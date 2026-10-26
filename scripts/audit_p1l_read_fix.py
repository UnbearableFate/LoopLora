"""WP-L4 no-training audit of final A1 checkpoints; compute only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm.evaluate import _slice
from looplora.llm.read_fix import controlled_score,diagnostics,question_terms
from looplora.llm.read_fix_run import require_prior
from looplora.llm.run import _write_json,source_metadata
from looplora.llm.train import make_task_spec
from looplora.llm.writer_fix import auxiliary_losses
from looplora.llm.writer_fix_run import load_inference
from looplora.sym.evaluate import stratify_queries
from looplora.sym.stats import load_samples


@torch.enable_grad()
def gradient_audit(experiment):
    params=experiment.trainable_parameters()
    names=[name for name,_ in experiment.adapter.named_parameters()]
    if experiment.value_head is not None:
        names += ['value_head.'+name for name,_ in experiment.value_head.named_parameters()]
    rows=[]
    spec=make_task_spec(experiment.config)
    for step in range(4):
        arrays=generate(spec,'train',step*32,32,derive_seed(17,'llm','writer-data'))
        batch=experiment.prepare(arrays)
        logits,x=experiment.score(batch,1,capture_inputs=True)
        main=F.cross_entropy(logits.float().flatten(0,1),batch.answer_ids.flatten())
        address,value=auxiliary_losses(experiment,batch,x)
        losses=dict(answer=main,address=address,value=value)
        vectors={}
        norms={}
        for label,loss in losses.items():
            grads=(torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                   if loss.requires_grad else [None]*len(params))
            vectors[label]=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,grads)])
            norms[label]={name:0. if g is None else float(g.norm()) for name,g in zip(names,grads)}
        total=sum(vectors.values())
        alignment={a+'__'+b:float(F.cosine_similarity(vectors[a][None],vectors[b][None]))
                   for a,b in (('answer','address'),('answer','value'),('address','value'))}
        rows.append(dict(batch=step,parameter_gradient_norms=norms,cosines=alignment,
                         total_norm=float(total.norm()),would_clip=bool(total.norm()>1)))
    return dict(training_batches=4,episodes=128,optimizer_steps=0,rows=rows,
                would_clip_fraction=float(np.mean([r['would_clip'] for r in rows])))


def load_head(experiment,path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    state=payload['value_head']
    if state is not None:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed(17,'read-fix-audit-head-container'))
            experiment.value_head=nn.Linear(experiment.adapter.cfg.r_mem,1024).to(experiment.device)
        experiment.value_head.load_state_dict(state,strict=True)


@torch.no_grad()
def audit_one(folder,out,count=1000,*,model=None,tokenizer=None):
    out.mkdir(parents=True,exist_ok=False)
    record=json.loads((folder/'record.json').read_text())
    assert record['status']=='complete' and record['extension'] and record['training']['actual_steps']==32900
    experiment=load_inference(folder/'writer.pt',model=model,tokenizer=tokenizer)
    try:
        load_head(experiment,folder/'checkpoint.pt')
        spec=make_task_spec(experiment.config)
        arrays=stratify_queries(generate(spec,'val',0,count,4242),spec,[1],4242)
        order=np.random.default_rng(derive_seed(17,'read-fix','audit-memory-shuffle')).permutation(count)
        pairing=np.empty(count,dtype=np.int64); pairing[order]=np.roll(order,1)
        chunks={}; vectors={}; max_drift=0.
        for begin in range(0,count,16):
            end=min(begin+16,count)
            part=_slice(arrays,slice(begin,end))
            batch=experiment.prepare(part)
            logits,x=controlled_score(experiment,batch)
            repeat,repeat_x=controlled_score(experiment,batch)
            max_drift=max(max_drift,float((repeat-logits).abs().max()))
            assert torch.equal(repeat,logits) and torch.equal(repeat_x,x)
            diag,exact=diagnostics(experiment,batch,x)
            assert float(diag['gram_relative_error'].max())<1e-4,'trace write masks/codes before interpreting'
            terms=question_terms(experiment,batch,x)
            donor=experiment.prepare(_slice(arrays,pairing[begin:end]))
            conditions={'original':logits}
            for name in ('zero','shuffle'):
                changed,changed_x=controlled_score(experiment,batch,name,donor if name=='shuffle' else None)
                assert torch.equal(changed_x,x),'question input changed under memory-only intervention'
                conditions[name]=changed
            shape=batch.answer_ids.shape
            values=dict(split=np.full(shape,'val'),condition=np.full(shape,'memory_only/original'),
                protocol=np.full(shape,'memory_only'),episode_id=part.episode_seed[:,None],
                query_id=np.zeros(shape,dtype=np.int64),hop=np.ones(shape,dtype=np.int64),
                L=np.ones(shape,dtype=np.int64),answer=batch.answer_ids,
                prediction=logits.argmax(-1),correct=logits.argmax(-1)==batch.answer_ids,**diag)
            for name,scores in conditions.items():
                nll=F.cross_entropy(scores.float().flatten(0,1),batch.answer_ids.flatten(),reduction='none')
                answer_logit=scores.gather(-1,batch.answer_ids[...,None]).squeeze(-1)
                other=scores.clone().scatter(-1,batch.answer_ids[...,None],-torch.inf).amax(-1)
                values.update({name+'_nll':nll,name+'_correct':scores.argmax(-1)==batch.answer_ids,
                    name+'_answer_logit':answer_logit,name+'_answer_margin':answer_logit-other})
            for key,value in values.items():
                if isinstance(value,torch.Tensor): value=value.cpu().numpy()
                chunks.setdefault(key,[]).append(np.asarray(value).ravel())
            dense=dict(b=exact['b'],a=exact['a'],d=exact['d'],gram=exact['gram'],keys=batch.keys,
                       values=batch.values,q=terms['q'],matrix=batch.snapshot.states[experiment.layer],
                       actual_stored_read=terms['stored'],actual_deployed_read=terms['m'],
                       reconstructed=exact['reconstructed'])
            for key,value in dense.items(): vectors.setdefault(key,[]).append(value.cpu().numpy())
        samples=load_samples({k:np.concatenate(v) for k,v in chunks.items()})
        np.savez_compressed(out/'samples.npz',**samples)
        np.savez_compressed(out/'coefficients.npz',**{k:np.concatenate(v) for k,v in vectors.items()})
        means={key:float(value.mean()) for key,value in samples.items() if value.dtype.kind in 'bif' and key not in {
            'episode_id','query_id','hop','L','answer','prediction'}}
        ages=[]
        for age in range(16):
            mask=samples['target_age']==age
            ages.append(dict(age=age,n=int(mask.sum()),**{
                key:float(samples[key][mask].mean()) if mask.any() else None
                for key in ('target_d','value_SIR','parallel_interference','relative_read_error','original_correct')}))
        result=dict(**source_metadata(),checkpoint_record=record,episodes=count,seed=17,eval_seed=4242,
            oracle_matrix_bypassed=experiment.arm=='OA',means=means,age_strata=ages,
            max_repeated_logit_drift=max_drift,identical_repeated_predictions=True,
            max_gram_relative_error=float(samples['gram_relative_error'].max()),
            value_head_available=experiment.value_head is not None,
            gradients=gradient_audit(experiment),optimizer_steps=0)
        _write_json(out/'summary.json',result)
        return result
    finally:
        experiment.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prior-decision',type=Path,required=True)
    parser.add_argument('--prior-report',type=Path,required=True)
    args=parser.parse_args()
    require_prior(args.prior_decision,args.prior_report)
    args.out.mkdir(parents=True,exist_ok=False)
    runs={arm:audit_one(args.root/'rerun'/arm,args.out/arm) for arm in ('OA','W5')}
    _write_json(args.out/'summary.json',dict(**source_metadata(),runs=runs,
        third_arm=None,third_arm_reason='A1 top-two selection includes OA; no third rerun arm exists'))
