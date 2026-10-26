"""WP-L4 gated writer arms; smoke and all runtime execute on compute."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed,stable_hash
from looplora.sym.evaluate import stratify_queries
from looplora.sym.stats import load_samples
from .evaluate import _slice
from .prefetch import BatchStream
from .read_fix import (ARMS,SIMPLICITY,ReadFixExperiment,configuration,diagnostics,extra_losses)
from .run import _write_json,source_metadata
from .train import make_task_spec,synchronize
from .writer_fix import alignment,auxiliary_losses


def save_checkpoint(experiment,optimizer,out,step):
    payload=dict(read_fix_version=1,arm=experiment.read_arm,config=experiment.config,step=step,
        signature=experiment.writer_signature(),adapter=experiment.inference_state())
    torch.save(payload,out/'writer.pt')
    torch.save({**payload,'optimizer':optimizer.state_dict(),
        'value_head':None if experiment.value_head is None else experiment.value_head.state_dict()},out/'checkpoint.pt')


def load_inference(path,*,model=None,tokenizer=None):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('read_fix_version')!=1 or 'value_head' in payload or 'optimizer' in payload:
        raise ValueError('expected a read-fix inference export')
    experiment=ReadFixExperiment(payload['config'],payload['arm'],model=model,tokenizer=tokenizer)
    try:
        if payload['signature']!=experiment.writer_signature():
            raise ValueError('read-fix signature mismatch')
        experiment.adapter.load_state_dict(payload['adapter'],strict=True)
        experiment.value_head=None
    except BaseException:
        experiment.close()
        raise
    return experiment


@torch.no_grad()
def validate(experiment,out,count=1000):
    out.mkdir(parents=True,exist_ok=False)
    spec=make_task_spec(experiment.config)
    arrays=stratify_queries(generate(spec,'val',0,count,4242),spec,[1],4242)
    pool=load_name_pool()
    ids=torch.tensor(pool.token_ids,device=experiment.device)
    chunks={}
    tick=time.perf_counter()
    for begin in range(0,count,16):
        part=_slice(arrays,slice(begin,min(begin+16,count)))
        batch=experiment.prepare(part)
        logits,x=experiment.score(batch,1,capture_inputs=True)
        prediction=logits.argmax(-1)
        restricted=logits.gather(-1,batch.candidate_ids[:,None,:]).masked_fill(~batch.candidate_mask[:,None,:],-torch.inf)
        candidate=batch.candidate_ids.gather(1,restricted.argmax(-1))
        tail_ids=ids[torch.as_tensor(part.facts[:,:,2],device=experiment.device,dtype=torch.long)]
        diag,_=diagnostics(experiment,batch,x)
        shape=prediction.shape
        condition='memory_only/oracle_addressing' if experiment.arm=='OA' else 'memory_only/original'
        values=dict(split=np.full(shape,'val'),condition=np.full(shape,condition),
            protocol=np.full(shape,'memory_only'),arm=np.full(shape,experiment.read_arm),
            episode_id=part.episode_seed[:,None],query_id=np.zeros(shape,dtype=np.int64),
            hop=np.ones(shape,dtype=np.int64),L=np.ones(shape,dtype=np.int64),
            answer=batch.answer_ids,prediction=prediction,correct=prediction==batch.answer_ids,
            full_correct=prediction==batch.answer_ids,full_prediction=prediction,
            candidate_prediction=candidate,candidate_correct=candidate==batch.answer_ids,
            candidate_size=batch.candidate_mask.sum(-1)[:,None],
            tail_prediction=(prediction[:,:,None]==tail_ids[:,None,:]).any(-1),
            nll=F.cross_entropy(logits.float().flatten(0,1),batch.answer_ids.flatten(),reduction='none'),
            **alignment(experiment,batch,x),**diag)
        for key,value in values.items():
            if isinstance(value,torch.Tensor): value=value.cpu().numpy()
            chunks.setdefault(key,[]).append(np.asarray(value).ravel())
    samples=load_samples({k:np.concatenate(v) for k,v in chunks.items()})
    np.savez_compressed(out/'samples.npz',**samples)
    metrics={key:float(samples[key].mean()) for key in samples if samples[key].dtype.kind in 'bif' and key not in {
        'episode_id','query_id','hop','L','answer','prediction','full_prediction','candidate_prediction'}}
    metrics.update(n=count,correct=int(samples['correct'].sum()),em=float(samples['correct'].mean()),
                   loss=float(samples['nll'].mean()),seconds=time.perf_counter()-tick)
    _write_json(out/'metrics.json',metrics)
    return metrics


def fit(experiment,out,steps,*,evaluate=True):
    options=experiment.config['train']
    seed=derive_seed(17,'llm','writer-data')
    params=experiment.trainable_parameters()
    optimizer=torch.optim.AdamW(params,lr=.003,weight_decay=.01)
    checkpoints={round(steps*k/5) for k in range(1,6)}
    stream=BatchStream(make_task_spec(experiment.config),'train',32,seed,steps,
                       tokenizer=experiment.tokenizer,workers=2)
    metrics,durations={},[]
    synchronize(experiment.device); tick=time.perf_counter()
    with stream,(out/'train_log.jsonl').open('w') as log:
        for step in range(1,steps+1):
            begin=time.perf_counter()
            arrays,tokens=next(stream)
            for group in optimizer.param_groups: group['lr']=.003*min(1.,step/20)
            optimizer.zero_grad(set_to_none=True)
            batch=experiment.prepare(arrays,tokens=tokens)
            logits,x=experiment.score(batch,1,capture_inputs=True)
            main=F.cross_entropy(logits.float().flatten(0,1),batch.answer_ids.flatten())
            address,value=auxiliary_losses(experiment,batch,x)
            read,decor=extra_losses(experiment,batch,x)
            loss=main+address+value+read+decor
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f'nonfinite step{step}')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step(); synchronize(experiment.device)
            duration=time.perf_counter()-begin; durations.append(duration)
            if step==1 or step%50==0 or step==steps:
                row=dict(step=step,loss=float(loss.detach()),main_loss=float(main.detach()),
                    address_loss=float(address.detach()),value_loss=float(value.detach()),
                    read_loss=float(read.detach()),decorrelation_loss=float(decor.detach()),
                    grad_norm=float(grad),clipped=bool(grad>1),step_seconds=duration,
                    lr=optimizer.param_groups[0]['lr'],data_seed=seed)
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                print(json.dumps(dict(arm=experiment.read_arm,**row)),flush=True)
            if evaluate and step in checkpoints:
                metrics[str(step)]=validate(experiment,out/f'val-{step}',experiment.config['eval']['episodes'])
            if time.perf_counter()-tick>3450 and step!=steps:
                raise RuntimeError('fixed WP-L4 schedule exceeded its measured allowance')
    save_checkpoint(experiment,optimizer,out,steps)
    result=dict(actual_steps=steps,training_episodes=steps*32,training_queries=steps*128,
        eval_steps=sorted(checkpoints) if evaluate else [],metrics=metrics,data_seed=seed,
        seconds=time.perf_counter()-tick,
        steady_step_mean_seconds=float(np.mean(durations[min(10,len(durations)-1):])),
        step_p95_seconds=float(np.quantile(durations,.95)))
    _write_json(out/'training.json',result)
    return result


def require_prior(decision,report):
    if decision is None or report is None or not Path(report).is_file():
        raise ValueError('finish and report WP-L3 before WP-L4 scientific submission')
    summary=json.loads(Path(decision).read_text())
    if summary.get('screen_only') is not False or summary.get('selected_arm') is not None or summary.get('stop_P1L') is not True:
        raise ValueError('WP-L4 requires final audited WP-L3 with no selected writer')
    if not summary.get('export_check') or summary.get('decision_pending_rerun'):
        raise ValueError('WP-L3 final inference audit is required')


def run_one(arm,out,steps,E=16,smoke=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    cfg=configuration(arm,steps,E=E,smoke=smoke)
    record=dict(**source_metadata(),arm=arm,config=cfg,config_hash=stable_hash(cfg),status='running',
        smoke=smoke,seed=17,eval_seed=4242,stage='writer',protocol='memory_only',rank=32,layer=14,L=1,
        host=socket.gethostname(),job_id=os.environ.get('PBS_JOBID'),created_at=datetime.now(timezone.utc).isoformat())
    experiment=None
    try:
        experiment=ReadFixExperiment(cfg,arm)
        record['standardization']=experiment.standardize()
        record['training']=fit(experiment,out,steps)
        record['inference_parameters']=sum(p.numel() for p in experiment.adapter.parameters())
        record['training_only_parameters']=0 if experiment.value_head is None else sum(p.numel() for p in experiment.value_head.parameters())
        record['runtime']=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        record.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        record.update(status='failed',failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        if experiment is not None: experiment.close()
        _write_json(out/'record.json',record)
    return record


def _task(args): return run_one(*args)


def choose(records):
    if len(records)!=4 or {r['arm'] for r in records}!=set(ARMS) or any(r['status']!='complete' or r['smoke'] for r in records):
        raise ValueError('decision requires all four complete scientific E16 arms')
    by_arm={r['arm']:r for r in records}
    scores={a:by_arm[a]['training']['metrics'][str(by_arm[a]['training']['actual_steps'])]['correct'] for a in SIMPLICITY}
    passed=[a for a in SIMPLICITY if scores[a]>=800]
    if passed: return dict(action='select',arm=passed[0],E=16,correct=scores)
    best=max(SIMPLICITY,key=lambda a:(scores[a],-SIMPLICITY.index(a)))
    if scores[best]>=500: return dict(action='retry_e8',arm=best,E=8,correct=scores)
    return dict(action='stop',arm=None,E=None,correct=scores)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',nargs='+',choices=ARMS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--prior-decision',type=Path)
    parser.add_argument('--prior-report',type=Path)
    parser.add_argument('--e8-decision',type=Path)
    args=parser.parse_args()
    if not 1<=len(args.arms)<=2 or len(set(args.arms))!=len(args.arms) or args.steps<5:
        raise ValueError('one job runs one or two distinct arms for at least five steps')
    E=16
    if not args.smoke:
        require_prior(args.prior_decision,args.prior_report)
        if args.steps%5: raise ValueError('scientific schedule requires five equal intervals')
        if args.e8_decision:
            decision=json.loads(args.e8_decision.read_text())
            if decision['action']!='retry_e8' or args.arms!=[decision['arm']]:
                raise ValueError('E8 requires the one locked eligible arm')
            E=8
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(_task,[(a,str(args.out/a),args.steps,E,args.smoke) for a in args.arms]))
    _write_json(args.out/('shard-'+'-'.join(args.arms)+'.json'),dict(runs=records,procs=len(args.arms)))
