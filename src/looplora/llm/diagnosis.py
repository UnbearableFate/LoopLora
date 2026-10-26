"""Bounded WP-L2 output-oracle and writer-budget diagnostics on compute."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed, stable_hash
from looplora.sym.evaluate import stratify_queries
from looplora.sym.stats import load_samples
from .evaluate import _slice, aggregate_samples
from .pilot1_select import writer_config
from .run import _write_json, source_metadata
from .train import LLMExperiment, make_task_spec, synchronize


def inject_last(output, codes, B, mask, alpha=1.):
    """Inject a supplied post-reader z only at the actual prediction token."""
    if not bool(mask.any(-1).all()):
        raise ValueError('oracle requires a real read token in every row')
    last = torch.where(mask,torch.arange(mask.shape[1],device=mask.device),-1).amax(-1)
    branch = alpha*F.linear(codes,B)
    delta = torch.zeros_like(output).index_put(
        (torch.arange(len(output),device=output.device),last),branch.to(output.dtype))
    return output+delta


class OutputOracle(LLMExperiment):
    """Answer identity is deliberately supplied; there is no writer/read path."""
    def __init__(self, config, *, model=None, tokenizer=None):
        baseline = copy.deepcopy(config)
        baseline.update(stage='baseline',variant='B0')
        super().__init__(baseline,model=model,tokenizer=tokenizer)
        self.config = config
        r,d = config['model']['r'],self.model.config.hidden_size
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed(17,'D2-answer-code'))
            self.table = nn.Embedding(config['task']['V'],r).to(device=self.device,dtype=torch.float32)
        self.B = nn.Parameter(torch.zeros(d,r,device=self.device,dtype=torch.float32))
        self.enabled = True
        self.codes = self.mask = None
        projection = self.model.model.layers[config['model']['layer']].self_attn.o_proj
        self.handle = projection.register_forward_hook(self._inject)

    def _inject(self,module,args,output):
        if not self.enabled:
            return output
        if self.codes is None or self.mask is None:
            raise RuntimeError('oracle codes must be scoped to a score call')
        return inject_last(output,self.codes,self.B,self.mask,self.config['model']['alpha'])

    def score(self, prepared, L=1, *, capture_inputs=False):
        B,Q,T = prepared.question_ids.shape
        self.codes = self.table(torch.as_tensor(prepared.arrays.answer,device=self.device,dtype=torch.long).flatten())
        self.mask = prepared.question_mask.reshape(B*Q,T)
        try:
            return super().score(prepared,L,capture_inputs=capture_inputs)
        finally:
            self.codes = self.mask = None

    def trainable_parameters(self):
        return [self.table.weight,self.B]

    def close(self):
        if hasattr(self,'handle'):
            self.handle.remove()
        super().close()


def matching(experiment, batch, write_x, read_x):
    """Direct source-fact key/query alignment, without changing memory or logits."""
    adapter = experiment.adapter
    B,Q,T = batch.question_ids.shape
    support_u = adapter.u(write_x.float())
    E = batch.arrays.facts.shape[1]
    fact_u = support_u[batch.fact_final].reshape(B,E,-1)
    keys = F.normalize(F.linear(fact_u,adapter.K),dim=-1,eps=adapter.cfg.eps)
    last = batch.question_mask.long().sum(-1)-1
    query_x = read_x[torch.arange(B,device=read_x.device)[:,None],
                     torch.arange(Q,device=read_x.device)[None,:],last]
    queries = F.normalize(F.linear(adapter.u(query_x.float()),adapter.Q),dim=-1,eps=adapter.cfg.eps)
    scores = torch.einsum('bqr,ber->bqe',queries,keys)
    facts = torch.as_tensor(batch.arrays.facts,device=read_x.device)
    head = torch.as_tensor(batch.arrays.q_start,device=read_x.device)
    rel = torch.as_tensor(batch.arrays.q_rel,device=read_x.device)
    correct_fact = (facts[:,None,:,0]==head[:,:,None]) & (facts[:,None,:,1]==rel[:,:,None])
    if not bool((correct_fact.sum(-1)==1).all()):
        raise AssertionError('T1 matching requires exactly one source fact')
    target = correct_fact.long().argmax(-1)
    target_score = scores.gather(-1,target[...,None]).squeeze(-1)
    non_target = scores.masked_fill(correct_fact,-torch.inf).amax(-1)
    return {'key_top1':(scores.argmax(-1)==target).flatten(),
            'key_target_cosine':target_score.flatten(),
            'key_margin':(target_score-non_target).flatten()}


@torch.no_grad()
def validate(experiment, out: Path, *, key_matching=False):
    """The unchanged Pilot-1 val questions, with per-answer NLL and flat samples."""
    out.mkdir(parents=True,exist_ok=False)
    config = experiment.config
    spec = make_task_spec(config)
    arrays = stratify_queries(generate(spec,'val',0,1000,4242),spec,[1],4242)
    chunks = {}
    started = time.perf_counter()
    for begin in range(0,1000,16):
        part = _slice(arrays,slice(begin,min(begin+16,1000)))
        captured,handle = [],None
        if key_matching:
            projection = experiment.bank.modules[experiment.layer]
            if getattr(experiment,'input_site','o_proj')=='residual':
                handle = experiment.model.model.layers[experiment.layer].register_forward_pre_hook(
                    lambda _,args: captured.append(args[0]))
            else:
                handle = projection.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
        try:
            batch = experiment.prepare(part)
            logits,read_x = experiment.score(batch,1,capture_inputs=key_matching)
        finally:
            if handle is not None:
                handle.remove()
        shape = part.answer.shape
        answers = batch.answer_ids
        prediction = logits.argmax(-1)
        candidate = logits.gather(-1,batch.candidate_ids[:,None,:].expand(-1,shape[1],-1))
        choice = candidate.masked_fill(~batch.candidate_mask[:,None,:],-torch.inf).argmax(-1)
        candidate_prediction = batch.candidate_ids.gather(1,choice)
        loss = F.cross_entropy(logits.float().flatten(0,1),answers.flatten(),reduction='none')
        values = dict(split=np.full(shape,'val'),condition=np.full(shape,'memory_only/original'),
            protocol=np.full(shape,'memory_only'),episode_id=part.episode_seed[:,None],
            query_id=np.zeros(shape,dtype=np.int64),hop=np.ones(shape,dtype=np.int64),
            L=np.ones(shape,dtype=np.int64),answer=answers,prediction=prediction,
            correct=prediction==answers,full_correct=prediction==answers,
            full_prediction=prediction,candidate_prediction=candidate_prediction,
            candidate_correct=candidate_prediction==answers,
            candidate_size=batch.candidate_mask.sum(-1)[:,None],nll=loss)
        if key_matching:
            if len(captured)!=2:
                raise AssertionError('expected one support and one read feature capture')
            # Residual-input score capture is supplied by the layer hook.
            qx = captured[1].reshape(*batch.question_ids.shape,-1) if getattr(
                experiment,'input_site','o_proj')=='residual' else read_x
            values.update(matching(experiment,batch,captured[0],qx))
        for key,value in values.items():
            if isinstance(value,torch.Tensor):
                value = value.detach().cpu().numpy()
            chunks.setdefault(key,[]).append(np.asarray(value).ravel())
    samples = load_samples({key:np.concatenate(value) for key,value in chunks.items()})
    if len(np.unique(samples['episode_id']))!=1000:
        raise AssertionError('validation must contain 1000 unique episodes')
    np.savez_compressed(out/'samples.npz',**samples)
    synchronize(experiment.device)
    result = dict(table=aggregate_samples(samples),n=1000,correct=int(samples['correct'].sum()),
                  em=float(samples['correct'].mean()),candidate_accuracy=float(samples['candidate_correct'].mean()),
                  loss=float(samples['nll'].mean()),seconds=time.perf_counter()-started)
    if key_matching:
        result['matching'] = {key:float(samples[key].mean()) for key in
                              ('key_top1','key_target_cosine','key_margin')}
        result['matching']['chance']=1/spec.E
    _write_json(out/'metrics.json',result)
    print(json.dumps(dict(validation=str(out),**result)),flush=True)
    return result


def _checkpoint(experiment,out,step):
    if isinstance(experiment,OutputOracle):
        payload = dict(table=experiment.table.weight.detach().cpu(),B=experiment.B.detach().cpu())
    else:
        payload = dict(adapter={key:value.detach().cpu() for key,value in experiment.adapter.state_dict().items()},
                       signature=experiment.writer_signature())
    torch.save(dict(config=experiment.config,step=step,**payload),out/'checkpoint.pt')


def fit(experiment,out: Path,*,steps: int, checkpoints: list[int], max_seconds=3500):
    config,options = experiment.config,experiment.config['train']
    spec = make_task_spec(config)
    data_seed = derive_seed(17,'llm','writer-data')
    params = experiment.trainable_parameters()
    optimizer = torch.optim.AdamW(params,lr=options['lr'],weight_decay=options['weight_decay'])
    evaluations,durations = {},[]
    synchronize(experiment.device)
    started = time.perf_counter()
    with (out/'train_log.jsonl').open('w') as log:
        for step in range(1,steps+1):
            begin = time.perf_counter()
            arrays = generate(spec,'train',(step-1)*32,32,data_seed)
            for group in optimizer.param_groups:
                group['lr']=options['lr']*min(1.,step/max(1,options['warmup_steps']))
            optimizer.zero_grad(set_to_none=True)
            batch = experiment.prepare(arrays)
            logits,_ = experiment.score(batch,1)
            loss = F.cross_entropy(logits.float().flatten(0,1),batch.answer_ids.flatten())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'nonfinite training loss at step {step}')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params,options['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            synchronize(experiment.device)
            seconds = time.perf_counter()-begin
            durations.append(seconds)
            timed_out = time.perf_counter()-started>=max_seconds
            if step==1 or step%50==0 or step==steps or timed_out:
                row = dict(step=step,loss=float(loss.detach()),grad_norm=float(grad),
                    accuracy=float((logits.argmax(-1)==batch.answer_ids).float().mean()),
                    step_seconds=seconds,lr=optimizer.param_groups[0]['lr'],data_seed=data_seed)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
                print(json.dumps(row),flush=True)
            if step in checkpoints or timed_out:
                evaluations[str(step)] = validate(experiment,out/f'val-{step}',
                                                  key_matching=not isinstance(experiment,OutputOracle))
            if timed_out:
                break
    _checkpoint(experiment,out,step)
    result = dict(requested_steps=steps,actual_steps=step,time_guard_reached=step!=steps,
                  train_episodes=step*32,train_queries=step*128,
                  seconds=time.perf_counter()-started,train_step_mean_seconds=float(np.mean(durations)),
                  train_step_p95_seconds=float(np.quantile(durations,.95)),
                  final_loss=float(loss.detach()),evaluations=evaluations,data_seed=data_seed)
    _write_json(out/'training.json',result)
    return result


def run_one(kind,lr,out,steps,input_site='o_proj'):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    config = writer_config(f'{kind}-{input_site}-{lr}',16,'fact_final',lr,32,steps)
    config['train']['log_every']=50
    if input_site!='o_proj':
        config['model']['input_site']=input_site
    provenance = source_metadata()
    record = dict(diagnostic=kind,status='running',config=config,config_hash=stable_hash(config),
        **provenance,created_at=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),
        job_id=os.environ.get('PBS_JOBID'),input_site=input_site,seed=17,eval_seed=4242,
        scope='T1 writer diagnosis only; train/val; E16; rank32 layer14 L1; batch32 Q4',
        dtype='bf16 backbone / fp32 trainable',deterministic=True)
    _write_json(out/'record.json',record)
    experiment = None
    try:
        experiment = OutputOracle(config) if kind=='D2' else LLMExperiment(config)
        if input_site!='o_proj':
            experiment.input_site=input_site
        if kind=='D2':
            experiment.enabled=False
            record['B0']=validate(experiment,out/'B0')
            experiment.enabled=True
            record['oracle']='z=c(answer) at final question read token only; alpha1; rho.25 bypassed; only table/B trained'
        checkpoints = [steps] if kind=='D2' else sorted({1500,steps//2,steps})
        record['training']=fit(experiment,out,steps=steps,checkpoints=checkpoints)
        record['status']='complete'
        record['completed_at']=datetime.now(timezone.utc).isoformat()
        record['runtime']={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    except BaseException as error:
        record.update(status='failed',failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        if experiment is not None:
            experiment.close()
        _write_json(out/'record.json',record)
    return record


def run_suite(kind,out: Path,steps: int):
    out.mkdir(parents=True,exist_ok=False)
    lrs = [.0003,.001,.003] if kind=='D2' else [.0003,.001]
    site = 'residual' if kind=='D4' else 'o_proj'
    tasks = [(kind,lr,str(out/f'lr-{lr:g}'),steps,site) for lr in lrs]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as executor:
        results = list(executor.map(_task,tasks))
    _write_json(out/'summary.json',dict(diagnostic=kind,procs=2,runs=results))


def _task(args):
    return run_one(*args)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic',choices=('D2','D3','D4'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int)
    args = parser.parse_args()
    steps = args.steps or (1500 if args.diagnostic=='D2' else 6400)
    if (args.diagnostic=='D2' and steps!=1500) or (args.diagnostic!='D2' and steps!=6400):
        raise ValueError('WP-L2 fixes D2 at1500 and budget trials at6400 requested steps')
    run_suite(args.diagnostic,args.out,steps)
