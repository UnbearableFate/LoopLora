"""WP-L3 fixed seven-arm screen and Amendment A1 locked fast rerun."""
from __future__ import annotations

import argparse
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
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed,stable_hash
from looplora.sym.evaluate import stratify_queries
from looplora.sym.stats import load_samples
from .evaluate import _slice,aggregate_samples
from .pilot1_select import writer_config
from .prefetch import BatchStream
from .run import _write_json,source_metadata
from .train import make_task_spec,synchronize
from .writer_fix import (ARMS,SIMPLICITY,WriterFixExperiment,alignment,auxiliary_losses)


def configuration(arm: str,steps=4800,*,prefetch_workers=0):
    if arm not in ARMS:
        raise ValueError('unknown WP-L3 arm')
    cfg = writer_config(f'writer-fix-{arm}',16,'fact_final',.003,32,steps)
    cfg['train']['log_every']=50
    if prefetch_workers:
        cfg['train']['prefetch_workers']=prefetch_workers
    cfg['writer_fix'] = dict(arm=arm,backend='core.scan_final',
        effective_write_policy='paired_pre_tail_tail' if arm=='W4' else 'fact_final',
        standardization=arm in {'W1','W5'},address_weight=1 if arm in {'W2','W5'} else 0,
        address_temperature=.1,value_weight=1 if arm in {'W3','W5'} else 0,
        value_head='training_only_32_to_1024_affine' if arm in {'W3','W5'} else None,
        oracle_addressing=arm=='OA')
    return cfg


@torch.no_grad()
def validate(experiment,out: Path):
    out.mkdir(parents=True,exist_ok=False)
    spec = make_task_spec(experiment.config)
    arrays = stratify_queries(generate(spec,'val',0,1000,4242),spec,[1],4242)
    started = time.perf_counter()
    chunks = {}
    pool = load_name_pool()
    tokens = torch.tensor(pool.token_ids,device=experiment.device)
    for begin in range(0,1000,16):
        part = _slice(arrays,slice(begin,min(begin+16,1000)))
        batch = experiment.prepare(part)
        logits,x = experiment.score(batch,1,capture_inputs=True)
        answer = batch.answer_ids
        prediction = logits.argmax(-1)
        restricted = logits.gather(-1,batch.candidate_ids[:,None,:])
        choice = restricted.masked_fill(~batch.candidate_mask[:,None,:],-torch.inf).argmax(-1)
        candidate = batch.candidate_ids.gather(1,choice)
        tails = tokens[torch.as_tensor(part.facts[:,:,2],device=experiment.device,dtype=torch.long)]
        tail_prediction = (prediction[:,:,None]==tails[:,None,:]).any(-1)
        shape = answer.shape
        values = dict(split=np.full(shape,'val'),condition=np.full(shape,
                'memory_only/oracle_addressing' if experiment.arm=='OA' else 'memory_only/original'),
            protocol=np.full(shape,'memory_only'),arm=np.full(shape,experiment.arm),
            episode_id=part.episode_seed[:,None],query_id=np.zeros(shape,dtype=np.int64),
            hop=np.ones(shape,dtype=np.int64),L=np.ones(shape,dtype=np.int64),
            answer=answer,prediction=prediction,correct=prediction==answer,full_correct=prediction==answer,
            full_prediction=prediction,candidate_prediction=candidate,candidate_correct=candidate==answer,
            candidate_size=batch.candidate_mask.sum(-1)[:,None],tail_prediction=tail_prediction,
            nll=F.cross_entropy(logits.float().flatten(0,1),answer.flatten(),reduction='none'),
            **alignment(experiment,batch,x))
        for name,value in values.items():
            if isinstance(value,torch.Tensor):
                value = value.detach().cpu().numpy()
            chunks.setdefault(name,[]).append(np.asarray(value).ravel())
    samples = load_samples({key:np.concatenate(value) for key,value in chunks.items()})
    if len(samples['correct'])!=1000 or len(np.unique(samples['episode_id']))!=1000:
        raise AssertionError('validation requires 1000 unique Pilot-1 episodes')
    np.savez_compressed(out/'samples.npz',**samples)
    synchronize(experiment.device)
    metric = dict(n=1000,correct=int(samples['correct'].sum()),em=float(samples['correct'].mean()),
        loss=float(samples['nll'].mean()),candidate_accuracy=float(samples['candidate_correct'].mean()),
        tail_prediction_fraction=float(samples['tail_prediction'].mean()),
        key_top1=float(samples['key_top1'].mean()),key_margin=float(samples['key_margin'].mean()),
        key_pairwise_cosine=float(samples['key_pairwise_cosine'].mean()),
        key_chance=1/16,table=aggregate_samples(samples),seconds=time.perf_counter()-started)
    _write_json(out/'metrics.json',metric)
    print(json.dumps(dict(validation=str(out),**metric)),flush=True)
    return metric


def save_checkpoint(experiment,optimizer,out: Path,step: int):
    state = dict(writer_fix_version=1,config=experiment.config,arm=experiment.arm,step=step,
                 signature=experiment.writer_signature(),adapter=experiment.inference_state())
    # Inference export deliberately has neither training head nor optimizer.
    torch.save(state,out/'writer.pt')
    torch.save({**state,'optimizer':optimizer.state_dict(),
                'value_head':None if experiment.value_head is None else
                    {key:value.detach().cpu() for key,value in experiment.value_head.state_dict().items()}},
               out/'checkpoint.pt')


def load_inference(path: Path,*,model=None,tokenizer=None):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('writer_fix_version')!=1 or 'value_head' in payload or 'optimizer' in payload:
        raise ValueError('expected a writer-fix inference export')
    experiment = WriterFixExperiment(payload['config'],payload['arm'],model=model,tokenizer=tokenizer)
    try:
        if payload['signature']!=experiment.writer_signature():
            raise ValueError('writer-fix checkpoint signature mismatch')
        experiment.adapter.load_state_dict(payload['adapter'],strict=True)
        experiment.value_head=None
    except BaseException:
        experiment.close()
        raise
    return experiment


def fit(experiment,out: Path,*,steps=4800,extension=False,max_seconds=3450,eval_steps=None):
    options = experiment.config['train']
    spec = make_task_spec(experiment.config)
    seed = derive_seed(17,'llm','writer-data')
    params = experiment.trainable_parameters()
    optimizer = torch.optim.AdamW(params,lr=.003,weight_decay=options['weight_decay'])
    metrics,durations = {},[]
    eval_steps=set((1200,2400,4800) if eval_steps is None else eval_steps)
    synchronize(experiment.device)
    started = time.perf_counter()
    stream=BatchStream(spec,'train',32,seed,steps,tokenizer=experiment.tokenizer,
                       workers=options.get('prefetch_workers',0))
    with stream, (out/'train_log.jsonl').open('w') as log:
        for step in range(1,steps+1):
            begin = time.perf_counter()
            arrays,tokens = next(stream)
            for group in optimizer.param_groups:
                group['lr']=.003*min(1.,step/20)
            optimizer.zero_grad(set_to_none=True)
            batch = experiment.prepare(arrays,tokens=tokens)
            logits,x = experiment.score(batch,1,capture_inputs=experiment.arm in {'W2','W5'})
            main = F.cross_entropy(logits.float().flatten(0,1),batch.answer_ids.flatten())
            address,value = auxiliary_losses(experiment,batch,x)
            loss = main+address+value
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'nonfinite loss at step {step}')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params,options['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            synchronize(experiment.device)
            seconds = time.perf_counter()-begin
            durations.append(seconds)
            done = step==steps
            if max_seconds is not None and time.perf_counter()-started>=max_seconds and not done:
                raise RuntimeError('fixed schedule exceeded its measured walltime allowance')
            if step==1 or step%50==0 or done:
                row = dict(step=step,loss=float(loss.detach()),main_loss=float(main.detach()),
                    address_loss=float(address.detach()),value_loss=float(value.detach()),
                    grad_norm=float(grad),lr=optimizer.param_groups[0]['lr'],step_seconds=seconds,data_seed=seed)
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                print(json.dumps(row),flush=True)
            if step in eval_steps:
                metrics[str(step)] = validate(experiment,out/f'val-{step}')
            if done:
                break
    save_checkpoint(experiment,optimizer,out,step)
    result = dict(actual_steps=step,requested_steps=steps,training_episodes=step*32,
        training_queries=step*128,data_seed=seed,extension=extension,
        termination='steps',eval_steps=sorted(eval_steps),
        seconds=time.perf_counter()-started,step_mean_seconds=float(np.mean(durations)),
        steady_step_mean_seconds=float(np.mean(durations[min(10,len(durations)-1):])),
        step_p95_seconds=float(np.quantile(durations,.95)),metrics=metrics)
    _write_json(out/'training.json',result)
    return result


def run_one(arm,out,extension=False,steps=4800):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    cfg = configuration(arm,steps,prefetch_workers=2 if extension else 0)
    record = dict(experiment_id=cfg['name'],status='running',arm=arm,config=cfg,
        config_hash=stable_hash(cfg),**source_metadata(),created_at=datetime.now(timezone.utc).isoformat(),
        seed=17,eval_seed=4242,stage='writer',variant='M1',protocol='memory_only',rank=32,layer=14,L=1,
        host=socket.gethostname(),job_id=os.environ.get('PBS_JOBID'),backend='core.scan_final',
        scientific_scope='WP-L3 T1 val-only seven-arm screen; no reader/joint/T2',extension=extension)
    _write_json(out/'record.json',record)
    experiment = None
    try:
        experiment = WriterFixExperiment(cfg,arm)
        record['standardization']=experiment.standardize()
        record['inference_parameters']=sum(p.numel() for p in experiment.adapter.parameters())
        record['training_only_parameters']=(0 if experiment.value_head is None else
                                            sum(p.numel() for p in experiment.value_head.parameters()))
        evaluation=[steps*k//5 for k in range(1,6)] if extension else [1200,2400,4800]
        record['training']=fit(experiment,out,steps=steps,extension=extension,
                              max_seconds=7000 if extension else 3450,eval_steps=evaluation)
        record['runtime']=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        record.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        record.update(status='failed',failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        if experiment is not None:
            experiment.close()
        _write_json(out/'record.json',record)
    return record


def _task(args):
    return run_one(*args)


def screen(arms,out: Path):
    if not 1<=len(arms)<=2 or len(set(arms))!=len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError('one job permits one or two distinct registered arms')
    out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(_task,[(arm,str(out/arm),False) for arm in arms]))
    _write_json(out/('shard-'+'-'.join(arms)+'.json'),dict(procs=min(2,len(arms)),runs=records))


def choose(records,final_main_losses=None):
    if len(records)!=7 or {r['arm'] for r in records}!=set(ARMS):
        raise ValueError('decision requires all seven distinct complete screen arms')
    by_arm={r['arm']:r for r in records}
    for rec in records:
        if rec['status']!='complete' or rec['training']['actual_steps']!=4800 or rec['extension']:
            raise ValueError('decision requires the complete fixed4800-step screen')
    correct={arm:by_arm[arm]['training']['metrics']['4800']['correct'] for arm in ARMS}
    passed=[arm for arm in SIMPLICITY if correct[arm]>=800]
    if passed:
        return dict(action='select',arm=passed[0],screen_correct=correct)
    if final_main_losses is None or set(final_main_losses)!=set(ARMS):
        raise ValueError('A1 ranking requires final main losses for all seven arms')
    if not all(np.isfinite(v) for v in final_main_losses.values()):
        raise ValueError('A1 ranking requires finite final main losses')
    ranked=sorted(ARMS,key=lambda arm:(-correct[arm],final_main_losses[arm],ARMS.index(arm)))
    arms=ranked[:2]
    if 'OA' not in arms:
        arms.append('OA')
    return dict(action='rerun',arm=None,arms=arms,screen_correct=correct,
                final_main_losses=final_main_losses,amendment='A1')


def rerun(selection: Path,out: Path,steps: int):
    lock=json.loads(selection.read_text())
    arms=lock['choice'].get('arms',[])
    if lock['choice']['action']!='rerun' or lock['choice'].get('amendment')!='A1':
        raise ValueError('rerun requires an audited A1 selection lock')
    if not 2<=len(arms)<=3 or len(set(arms))!=len(arms) or 'OA' not in arms or any(a not in ARMS for a in arms):
        raise ValueError('A1 requires the top two arms plus OA if absent')
    if steps<=4800 or steps%5:
        raise ValueError('A1 rerun steps must exceed4800 and divide into five equal intervals')
    out.mkdir(parents=True,exist_ok=False)
    _write_json(out/'selection.json',lock)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(_task,[(a,str(out/a),True,steps) for a in arms]))
    _write_json(out/'rerun.json',dict(procs=2,steps=steps,runs=records,selection=lock))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',choices=ARMS,nargs='+')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rerun-selection',type=Path)
    parser.add_argument('--steps',type=int,default=4800)
    args=parser.parse_args()
    if bool(args.arms)==bool(args.rerun_selection):
        raise ValueError('specify screen arms or an audited A1 selection lock')
    if args.rerun_selection:
        rerun(args.rerun_selection,args.out,args.steps)
    else:
        screen(args.arms,args.out)
