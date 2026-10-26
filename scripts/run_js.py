#!/usr/bin/env python3
"""Registered JS design, random-loop training, validation lock and blind traces."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from run_p1s_f2 import csv_table,write

ROOT=Path('/work/xg24i002/x10041/LoopLora-wt/symf');RUNS=ROOT/'runs/js';REPORTS=ROOT/'reports/js'
WRITERS=('oracle_random','joint_fact');CONDITIONS=('RS','RS+J');SEEDS=(17,29,43)
THRESHOLDS=[0,.001,.003,.01,.03,.1,.2,.3,.5,.7,1,1.5,2,4]


def config(w,condition,seed,smoke=False):
    from looplora.sym.config import resolve_config
    return resolve_config(dict(name=f'{"smoke" if smoke else "js"}-{w}-{condition.replace("+","")}-s{seed}',
        seed=seed,backbone_seed=31415,device='cuda:0',writer=w,variant='M1',activation='relu',
        model=dict(d_e=64,r=32,r_mem=32,reference_r=32,rho=.25,alpha=1.),
        task=dict(name='t2_chain_end',E=12,Q=4,V=1024,n_rel=4,chain_lengths=[8,4],hops=[1,2,3]),
        train=dict(steps=1500 if smoke else 15000,batch_size=64,lr=.003,depths=[4,5,6,7,8]),
        eval=dict(episodes=200 if smoke else 2000,batch_size=64,splits=['val'],seed=4242,depths=list(range(1,17)),
            hops=list(range(1,9)),counterfactuals=False,fixed_depth_sweep=True),runtime=dict(threads=1)))


def lock():
    from analyze_p1s_main import file_hash
    from looplora.infra.seeds import stable_hash
    if (REPORTS/'design.json').exists(): raise FileExistsError('JS design already locked')
    method=json.loads((REPORTS/'method.json').read_text());refs=[]
    for r in json.loads((ROOT/'runs/is/reference_plan.json').read_text())['runs']:
        if r['writer'] not in WRITERS or r['reader'] not in ('M1-L4','M1-mixed'): continue
        if file_hash(Path(r['checkpoint']))!=r['checkpoint_hash']: raise ValueError('reference checkpoint changed')
        refs.append(dict(r,condition=r['reader']))
    expected={(w,c,s) for w in WRITERS for c in ('M1-L4','M1-mixed') for s in SEEDS}
    if len(refs)!=12 or {(r['writer'],r['condition'],r['seed']) for r in refs}!=expected: raise ValueError('JS reference matrix differs')
    stages={}
    for stage in ('smoke','main'):
        entries=[]
        for w in WRITERS:
            for condition in CONDITIONS:
                for seed in ((17,) if stage=='smoke' else SEEDS):
                    c=config(w,condition,seed,stage=='smoke')
                    entries.append(dict(name=c['name'],writer=w,condition=condition,seed=seed,config=c,
                        training_dir=str(RUNS/stage/c['name']),checkpoint=str(RUNS/stage/c['name']/'checkpoint.pt')))
        stages[stage]=dict(runs=entries);write(RUNS/f'{stage}_plan.json',stages[stage])
    write(RUNS/'reference_plan.json',dict(runs=refs))
    design=dict(prereg='docs/prereg/JS.md',writers=list(WRITERS),conditions=list(CONDITIONS),seeds=list(SEEDS),
        model=dict(r=32,r_mem=32,variant='M1',activation='relu',rho=.25),task=dict(E=12,chain_lengths=[8,4],train_hops=[1,2,3]),
        training=dict(steps=15000,batch_size=64,lr=.003,loop_sampling='one independent Uniform{4,5,6,7,8} draw per batch',
            paired_randomness='same data and depth RNG streams across RS and RS+J at each training seed',
            data_seed='derive_seed(training_seed, train_data)',depth_seed='derive_seed(training_seed, train, depths)'),
        penalty=dict(gamma=.9,lambda_J=1,statistic='mean_episode mean_query relu(spectral_radius(J(z_L))-.9)^2',
            method=method,gradient_gate='ReLU derivative at zero is zero; the boolean gate has zero local derivative',
            backend_fixed_before_scores=True),
        stopping=dict(rule='WP-C3 relative_change',max_steps=16,eps=1e-12,thresholds=THRESHOLDS,
            validation_seed=4242,validation_episodes=2000,scope='one threshold per writer/condition, shared over three seeds; references included',
            selection='maximum validation correct count across hops1-8 and seeds; ties fewer total steps, then smaller threshold',
            selected_thresholds_committed_before_confirmation=True),
        confirmation=dict(seed=999003,episodes=2000,hops=list(range(1,9)),depths=list(range(1,17)),radius_depths=[4,8,16],
            after_all_12_training=True,aggregate_after_all_24_traces=True),
        statistics=dict(confidence=.975,resamples=2000,seed=2718,cluster='paired episode; fixed training seeds averaged',
            primary='DeltaS = (RS+J@16 - RS+J@4) - (RS@16 - RS@4), hops1-3',
            removes_overshoot='DeltaS point > 0 and CI low > 0 and S(RS+J) point >= -.01',
            descriptive='RS minus each reference at L4/16, hops1-3,4-8,1-8; hop1 by every L; radius distributions'),
        checkpoints=[{k:r[k] for k in ('name','writer','condition','seed','checkpoint','checkpoint_hash','config_hash')} for r in refs],
        main_plan_hash=stable_hash(stages['main']),smoke_plan_hash=stable_hash(stages['smoke']),reference_plan_hash=stable_hash(dict(runs=refs)),
        budget_node_hours=1.5,execution='IS-tested all-valid support writes; shared reader update unchanged')
    write(REPORTS/'design.json',design)
    csv_table(REPORTS/'training_manifest.csv',[{k:r[k] for k in ('name','writer','condition','seed','checkpoint')}|dict(steps=15000,batch_size=64,lr=.003) for r in stages['main']['runs']])
    print('JS design locked: 12 new runs, 12 read-only references, exact Jacobian method',flush=True)


def design():
    from looplora.infra.seeds import stable_hash
    d=json.loads((REPORTS/'design.json').read_text())
    for stage in ('smoke','main','reference'):
        if stable_hash(json.loads((RUNS/f'{stage}_plan.json').read_text()))!=d[f'{stage}_plan_hash']: raise ValueError('JS locked plan changed')
    return d


def train_one(stage,name):
    import torch
    from torch.nn import functional as F
    from looplora.data.relations import generate
    from looplora.infra.seeds import derive_seed,stable_hash
    from looplora.sym.model import SymbolicModel,make_task_spec
    from looplora.sym.run import source_metadata
    from is_runtime import install
    from js_jacobian import penalty
    d=design();row=next(r for r in json.loads((RUNS/f'{stage}_plan.json').read_text())['runs'] if r['name']==name)
    c=row['config'];out=Path(row['training_dir']);out.mkdir(parents=True,exist_ok=False)
    write(out/'config.json',dict(config=c,config_hash=stable_hash(c)))
    rec=dict(status='running',name=name,condition=row['condition'],writer=row['writer'],seed=row['seed'],config_hash=stable_hash(c),**source_metadata())
    write(out/'record.json',rec)
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();install()
    model=SymbolicModel(c);model.train();options=c['train'];spec=make_task_spec(c)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003,weight_decay=options['weight_decay'])
    depth_rng=np.random.default_rng(derive_seed(c['seed'],'train','depths'));data_seed=derive_seed(c['seed'],'train_data')
    active_batches=0;active_queries=0;max_radius=0.;max_penalty=0.;depth_sum=0;torch.cuda.synchronize();started=time.perf_counter()
    try:
        with (out/'train_log.jsonl').open('w') as log:
            for step in range(options['steps']):
                arrays=generate(spec,'train',step*64,64,data_seed);L=int(depth_rng.choice(options['depths']));depth_sum+=L
                for group in optimizer.param_groups: group['lr']=.003*min(1.,(step+1)/max(1,options['warmup_steps']))
                optimizer.zero_grad(set_to_none=True);prepared=model.prepare(arrays);result=model.read(prepared,L)
                restricted=result['logits'].masked_fill(~prepared.candidate_mask[:,None,:],-torch.inf)
                ce=F.cross_entropy(restricted.flatten(0,1),prepared.answer.flatten())
                reg=ce.new_zeros(());radius=None
                if row['condition']=='RS+J':
                    reg,radius=penalty(model.adapter,prepared,result['z'],d['penalty']['method']['backend'])
                    active=int((radius.detach()>.9).sum());active_queries+=active;active_batches+=active>0
                    max_radius=max(max_radius,float(radius.detach().max()));max_penalty=max(max_penalty,float(reg.detach()))
                loss=ce+reg
                if not torch.isfinite(loss): raise FloatingPointError(f'JS nonfinite loss at step {step}')
                loss.backward();grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),options['grad_clip'],error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                if step==0 or (step+1)%options['log_every']==0 or step+1==options['steps']:
                    log.write(json.dumps(dict(step=step+1,L=L,ce=float(ce.detach()),penalty=float(reg.detach()),
                        total_loss=float(loss.detach()),grad_norm=float(grad_norm),mean_L=depth_sum/(step+1),
                        radius_mean=float(radius.detach().mean()) if radius is not None else None))+ '\n');log.flush()
        summary=dict(steps=options['steps'],train_seconds=time.perf_counter()-started,mean_L=depth_sum/options['steps'],
            peak_memory_bytes=torch.cuda.max_memory_allocated(),penalty_finite=True,active_batches=active_batches,
            active_queries=active_queries,max_radius=max_radius,max_penalty=max_penalty,design_penalty=d['penalty'])
        torch.save(dict(config=c,model=model.state_dict(),summary=summary),out/'checkpoint.pt')
        rec.update(status='complete',**summary);write(out/'record.json',rec)
    except BaseException as error:
        rec.update(status='failed',error=f'{type(error).__name__}: {error}');write(out/'record.json',rec);raise


def train_stage(stage,procs):
    design();plan=json.loads((RUNS/f'{stage}_plan.json').read_text());pending=[r['name'] for r in plan['runs']];running=[];started=time.perf_counter()
    while pending or running:
        for child in list(running):
            code=child.poll()
            if code is not None:
                running.remove(child)
                if code:
                    for other in running: other.terminate()
                    raise RuntimeError(f'JS training child failed {code}')
        while pending and len(running)<procs:
            running.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'train-one','--stage',stage,'--name',pending.pop(0)],env={**os.environ,'OMP_NUM_THREADS':'1'}))
        if running: time.sleep(.2)
    if stage=='smoke':
        timings=[]
        for r in plan['runs']:
            rec=json.loads((Path(r['training_dir'])/'record.json').read_text())
            from looplora.sym.train import load_checkpoint
            r=dict(r,checkpoint_hash=None)
            collect(r,'val',4242,200,RUNS/'smoke_evaluation'/r['name'],True)
            timings.append({k:rec[k] for k in ('writer','condition','train_seconds','peak_memory_bytes','penalty_finite','active_batches','active_queries','max_radius','max_penalty')})
        write(REPORTS/'smoke_timing.json',dict(status='passed',steps=1500,episodes=200,procs=procs,suite_seconds=time.perf_counter()-started,
            timing=timings,scores_read=False,job_id=os.environ.get('PBS_JOBID')))
    print(f'JS {stage} {len(plan["runs"])} runs complete in {time.perf_counter()-started:.2f}s',flush=True)


def all_training():
    from looplora.infra.seeds import stable_hash
    design();plan=json.loads((RUNS/'main_plan.json').read_text())
    expected={(w,c,s) for w in WRITERS for c in CONDITIONS for s in SEEDS}
    if len(plan['runs'])!=12 or {(r['writer'],r['condition'],r['seed']) for r in plan['runs']}!=expected: raise ValueError('JS training matrix differs')
    for r in plan['runs']:
        folder=Path(r['training_dir']);rec=json.loads((folder/'record.json').read_text());c=json.loads((folder/'config.json').read_text())['config']
        if (rec['status']!='complete' or rec['steps']!=15000 or c!=config(r['writer'],r['condition'],r['seed'])
            or c!=r['config'] or rec['config_hash']!=stable_hash(c) or not (folder/'checkpoint.pt').is_file()):
            raise ValueError('JS global training barrier')
    return plan


def collect(row,split,seed,episodes,out,measure_radius):
    import torch
    from looplora.data.relations import generate
    from looplora.sym.model import make_task_spec
    from looplora.sym.train import load_checkpoint
    from looplora.sym.evaluate import stratify_queries,_slice,_snapshot
    from looplora.sym.run import source_metadata
    from looplora.core.stopping import EPS
    from analyze_p1s_main import file_hash
    from is_runtime import install
    from js_jacobian import jacobian,spectral_radius
    d=design();path=Path(row['checkpoint']);before=file_hash(path)
    if row.get('checkpoint_hash') and before!=row['checkpoint_hash']: raise ValueError('JS checkpoint seal differs')
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);torch.cuda.set_device(0);install();model=load_checkpoint(path)
    if model.checkpoint_config!=row['config']: raise ValueError('JS checkpoint config changed')
    c=copy.deepcopy(row['config']);c['eval'].update(seed=seed,episodes=episodes,splits=[split],depths=list(range(1,17)),hops=list(range(1,9)),counterfactuals=False)
    spec=make_task_spec(c,evaluation=True);arrays=stratify_queries(generate(spec,split,0,episodes,seed),spec,c['eval']['hops'],seed)
    chunks={k:[] for k in ('episode_id','query_id','hop','answer','candidate_size','predictions','relative_change','radii')};hashes=[]
    started=time.perf_counter()
    with torch.no_grad():
        for begin in range(0,episodes,64):
            part=_slice(arrays,slice(begin,min(begin+64,episodes)));prepared=model.prepare(part);snapshot=_snapshot(prepared)
            B,Q=prepared.answer.shape;memory=prepared.memory[:,None].expand(-1,Q,-1,-1);z=prepared.u
            predictions=[];changes=[];radii=[]
            for L in range(17):
                if L:
                    previous=z
                    z=prepared.u+model.adapter.cfg.rho*torch.relu(torch.nn.functional.linear(z,model.adapter.R)+model.adapter._query(z,memory,model.adapter.Q,model.adapter.U))
                    changes.append(torch.linalg.vector_norm(z-previous,dim=-1)/torch.linalg.vector_norm(previous,dim=-1).clamp_min(EPS))
                    after=_snapshot(prepared)
                    if after!=snapshot: raise ValueError('JS reader mutated prepared inputs/memory')
                    hashes.append(dict(batch_start=begin,L=L,**after))
                else: changes.append(torch.full_like(prepared.answer,float('nan'),dtype=z.dtype))
                logits=torch.einsum('bqd,bvd->bqv',model.adapter.branch(z),prepared.codes)
                if not torch.isfinite(logits).all() or not torch.isfinite(z).all(): raise FloatingPointError('JS nonfinite evaluation')
                pred=logits.masked_fill(~prepared.candidate_mask[:,None,:],-torch.inf).argmax(-1)
                if not prepared.candidate_mask[:,None,:].expand(-1,Q,-1).gather(-1,pred[...,None]).all(): raise ValueError('JS prediction outside candidates')
                predictions.append(pred)
                if measure_radius and L in (4,8,16):
                    radii.append(spectral_radius(jacobian(model.adapter.R,model.adapter.U,model.adapter.Q,prepared.memory,z,model.adapter.cfg.rho),d['penalty']['method']['backend']))
            if not np.array_equal(prepared.answer.cpu().numpy(),part.answer): raise ValueError('JS prepared targets differ')
            values=dict(episode_id=np.broadcast_to(part.episode_seed[:,None],(B,Q)).ravel(),
                query_id=np.broadcast_to(np.arange(Q,dtype=np.int16),(B,Q)).ravel(),hop=part.q_hop_true.ravel(),answer=part.answer.ravel(),
                candidate_size=np.broadcast_to(prepared.candidate_mask.sum(-1).cpu().numpy()[:,None],(B,Q)).ravel(),
                predictions=torch.stack(predictions,-1).reshape(-1,17).cpu().numpy(),
                relative_change=torch.stack(changes,-1).reshape(-1,17).cpu().numpy(),
                radii=torch.stack(radii,-1).reshape(-1,3).cpu().numpy() if measure_radius else np.empty((B*Q,0),np.float32))
            for k,v in values.items(): chunks[k].append(v)
    trace={k:np.concatenate(v) for k,v in chunks.items()};np.savez_compressed(out/'trace.npz',**trace)
    if file_hash(path)!=before: raise ValueError('JS evaluation changed checkpoint')
    write(out/'evaluation.json',dict(status='complete',aggregation_deferred=True,split=split,seed=seed,episodes=episodes,
        config=c,checkpoint_config=row['config'],checkpoint=str(path),checkpoint_hash=before,trace_hash=file_hash(out/'trace.npz'),
        memory_hashes=hashes,gate0_passed=True,radius_depths=[4,8,16] if measure_radius else [],
        seconds=time.perf_counter()-started,**source_metadata()))


def evaluation_plan():
    from analyze_p1s_main import file_hash
    from looplora.infra.seeds import stable_hash
    trained=all_training();refs=json.loads((RUNS/'reference_plan.json').read_text())['runs'];entries=[]
    for r in [*trained['runs'],*refs]:
        r=copy.deepcopy(r);digest=file_hash(Path(r['checkpoint']))
        if r.get('checkpoint_hash') and digest!=r['checkpoint_hash']: raise ValueError('JS reference changed since design lock')
        r['checkpoint_hash']=digest;entries.append(r)
    plan=dict(runs=entries,training_plan_hash=stable_hash(trained));write(RUNS/'evaluation_plan.json',plan)
    write(RUNS/'training_barrier.json',dict(status='passed',training_stages=12,created_at=datetime.now(timezone.utc).isoformat(),evaluation_plan_hash=stable_hash(plan)))
    csv_table(REPORTS/'checkpoint_manifest.csv',[{k:r[k] for k in ('name','writer','condition','seed','checkpoint','checkpoint_hash')} for r in entries])
    return plan


def collect_stage(stage,procs):
    from looplora.infra.seeds import stable_hash
    if stage=='validation': plan=evaluation_plan()
    else:
        all_training();plan=json.loads((RUNS/'evaluation_plan.json').read_text())
        selection=json.loads((REPORTS/'thresholds.json').read_text())
        if selection['evaluation_plan_hash']!=stable_hash(plan): raise ValueError('JS threshold plan mismatch')
    pending=[r['name'] for r in plan['runs']];running=[]
    while pending or running:
        for child in list(running):
            code=child.poll()
            if code is not None:
                running.remove(child)
                if code:
                    for other in running: other.terminate()
                    raise RuntimeError(f'JS {stage} child failed {code}')
        while pending and len(running)<procs:
            running.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'collect-one','--stage',stage,'--name',pending.pop(0)],env={**os.environ,'OMP_NUM_THREADS':'1'}))
        if running: time.sleep(.2)
    paths=[str(RUNS/stage/r['name']/'trace.npz') for r in plan['runs']]
    if not all(Path(p).is_file() for p in paths): raise ValueError('JS incomplete trace set')
    write(RUNS/f'{stage}_barrier.json',dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),
        evaluation_plan_hash=stable_hash(plan),runs=[r['name'] for r in plan['runs']],traces=paths))
    print(f'JS {stage}: all 24 blind traces complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('lock','train','train-one','collect','collect-one'))
    p.add_argument('--stage',default='main');p.add_argument('--procs',type=int,default=12);p.add_argument('--name');a=p.parse_args()
    if a.action=='lock': lock()
    elif a.action=='train': train_stage(a.stage,a.procs)
    elif a.action=='train-one': train_one(a.stage,a.name)
    elif a.action=='collect': collect_stage(a.stage,a.procs)
    else:
        all_training();row=next(r for r in json.loads((RUNS/'evaluation_plan.json').read_text())['runs'] if r['name']==a.name)
        collect(row,'val' if a.stage=='validation' else 'test',4242 if a.stage=='validation' else 999003,2000,
            RUNS/a.stage/row['name'],a.stage!='validation')

if __name__=='__main__': main()
