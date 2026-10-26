#!/usr/bin/env python3
"""Registered IS design lock, timed smoke, trained-at-load barrier and evaluation."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_p1s_f2 import csv_table, write, write_suite

ROOT=Path('/work/xg24i002/x10041/LoopLora-wt/symf')
RUNS=ROOT/'runs/is'
REPORTS=ROOT/'reports/is'
SEEDS=(17,29,43)
WRITERS=('oracle_random','joint_fact')
ALL_WRITERS=(*WRITERS,'joint')
LAYOUTS={6:[4,2],12:[8,4],18:[8,8,2],24:[8,8,8],36:[8,8,8,8,4]}
READERS=('M1-mixed','M1-L4','B8','B4','B5','B3')


def reference_rows() -> list[dict]:
    from analyze_p1s_main import file_hash
    from looplora.infra.seeds import stable_hash
    rows=[]
    for source,path in [('wph',ROOT/'reports/p1s/plan.json'),('f2',ROOT/'runs/p1s/f2/main_plan.json'),
                        ('p2s',ROOT/'runs/p2s/main_plan.json'),('p2sf1',ROOT/'runs/p2s-f1/main_plan.json')]:
        for row in json.loads(path.read_text())['runs']:
            w,v=row['writer'],row['variant']
            reader=None
            if source=='wph' and row['group']=='primary' and row['r']==32 and w in ('joint','oracle_random') and v in ('M1','B4','B5','B3'):
                reader='M1-mixed' if v=='M1' else v
            elif source=='f2' and row['group']=='primary' and w=='joint_fact' and v in ('M1','B4','B5','B3'):
                reader='M1-mixed' if v=='M1' else v
            elif source=='p2s' and w in ('joint','oracle_random') and v=='B8': reader='B8'
            elif source=='p2sf1' and ((v=='M1' and w in ALL_WRITERS) or (v=='B8' and w=='joint_fact')):
                reader='M1-L4' if v=='M1' else v
            if reader is None: continue
            folder=Path(row['run_dir'] if source=='wph' else row['training_dir'])
            c=json.loads((folder/'config.json').read_text())['config'];rec=json.loads((folder/'record.json').read_text())
            if (rec['status']!='complete' or c!=row['config'] or c['train']['steps']!=15000
                    or c['train']['batch_size']!=64 or c['model']['r']!=32 or c['model']['r_mem']!=32
                    or c['task']['E']!=12 or c['task']['chain_lengths']!=[8,4]):
                raise ValueError('E12 reference identity/budget differs')
            checkpoint=folder/'checkpoint.pt'
            rows.append(dict(name=f'E12-{w}-{reader}-s{c["seed"]}',source=source,writer=w,reader=reader,
                variant=v,seed=c['seed'],train_E=12,config=c,training_dir=str(folder),
                checkpoint=str(checkpoint),checkpoint_hash=file_hash(checkpoint),config_hash=stable_hash(c)))
    expected={(w,v,s) for w in ALL_WRITERS for v in READERS for s in SEEDS}
    if len(rows)!=54 or {(r['writer'],r['reader'],r['seed']) for r in rows}!=expected:
        raise ValueError('expected all 54 E12 checkpoints')
    return rows


def new_config(writer: str,variant: str,E: int,seed: int,lr: float,steps: int=15000,smoke: bool=False) -> dict:
    from looplora.sym.config import resolve_config
    return resolve_config(dict(name=f'{"smoke" if smoke else "main"}-E{E}-{writer}-{variant}-s{seed}',
        seed=seed,backbone_seed=31415,device='cuda:0',writer=writer,variant=variant,
        model=dict(d_e=64,r=32,r_mem=32,reference_r=32,rho=.25,alpha=1.),
        task=dict(name='t2_chain_end',E=E,Q=4,V=1024,n_rel=4,chain_lengths=LAYOUTS[E],hops=[1,2,3]),
        train=dict(steps=steps,batch_size=64,lr=lr,depths=[1,2,4] if variant=='M1' else [4] if variant=='B8' else [1]),
        eval=dict(episodes=200 if smoke else 1000,batch_size=64,splits=['val'],seed=4242,
            counterfactuals=False,depths=list(range(1,9)) if smoke and variant=='M1' else [4 if variant in ('M1','B8') else 1],
            hops=list(range(1,9)) if smoke else [1,2,3],fixed_depth_sweep=True),runtime=dict(threads=1)))


def lock() -> None:
    from looplora.data.relations import TaskSpec,generate
    from looplora.infra.seeds import stable_hash
    if (REPORTS/'design.json').exists(): raise FileExistsError('IS design already exists')
    structural=[]
    for E,chains in LAYOUTS.items():
        spec=TaskSpec(name='t2_chain_end',E=E,Q=4,chain_lengths=chains,hops=range(1,min(8,max(chains))+1))
        sample=generate(spec,'val',0,2,4242)
        structural.append(dict(E=E,chain_lengths=chains,eval_hops=list(spec.hops),validated_rows=len(sample.episode_seed)))
    update=TaskSpec(name='t3_update',base='t2_chain_end',E=14,Q=4,chain_lengths=[8,4],hops=[1,2,3,4])
    generate(update,'val',0,2,4242)
    references=reference_rows();lrs=[]
    for w in WRITERS:
        for v in ('M1','B8','B4'):
            chosen=[r for r in references if r['writer']==w and r['reader']==('M1-mixed' if v=='M1' else v)]
            values={r['config']['train']['lr'] for r in chosen}
            if len(chosen)!=3 or len(values)!=1: raise ValueError('E12 LR differs across seeds')
            lrs.append(dict(writer=w,variant=v,lr=values.pop(),source=chosen[0]['source']))
    configs=[new_config(row['writer'],row['variant'],E,s,row['lr']) for E in (24,36) for row in lrs for s in SEEDS]
    write_suite(configs,RUNS/'plans/main',4)
    main=[dict(name=c['name'],writer=c['writer'],variant=c['variant'],reader='M1-mixed' if c['variant']=='M1' else c['variant'],
        seed=c['seed'],train_E=c['task']['E'],config=c,training_dir=str(RUNS/'main'/f'shard-{i%4:02d}'/c['name'])) for i,c in enumerate(configs)]
    smoke=[new_config(w,'M1',36,17,next(r['lr'] for r in lrs if r['writer']==w and r['variant']=='M1'),1500,True) for w in WRITERS]
    write_suite(smoke,RUNS/'plans/smoke',1)
    write(RUNS/'reference_plan.json',dict(runs=references))
    write(RUNS/'main_plan.json',dict(runs=main,steps=15000))
    design=dict(prereg='docs/prereg/IS.md',writers=list(WRITERS),descriptive_writer='joint',seeds=list(SEEDS),
        layouts=structural,r=32,r_mem=32,main_steps=15000,batch_size=64,locked_lrs=lrs,
        confirmation=dict(seed=999001,episodes=2000,split='test',after_all_36_training=True),
        same_checkpoint_depths=dict(M1_mixed=list(range(1,9)),M1_L4=[4],B8=[4],B4=[1],B5=[4],B3=[4]),
        t3_update=dict(base='t2_chain_end',E=14,current_edges=12,chain_lengths=[8,4],n_updates=2,
            update_query_rate=.5,eval_hops=[1,2,3,4],depths=list(range(1,9)),B8_depths=[4],
            query_protocol='one query per hop from unchanged current support; generator defaults otherwise'),
        IS_E4=dict(updated_path='current path includes a key with two chronological facts',
            obsolete_graph='first occurrence per key; all repeated keys reverted to obsolete tails',
            stale_terminal='terminal from query start in obsolete graph; generator obsolete tails are terminal',
            stale_rate='prediction equals stale terminal, denominator updated-path queries only',
            untouched_stale_rate=None,breakdowns=['writer','reader','hop','L','training_seed','updated_path'],
            unavailable_strata='n=0, rates null; no imputation'),
        best_L='highest pooled hops-2/3 accuracy of mixed M1 across L1..8; ties choose smaller L; descriptive only',
        statistics=dict(confidence=.975,resamples=2000,seed=2718,within_load='paired episodes shared across fixed training seeds',
            across_load='independent bootstrap streams keyed by load; DeltaD uses independent D36 minus D12 draws',
            capability_rule='report largest trained E meeting difference >= .03 and CI low > 0 separately per writer and reader'),
        deviations=['Inherited E12 LRs without a new pilot, as preregistered.',
            'Q3 E=14 total stored facts preserves registered 12 current edges (8,4) plus two obsolete facts; generator counts obsolete facts in E.',
            'Q3 hops1-4 preserves default mixed update_query_rate=.5: (8,4) cannot supply updated and untouched chains at every hop above4.',
            'Q3 B8 is defined only at L4; other readers scanned L1-8. E6 has no hops6-8 and those summaries are unavailable.'],
        checkpoints=[{k:r[k] for k in ('name','writer','reader','variant','seed','train_E','checkpoint','checkpoint_hash','config_hash')} for r in references],
        main_plan_hash=stable_hash(dict(runs=main,steps=15000)),reference_plan_hash=stable_hash(dict(runs=references)),
        budget_node_hours=2)
    write(REPORTS/'design.json',design)
    csv_table(REPORTS/'training_manifest.csv',[dict(name=r['name'],writer=r['writer'],reader=r['reader'],seed=r['seed'],
        train_E=r['train_E'],steps=15000,lr=r['config']['train']['lr'],training_dir=r['training_dir']) for r in main])
    print('IS design locked structurally: 54 reference checkpoints, 36 main runs, 6 inherited LRs',flush=True)


def design() -> dict:
    from looplora.infra.seeds import stable_hash
    result=json.loads((REPORTS/'design.json').read_text())
    for name in ('main','reference'):
        plan=json.loads((RUNS/f'{name}_plan.json').read_text())
        if stable_hash(plan)!=result[f'{name}_plan_hash']: raise ValueError('IS locked plan changed')
    return result


def run(stage: str,shard: int,procs: int) -> None:
    from looplora.sym.run import run_suite
    design();started=time.perf_counter()
    rc=run_suite(RUNS/'plans'/stage/f'shard-{shard:02d}.json',RUNS/stage/f'shard-{shard:02d}',procs)
    if rc: raise RuntimeError(f'IS {stage} failed: {rc}')
    if stage=='smoke':
        records=[json.loads(p.read_text()) for p in (RUNS/'smoke/shard-00').glob('*/record.json')]
        if len(records)!=2 or any(r['status']!='complete' for r in records): raise ValueError('smoke incomplete')
        write(REPORTS/'smoke_timing.json',dict(status='passed',suite_seconds=time.perf_counter()-started,
            steps=1500,episodes=200,writers=list(WRITERS),procs=procs,job_id=os.environ.get('PBS_JOBID'),
            timing=[{k:r[k] for k in ('writer','train_seconds','peak_memory_bytes')} for r in records],
            metrics_read=False))
    print(f'IS {stage} shard {shard} complete in {time.perf_counter()-started:.2f}s',flush=True)


def packed(stage: str,shard: int,procs: int) -> None:
    design();plan=json.loads((RUNS/'main_plan.json').read_text())
    rows=([r for r in plan['runs'] if r['seed']==17] if stage=='packing' else plan['runs'][shard::3])
    jobs=[]
    for r in rows:
        c=copy.deepcopy(r['config']);out=Path(r['training_dir'])
        if stage=='packing':
            c['name']='packing-'+c['name'];c['train']['steps']=150
            c['eval'].update(episodes=20)
            out=RUNS/'packing'/c['name']
        path=RUNS/'packed_configs'/f"{c['name']}.json";write(path,c)
        jobs.append((path,out))
    started=time.perf_counter();running=[]
    while jobs or running:
        for child in list(running):
            rc=child.poll()
            if rc is not None:
                running.remove(child)
                if rc:
                    for other in running: other.terminate()
                    raise RuntimeError(f'IS packed training failed: {rc}')
        while jobs and len(running)<procs:
            path,out=jobs.pop(0)
            running.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'train-one',
                '--config',str(path),'--out',str(out)],env={**os.environ,'OMP_NUM_THREADS':'1'}))
        if running: time.sleep(.2)
    if stage=='packing':
        records=[json.loads(p.read_text()) for p in (RUNS/'packing').glob('*/record.json')]
        if len(records)!=12 or any(r['status']!='complete' for r in records): raise ValueError('packing matrix incomplete')
        write(REPORTS/'packing_timing.json',dict(status='passed',suite_seconds=time.perf_counter()-started,
            steps=150,procs=procs,job_id=os.environ.get('PBS_JOBID'),metrics_read=False,
            timing=[{k:r[k] for k in ('writer','variant','train_seconds','peak_memory_bytes')} for r in records],
            projected_main_node_hours=max(r['train_seconds'] for r in records)*100*3/3600))
    print(f'IS {stage} shard {shard}: {len(rows)} runs in {time.perf_counter()-started:.2f}s',flush=True)


def train_one(config_path: str,out: str) -> None:
    from is_runtime import install
    from looplora.sym.run import run_one
    install();run_one(json.loads(Path(config_path).read_text()),out)


def all_training() -> dict:
    from looplora.infra.seeds import stable_hash
    locked=design();plan=json.loads((RUNS/'main_plan.json').read_text())
    expected={(w,v,E,s) for w in WRITERS for v in ('M1','B8','B4') for E in (24,36) for s in SEEDS}
    if len(plan['runs'])!=36 or {(r['writer'],r['variant'],r['train_E'],r['seed']) for r in plan['runs']}!=expected:
        raise ValueError('IS training matrix differs')
    for r in plan['runs']:
        folder=Path(r['training_dir']);rec=json.loads((folder/'record.json').read_text())
        c=json.loads((folder/'config.json').read_text())['config']
        lr=next(v['lr'] for v in locked['locked_lrs'] if v['writer']==r['writer'] and v['variant']==r['variant'])
        if (rec['status']!='complete' or c!=r['config'] or c!=new_config(r['writer'],r['variant'],r['train_E'],r['seed'],lr)
            or rec['config_hash']!=stable_hash(c) or not (folder/'checkpoint.pt').is_file()):
            raise ValueError(f'IS training barrier: {r["name"]}')
    return plan


def evaluation_config(row: dict,E: int,update: bool=False,episodes: int=2000,seed: int=999001,split: str='test') -> dict:
    c=copy.deepcopy(row['config'])
    c['task'].update(name='t3_update' if update else 't2_chain_end',E=14 if update else E,
        chain_lengths=[8,4] if update else LAYOUTS[E],hops=[1,2,3])
    if update: c['task'].update(base='t2_chain_end',n_updates=2,update_query_rate=.5)
    depths=([4] if row['reader']=='B8' else list(range(1,9))) if update else (
        list(range(1,9)) if row['reader']=='M1-mixed' else [1] if row['reader']=='B4' else [4])
    c['eval'].update(seed=seed,episodes=episodes,batch_size=64,splits=[split],hops=[1,2,3,4] if update else list(range(1,min(8,max(LAYOUTS[E]))+1)),
        depths=depths,fixed_depth_sweep=True,counterfactuals=False,swap_successors=False)
    return c


def evaluation_plan() -> dict:
    from analyze_p1s_main import file_hash
    from looplora.infra.seeds import stable_hash
    trained=all_training();refs=json.loads((RUNS/'reference_plan.json').read_text())['runs'];entries=[]
    for r in refs:
        if file_hash(Path(r['checkpoint']))!=r['checkpoint_hash']: raise ValueError('locked E12 checkpoint changed')
        for E in LAYOUTS: entries.append(dict(r,eval_E=E,update=False,name=f'{r["name"]}-evalE{E}'))
        entries.append(dict(r,eval_E=12,update=True,name=f'{r["name"]}-update'))
    for r in trained['runs']:
        checkpoint=Path(r['training_dir'])/'checkpoint.pt'
        entries.append(dict(r,checkpoint=str(checkpoint),checkpoint_hash=file_hash(checkpoint),eval_E=r['train_E'],update=False))
    for r in entries: r['run_dir']=str(RUNS/'evaluation'/r['name'])
    if len(entries)!=360 or len({r['name'] for r in entries})!=360: raise ValueError('IS evaluation matrix differs')
    plan=dict(runs=entries,training_plan_hash=stable_hash(trained))
    write(RUNS/'evaluation_plan.json',plan)
    write(RUNS/'training_barrier.json',dict(status='passed',training_stages=36,created_at=datetime.now(timezone.utc).isoformat(),
        evaluation_plan_hash=stable_hash(plan),training_plan_hash=stable_hash(trained)))
    return plan


def evaluate_one(name: str) -> None:
    import torch
    from analyze_p1s_main import file_hash
    from looplora.sym.evaluate import evaluate
    from looplora.sym.run import source_metadata
    from looplora.sym.train import load_checkpoint
    from is_runtime import install
    install()
    all_training();plan=json.loads((RUNS/'evaluation_plan.json').read_text())
    row=next(r for r in plan['runs'] if r['name']==name);path=Path(row['checkpoint']);out=Path(row['run_dir'])
    if out.exists(): raise FileExistsError(out)
    if file_hash(path)!=row['checkpoint_hash']: raise ValueError('checkpoint changed before evaluation')
    torch.set_num_threads(1);torch.cuda.set_device(0);model=load_checkpoint(path)
    if model.checkpoint_config!=row['config']: raise ValueError('checkpoint configuration differs')
    out.mkdir(parents=True);write(out/'record.json',dict(status='running',name=name))
    try:
        c=evaluation_config(row,row['eval_E'],row['update'])
        evaluate(model,c,out/'confirmation',samples_only=True,
            provenance=dict(checkpoint=str(path),checkpoint_config=row['config'],**source_metadata()))
        if file_hash(path)!=row['checkpoint_hash']: raise ValueError('evaluation changed checkpoint')
        write(out/'record.json',dict(status='complete',checkpoint_unchanged=True))
    except BaseException as error:
        write(out/'record.json',dict(status='failed',error=f'{type(error).__name__}: {error}'));raise


def evaluate_all(procs: int) -> None:
    from looplora.infra.seeds import stable_hash
    plan=evaluation_plan();pending=[r['name'] for r in plan['runs']];running=[]
    while pending or running:
        for child in list(running):
            rc=child.poll()
            if rc is not None:
                running.remove(child)
                if rc:
                    for other in running: other.terminate()
                    raise RuntimeError(f'IS evaluation child failed: {rc}')
        while pending and len(running)<procs:
            running.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'evaluate-one','--name',pending.pop(0)],
                env={**os.environ,'OMP_NUM_THREADS':'1'}))
        if running: time.sleep(.2)
    paths=[]
    for r in plan['runs']:
        folder=Path(r['run_dir']);rec=json.loads((folder/'record.json').read_text());meta=json.loads((folder/'confirmation/evaluation.json').read_text())
        if (rec['status']!='complete' or not meta['aggregation_deferred'] or meta['config']!=evaluation_config(r,r['eval_E'],r['update'])
            or (folder/'confirmation/metrics.json').exists()): raise ValueError('IS sample barrier failed')
        path=folder/'confirmation/samples.npz'
        if not path.is_file(): raise FileNotFoundError(path)
        paths.append(str(path))
    write(RUNS/'barrier.json',dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),
        evaluation_plan_hash=stable_hash(plan),runs=[r['name'] for r in plan['runs']],samples=paths))
    print('IS release: 36 complete training stages and 360 blind sample sets',flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('lock','smoke','main','packing','packed-main','train-one','evaluate','evaluate-one'))
    p.add_argument('--shard',type=int,default=0);p.add_argument('--procs',type=int,default=9);p.add_argument('--name');p.add_argument('--config');p.add_argument('--out')
    a=p.parse_args()
    if a.stage=='lock': lock()
    elif a.stage=='train-one': train_one(a.config,a.out)
    elif a.stage in ('packing','packed-main'): packed(a.stage,a.shard,a.procs)
    elif a.stage=='evaluate': evaluate_all(a.procs)
    elif a.stage=='evaluate-one': evaluate_one(a.name)
    else: run(a.stage,a.shard,a.procs)

if __name__=='__main__': main()
