#!/usr/bin/env python3
"""Preregistered P2-S pilots, training barrier and read-only reference evaluation."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_p1s_f2 import csv_table, write, write_suite

ROOT=Path('/work/xg24i002/x10041/LoopLora-wt/symf')
RUNS=ROOT/'runs/p2s'
REPORTS=ROOT/'reports/p2s'
SEEDS=(17,29,43)
LRS=(.0003,.001,.003)
DEPTHS=[0,1,2,4,8,16]
WRITERS=('joint','oracle_random')


def families() -> list[dict]:
    rows=[dict(writer=w,variant='B6',r=r,r_mem=r) for w in WRITERS for r in (64,128)]
    rows.extend(dict(writer='joint',variant='B7',r=32,r_mem=m) for m in (64,128))
    rows.extend(dict(writer=w,variant='B8',r=32,r_mem=32) for w in WRITERS)
    return rows


def family_name(row: dict) -> str:
    return f"{row['writer']}-{row['variant']}-r{row['r']}-m{row['r_mem']}"


def config(row: dict,seed: int,steps: int,lr: float,*,smoke: bool=False) -> dict:
    from looplora.sym.config import resolve_config
    name=f"{'smoke' if smoke else 'pilot' if steps==5000 else 'main'}-{family_name(row)}-s{seed}-lr{lr:g}"
    return resolve_config(dict(name=name,writer=row['writer'],variant=row['variant'],seed=seed,
        device='cuda:0',backbone_seed=31415,model=dict(d_e=64,r=row['r'],r_mem=row['r_mem'],
        reference_r=32,rho=.25,alpha=1.),task=dict(name='t2_chain_end',E=12,Q=4,V=1024,
        n_rel=4,chain_lengths=[8,4],hops=[1,2,3]),
        train=dict(steps=steps,batch_size=64,lr=lr),
        eval=dict(episodes=100 if smoke else 1000,batch_size=64,splits=['val'],seed=4242,
                  counterfactuals=False,depths=DEPTHS if smoke else [4 if row['variant']=='B8' else 1],
                  fixed_depth_sweep=smoke),runtime=dict(threads=1)))


def plan() -> None:
    write_suite([config(f,17,500,.001,smoke=True) for f in families()],RUNS/'plans/smoke',1)
    write_suite([config(f,17,5000,lr) for f in families() for lr in LRS],RUNS/'plans/pilot',3)
    write(REPORTS/'design.json',dict(prereg='docs/prereg/P2S.md',families=families(),
        seeds=list(SEEDS),pilot_steps=5000,main_steps=15000,batch_size=64,
        confirmation_seed=888001,confirmation_episodes=2000,confirmation_depths=DEPTHS,
        benchmark=dict(warmup=5,repeats=20,batch_sizes=[64,1],batch_one_queries=8),
        total_budget_node_hours=3,pilot_budget_node_hours=1))


def run(stage: str,shard: int,procs: int) -> None:
    from looplora.sym.run import run_suite
    started=time.perf_counter()
    rc=run_suite(RUNS/'plans'/stage/f'shard-{shard:02d}.json',RUNS/stage/f'shard-{shard:02d}',procs)
    if rc: raise RuntimeError(f'{stage} shard {shard}: {rc}')
    elapsed=time.perf_counter()-started
    if stage=='smoke':
        from looplora.sym.train import load_checkpoint
        from audit_p2s import audit_evaluation
        timings=[]
        for f in families():
            c=config(f,17,500,.001,smoke=True)
            folder=RUNS/'smoke/shard-00'/c['name']
            model=load_checkpoint(folder/'checkpoint.pt')
            if model.checkpoint_config!=c: raise ValueError('smoke checkpoint config changed')
            audit_evaluation(folder,c,samples_only=False)
            rec=json.loads((folder/'record.json').read_text())
            timings.append(dict(**f,train_seconds=rec['train_seconds'],
                step_seconds=rec['train_seconds']/500,peak_memory_bytes=rec['peak_memory_bytes']))
        csv_table(REPORTS/'smoke.csv',timings)
        write(REPORTS/'smoke_timing.json',dict(status='passed',families=8,steps=500,procs=procs,
            suite_seconds=elapsed,pilot_projected_seconds=elapsed*10*3,
            main_projected_seconds=elapsed*30*3,job_id=os.environ.get('PBS_JOBID')))
    print(f'{stage} shard {shard} complete in {elapsed:.2f}s',flush=True)


def select() -> None:
    from looplora.infra.seeds import stable_hash
    all_rows=[]
    for f in families():
        for lr in LRS:
            c=config(f,17,5000,lr)
            index=LRS.index(lr)
            folder=RUNS/'pilot'/f'shard-{index:02d}'/c['name']
            record=json.loads((folder/'record.json').read_text())
            actual=json.loads((folder/'config.json').read_text())['config']
            if record['status']!='complete' or actual!=c or record['config_hash']!=stable_hash(c):
                raise ValueError('pilot incomplete or configuration changed')
            metrics=json.loads((folder/'metrics.json').read_text())
            cells=[r for r in metrics['table'] if r['split']=='val' and r['condition']=='original'
                   and r['hop'] in (1,2,3) and r['L']==c['eval']['depths'][0]]
            if sum(r['n'] for r in cells)!=3000: raise ValueError('pilot query coverage differs')
            all_rows.append(dict(**f,lr=lr,val_accuracy=sum(r['n']*r['accuracy'] for r in cells)/3000,
                depth=c['eval']['depths'][0],train_seconds=record['train_seconds']))
    selected=[]
    for f in families():
        matching=[r for r in all_rows if all(r[k]==v for k,v in f.items())]
        selected.append(max(matching,key=lambda r:(r['val_accuracy'],-r['lr'])))
    csv_table(REPORTS/'pilot_grid.csv',all_rows)
    csv_table(REPORTS/'selected_lrs.csv',selected)
    configs=[config(f,s,15000,f['lr']) for f in selected for s in SEEDS]
    write_suite(configs,RUNS/'plans/main',3)
    entries=[dict(name=c['name'],group='p2s',writer=c['writer'],variant=c['variant'],seed=c['seed'],
                  r=c['model']['r'],r_mem=c['model']['r_mem'],config=c,
                  training_dir=str(RUNS/'main'/f'shard-{i%3:02d}'/c['name'])) for i,c in enumerate(configs)]
    write(RUNS/'main_plan.json',dict(runs=entries,steps=15000,selection_hash=stable_hash(selected)))
    csv_table(REPORTS/'training_manifest.csv',[dict(name=r['name'],writer=r['writer'],variant=r['variant'],
        r=r['r'],r_mem=r['r_mem'],seed=r['seed'],steps=15000,lr=r['config']['train']['lr'],
        training_dir=r['training_dir']) for r in entries])
    print('Locked eight validation LR choices and planned 24 main runs; commit LR table before training',flush=True)


def all_training() -> dict:
    from looplora.infra.seeds import stable_hash
    plan=json.loads((RUNS/'main_plan.json').read_text())
    expected={(family_name(f),s) for f in families() for s in SEEDS}
    if len(plan['runs'])!=24 or {(family_name(r),r['seed']) for r in plan['runs']}!=expected:
        raise ValueError('P2-S training matrix differs')
    with (REPORTS/'selected_lrs.csv').open() as f:
        locked={family_name(dict(r,r=int(r['r']),r_mem=int(r['r_mem']))):float(r['lr'])
                for r in csv.DictReader(f)}
    if set(locked)!={family_name(f) for f in families()}: raise ValueError('locked LR matrix differs')
    for row in plan['runs']:
        folder=Path(row['training_dir'])
        rec=json.loads((folder/'record.json').read_text())
        c=json.loads((folder/'config.json').read_text())['config']
        if (rec['status']!='complete' or c!=row['config'] or rec['config_hash']!=stable_hash(c)
                or c['train']['steps']!=15000 or c['train']['batch_size']!=64
                or c['train']['lr']!=locked[family_name(row)]
                or not (folder/'checkpoint.pt').is_file()):
            raise ValueError(f'P2-S global training barrier: {row["name"]}')
    return plan


def confirmation_config(c: dict) -> dict:
    result=copy.deepcopy(c)
    result['eval'].update(seed=888001,episodes=2000,batch_size=64,splits=['test'],
        counterfactuals=False,swap_successors=False,fixed_depth_sweep=True,
        depths=[4] if c['variant']=='B8' else DEPTHS,hops=list(range(1,9)))
    return result


def evaluation_plan() -> dict:
    from analyze_p1s_main import file_hash
    from looplora.infra.seeds import stable_hash
    trained=all_training()  # Never open or generate P2-S confirmation before this check.
    entries=[dict(r,source='p2s',steps=15000,descriptive_only=False) for r in trained['runs']]
    for source,path,variants in (
        ('wph',ROOT/'reports/p1s/plan.json',('M1','B4','B5','B3')),
        ('f1',ROOT/'reports/p1s-f1/evaluation_plan.json',('M1','B5','B3'))):
        released=json.loads(path.read_text())
        selected=[r for r in released['runs'] if r['writer'] in WRITERS and r['variant'] in variants
                  and (source=='f1' or (r['group']=='primary' and r['r']==32))]
        expected={(w,v,r,s) for w in WRITERS for v in variants
                  for r in ((32,) if source=='wph' else (64,128)) for s in SEEDS}
        if len(selected)!=len(expected) or {(r['writer'],r['variant'],r['r'],r['seed']) for r in selected}!=expected:
            raise ValueError('reference matrix differs')
        for r in selected:
            folder=Path(r['run_dir'] if source=='wph' else r['training_dir'])
            c=json.loads((folder/'config.json').read_text())['config']
            record=json.loads((folder/'record.json').read_text())
            steps=15000 if source=='wph' else 20000
            if (record['status']!='complete' or c!=r['config'] or c['train']['steps']!=steps
                    or c['train']['batch_size']!=64 or c['model']['r']!=r['r']):
                raise ValueError('reference training identity/budget differs')
            entries.append(dict(name=source+'-'+r['name'],source=source,steps=steps,
                descriptive_only=source=='f1',writer=c['writer'],variant=c['variant'],seed=c['seed'],
                r=c['model']['r'],r_mem=c['model']['r_mem'],config=c,training_dir=str(folder)))
    for row in entries:
        row['checkpoint']=str(Path(row['training_dir'])/'checkpoint.pt')
        row['checkpoint_hash']=file_hash(Path(row['checkpoint']))
        row['run_dir']=str(RUNS/'evaluation'/row['name'])
    if len(entries)!=84 or len({r['name'] for r in entries})!=84: raise ValueError('expected 84 evaluations')
    plan=dict(root=str(RUNS),runs=entries,training_plan_hash=stable_hash(trained))
    write(RUNS/'evaluation_plan.json',plan)
    write(RUNS/'training_barrier.json',dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),
        training_plan_hash=stable_hash(trained),training_stages=24,evaluation_plan_hash=stable_hash(plan)))
    csv_table(REPORTS/'checkpoint_manifest.csv',[{k:r[k] for k in
        ('name','source','writer','variant','r','r_mem','seed','steps','descriptive_only','checkpoint','checkpoint_hash')}
        for r in entries])
    return plan


def evaluate_one(name: str) -> None:
    import torch
    from analyze_p1s_main import file_hash
    from looplora.sym.evaluate import evaluate
    from looplora.sym.run import source_metadata
    from looplora.sym.train import load_checkpoint
    all_training()
    plan=json.loads((RUNS/'evaluation_plan.json').read_text())
    run=next(r for r in plan['runs'] if r['name']==name)
    path=Path(run['checkpoint']);out=Path(run['run_dir'])
    if out.exists(): raise FileExistsError(out)
    if file_hash(path)!=run['checkpoint_hash']: raise ValueError('checkpoint changed before evaluation')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    model=load_checkpoint(path)
    if model.checkpoint_config!=run['config']: raise ValueError('checkpoint configuration differs')
    out.mkdir(parents=True)
    rec=dict(name=name,status='running',started_at=datetime.now(timezone.utc).isoformat())
    write(out/'record.json',rec)
    try:
        c=confirmation_config(run['config'])
        evaluate(model,c,out/'confirmation',samples_only=True,
                 provenance=dict(checkpoint=str(path),checkpoint_config=run['config'],**source_metadata()))
        if file_hash(path)!=run['checkpoint_hash']: raise ValueError('evaluation mutated checkpoint')
        rec.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
                   checkpoint_unchanged=True,checkpoint_hash=run['checkpoint_hash'])
    except BaseException as error:
        rec.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally: write(out/'record.json',rec)


def evaluate_all(procs: int) -> None:
    plan=evaluation_plan()
    pending=[r['name'] for r in plan['runs']];running=[]
    while pending or running:
        for p in list(running):
            rc=p.poll()
            if rc is not None:
                running.remove(p)
                if rc:
                    for child in running: child.terminate()
                    raise RuntimeError(f'P2-S evaluation child failed: {rc}')
        while pending and len(running)<procs:
            running.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                'evaluate-one','--name',pending.pop(0)],env={**os.environ,'OMP_NUM_THREADS':'1'}))
        if running: time.sleep(.2)
    from looplora.infra.seeds import stable_hash
    paths=[]
    for r in plan['runs']:
        folder=Path(r['run_dir'])
        record=json.loads((folder/'record.json').read_text())
        meta=json.loads((folder/'confirmation/evaluation.json').read_text())
        if (record['status']!='complete' or not meta['aggregation_deferred']
                or meta['config']!=confirmation_config(r['config'])
                or (folder/'confirmation/metrics.json').exists()):
            raise ValueError('P2-S blind-set barrier failed')
        path=folder/'confirmation/samples.npz'
        if not path.is_file(): raise FileNotFoundError(path)
        paths.append(str(path))
    write(RUNS/'barrier.json',dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),
          evaluation_plan_hash=stable_hash(plan),runs=[r['name'] for r in plan['runs']],confirmation_samples=paths))
    print('P2-S release: 24 trained stages,84 blind confirmation sets',flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','smoke','pilot','select','main','evaluate','evaluate-one'))
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--procs',type=int,default=8)
    p.add_argument('--name')
    args=p.parse_args()
    if args.stage=='plan': plan()
    elif args.stage=='select': select()
    elif args.stage=='evaluate': evaluate_all(args.procs)
    elif args.stage=='evaluate-one': evaluate_one(args.name)
    else: run(args.stage,args.shard,args.procs)


if __name__=='__main__': main()
