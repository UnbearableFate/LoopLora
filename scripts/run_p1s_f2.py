#!/usr/bin/env python3
"""Preregistered F2 pilots and main manifests. Execute through submit.py."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/work/xg24i002/x10041/LoopLora-wt/symf')
RUNS = ROOT / 'runs/p1s/f2'
REPORTS = ROOT / 'reports/p1s-f2'
WRITERS = ('t1_frozen_fact', 'joint_fact')
VARIANTS = ('M1', 'M1-lin', 'B5', 'B5-lin', 'B3', 'B4')
SEEDS = (17, 29, 43)
LRS = (.0003, .001, .003)
DEPTHS = [0, 1, 2, 3, 4, 6, 8, 12, 16]


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def csv_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def config(name: str, writer: str, variant: str, seed: int, steps: int, lr: float,
           *, t1: bool = False, checkpoint: str | None = None) -> dict:
    from looplora.sym.config import resolve_config
    task = dict(name='t1' if t1 else 't2_chain_end', E=12, Q=4, V=1024, n_rel=4,
                head_sharing=.3 if t1 else 0., hops=[1] if t1 else [1, 2, 3])
    if not t1:
        task['chain_lengths'] = [8, 4]
    return resolve_config(dict(name=name, seed=seed, backbone_seed=31415, device='cuda:0',
        writer=writer, variant=variant, model=dict(d_e=64, r=32, r_mem=32, reference_r=32),
        task=task, train=dict(steps=steps, batch_size=64, lr=lr),
        pretrain=dict(steps=steps, batch_size=64, lr=lr, writer_checkpoint=checkpoint,
                      task=dict(head_sharing=.3, n_rel=4)),
        eval=dict(episodes=1000, batch_size=64, splits=['val'], seed=4242,
                  depths=[1] if t1 else DEPTHS, counterfactuals=False,
                  fixed_depth_sweep=not t1 and steps==15000), runtime=dict(threads=1)))


def write_suite(configs: list[dict], folder: Path, shards: int) -> None:
    if folder.exists():
        raise FileExistsError(folder)
    entries = []
    for c in configs:
        path = folder / (c['name']+'.json')
        write(path, c)
        entries.append(dict(name=c['name'], config=str(path)))
    for i in range(shards):
        write(folder / f'shard-{i:02d}.json', entries[i::shards])
    write(folder / 'manifest.json', dict(names=[c['name'] for c in configs], shards=shards))


def plan_writers() -> None:
    configs = [config(f'p-writer-lr{lr:g}', 'joint_fact', 'B4', 17, 5000, lr, t1=True)
               for lr in LRS]
    write_suite(configs, RUNS/'plans/pilot-writers', 1)


def complete_rows(stage: str) -> list[dict]:
    plan = json.loads((RUNS/'plans'/stage/'manifest.json').read_text())
    paths = list((RUNS/stage).glob('shard-*/*/record.json'))
    rows = []
    for path in paths:
        record = json.loads(path.read_text())
        if record['status'] != 'complete':
            raise ValueError(f'incomplete {path}')
        c = json.loads(path.with_name('config.json').read_text())['config']
        metrics = json.loads(path.with_name('metrics.json').read_text())
        if c['eval']['splits'] != ['val'] or c['eval']['seed'] != 4242:
            raise ValueError('selection must use validation only')
        depth = 1 if c['variant']=='B4' else 4
        cells = [r for r in metrics['table'] if r['condition']=='original' and r['split']=='val'
                 and r['L']==depth and r['hop'] in (1, 2, 3)]
        n = sum(r['n'] for r in cells)
        rows.append(dict(name=c['name'], writer=c['writer'], variant=c['variant'], seed=c['seed'],
                         lr=c['train']['lr'], steps=c['train']['steps'], depth=depth,
                         val_accuracy=sum(r['n']*r['accuracy'] for r in cells)/n,
                         train_seconds=record['train_seconds'], run_dir=str(path.parent),
                         config_hash=record['config_hash'], git_commit=record['git_commit']))
    if len(rows)!=len(plan['names']) or {r['name'] for r in rows}!=set(plan['names']):
        raise ValueError(f'{stage} matrix incomplete')
    return rows


def best(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (r['val_accuracy'], -r['lr']))


def select_writer() -> None:
    rows = complete_rows('pilot-writers')
    selected = best(rows)
    write(REPORTS/'pilot_writer_selection.json', selected)
    csv_table(REPORTS/'pilot_writer_grid.csv', rows)
    checkpoint = str(Path(selected['run_dir'])/'writer.pt')
    configs = [config(f'p-reader-{w}-{v}-lr{lr:g}', w, v, 17, 5000, lr,
                      checkpoint=checkpoint if w=='t1_frozen_fact' else None)
               for w in WRITERS for v in VARIANTS for lr in LRS]
    write_suite(configs, RUNS/'plans/pilot-readers', 4)
    print('T1 writer LR selected; planned 36 validation reader pilots', flush=True)


def select_readers() -> None:
    rows = complete_rows('pilot-readers')
    selected = [best([r for r in rows if r['writer']==w and r['variant']==v])
                for w in WRITERS for v in VARIANTS]
    writer = json.loads((REPORTS/'pilot_writer_selection.json').read_text())
    write(REPORTS/'pilot_selection.json', dict(writer_lr=writer['lr'], readers=selected,
                                            rule='maximum validation accuracy; smaller LR breaks ties'))
    csv_table(REPORTS/'pilot_reader_grid.csv', rows)
    csv_table(REPORTS/'selected_lrs.csv', [dict(writer=r['writer'], variant=r['variant'],
                                             lr=r['lr'], val_accuracy=r['val_accuracy']) for r in selected])
    writers = [config(f'f2-writer-s{s}', 'joint_fact', 'B4', s, 15000, writer['lr'], t1=True)
               for s in SEEDS]
    write_suite(writers, RUNS/'plans/main-writers', 1)
    readers = []
    entries = []
    for c in writers:
        entries.append(dict(name=c['name'], group='writer', writer=c['writer'], variant='B4',
                            seed=c['seed'], r=32, config=c,
                            training_dir=str(RUNS/'main-writers/shard-00'/c['name']),
                            run_dir=str(RUNS/'evaluation'/c['name'])))
    for w in WRITERS:
        for s in SEEDS:
            checkpoint = str(RUNS/'main-writers/shard-00'/f'f2-writer-s{s}'/'writer.pt')
            for v in VARIANTS:
                lr = next(r['lr'] for r in selected if r['writer']==w and r['variant']==v)
                c = config(f'f2-main-{w}-{v}-s{s}', w, v, s, 15000, lr,
                           checkpoint=checkpoint if w=='t1_frozen_fact' else None)
                i = len(readers)
                readers.append(c)
                entries.append(dict(name=c['name'], group='primary', writer=w, variant=v, seed=s,
                                    r=32, config=c, training_dir=str(RUNS/'main-readers'/f'shard-{i%4:02d}'/c['name']),
                                    run_dir=str(RUNS/'evaluation'/c['name'])))
    write_suite(readers, RUNS/'plans/main-readers', 4)
    write(RUNS/'main_plan.json', dict(root=str(RUNS), steps=15000, runs=entries,
                                     prereg='docs/prereg/P1S-F2.md'))
    csv_table(REPORTS/'main_manifest.csv', [dict(name=r['name'], group=r['group'],
              writer=r['writer'], variant=r['variant'], seed=r['seed'], steps=15000,
              lr=r['config']['train']['lr'], training_dir=r['training_dir']) for r in entries])
    print('Locked selected LRs, 3 fresh main writers and 36 main readers', flush=True)


def diagnose_writer() -> None:
    from diagnose_p1s_writer import _diagnose_run
    from looplora.sym.pilot import loss_diagnostic
    selected=json.loads((REPORTS/'pilot_writer_selection.json').read_text())
    row=dict(selected,r=32,E=12,accuracy=selected['val_accuracy'],n=1000,
             **loss_diagnostic(Path(selected['run_dir'])/'train_log.jsonl'))
    result=_diagnose_run(row,REPORTS)
    write(REPORTS/'pilot_writer_diagnosis.json',result)
    csv_table(REPORTS/'pilot_writer_diagnosis.csv',
              [dict(stratum=n,**values) for n,values in result['head_multiplicity'].items()])


def run(stage: str, shard: int, procs: int) -> None:
    from looplora.sym.run import run_suite
    rc = run_suite(RUNS/'plans'/stage/f'shard-{shard:02d}.json', RUNS/stage/f'shard-{shard:02d}', procs)
    if rc:
        raise SystemExit(rc)


def all_training() -> dict:
    from looplora.infra.seeds import stable_hash
    plan = json.loads((RUNS/'main_plan.json').read_text())
    if len(plan['runs'])!=39 or sum(r['group']=='primary' for r in plan['runs'])!=36:
        raise ValueError('F2 requires 3 writers and 36 readers')
    for run in plan['runs']:
        folder = Path(run['training_dir'])
        rec = json.loads((folder/'record.json').read_text())
        c = json.loads((folder/'config.json').read_text())['config']
        if (rec['status']!='complete' or c!=run['config'] or rec['config_hash']!=stable_hash(c)
                or c['train']['steps']!=15000 or not (folder/'checkpoint.pt').is_file()):
            raise ValueError(f'F2 global training barrier failed: {run["name"]}')
    return plan


def evaluate_one(name: str) -> None:
    import torch
    from looplora.infra.seeds import stable_hash
    from looplora.sym.evaluate import evaluate
    from looplora.sym.run import source_metadata
    from looplora.sym.train import load_checkpoint
    plan = all_training()
    entry = next(r for r in plan['runs'] if r['name']==name)
    source, out = Path(entry['training_dir']), Path(entry['run_dir'])
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    for filename in ('train_log.jsonl', 'checkpoint.pt', 'writer.pt'):
        (out/filename).symlink_to(source/filename)
    torch.set_num_threads(1)
    torch.cuda.init()
    torch.cuda.set_device(0)
    model = load_checkpoint(source/'checkpoint.pt')
    c = copy.deepcopy(model.checkpoint_config)
    if c!=entry['config']: raise ValueError('F2 checkpoint configuration differs')
    provenance = dict(checkpoint=str(source/'checkpoint.pt'), checkpoint_config=c, **source_metadata())
    rec = json.loads((source/'record.json').read_text())
    rec.update(status='running', config=c, training_complete=True,
               evaluation_source=source_metadata(), confirmation_samples_written=False)
    write(out/'record.json', rec)
    try:
        if entry['group']=='writer':
            metrics, timing = evaluate(model,c,out/'writer_validation',provenance=provenance)
            write(out/'writer_validation/timing.json', timing)
            rec.update(samples=str(out/'writer_validation/samples.npz'), scores=metrics['table'])
        else:
            screen = copy.deepcopy(c)
            screen['eval'].update(episodes=2000, splits=['test','test_unseen'], seed=4242,
                                  counterfactuals=True, swap_successors=True)
            metrics, timing = evaluate(model, screen, out/'screening',provenance=provenance)
            write(out/'screening/timing.json',timing)
            confirmation = copy.deepcopy(screen)
            confirmation['eval'].update(splits=['test'],seed=777001,counterfactuals=False)
            _, confirm_timing = evaluate(model,confirmation,out/'confirmation',samples_only=True,
                                         provenance=provenance)
            rec.update(samples=str(out/'screening/samples.npz'), scores=metrics['table'],
                       confirmation_samples_written=True,
                       confirmation_samples=str(out/'confirmation/samples.npz'),
                       eval_samples=sum(r['n'] for r in metrics['table']),
                       latency=dict(screening=timing, confirmation=confirm_timing))
        rec.update(status='complete', evaluation_finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        rec.update(status='failed', failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(out/'record.json',rec)


def evaluation_shard(shard: int, procs: int) -> None:
    from looplora.infra.seeds import stable_hash
    plan = all_training()
    write(RUNS/'training_barriers'/f'shard-{shard:02d}.json',dict(status='passed',
          plan_hash=stable_hash(plan),runs=[r['name'] for r in plan['runs']],
          created_at=datetime.now(timezone.utc).isoformat()))
    pending = [r['name'] for r in plan['runs'][shard::4]]
    running = []
    while pending or running:
        for p in list(running):
            rc = p.poll()
            if rc is not None:
                running.remove(p)
                if rc:
                    for child in running: child.terminate()
                    raise RuntimeError(f'F2 evaluation child failed: {rc}')
        while pending and len(running)<procs:
            running.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                       'evaluate-one','--name',pending.pop(0)],env={**os.environ,'OMP_NUM_THREADS':'1'}))
        if running: time.sleep(.2)


def seal() -> None:
    from looplora.infra.seeds import stable_hash
    plan = all_training()
    paths = []
    for r in plan['runs']:
        folder = Path(r['run_dir'])
        rec = json.loads((folder/'record.json').read_text())
        if rec['status']!='complete': raise ValueError('F2 evaluation incomplete')
        if r['group']=='writer': continue
        meta = json.loads((folder/'confirmation/evaluation.json').read_text())
        settings = meta['config']['eval']
        if (not meta['aggregation_deferred'] or 'table' in meta or 'diagnostics' in meta
                or settings['seed']!=777001 or settings['episodes']!=2000
                or settings['splits']!=['test'] or settings['counterfactuals']
                or settings['depths']!=DEPTHS or settings['hops']!=list(range(1,9))
                or meta['checkpoint_config']!=r['config']
                or (folder/'confirmation/metrics.json').exists()):
            raise ValueError('F2 confirmation embargo/protocol mismatch')
        sample=folder/'confirmation/samples.npz'
        if not sample.is_file(): raise FileNotFoundError(sample)
        paths.append(str(sample))
    write(RUNS/'barrier.json',dict(status='passed',plan_hash=stable_hash(plan),
          created_at=datetime.now(timezone.utc).isoformat(),runs=[r['name'] for r in plan['runs']],
          confirmation_samples=paths))
    print('F2 release: 39 completed stages and 36 blind confirmation sets',flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('plan-writers','pilot-writers','select-writer','pilot-readers',
                                      'select-readers','main-writers','main-readers',
                                      'evaluate','evaluate-one','seal','diagnose-writer'))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--name')
    args = ap.parse_args()
    if args.stage=='plan-writers': plan_writers()
    elif args.stage=='select-writer': select_writer()
    elif args.stage=='select-readers': select_readers()
    elif args.stage=='evaluate': evaluation_shard(args.shard,args.procs)
    elif args.stage=='evaluate-one': evaluate_one(args.name)
    elif args.stage=='seal': seal()
    elif args.stage=='diagnose-writer': diagnose_writer()
    else: run(args.stage,args.shard,args.procs)


if __name__=='__main__':
    main()
