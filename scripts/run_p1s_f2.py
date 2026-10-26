#!/usr/bin/env python3
"""Preregistered F2 pilots and main manifests. Execute through submit.py."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

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


def run(stage: str, shard: int, procs: int) -> None:
    from looplora.sym.run import run_suite
    rc = run_suite(RUNS/'plans'/stage/f'shard-{shard:02d}.json', RUNS/stage/f'shard-{shard:02d}', procs)
    if rc:
        raise SystemExit(rc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('plan-writers','pilot-writers','select-writer','pilot-readers',
                                      'select-readers','main-writers','main-readers'))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--procs', type=int, default=8)
    args = ap.parse_args()
    if args.stage=='plan-writers': plan_writers()
    elif args.stage=='select-writer': select_writer()
    elif args.stage=='select-readers': select_readers()
    else: run(args.stage,args.shard,args.procs)


if __name__=='__main__':
    main()
