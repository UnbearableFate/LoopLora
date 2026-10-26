#!/usr/bin/env python3
"""WP-C4: released r32 checkpoints, unchanged WP-C3 selection and reader."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time

import torch

from analyze_p4s_stopping import select_thresholds
from analyze_p4s_r32 import summarize
from audit_p4s_stopping import digest_checkpoint, no_confirmation_path, require, require_compute
from looplora.infra.seeds import stable_hash
from looplora.sym.run import source_metadata
from run_p4s_stopping import collect, load_model, save

F2_ROOT = Path('/work/xg24i002/x10041/LoopLora-wt/symf/runs/p1s/f2')
H_ROOT = Path('/work/xg24i002/x10041/LoopLora/runs/p1s/main/20261026')


def checkpoint_manifest(f2_manifest: Path, h_root: Path = H_ROOT,
                        f2_root: Path = F2_ROOT) -> list[dict]:
    with f2_manifest.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream)
                if r['writer'] in {'joint_fact', 't1_frozen_fact'}
                and r['variant'] in {'M1', 'M1-lin'} and r['group'] == 'primary']
    for row in rows:
        row.update(r=32, seed=int(row['seed']),
                   reference_metrics=str(f2_root / 'evaluation' / row['name'] / 'screening/metrics.json'))
    for variant in ('M1', 'M1-lin'):
        for seed in (17, 29, 43):
            name = f'primary-joint-{variant}-r32-s{seed}'
            rows.append(dict(name=name, writer='joint', r=32, variant=variant, seed=seed,
                             training_dir=str(h_root / 'runs' / name),
                             reference_metrics=str(h_root / 'runs' / name / 'screening/metrics.json')))
    expected = {(w, 32, v, s) for w in ('joint_fact', 't1_frozen_fact', 'joint')
                for v in ('M1', 'M1-lin') for s in (17, 29, 43)}
    require(len(rows) == 18 and {(r['writer'], r['r'], r['variant'], r['seed']) for r in rows} == expected,
            'WP-C4 requires exactly the 18 released r32 reader identities')
    runs = []
    for row in rows:
        path = Path(row['training_dir']) / 'checkpoint.pt'
        no_confirmation_path(path)
        no_confirmation_path(Path(row['reference_metrics']))
        require(path.is_file(), f'Missing released checkpoint: {path}')
        record = json.loads(path.with_name('record.json').read_text())
        require(record['status'] == 'complete', f'Incomplete training: {path}')
        run = {k: row[k] for k in ('name', 'writer', 'r', 'variant', 'seed', 'reference_metrics')}
        run.update(family=f"{row['writer']}/r32/{row['variant']}",
                   checkpoint=str(path.resolve()), checkpoint_blake2b=digest_checkpoint(path))
        runs.append(run)
    return sorted(runs, key=lambda r: r['name'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--design', type=Path, default=Path('reports/p4s-r32/design.json'))
    parser.add_argument('--f2-manifest', type=Path, default=Path('reports/p1s-f2/main_manifest.csv'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    require_compute()
    require(torch.cuda.is_available(), 'WP-C4 requires a GPU allocation')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    design = json.loads(args.design.read_text())
    require(design['collect_certificates'] is False and design['latency'] is False,
            'WP-C4 omits certificates and latency')
    previous = json.loads(Path('reports/p4s/design.json').read_text())
    for key in ('read_norm_thresholds', 'relative_change_thresholds', 'output_stability_thresholds',
                'evaluation_seed', 'hops', 'max_steps', 'minimum_steps', 'normalization_epsilon'):
        require(design[key] == previous[key], f'WP-C3 convention changed: {key}')
    runs = checkpoint_manifest(args.f2_manifest)
    if args.smoke:
        runs = [r for r in runs if r['seed'] == 17]
        design['episodes_per_split'] = 200
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = dict(design=design, runs=runs, smoke=args.smoke, **source_metadata(),
                    started_at=datetime.now(timezone.utc).isoformat(),
                    runtime=dict(host=socket.gethostname(), job_id=os.environ['PBS_JOBID'],
                                 torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                                 dtype='float32', threads=1))
    save(root / 'manifest.json', metadata)
    timing = []
    for run in runs:
        model = load_model(run)
        row, _ = collect(model, 'val', design, root / 'traces' / run['name'] / 'val.npz')
        timing.append(dict(name=run['name'], **row))
        print(json.dumps(dict(stage='val', name=run['name'], seconds=row['total_seconds'])), flush=True)
        del model
    if not args.smoke:
        selection = select_thresholds(root, runs, design)
        selection['locked_at'] = datetime.now(timezone.utc).isoformat()
        save(root / 'thresholds.json', selection)
        for run in runs:
            model = load_model(run)
            row, _ = collect(model, 'test', design, root / 'traces' / run['name'] / 'test.npz')
            timing.append(dict(name=run['name'], **row))
            print(json.dumps(dict(stage='test', name=run['name'], seconds=row['total_seconds'])), flush=True)
            del model
        summarize(root, runs, design, selection, root / 'summaries')
    for run in runs:
        require(digest_checkpoint(Path(run['checkpoint'])) == run['checkpoint_blake2b'],
                f"Checkpoint changed: {run['name']}")
    save(root / 'timing.json', timing)
    metadata.update(status='complete', total_seconds=time.perf_counter() - started,
                    finished_at=datetime.now(timezone.utc).isoformat(), design_hash=stable_hash(design),
                    checkpoints_unchanged=True, peak_memory_bytes=torch.cuda.max_memory_allocated())
    save(root / 'manifest.json', metadata)
    print(json.dumps(dict(status='complete', smoke=args.smoke,
                          seconds=metadata['total_seconds'], root=str(root))), flush=True)


if __name__ == '__main__':
    main()
