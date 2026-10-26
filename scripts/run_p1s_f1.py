#!/usr/bin/env python3
"""Fixed WP-F1 manifests and execution; application modes require PBS compute."""
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

ROOT = Path('/work/xg24i002/x10041/LoopLora-wt/symf')
RUNS = ROOT / 'runs/p1s/f1'
REPORTS = ROOT / 'reports/p1s-f1'
SEEDS = (17, 29, 43)
VARIANTS = ('M1', 'M1-lin', 'B5', 'B5-lin', 'B3')
LRS = {'oracle_random': (.003, .003, .003, .001, .0003),
       'joint': (.003, .001, .003, .003, .0003)}
DEPTHS = [0, 1, 2, 3, 4, 6, 8, 12, 16]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def configurations():
    from looplora.sym.config import resolve_config
    configs = []
    for writer in LRS:
        for rank in (64, 128):
            for seed in SEEDS:
                for variant, lr in zip(VARIANTS, LRS[writer]):
                    configs.append(resolve_config({
                        'name': f'f1-{writer}-r{rank}-{variant}-s{seed}',
                        'seed': seed, 'backbone_seed': 31415, 'device': 'cuda:0',
                        'writer': writer, 'variant': variant,
                        'model': {'d_e': 64, 'r': rank, 'r_mem': rank, 'reference_r': rank},
                        'task': {'name': 't2_chain_end', 'E': 12, 'Q': 4, 'V': 1024,
                                 'n_rel': 4, 'chain_lengths': [8, 4], 'hops': [1, 2, 3]},
                        'train': {'steps': 20000, 'batch_size': 64, 'lr': lr},
                        'eval': {'episodes': 2000, 'batch_size': 64, 'depths': DEPTHS,
                                 'splits': ['test', 'test_unseen'], 'seed': 4242,
                                 'counterfactuals': True},
                        'runtime': {'threads': 1},
                    }))
    return configs


def plan():
    configs = configurations()
    if (REPORTS / 'manifest.json').exists():
        raise FileExistsError('F1 manifest already exists')
    entries = []
    for config in configs:
        name = config['name']
        write(REPORTS / 'configs' / f'{name}.json', config)
        entries.append({'name': name, 'config': f'configs/{name}.json'})
    # Each shard has a balanced mix of seeds, writers, ranks and readers.
    for shard in range(6):
        write(REPORTS / f'shard-{shard:02d}.json', entries[shard::6])
    write(REPORTS / 'manifest.json', {
        'prereg': 'docs/prereg/P1S-F1.md', 'runs': 60, 'shards': 6,
        'names': [c['name'] for c in configs], 'seeds': list(SEEDS),
        'confirmation_barrier': 'all 60 record.json files complete before confirmation starts',
        'confirmation': {'split': 'test', 'seed': 777001, 'episodes': 2000},
        'budget_node_hours': 6,
    })
    smoke = [c for c in configs if c['seed'] == 17 and c['variant'] in ('M1', 'M1-lin')]
    for procs in (4, 8):
        smoke_entries = []
        for original in smoke:
            config = copy.deepcopy(original)
            config['name'] = 'smoke-' + config['name']
            config['train']['steps'] = 2000
            config['eval'].update(episodes=200, splits=['val'], counterfactuals=False)
            path = RUNS / 'plans' / f'p{procs}' / (config['name'] + '.json')
            write(path, config)
            smoke_entries.append({'name': config['name'], 'config': str(path)})
        write(RUNS / 'plans' / f'smoke-p{procs}.json', smoke_entries)
    print('Planned 60 formal runs and two eight-run timing packs', flush=True)


def smoke():
    from looplora.sym.run import run_suite
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_sym_model.py',
                    'tests/test_sym_stats.py', 'tests/test_sym_eval_overrides.py'], check=True)
    for procs in (4, 8):
        start = time.perf_counter()
        rc = run_suite(RUNS / 'plans' / f'smoke-p{procs}.json', RUNS / f'smoke-p{procs}', procs)
        if rc:
            raise RuntimeError(f'smoke p{procs} failed')
        elapsed = time.perf_counter() - start
        timings = []
        for path in sorted((RUNS / f'smoke-p{procs}').glob('*/record.json')):
            rec = json.loads(path.read_text())
            timings.append({'name': rec['experiment_id'], 'train_seconds': rec['train_seconds'],
                            'peak_memory_bytes': rec['peak_memory_bytes']})
        result = {'procs': procs, 'seconds': elapsed, 'runs': timings,
                  'job_id': os.environ.get('PBS_JOBID')}
        write(REPORTS / f'smoke-p{procs}.json', result)
        print(json.dumps(result), flush=True)


def all_training():
    manifest = json.loads((REPORTS / 'manifest.json').read_text())
    paths = list((RUNS / 'training').glob('shard-*/*/record.json'))
    by_name = {json.loads(p.read_text())['experiment_id']: p.parent for p in paths}
    if len(paths) != 60 or set(by_name) != set(manifest['names']):
        raise RuntimeError('global F1 training barrier: expected exactly 60 runs')
    records = []
    for name in manifest['names']:
        directory = by_name[name]
        record = json.loads((directory / 'record.json').read_text())
        if record['status'] != 'complete' or not (directory / 'checkpoint.pt').exists():
            raise RuntimeError(f'global F1 training barrier: incomplete {name}')
        records.append({'name': name, 'directory': str(directory), 'finished_at': record['finished_at'],
                        'git_commit': record['git_commit'], 'config_hash': record['config_hash']})
    return records


def evaluation_plan():
    rows = all_training()
    entries = []
    for row in rows:
        source = Path(row['directory'])
        config = json.loads((source / 'config.json').read_text())['config']
        expected = json.loads((REPORTS / 'configs' / (row['name'] + '.json')).read_text())
        if config != expected:
            raise ValueError('F1 trained configuration differs from its frozen manifest')
        if config['train']['steps'] != 20000 or config['train']['batch_size'] != 64:
            raise ValueError('F1 training budget changed')
        entries.append(dict(name=row['name'], group='primary', writer=config['writer'],
                            variant=config['variant'], seed=config['seed'], r=config['model']['r'],
                            config=config, run_dir=str(RUNS / 'evaluation' / row['name']),
                            training_dir=str(source)))
    return dict(steps=20000, runs=entries, root=str(RUNS),
                prereg='docs/prereg/P1S-F1.md', training_commit=rows[0]['git_commit'])


def evaluate_one(name):
    import torch
    from looplora.infra.seeds import stable_hash
    from looplora.sym.config import resolve_config
    from looplora.sym.evaluate import evaluate
    from looplora.sym.run import source_metadata
    from looplora.sym.train import load_checkpoint

    plan = evaluation_plan()  # Check the global training barrier before loading weights.
    entry = next(row for row in plan['runs'] if row['name'] == name)
    training_dir, out = Path(entry['training_dir']), Path(entry['run_dir'])
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / 'train_log.jsonl').symlink_to(training_dir / 'train_log.jsonl')
    torch.set_num_threads(1)
    torch.cuda.init()
    torch.cuda.set_device(0)
    model = load_checkpoint(training_dir / 'checkpoint.pt')
    checkpoint_config = copy.deepcopy(model.checkpoint_config)
    if checkpoint_config != entry['config']:
        raise ValueError('checkpoint and training configuration differ')
    provenance = dict(checkpoint=str(training_dir / 'checkpoint.pt'),
                      checkpoint_config=checkpoint_config, **source_metadata())
    record = json.loads((training_dir / 'record.json').read_text())
    record.update(status='running', config=checkpoint_config,
                  training_directory=str(training_dir), training_complete=True,
                  evaluation_source=source_metadata(), confirmation_samples_written=False)
    write(out / 'record.json', record)
    try:
        screen = resolve_config(checkpoint_config)
        screen['eval']['swap_successors'] = True
        metrics, timing = evaluate(model, screen, out / 'screening', provenance=provenance)
        write(out / 'screening/timing.json', timing)
        write(out / 'screening/config.json', dict(config=screen, **provenance))
        confirmation = copy.deepcopy(screen)
        confirmation['eval'].update(splits=['test'], seed=777001, counterfactuals=False)
        _, confirmation_timing = evaluate(model, confirmation, out / 'confirmation', samples_only=True,
                         provenance={**provenance, 'eval_overrides': {'eval.seed': 777001,
                                     'eval.splits': ['test'], 'eval.counterfactuals': False}})
        record.update(status='complete', confirmation_samples_written=True,
                      samples=str(out / 'screening/samples.npz'), scores=metrics['table'],
                      eval_samples=sum(row['n'] for row in metrics['table']),
                      confirmation_eval_samples=2000*8*len(DEPTHS),
                      latency=dict(screening=timing, confirmation=confirmation_timing),
                      config_hash=stable_hash(checkpoint_config),
                      confirmation_samples=str(out / 'confirmation/samples.npz'),
                      evaluation_finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        record.update(status='failed', failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(out / 'record.json', record)


def evaluation_shard(shard, procs):
    entries = evaluation_plan()['runs'][shard::6]
    pending = [row['name'] for row in entries]
    running = []
    while pending or running:
        for process in list(running):
            rc = process.poll()
            if rc is not None:
                running.remove(process)
                if rc:
                    for child in running:
                        child.terminate()
                    raise RuntimeError(f'evaluation child failed: {rc}')
        while pending and len(running) < procs:
            running.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                             'evaluate-one', '--name', pending.pop(0)],
                                            env={**os.environ, 'OMP_NUM_THREADS': '1'}))
        if running:
            time.sleep(.2)


def seal():
    from looplora.infra.seeds import stable_hash
    plan = evaluation_plan()
    paths = []
    for run in plan['runs']:
        folder = Path(run['run_dir'])
        record = json.loads((folder / 'record.json').read_text())
        if record['status'] != 'complete' or not record['confirmation_samples_written']:
            raise ValueError('F1 evaluation incomplete')
        meta = json.loads((folder / 'confirmation/evaluation.json').read_text())
        settings = meta['config']['eval']
        if (not meta.get('aggregation_deferred') or 'table' in meta or 'diagnostics' in meta
                or settings['seed'] != 777001 or settings['episodes'] != 2000
                or settings['splits'] != ['test'] or settings['counterfactuals']
                or settings['depths'] != DEPTHS or settings['hops'] != list(range(1, 9))
                or meta['checkpoint_config'] != run['config']):
            raise ValueError('F1 confirmation protocol differs')
        path = folder / 'confirmation/samples.npz'
        if not path.is_file() or (folder / 'confirmation/metrics.json').exists():
            raise ValueError('F1 confirmation missing or prematurely aggregated')
        paths.append(str(path))
    write(RUNS / 'evaluation_plan.json', plan)
    write(RUNS / 'barrier.json', dict(status='passed', plan_hash=stable_hash(plan),
          created_at=datetime.now(timezone.utc).isoformat(),
          runs=[row['name'] for row in plan['runs']], confirmation_samples=paths))
    print('F1 global release: 60 trained runs and 60 blind confirmation sample sets', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'smoke', 'train', 'barrier',
                                        'evaluate', 'evaluate-one', 'seal'])
    parser.add_argument('--shard', type=int, choices=range(6), default=0)
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.stage == 'plan':
        plan()
    elif args.stage == 'smoke':
        smoke()
    elif args.stage == 'train':
        from looplora.sym.run import run_suite
        rc = run_suite(REPORTS / f'shard-{args.shard:02d}.json',
                       RUNS / 'training' / f'shard-{args.shard:02d}', args.procs)
        if rc:
            raise SystemExit(rc)
    elif args.stage == 'barrier':
        write(RUNS / 'training_barrier.json', all_training())
    elif args.stage == 'evaluate':
        evaluation_shard(args.shard, args.procs)
    elif args.stage == 'evaluate-one':
        evaluate_one(args.name)
    else:
        seal()


if __name__ == '__main__':
    main()
