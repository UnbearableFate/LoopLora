#!/usr/bin/env python3
"""Lock P4-S-L readers and references before any halting-head training.

Compute only: collect six oracle_random/r32 validation traces and reuse C4's
18 existing threshold rows. Never train a head or open confirmation artifacts.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

REPO = Path(__file__).resolve().parents[1]
H_ROOT = Path('/work/xg24i002/x10041/LoopLora/runs/p1s/main/20261026')
SEEDS = (17, 29, 43)
VARIANTS = ('M1', 'M1-lin')
C4_WRITERS = ('joint_fact', 't1_frozen_fact', 'joint')
CRITERIA = ('read_norm', 'relative_change', 'output_stability')
CLARIFICATION = (
    'No oracle_random/r32 label-free thresholds existed. Select them on 2000 validation episodes '
    'with generator seed 4242 using the unchanged WP-C3 grids and tie rule BEFORE ANY HEAD TRAINING; '
    "reuse WP-C4's 18 threshold rows for the other six families."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest_blake2b(path):
    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(run):
    return run['writer'], run['r'], run['variant'], run['seed']


def _check_sidecars(run):
    checkpoint = Path(run['checkpoint'])
    require(not any('confirmation' in part.lower() for part in checkpoint.resolve().parts),
            'Confirmation artifacts are outside preparation scope')
    require(checkpoint.name == 'checkpoint.pt' and checkpoint.is_file(), f'Missing released checkpoint: {checkpoint}')
    require(Path(run['name']).name == run['name'] and run['name'] not in ('.', '..'), 'Invalid reader name')
    config = read_json(checkpoint.with_name('config.json'))['config']
    record = read_json(checkpoint.with_name('record.json'))
    require(record['status'] == 'complete' and record['experiment_id'] == run['name'],
            f'Incomplete or misidentified reader record: {run["name"]}')
    expected = identity(run)
    require((config['writer'], config['model']['r'], config['variant'], config['seed']) == expected,
            f'Reader config identity differs: {run["name"]}')
    require((record['writer'], record['rank'], record['variant'], record['train_seed']) == expected,
            f'Reader record identity differs: {run["name"]}')
    require(config['model']['r_mem'] == 32, f'Reader memory rank differs: {run["name"]}')
    require(run['family'] == f"{run['writer']}/r32/{run['variant']}", 'Family identity differs')


def build_runs(c4_manifest, h_root):
    require(c4_manifest.get('status') == 'complete' and c4_manifest.get('smoke') is False
            and c4_manifest.get('checkpoints_unchanged') is True, 'C4 manifest must be complete and sealed')
    runs = copy.deepcopy(c4_manifest['runs'])
    expected_c4 = {(writer, 32, variant, seed) for writer in C4_WRITERS
                   for variant in VARIANTS for seed in SEEDS}
    require(len(runs) == 18 and {identity(run) for run in runs} == expected_c4,
            'C4 must contain exactly its 18 released r32 identities')
    for run in runs:
        _check_sidecars(run)
        require(digest_blake2b(run['checkpoint']) == run['checkpoint_blake2b'],
                f'C4 checkpoint seal changed: {run["name"]}')
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f'primary-oracle_random-{variant}-r32-s{seed}'
            checkpoint = (Path(h_root) / 'runs' / name / 'checkpoint.pt').resolve()
            run = dict(name=name, writer='oracle_random', r=32, variant=variant, seed=seed,
                       family=f'oracle_random/r32/{variant}', checkpoint=str(checkpoint))
            _check_sidecars(run)
            run['checkpoint_blake2b'] = digest_blake2b(checkpoint)
            runs.append(run)
    expected = expected_c4 | {('oracle_random', 32, variant, seed) for variant in VARIANTS for seed in SEEDS}
    require(len(runs) == 24 and len({run['name'] for run in runs}) == 24
            and {identity(run) for run in runs} == expected, 'P4-S-L requires exactly 24 frozen readers')
    return sorted(runs, key=lambda run: run['name'])


def collection_design(c4_design):
    design = copy.deepcopy(c4_design)
    require(design['episodes_per_split'] == 2000 and design['evaluation_seed'] == 4242
            and design['hops'] == list(range(1, 9)) and design['max_steps'] == 16
            and design['minimum_steps'] == 1 and design['normalization_epsilon'] == 1e-12,
            'C4 stopping conventions differ from the registered design')
    design['families'] = dict(writer=['oracle_random'], r=[32], variant=list(VARIANTS))
    design['collect_certificates'] = False
    design['latency'] = False
    # Existing C3 selection metadata requires these two protocol split names;
    # this preparation invokes collect only with split='val'.
    require(design['splits'] == ['val', 'test'] and design['training_seeds'] == list(SEEDS),
            'C3 selection protocol metadata differs')
    return design


def read_c4_thresholds(path, design):
    with Path(path).open(newline='') as stream:
        raw = list(csv.DictReader(stream))
    expected = {(f'{writer}/r32/{variant}', criterion) for writer in C4_WRITERS
                for variant in VARIANTS for criterion in CRITERIA}
    rows = []
    for row in raw:
        parsed = {key: row[key] for key in ('family', 'criterion')}
        for key in ('threshold', 'validation_accuracy', 'validation_mean_steps'):
            parsed[key] = float(row[key])
            require(math.isfinite(parsed[key]), f'Nonfinite C4 threshold row: {row}')
        criterion = parsed['criterion']
        require(criterion in CRITERIA and parsed['threshold'] in design[f'{criterion}_thresholds'],
                'C4 threshold is outside its locked grid')
        if criterion == 'output_stability':
            parsed['threshold'] = int(parsed['threshold'])
        require(0 <= parsed['validation_accuracy'] <= 1 and 1 <= parsed['validation_mean_steps'] <= 16,
                'Invalid C4 validation score/selected-step mean')
        rows.append(parsed)
    require(len(rows) == 18 and {(row['family'], row['criterion']) for row in rows} == expected,
            'C4 must supply exactly 18 rows for six families; random/r32 must not be inferred from r64')
    return rows


def attach_source(rows, path, kind):
    source = dict(kind=kind, path=str(Path(path).resolve()), blake2b=digest_blake2b(path),
                  selection_split='val', generator_seed=4242, episodes_per_seed=2000,
                  training_seeds=list(SEEDS), selected_before_any_head_training=True)
    return [{**copy.deepcopy(row), 'source': copy.deepcopy(source)} for row in rows]


def best_label_free(rows):
    families = sorted({row['family'] for row in rows})
    result = {}
    for family in families:
        choices = [row for row in rows if row['family'] == family]
        require(len(choices) == 3 and {row['criterion'] for row in choices} == set(CRITERIA),
                f'Incomplete label-free criteria: {family}')
        result[family] = copy.deepcopy(min(choices, key=lambda row: (
            -row['validation_accuracy'], CRITERIA.index(row['criterion']))))
    return result


def locked_design(runs, thresholds, sources, raw_root):
    families = sorted({run['family'] for run in runs})
    require(len(thresholds) == 24 and len(families) == 8
            and {row['family'] for row in thresholds} == set(families), 'Incomplete design references')
    return dict(
        schema_version=1, task='WP-P4L', study='P4-S-L', status='prepared_for_commit',
        runs=copy.deepcopy(runs), label_free_thresholds=copy.deepcopy(thresholds),
        best_label_free=best_label_free(thresholds),
        primary_families=[family for family in families if not family.startswith('joint/')],
        descriptive_families=[family for family in families if family.startswith('joint/')],
        r=32, max_steps=16, minimum_steps=1, hops=list(range(1, 9)), normalization_epsilon=1e-12,
        training_seeds=list(SEEDS), theta_grid=[round(i / 20, 2) for i in range(1, 20)],
        theta_selection=dict(split='val', accuracy='equal mean over hops1..8 and three training seeds',
                             tiebreak=['higher_accuracy', 'fewer_mean_selected_steps', 'smaller_threshold'],
                             inference='first s in1..16 with p_s>=theta; otherwise earliest argmax_s p_s',
                             barrier='commit theta lock before any confirmation evaluation'),
        head=dict(input_dim=51, hidden_dim=64, hidden_activation='relu', output_dim=1,
                  output_activation='sigmoid',
                  features=[dict(name='post_update_state', width=32), dict(name='state_l2_norm', width=1),
                            dict(name='relative_change', width=1), dict(name='read_norm_ratio', width=1),
                            dict(name='one_hot_step1..16', width=16)],
                  relative_change='norm(z_s-z_previous)/max(norm(z_previous),EPS)',
                  read_norm_ratio='post-state read norm / max(initial-state read norm,EPS)',
                  target='candidate argmax at this post-update step equals the answer'),
        training=dict(split='train', generator_seed=31337, episodes=20000, steps=2000,
                      optimizer='adam', lr=.001, batch_episodes=256, hops_per_episode=8,
                      steps_per_query=16, loss='binary_cross_entropy',
                      minibatch='all eight queries and all sixteen post-update steps for each sampled episode'),
        validation=dict(split='val', generator_seed=4242, episodes=2000),
        confirmation=dict(split='test', generator_seed=999002, episodes=2000),
        bootstrap=dict(resamples=2000, confidence=1 - .05 / 6, seed=2718,
                       resampling='paired episode bootstrap, shared episode draws across three seeds'),
        latency=dict(batch_size=64, warmup=5, repeats=20),
        seed_derivation=dict(head_init="derive_seed(seed,'p4s-l',family,'head_init')",
                             episode_sampler="derive_seed(seed,'p4s-l',family,'episode_sampler')"),
        user_clarification=CLARIFICATION,
        label_free_selection=dict(grid='unchanged WP-C3 grids',
                                  tiebreak=['higher_accuracy', 'fewer_mean_selected_steps', 'smaller_threshold'],
                                  c4_families='reuse all18 existing rows unchanged; no reselection',
                                  oracle_random='select only two r32 families on validation before head training',
                                  best_rule_tiebreak=list(CRITERIA)),
        sources=copy.deepcopy(sources), preparation_root=str(Path(raw_root).resolve()),
        prepared_at=datetime.now(timezone.utc).isoformat(),
        next_required_action='commit this full design and reference rows before any head training')


def prepare(root, out, c4_manifest_path, c4_thresholds_path, h_root):
    from audit_p4s_stopping import require_compute
    require_compute()  # Before numerical/model imports or source checkpoint loading.
    import torch
    from analyze_p4s_stopping import select_thresholds
    from looplora.infra.seeds import stable_hash
    from looplora.sym.run import source_metadata
    from run_p4s_stopping import collect, load_model

    require(torch.cuda.is_available(), 'Preparation requires the Miyabi-G GPU')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    root, out = Path(root).resolve(), Path(out).resolve()
    require(not root.exists(), f'Preparation raw root must be new: {root}')
    require(not (out / 'design.json').exists(), 'Refusing to overwrite an existing head-training design lock')
    c4 = read_json(c4_manifest_path)
    runs = build_runs(c4, h_root)
    trace_design = collection_design(c4['design'])
    original_c3 = read_json(REPO / 'reports/p4s/design.json')
    for key in ('read_norm_thresholds', 'relative_change_thresholds', 'output_stability_thresholds'):
        require(trace_design[key] == original_c3[key], f'WP-C3 threshold grid changed: {key}')
    reused = read_c4_thresholds(c4_thresholds_path, trace_design)
    for run in runs:
        config = torch.load(run['checkpoint'], map_location='cpu', weights_only=True)['config']
        require((config['writer'], config['model']['r'], config['variant'], config['seed']) == identity(run)
                and config['model']['r_mem'] == 32, f'Checkpoint payload identity differs: {run["name"]}')
    root.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = dict(status='running', task='WP-P4L-preparation', runs=runs,
                    collection_splits=['val'], collection_writers=['oracle_random'],
                    design=trace_design, user_clarification=CLARIFICATION, **source_metadata())
    save(root / 'manifest.json', metadata)
    try:
        random_runs = [run for run in runs if run['writer'] == 'oracle_random']
        timing = []
        for run in random_runs:
            model = load_model(run)
            row, batch = collect(model, 'val', trace_design, root / 'traces' / run['name'] / 'val.npz')
            require(row['episodes'] == 2000 and row['queries'] == 16000, 'Random validation trace count differs')
            timing.append(dict(name=run['name'], **row))
            del model, batch
        random_selection = select_thresholds(root, random_runs, trace_design)
        random_selection['locked_at'] = datetime.now(timezone.utc).isoformat()
        save(root / 'thresholds.json', random_selection)
        sources = {
            'c4_manifest': dict(path=str(Path(c4_manifest_path).resolve()), blake2b=digest_blake2b(c4_manifest_path)),
            'c4_thresholds': dict(path=str(Path(c4_thresholds_path).resolve()), blake2b=digest_blake2b(c4_thresholds_path)),
            'random_selection': dict(path=str(root / 'thresholds.json'), blake2b=digest_blake2b(root / 'thresholds.json')),
        }
        thresholds = attach_source(reused, c4_thresholds_path, 'reused_C4') + attach_source(
            random_selection['selected'], root / 'thresholds.json', 'new_random_r32_validation_before_head_training')
        thresholds.sort(key=lambda row: (row['family'], CRITERIA.index(row['criterion'])))
        design = locked_design(runs, thresholds, sources, root)
        for run in runs:
            require(digest_blake2b(run['checkpoint']) == run['checkpoint_blake2b'],
                    f'Checkpoint changed during preparation: {run["name"]}')
        require(set((root / 'traces').glob('*/*')) == {
            root / 'traces' / run['name'] / 'val.npz' for run in random_runs}, 'Preparation must retain only six validation archives')
        save(root / 'timing.json', timing)
        save(out / 'design.json', design)
        with (out / 'label_free_thresholds.csv').open('w', newline='') as stream:
            fields = ('family', 'criterion', 'threshold', 'validation_accuracy', 'validation_mean_steps',
                      'source_kind', 'source_path', 'source_blake2b')
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for row in thresholds:
                writer.writerow({**{key: row[key] for key in fields[:5]},
                                 **{'source_' + key: row['source'][key] for key in ('kind', 'path', 'blake2b')}})
        audit = dict(status='complete', runs=24, checkpoint_seals_verified=24, checkpoints_unchanged=True,
                     reused_threshold_rows=18, newly_selected_threshold_rows=6, primary_families=6,
                     validation_trace_files=6, validation_episodes=12000, validation_queries=96000,
                     head_training_performed=False, confirmation_inputs_opened=False,
                     design_path=str(out / 'design.json'), design_hash=stable_hash(design),
                     runtime=dict(host=socket.gethostname(), job_id=os.environ['PBS_JOBID'],
                                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0), threads=1),
                     seconds=time.perf_counter() - started)
        save(root / 'preparation.json', audit)
        save(out / 'preparation_audit.json', audit)
        metadata.update(status='complete', checkpoints_unchanged=True, total_seconds=audit['seconds'],
                        design_hash=audit['design_hash'], completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        metadata.update(status='failed', failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        save(root / 'manifest.json', metadata)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--c4-manifest', type=Path, default=REPO / 'reports/p4s-r32/manifest.json')
    parser.add_argument('--c4-thresholds', type=Path, default=REPO / 'reports/p4s-r32/thresholds.csv')
    parser.add_argument('--h-root', type=Path, default=H_ROOT)
    args = parser.parse_args(argv)
    result = prepare(args.root, args.out, args.c4_manifest, args.c4_thresholds, args.h_root)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
