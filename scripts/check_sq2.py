"""Compute-only SQ2 gates before the schedule release is committed (docs/prereg/SQ2.md).

step0   The registered step-0 check: `S-DE` at step 0 (D128-3's final adapter and value head under the delta store) must
        reproduce D128-3's recorded final validation exactly: 608 correct, the NLL bit for bit (and, when the parent's validation
        samples are on disk, every prediction). It prints and writes a result with `passed`; if false the continuation is void.
reload  One smoke arm: its export reloads with `sq2.load_inference`; the reloaded config is the smoke record's and the one the code
        builds for the smoke; a fresh 100-episode validation of the reloaded writer predicts exactly what the smoke's last
        evaluation predicted. Prints identity and booleans only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from looplora.llm.read_fix_run import validate
from looplora.llm.run import source_metadata
from looplora.llm.sq2 import SQ2Experiment, configuration, load_continuation, load_inference
from looplora.llm.sq2_release import N_MAX, PARENT, SPECS, file_digest
from looplora.sym.stats import load_samples

ROOT = Path(__file__).resolve().parents[1]
EPISODES = 100  # the smoke's evaluation size
PARENT_FINAL = 'val-%d' % PARENT['steps']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def step0(out, parent_folder=None):
    folder = Path(PARENT['folder'] if parent_folder is None else parent_folder)
    recorded = json.loads((ROOT / PARENT['record']).read_text())['training']['metrics'][str(PARENT['steps'])]
    experiment = SQ2Experiment(configuration('S-DE', N_MAX), 'S-DE')
    try:
        load_continuation(experiment, folder)
        metrics = validate(experiment, Path(out) / 'val-0', 1000)
    finally:
        experiment.close()
    samples = load_samples(Path(out) / 'val-0' / 'samples.npz')
    parent_samples = folder / PARENT_FINAL / 'samples.npz'
    predictions = None
    if parent_samples.exists():
        parent = load_samples(parent_samples)
        predictions = bool(all(np.array_equal(samples[key], parent[key]) for key in ('episode_id', 'answer', 'prediction')))
    passed = metrics['correct'] == recorded['correct'] == 608 and metrics['nll'] == recorded['nll'] and predictions is not False
    return dict(study='SQ2', check='step0', arm='S-DE', passed=bool(passed), correct=metrics['correct'], parent_correct=recorded['correct'],
                nll=metrics['nll'], parent_nll=recorded['nll'], nll_bit_identical=metrics['nll'] == recorded['nll'],
                predictions_identical=predictions, parent_digests={k: file_digest(folder / k) for k in PARENT['digests']},
                checked_at_commit=source_metadata()['git_commit'], job_id=os.environ.get('PBS_JOBID'))


def reload(release_path, root, arm):
    release = json.loads(Path(release_path).read_text())
    steps = release['smoke']['steps']
    rows = [row for row in release['smoke']['rows'] if row['arm'] == arm]
    require(len(rows) == 1, f'{arm}: not an arm of this SQ2 release')
    folder = Path(root) / arm
    record = json.loads((folder / 'record.json').read_text())
    require([record[k] for k in ('job_id', 'git_commit', 'source_hash')] == [rows[0][k] for k in ('job_id', 'git_commit', 'source_hash')],
            f'{arm}: this smoke record is not the one the release was sized from')
    numerics = release['numerics']
    store, coefficient = SPECS[arm]
    options = dict(eta=numerics['eta'] if store == 'ridge' else None, w_c=numerics['w_c'] if coefficient else None,
                   numerics_digest=numerics['file_blake2b_128'])
    expected = json.loads(json.dumps(configuration(arm, steps, smoke=True, **options)))
    require(record['config'] == expected, f'{arm}: the smoke ran another config than the code builds')
    experiment = load_inference(folder / 'writer.pt')
    try:
        require(json.loads(json.dumps(experiment.config)) == expected, f'{arm}: the reloaded export has another config')
        require(experiment.value_head is None, f'{arm}: the reloaded export holds a training-only value head')
        require(all(bool(torch.isfinite(v).all()) for v in experiment.adapter.state_dict().values() if v.is_floating_point()),
                f'{arm}: the reloaded export has non-finite parameters')
        reloaded_folder = Path(root) / f'export-check-{arm}'
        validate(experiment, reloaded_folder, EPISODES)
    finally:
        experiment.close()
    reloaded = load_samples(reloaded_folder / 'samples.npz')
    saved = load_samples(folder / f'val-{steps}' / 'samples.npz')
    require(all(np.array_equal(reloaded[k], saved[k]) for k in ('episode_id', 'answer')), f'{arm}: not the smoke\'s episodes')
    mismatches = int((reloaded['prediction'] != saved['prediction']).sum())
    require(mismatches == 0, f'{arm}: the reloaded writer differs from the smoke in {mismatches} of {EPISODES} predictions')
    result = dict(study='SQ2', arm=arm, steps=steps, config_equal=True, reload_identical=True, prediction_mismatches=mismatches,
                  science_commit=release['science_commit'], checked_at_commit=source_metadata()['git_commit'], quality_scores_exposed=False)
    (reloaded_folder / 'reload-check.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    first = sub.add_parser('step0')
    first.add_argument('--out', type=Path, required=True)
    second = sub.add_parser('reload')
    second.add_argument('--release', type=Path, required=True)
    second.add_argument('--root', type=Path, required=True)
    second.add_argument('--arm', required=True)
    args = parser.parse_args()
    if args.mode == 'step0':
        args.out.mkdir(parents=True, exist_ok=False)
        result = step0(args.out)
        with (args.out / 'step0.json').open('x') as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['passed'] else 3)
    print(json.dumps(reload(args.release, args.root, args.arm)), flush=True)
