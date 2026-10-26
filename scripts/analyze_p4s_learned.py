#!/usr/bin/env python3
"""Locked P4-S-L validation selection and confirmation analysis (compute only).

    PYTHONPATH=src python scripts/analyze_p4s_learned.py --root RAWROOT --out REPORT

``select_theta(root,runs,design)`` opens validation traces only and returns
design_hash/selected/grid; the runner adds locked_at and commits theta.json.
``validate_theta(selection,runs,design)`` checks the frozen grid without trace
I/O. Analysis validates both Git locks and seals before confirmation reads.
Chosen states, executed updates and memory reads are reported separately.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from analyze_p4s_stopping import CRITERIA, halting_steps
from audit_p4s_stopping import digest_checkpoint, markdown_table, require, require_compute
from looplora.infra.seeds import stable_hash
from looplora.sym.stats import pooled_compare_runs, pooled_walk_specificity

WRITERS = ('joint_fact', 't1_frozen_fact', 'oracle_random', 'joint')
VARIANTS = ('M1', 'M1-lin')
SEEDS = (17, 29, 43)
CONFIDENCE = 1 - .05 / 6
THETAS = tuple(i / 20 for i in range(1, 20))


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f'Empty report table: {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _runs(runs: list[dict]) -> list[str]:
    expected = {(w, 32, v, s) for w in WRITERS for v in VARIANTS for s in SEEDS}
    require(len(runs) == 24 and {(r['writer'], r['r'], r['variant'], r['seed']) for r in runs} == expected,
            'P4-S-L requires exactly 24 registered reader/head identities')
    require(len({r['name'] for r in runs}) == 24, 'Duplicate run names')
    for run in runs:
        require(run['name'] not in ('.', '..') and Path(run['name']).name == run['name'], 'Unsafe run name')
        require(run['family'] == f"{run['writer']}/r32/{run['variant']}", 'Run family differs from identity')
    return sorted({r['family'] for r in runs})


def learned_steps(probabilities: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """First inclusive hit, else earliest best state; fallback executes all 16."""
    require(probabilities.ndim == 3 and probabilities.shape[-1] == 16
            and np.all(np.isfinite(probabilities)) and np.all((probabilities >= 0) & (probabilities <= 1)),
            'Halting probabilities must be finite [E,8,16] data in [0,1]')
    require(np.isfinite(theta) and 0 <= theta <= 1, 'Invalid halting theta')
    hits = probabilities >= theta
    found = hits.any(axis=-1)
    first = hits.argmax(axis=-1) + 1
    selected = np.where(found, first, probabilities.argmax(axis=-1) + 1).astype(np.int64)
    executed = np.where(found, selected, 16).astype(np.int64)
    return selected, executed


def load_trace(root: Path, run: dict, split: str, design: dict) -> dict[str, np.ndarray]:
    require(split in ('val', 'test'), 'Only registered validation/confirmation splits are supported')
    path = Path(root) / 'traces' / run['name'] / f'{split}.npz'
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    shape = trace['answer'].shape
    require(shape == (design['validation' if split == 'val' else 'confirmation']['episodes'], 8),
            f'Wrong trace episode/hop dimensions: {path}')
    for field in ('answer', 'episode_id', 'hop'):
        require(trace[field].shape == shape and trace[field].dtype.kind in 'iu', f'Invalid {field}: {path}')
    require(np.array_equal(trace['hop'], np.broadcast_to(np.arange(1, 9), shape)), 'Hop order must be 1..8')
    ids = trace['episode_id']
    require(np.all(ids == ids[:, :1]) and len(np.unique(ids[:, 0])) == shape[0], 'Episode ids are not unique/shared')
    require(trace['predictions'].shape == (*shape, 17) and trace['predictions'].dtype.kind in 'iu', 'Invalid predictions')
    learned_steps(trace['probabilities'], .5)
    require(trace['probabilities'].shape == (*shape, 16), 'Probability dimensions differ from query labels')
    for field in ('read_norm', 'relative_change'):
        require(trace[field].shape == (*shape, 17) and np.all(np.isfinite(trace[field][..., 1:]))
                and np.all(trace[field][..., 1:] >= 0), f'Invalid {field}')
    order = np.argsort(ids[:, 0], kind='stable')
    require(all(value.shape[:2] == shape for value in trace.values()), 'Unexpected trace array dimensions')
    return {key: value[order] for key, value in trace.items()}


def _correct(trace: dict, steps: np.ndarray) -> np.ndarray:
    return np.take_along_axis(trace['predictions'], steps[..., None], axis=-1)[..., 0] == trace['answer']


def select_theta(root: Path, runs: list[dict], design: dict) -> dict:
    """Validation-only family selection: accuracy, chosen steps, then theta."""
    families = _runs(runs)
    require(list(design['theta_grid']) == list(THETAS), 'Theta grid differs from registration')
    scores = {}
    for run in sorted(runs, key=lambda r: r['name']):
        trace = load_trace(root, run, 'val', design)
        for theta in THETAS:
            selected, executed = learned_steps(trace['probabilities'], theta)
            scores.setdefault((run['family'], theta), []).append(
                (float(_correct(trace, selected).mean()), float(selected.mean()), float(executed.mean())))
    grid, selected = [], []
    for family in families:
        candidates = []
        for theta in THETAS:
            values = np.asarray(scores[family, theta])
            require(values.shape == (3, 3), 'Missing seed in validation theta selection')
            candidates.append(dict(family=family, theta=theta, validation_accuracy=float(values[:, 0].mean()),
                                   validation_mean_steps=float(values[:, 1].mean()),
                                   validation_mean_executed_steps=float(values[:, 2].mean())))
        grid.extend(candidates)
        selected.append(dict(min(candidates, key=lambda r: (-r['validation_accuracy'], r['validation_mean_steps'], r['theta']))))
    return dict(design_hash=stable_hash(design), selected=selected, grid=grid)


def validate_theta(selection: dict, runs: list[dict], design: dict) -> dict[str, float]:
    """Check the locked theta grid without opening any trace or checkpoint."""
    families = _runs(runs)
    require(selection['design_hash'] == stable_hash(design), 'Theta lock belongs to a different design')
    require(len(selection['selected']) == 8 and {r['family'] for r in selection['selected']} == set(families), 'Incomplete theta selection')
    require(len(selection['grid']) == 152, 'Incomplete theta grid')
    result = {}
    for row in selection['selected']:
        candidates = [r for r in selection['grid'] if r['family'] == row['family']]
        require(len(candidates) == 19 and {r['theta'] for r in candidates} == set(THETAS), 'Missing or repeated theta grid row')
        for candidate in candidates:
            require(all(math.isfinite(candidate[k]) for k in ('theta', 'validation_accuracy', 'validation_mean_steps', 'validation_mean_executed_steps'))
                    and 0 <= candidate['validation_accuracy'] <= 1
                    and 1 <= candidate['validation_mean_steps'] <= candidate['validation_mean_executed_steps'] <= 16,
                    'Invalid frozen theta validation score/count')
        best = min(candidates, key=lambda r: (-r['validation_accuracy'], r['validation_mean_steps'], r['theta']))
        require(row == best, 'Theta selection disagrees with frozen validation grid/tie policy')
        result[row['family']] = row['theta']
    return result


def _references(design: dict, runs: list[dict]) -> tuple[dict, dict]:
    families = _runs(runs)
    rows = design['label_free_thresholds']
    require(len(rows) == 24 and {(r['family'], r['criterion']) for r in rows} ==
            {(f, c) for f in families for c in CRITERIA}, 'Incomplete locked label-free references')
    lookup = {(r['family'], r['criterion']): r for r in rows}
    best = design['best_label_free']
    require(isinstance(best, dict) and set(best) == set(families), 'Incomplete best-label-free mapping')
    for family in families:
        candidates = [lookup[family, c] for c in CRITERIA]
        require(all(np.isfinite(r['validation_accuracy']) and 0 <= r['validation_accuracy'] <= 1 for r in candidates), 'Invalid label-free validation accuracy')
        expected = min(candidates, key=lambda r: (-r['validation_accuracy'], CRITERIA.index(r['criterion'])))
        require(all(best[family][key] == expected[key] for key in ('family', 'criterion', 'threshold', 'validation_accuracy')),
                'Best label-free reference was not selected by locked validation accuracy')
    return lookup, best


def _git_locked_file(path: Path, commit: str) -> bytes:
    require(isinstance(commit, str) and bool(commit), 'Missing Git lock commit')
    root = Path(subprocess.check_output(['git', '-C', str(path.parent), 'rev-parse', '--show-toplevel'], text=True).strip())
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return subprocess.check_output(['git', '-C', str(root), 'show', f'{commit}:{relative}'])


def audit_locks(root: Path) -> tuple[dict, dict, dict]:
    """All lock, manifest and source seal checks precede confirmation loading."""
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['status'] == 'complete' and manifest['smoke'] is False, 'Analysis requires completed main confirmation artifacts')
    require(manifest.get('checkpoints_unchanged') is True, 'Runner did not certify frozen readers unchanged')
    runs, design = manifest['runs'], manifest['design']
    _runs(runs)
    require(manifest['design_hash'] == stable_hash(design), 'Manifest design hash differs')
    require(design['validation'] == dict(split='val', generator_seed=4242, episodes=2000)
            and design['confirmation'] == dict(split='test', generator_seed=999002, episodes=2000), 'Unregistered evaluation set')
    require(design['bootstrap']['resamples'] == 2000 and design['bootstrap']['confidence'] == CONFIDENCE
            and design['bootstrap']['seed'] == 2718, 'Unregistered bootstrap settings')
    theta_path = Path(manifest['theta_path'])
    require(theta_path.is_absolute() and digest_checkpoint(theta_path) == manifest['theta_blake2b'], 'Theta file seal differs')
    require(_git_locked_file(theta_path, manifest['theta_lock_commit']) == theta_path.read_bytes(), 'Theta file differs from its Git lock')
    repo = Path(subprocess.check_output(['git', '-C', str(theta_path.parent), 'rev-parse', '--show-toplevel'], text=True).strip())
    design_blob = _git_locked_file(repo / 'reports/p4s-l/design.json', manifest['design_lock_commit'])
    require(json.loads(design_blob) == design, 'Manifest design differs from the pre-head Git lock')
    selection = json.loads(theta_path.read_text())
    validate_theta(selection, runs, design)
    _references(design, runs)
    locked = datetime.fromisoformat(selection['locked_at'].replace('Z', '+00:00')).timestamp()
    design_runs = {r['name']: r for r in design['runs']}
    require(set(design_runs) == {r['name'] for r in runs}, 'Head runs differ from the pre-head locked readers')
    for run in runs:
        require(all(run[k] == design_runs[run['name']][k] for k in ('writer', 'r', 'variant', 'seed', 'family', 'checkpoint', 'checkpoint_blake2b')),
                'Head run reader differs from the pre-head design lock')
        for filename, seal in ((run['checkpoint'], run['checkpoint_blake2b']), (run['head_checkpoint'], run['head_blake2b'])):
            require(digest_checkpoint(Path(filename)) == seal, f'Checkpoint/head seal changed: {run["name"]}')
        val = root / 'traces' / run['name'] / 'val.npz'
        test = val.with_name('test.npz')
        require(val.stat().st_mtime <= locked < test.stat().st_mtime, 'Theta lock did not separate validation and confirmation')
    return manifest, design, selection


def _sample_mapping(traces: dict[int, dict], correct: dict[int, np.ndarray], depth: int) -> dict:
    result = {}
    for seed, trace in traces.items():
        shape = trace['answer'].shape
        result[seed] = dict(episode_id=trace['episode_id'].ravel(),
                            query_id=np.broadcast_to(np.arange(8), shape).ravel(), hop=trace['hop'].ravel(),
                            answer=trace['answer'].ravel(), correct=correct[seed].ravel(),
                            L=np.full(np.prod(shape), depth, dtype=np.int64),
                            split=np.full(np.prod(shape), 'test'), condition=np.full(np.prod(shape), 'original'))
    return result


def endpoint_statistics(traces: dict[int, dict], learned: dict[int, np.ndarray], reference: dict[int, np.ndarray]) -> dict:
    """Reuse P1-S's joint episode draws; synthetic L labels encode two methods."""
    a = _sample_mapping(traces, learned, 4)
    b = _sample_mapping(traces, reference, 1)
    merged = {s: {k: np.concatenate((a[s][k], b[s][k])) for k in a[s]} for s in a}
    options = dict(split='test', condition='original', resamples=2000, seed=2718, confidence=CONFIDENCE)
    output = {name: pooled_compare_runs(a, b, 4, 1, hops=hops, **options)
              for name, hops in (('hop1', [1]), ('reach', [2, 3]), ('far', [6, 7, 8]), ('overall', list(range(1, 9))))}
    output['specificity'] = pooled_walk_specificity(merged, 4, 1, **options)
    for stat in output.values():
        stat.update(method_a='learned', method_b='fixed_1', depth_labels_are_synthetic=True)
        if stat['statistic'] != 'mean_per_seed_episode_walk_specificity':
            stat['comparison'] = 'learned - fixed_1'
    return output


def qualification(stats: dict, primary: bool) -> dict:
    criteria = dict(hop1_preserved=stats['hop1']['difference'] >= -.01,
                    reach_at_least_3pp=stats['reach']['difference'] >= .03,
                    reach_ci_positive=stats['reach']['ci_low'] > 0,
                    specificity_ci_positive=stats['specificity']['ci_low'] > 0)
    return dict(primary=primary, criteria=criteria, endpoints_satisfied=all(criteria.values()),
                qualifies=primary and all(criteria.values()))


def _metric(trace: dict, correct: np.ndarray, selected: np.ndarray | None,
            executed: np.ndarray | None, reads: np.ndarray | None) -> list[dict]:
    rows = []
    for hop in (*range(1, 9), 'overall'):
        mask = np.ones(trace['answer'].shape, dtype=bool) if hop == 'overall' else trace['hop'] == hop
        rows.append(dict(hop=hop, n=int(mask.sum()), n_correct=int(correct[mask].sum()), accuracy=float(correct[mask].mean()),
                         mean_selected_steps=float(selected[mask].mean()) if selected is not None else None,
                         mean_executed_steps=float(executed[mask].mean()) if executed is not None else None,
                         mean_read_calls=float(reads[mask].mean()) if reads is not None else None))
    return rows


def analyze(root: Path, out: Path) -> dict:
    root, out = Path(root), Path(out)
    manifest, design, selection = audit_locks(root)
    runs = manifest['runs']
    theta = validate_theta(selection, runs, design)
    # Recompute validation selection only after metadata/Git-lock checks, and
    # still before the first confirmation trace is opened.
    actual = select_theta(root, runs, design)
    require({k: selection[k] for k in actual} == actual, 'Frozen theta differs from validation-only recomputation')
    references, best = _references(design, runs)
    per_seed, endpoints, per_seed_endpoints, audited_queries = [], [], [], 0
    for family in _runs(runs):
        members = sorted((r for r in runs if r['family'] == family), key=lambda r: r['seed'])
        test_traces, methods_test = {}, {}
        for run in members:
            for split in ('val', 'test'):
                trace = load_trace(root, run, split, design)
                selected, executed = learned_steps(trace['probabilities'], theta[family])
                if split == 'test':
                    for field, expected in (('selected_steps', selected), ('executed_steps', executed), ('read_calls', executed + 1)):
                        require(field in trace and np.array_equal(trace[field], expected), f'Saved learned {field} disagrees with locked theta policy')
                    val = load_trace(root, run, 'val', design)
                    require(not np.intersect1d(val['episode_id'][:, 0], trace['episode_id'][:, 0]).size, 'Validation and confirmation episodes overlap')
                methods = {'learned': (selected, executed, executed + 1)}
                for depth in (1, 4, 16):
                    fixed = np.full(trace['answer'].shape, depth, dtype=np.int64)
                    methods[f'fixed_{depth}'] = fixed, fixed, fixed
                for criterion in CRITERIA:
                    steps = halting_steps(trace, criterion, references[family, criterion]['threshold'])
                    field = f'reference_{criterion}_steps'
                    require(field not in trace or np.array_equal(trace[field], steps), f'Saved {criterion} steps disagree with locked reference')
                    methods[criterion] = steps, steps, steps + int(criterion == 'read_norm')
                methods['oracle_true_hop'] = trace['hop'], trace['hop'], trace['hop']
                correct = {name: _correct(trace, values[0]) for name, values in methods.items()}
                correct['oracle_any_step'] = (trace['predictions'][..., 1:] == trace['answer'][..., None]).any(axis=-1)
                base = {k: run[k] for k in ('name', 'writer', 'r', 'variant', 'seed', 'family')}
                for method, flags in correct.items():
                    counts = methods.get(method, (None, None, None))
                    per_seed.extend(dict(base, split=split, method=method, **row) for row in _metric(trace, flags, *counts))
                if split == 'test':
                    test_traces[run['seed']] = trace
                    methods_test[run['seed']] = correct
                audited_queries += trace['answer'].size
        learned = {s: methods_test[s]['learned'] for s in SEEDS}
        l1 = {s: methods_test[s]['fixed_1'] for s in SEEDS}
        stats = endpoint_statistics(test_traces, learned, l1)
        descriptive = {}
        for label, left, right in (
            ('learned_minus_best_label_free', 'learned', best[family]['criterion']),
            ('any_step_oracle_minus_learned', 'oracle_any_step', 'learned'),
            ('learned_minus_true_hop', 'learned', 'oracle_true_hop')):
            a = _sample_mapping(test_traces, {s: methods_test[s][left] for s in SEEDS}, 4)
            b = _sample_mapping(test_traces, {s: methods_test[s][right] for s in SEEDS}, 1)
            descriptive[label] = dict(overall=pooled_compare_runs(
                a, b, 4, 1, hops=list(range(1, 9)), split='test', condition='original',
                resamples=2000, seed=2718, confidence=CONFIDENCE))
            descriptive[label]['overall'].update(method_a=left, method_b=right, depth_labels_are_synthetic=True,
                                                  comparison=f'{left} - {right}')
        row = dict(family=family, writer=members[0]['writer'], variant=members[0]['variant'], r=32,
                   theta=theta[family], best_label_free=best[family], statistics=stats, descriptive=descriptive,
                   **qualification(stats, members[0]['writer'] != 'joint'))
        endpoints.append(row)
        for name, value in stats.items():
            for item in value['per_seed']:
                per_seed_endpoints.append(dict(family=family, endpoint=name, **item))
    pooled = []
    for family in _runs(runs):
        rows = [r for r in per_seed if r['family'] == family]
        for split in ('val', 'test'):
            for method in sorted({r['method'] for r in rows}):
                for hop in (*range(1, 9), 'overall'):
                    selected = [r for r in rows if (r['split'], r['method'], r['hop']) == (split, method, hop)]
                    require(len(selected) == 3, 'Accuracy table omits a training seed')
                    pooled.append(dict(family=family, writer=selected[0]['writer'], variant=selected[0]['variant'], r=32,
                                       split=split, method=method, hop=hop, seeds=3, n=sum(r['n'] for r in selected),
                                       n_correct=sum(r['n_correct'] for r in selected),
                                       **{k: float(np.mean([r[k] for r in selected])) if selected[0][k] is not None else None
                                          for k in ('accuracy', 'mean_selected_steps', 'mean_executed_steps', 'mean_read_calls')}))
    latency = analyze_latency(root, runs)
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / 'accuracy_per_seed.csv', per_seed)
    _csv(out / 'accuracy.csv', pooled)
    _csv(out / 'endpoints_per_seed.csv', per_seed_endpoints)
    _save(out / 'endpoints.json', endpoints)
    flat_endpoints = []
    for row in endpoints:
        values = list(row['statistics'].items()) + [
            (name, value['overall']) for name, value in row['descriptive'].items()]
        for name, stat in values:
            flat_endpoints.append(dict(family=row['family'], writer=row['writer'], variant=row['variant'], r=32,
                                       primary=row['primary'], theta=row['theta'], endpoint=name,
                                       **{key: stat[key] for key in ('difference', 'ci_low', 'ci_high', 'confidence',
                                                                     'n_episodes', 'n_queries_per_seed', 'n_seeds', 'resamples')}))
    _csv(out / 'endpoints.csv', flat_endpoints)
    _csv(out / 'qualification.csv', [dict(family=row['family'], primary=row['primary'],
                                         **row['criteria'], endpoints_satisfied=row['endpoints_satisfied'], qualifies=row['qualifies'])
                                    for row in endpoints])
    _csv(out / 'theta_grid.csv', selection['grid'])
    _csv(out / 'theta.csv', selection['selected'])
    _csv(out / 'latency_per_seed.csv', latency['per_seed'])
    _csv(out / 'latency.csv', latency['pooled'])
    render(out, endpoints, pooled, latency['pooled'])
    summary = dict(status='passed', study='P4-S-L', runs=24, families=8, primary_families=6,
                   qualified_families=[r['family'] for r in endpoints if r['qualifies']],
                   design_hash=manifest['design_hash'], design_lock_commit=manifest['design_lock_commit'],
                   theta_lock_commit=manifest['theta_lock_commit'], theta_blake2b=manifest['theta_blake2b'],
                   confidence=CONFIDENCE, resamples=2000, audited_queries=audited_queries,
                   selected_and_executed_steps_separate=True, learned_read_calls='executed_updates+1',
                   runtime_recomputed=False, latency_rows=len(latency['per_seed']),
                   analyzed_at=datetime.now(timezone.utc).isoformat())
    _save(out / 'analysis.json', summary)
    return summary


def analyze_latency(root: Path, runs: list[dict]) -> dict:
    rows = json.loads((root / 'latency.json').read_text())
    sources = {run['name']: run for run in runs}
    expected = {(name, method) for name in sources for method in ('learned', 'fixed_4', 'fixed_16')}
    seen, groups = set(), {}
    clocks = [f'{clock}_ms_{stat}' for clock in ('cuda', 'wall') for stat in ('mean', 'median', 'p95')]
    for row in rows:
        key = row['name'], row['method']
        require(key in expected and key not in seen, 'Unexpected or duplicate learned-halting latency row')
        seen.add(key)
        run = sources[row['name']]
        require(all(row[k] == run[k] for k in ('family', 'writer', 'variant', 'seed', 'r')), 'Latency run identity differs')
        require(row['batch_size'] == 64 and row['warmup'] == 5 and row['repeats'] == 20, 'Latency protocol differs')
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in clocks), 'Invalid latency measurement')
        require(1 <= row['mean_steps'] <= row['mean_executed_steps'] <= 16
                and row['mean_reads'] == row['mean_executed_steps'] + int(row['method'] == 'learned'),
                'Latency chosen/executed/read counters disagree')
        if row['method'] != 'learned':
            require(row['mean_steps'] == row['mean_executed_steps'] == int(row['method'].split('_')[1]), 'Fixed latency depth differs')
        require(row['audit_prediction_steps_exact'] is True
                and all(math.isfinite(row[k]) and row[k] >= 0 for k in ('audit_state_rel_l2_max', 'audit_state_abs_max')),
                'Latency streaming/offline audit is missing or invalid')
        groups.setdefault((run['family'], row['method']), []).append(row)
    require(seen == expected and len(rows) == 72, 'Latency must contain all 72 run/method rows')
    pooled = [dict(family=family, method=method, seeds=3, batch_size=64,
                   **{k: float(np.mean([row[k] for row in members]))
                      for k in (*clocks, 'mean_steps', 'mean_executed_steps', 'mean_reads')},
                   audit_state_rel_l2_max=max(row['audit_state_rel_l2_max'] for row in members),
                   audit_state_abs_max=max(row['audit_state_abs_max'] for row in members))
              for (family, method), members in sorted(groups.items())]
    return dict(per_seed=rows, pooled=pooled)


def render(out: Path, endpoints: list[dict], accuracy: list[dict], latency: list[dict]) -> None:
    def effect(stat):
        return f"{100*stat['difference']:+.4f} [{100*stat['ci_low']:+.4f}, {100*stat['ci_high']:+.4f}]"

    lines = ['# P4-S-L learned halting', '',
             f'Paired episode bootstrap: 2000 joint draws across training seeds; {100*CONFIDENCE:.6f}% CIs. '
             'Differences are percentage points. Six primary families; joint families are descriptive. '
             'Any-step oracle uses only states 1..16 and is label-informed; true-hop is not an upper bound. '
             'Chosen steps differ from executed updates on fallback, which executes 16 updates/17 reads.', '']
    for row in endpoints:
        family, stats = row['family'], row['statistics']
        lines += [f'## {family}', '', f"Theta {row['theta']:g}; best locked label-free rule: {row['best_label_free']['criterion']}. "
                  f"Primary: {row['primary']}; endpoints satisfied: {row['endpoints_satisfied']}; qualifies: {row['qualifies']}.", '']
        lines += markdown_table(['Endpoint vs L1', 'Estimate [adjusted CI] (pp)', 'Per-seed effect (pp)'],
                                [[name, effect(value), ', '.join(f"{r['seed']}: {100*r['difference']:+.4f}" for r in value['per_seed'])]
                                 for name, value in stats.items()])
        lines += markdown_table(['Descriptive contrast, hops1..8', 'Estimate [adjusted CI] (pp)'],
                                [[name, effect(value['overall'])] for name, value in row['descriptive'].items()])
        methods = sorted({r['method'] for r in accuracy if r['family'] == family})
        cells = {(r['method'], r['hop']): r for r in accuracy if r['family'] == family and r['split'] == 'test'}
        lines += markdown_table(['Rule', *[f'h{h} (%)' for h in range(1, 9)], 'Overall (%)', 'Chosen', 'Executed', 'Reads', 'Queries'],
                                [[m, *[f"{100*cells[m,h]['accuracy']:.3f}" for h in range(1, 9)],
                                  f"{100*cells[m,'overall']['accuracy']:.3f}",
                                  cells[m,'overall']['mean_selected_steps'], cells[m,'overall']['mean_executed_steps'],
                                  cells[m,'overall']['mean_read_calls'], cells[m,'overall']['n']] for m in methods])
        lines += markdown_table(['Latency method', 'CUDA mean ms / prepared batch64', 'Wall mean ms'],
                                [[r['method'], r['cuda_ms_mean'], r['wall_ms_mean']] for r in latency if r['family'] == family])
        lines.append('')
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    print(json.dumps(analyze(args.root, args.out)), flush=True)


if __name__ == '__main__':
    main()
