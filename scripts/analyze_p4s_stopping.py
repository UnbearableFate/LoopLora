#!/usr/bin/env python3
"""Validation-only selection and screening summaries for the locked WP-C3 pilot.

This module deliberately has no model/runtime dependency. Trace index L denotes
the prediction after update L; index zero can never be a stopping decision.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CRITERIA = ('read_norm', 'relative_change', 'output_stability')
ACCURACY_COLUMNS = ('writer', 'r', 'variant', 'family', 'split', 'method', 'hop',
                    'accuracy', 'mean_steps', 'mean_reads', 'n')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _write_csv(path, rows, columns=None):
    columns = list(columns or (rows[0].keys() if rows else []))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _family(run):
    return f"{run['writer']}/r{run['r']}/{run['variant']}"


def _metadata(runs, design):
    """Persist the full locked design and source identities without reading data."""
    _require(runs, 'No runs supplied')
    identities = []
    for run in runs:
        family = _family(run)
        _require(run['family'] == family, 'Run family disagrees with writer/r/variant')
        _require(Path(run['name']).name == run['name'] and run['name'] not in ('.', '..'),
                 'Run name must be a single path component')
        identities.append({k: run[k] for k in
                           ('name', 'writer', 'r', 'variant', 'family', 'seed', 'checkpoint')})
    _require(len({r['name'] for r in identities}) == len(identities), 'Duplicate run name')
    expected_families = {f'{w}/r{r}/{v}' for w in design['families']['writer']
                         for r in design['families']['r'] for v in design['families']['variant']}
    _require({r['family'] for r in identities} == expected_families,
             'Runs do not cover the locked families')
    for family in expected_families:
        seeds = [r['seed'] for r in identities if r['family'] == family]
        _require(len(seeds) == len(set(seeds)) and set(seeds) == set(design['training_seeds']),
                 f'Family {family} must contain each training seed exactly once')
    _require(design['minimum_steps'] == 1, 'Only post-update stopping is supported')
    _require(design['splits'] == ['val', 'test'], 'Only validation and screening test are allowed')
    _require(len(set(design['hops'])) == len(design['hops']) and design['hops'], 'Invalid hops')
    return json.loads(json.dumps(dict(
        schema_version=1, selection_split='val',
        selection_weighting='equal mean over training seeds and hops',
        selection_tiebreak=['higher_accuracy', 'fewer_steps', 'lower_threshold'],
        design=design, runs=sorted(identities, key=lambda r: r['name']))))


def _grid(design, criterion):
    thresholds = design[f'{criterion}_thresholds']
    _require(thresholds and len(set(thresholds)) == len(thresholds), 'Invalid threshold grid')
    for value in thresholds:
        _require(np.isfinite(value) and value >= 0, 'Thresholds must be finite and nonnegative')
        if criterion == 'output_stability':
            _require(value >= 1 and int(value) == value, 'Stability patience must be a positive integer')
    return sorted(thresholds)


def _load_trace(root, run, split, design):
    _require(split in ('val', 'test'), 'Confirmation data are outside WP-C3 scope')
    path = Path(root) / 'traces' / run['name'] / f'{split}.npz'
    certificates = design.get('collect_certificates', True)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in
                ('predictions', 'read_norm', 'relative_change', 'answer', 'hop',
                 'episode_id')}
        if certificates:
            data['certified_c'] = archive['certified_c']
    shape = data['answer'].shape
    _require(len(shape) == 2 and shape[0] > 0 and shape[1] == len(design['hops']),
             f'Invalid answer shape in {path}')
    expected = (*shape, design['max_steps'] + 1)
    for key in ('predictions', 'read_norm', 'relative_change'):
        _require(data[key].shape == expected, f'Invalid {key} shape in {path}')
    for key in ('hop', 'episode_id'):
        _require(data[key].shape == shape, f'Invalid {key} shape in {path}')
    _require(np.issubdtype(data['predictions'].dtype, np.integer)
             and np.issubdtype(data['answer'].dtype, np.integer), 'Predictions and labels must be integers')
    _require(np.array_equal(np.sort(data['hop'], axis=1),
                           np.broadcast_to(sorted(design['hops']), shape)),
             f'Each episode must contain each locked hop once in {path}')
    _require(np.all(data['episode_id'] == data['episode_id'][:, :1])
             and len(np.unique(data['episode_id'][:, 0])) == shape[0],
             f'Episode identities must be unique and shared across hops in {path}')
    if certificates:
        _require(data['certified_c'].shape == (shape[0],)
                 and np.all(np.isfinite(data['certified_c'])) and np.all(data['certified_c'] >= 0),
                 f'Invalid contraction certificates in {path}')
    for key in ('read_norm', 'relative_change'):
        values = data[key][..., 1:]
        _require(np.all(np.isfinite(values)) and np.all(values >= 0), f'Invalid {key} in {path}')
    return data


def halting_steps(trace, criterion, threshold):
    """NumPy equivalent of the label-free reader's first-hit convention."""
    _require(criterion in CRITERIA, 'Unknown criterion')
    _require(np.isfinite(threshold) and threshold >= 0, 'Invalid threshold')
    prediction = np.asarray(trace['predictions'])
    cap = prediction.shape[-1] - 1
    _require(cap >= 1, 'Trace needs at least one update')
    if criterion == 'output_stability':
        _require(threshold >= 1 and int(threshold) == threshold,
                 'Stability patience must be a positive integer')
        unchanged = prediction[..., 1:] == prediction[..., :-1]
        streak = np.zeros(prediction.shape[:-1], dtype=np.int64)
        hits = np.zeros_like(unchanged, dtype=bool)
        for index in range(cap):
            streak = np.where(unchanged[..., index], streak + 1, 0)
            hits[..., index] = streak >= threshold
    else:
        hits = np.asarray(trace[criterion])[..., 1:] <= threshold
    return np.where(hits.any(axis=-1), hits.argmax(axis=-1) + 1, cap).astype(np.int64)


def _metric(trace, steps, hops, extra_read=False):
    predictions = np.take_along_axis(trace['predictions'], steps[..., None], axis=-1)[..., 0]
    correct = predictions == trace['answer']
    by_hop = []
    for hop in hops:
        mask = trace['hop'] == hop
        _require(mask.any(), f'Missing hop {hop}')
        by_hop.append(dict(hop=hop, accuracy=float(correct[mask].mean()),
                           mean_steps=float(steps[mask].mean()),
                           mean_reads=float(steps[mask].mean() + int(extra_read)), n=int(mask.sum())))
    by_hop.append(dict(hop='overall', **{
        key: float(np.mean([row[key] for row in by_hop]))
        for key in ('accuracy', 'mean_steps', 'mean_reads')}, n=int(correct.size)))
    return by_hop


def select_thresholds(root, runs, design):
    """Choose one threshold per family/criterion using validation traces only."""
    metadata = _metadata(runs, design)
    grids = {criterion: _grid(design, criterion) for criterion in CRITERIA}
    selected, grid = [], []
    # Hold one run at a time, including when the production traces are large.
    scores = {}
    for run in sorted(runs, key=lambda r: r['name']):
        trace = _load_trace(root, run, 'val', design)
        for criterion in CRITERIA:
            for threshold in grids[criterion]:
                steps = halting_steps(trace, criterion, threshold)
                metric = _metric(trace, steps, design['hops'])[-1]
                scores.setdefault((run['family'], criterion, threshold), []).append(metric)
    for family in sorted({run['family'] for run in runs}):
        for criterion in CRITERIA:
            candidates = []
            for threshold in grids[criterion]:
                metrics = scores[family, criterion, threshold]
                row = dict(family=family, criterion=criterion, threshold=threshold,
                           validation_accuracy=float(np.mean([m['accuracy'] for m in metrics])),
                           validation_mean_steps=float(np.mean([m['mean_steps'] for m in metrics])))
                candidates.append(row)
                grid.append(row)
            best = min(candidates, key=lambda row: (-row['validation_accuracy'],
                                                    row['validation_mean_steps'], row['threshold']))
            selected.append(dict(best))
    return dict(selected=selected, grid=grid, metadata=metadata)


def _validate_selection(selection, runs, design):
    _require(selection.get('metadata') == _metadata(runs, design),
             'Selection metadata differs from the frozen runs/design')
    expected = {(r['family'], c) for r in runs for c in CRITERIA}
    rows = selection['selected']
    _require(len(rows) == len(expected) and {(r['family'], r['criterion']) for r in rows} == expected,
             'Selection must have exactly one threshold per family/criterion')
    lookup = {}
    for row in rows:
        _require('seed' not in row and row['threshold'] in _grid(design, row['criterion']),
                 'Selected thresholds must be family-wide members of the locked grid')
        candidates = [g for g in selection['grid'] if
                      (g['family'], g['criterion']) == (row['family'], row['criterion'])]
        _require(len(candidates) == len(_grid(design, row['criterion']))
                 and {g['threshold'] for g in candidates} == set(_grid(design, row['criterion'])),
                 'Frozen selection grid is incomplete')
        best = min(candidates, key=lambda g: (-g['validation_accuracy'],
                                             g['validation_mean_steps'], g['threshold']))
        _require(row == best, 'Selected threshold disagrees with frozen validation scores')
        lookup[row['family'], row['criterion']] = row['threshold']
    return lookup


def _certificate(values):
    values = np.asarray(values)
    count = int(np.count_nonzero(values < 1))
    return dict(n=int(values.size), c_lt_1_count=count, c_lt_1_percent=100 * count / values.size,
                c_min=float(values.min()), c_median=float(np.median(values)),
                c_p95=float(np.percentile(values, 95)), c_max=float(values.max()),
                sufficient_bound_vacuous_for_all_episodes=count == 0)


def _pool_accuracy(per_seed):
    groups = {}
    keys = ACCURACY_COLUMNS[:7]
    for row in per_seed:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return [dict(zip(keys, key), **{k: float(np.mean([r[k] for r in rows]))
                                  for k in ('accuracy', 'mean_steps', 'mean_reads')},
                 n=sum(r['n'] for r in rows)) for key, rows in groups.items()]


def _latency(root, runs, out):
    path = Path(root) / 'latency.json'
    if not path.exists():
        return dict(available=False, per_run_rows=0, family_rows=0)
    rows = json.loads(path.read_text())
    _require(isinstance(rows, list), 'latency.json must be a list of per-run rows')
    sources = {r['name']: r for r in runs}
    groups = {}
    valid_methods = {*(f'fixed_{n}' for n in (1, 2, 4, 8, 16)), *CRITERIA}
    identity = ('family', 'writer', 'r', 'variant', 'seed')
    group_keys = ('writer', 'r', 'variant', 'family', 'method', 'batch_size', 'warmup', 'repeats')
    fields = ('mean_steps', 'mean_reads', 'cuda_ms_mean', 'cuda_ms_median', 'cuda_ms_p95',
              'wall_ms_mean', 'wall_ms_median', 'wall_ms_p95')
    seen = set()
    for row in rows:
        _require(row['name'] in sources and all(row[k] == sources[row['name']][k] for k in identity),
                 'Latency row run identity differs')
        _require(row['method'] in valid_methods, 'Unknown latency method')
        hop = row.get('latency_hop', 0)
        _require(isinstance(hop, int) and 0 <= hop <= 8, 'Invalid latency query hop')
        key = (row['name'], row['method'], row['batch_size'], hop)
        _require(key not in seen, 'Duplicate latency row')
        seen.add(key)
        _require(all(np.isfinite(row[k]) and row[k] >= 0 for k in fields), 'Invalid latency metric')
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    pooled = []
    for key, members in groups.items():
        family = members[0]['family']
        seeds = {r['seed'] for r in members}
        _require(seeds == {r['seed'] for r in runs if r['family'] == family},
                 'Latency family group must include every training seed')
        seed_hops = {seed: {r.get('latency_hop', 0) for r in members if r['seed'] == seed}
                     for seed in seeds}
        hops = seed_hops[members[0]['seed']]
        _require(all(member_hops == hops for member_hops in seed_hops.values()),
                 'Latency training seeds must include matching query hops')
        if any('latency_hop' in r for r in members):
            _require(all('latency_hop' in r for r in members), 'Latency hop metadata is incomplete')
            expected_hops = set(range(1, 9)) if members[0]['batch_size'] == 1 else {0}
            _require(hops == expected_hops,
                     'Latency batch 1 requires all eight query hops; batch 64 requires hop 0')
        pooled.append(dict(zip(group_keys, key), seeds=len(seeds),
                           **{k: float(np.mean([r[k] for r in members])) for k in fields}))
    _write_csv(out / 'latency_per_seed.csv', rows)
    _write_csv(out / 'latency.csv', pooled)
    return dict(available=True, per_run_rows=len(rows), family_rows=len(pooled),
                aggregation='equal mean over seeds and query hops; percentiles are means of '
                'per-run/query-hop percentiles')


def _compact_family(run, rows, certificates, thresholds):
    family = run['family']
    test = {(r['method'], r['hop']): r for r in rows if r['family'] == family and r['split'] == 'test'}

    def score(method, hops):
        members = [test[method, hop] for hop in hops]
        return {k: float(np.mean([r[k] for r in members]))
                for k in ('accuracy', 'mean_steps', 'mean_reads')}

    endpoints = {}
    for label, hops in (('hop1_overshoot_recovery', [1]),
                        ('long_hops_4to8', [4, 5, 6, 7, 8]), ('overall', ['overall'])):
        fixed1, fixed4 = score('fixed_1', hops), score('fixed_4', hops)
        adaptive = {}
        for criterion in CRITERIA:
            result = score(criterion, hops)
            result.update(delta_fixed_1=result['accuracy'] - fixed1['accuracy'],
                          delta_fixed_4=result['accuracy'] - fixed4['accuracy'])
            if label == 'hop1_overshoot_recovery':
                gap = fixed1['accuracy'] - fixed4['accuracy']
                result['recovered_fraction_of_fixed1_minus_fixed4'] = (
                    result['delta_fixed_4'] / gap if gap > 0 else None)
            adaptive[criterion] = result
        endpoints[label] = dict(fixed_1=fixed1, fixed_4=fixed4, adaptive=adaptive)
    return dict(writer=run['writer'], r=run['r'], variant=run['variant'], family=family,
                thresholds={criterion: thresholds[family, criterion] for criterion in CRITERIA},
                contraction={r['split']: {k: v for k, v in r.items()
                                         if k not in ('writer', 'r', 'variant', 'family', 'split')}
                             for r in certificates if r['family'] == family},
                screening_test=endpoints)


def summarize(root, runs, design, selection, out):
    """Apply frozen family thresholds; write compact CSV tables and summary JSON."""
    thresholds = _validate_selection(selection, runs, design)  # Before opening test.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed, per_run_c, all_c = [], [], {}
    for run in sorted(runs, key=lambda r: r['name']):
        base = {k: run[k] for k in ('writer', 'r', 'variant', 'family')}
        for split in ('val', 'test'):
            trace = _load_trace(root, run, split, design)
            methods = [(f'fixed_{n}', np.full(trace['answer'].shape, n, dtype=np.int64), False)
                       for n in design['fixed_references']]
            methods += [(criterion, halting_steps(trace, criterion, thresholds[run['family'], criterion]),
                         criterion == 'read_norm') for criterion in CRITERIA]
            methods.append(('oracle_true_hop', trace['hop'].astype(np.int64), False))
            for method, steps, extra_read in methods:
                _require(np.all((steps >= 1) & (steps <= design['max_steps'])), 'Step outside trace')
                per_seed.extend(dict(base, split=split, method=method, **row,
                                     name=run['name'], seed=run['seed'])
                                for row in _metric(trace, steps, design['hops'], extra_read))
            per_run_c.append(dict(base, split=split, name=run['name'], seed=run['seed'],
                                  **_certificate(trace['certified_c'])))
            all_c.setdefault((run['family'], split), []).append(trace['certified_c'])
    accuracy = _pool_accuracy(per_seed)
    by_family = {r['family']: r for r in runs}
    contraction = []
    for (family, split), values in sorted(all_c.items()):
        base = {k: by_family[family][k] for k in ('writer', 'r', 'variant', 'family')}
        contraction.append(dict(base, split=split, **_certificate(np.concatenate(values))))
    _write_csv(out / 'accuracy.csv', accuracy, ACCURACY_COLUMNS)
    _write_csv(out / 'accuracy_per_seed.csv', per_seed, (*ACCURACY_COLUMNS, 'name', 'seed'))
    _write_csv(out / 'contraction.csv', contraction)
    _write_csv(out / 'contraction_per_run.csv', per_run_c)
    _write_csv(out / 'thresholds.csv', selection['selected'])
    _write_csv(out / 'threshold_grid.csv', selection['grid'])
    latency = _latency(root, runs, out)
    summary = dict(task='WP-C3', runs=len(runs), selection_split='val', report_split='test',
                   accuracy_aggregation='equal mean over training seeds and hops',
                   halting_convention=design['halting_convention'],
                   oracle_reference=design['oracle_reference'],
                   contraction_interpretation='c < 1 is sufficient for contraction, not correctness; '
                   'c >= 1 is inconclusive, not evidence of divergence', latency=latency,
                   families=[_compact_family(by_family[f], accuracy, contraction, thresholds)
                             for f in sorted(by_family)])
    _write_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    selection = json.loads((args.root / 'thresholds.json').read_text())
    summarize(args.root, manifest['runs'], manifest['design'], selection, args.out or args.root)


if __name__ == '__main__':
    main()
