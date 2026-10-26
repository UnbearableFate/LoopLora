#!/usr/bin/env python3
"""WP-C4 screening point estimates and compact audit; run on compute."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket

import numpy as np

from analyze_p4s_stopping import (
    ACCURACY_COLUMNS, CRITERIA, _load_trace, _metric, _pool_accuracy,
    _validate_selection, halting_steps, select_thresholds,
)
from audit_p4s_stopping import (
    audit_threshold_ties, digest_checkpoint, markdown_table, no_confirmation_path, require, require_compute,
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, columns=None):
    columns = columns or list(rows[0])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def g1_estimates(trace, steps):
    """Paired stopped-minus-L1 accuracy differences, with equal hop weights."""
    stopped = np.take_along_axis(trace['predictions'], steps[..., None], axis=-1)[..., 0]
    difference = ((stopped == trace['answer']).astype(np.float64)
                  - (trace['predictions'][..., 1] == trace['answer']))

    def delta(hops):
        return float(np.mean([difference[trace['hop'] == h].mean() for h in hops]))

    h1, reach, far = delta([1]), delta([2, 3]), delta([6, 7, 8])
    return dict(hop1_delta=h1, reach_delta=reach, far_delta=far, walk_specificity=reach - far)


def candidate(row):
    # The F2 hop-1 requirement is nondegradation by more than 1 pp.
    return bool(row['hop1_delta'] >= -.01 - 1e-12 and row['reach_delta'] > 0
                and row['walk_specificity'] > 0)


def summarize(root, runs, design, selection, out):
    thresholds = _validate_selection(selection, runs, design)  # Before opening test traces.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed, endpoints_per_seed = [], []
    for run in sorted(runs, key=lambda r: r['name']):
        base = {k: run[k] for k in ('writer', 'r', 'variant', 'family')}
        for split in ('val', 'test'):
            trace = _load_trace(root, run, split, design)
            methods = [(f'fixed_{n}', np.full(trace['answer'].shape, n, dtype=np.int64))
                       for n in design['fixed_references']]
            methods += [(c, halting_steps(trace, c, thresholds[run['family'], c])) for c in CRITERIA]
            methods.append(('oracle_true_hop', trace['hop']))
            for method, steps in methods:
                per_seed.extend(dict(base, split=split, method=method, name=run['name'], seed=run['seed'], **m)
                                for m in _metric(trace, steps, design['hops'], method == 'read_norm'))
                if split == 'test':
                    endpoints_per_seed.append(dict(base, method=method, name=run['name'], seed=run['seed'],
                                                   **g1_estimates(trace, steps)))
    accuracy = _pool_accuracy(per_seed)
    groups = {}
    for row in endpoints_per_seed:
        groups.setdefault((row['family'], row['method']), []).append(row)
    endpoints = []
    for (family, method), rows in sorted(groups.items()):
        row = {k: rows[0][k] for k in ('writer', 'r', 'variant', 'family', 'method')}
        row.update({k: float(np.mean([r[k] for r in rows]))
                    for k in ('hop1_delta', 'reach_delta', 'far_delta', 'walk_specificity')})
        row.update(screening_candidate=candidate(row),
                   min_seed_reach_delta=min(r['reach_delta'] for r in rows),
                   min_seed_walk_specificity=min(r['walk_specificity'] for r in rows))
        endpoints.append(row)
    write_csv(out / 'accuracy.csv', accuracy, ACCURACY_COLUMNS)
    write_csv(out / 'accuracy_per_seed.csv', per_seed, (*ACCURACY_COLUMNS, 'name', 'seed'))
    write_csv(out / 'endpoints.csv', endpoints)
    write_csv(out / 'endpoints_per_seed.csv', endpoints_per_seed)
    write_csv(out / 'thresholds.csv', selection['selected'])
    write_csv(out / 'threshold_grid.csv', selection['grid'])
    lines = ['# WP-C4 screening tables', '',
             'Accuracy is percent; deltas are percentage points. Equal means over seeds 17/29/43 and '
             'the specified hops. Thresholds maximize validation accuracy over hops 1–8, as WP-C3. '
             'No bootstrap was implemented in WP-C3; all quantities here are screening point estimates.', '',
             'First qualifying post-update state z^L, 1≤L≤16; read norm costs L+1 reads, other rules L. '
             'True-hop stopping is label-informed, not an upper bound. The candidate flag requires '
             'hop-1 delta ≥−1 pp, positive reach and positive walk specificity; it is not formal G1 acceptance.', '']
    for family in sorted({r['family'] for r in runs}):
        lines += [f'## {family}', '', 'Validation selection:', '']
        lines += markdown_table(['Rule', 'Threshold', 'Val accuracy (%)', 'Val steps'],
                                [[r['criterion'], r['threshold'], f"{100*r['validation_accuracy']:.3f}",
                                  f"{r['validation_mean_steps']:.3f}"]
                                 for r in selection['selected'] if r['family'] == family])
        lines += ['', 'Screening test:', '']
        lookup = {(r['method'], r['hop']): r for r in accuracy
                  if r['family'] == family and r['split'] == 'test'}
        methods = [*(f'fixed_{n}' for n in design['fixed_references']), *CRITERIA, 'oracle_true_hop']
        lines += markdown_table(['Method', *[f'h{h}' for h in design['hops']], 'Overall', 'Steps', 'Reads'],
                                [[m, *[f"{100*lookup[m, h]['accuracy']:.2f}" for h in design['hops']],
                                  f"{100*lookup[m, 'overall']['accuracy']:.2f}",
                                  f"{lookup[m, 'overall']['mean_steps']:.3f}",
                                  f"{lookup[m, 'overall']['mean_reads']:.3f}"] for m in methods])
        lines += ['', 'Stopped-minus-L1 screening endpoints:', '']
        lines += markdown_table(['Method', 'Hop1 Δ', 'Reach Δ (2,3)', 'Far Δ (6,7,8)',
                                 'Walk specificity', 'Candidate'],
                                [[r['method'], *[f"{100*r[k]:+.3f}" for k in
                                  ('hop1_delta', 'reach_delta', 'far_delta', 'walk_specificity')],
                                  r['screening_candidate']] for r in endpoints if r['family'] == family])
        lines.append('')
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    return endpoints


def audit(root, out):
    root, out = Path(root), Path(out)
    no_confirmation_path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['status'] == 'complete' and manifest['smoke'] is False, 'Main run must be complete')
    runs, design = manifest['runs'], manifest['design']
    require(len(runs) == 18 and design['episodes_per_split'] == 2000
            and design['evaluation_seed'] == 4242 and design['splits'] == ['val', 'test']
            and design['collect_certificates'] is False and design['latency'] is False,
            'Unexpected WP-C4 evaluation scope')
    selection = json.loads((root / 'thresholds.json').read_text())
    recomputed = select_thresholds(root, runs, design)
    require({k: v for k, v in selection.items() if k != 'locked_at'} == recomputed,
            'Frozen selection differs from validation-only recomputation')
    lock = datetime.fromisoformat(selection['locked_at']).timestamp()
    val_times, test_times, comparisons = [], [], []
    episodes = queries = 0
    for run in runs:
        checkpoint = Path(run['checkpoint'])
        no_confirmation_path(checkpoint)
        require(digest_checkpoint(checkpoint) == run['checkpoint_blake2b'], 'Checkpoint changed')
        traces = {}
        for split, timestamps in [('val', val_times), ('test', test_times)]:
            path = root / 'traces' / run['name'] / f'{split}.npz'
            timestamps.append(path.stat().st_mtime)
            trace = traces[split] = _load_trace(root, run, split, design)
            require(trace['answer'].shape == (2000, 8), 'Incomplete trace')
            with np.load(path, allow_pickle=False) as archive:
                require('certified_c' not in archive.files, 'Unexpected certificate evaluation')
            episodes += trace['answer'].shape[0]
            queries += trace['answer'].size
        require(np.intersect1d(traces['val']['episode_id'][:, 0],
                              traces['test']['episode_id'][:, 0]).size == 0, 'Val/test episodes overlap')
        reference_path = Path(run['reference_metrics'])
        no_confirmation_path(reference_path)
        reference = json.loads(reference_path.read_text())
        cells = {(r['L'], r['hop']): r for r in reference['table']
                 if r['split'] == 'test' and r['condition'] == 'original'
                 and r['L'] in design['fixed_references'] and r['hop'] in design['hops']}
        require(len(cells) == 32, 'Released screening reference is incomplete')
        trace = traces['test']
        for depth in design['fixed_references']:
            for hop in design['hops']:
                mask = trace['hop'] == hop
                actual = float((trace['predictions'][..., depth][mask] == trace['answer'][mask]).mean())
                require(cells[depth, hop]['n'] == 2000, 'Reference query count differs')
                comparisons.append(dict(name=run['name'], L=depth, hop=hop, accuracy=actual,
                                        reference_accuracy=cells[depth, hop]['accuracy'],
                                        delta=actual - cells[depth, hop]['accuracy']))
    require(max(val_times) <= lock < min(test_times), 'Threshold lock did not precede screening')
    timing = json.loads((root / 'timing.json').read_text())
    require(len(timing) == 36 and all(r['certificate_seconds'] == 0 for r in timing)
            and not (root / 'latency.json').exists(), 'Unexpected certificate or latency workload')
    out.mkdir(parents=True, exist_ok=True)
    endpoints = summarize(root, runs, design, selection, out)
    write_csv(out / 'reference_check.csv', comparisons)
    shutil.copy2(root / 'manifest.json', out / 'manifest.json')
    shutil.copy2(root / 'thresholds.json', out / 'selection.json')
    ties = audit_threshold_ties(selection, {r['family']: 48000 for r in runs})
    changed = [r for r in comparisons if r['delta'] != 0]
    result = dict(status='passed', task='WP-C4', audited_at=datetime.now(timezone.utc).isoformat(),
                  runtime=dict(host=socket.gethostname(), job_id=os.environ.get('PBS_JOBID')),
                  root=str(root), runs=18, families=6, checkpoints_unchanged=True,
                  confirmation_inputs_opened=False, latency_evaluated=False, certificates_evaluated=False,
                  traces=dict(files=36, episodes=episodes, queries=queries, all_metrics_finite=True),
                  threshold_lock=dict(locked_at=selection['locked_at'],
                                      seconds_after_last_val=lock-max(val_times),
                                      seconds_before_first_test=min(test_times)-lock),
                  threshold_integer_ties=ties,
                  reference_check=dict(cells=len(comparisons), exact_cells=len(comparisons)-len(changed),
                                       differing_cells=len(changed),
                                       max_absolute_delta=max(abs(r['delta']) for r in comparisons)),
                  adaptive_screening_candidates=[r for r in endpoints
                                                 if r['method'] in CRITERIA and r['screening_candidate']])
    save(out / 'audit.json', result)
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    require(size <= design['git_artifact_max_bytes'], 'Compact artifacts exceed 1 MB')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    print(json.dumps(audit(args.root, args.out)), flush=True)


if __name__ == '__main__':
    main()
