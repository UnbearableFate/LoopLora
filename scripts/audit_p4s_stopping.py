#!/usr/bin/env python3
"""Audit the completed WP-C3 screening run and export small report artifacts.

Run through scripts/submit.py in a Miyabi-G allocation. No confirmation archive
or model is opened; the F1 comparison uses only screening rows in its CSV.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket


FIXED = (1, 2, 4, 8, 16)
ADAPTIVE = ('read_norm', 'relative_change', 'output_stability')
METHODS = tuple(f'fixed_{step}' for step in FIXED) + ADAPTIVE
REPORT_METHODS = METHODS + ('oracle_true_hop',)
SMALL_TABLES = ('accuracy.csv', 'contraction.csv', 'latency.csv', 'thresholds.csv',
                'threshold_grid.csv', 'summary.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_compute():
    host = socket.gethostname().split('.')[0]
    nodefile = Path(os.environ.get('PBS_NODEFILE', '/nonexistent-pbs-nodefile'))
    hosts = {line.strip().split('.')[0] for line in nodefile.read_text().splitlines()} if nodefile.is_file() else set()
    require(bool(re.fullmatch(r'mg\d+', host)) and bool(os.environ.get('PBS_JOBID')) and host in hosts,
            'Run only inside a confirmed Miyabi-G PBS allocation via scripts/submit.py')


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest_checkpoint(path):
    digest = hashlib.blake2b(digest_size=16)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def no_confirmation_path(path):
    require(not any('confirmation' in part.lower() for part in path.resolve().parts),
            f'Confirmation input is outside the audit scope: {path}')


def check_manifest(manifest):
    require(manifest.get('status') == 'complete' and manifest.get('smoke') is False,
            'Audit requires a completed main run, not smoke artifacts')
    runs, design = manifest['runs'], manifest['design']
    expected = {(writer, rank, variant, seed) for writer in ('oracle_random', 'joint')
                for rank in (64, 128) for variant in ('M1', 'M1-lin') for seed in (17, 29, 43)}
    actual = [(r['writer'], r['r'], r['variant'], r['seed']) for r in runs]
    require(len(runs) == 24 and set(actual) == expected and len(set(actual)) == 24,
            'Manifest must contain exactly the 24 locked F1 checkpoint identities')
    require(len({r['name'] for r in runs}) == 24, 'Manifest run names are not unique')
    require(design['episodes_per_split'] == 2000 and design['hops'] == list(range(1, 9))
            and design['max_steps'] == 16 and design['minimum_steps'] == 1
            and design['splits'] == ['val', 'test'] and design['evaluation_seed'] == 4242
            and design['fixed_references'] == list(FIXED), 'Main evaluation design differs')
    require(design['latency']['warmup'] == 5 and design['latency']['repeats'] == 20,
            'Latency protocol differs from the locked design')
    return runs, design


def check_latency(rows, runs, design):
    sources = {r['name']: r for r in runs}
    expected = {(r['name'], method, batch, hop) for r in runs for method in METHODS
                for batch, hop in [(64, 0), *((1, h) for h in range(1, 9))]}
    actual = set()
    state_errors = {criterion: [] for criterion in ADAPTIVE}
    timing_fields = tuple(f'{clock}_ms_{stat}' for clock in ('cuda', 'wall')
                          for stat in ('mean', 'median', 'p95'))
    for row in rows:
        key = (row['name'], row['method'], row['batch_size'], row['latency_hop'])
        require(key in expected and key not in actual, f'Unexpected or duplicate latency cell: {key}')
        actual.add(key)
        run = sources[row['name']]
        require(all(row[k] == run[k] for k in ('family', 'writer', 'r', 'variant', 'seed')),
                f'Latency source identity differs: {key}')
        require(row['warmup'] == design['latency']['warmup']
                and row['repeats'] == design['latency']['repeats'], f'Latency timing settings differ: {key}')
        require(all(math.isfinite(row[field]) and row[field] >= 0 for field in timing_fields),
                f'Nonfinite/negative latency measurement: {key}')
        require(1 <= row['mean_steps'] <= 16
                and row['mean_reads'] == row['mean_steps'] + int(row['method'] == 'read_norm'),
                f'Latency step/read counts differ: {key}')
        if row['method'].startswith('fixed_'):
            require(row['mean_steps'] == int(row['method'].split('_')[1]), f'Fixed latency depth differs: {key}')
        else:
            relative, absolute = row['audit_state_rel_l2_max'], row['audit_state_abs_max']
            require(math.isfinite(relative) and relative >= 0 and math.isfinite(absolute) and absolute >= 0,
                    f'Adaptive state audit is missing or nonfinite: {key}')
            state_errors[row['method']].append((relative, absolute))
    require(len(rows) == 1728 and actual == expected, 'Latency matrix must have exactly 1728 cells')
    return dict(rows=len(rows), runs=24, methods=8, batch64_rows=192, batch1_rows=1536,
                batch1_hops=list(range(1, 9)),
                max_state_relative_l2_drift=max(v[0] for values in state_errors.values() for v in values),
                max_state_absolute_component_drift=max(v[1] for values in state_errors.values() for v in values),
                per_criterion={criterion: dict(cells=len(values), max_relative_l2=max(v[0] for v in values),
                                               max_absolute_component=max(v[1] for v in values))
                               for criterion, values in state_errors.items()})


def compare_f1(reference_csv, fixed_cells, runs):
    """Stream the existing CSV; retain only the requested screening fixed cells."""
    no_confirmation_path(reference_csv)
    expected = {(r['name'], step, hop) for r in runs for step in FIXED for hop in range(1, 9)}
    names = {r['name'] for r in runs}
    reference = {}
    with reference_csv.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if (row['dataset'], row['split'], row['condition']) != ('screening', 'test', 'original'):
                continue
            if row['name'] not in names or int(row['L']) not in FIXED:
                continue
            key = (row['name'], int(row['L']), int(row['hop']))
            require(key in expected and key not in reference, f'Duplicate/unexpected F1 screening cell: {key}')
            require(int(row['n']) == 2000, f'F1 reference cell has wrong episode count: {key}')
            value = float(row['accuracy'])
            require(math.isfinite(value) and 0 <= value <= 1, f'Invalid F1 reference accuracy: {key}')
            reference[key] = value
    require(len(reference) == len(fixed_cells) == 960 and set(reference) == set(fixed_cells) == expected,
            'F1 comparison requires all 960 matching screening fixed-depth cells')
    deltas = [(key, fixed_cells[key] - reference[key]) for key in sorted(expected)]
    changed = [(key, delta) for key, delta in deltas if delta != 0]
    explanation = ('Every fixed-depth screening accuracy is an exact match.' if not changed else
                   'Nonzero deltas are retained and do not fail this comparison. FP32 GEMM shapes and '
                   'near-tied argmax predictions can cause small differences, but aggregate CSV '
                   'accuracies alone cannot establish their cause; no rounding or tolerance hides them.')
    return dict(expected_cells=960, matched_cells=len(deltas), exact_cells=len(deltas) - len(changed),
                differing_cells=len(changed), max_absolute_accuracy_delta=max(abs(v) for _, v in deltas),
                mean_absolute_accuracy_delta=sum(abs(v) for _, v in deltas) / len(deltas),
                largest_differences=[dict(name=key[0], L=key[1], hop=key[2],
                                          p4s_accuracy=fixed_cells[key], f1_accuracy=reference[key], delta=delta)
                                     for key, delta in sorted(changed, key=lambda p: -abs(p[1]))[:8]],
                explanation=explanation,
                reference_scope='CSV rows with dataset=screening, split=test, condition=original only')


def audit_threshold_ties(selection, family_queries):
    """Report true equal-count ties without changing any frozen threshold."""
    require(set(family_queries.values()) == {48000},
            'Every validation family must contain 2000 episodes x 8 hops x 3 seeds')
    groups = {}
    for row in selection['grid']:
        n = family_queries[row['family']]
        accuracy, steps = row['validation_accuracy'], row['validation_mean_steps']
        require(math.isfinite(accuracy) and 0 <= accuracy <= 1 and math.isfinite(steps)
                and 1 <= steps <= 16, 'Invalid frozen threshold-grid validation score')
        correct_float, steps_float = accuracy * n, steps * n
        correct, total_steps = round(correct_float), round(steps_float)
        require(abs(correct_float - correct) < 1e-6 and abs(steps_float - total_steps) < 1e-6,
                'Frozen validation scores do not correspond to integer query counts')
        groups.setdefault((row['family'], row['criterion']), []).append(
            dict(threshold=row['threshold'], correct=correct, total_steps=total_steps))
    selected = {(r['family'], r['criterion']): r for r in selection['selected']}
    mismatches, true_tie_groups = [], 0
    for key, grid in sorted(groups.items()):
        best = min(grid, key=lambda r: (-r['correct'], r['total_steps'], r['threshold']))
        maximum = max(row['correct'] for row in grid)
        tied = [row for row in grid if row['correct'] == maximum]
        true_tie_groups += int(len(tied) > 1)
        actual = next(row for row in grid if row['threshold'] == selected[key]['threshold'])
        if actual['threshold'] != best['threshold']:
            mismatches.append(dict(family=key[0], criterion=key[1],
                                   selected_threshold=actual['threshold'],
                                   integer_tiebreak_threshold=best['threshold'],
                                   selected_correct=actual['correct'], maximum_correct=maximum,
                                   selected_total_steps=actual['total_steps'],
                                   integer_tiebreak_total_steps=best['total_steps'],
                                   true_equal_accuracy_count=actual['correct'] == maximum))
    return dict(queries_per_family=48000, family_criteria=len(groups),
                groups_with_equal_maximum_correct_counts=true_tie_groups,
                mismatch_count=len(mismatches), mismatches=mismatches,
                interpretation='Integer-equivalent accuracy and step totals expose true ties that '
                'can differ by one floating-point ulp after averaging. Mismatches are reported '
                'for review; frozen selections are not changed by this audit.')


def markdown_table(columns, rows):
    return ['| ' + ' | '.join(str(c) for c in columns) + ' |',
            '| ' + ' | '.join('---' for _ in columns) + ' |',
            *('| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows)]


def write_tables(out, tables, selection, manifest, audit):
    accuracy = read_csv(tables / 'accuracy.csv')
    contraction = read_csv(tables / 'contraction.csv')
    latency = read_csv(tables / 'latency.csv')
    families = sorted({r['family'] for r in manifest['runs']})
    expected_accuracy = {(family, split, method, str(hop)) for family in families for split in ('val', 'test')
                         for method in REPORT_METHODS for hop in [*range(1, 9), 'overall']}
    actual_accuracy = [(r['family'], r['split'], r['method'], r['hop']) for r in accuracy]
    require(len(actual_accuracy) == len(expected_accuracy) and set(actual_accuracy) == expected_accuracy,
            'Family accuracy table coverage differs')
    require(all(int(r['n']) == (48000 if r['hop'] == 'overall' else 6000) for r in accuracy),
            'Family accuracy table query counts differ')
    acc = {(r['family'], r['method'], r['hop']): r for r in accuracy if r['split'] == 'test'}
    latency_keys = [(r['family'], r['method'], int(r['batch_size'])) for r in latency]
    require(len(latency_keys) == 128 and set(latency_keys) == {
        (family, method, batch) for family in families for method in METHODS for batch in (1, 64)},
        'Family latency table coverage differs')
    require(all(int(r['seeds']) == 3 for r in latency), 'Family latency seed counts differ')
    require(len(contraction) == 16 and {(r['family'], r['split']) for r in contraction} == {
        (family, split) for family in families for split in ('val', 'test')},
        'Family contraction table coverage differs')
    headlines = []
    for family in families:
        for method in REPORT_METHODS:
            overall = acc[family, method, 'overall']
            headlines.append(dict(writer=overall['writer'], r=int(overall['r']),
                                  variant=overall['variant'], family=family, method=method,
                                  hop1_accuracy=float(acc[family, method, '1']['accuracy']),
                                  long_hops_4to8_accuracy=sum(float(acc[family, method, str(h)]['accuracy'])
                                                             for h in range(4, 9)) / 5,
                                  overall_accuracy=float(overall['accuracy']),
                                  mean_steps=float(overall['mean_steps']),
                                  mean_reads=float(overall['mean_reads'])))
    with (out / 'headline.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(headlines[0]))
        writer.writeheader()
        writer.writerows(headlines)
    lines = ['# WP-C3 screening tables', '',
             'Accuracy is in percent, equally averaging three training seeds and eight hops. '
             'All reported accuracy below is screening test data; thresholds were selected on validation.', '',
             manifest['design']['halting_convention'], '',
             'The true-hop oracle is label informed and is not a correctness guarantee. '
             'Read-norm stopping costs one extra diagnostic memory read. '
             'A certificate c < 1 is sufficient for contraction, not correctness; c >= 1 is inconclusive.', '',
             'Latency includes prepared-memory reader, candidate prediction and stopping bookkeeping. '
             'Each run uses 5 warmups and 20 repeats. Batch-1 summaries equally average one separate '
             'query per hop 1–8 and all three seeds; batch-64 summaries average the three seeds. '
             'Median/p95 columns are means of run/query-hop percentiles, not pooled percentiles.', '']
    for family in families:
        lines += [f'## {family}', '', 'Validation threshold selection:', '']
        lines += markdown_table(['Criterion', 'Threshold', 'Validation accuracy (%)', 'Validation mean steps'],
                                [[r['criterion'], r['threshold'], f"{100*r['validation_accuracy']:.3f}",
                                  f"{r['validation_mean_steps']:.3f}"]
                                 for r in selection['selected'] if r['family'] == family])
        lines += ['', 'Screening accuracy by hop and mean reader cost:', '']
        values = []
        for method in REPORT_METHODS:
            overall = acc[family, method, 'overall']
            long_accuracy = sum(float(acc[family, method, str(h)]['accuracy']) for h in range(4, 9)) / 5
            values.append([method, *(f"{100*float(acc[family, method, str(h)]['accuracy']):.3f}"
                                      for h in range(1, 9)), f'{100*long_accuracy:.3f}',
                           f"{100*float(overall['accuracy']):.3f}",
                           f"{float(overall['mean_steps']):.3f}", f"{float(overall['mean_reads']):.3f}"])
        lines += markdown_table(['Method', *(f'H{h}' for h in range(1, 9)), 'H4–8', 'Overall', 'Steps', 'Reads'], values)
        lines += ['', 'Latency in milliseconds:', '']
        lat = {(r['method'], int(r['batch_size'])): r for r in latency if r['family'] == family}
        lines += markdown_table(['Method', 'Batch', 'GPU mean/median/p95', 'Wall mean/median/p95', 'Steps', 'Reads'],
                                [[method, batch,
                                  '/'.join(f"{float(lat[method,batch][f'cuda_ms_{s}']):.4f}"
                                           for s in ('mean', 'median', 'p95')),
                                  '/'.join(f"{float(lat[method,batch][f'wall_ms_{s}']):.4f}"
                                           for s in ('mean', 'median', 'p95')),
                                  f"{float(lat[method,batch]['mean_steps']):.3f}",
                                  f"{float(lat[method,batch]['mean_reads']):.3f}"]
                                 for method in METHODS for batch in (1, 64)])
        lines += ['', 'Contraction certificates (one value per episode):', '']
        c_rows = [r for r in contraction if r['family'] == family]
        lines += markdown_table(['Split', 'n', 'c<1 count', 'c<1 (%)', 'Minimum', 'Median', 'p95', 'Maximum'],
                                [[r['split'], r['n'], r['c_lt_1_count'], f"{float(r['c_lt_1_percent']):.3f}",
                                  *(f"{float(r[k]):.6g}" for k in ('c_min', 'c_median', 'c_p95', 'c_max'))]
                                 for r in c_rows])
        vacuous = [r['split'] for r in c_rows if int(r['c_lt_1_count']) == 0]
        if vacuous:
            lines += ['', 'The sufficient contraction bound is vacuous for every episode on: '
                      + ', '.join(vacuous) + '.']
        lines += ['']
    reference = audit['f1_fixed_reference']
    lines += ['## Artifact audit', '',
              f"Validated {audit['traces']['files']} trace files, {audit['traces']['episodes']} episode evaluations, "
              f"{audit['traces']['queries']} queries and {audit['latency']['rows']} latency cells. "
              'Checkpoint seals are unchanged and thresholds were locked before every test archive.', '',
              f"F1 fixed-reference comparison: {reference['exact_cells']}/960 exact cells; maximum absolute "
              f"difference {100*reference['max_absolute_accuracy_delta']:.6g} percentage points. "
              + reference['explanation'], '',
              f"Maximum adaptive/full-trace state relative L2 drift: "
              f"{audit['latency']['max_state_relative_l2_drift']:.6g}.", '',
              f"Integer-equivalent validation tie audit: "
              f"{audit['threshold_integer_ties']['mismatch_count']} frozen choices differ from "
              'the maximum correct count, then fewer total steps, then smaller threshold ordering. '
              'Any mismatches are recorded in audit.json for review; the audit never changes thresholds.', '']
    (out / 'tables.md').write_text('\n'.join(lines))


def audit(root, out, reference_csv):
    require_compute()  # Before importing application analysis/numerical dependencies.
    import numpy as np
    from analyze_p4s_stopping import _load_trace, _validate_selection

    root, out, reference_csv = Path(root).resolve(), Path(out).resolve(), Path(reference_csv).resolve()
    no_confirmation_path(root)
    manifest = read_json(root / 'manifest.json')
    runs, design = check_manifest(manifest)
    selection_path = root / 'thresholds.json'
    selection = read_json(selection_path)
    _validate_selection(selection, runs, design)
    locked = datetime.fromisoformat(selection['locked_at'].replace('Z', '+00:00'))
    require(locked.tzinfo is not None, 'Threshold lock needs a timezone-aware timestamp')
    lock_seconds = locked.timestamp()
    require({p.name for p in (root / 'traces').iterdir()} == {r['name'] for r in runs},
            'Trace root includes unexpected or missing run directories')
    expected_paths = {root / 'traces' / r['name'] / f'{split}.npz' for r in runs for split in ('val', 'test')}
    require(set((root / 'traces').glob('*/*')) == expected_paths, 'Trace files must be exactly val/test per run')
    episodes = queries = files = checkpoint_bytes = 0
    fixed_cells = {}
    family_val_queries = {r['family']: 0 for r in runs}
    earliest_test = math.inf
    latest_validation = -math.inf
    for run in runs:
        checkpoint = Path(run['checkpoint'])
        no_confirmation_path(checkpoint)
        require(digest_checkpoint(checkpoint) == run['checkpoint_blake2b'],
                f"Checkpoint seal changed: {run['name']}")
        checkpoint_bytes += checkpoint.stat().st_size
        for split in ('val', 'test'):
            path = root / 'traces' / run['name'] / f'{split}.npz'
            no_confirmation_path(path)
            require(not path.is_symlink(), f'Trace archive must not be a symlink: {path}')
            mtime = path.stat().st_mtime
            if split == 'val':
                latest_validation = max(latest_validation, mtime)
                require(mtime <= lock_seconds, 'Validation trace was modified after threshold locking')
            else:
                earliest_test = min(earliest_test, mtime)
                require(lock_seconds < mtime and selection_path.stat().st_mtime < mtime,
                        'Threshold lock/file must precede every screening test archive')
            data = _load_trace(root, run, split, design)
            require(data['predictions'].shape == (2000, 8, 17), 'Main trace must have shape 2000x8x17')
            require(np.all(np.isfinite(data['predictions'])) and np.all(np.isfinite(data['answer'])),
                    'Nonfinite trace predictions or answers')
            files += 1
            episodes += data['answer'].shape[0]
            queries += data['answer'].size
            if split == 'val':
                family_val_queries[run['family']] += data['answer'].size
            if split == 'test':
                for step in FIXED:
                    correct = data['predictions'][..., step] == data['answer']
                    for hop in range(1, 9):
                        fixed_cells[run['name'], step, hop] = float(correct[data['hop'] == hop].mean())
    require((files, episodes, queries) == (48, 96000, 768000), 'Main trace totals differ')
    latency = check_latency(read_json(root / 'latency.json'), runs, design)
    reference = compare_f1(reference_csv, fixed_cells, runs)
    threshold_ties = audit_threshold_ties(selection, family_val_queries)
    result = dict(status='passed', task='WP-C3', audited_at=datetime.now(timezone.utc).isoformat(),
                  root=str(root), runtime=dict(host=socket.gethostname(), job_id=os.environ['PBS_JOBID']),
                  runs=24, families=8, confirmation_inputs_opened=False,
                  checkpoints=dict(verified=24, seal='blake2b-128', bytes_streamed=checkpoint_bytes,
                                   all_unchanged=True),
                  threshold_lock=dict(locked_at=selection['locked_at'],
                                      seconds_after_last_val=lock_seconds - latest_validation,
                                      seconds_before_first_test=earliest_test - lock_seconds,
                                      metadata_frozen=True),
                  traces=dict(files=files, episodes=episodes, queries=queries,
                              shape=[2000, 8, 17], all_noninitial_metrics_finite=True,
                              all_certificates_finite=True),
                  latency=latency, f1_fixed_reference=reference, threshold_integer_ties=threshold_ties)
    tables = root / 'summaries'
    require(all((tables / name).is_file() for name in SMALL_TABLES), 'Source summary tables are incomplete')
    out.mkdir(parents=True, exist_ok=True)
    write_tables(out, tables, selection, manifest, result)
    for source, destination in [*((tables / name, out / name) for name in SMALL_TABLES),
                                (root / 'manifest.json', out / 'manifest.json'),
                                (selection_path, out / 'selection.json')]:
        require(source.stat().st_size < 5 * 1024 * 1024, f'Summary unexpectedly large: {source}')
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    result['exported'] = [*SMALL_TABLES, 'manifest.json', 'selection.json', 'headline.csv', 'tables.md', 'audit.json']
    (out / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-csv', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.out, args.reference_csv)
    print(json.dumps(dict(status=result['status'], runs=result['runs'], traces=result['traces'],
                          f1_fixed_reference=result['f1_fixed_reference'])), flush=True)


if __name__ == '__main__':
    main()
