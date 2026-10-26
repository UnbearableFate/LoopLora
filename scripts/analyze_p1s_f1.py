#!/usr/bin/env python3
"""WP-F1 analysis using the tested WP-H evaluator audits and paired inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import analyze_p1s_main as shared
from looplora.infra.seeds import stable_hash

WRITERS = ('oracle_random', 'joint')
RANKS = (64, 128)
READERS = ('M1', 'M1-lin')
ENDPOINTS = ('reach', 'far', 'specificity', 'hop1', 'vs_fixed', 'vs_loop')


def evaluate_f1_gates(endpoints, *, evaluation_seed):
    """F1 has seven predicates and no replication requirement; intervals are raw."""
    from looplora.sym.gates import _validate_endpoint
    expected = {(w, r, p) for w in WRITERS for r in RANKS for p in READERS}
    shared.require(evaluation_seed == 777001, 'F1 gates require confirmation seed 777001')
    shared.require(set(endpoints) == expected, 'F1 gates require exactly eight writer/rank/reader rows')
    rows = []
    for writer, rank, reader in sorted(expected):
        stats = endpoints[writer, rank, reader]
        shared.require(set(stats) == set(ENDPOINTS), 'F1 requires exactly six endpoints')
        for name, pair in stats.items():
            _validate_endpoint(name, pair)
        values = [('reach_at_least_3pp', stats['reach']['gate']['difference'], .03, True),
                  ('reach_ci_positive', stats['reach']['gate']['ci_low'], 0., False),
                  ('reach_each_seed_positive', min(s['difference'] for s in stats['reach']['gate']['per_seed']), 0., False),
                  ('specificity_ci_positive', stats['specificity']['gate']['ci_low'], 0., False),
                  ('hop1_degradation_at_most_1pp', stats['hop1']['gate']['difference'], -.01, True),
                  ('vs_fixed_ci_positive', stats['vs_fixed']['gate']['ci_low'], 0., False),
                  ('vs_loop_ci_positive', stats['vs_loop']['gate']['ci_low'], 0., False)]
        criteria = {name: dict(value=value, threshold=threshold, operator='>=' if inclusive else '>',
                              passed=bool(value >= threshold if inclusive else value > threshold))
                    for name, value, threshold, inclusive in values}
        g1 = all(criteria[name]['passed'] for name, *_ in values[:5])
        g2 = all(criteria[name]['passed'] for name, *_ in values[5:])
        rows.append(dict(writer=writer, r=rank, reader=reader,
                         fixed_query_comparator='B5' if reader == 'M1' else 'B5-lin',
                         evaluation_seed=evaluation_seed, gate_confidence=.975, report_confidence=.95,
                         replication_required=False, weaker_than_P1S_main=True,
                         criteria=criteria, g1_passed=g1, g2_passed=g2, passed=g1 and g2,
                         statistics=stats))
    return rows


def verify_release(root):
    plan = shared.read_json(root / 'evaluation_plan.json')
    barrier = shared.read_json(root / 'barrier.json')
    expected = {(w, r, v, s) for w in WRITERS for r in RANKS
                for v in ('M1', 'M1-lin', 'B5', 'B5-lin', 'B3') for s in (17, 29, 43)}
    actual = {(r['writer'], r['r'], r['variant'], r['seed']) for r in plan['runs']}
    shared.require(len(plan['runs']) == 60 and actual == expected and plan['steps'] == 20000,
                   'F1 released matrix differs')
    shared.require(barrier['status'] == 'passed' and barrier['plan_hash'] == stable_hash(plan),
                   'F1 global release barrier has not passed')
    shared.require(set(barrier['runs']) == {r['name'] for r in plan['runs']}
                   and len(barrier['runs']) == 60, 'F1 barrier does not cover every run')
    paths = []
    for run in plan['runs']:
        folder = Path(run['run_dir'])
        record = shared.read_json(folder / 'record.json')
        original = shared.read_json(Path(run['training_dir']) / 'record.json')
        config = shared.read_json(Path(run['training_dir']) / 'config.json')['config']
        shared.require(record['status'] == original['status'] == 'complete'
                       and record['training_complete'] and record['confirmation_samples_written'],
                       'F1 training/evaluation incomplete')
        shared.require(config == run['config'] == record['config']
                       and stable_hash(config) == original['config_hash'] == record['config_hash']
                       and config['train']['steps'] == 20000 and config['train']['batch_size'] == 64,
                       'F1 training configuration differs')
        shared.require((Path(run['training_dir']) / 'checkpoint.pt').is_file(), 'F1 checkpoint missing')
        meta = shared.read_json(folder / 'confirmation/evaluation.json')
        settings = meta['config']['eval']
        shared.require(meta['aggregation_deferred'] and 'table' not in meta and 'diagnostics' not in meta
                       and meta['checkpoint_config'] == config and settings['seed'] == 777001
                       and settings['episodes'] == 2000 and settings['splits'] == ['test']
                       and not settings['counterfactuals']
                       and settings['depths'] == list(shared.DEPTHS) and settings['hops'] == list(shared.HOPS),
                       'F1 confirmation scope or embargo differs')
        path = folder / 'confirmation/samples.npz'
        shared.require(path.is_file() and not (folder / 'confirmation/metrics.json').exists(),
                       'F1 confirmation missing or already finalized')
        paths.append(str(path))
    shared.require(len(barrier['confirmation_samples']) == 60 and set(paths) == set(barrier['confirmation_samples']),
                   'F1 barrier does not cover all confirmation files')
    return plan, barrier


def statistics(plan, cache):
    endpoints, reported = {}, []
    def select(writer, rank, variant, dataset):
        return [r for r in plan['runs'] if r['writer'] == writer and r['r'] == rank
                and r['variant'] == variant and (dataset != 'screening' or r['seed'] == 17)]
    for dataset in ('confirmation', 'screening', 'unseen'):
        for writer in WRITERS:
            for rank in RANKS:
                for reader in READERS:
                    p = select(writer, rank, reader, dataset)
                    fixed = select(writer, rank, 'B5' if reader == 'M1' else 'B5-lin', dataset)
                    loop = select(writer, rank, 'B3', dataset)
                    pairs = {
                        'reach': shared.paired(cache, p, p, dataset, 4, 1, (2, 3)),
                        'far': shared.paired(cache, p, p, dataset, 4, 1, (6, 7, 8)),
                        'specificity': shared.paired(cache, p, p, dataset, 4, 1, (2, 3), specificity=True),
                        'hop1': shared.paired(cache, p, p, dataset, 4, 1, (1,)),
                        'vs_fixed': shared.paired(cache, p, fixed, dataset, 4, 4, (2, 3)),
                        'vs_loop': shared.paired(cache, p, loop, dataset, 4, 4, (2, 3)),
                    }
                    if dataset == 'confirmation':
                        endpoints[writer, rank, reader] = pairs
                    else:
                        reported.extend(dict(writer=writer, r=rank, reader=reader, dataset=dataset,
                                             endpoint=name, statistics=pair, gating=False)
                                        for name, pair in pairs.items())
                    reported.append(dict(writer=writer, r=rank, reader=reader, dataset=dataset,
                                         endpoint='H6_L8_minus_L4_hops4to8', gating=False,
                                         statistics=shared.paired(cache, p, p, dataset, 8, 4, (4, 5, 6, 7, 8))))
                reported.append(dict(writer=writer, r=rank, reader='M1', dataset=dataset,
                                     endpoint='M1_minus_M1lin_L4_reach', gating=False,
                                     statistics=shared.paired(cache, select(writer, rank, 'M1', dataset),
                                     select(writer, rank, 'M1-lin', dataset), dataset, 4, 4, (2, 3))))
    return evaluate_f1_gates(endpoints, evaluation_seed=777001), reported


def tables(out, accuracy, gates, reported):
    for name, rows in (('accuracy_tables', accuracy), ('gates', gates), ('reported_contrasts', reported)):
        shared.write_json(out / f'{name}.json', rows)
    shared.write_csv(out / 'accuracy_tables.csv', accuracy)
    shared.write_csv(out / 'reported_contrasts.csv', [
        {k: row[k] for k in ('writer', 'r', 'reader', 'dataset', 'endpoint')} |
        {'confidence': stat['confidence'], 'difference': stat['difference'],
         'ci_low': stat['ci_low'], 'ci_high': stat['ci_high'], 'per_seed': json.dumps(stat['per_seed'])}
        for row in reported for stat in row['statistics'].values()])
    lines = ['# F1 confirmation gates', '', 'Differences and intervals are percentage points. '
             'Decisions use unrounded 97.5% intervals. F1 omits replication and is a weaker design.', '',
             shared._table(['Writer', 'r', 'Reader', 'G1', 'G2', 'Pass'],
                           [[r['writer'], r['r'], r['reader'], r['g1_passed'], r['g2_passed'], r['passed']] for r in gates])]
    for row in gates:
        lines += [f"## {row['writer']} / r={row['r']} / {row['reader']}", '',
                  shared._table(['Endpoint', 'Estimate [95% CI]', 'Estimate [97.5% CI]', 'Per-seed effect (pp)'],
                                [[name, shared._interval(pair['report']), shared._interval(pair['gate']),
                                  ', '.join(f"{s['seed']}: {100*s['difference']:+.2f}" for s in pair['gate']['per_seed'])]
                                 for name, pair in row['statistics'].items()]),
                  shared._table(['Criterion', 'Raw value', 'Threshold', 'Pass'],
                                [[name, c['value'], f"{c['operator']} {c['threshold']}", c['passed']]
                                 for name, c in row['criteria'].items()])]
    (out / 'gates.md').write_text('\n'.join(lines))
    original = {(r['name'], r['dataset'], r['split'], r['hop'], r['L']): r
                for r in accuracy if r['condition'] == 'original'}
    cf = []
    for row in accuracy:
        if row['condition'] != 'original':
            baseline = original[row['name'], row['dataset'], row['split'], row['hop'], row['L']]
            cf.append(row | dict(original_accuracy=baseline['accuracy'], gating=False,
                                 accuracy_change_from_original=row['accuracy']-baseline['accuracy'],
                                 accuracy_minus_chance=row['accuracy']-row['chance_accuracy']))
    shared.write_json(out / 'counterfactual_consistency.json', cf)
    shared.write_csv(out / 'counterfactual_consistency.csv', cf)


def trend(out, gates, reference=None):
    rows = []
    for gate in gates:
        row = dict(writer=gate['writer'], r=gate['r'], reader=gate['reader'], status='complete',
                   dataset='confirmation', evaluation_seed=777001)
        for name in ('reach', 'far', 'hop1'):
            stat = gate['statistics'][name]['gate']
            row.update({name: stat['difference'], name+'_ci_low': stat['ci_low'], name+'_ci_high': stat['ci_high']})
        rows.append(row)
    if reference and reference.is_file():
        for gate in json.loads(reference.read_text()):
            if gate['writer'] in (*WRITERS, 'oracle_orth'):
                row = dict(writer=gate['writer'], r=32, reader=gate['reader'], status='complete',
                           dataset='WP-H confirmation', evaluation_seed=777001)
                for name in ('reach', 'far', 'hop1'):
                    stat = gate['statistics'][name]['gate']
                    row.update({name: stat['difference'], name+'_ci_low': stat['ci_low'], name+'_ci_high': stat['ci_high']})
                rows.append(row)
    else:
        rows += [dict(writer=w, r=32, reader=p, status='pending WP-H on master',
                      dataset='WP-H confirmation', reach=None, far=None, hop1=None)
                 for w in (*WRITERS, 'oracle_orth') for p in READERS]
    shared.write_json(out / 'rank_trend.json', rows)
    shared.write_csv(out / 'rank_trend.csv', rows)
    (out / 'rank_trend.md').write_text('# F1 rank trends\n\nAll values are percentage points. '
        'Missing r=32 WP-H references and the orthogonal ceiling remain explicit placeholders.\n\n' +
        shared._table(['Writer', 'r', 'Reader', 'Reach', 'Far', 'Hop-1 change', 'Status'],
                      [[r['writer'], r['r'], r['reader'],
                        *(f"{100*r[n]:+.2f}" if r.get(n) is not None else 'pending' for n in ('reach', 'far', 'hop1')),
                        r['status']] for r in rows]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for i, writer in enumerate(WRITERS):
        for j, name in enumerate(('reach', 'far', 'hop1')):
            ax = axes[i, j]
            for reader in READERS:
                points = sorted((r for r in rows if r['writer'] == writer and r['reader'] == reader
                                 and r.get(name) is not None), key=lambda r: r['r'])
                values = [100*r[name] for r in points]
                ax.errorbar([r['r'] for r in points], values,
                            yerr=[[100*(r[name]-r[name+'_ci_low']) for r in points],
                                  [100*(r[name+'_ci_high']-r[name]) for r in points]], marker='o', label=reader)
            ax.axhline(0, color='grey', linewidth=.7)
            ax.set_title(f'{writer}: {name}')
            ax.set_xticks([32, 64, 128]); ax.set_ylabel('Difference (pp)'); ax.set_xlabel('Rank')
            ax.legend()
    fig.suptitle('F1 confirmation rank trends; 97.5% CIs. r=32 references pending WP-H.'
                 if not reference else 'F1 and WP-H confirmation rank trends; 97.5% CIs')
    fig.tight_layout()
    fig.savefig(out / 'rank_trend.png', dpi=170)
    fig.savefig(out / 'rank_trend.pdf')
    plt.close(fig)


def analyze(root, out):
    plan, barrier = verify_release(root)  # Metadata only, before opening any confirmation archive.
    shared.require(not (out / 'analysis_started.json').exists(), 'F1 analysis already started')
    shared.write_json(out / 'analysis_started.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
                      plan_hash=stable_hash(plan), barrier_created_at=barrier['created_at'],
                      confirmation_metrics_previously_read=False))
    cache = shared.CompactCache(root / 'analysis_cache', limit=12)
    accuracy, audits, successors, pooled = [], [], [], []
    for rank in RANKS:
        rank_out = out / f'r{rank}'
        rank_out.mkdir(parents=True, exist_ok=True)
        subset = dict(plan, runs=[r for r in plan['runs'] if r['r'] == rank])
        table, logs, audit = shared.collect(subset, barrier, rank_out, cache)
        accuracy.extend(table)
        audits.append(dict(r=rank, **audit))
        successors += [dict(r=rank, **r) for r in json.loads((rank_out / 'swap_successors_per_seed.json').read_text())]
        pooled += [dict(r=rank, **r) for r in json.loads((rank_out / 'swap_successors.json').read_text())]
    shared.write_json(out / 'artifact_audit.json', dict(status='passed', runs=60,
                      sample_rows=sum(a['sample_rows'] for a in audits), ranks=audits))
    for name, rows in (('swap_successors_per_seed', successors), ('swap_successors', pooled)):
        shared.write_json(out / f'{name}.json', rows)
        shared.write_csv(out / f'{name}.csv', rows)
    gates, reported = statistics(plan, cache)
    tables(out, accuracy, gates, reported)
    trend(out, gates)
    from collect_registry import write_registry
    write_registry([Path(r['run_dir']) / 'record.json' for r in plan['runs']], out / 'experiment_registry.csv')
    shared.write_json(out / 'analysis_summary.json', dict(status='complete', runs=60, gate_rows=8,
                      confirmation_sets=60, plan_hash=stable_hash(plan),
                      passing_rows=[dict(writer=r['writer'], r=r['r'], reader=r['reader']) for r in gates if r['passed']],
                      confirmation_look_count=1, replication_required=False,
                      training_commit=plan['training_commit'], shared_pipeline='b900d92',
                      completed_at=datetime.now(timezone.utc).isoformat()))
    print('F1 analysis complete: 60 runs, 8 gate rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.out)
