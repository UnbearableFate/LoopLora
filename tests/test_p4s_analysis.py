"""Synthetic regression cases for validation-only WP-C3 reporting."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_p4s_stopping.py'
SPEC = importlib.util.spec_from_file_location('p4s_analysis', SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.fixture
def study(tmp_path):
    design = dict(families=dict(writer=['joint'], r=[64], variant=['M1']),
                  training_seeds=[17, 29, 43], splits=['val', 'test'],
                  hops=list(range(1, 9)), max_steps=16, minimum_steps=1,
                  fixed_references=[1, 2, 4, 8, 16],
                  read_norm_thresholds=[0, .5, 1],
                  relative_change_thresholds=[0, .5, 1],
                  output_stability_thresholds=[1, 2, 3],
                  halting_convention='predict at first post-update hit L>=1',
                  oracle_reference='label-informed true-hop reference')
    runs = [dict(name=f'joint64_{seed}', writer='joint', r=64, variant='M1',
                 family='joint/r64/M1', seed=seed, checkpoint=f'/checkpoint/{seed}.pt')
            for seed in design['training_seeds']]
    return tmp_path, runs, design


def trace(episodes=1):
    hops = np.broadcast_to(np.arange(1, 9), (episodes, 8)).copy()
    norms = np.full((episodes, 8, 17), 2.)
    norms[..., 0] = 1.
    norms[..., 2] = .5
    norms[..., 3:] = 0.
    change = norms.copy()
    change[..., 0] = np.nan
    return dict(predictions=np.ones((episodes, 8, 17), dtype=np.int64),
                read_norm=norms, relative_change=change,
                answer=np.ones((episodes, 8), dtype=np.int64), hop=hops,
                episode_id=np.broadcast_to(np.arange(episodes)[:, None], (episodes, 8)).copy(),
                certified_c=np.linspace(.2, 2., episodes))


def write_trace(root, run, split, data):
    folder = root / 'traces' / run['name']
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f'{split}.npz', **data)


def populate(study):
    root, runs, _ = study
    for run in runs:
        for split in ('val', 'test'):
            write_trace(root, run, split, trace())


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_first_post_update_hit_is_inclusive_and_falls_back_to_cap():
    data = trace()
    data['read_norm'][..., 0] = 0.  # This is never a permissible stopping decision.
    assert np.all(analysis.halting_steps(data, 'read_norm', .5) == 2)
    assert np.all(analysis.halting_steps(data, 'relative_change', 0) == 3)
    data['read_norm'][...] = 2.
    assert np.all(analysis.halting_steps(data, 'read_norm', .5) == 16)


def test_stability_counts_unchanged_pairs_and_resets_after_a_change():
    data = trace()
    data['predictions'][...] = np.array([0, 0, 1, 1, 1] + [2, 3] * 6)
    assert np.all(analysis.halting_steps(data, 'output_stability', 1) == 1)
    assert np.all(analysis.halting_steps(data, 'output_stability', 2) == 4)
    assert np.all(analysis.halting_steps(data, 'output_stability', 3) == 16)
    with pytest.raises(ValueError, match='positive integer'):
        analysis.halting_steps(data, 'output_stability', 1.5)


def test_selection_opens_only_validation_and_freezes_metadata(study, monkeypatch):
    root, runs, design = study
    populate(study)
    for run in runs:
        (root / 'traces' / run['name'] / 'test.npz').write_bytes(b'not an archive')
    real_load = analysis.np.load
    opened = []

    def guarded_load(path, *args, **kwargs):
        opened.append(Path(path))
        assert Path(path).name == 'val.npz'
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(analysis.np, 'load', guarded_load)
    selection = analysis.select_thresholds(root, runs, design)
    assert len(opened) == 3
    assert len(selection['selected']) == 3  # One family, not one per seed.
    assert all('seed' not in row for row in selection['selected'])
    frozen_design = copy.deepcopy(selection['metadata']['design'])
    design['training_seeds'].append(99)
    assert selection['metadata']['design'] == frozen_design


def test_selection_averages_seeds_equally_instead_of_pooling_examples(study):
    root, runs, design = study
    design['read_norm_thresholds'] = [0, 1]
    for index, run in enumerate(runs):
        # Seed 17 has eight times as many episodes but still only one seed's weight.
        data = trace(8 if index == 0 else 1)
        data['read_norm'][..., 1] = 1.
        data['read_norm'][..., 2:] = 0.
        data['predictions'][..., 1] = 1 if index == 0 else 0
        data['predictions'][..., 2] = 0 if index == 0 else 1
        write_trace(root, run, 'val', data)
        write_trace(root, run, 'test', data)
    selection = analysis.select_thresholds(root, runs, design)
    selected = next(r for r in selection['selected'] if r['criterion'] == 'read_norm')
    assert selected['threshold'] == 0
    assert selected['validation_accuracy'] == pytest.approx(2 / 3)
    assert selected['validation_mean_steps'] == 2
    analysis.summarize(root, runs, design, selection, root / 'out')
    rows = read_csv(root / 'out' / 'accuracy.csv')
    fixed = next(r for r in rows if r['method'] == 'fixed_1' and r['split'] == 'test'
                 and r['hop'] == 'overall')
    assert float(fixed['accuracy']) == pytest.approx(1 / 3)


def test_selection_prefers_accuracy_then_fewer_steps_then_lower_threshold(study):
    root, runs, design = study
    populate(study)
    selection = analysis.select_thresholds(root, runs, design)
    for row in selection['selected']:
        if row['criterion'] in ('read_norm', 'relative_change'):
            # All correct: .5 and 1 stop at update 2; 0 stops at update 3.
            assert row['threshold'] == .5
            assert row['validation_mean_steps'] == 2
        else:
            assert row['threshold'] == 1
    for run in runs:
        data = trace()
        data['predictions'][..., 2] = 0  # Fewer steps may not override accuracy.
        write_trace(root, run, 'val', data)
    chosen = analysis.select_thresholds(root, runs, design)['selected']
    assert next(r for r in chosen if r['criterion'] == 'read_norm')['threshold'] == 0


@pytest.mark.parametrize('mutation', ['design', 'seed_threshold', 'selected_score', 'missing_seed'])
def test_frozen_selection_rejects_scope_or_threshold_drift_before_test(study, monkeypatch, mutation):
    root, runs, design = study
    populate(study)
    selection = analysis.select_thresholds(root, runs, design)
    if mutation == 'design':
        design['read_norm_thresholds'].append(2)
    elif mutation == 'seed_threshold':
        selection['selected'][0]['seed'] = 17
    elif mutation == 'selected_score':
        selection['selected'][0]['threshold'] = 0
    else:
        runs.pop()

    def no_open(*args, **kwargs):
        pytest.fail('Trace opened before frozen selection validation')

    monkeypatch.setattr(analysis.np, 'load', no_open)
    with pytest.raises(ValueError):
        analysis.summarize(root, runs, design, selection, root / 'out')


def test_oracle_selects_true_hop_and_summary_uses_seed_means_and_episode_certificates(study):
    root, runs, design = study
    for index, run in enumerate(runs):
        data = trace(8 if index == 0 else 1)
        data['predictions'][...] = 0
        np.put_along_axis(data['predictions'], data['hop'][..., None], 1, axis=-1)
        data['certified_c'][...] = [.2, 1., 2.][index]
        for split in ('val', 'test'):
            write_trace(root, run, split, data)
    selection = analysis.select_thresholds(root, runs, design)
    out = root / 'out'
    result = analysis.summarize(root, runs, design, selection, out)
    rows = read_csv(out / 'accuracy.csv')
    assert tuple(rows[0]) == analysis.ACCURACY_COLUMNS
    oracle = next(r for r in rows if r['split'] == 'test' and r['method'] == 'oracle_true_hop'
                  and r['hop'] == 'overall')
    assert float(oracle['accuracy']) == 1
    assert float(oracle['mean_steps']) == 4.5
    assert int(oracle['n']) == 80
    for row in rows:
        if row['method'] == 'read_norm':
            assert float(row['mean_reads']) == float(row['mean_steps']) + 1
    c = next(r for r in read_csv(out / 'contraction.csv') if r['split'] == 'test')
    assert int(c['n']) == 10  # Not 80 query-level copies of the same certificates.
    assert int(c['c_lt_1_count']) == 8  # Boundary c=1 is not certified.
    assert float(c['c_lt_1_percent']) == 80
    assert float(c['c_min']) == .2
    assert float(c['c_max']) == 2
    assert result['latency']['available'] is False
    family = result['families'][0]
    assert set(family['screening_test']) == {'hop1_overshoot_recovery', 'long_hops_4to8', 'overall'}
    recovery = family['screening_test']['hop1_overshoot_recovery']
    for criterion in analysis.CRITERIA:
        assert recovery['adaptive'][criterion]['delta_fixed_4'] == pytest.approx(
            recovery['adaptive'][criterion]['accuracy'] - recovery['fixed_4']['accuracy'])
    assert json.loads((out / 'summary.json').read_text()) == result
    assert len(read_csv(out / 'accuracy_per_seed.csv')) == 3 * 2 * 9 * 9


def test_latency_averages_run_statistics_equally_and_preserves_details(study):
    root, runs, design = study
    populate(study)
    latency = []
    for index, run in enumerate(runs):
        latency.append({k: run[k] for k in ('name', 'writer', 'r', 'variant', 'family', 'seed')} |
                       dict(method='fixed_1', batch_size=64, warmup=5, repeats=20,
                            mean_steps=1, mean_reads=1,
                            **{f'{clock}_ms_{stat}': index + 1 for clock in ('cuda', 'wall')
                               for stat in ('mean', 'median', 'p95')}))
    (root / 'latency.json').write_text(json.dumps(latency))
    selection = analysis.select_thresholds(root, runs, design)
    result = analysis.summarize(root, runs, design, selection, root / 'out')
    assert result['latency']['available'] is True
    pooled = read_csv(root / 'out' / 'latency.csv')
    assert len(pooled) == 1 and int(pooled[0]['seeds']) == 3
    assert float(pooled[0]['cuda_ms_mean']) == 2
    assert len(read_csv(root / 'out' / 'latency_per_seed.csv')) == 3


def test_latency_single_query_hops_pool_equally_and_count_unique_seeds(study):
    root, runs, _ = study
    rows = []
    for index, run in enumerate(runs):
        for hop in range(1, 9):
            rows.append({k: run[k] for k in ('name', 'writer', 'r', 'variant', 'family', 'seed')} |
                        dict(method='read_norm', batch_size=1, latency_hop=hop,
                             warmup=5, repeats=20, mean_steps=hop, mean_reads=hop + 1,
                             **{f'{clock}_ms_{stat}': index * 10 + hop
                                for clock in ('cuda', 'wall') for stat in ('mean', 'median', 'p95')}))
    (root / 'latency.json').write_text(json.dumps(rows))
    out = root / 'out'
    out.mkdir()
    result = analysis._latency(root, runs, out)
    assert result['per_run_rows'] == 24 and result['family_rows'] == 1
    pooled = read_csv(out / 'latency.csv')[0]
    assert int(pooled['seeds']) == 3
    assert float(pooled['mean_steps']) == 4.5
    assert float(pooled['mean_reads']) == 5.5
    assert float(pooled['cuda_ms_mean']) == 14.5
    assert len(read_csv(out / 'latency_per_seed.csv')) == 24
    # A missing hop for only one seed cannot silently change the seed weighting.
    (root / 'latency.json').write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match='matching query hops'):
        analysis._latency(root, runs, out)
    # Even a matching subset across seeds fails the explicit eight-hop protocol.
    subset = [row for row in rows if row['latency_hop'] != 8]
    (root / 'latency.json').write_text(json.dumps(subset))
    with pytest.raises(ValueError, match='all eight query hops'):
        analysis._latency(root, runs, out)


def test_hop_overall_is_balanced_even_if_trace_columns_are_permuted(study):
    root, runs, design = study
    for run in runs:
        data = trace(2)
        for key in ('predictions', 'read_norm', 'relative_change', 'answer', 'hop', 'episode_id'):
            data[key][1] = data[key][1, ::-1].copy()
        data['predictions'][data['hop'] == 1] = 0
        write_trace(root, run, 'val', data)
    selection = analysis.select_thresholds(root, runs, design)
    assert all(row['validation_accuracy'] == 7 / 8 for row in selection['selected'])


def test_duplicate_seed_or_missing_hop_is_rejected(study):
    root, runs, design = study
    populate(study)
    runs[1]['seed'] = runs[0]['seed']
    with pytest.raises(ValueError, match='each training seed'):
        analysis.select_thresholds(root, runs, design)
    runs[1]['seed'] = 29
    data = trace()
    data['hop'][0, 0] = 2
    write_trace(root, runs[0], 'val', data)
    with pytest.raises(ValueError, match='each locked hop'):
        analysis.select_thresholds(root, runs, design)
