"""Synthetic learned-halting selection, paired inference and reporting checks."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
_SPEC = importlib.util.spec_from_file_location('analyze_p4s_learned_test', SCRIPTS / 'analyze_p4s_learned.py')
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def trace(episodes=2, offset=0):
    shape = (episodes, 8)
    prediction = np.zeros((*shape, 17), dtype=np.int64)
    # State zero is correct at hop1 only; the any-step oracle must exclude it.
    prediction[:, 0, 0] = 1
    prediction[:, 1:, 2:] = 1
    prediction[:, 5:, 1] = 1
    probability = np.full((*shape, 16), .1)
    probability[..., 1] = .8
    return dict(predictions=prediction, answer=np.ones(shape, dtype=np.int64),
                episode_id=np.broadcast_to((np.arange(episodes) + offset)[:, None], shape).copy(),
                hop=np.broadcast_to(np.arange(1, 9), shape).copy(),
                probabilities=probability, read_norm=np.ones((*shape, 17)),
                relative_change=np.ones((*shape, 17)))


def fixture(tmp_path):
    runs = [dict(name=f'{w}-{v}-{s}', writer=w, variant=v, r=32, seed=s,
                 family=f'{w}/r32/{v}', checkpoint=f'/released/{w}-{v}-{s}.pt')
            for w in analysis.WRITERS for v in analysis.VARIANTS for s in analysis.SEEDS]
    references = [dict(family=f, criterion=c, threshold=1 if c == 'output_stability' else 0,
                       validation_accuracy=.1, validation_mean_steps=16)
                  for f in sorted({r['family'] for r in runs}) for c in analysis.CRITERIA]
    design = dict(theta_grid=list(analysis.THETAS), runs=copy.deepcopy(runs),
                  validation=dict(split='val', generator_seed=4242, episodes=2),
                  confirmation=dict(split='test', generator_seed=999002, episodes=2),
                  bootstrap=dict(resamples=2000, confidence=analysis.CONFIDENCE, seed=2718),
                  label_free_thresholds=references,
                  best_label_free={f: next(r for r in references if r['family'] == f and r['criterion'] == 'read_norm')
                                   for f in {r['family'] for r in runs}})
    for run in runs:
        folder = tmp_path / 'traces' / run['name']
        folder.mkdir(parents=True)
        for split, offset in (('val', 0), ('test', 100)):
            data = trace(offset=offset)
            if split == 'test':
                chosen, executed = analysis.learned_steps(data['probabilities'], .15)
                data.update(selected_steps=chosen, executed_steps=executed, read_calls=executed + 1)
            np.savez(folder / f'{split}.npz', **data)
    return tmp_path, runs, design


def test_first_inclusive_hit_and_earliest_fallback_argmax_have_distinct_execution_cost():
    probabilities = np.full((1, 8, 16), .1)
    probabilities[0, 0, [2, 4]] = .5
    probabilities[0, 1:, [3]] = .4
    chosen, executed = analysis.learned_steps(probabilities, .5)
    assert chosen.tolist() == [[3, 4, 4, 4, 4, 4, 4, 4]]
    assert executed.tolist() == [[3, 16, 16, 16, 16, 16, 16, 16]]
    assert (executed + 1).tolist() == [[4, 17, 17, 17, 17, 17, 17, 17]]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1])
def test_nonfinite_or_invalid_probabilities_fail(value):
    probabilities = np.full((1, 8, 16), .1)
    probabilities[0, 0, 0] = value
    with pytest.raises(ValueError, match='probabilities'):
        analysis.learned_steps(probabilities, .5)


def test_theta_selection_uses_validation_only_and_ties_prefer_selected_steps_then_theta(tmp_path, monkeypatch):
    root, runs, design = fixture(tmp_path)
    original = analysis.load_trace

    def validation_only(root, run, split, design):
        assert split == 'val'
        return original(root, run, split, design)

    monkeypatch.setattr(analysis, 'load_trace', validation_only)
    result = analysis.select_theta(root, runs, design)
    assert len(result['selected']) == 8 and len(result['grid']) == 152
    assert all(row['theta'] == .15 for row in result['selected'])
    assert all(row['validation_mean_steps'] == row['validation_mean_executed_steps'] == 2 for row in result['selected'])
    # .85 has the same chosen state/accuracy but executes to16; the registered
    # tie rule uses chosen steps, then theta, not executed steps.
    for family in {r['family'] for r in runs}:
        candidate = next(r for r in result['grid'] if r['family'] == family and r['theta'] == .85)
        assert candidate['validation_mean_steps'] == 2
        assert candidate['validation_mean_executed_steps'] == 16
    assert len(analysis.validate_theta(result, runs, design)) == 8


def test_theta_validation_rejects_grid_or_design_drift_without_trace_io(tmp_path, monkeypatch):
    root, runs, design = fixture(tmp_path)
    result = analysis.select_theta(root, runs, design)
    monkeypatch.setattr(analysis, 'load_trace', lambda *args: pytest.fail('lock validation opened a trace'))
    invalid = copy.deepcopy(result)
    invalid['selected'][0]['theta'] = .95
    with pytest.raises(ValueError, match='Theta selection'):
        analysis.validate_theta(invalid, runs, design)
    invalid = copy.deepcopy(result)
    invalid['grid'].pop()
    with pytest.raises(ValueError, match='Incomplete theta grid'):
        analysis.validate_theta(invalid, runs, design)
    with pytest.raises(ValueError, match='different design'):
        analysis.validate_theta(result, runs, dict(design, extra='drift'))


def test_validation_selection_requires_identical_episode_labels_across_seeds(tmp_path):
    root, runs, design = fixture(tmp_path)
    run = next(r for r in runs if r['writer'] == 'joint_fact' and r['variant'] == 'M1' and r['seed'] == 29)
    path = root / 'traces' / run['name'] / 'val.npz'
    data = trace()
    data['answer'][0, 0] = 99
    np.savez(path, **data)
    with pytest.raises(ValueError, match='Validation.*labels differ'):
        analysis.select_theta(root, runs, design)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'wrong_family'])
def test_manifest_requires_all_24_reader_head_identities(tmp_path, mutation):
    _, runs, _ = fixture(tmp_path)
    if mutation == 'duplicate':
        runs[-1] = runs[0]
    elif mutation == 'missing':
        runs.pop()
    else:
        runs[0]['family'] = 'joint/r32/M1'
    with pytest.raises(ValueError):
        analysis._runs(runs)


def test_best_label_free_is_locked_validation_choice_not_confirmation_choice(tmp_path):
    _, runs, design = fixture(tmp_path)
    _, best = analysis._references(design, runs)
    assert all(row['criterion'] == 'read_norm' for row in best.values())
    altered = copy.deepcopy(design)
    first = next(iter(altered['best_label_free']))
    altered['best_label_free'][first] = dict(altered['best_label_free'][first], criterion='relative_change')
    with pytest.raises(ValueError, match='Best label-free'):
        analysis._references(altered, runs)


def test_adjusted_endpoint_statistics_preserve_joint_seed_and_subset_covariance():
    traces = {s: trace(episodes=4) for s in analysis.SEEDS}
    shared = np.broadcast_to(np.asarray([1, 0, 1, 0])[:, None], (4, 8)).astype(bool)
    learned = {17: shared, 29: ~shared, 43: np.zeros_like(shared)}
    reference = {17: np.zeros_like(shared), 29: np.ones_like(shared), 43: np.zeros_like(shared)}
    result = analysis.endpoint_statistics(traces, learned, reference)
    assert set(result) == {'hop1', 'reach', 'far', 'overall', 'specificity'}
    assert all(row['confidence'] == 1 - .05 / 6 and row['resamples'] == 2000 for row in result.values())
    assert all(row['difference'] == row['ci_low'] == row['ci_high'] == 0 for row in result.values())


def test_endpoint_statistics_reject_different_episode_query_labels_across_seeds():
    traces = {s: trace() for s in analysis.SEEDS}
    traces[29]['answer'][0, 0] = 9
    flags = {s: np.zeros((2, 8), dtype=bool) for s in analysis.SEEDS}
    with pytest.raises(ValueError, match='different answer labels'):
        analysis.endpoint_statistics(traces, flags, flags)


@pytest.mark.parametrize('hop1,reach,reach_low,specificity_low,primary,expected', [
    (-.01, .03, .001, .001, True, True),
    (-.01001, .03, .001, .001, True, False),
    (0, .02999, .001, .001, True, False),
    (0, .03, 0, .001, True, False),
    (0, .03, .001, 0, True, False),
    (0, .03, .001, .001, False, False),
])
def test_exact_registered_endpoint_thresholds_and_joint_descriptive_status(hop1, reach, reach_low, specificity_low, primary, expected):
    stats = dict(hop1=dict(difference=hop1), reach=dict(difference=reach, ci_low=reach_low),
                 specificity=dict(ci_low=specificity_low))
    assert analysis.qualification(stats, primary)['qualifies'] is expected


def latency_rows(runs):
    return [dict(name=r['name'], family=r['family'], writer=r['writer'], variant=r['variant'], seed=r['seed'], r=32,
                 method=m, batch_size=64, warmup=5, repeats=20,
                 mean_steps=2 if m == 'learned' else int(m.split('_')[1]),
                 mean_executed_steps=2 if m == 'learned' else int(m.split('_')[1]),
                 mean_reads=3 if m == 'learned' else int(m.split('_')[1]),
                 audit_prediction_steps_exact=True, audit_state_rel_l2_max=0, audit_state_abs_max=0,
                 **{f'{clock}_ms_{stat}': 1 for clock in ('cuda', 'wall') for stat in ('mean', 'median', 'p95')})
            for r in runs for m in ('learned', 'fixed_4', 'fixed_16')]


def test_synthetic_full_report_counts_oracle_excludes_state_zero_and_joint_is_descriptive(tmp_path, monkeypatch):
    root, runs, design = fixture(tmp_path)
    # The real descriptive bootstrap must run, not a mocked result: force the
    # locked read-norm reference to choose L1, with exactly 4/8 less accuracy
    # than learned L2 in this fixture.
    for row in design['label_free_thresholds']:
        if row['criterion'] == 'read_norm':
            row['threshold'] = 1
    selection = analysis.select_theta(root, runs, design)
    manifest = dict(runs=runs, design=design, design_hash=analysis.stable_hash(design),
                    design_lock_commit='design', theta_lock_commit='theta', theta_blake2b='0' * 32)
    monkeypatch.setattr(analysis, 'audit_locks', lambda root: (manifest, design, selection))
    (root / 'latency.json').write_text(json.dumps(latency_rows(runs)))
    result = analysis.analyze(root, root / 'out')
    assert result['runs'] == 24 and result['families'] == 8
    assert result['confidence'] == 1 - .05 / 6
    assert result['qualified_families'] == sorted(f'{w}/r32/{v}' for w in analysis.WRITERS if w != 'joint' for v in analysis.VARIANTS)
    import csv
    with (root / 'out/accuracy.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    oracle_hop1 = [r for r in rows if r['method'] == 'oracle_any_step' and r['hop'] == '1']
    assert all(float(row['accuracy']) == 0 for row in oracle_hop1)
    assert all(row['mean_executed_steps'] == '' for row in oracle_hop1)
    learned = [r for r in rows if r['method'] == 'learned']
    assert all(float(row['mean_selected_steps']) == float(row['mean_executed_steps']) == 2 for row in learned)
    assert all(float(row['mean_read_calls']) == 3 for row in learned)
    endpoints = json.loads((root / 'out/endpoints.json').read_text())
    assert all(not row['qualifies'] and row['endpoints_satisfied'] for row in endpoints if row['writer'] == 'joint')
    assert all(row['descriptive']['any_step_oracle_minus_learned']['overall']['difference'] == 0 for row in endpoints)
    assert all(row['descriptive']['learned_minus_best_label_free']['overall']['difference'] == .5 for row in endpoints)
    assert all(row['descriptive']['learned_minus_best_label_free']['overall']['comparison'] == 'learned - read_norm' for row in endpoints)
    assert all(row['descriptive']['learned_minus_best_label_free']['overall']['confidence'] == 1 - .05 / 6 for row in endpoints)
    assert (root / 'out/tables.md').read_text().count('## ') == 8


def test_missing_lock_blocks_analysis_before_any_validation_or_confirmation_read(tmp_path, monkeypatch):
    root, runs, design = fixture(tmp_path)
    (root / 'manifest.json').write_text(json.dumps(dict(status='running', smoke=False, runs=runs, design=design)))
    monkeypatch.setattr(analysis, 'load_trace', lambda *args: pytest.fail('trace opened before lock check'))
    with pytest.raises(ValueError, match='completed main'):
        analysis.analyze(root, root / 'out')
    assert not (root / 'out').exists()


def test_theta_git_lock_mismatch_blocks_every_trace_read(tmp_path, monkeypatch):
    root, runs, design = fixture(tmp_path)
    selection = analysis.select_theta(root, runs, design)
    design['validation']['episodes'] = design['confirmation']['episodes'] = 2000
    theta = root / 'theta.json'
    theta.write_text(json.dumps(selection))
    manifest = dict(status='complete', smoke=False, checkpoints_unchanged=True, runs=runs, design=design,
                    design_hash=analysis.stable_hash(design), theta_path=str(theta.resolve()),
                    theta_blake2b=analysis.digest_checkpoint(theta), theta_lock_commit='changed')
    (root / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(analysis, '_git_locked_file', lambda *args: b'changed contents')
    monkeypatch.setattr(analysis, 'load_trace', lambda *args: pytest.fail('trace opened before Git lock check'))
    with pytest.raises(ValueError, match='Theta file differs from its Git lock'):
        analysis.analyze(root, root / 'out')
    assert not (root / 'out').exists()


def test_saved_execution_counters_cannot_claim_fewer_updates_than_policy(tmp_path, monkeypatch):
    root, runs, design = fixture(tmp_path)
    selection = analysis.select_theta(root, runs, design)
    manifest = dict(runs=runs, design=design)
    monkeypatch.setattr(analysis, 'audit_locks', lambda root: (manifest, design, selection))
    run = next(r for r in runs if r['family'] == 'joint/r32/M1' and r['seed'] == 17)
    path = root / 'traces' / run['name'] / 'test.npz'
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data['executed_steps'][0, 0] = 1
    np.savez(path, **data)
    with pytest.raises(ValueError, match='executed_steps'):
        analysis.analyze(root, root / 'out')


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'bad_time', 'bad_batch', 'bad_counter', 'bad_audit'])
def test_latency_requires_complete_registered_72_row_matrix(tmp_path, failure):
    root, runs, _ = fixture(tmp_path)
    rows = latency_rows(runs)
    if failure == 'missing':
        rows.pop()
    elif failure == 'duplicate':
        rows[-1] = rows[0]
    elif failure == 'bad_time':
        rows[0]['cuda_ms_mean'] = float('nan')
    elif failure == 'bad_batch':
        rows[0]['batch_size'] = 1
    elif failure == 'bad_counter':
        rows[0]['mean_reads'] = 2
    else:
        rows[0]['audit_prediction_steps_exact'] = False
    (root / 'latency.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        analysis.analyze_latency(root, runs)
