"""F1-specific gate boundaries and global release protection, synthetic data only."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze_p1s_f1 as f1
import run_p1s_f1 as driver


def endpoint_table():
    spec = importlib.util.spec_from_file_location('main_gate_fixtures', ROOT / 'tests/test_sym_gates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = module._table()
    return {(w, r, p): {name: copy.deepcopy(main[w, p][name]) for name in f1.ENDPOINTS}
            for w in f1.WRITERS for r in f1.RANKS for p in f1.READERS}


def set_point(table, endpoint, value):
    for row in table.values():
        for role in ('report', 'gate'):
            row[endpoint][role]['difference'] = value
            for seed in row[endpoint][role]['per_seed']:
                seed['difference'] = value


def test_f1_gates_have_no_replication_and_preserve_inclusive_point_boundaries():
    table = endpoint_table()
    set_point(table, 'reach', .03)
    set_point(table, 'hop1', -.01)
    before = copy.deepcopy(table)
    rows = f1.evaluate_f1_gates(table, evaluation_seed=777001)
    assert len(rows) == 8 and all(row['passed'] for row in rows)
    assert all(len(row['criteria']) == 7 and row['replication_required'] is False for row in rows)
    assert table == before
    for row in rows:
        assert row['fixed_query_comparator'] == ('B5' if row['reader'] == 'M1' else 'B5-lin')


@pytest.mark.parametrize('endpoint', ['reach', 'specificity', 'vs_fixed', 'vs_loop'])
def test_one_zero_gate_ci_fails_only_its_writer_rank_reader(endpoint):
    table = endpoint_table()
    table['joint', 128, 'M1-lin'][endpoint]['gate']['ci_low'] = 0.
    rows = f1.evaluate_f1_gates(table, evaluation_seed=777001)
    assert {(r['writer'], r['r'], r['reader']) for r in rows if not r['passed']} == {('joint', 128, 'M1-lin')}


@pytest.mark.parametrize('endpoint,value', [('reach', .029999), ('hop1', -.010001)])
def test_f1_point_threshold_failures(endpoint, value):
    table = endpoint_table()
    set_point(table, endpoint, value)
    assert not any(row['g1_passed'] for row in f1.evaluate_f1_gates(table, evaluation_seed=777001))


def test_a_zero_reach_seed_fails_despite_positive_pooled_effect():
    table = endpoint_table()
    for row in table.values():
        for role in ('report', 'gate'):
            row['reach'][role]['per_seed'] = [{'seed': 17, 'difference': 0.},
                                             {'seed': 29, 'difference': .06},
                                             {'seed': 43, 'difference': .06}]
    assert not any(row['passed'] for row in f1.evaluate_f1_gates(table, evaluation_seed=777001))


def test_missing_rank_or_wrong_confirmation_seed_is_rejected():
    table = endpoint_table()
    with pytest.raises(ValueError, match='confirmation'):
        f1.evaluate_f1_gates(table, evaluation_seed=4242)
    table.pop(('joint', 128, 'M1'))
    with pytest.raises(ValueError, match='eight'):
        f1.evaluate_f1_gates(table, evaluation_seed=777001)


def test_global_training_barrier_rejects_partial_matrix_before_weight_loading(tmp_path, monkeypatch):
    reports = tmp_path / 'reports'
    reports.mkdir()
    (reports / 'manifest.json').write_text('{"names": []}')
    monkeypatch.setattr(driver, 'REPORTS', reports)
    monkeypatch.setattr(driver, 'RUNS', tmp_path / 'runs')
    with pytest.raises(RuntimeError, match='60 runs'):
        driver.evaluation_plan()


def test_rank_trend_retains_explicit_wp_h_placeholders(tmp_path):
    rows = f1.evaluate_f1_gates(endpoint_table(), evaluation_seed=777001)
    f1.trend(tmp_path, rows)
    import json
    trend = json.loads((tmp_path / 'rank_trend.json').read_text())
    assert len(trend) == 14
    pending = [r for r in trend if r['r'] == 32]
    assert len(pending) == 6 and all(r['reach'] is None for r in pending)
    assert (tmp_path / 'rank_trend.png').stat().st_size > 1000
