"""scripts/analyze_lsqt.py (stdlib only): the registered decision rule of docs/prereg/LSQT.md and the committed records."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('analyze_lsqt', ROOT / 'scripts' / 'analyze_lsqt.py')
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def pick(k4, rg001, rg1):
    return analyze.decide({'LT-K4': k4, 'LT-RG001': rg001, 'LT-RG1': rg1})


def test_outcome_one_flags_the_highest_arm_at_800_with_ties_in_candidate_order():
    assert pick(799, 799, 799)['outcome'] == 'no_candidate' and pick(799, 799, 799)['e8_retrain'] is False
    assert pick(800, 100, 100)['candidate'] == 'LT-K4' and pick(100, 800, 100)['candidate'] == 'LT-RG001'
    assert pick(800, 800, 800)['candidate'] == 'LT-K4' and pick(850, 900, 800)['candidate'] == 'LT-RG001'
    assert pick(100, 800, 800)['candidate'] == 'LT-RG001' and pick(900, 850, 950)['candidate'] == 'LT-RG1'


def test_outcome_two_bands_read_the_best_ridge_arm_only():
    assert pick(0, 485, 0)['best_rg_band'] == 'w3_trigger' and pick(0, 484, 0)['best_rg_band'] == 'open'
    assert pick(0, 400, 0)['best_rg_band'] == 'open' and pick(0, 399, 399)['best_rg_band'] == 'closed'
    assert pick(0, 399, 399)['trained_lsq_line'] == 'closed' and pick(0, 400, 0)['trained_lsq_line'] == 'open'
    assert pick(900, 100, 100)['best_rg_band'] == 'closed'      # K4 does not rescue the ridge band
    assert pick(0, 450, 450)['best_rg'] == 'LT-RG001' and pick(0, 450, 451)['best_rg'] == 'LT-RG1'
    assert pick(400, 0, 0)['u_control_proposed'] is True and pick(399, 399, 399)['u_control_proposed'] is False
    assert analyze.R_R128 == 335 and analyze.D128_3 == 608
    result = pick(383, 461, 426)    # the recorded result
    assert result['best_rg'] == 'LT-RG001' and result['best_rg_band'] == 'open' and result['u_control_proposed'] is True
    assert result['vs_r_r128_pp'] == {'LT-K4': 4.8, 'LT-RG001': 12.6, 'LT-RG1': 9.1}


def test_the_committed_records_verify_and_give_the_recorded_decision():
    root = ROOT / 'reports' / 'p1l-lsqt'
    if not (root / 'decision.json').exists():
        pytest.skip('the LSQT records are committed with the report')
    release = json.loads((root / 'release.json').read_text())
    runs = {a: analyze.load(root / 'arms' / a) for a in analyze.ARMS}
    facts = analyze.verify(root, release, runs)
    assert all(f['finite'] for f in facts.values())
    d = analyze.decide({a: m[max(m)]['correct'] for a, (r, m) in runs.items()})
    assert d == json.loads((root / 'decision.json').read_text()) and d['correct'] == {'LT-K4': 383, 'LT-RG001': 461, 'LT-RG1': 426}
    assert [max(m) for r, m in runs.values()] == [17970, 28005, 28245]
