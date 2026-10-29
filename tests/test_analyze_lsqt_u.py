"""scripts/analyze_lsqt_u.py (stdlib only): the split of the ridge gain over R-r128, its standard errors, the finiteness gate and the committed U records."""
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('analyze_lsqt_u', ROOT / 'scripts' / 'analyze_lsqt_u.py')
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

RECORDED = {'LT-U': 360, 'LT-K4': 383, 'LT-RG001': 461, 'LT-RG1': 426}


def test_the_split_adds_up_to_the_total_and_carries_unpaired_binomial_standard_errors():
    parts = analyze.split(RECORDED)
    assert parts['total_LT-RG001']['pp'] == pytest.approx(12.6) and parts['unit_query']['pp'] == pytest.approx(2.5)
    assert parts['store_LT-RG001']['pp'] == pytest.approx(10.1) and parts['store_LT-RG1']['pp'] == pytest.approx(6.6)
    for arm in ('LT-K4', 'LT-RG001', 'LT-RG1'):
        assert parts['unit_query']['pp'] + parts[f'store_{arm}']['pp'] == pytest.approx(parts[f'total_{arm}']['pp'])
    # sqrt(p (1 - p) / n + q (1 - q) / n), in pp
    assert parts['unit_query']['se_pp'] == pytest.approx(100 * math.sqrt(.36 * .64 / 1000 + .335 * .665 / 1000))
    assert 2.1 < parts['unit_query']['se_pp'] < 2.2 and 2.1 < parts['store_LT-RG001']['se_pp'] < 2.3
    assert analyze.se_diff(500, 500) == pytest.approx(100 * math.sqrt(.5 / 1000), rel=1e-12) and 2.2 < analyze.se_diff(500, 500) < 2.3   # the registered "about 2.2 pp"
    assert analyze.se_diff(360, 335) == analyze.se_diff(335, 360) and analyze.se_diff(0, 0) == 0 == analyze.se_diff(1000, 1000)
    assert analyze.se_one(500) == pytest.approx(1.5811, abs=1e-4)


def test_every_non_finite_number_is_refused_in_the_record_and_the_train_log(tmp_path):
    with pytest.raises(ValueError):
        analyze.strict('{"loss": NaN}')
    with pytest.raises(ValueError):
        analyze.strict('{"loss": -Infinity}')
    assert analyze.non_finite(analyze.strict('{"a": {"b": [1.0, 1e999]}}')) == ['.a.b[1]']      # an overflow parses to inf
    root = ROOT / 'reports' / 'p1l-lsqt-u' / 'arms' / 'LT-U'
    if not root.exists():
        pytest.skip('the U records are committed with the report')
    copy = tmp_path / 'LT-U'
    shutil.copytree(root, copy)
    assert analyze.check_all_finite(copy) > 5000
    log = copy / 'train_log.jsonl'
    lines = log.read_text().splitlines()
    for poisoned in (lines[7].replace('"grad_norm": ', '"grad_norm": NaN, "was": '), lines[7].replace('"loss": ', '"loss": 1e999, "was": ')):
        log.write_text('\n'.join(lines[:7] + [poisoned] + lines[8:]) + '\n')
        with pytest.raises((ValueError, AssertionError)):
            analyze.check_all_finite(copy)
    log.write_text('\n'.join(lines) + '\n')
    metrics = copy / 'val-24560' / 'metrics.json'
    metrics.write_text(metrics.read_text().replace('"nll": ', '"nll": Infinity, "was": '))
    with pytest.raises(ValueError):
        analyze.check_all_finite(copy)


def test_the_committed_u_records_verify_and_give_the_recorded_split(tmp_path):
    root = ROOT / 'reports' / 'p1l-lsqt-u'
    if not (root / 'split.json').exists():
        pytest.skip('the U records are committed with the report')
    result = analyze.main(str(root), str(ROOT / 'reports' / 'p1l-lsqt'), str(ROOT / 'reports' / 'p1l-dpfp'), split_out=tmp_path / 'split.json')
    assert result == json.loads((tmp_path / 'split.json').read_text()) == json.loads((root / 'split.json').read_text())
    assert result['correct'] == RECORDED and result['steps'] == 24560 and result['u_reaches_0_80'] is False and result['enters_registered_outcomes'] is False
    assert result['split']['store_LT-RG001']['pp'] + result['split']['unit_query']['pp'] == pytest.approx(result['split']['total_LT-RG001']['pp'])
    audit = json.loads((root / 'audit.json').read_text())['runs']['LT-U-E16']
    assert audit['original_matches_record']['correct'] is True and audit['optimizer_steps'] == 0 and audit['max_repeated_logit_drift'] == 0.0
    lsqt = json.loads((ROOT / 'reports' / 'p1l-lsqt' / 'decision.json').read_text())
    assert lsqt['outcome'] == 'no_candidate' and lsqt['best_rg'] == result['best_rg'] == 'LT-RG001'     # U does not change the registered decision
