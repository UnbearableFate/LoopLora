"""scripts/analyze_sq2.py (stdlib only): the registered decision rule of docs/prereg/SQ2.md and the verification of the committed records."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('analyze_sq2', ROOT / 'scripts' / 'analyze_sq2.py')
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

ARMS = analyze.ARMS


def inputs(**changes):
    correct = dict(zip(ARMS, (600, 600, 650, 720)))
    read_error = dict(zip(ARMS, (.40, .40, .30, .20)))
    clean = {a: [.93] * 5 for a in ARMS}
    finite = {a: True for a in ARMS}
    for key, update in changes.items():
        {'correct': correct, 'read_error': read_error, 'clean': clean, 'finite': finite}[key].update(update)
    return correct, read_error, clean, finite


def test_the_primary_contrast_needs_all_three_conditions_at_their_exact_thresholds():
    assert analyze.decide(*inputs())['advances'] is True
    d = analyze.decide(*inputs(correct={'S-RC': 699}))
    assert d['em_gap_pp'] == 9.9 and d['advances'] is False and d['line'] == 'learned-writer line stays stopped'
    assert analyze.decide(*inputs(correct={'S-RC': 700}))['advances'] is True       # exactly +10.0 pp
    assert analyze.decide(*inputs(read_error={'S-RC': .2}))['conditions']['read_error_halved'] is True   # exactly half
    assert analyze.decide(*inputs(read_error={'S-RC': .2001}))['advances'] is False
    floor = .93 - .05
    assert analyze.decide(*inputs(clean={'S-RC': [.93, .93, floor + 1e-9, .93, .93]}))['advances'] is True
    assert analyze.decide(*inputs(clean={'S-RC': [.93, .93, floor - 1e-3, .93, .93]}))['conditions']['stable_decoding'] is False
    assert analyze.decide(*inputs(finite={'S-RC': False}))['conditions']['stable_decoding'] is False
    # the reference is S-DE's FINAL clean-value accuracy, not its mean or an earlier checkpoint
    assert analyze.decide(*inputs(clean={'S-DE': [.99, .99, .99, .99, .90], 'S-RC': [.86] * 5}))['conditions']['stable_decoding'] is True


def test_candidate_interaction_alone_cells_and_the_registered_prediction():
    d = analyze.decide(*inputs(correct={'S-RC': 800}))
    assert d['candidate'] and 'flagged' in d['candidate'] and d['non_primary_at_080'] == []
    d = analyze.decide(*inputs(correct={'S-RC': 790, 'S-RE': 801}))   # a non-primary cell at 0.80 is reported, never promoted
    assert d['candidate'] is None and d['non_primary_at_080'] == ['S-RE']
    assert analyze.decide(*inputs(correct={'S-RC': 600, 'S-RE': 800}))['advances'] is False
    correct, read_error, clean, finite = inputs()
    d = analyze.decide(correct, read_error, clean, finite)
    assert d['interaction_pp'] == ((720 - 650) - (600 - 600)) / 10 == 7.0
    assert d['prediction_holds'] is True and d['alone']['S-RE']['meets_1_and_2'] is False
    # a single change that meets conditions 1 and 2 breaks the registered prediction (but not the decision)
    d = analyze.decide(*inputs(correct={'S-RE': 705}, read_error={'S-RE': .19}))
    assert d['alone']['S-RE']['meets_1_and_2'] is True and d['prediction_holds'] is False and d['advances'] is True


def test_the_committed_records_verify_and_give_the_recorded_decision():
    root = ROOT / 'reports' / 'p1l-sq2'
    if not (root / 'arms' / 'S-RC' / 'record.json').exists():
        pytest.skip('the SQ2 records are committed with the report')
    release = json.loads((root / 'release.json').read_text())
    runs = {a: analyze.load(root / 'arms' / a) for a in ARMS}
    facts = analyze.verify(root, release, runs)
    assert all(f['finite'] for f in facts.values())
    final = {a: m[6000] for a, (r, m) in runs.items()}
    d = analyze.decide({a: final[a]['correct'] for a in ARMS}, {a: final[a]['relative_read_error'] for a in ARMS},
                       {a: [m[s]['clean_value_correct'] for s in sorted(m) if s] for a, (r, m) in runs.items()}, {a: True for a in ARMS})
    recorded = json.loads((root / 'decision.json').read_text())
    assert d == recorded and d['correct'] == {'S-DE': 654, 'S-DC': 630, 'S-RE': 723, 'S-RC': 695}
    assert d['advances'] is False and d['conditions'] == dict(em_gain=False, read_error_halved=False, stable_decoding=True)
    assert runs['S-DE'][1][0]['correct'] == 608 and runs['S-DE'][1][0]['nll'] == 2.1291537284851074  # the step-0 check, again
