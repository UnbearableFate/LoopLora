"""scripts/analyze_slot_study.py (standard library and numpy): the S0 decision rows and the calibration re-verification, the S1A dev
selection and its trigger, the S1A confirmation verdict and its bootstrap, determinism, blinding and the source index
(docs/prereg/SLOT-LOOP-v1.md). No torch."""
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'analyze_slot_study.py'
spec = importlib.util.spec_from_file_location('analyze_slot_study', SCRIPT)
slot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slot)

AV, FO = 'all_valid_question', 'final_only'
CHECKS = ('native_replay', 'matrix_equals_native', 'owner_mask', 'isolation', 'frozen_set', 'bytes', 'capacity',
          'feature_consistency', 'd128_3_replay')
GRID = ([('matrix', g, None) for g in (0.5, 1.0, 2.0)] + [('soft', g, t) for t in (0.03, 0.1, 0.3, 1.0) for g in (0.5, 1.0, 2.0)]
        + [('hard', g, None) for g in (0.5, 1.0, 2.0)])
SELECTED = dict(matrix=dict(gain=1.0), soft=dict(gain=1.0, tau=0.3), hard=dict(gain=1.0))
N = 1000


def blake(data):
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def arguments(argv):
    argv = [str(a) for a in argv]
    if '--manifest' not in argv:
        root = Path(argv[argv.index('--records') + 1])
        path = root / 'manifest.lock.json'
        if not path.exists():
            write_manifest(root)
        argv += ['--manifest', str(path)]
    return argv


def run(*argv):
    assert slot.main(arguments(argv)) == 0


def refused(match, *argv):
    with pytest.raises(SystemExit, match=match) as caught:
        slot.main(arguments(argv))
    assert str(caught.value).startswith('refused: ') and '\n' not in str(caught.value)


# ---------------------------------------------------------------------------------------------------------------- S0 inputs

def parameters(role, selected):
    """The (gain, tau) a role's read must carry: g = 1 and tau = 1 uncalibrated, the selection otherwise."""
    kind = next((k for k in ('matrix', 'soft', 'hard') if role.startswith(k)), None)
    if kind is None:
        return None, None
    if role.endswith('uncal'):
        return 1.0, 1.0 if kind == 'soft' else None
    return selected[kind]['gain'], selected[kind].get('tau')


def condition(role, scope, correct, rel_err, selected=SELECTED, n=N, **extra):
    gain, tau = parameters(role, selected)
    return dict(role=role, kind=role.split('_')[0], gain=gain, tau=tau, scope=scope, recency='rec' in role, n=n, correct=correct,
                em=correct / n, nll=1.5, rel_err=rel_err, cos=0.9, norm_ratio=1.0, seconds=1.0, **extra)


def candidates(nll=None):
    """The 18 candidates; the NLL is lowest at Matrix g 1, SoftSlot (g 1, tau 0.3) and HardSlot g 1 unless `nll` {(kind, g, tau): x} changes it."""
    return [dict(kind=k, gain=g, tau=t, nll=(nll or {}).get((k, g, t), 1.5 + 0.1 * abs(g - 1) + (0.0 if k != 'soft' or t == 0.3 else 0.05)),
                 correct=1000, seconds=15.0) for k, g, t in GRID]


def episode_ids(n):
    """The dev episode ids shared by every condition (distinct int64)."""
    return np.random.default_rng(2026).integers(0, 2 ** 62, size=n, dtype=np.int64)


def ids_digest(ids):
    """blake2b-128 of the ids as little-endian int64: the digest metrics.json carries."""
    return blake(np.asarray(ids).astype('<i8').tobytes())


def write_manifest(root, s1a_n=400, s0_n=N, dev_n=1000, dev_ids='cd' * 16):
    """A locked synthetic registry, fixed independently of the result files under test."""
    values = {'S0/calibration': (2048, '1' * 32), 'S0/dev': (s0_n, ids_digest(episode_ids(s0_n))),
              'S1A/dev': (dev_n, dev_ids),
              'S1A/confirmation': (s1a_n, ids_digest(np.arange(s1a_n, dtype=np.int64) + 1000)),
              'S1A/unseen': (s1a_n, ids_digest(np.arange(s1a_n, dtype=np.int64) + 10 ** 6))}
    locked = dict(version=1, status='locked', study='slot-loop-v1',
                  manifest=dict(sets={key: dict(count=n) for key, (n, _) in values.items()},
                                ids={f'eval/{key}': dict(count=n, ids_blake2b_128=own) for key, (n, own) in values.items()}))
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'manifest.lock.json'
    path.write_text(json.dumps(locked))
    return path


def s0_metrics(correct=None, rel_err=None, selected=SELECTED, nll=None, n=N, d128=None):
    """A consistent S0 record. Every condition has 700 correct and read error 0.20 (decision row R5) unless 'role/scope' says otherwise."""
    correct, rel_err = correct or {}, rel_err or {}
    conditions = {}
    for role, scope in slot.CONDITIONS:
        key = f'{role}/{scope}'
        extra = dict(w_true=0.7, w_gap=0.2, w_entropy=1.1) if role.startswith('soft') else {}
        conditions[key] = condition(role, scope, correct.get(key, 700), rel_err.get(key, 0.2), selected, n, **extra)
    old = dict(correct=608, n=1000, loss_float32=2.1291537284851074)
    counts = dict(native=608, matrix=600, soft=640, hard=630) if d128 is None else d128
    return dict(version=1, stage='S0', n=n, ids_blake2b_128=ids_digest(episode_ids(n)),
                calibration=dict(n=2048, ids_blake2b_128='1' * 32, candidates=candidates(nll), selected=selected), conditions=conditions,
                common=dict(key_top1=0.87, key_margin=0.3),
                d128_3=dict(conditions={f'{r}/{AV}': condition(r, AV, c, 0.3, selected, n) for r, c in counts.items()}, old_range=old))


def s0_samples(metrics, seed=0):
    """Per-sample arrays whose correct counts are the metrics' counts; the episode ids are shared by every condition."""
    rng = np.random.default_rng(seed)
    n = metrics['n']
    ids = episode_ids(n)
    arrays = {}
    for key, row in metrics['conditions'].items():
        arrays[f'{key}/episode_id'] = ids
        arrays[f'{key}/correct'] = rng.permutation(n) < row['correct']
        arrays[f'{key}/nll'] = rng.uniform(0.5, 3.0, n)
        arrays[f'{key}/rel_err'] = rng.uniform(0.0, 1.0, n)
    native, matrix = f'native/{AV}', f'matrix_uncal/{AV}'
    if metrics['conditions'][native]['correct'] == metrics['conditions'][matrix]['correct']:
        arrays[f'{matrix}/correct'] = arrays[f'{native}/correct'].copy()
    arrays[f'{matrix}/nll'] = arrays[f'{native}/nll'].copy()
    return arrays


def passing(**changes):
    return dict(checks={name: dict(passed=changes.get(name, True), detail='x') for name in CHECKS}, passed=all(changes.values()))


def write_s0(root, metrics, preflight=None, arrays=None):
    write_manifest(root)
    (root / 'metrics.json').write_text(json.dumps(metrics))
    np.savez(root / 'samples.npz', **(s0_samples(metrics) if arrays is None else arrays))
    (root / 'preflight.json').write_text(json.dumps(passing() if preflight is None else preflight))
    return root


def run_s0(tmp_path, metrics, preflight=None, arrays=None, out='out'):
    root = write_s0(tmp_path / 'run', metrics, preflight, arrays)
    run('--stage', 'S0', '--records', root, '--preflight', root / 'preflight.json', '--out', tmp_path / out)
    return json.loads((tmp_path / out / 'decision_s0.json').read_text())


def mini(n=N, **kv):
    """Only the fields the decision reads: counts and read errors of matrix_cal, soft_cal, hard_cal and clean_final."""
    c = {f'{r}/{s}': dict(correct=700, rel_err=0.2) for r, s in (('matrix_cal', AV), ('soft_cal', AV), ('hard_cal', AV), ('matrix_cal', FO),
                                                                ('soft_cal', FO), ('hard_cal', FO), ('clean_final', FO))}
    for key, (correct, rel_err) in kv.items():
        c[key.replace('__', '/')] = dict(correct=correct, rel_err=rel_err)
    return c, n


# ------------------------------------------------------------------------------------------------------ S0: decision rows

S0_ROWS = {
    'R1': ({'soft_cal/all_valid_question': 750}, {'soft_cal/all_valid_question': 0.1}, 'full adaptation'),
    'R2': ({'hard_cal/all_valid_question': 750}, {'hard_cal/all_valid_question': 0.1},
           'full adaptation; soft mixture or temperature suspected, HardSlot stays diagnostic'),
    'R3': ({'soft_cal/final_only': 750}, {'soft_cal/final_only': 0.1},
           'limited adaptation; final_only reported separately; a read-position protocol study is flagged (not authorized here)'),
    'R4': ({'clean_final/final_only': 750}, {},
           'limited adaptation; addressing, query or role interface flagged as the main bottleneck (S2 priority)'),
    'R5': ({}, {}, 'limited adaptation: one pre-budgeted pass'),
}


@pytest.mark.parametrize('row', sorted(S0_ROWS))
def test_rows_r1_to_r5_are_each_reached_with_the_s1a_choices_of_the_verified_selection(tmp_path, row):
    correct, rel_err, label = S0_ROWS[row]
    d = run_s0(tmp_path, s0_metrics(correct, rel_err))
    assert d['row'] == row and d['label'] == label
    assert d['s1a'] == dict(arms=['Matrix-L1', 'SoftSlot-L1'], matrix=dict(gain=1.0), soft_slot=dict(gain=1.0, tau=0.3))
    assert 'decision_s0.md' in {p.name for p in (tmp_path / 'out').iterdir()}


def test_row_r0_and_the_labels_are_the_registered_text(tmp_path):
    root = tmp_path / 'p'
    root.mkdir()
    (root / 'preflight.json').write_text(json.dumps(passing(matrix_equals_native=False)))
    run('--stage', 'S0', '--records', root, '--preflight', root / 'preflight.json', '--out', tmp_path / 'out')
    d = json.loads((tmp_path / 'out' / 'decision_s0.json').read_text())
    assert d['row'] == 'R0' and d['label'] == 'stop, fix the interface, no comparison, no S1A' and d['s1a'] is None
    assert slot.LABELS['R5'] == S0_ROWS['R5'][2] and slot.CHECKS == CHECKS


@pytest.mark.parametrize('mode', ['failed', 'missing', 'not_boolean'])
@pytest.mark.parametrize('name', CHECKS)
def test_r0_for_each_failing_missing_or_non_boolean_required_check(tmp_path, name, mode):
    pre = passing()
    if mode == 'failed':
        pre['checks'][name]['passed'] = False
    elif mode == 'missing':
        del pre['checks'][name]
    else:
        pre['checks'][name]['passed'] = 'true'       # a string is not the boolean True
    root = tmp_path / 'p'
    root.mkdir()
    (root / 'preflight.json').write_text(json.dumps(pre))
    # no metrics.json and no samples.npz exist: after a failed gate no S0 data are opened
    run('--stage', 'S0', '--records', root, '--preflight', root / 'preflight.json', '--out', tmp_path / 'out')
    d = json.loads((tmp_path / 'out' / 'decision_s0.json').read_text())
    assert d['row'] == 'R0' and d['preflight']['failed'] == [name] and d['preflight']['required'][name] is False
    index = json.loads((tmp_path / 'out' / 'source_index.json').read_text())
    assert sorted(index['inputs']) == sorted([str((root / 'preflight.json').resolve()), str((root / 'manifest.lock.json').resolve())])


def test_r0_lists_every_missing_check_and_wins_over_a_good_comparison(tmp_path):
    d = run_s0(tmp_path, s0_metrics({'soft_cal/all_valid_question': 800}, {'soft_cal/all_valid_question': 0.05}), preflight=dict(passed=True))
    assert d['row'] == 'R0' and d['preflight']['failed'] == list(CHECKS)
    assert 'differences' not in d and 'conditions' not in d


@pytest.mark.parametrize('field', ['correct', 'nll'])
def test_s0_dev_per_sample_native_matrix_mismatch_is_r0_even_with_equal_aggregate_means(tmp_path, field):
    m = s0_metrics({'soft_cal/all_valid_question': 800}, {'soft_cal/all_valid_question': 0.05})
    arrays = s0_samples(m)
    key = f'matrix_uncal/{AV}/{field}'
    a = arrays[key]
    if field == 'correct':
        i, j = np.flatnonzero(a)[0], np.flatnonzero(~a)[0]
    else:
        i, j = 0, 1
    a[[i, j]] = a[[j, i]]
    d = run_s0(tmp_path, m, arrays=arrays)
    assert d['row'] == 'R0' and d['s1a'] is None
    assert d['native_matrix_identity'][field] is False
    assert 'differences' not in d and 'tests' not in d
    assert 'differ per sample on S0 dev' in (tmp_path / 'out' / 'decision_s0.md').read_text()


def test_s0_native_matrix_nll_identity_has_no_tolerance_and_requires_both_arrays(tmp_path):
    m = s0_metrics()
    arrays = s0_samples(m)
    arrays[f'matrix_uncal/{AV}/nll'][0] = np.nextafter(arrays[f'native/{AV}/nll'][0], np.inf)
    assert run_s0(tmp_path, m, arrays=arrays)['row'] == 'R0'
    del arrays[f'matrix_uncal/{AV}/nll']
    root = write_s0(tmp_path / 'missing', m, arrays=arrays)
    refused('lacks.*matrix_uncal/all_valid_question/nll', '--stage', 'S0', '--records', root,
            '--preflight', root / 'preflight.json', '--out', tmp_path / 'missing-out')


def test_the_five_point_threshold_is_exact_integer_arithmetic():
    for n, needed in ((1000, 50), (1001, 51), (999, 50), (2000, 100)):
        assert needed * 20 >= n > (needed - 1) * 20
        assert slot.decide_s0(*mini(n, soft_cal__all_valid_question=(700 + needed, 0.1)))[0] == 'R1'
        assert slot.decide_s0(*mini(n, soft_cal__all_valid_question=(700 + needed - 1, 0.1)))[0] == 'R5'
    row, tests = slot.decide_s0(*mini(soft_cal__all_valid_question=(750, 0.1)))
    assert tests['soft_cal/all_valid_question'] == dict(vs='matrix_cal/all_valid_question', d_correct=50, five_pp=True,
                                                         read_error_lower=True, improves=True)
    assert slot.decide_s0(*mini(hard_cal__all_valid_question=(750, 0.1)))[0] == 'R2'
    assert slot.decide_s0(*mini(hard_cal__all_valid_question=(749, 0.1)))[0] == 'R5'


def test_improves_also_needs_a_strictly_lower_mean_read_error():
    assert slot.decide_s0(*mini(soft_cal__all_valid_question=(750, 0.2)))[0] == 'R5'            # equal read error
    assert slot.decide_s0(*mini(soft_cal__all_valid_question=(750, 0.25)))[0] == 'R5'           # higher read error
    assert slot.decide_s0(*mini(soft_cal__all_valid_question=(750, 0.1999999)))[0] == 'R1'
    assert slot.decide_s0(*mini(soft_cal__all_valid_question=(900, 0.25), hard_cal__all_valid_question=(750, 0.1)))[0] == 'R2'
    assert slot.decide_s0(*mini(soft_cal__final_only=(750, 0.25), hard_cal__final_only=(750, 0.2)))[0] == 'R5'
    assert slot.decide_s0(*mini(soft_cal__final_only=(750, 0.19)))[0] == 'R3'


def test_rows_follow_the_registered_order_and_clean_final_needs_only_the_count():
    both = dict(soft_cal__all_valid_question=(750, 0.1), hard_cal__all_valid_question=(750, 0.1), soft_cal__final_only=(750, 0.1),
                clean_final__final_only=(750, 0.0))
    assert slot.decide_s0(*mini(**both))[0] == 'R1'
    del both['soft_cal__all_valid_question']
    assert slot.decide_s0(*mini(**both))[0] == 'R2'
    del both['hard_cal__all_valid_question']
    assert slot.decide_s0(*mini(**both))[0] == 'R3'
    del both['soft_cal__final_only']
    assert slot.decide_s0(*mini(**both))[0] == 'R4'
    assert slot.decide_s0(*mini(clean_final__final_only=(750, 0.5)))[0] == 'R4'                  # no read-error condition
    assert slot.decide_s0(*mini(clean_final__final_only=(749, 0.0)))[0] == 'R5'


def test_every_comparison_uses_the_matrix_cal_of_its_own_scope():
    # all_valid matrix_cal 600, final_only matrix_cal 720: the final_only reads (700) beat the other scope's baseline by 100 but not their own
    c, n = mini(matrix_cal__all_valid_question=(600, 0.3), soft_cal__all_valid_question=(600, 0.1), hard_cal__all_valid_question=(600, 0.1),
                matrix_cal__final_only=(720, 0.3), soft_cal__final_only=(700, 0.1), hard_cal__final_only=(700, 0.1),
                clean_final__final_only=(700, 0.0))
    assert slot.decide_s0(c, n)[0] == 'R5'
    c['soft_cal/final_only'] = dict(correct=770, rel_err=0.1)
    row, tests = slot.decide_s0(c, n)
    assert row == 'R3' and tests['soft_cal/final_only']['d_correct'] == 50 and tests['soft_cal/final_only']['vs'] == 'matrix_cal/final_only'
    c['matrix_cal/all_valid_question'] = dict(correct=800, rel_err=0.3)       # soft_cal at all_valid (600) is 200 below its own baseline
    c['soft_cal/final_only'] = dict(correct=700, rel_err=0.1)
    assert slot.decide_s0(c, n)[0] == 'R5'


def test_notes_soft_rec_and_the_d128_3_recheck(tmp_path):
    selected = dict(matrix=dict(gain=1.0), soft=dict(gain=1.0, tau=0.03), hard=dict(gain=1.0))
    nll = {('soft', 1.0, 0.03): 1.0}
    m = s0_metrics({'soft_rec_cal/all_valid_question': 703, 'soft_rec_uncal/all_valid_question': 697}, selected=selected, nll=nll,
                   d128=dict(native=608, matrix=640, soft=600, hard=650))
    d = run_s0(tmp_path, m)
    assert 'tau = 0.03 selected: HardSlot is the tau -> 0 limit; the grid is not extended' in d['notes']
    assert d['soft_rec_vs_soft']['largest_abs_d_correct'] == 3 and d['soft_rec_vs_soft']['uncal']['d_correct'] == -3
    assert any('largest EM-count difference 3 of 1000' in x and '0.999999' in x for x in d['notes'])
    r = d['d128_3']['recheck']
    assert r['soft_minus_matrix']['direction'] == 'lower' and r['hard_minus_matrix']['direction'] == 'higher'
    assert r['s_re_soft_minus_matrix']['direction'] == 'equal' and r['same_direction_as_s_re'] is False
    assert d['d128_3']['old_range'] == dict(correct=608, n=1000, loss_float32=2.1291537284851074)
    other = run_s0(tmp_path, s0_metrics(), out='other')
    assert not any('tau = 0.03' in x for x in other['notes'])
    md = (tmp_path / 'out' / 'decision_s0.md').read_text()
    assert '2.1291537284851074' in md and 'reported, not judged' in md


def test_s0_tables_carry_every_condition_the_calibration_grid_and_the_descriptive_intervals(tmp_path):
    d = run_s0(tmp_path, s0_metrics({'soft_cal/all_valid_question': 750}, {'soft_cal/all_valid_question': 0.1}))
    assert sorted(d['conditions']) == sorted(f'{r}/{s}' for r, s in slot.CONDITIONS) and len(d['conditions']) == 13
    assert [(c['kind'], c['gain'], c['tau']) for c in d['calibration']['candidates']] == GRID
    assert d['calibration']['selected'] == SELECTED
    diff = d['differences']
    assert sorted(diff) == sorted(f'{r}/{s}' for r, s in slot.COMPARED) and diff['soft_cal/all_valid_question']['d_correct'] == 50
    assert diff['soft_cal/all_valid_question']['pp'] == 5.0 and diff['native/all_valid_question']['vs'] == 'matrix_cal/all_valid_question'
    assert d['bootstrap'] == dict(seed=2079281455337892545, resamples=10000, block=500, descriptive=True)
    assert d['ids_blake2b_128'] == ids_digest(episode_ids(N)) and d['n'] == N
    md = (tmp_path / 'out' / 'decision_s0.md').read_text()
    assert '| soft | 1 | 0.30 | 1.500000 | 1000 | 15.00 | yes |' in md and 'never used in the decision' in md and 'w_entropy' in md


def numpy_interval_pp(vector, n, seed):
    """The registered bootstrap re-implemented with numpy only: 20 sequential blocks of 500 index rows, np.percentile at 2.5 and 97.5."""
    rng = np.random.default_rng(seed)
    means = np.concatenate([vector[rng.integers(0, n, size=(500, n))].mean(axis=1) for _ in range(20)])
    return 100 * np.percentile(means, [2.5, 97.5])


def test_s0_intervals_reproduce_with_numpy_and_never_touch_the_decision(tmp_path):
    m = s0_metrics({'soft_cal/all_valid_question': 750, 'clean_final/final_only': 712}, {'soft_cal/all_valid_question': 0.1})
    arrays = s0_samples(m, seed=3)
    d = run_s0(tmp_path, m, arrays=arrays)
    for key, base in (('soft_cal/all_valid_question', 'matrix_cal/all_valid_question'), ('clean_final/final_only', 'matrix_cal/final_only')):
        vector = arrays[f'{key}/correct'].astype(float) - arrays[f'{base}/correct']
        assert d['differences'][key]['ci95_pp'] == pytest.approx(list(numpy_interval_pp(vector, N, 2079281455337892545)), abs=1e-9)
    # a different sample draw (other intervals, same counts) cannot change the row
    again = run_s0(tmp_path, m, arrays=s0_samples(m, seed=4), out='again')
    assert again['row'] == d['row'] == 'R1' and again['differences'] != d['differences']


def test_s0_outputs_are_byte_identical_between_runs(tmp_path):
    m = s0_metrics({'soft_cal/all_valid_question': 750}, {'soft_cal/all_valid_question': 0.1})
    arrays = s0_samples(m)
    root = write_s0(tmp_path / 'run', m, arrays=arrays)
    for out in ('a', 'b'):
        run('--stage', 'S0', '--records', root, '--preflight', root / 'preflight.json', '--out', tmp_path / out / 'nested')
    names = sorted(p.name for p in (tmp_path / 'a' / 'nested').iterdir())
    assert names == ['decision_s0.json', 'decision_s0.md', 'source_index.json']
    for name in names:
        assert (tmp_path / 'a' / 'nested' / name).read_bytes() == (tmp_path / 'b' / 'nested' / name).read_bytes()


# ------------------------------------------------------------------------------------------------------ S0: calibration rule

def cal(nll=None, selected=SELECTED):
    return dict(n=2048, candidates=candidates(nll), selected=selected)


def oracle(raw):
    """The same rule written out independently: lowest finite NLL, 1e-4 ties to the gain nearest 1, then the larger tau."""
    out = {}
    for kind in ('matrix', 'soft', 'hard'):
        pool = [c for c in raw['candidates'] if c['kind'] == kind and c['nll'] is not None and c['nll'] == c['nll']]
        low = min(c['nll'] for c in pool)
        tied = [c for c in pool if c['nll'] - low <= 1e-4]
        best = sorted(tied, key=lambda c: (abs(c['gain'] - 1), -(c['tau'] or 0)))[0]
        out[kind] = dict(gain=best['gain'], tau=best['tau']) if kind == 'soft' else dict(gain=best['gain'])
    return out


def test_calibration_tie_goes_to_the_gain_nearest_one_then_the_larger_tau():
    # Matrix: g = 2 is lowest by 1e-5, g = 0.5 lower than g = 1 by 5e-5; all within 1e-4 -> g = 1
    nll = {('matrix', 2.0, None): 1.00000, ('matrix', 1.0, None): 1.00001, ('matrix', 0.5, None): 1.00006}
    assert slot.parse_calibration(cal(nll, dict(SELECTED, matrix=dict(gain=1.0))))['selected']['matrix'] == dict(gain=1.0)
    # SoftSlot: (0.5, 1.0), (1.0, 0.1) and (1.0, 0.3) are tied; g = 1 beats g = 0.5, the larger tau wins among g = 1
    nll = {('soft', 0.5, 1.0): 1.00000, ('soft', 1.0, 0.1): 1.00004, ('soft', 1.0, 0.3): 1.00008, ('soft', 2.0, 0.3): 1.0001}
    assert slot.parse_calibration(cal(nll, dict(SELECTED, soft=dict(gain=1.0, tau=0.3))))['selected']['soft'] == dict(gain=1.0, tau=0.3)
    # with g = 0.5 and g = 2 tied (|g - 1| = 0.5 against 1), g = 0.5 wins; the tie set excludes the candidate 1.1e-4 above the minimum
    nll = {('hard', 0.5, None): 1.00000, ('hard', 2.0, None): 1.00003, ('hard', 1.0, None): 1.00011}
    assert slot.parse_calibration(cal(nll, dict(SELECTED, hard=dict(gain=0.5))))['selected']['hard'] == dict(gain=0.5)
    # a candidate 1.1e-4 above the minimum is not tied: the minimum itself is selected even though g = 1 is nearer
    nll = {('matrix', 0.5, None): 1.00000, ('matrix', 1.0, None): 1.00011}
    assert slot.parse_calibration(cal(nll, dict(SELECTED, matrix=dict(gain=0.5))))['selected']['matrix'] == dict(gain=0.5)
    # ties between equal NLLs follow the same rule
    nll = {('soft', 1.0, 0.1): 1.0, ('soft', 1.0, 1.0): 1.0, ('soft', 1.0, 0.3): 1.0}
    assert slot.parse_calibration(cal(nll, dict(SELECTED, soft=dict(gain=1.0, tau=1.0))))['selected']['soft'] == dict(gain=1.0, tau=1.0)


def test_a_wrong_selected_value_is_refused_whatever_the_error():
    tie = {('soft', 0.5, 1.0): 1.00000, ('soft', 1.0, 0.1): 1.00004, ('soft', 1.0, 0.3): 1.00008}
    for wrong in (dict(gain=0.5, tau=1.0), dict(gain=1.0, tau=0.1), dict(gain=2.0, tau=0.3), dict(gain=1.0, tau=0.03)):
        with pytest.raises(SystemExit, match='selected soft differs'):
            slot.parse_calibration(cal(tie, dict(SELECTED, soft=wrong)))
    with pytest.raises(SystemExit, match='selected matrix differs'):
        slot.parse_calibration(cal(None, dict(SELECTED, matrix=dict(gain=2.0))))
    with pytest.raises(SystemExit, match='selected hard differs'):
        slot.parse_calibration(cal(None, dict(SELECTED, hard=dict(gain=0.5))))
    with pytest.raises(SystemExit, match="lacks 'hard'"):
        slot.parse_calibration(cal(None, dict(matrix=dict(gain=1.0), soft=dict(gain=1.0, tau=0.3))))
    assert slot.parse_calibration(cal(None, SELECTED))['selected'] == SELECTED


def test_calibration_never_selects_a_non_finite_candidate_and_needs_the_whole_registered_grid():
    raw = cal(None, dict(SELECTED, matrix=dict(gain=2.0)))
    raw['candidates'][0]['nll'] = float('nan')                       # Matrix g 0.5: a naive comparison with NaN would let it win or block
    raw['candidates'][1]['nll'] = None                               # Matrix g 1 (the lowest) is not finite either
    assert slot.parse_calibration(raw)['selected']['matrix'] == dict(gain=2.0)
    raw['candidates'][2]['nll'] = float('inf')
    with pytest.raises(SystemExit, match='no finite matrix candidate'):
        slot.parse_calibration(raw)
    short = cal(None)
    del short['candidates'][5]
    with pytest.raises(SystemExit, match='17 of the 18'):
        slot.parse_calibration(short)
    twice = cal(None)
    twice['candidates'][5] = dict(twice['candidates'][4])
    with pytest.raises(SystemExit, match='appears twice'):
        slot.parse_calibration(twice)
    other = cal(None)
    other['candidates'][4]['gain'] = 4.0
    with pytest.raises(SystemExit, match='not in the registered grid'):
        slot.parse_calibration(other)


def test_a_wrong_calibration_selection_is_refused_through_main(tmp_path):
    root = write_s0(tmp_path / 'run', s0_metrics(selected=dict(SELECTED, soft=dict(gain=2.0, tau=0.3))))
    refused('selected soft differs', '--stage', 'S0', '--records', root, '--preflight', root / 'preflight.json', '--out', tmp_path / 'out')


def test_the_script_agrees_with_an_independent_statement_of_the_rule_on_tie_heavy_surfaces():
    rng = np.random.default_rng(11)
    for _ in range(25):
        nll = {key: 1.0 + float(rng.integers(0, 4)) * 4e-5 for key in GRID}      # steps of 4e-5: ties inside 1e-4, never on its edge
        raw = cal(nll)
        raw['selected'] = oracle(raw)
        assert slot.parse_calibration(raw)['selected'] == raw['selected']


# --------------------------------------------------------------------------------------------- S0: refusals of inconsistent data

def test_s0_refuses_inconsistent_or_missing_data(tmp_path):
    def attempt(name, match, metrics=None, arrays=None, tweak=None, drop=()):
        metrics = s0_metrics() if metrics is None else metrics
        root = write_s0(tmp_path / name, metrics, arrays=arrays)
        if tweak:
            tweak(root)
        for file in drop:
            (root / file).unlink()
        refused(match, '--stage', 'S0', '--records', root, '--preflight', root / 'preflight.json', '--out', tmp_path / (name + '-out'))

    attempt('no-metrics', 'cannot read metrics.json', drop=['metrics.json'])
    attempt('no-samples', 'cannot read samples.npz', drop=['samples.npz'])
    refused('cannot read', '--stage', 'S0', '--records', tmp_path, '--preflight', tmp_path / 'nothing.json', '--out', tmp_path / 'x')
    bad_ids = s0_samples(s0_metrics())
    bad_ids['soft_cal/final_only/episode_id'] = bad_ids['soft_cal/final_only/episode_id'][::-1].copy()
    attempt('ids', 'episode_id of soft_cal/final_only differs', arrays=bad_ids)
    short = s0_samples(s0_metrics())
    for field in ('episode_id', 'correct', 'nll', 'rel_err'):
        short[f'hard_cal/all_valid_question/{field}'] = short[f'hard_cal/all_valid_question/{field}'][:-1]
    attempt('n', 'not an int64 array of n episodes', arrays=short)
    wrong_count = s0_samples(s0_metrics())
    wrong_count['soft_cal/all_valid_question/correct'] = ~wrong_count['soft_cal/all_valid_question/correct']
    attempt('count', 'another n or correct count', arrays=wrong_count)
    missing_array = s0_samples(s0_metrics())
    del missing_array['clean_final/final_only/correct']
    attempt('array', 'lacks episode_id or correct of clean_final/final_only', arrays=missing_array)
    m = s0_metrics()
    m['conditions']['hard_cal/final_only']['n'] = 999
    attempt('cond-n', 'different n than the record', metrics=m)
    m = s0_metrics()
    m['conditions']['hard_cal/final_only']['rel_err'] = float('nan')
    attempt('nan', 'rel_err is not a finite number', metrics=m)
    m = s0_metrics()
    del m['conditions']['clean_final/final_only']
    attempt('missing', "lacks 'clean_final/final_only'", metrics=m)
    m = s0_metrics()
    m['conditions']['soft_cal/all_valid_question']['tau'] = 1.0
    attempt('tau', 'not run at the registered gain and tau', metrics=m)
    m = s0_metrics()
    m['d128_3']['conditions'][f'soft/{AV}']['gain'] = 2.0
    attempt('d128', 'not run at the registered gain and tau', metrics=m)
    m = s0_metrics()
    m['conditions']['soft_cal/all_valid_question']['role'] = 'hard_cal'
    attempt('role', 'not a condition row of that role', metrics=m)
    m = s0_metrics()
    arrays = s0_samples(m)
    m['n'] = 1000.0
    attempt('float-n', 'not an integer', metrics=m, arrays=arrays)
    attempt('dup', 'duplicate key', tweak=lambda root: (root / 'metrics.json').write_text(
        (root / 'metrics.json').read_text().replace('"stage": "S0"', '"stage": "S0", "stage": "S0"')))
    attempt('garbage', 'not valid JSON', tweak=lambda root: (root / 'metrics.json').write_text('{'))


def test_the_episode_ids_of_samples_must_match_the_digest_of_metrics_json(tmp_path):
    ids = episode_ids(N)
    assert len({ids_digest(ids), blake(ids.astype('>i8').tobytes()), ids_digest(ids[::-1])}) == 3    # byte order and order both matter
    good = s0_metrics()
    assert good['ids_blake2b_128'] == ids_digest(ids)
    tampered = {'wrong': dict(good, ids_blake2b_128='0' * 32), 'big-endian': dict(good, ids_blake2b_128=blake(ids.astype('>i8').tobytes())),
                'reversed': dict(good, ids_blake2b_128=ids_digest(ids[::-1]))}
    for name, m in tampered.items():
        root = write_s0(tmp_path / name, m)
        refused('episode ids of samples.npz do not match ids_blake2b_128', '--stage', 'S0', '--records', root,
                '--preflight', root / 'preflight.json', '--out', tmp_path / (name + '-out'))
    absent = dict(good)
    del absent['ids_blake2b_128']
    root = write_s0(tmp_path / 'absent', absent)
    refused('episode ids of samples.npz do not match', '--stage', 'S0', '--records', root, '--preflight', root / 'preflight.json',
            '--out', tmp_path / 'absent-out')
    # A coherent replacement set still differs from the independent locked registry and must be refused.
    other = ids[::-1].copy()
    arrays = {k: (other if k.endswith('/episode_id') else v) for k, v in s0_samples(good).items()}
    root = write_s0(tmp_path / 'other', dict(good, ids_blake2b_128=ids_digest(other)), arrays=arrays)
    refused('S0/dev: ids digest differs from locked manifest', '--stage', 'S0', '--records', root,
            '--preflight', root / 'preflight.json', '--out', tmp_path / 'other-out')


# -------------------------------------------------------------------------------------------------------------------- S1A-DEV

def dev_record(arm, seed, nll, correct, steps=1000, **over):
    """One run record. `nll` and `correct` hold step 0 and the ten selectable rows; the registered rule picks `selected_step`."""
    marks = [0] + [steps * k // 10 for k in range(1, 11)]
    rows = [dict(step=m, selectable=m >= 1, n=1000, correct=c, nll=v, em=c / 1000) for m, v, c in zip(marks, nll, correct)]
    rec = dict(status='complete', ablation=False, arm=arm, seed=seed, steps=steps, dev=rows, selected_step=None, selected_checkpoint_blake2b_128='ab' * 16,
               dev_ids_blake2b_128='cd' * 16, training=dict(all_finite=True, clipped_fraction=0.1, grad_norm_max=3.0))
    rec.update(over)
    return rec


FALLING = [2.0, 1.5, 1.2, 1.0, 0.9, 0.85, 0.8, 0.78, 0.77, 0.76, 0.75]          # selected: the last row, step 1000
COUNTS = [600, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695]


def dev_tree(root, soft_extra=(30, 30, 30), mutate=None, nll=FALLING):
    """Six records; the soft arm gains `soft_extra[seed]` correct at every selectable step."""
    write_manifest(root)
    for arm in ('matrix_L1', 'soft_slot_L1'):
        for i, seed in enumerate((17, 29, 43)):
            extra = soft_extra[i] if arm == 'soft_slot_L1' else 0
            rec = dev_record(arm, seed, nll, [COUNTS[0]] + [c + extra for c in COUNTS[1:]], selected_step=1000)
            if mutate:
                rec = mutate(arm, seed, rec) or rec
            path = root / arm / f's{seed}'
            path.mkdir(parents=True, exist_ok=True)
            (path / 'record.json').write_text(json.dumps(rec))
    return root


def run_dev(tmp_path, out='out', **kw):
    root = dev_tree(tmp_path / 'dev', **kw)
    run('--stage', 'S1A-DEV', '--records', root, '--out', tmp_path / out)
    return json.loads((tmp_path / out / 's1a-selection.json').read_text())


def test_dev_trigger_boundary_is_exact_integer_arithmetic(tmp_path):
    d = run_dev(tmp_path, soft_extra=(30, 30, 30))          # sum of count differences 90: 90 * 100 == 9 * 1000, not below
    assert d['outcome'] == 'no_trigger' and d['differences']['sum_d_correct'] == 90 and d['differences']['mean_pp'] == 3.0
    d = run_dev(tmp_path, out='b', soft_extra=(30, 30, 29))
    assert d['outcome'] == 'ablation' and d['differences']['sum_d_correct'] == 89 and d['reasons'] == []
    assert d['differences']['per_seed']['43'] == dict(d_correct=29, pp=2.9)
    d = run_dev(tmp_path, out='c', soft_extra=(40, 30, 21))   # per-seed differences may be below 3.0 pp; only the mean counts
    assert d['outcome'] == 'no_trigger'
    d = run_dev(tmp_path, out='d', soft_extra=(-50, 100, 40))
    assert d['outcome'] == 'no_trigger' and d['differences']['sum_d_correct'] == 90
    md = (tmp_path / 'd' / 's1a-selection.md').read_text()
    assert 'no ablation' in md and 'matrix_L1/s17' in md


def test_dev_selection_rule_with_the_one_micro_nat_tie_to_the_earlier_row(tmp_path):
    def curve(*rows):
        base = [2.0] + [5.0] * 10
        for step_index, value in rows:
            base[step_index] = value
        return base
    expected_by_rows = {
        'tie to the earlier': ([(1, 1.0), (2, 0.9999995)], 100),               # lower by 5e-7: a tie
        'strictly better by more than 1e-6': ([(1, 1.0), (2, 0.999998)], 200),
        'chain: the best so far is the reference': ([(1, 1.0), (2, 0.9999995), (3, 0.9999985)], 300),
        'a dip, then worse': ([(1, 1.5), (5, 0.9), (9, 1.0)], 500),
        'a later equal value does not replace': ([(2, 1.2), (7, 1.2)], 200),
    }
    for name, (rows, step) in expected_by_rows.items():
        def mutate(arm, seed, rec, rows=rows, step=step):
            values = curve(*rows)
            rec = dev_record(arm, seed, values, [600] + [650] * 10, selected_step=step)
            return rec
        d = run_dev(tmp_path, out=name.replace(' ', '_').replace(':', ''), mutate=mutate)
        assert {r['selected_step'] for r in d['runs'].values()} == {step}, name
    # step 0 is recorded but never selectable even when it is the lowest row; the run is then unstable (selected above step 0)
    def low_start(arm, seed, rec):
        return dev_record(arm, seed, [0.1] + [1.0] + [2.0] * 9, [600] + [650] * 10, selected_step=100)
    d = run_dev(tmp_path, out='low', mutate=low_start)
    assert d['outcome'] == 'stop_unstable' and all(r['selected_step'] == 100 for r in d['runs'].values())


def test_an_inconsistent_selected_step_is_refused(tmp_path):
    for step, name in ((900, 'earlier'), (None, 'none'), (0, 'zero')):
        def mutate(arm, seed, rec, step=step):
            if (arm, seed) == ('soft_slot_L1', 29):
                rec['selected_step'] = step
        root = dev_tree(tmp_path / name, mutate=mutate)
        refused('soft_slot_L1/s29/record.json: selected_step differs from the registered rule',
                '--stage', 'S1A-DEV', '--records', root, '--out', tmp_path / (name + '-out'))
    # the tie rule is part of the check: a record that picked the strictly lowest of a 5e-7 tie is refused
    values = [2.0, 1.0, 0.9999995] + [5.0] * 8
    root = dev_tree(tmp_path / 'tie', nll=values, mutate=lambda arm, seed, rec: rec.update(selected_step=200))
    refused('selected_step differs', '--stage', 'S1A-DEV', '--records', root, '--out', tmp_path / 'tie-out')


def test_unstable_runs_stop_with_the_failing_runs_listed(tmp_path):
    def above(arm, seed, rec):                                 # selected dev NLL above its own step-0 NLL
        if (arm, seed) == ('matrix_L1', 43):
            return dev_record(arm, seed, [1.0, 1.1, 1.2, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], COUNTS, selected_step=100)
    d = run_dev(tmp_path, mutate=above)
    assert d['outcome'] == 'stop_unstable' and len(d['reasons']) == 1 and d['reasons'][0].startswith('matrix_L1/s43:')
    assert 'above the step-0 dev NLL' in d['reasons'][0]
    assert 'STOP for the orchestrator' in (tmp_path / 'out' / 's1a-selection.md').read_text()

    def not_finite(arm, seed, rec):                            # a run whose training had a non-finite loss or gradient
        if (arm, seed) == ('soft_slot_L1', 17):
            rec['training']['all_finite'] = False
    d = run_dev(tmp_path, out='b', mutate=not_finite)
    assert d['outcome'] == 'stop_unstable' and d['reasons'] == ['soft_slot_L1/s17: training.all_finite is false']

    def nothing_finite(arm, seed, rec):                        # every selectable dev NLL is non-finite: no selectable row at all
        if (arm, seed) == ('matrix_L1', 17):
            return dev_record(arm, seed, [2.0] + [float('nan')] * 10, COUNTS, selected_step=None)
    d = run_dev(tmp_path, out='c', mutate=nothing_finite)
    assert d['outcome'] == 'stop_unstable' and d['differences'] is None
    assert d['reasons'] == ['matrix_L1/s17: no finite selectable dev row']
    assert d['runs']['matrix_L1/s17']['selected_step'] is None

    def several(arm, seed, rec):
        if arm == 'soft_slot_L1':
            rec['training']['all_finite'] = False
    d = run_dev(tmp_path, out='d', mutate=several)
    assert len(d['reasons']) == 3 and all(x.startswith('soft_slot_L1/') for x in d['reasons'])


def test_a_selected_dev_nll_equal_to_the_step_0_nll_is_stable_and_one_above_it_is_not(tmp_path):
    def curve(value):
        def mutate(arm, seed, rec):
            if (arm, seed) == ('matrix_L1', 17):
                return dev_record(arm, seed, [1.0, value] + [5.0] * 9, [600] + [650] * 10, selected_step=100)
        return mutate
    d = run_dev(tmp_path, mutate=curve(1.0))                       # stable means selected NLL <= step-0 NLL
    assert d['reasons'] == [] and d['outcome'] != 'stop_unstable' and d['runs']['matrix_L1/s17']['selected_step'] == 100
    d = run_dev(tmp_path, out='above', mutate=curve(1.0000001))
    assert d['outcome'] == 'stop_unstable' and len(d['reasons']) == 1 and 'above the step-0 dev NLL' in d['reasons'][0]


def test_a_non_finite_row_that_is_not_selected_does_not_by_itself_stop(tmp_path):
    def one_bad_row(arm, seed, rec):
        if (arm, seed) == ('matrix_L1', 29):
            rec['dev'][3]['nll'] = float('inf')                # excluded from the selection; the others are unchanged
    d = run_dev(tmp_path, mutate=one_bad_row)
    assert d['outcome'] == 'no_trigger' and d['runs']['matrix_L1/s29']['dev'][3]['nll'] is None


def test_dev_refuses_incomplete_missing_or_mismatched_records(tmp_path):
    def attempt(name, match, mutate=None, drop=None):
        root = dev_tree(tmp_path / name, mutate=mutate)
        if drop:
            (root / drop / 'record.json').unlink()
        refused(match, '--stage', 'S1A-DEV', '--records', root, '--out', tmp_path / (name + '-out'))

    def status(arm, seed, rec):
        if (arm, seed) == ('matrix_L1', 17):
            rec['status'] = 'running'
    attempt('status', 'matrix_L1/s17/record.json has status', status)

    def failed(arm, seed, rec):                                   # the runner's record of a crashed run: the first line of the reason
        if (arm, seed) == ('soft_slot_L1', 29):
            rec.update(status='failed', failure_reason='FloatingPointError: non-finite logged value at step 12\nTraceback (most recent call last)')
    attempt('failed', r"soft_slot_L1/s29/record.json has status 'failed', not complete \(FloatingPointError: non-finite logged value at step 12\)", failed)
    attempt('missing', 'cannot read soft_slot_L1/s43/record.json', drop='soft_slot_L1/s43')

    def ids(arm, seed, rec):
        if (arm, seed) == ('soft_slot_L1', 43):
            rec['dev_ids_blake2b_128'] = 'ee' * 16
    attempt('ids', 'differ in dev_ids_blake2b_128', ids)

    def other_n(arm, seed, rec):
        if (arm, seed) == ('soft_slot_L1', 43):
            for row in rec['dev']:
                row['n'] = 999
    attempt('n', 'differ in dev n', other_n)

    def other_steps(arm, seed, rec):
        if (arm, seed) == ('soft_slot_L1', 43):
            return dev_record(arm, seed, FALLING, COUNTS, steps=2000, selected_step=2000)
    attempt('steps', 'differ in steps', other_steps)

    def label(arm, seed, rec):
        if (arm, seed) == ('soft_slot_L1', 29):
            rec['seed'] = 17                                  # a duplicate of another run's label
    attempt('label', 'carries another arm or seed', label)

    def rows(arm, seed, rec):
        if (arm, seed) == ('matrix_L1', 29):
            rec['dev'].pop()
    attempt('rows', 'dev rows are not step 0 and', rows)

    def start(arm, seed, rec):
        if (arm, seed) == ('matrix_L1', 29):
            rec['dev'][0]['nll'] = float('nan')
    attempt('start', 'step-0 dev NLL is not finite', start)

    def digest(arm, seed, rec):
        if (arm, seed) == ('matrix_L1', 29):
            rec['selected_checkpoint_blake2b_128'] = 'xyz'
    attempt('digest', 'not 32 hex digits', digest)

    def flag(arm, seed, rec):
        if (arm, seed) == ('matrix_L1', 29):
            rec['training']['all_finite'] = 'yes'
    attempt('flag', 'all_finite is not a boolean', flag)


# ----------------------------------------------------------------------------------------------------------------- S1A-ABL

def ablation_tree(root, gap=30, n=1000):
    dev_tree(root)
    write_manifest(root, dev_n=n)
    for path in root.glob('*/s*/record.json'):
        rec = json.loads(path.read_text())
        for row in rec['dev']:
            row.update(n=n, correct=10)
        path.write_text(json.dumps(rec))
    for arm in slot.ARMS:
        # The last row has the largest EM, but dev NLL selects step 100, earlier on a micro-nat tie.
        rec = dev_record(arm, 17, [2.0, 1.0, 0.9999995] + [1.2] * 8,
                         [10, 10 + (gap if arm == 'soft_slot_L1' else 0)] + [n] * 9,
                         selected_step=100, ablation=True)
        for row in rec['dev']:
            row['n'] = n
        path = root / 'ablation' / arm / 's17' / 'record.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))
    return root


@pytest.mark.parametrize('n,gap,flag', [(1000, 30, True), (1000, 29, False), (101, 3, False), (101, 4, True), (1000, -1, False)])
def test_ablation_uses_nll_selected_checkpoint_and_exact_three_point_boundary(tmp_path, capsys, n, gap, flag):
    root = ablation_tree(tmp_path / 'tree', gap, n)
    run('--stage', 'S1A-ABL', '--records', root, '--out', tmp_path / 'out')
    result = json.loads((tmp_path / 'out' / 's1a-ablation.json').read_text())
    assert result == dict(stage='S1A-ABL', n=n, correct=dict(matrix_L1=10, soft_slot_L1=10 + gap), flag=flag)
    assert capsys.readouterr().out.splitlines() == [f'n {n}', 'matrix_L1 correct 10', f'soft_slot_L1 correct {10 + gap}', f'flag {str(flag).lower()}']
    index = read_index(tmp_path / 'out')
    assert len(index['inputs']) == 9  # six main records, two ablation records, locked manifest


@pytest.mark.parametrize('value', [True, None, 0, 'false'])
def test_six_main_records_require_the_exact_false_ablation_flag(tmp_path, value):
    root = dev_tree(tmp_path / 'tree', mutate=lambda arm, seed, rec: rec.update(ablation=value))
    refused('ablation must be False', '--stage', 'S1A-DEV', '--records', root, '--out', tmp_path / 'out')
    root = ablation_tree(tmp_path / 'abl')
    path = root / 'matrix_L1' / 's17' / 'record.json'
    rec = json.loads(path.read_text())
    rec.pop('ablation')
    path.write_text(json.dumps(rec))
    refused("lacks 'ablation'", '--stage', 'S1A-ABL', '--records', root, '--out', tmp_path / 'abl-out')


@pytest.mark.parametrize('failure', ['missing_main', 'missing_ablation', 'failed', 'flag', 'ids', 'n', 'selection'])
def test_ablation_refuses_incomplete_mislabelled_mismatched_or_wrong_selected_inputs(tmp_path, failure):
    root = ablation_tree(tmp_path / 'tree')
    path = root / 'ablation' / 'soft_slot_L1' / 's17' / 'record.json'
    rec = json.loads(path.read_text())
    if failure == 'missing_main':
        (root / 'matrix_L1' / 's29' / 'record.json').unlink()
        match = 'cannot read matrix_L1/s29/record.json'
    elif failure == 'missing_ablation':
        path.unlink()
        match = 'cannot read ablation/soft_slot_L1/s17/record.json'
    else:
        if failure == 'failed':
            rec['status'] = 'failed'
            match = "not complete"
        elif failure == 'flag':
            rec['ablation'] = False
            match = 'ablation must be True'
        elif failure == 'ids':
            rec['dev_ids_blake2b_128'] = '0' * 32
            match = 'S1A/dev: ids digest differs'
        elif failure == 'n':
            for row in rec['dev']:
                row['n'] = 1001
            match = 'S1A/dev: n .* differs'
        else:
            rec['selected_step'] = 200
            match = 'selected_step differs'
        path.write_text(json.dumps(rec))
    refused(match, '--stage', 'S1A-ABL', '--records', root, '--out', tmp_path / 'out')


# ---------------------------------------------------------------------------------------------------------------------- S1A

def correctness(n, counts, shift=0):
    """counts: seed -> (both correct, only soft correct, only matrix correct); episodes are rolled by `shift` per seed index."""
    out = {}
    for i, (seed, (both, wins, losses)) in enumerate(counts.items()):
        mat, soft = np.zeros(n, bool), np.zeros(n, bool)
        mat[:both] = soft[:both] = True
        soft[both:both + wins] = True
        mat[both + wins:both + wins + losses] = True
        out[f'matrix_L1_s{seed}'], out[f'soft_slot_L1_s{seed}'] = np.roll(mat, shift * i), np.roll(soft, shift * i)
    return out


def write_set(directory, ids, correct, rel_err=True, seed=5):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for label, c in correct.items():
        arrays = dict(episode_id=ids, correct=c, nll=rng.uniform(0.5, 2.0, len(ids)))
        if rel_err:
            arrays['rel_err'] = rng.uniform(0.0, 1.0, len(ids))
        np.savez(directory / f'{label}.npz', **arrays)


def s1a_tree(root, n, counts, shift=0, unseen=None, references=True):
    """confirmation/ and unseen/ with the six labels (+ the three references); returns the confirmation correctness arrays."""
    write_manifest(root, s1a_n=n)
    confirmation = correctness(n, counts, shift)
    extras = {label: np.random.default_rng(i).random(n) < 0.6 for i, label in enumerate(slot.REFERENCES)} if references else {}
    write_set(root / 'confirmation', np.arange(n, dtype=np.int64) + 1000, dict(confirmation, **extras))
    write_set(root / 'unseen', np.arange(n, dtype=np.int64) + 10 ** 6, dict(correctness(n, unseen or counts, shift), **extras), seed=6)
    return confirmation


def run_s1a(tmp_path, n, counts, out='out', name='s1a', **kw):
    root = tmp_path / name
    arrays = s1a_tree(root, n, counts, **kw)
    run('--stage', 'S1A', '--records', root, '--out', tmp_path / out)
    return json.loads((tmp_path / out / 'summary.json').read_text()), arrays


ADVANCE = {17: (200, 14, 0), 29: (200, 12, 0), 43: (200, 10, 0)}        # n = 400: 36 correct more in total = exactly +3.0 pp


def test_s1a_advances_at_exactly_three_points_with_every_seed_positive_and_the_lower_end_above_zero(tmp_path):
    s, _ = run_s1a(tmp_path, 400, ADVANCE, shift=37)
    c = s['confirmation']['contrast']
    assert c['sum_delta_correct'] == 36 and c['delta_pp'] == 3.0 and c['sum_delta_correct'] * 100 == 9 * 400
    assert s['verdict'] == 'ADVANCE' and s['tests'] == dict(delta_at_least_3pp=True, lower_end_positive=True, seeds_all_positive=True)
    assert c['lower_pp'] > 0 and c['lower_pp'] < 3.0 < c['upper_pp'] and c['lower_end_positive'] is True
    assert [c['seeds'][k]['delta_correct'] for k in ('17', '29', '43')] == [14, 12, 10]
    assert s['unseen']['contrast']['sum_delta_correct'] == 36 and 'tests' not in s['unseen']
    # the registered size: 2,000 confirmation episodes, +3.0 pp is 180 correct more over three seeds
    s, _ = run_s1a(tmp_path, 2000, {17: (1000, 50, 0), 29: (1000, 60, 0), 43: (1000, 70, 0)}, out='full', name='full', shift=5)
    assert s['verdict'] == 'ADVANCE' and s['confirmation']['n'] == s['confirmation']['expected_n'] == 2000
    assert s['confirmation']['contrast']['delta_pp'] == 3.0


def test_s1a_does_not_advance_one_correct_answer_below_the_pooled_boundary(tmp_path):
    s, _ = run_s1a(tmp_path, 400, {17: (200, 12, 0), 29: (200, 12, 0), 43: (200, 11, 0)}, shift=37)
    c = s['confirmation']['contrast']
    assert c['sum_delta_correct'] == 35 and c['delta_pp'] < 3.0 and c['lower_pp'] > 0
    assert s['verdict'] == 'NOT ADVANCED' and s['tests'] == dict(delta_at_least_3pp=False, lower_end_positive=True, seeds_all_positive=True)


def test_s1a_does_not_advance_when_one_seed_is_not_positive(tmp_path):
    for third, wins in ((0, 0), (-1, 0)):
        counts = {17: (200, 18 + (1 if third else 0), 0), 29: (200, 18, 0), 43: (200, wins, 1 if third else 0)}
        s, _ = run_s1a(tmp_path, 400, counts, out=f'o{third}', name=f's{third}', shift=37)
        c = s['confirmation']['contrast']
        assert c['sum_delta_correct'] == 36 and c['lower_pp'] > 0                 # pooled gain and interval would pass on their own
        assert c['seeds']['43']['delta_correct'] == third
        assert s['verdict'] == 'NOT ADVANCED' and s['tests']['seeds_all_positive'] is False


def test_s1a_does_not_advance_when_the_lower_end_is_not_above_zero(tmp_path):
    # +3.0 pp pooled and every seed positive, but the same 180 wins against 168 losses in every seed: the interval reaches below zero
    counts = {s: (0, 180, 168) for s in (17, 29, 43)}
    s, _ = run_s1a(tmp_path, 400, counts)
    c = s['confirmation']['contrast']
    assert c['sum_delta_correct'] == 36 and all(v['delta_correct'] == 12 for v in c['seeds'].values())
    assert c['lower_pp'] < 0 < c['upper_pp'] and c['lower_end_positive'] is False
    assert s['tests'] == dict(delta_at_least_3pp=True, lower_end_positive=False, seeds_all_positive=True) and s['verdict'] == 'NOT ADVANCED'


def test_s1a_does_not_advance_when_the_lower_end_is_exactly_zero(tmp_path):
    # the same 25 wins and 13 losses in every seed: the pooled resample sums are multiples of 3 and the registered bootstrap puts the
    # 2.5th percentile exactly at 0, so L > 0 is strict: +3.0 pp and three positive seeds are not enough
    s, _ = run_s1a(tmp_path, 400, {seed: (0, 25, 13) for seed in (17, 29, 43)})
    c = s['confirmation']['contrast']
    assert c['sum_delta_correct'] == 36 and c['delta_pp'] == 3.0 and c['lower_pp'] == 0.0 and c['lower_end_positive'] is False
    assert s['tests'] == dict(delta_at_least_3pp=True, lower_end_positive=False, seeds_all_positive=True) and s['verdict'] == 'NOT ADVANCED'


def test_the_verdict_comes_from_the_confirmation_set_only(tmp_path):
    s, _ = run_s1a(tmp_path, 400, ADVANCE, shift=37, unseen={17: (0, 0, 200), 29: (0, 0, 200), 43: (0, 0, 200)})
    assert s['verdict'] == 'ADVANCE' and s['unseen']['contrast']['delta_pp'] < 0
    s, _ = run_s1a(tmp_path, 400, {17: (200, 5, 0), 29: (200, 5, 0), 43: (200, 5, 0)}, out='b', name='b', shift=37,
                   unseen={17: (0, 200, 0), 29: (0, 200, 0), 43: (0, 200, 0)})
    assert s['verdict'] == 'NOT ADVANCED' and s['unseen']['contrast']['delta_pp'] > 3.0


def test_the_bootstrap_is_the_registered_one_and_reproduces_with_numpy(tmp_path):
    n = 400
    s, arrays = run_s1a(tmp_path, n, ADVANCE, shift=37)
    seed = 9056755664197379988
    diffs = {k: arrays[f'soft_slot_L1_s{k}'].astype(float) - arrays[f'matrix_L1_s{k}'] for k in (17, 29, 43)}
    d = sum(diffs.values()) / 3                                              # d_i = (1/3) sum_s (c_soft - c_matrix)
    low, high = numpy_interval_pp(d, n, seed)
    c = s['confirmation']['contrast']
    assert c['delta_pp'] == pytest.approx(100 * d.mean(), abs=1e-12)
    assert (c['lower_pp'], c['upper_pp']) == pytest.approx((low, high), abs=1e-9)
    for k, v in diffs.items():                                               # the same index matrix for every seed
        low, high = numpy_interval_pp(v, n, seed)
        assert (c['seeds'][str(k)]['lower_pp'], c['seeds'][str(k)]['upper_pp']) == pytest.approx((low, high), abs=1e-9)
        assert c['seeds'][str(k)]['delta_pp'] == pytest.approx(100 * v.mean(), abs=1e-12)
    assert s['bootstrap'] == dict(seed=seed, resamples=10000, block=500, interval='percentile 95 %')
    # the sums themselves: 20 sequential blocks of rng.integers(0, n, size=(500, n)), sorted
    rng = np.random.default_rng(seed)
    v = np.arange(n) % 3 - 1
    expected = np.sort(np.concatenate([v[rng.integers(0, n, size=(500, n))].sum(axis=1) for _ in range(20)])).tolist()
    assert slot.resample_sums({'v': v.astype(np.int64)}, n, seed)['v'] == expected
    # the blocks only save memory: they are the rows of one 10,000-resample draw, which an independent re-implementation may use
    one = np.sort(v[np.random.default_rng(seed).integers(0, n, size=(10000, n))].sum(axis=1)).tolist()
    assert slot.resample_sums({'v': v.astype(np.int64)}, n, seed)['v'] == one


def test_the_interval_end_is_exact_rational_arithmetic():
    from fractions import Fraction
    sums = [-39] * 250 + [1] * 9750                      # linear interpolation at 2.5 %: -39 + 0.975 * 40 is exactly 0
    low, high = slot.interval(sums)
    assert low == 0 and not low > 0 and high == 1 and isinstance(low, Fraction)
    ramp = list(range(10000))
    low, high = slot.interval(ramp)
    assert low == Fraction(249975, 1000) and high == Fraction(9749025, 1000)
    assert float(low) == pytest.approx(np.percentile(ramp, 2.5), abs=1e-9) and float(high) == pytest.approx(np.percentile(ramp, 97.5), abs=1e-9)


def test_s1a_outputs_are_byte_identical_between_runs_and_carry_the_reference_rows(tmp_path):
    root = tmp_path / 'tree'
    s1a_tree(root, 400, ADVANCE, shift=37)
    for out in ('a', 'b'):
        run('--stage', 'S1A', '--records', root, '--out', tmp_path / out)
    for name in ('summary.json', 'verdict.md', 'source_index.json'):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()
    s = json.loads((tmp_path / 'a' / 'summary.json').read_text())
    assert sorted(s['confirmation']['references']) == sorted(slot.REFERENCES) and sorted(s['unseen']['references']) == sorted(slot.REFERENCES)
    arm = s['confirmation']['arms']['soft_slot_L1']
    assert sorted(arm['seeds']) == ['17', '29', '43'] and arm['em'] == pytest.approx(sum(v['correct'] for v in arm['seeds'].values()) / 1200)
    assert arm['nll'] is not None and arm['rel_err'] is not None
    md = (tmp_path / 'a' / 'verdict.md').read_text()
    assert md.startswith('# S1A verdict: ADVANCE') and 's_re_native' in md and 'descriptive, no verdict' in md
    # All three references are mandatory for each opened set.
    plain = tmp_path / 'plain'
    s1a_tree(plain, 400, ADVANCE, shift=37, references=False)
    refused('confirmation/s_re_native.npz is missing', '--stage', 'S1A', '--records', plain, '--out', tmp_path / 'c')


def test_s1a_stdout_is_blinded(tmp_path, capsys):
    root = tmp_path / 'tree'
    s1a_tree(root, 400, ADVANCE, shift=37)
    run('--stage', 'S1A', '--records', root, '--out', tmp_path / 'out')
    out = capsys.readouterr().out
    s = json.loads((tmp_path / 'out' / 'summary.json').read_text())
    c = s['confirmation']['contrast']
    assert out.split('\n')[0] == 'stage S1A' and 'verdict ADVANCE' in out
    for name in ('summary.json', 'verdict.md', 'source_index.json'):
        assert re.search(rf'^{re.escape(name)} {blake((tmp_path / "out" / name).read_bytes())}$', out, re.M)
    assert not re.search(r'\d+\.\d+', out)                          # no decimal number: no metric, no interval, no EM
    for value in (c['delta_pp'], c['lower_pp'], c['upper_pp'], c['seeds']['17']['delta_pp'], c['seeds']['17']['lower_pp']):
        assert repr(value) not in out and f'{value:.1f}' not in out and f'{value:.3f}' not in out
    assert 'confirmation n 400 expected 400 files 9' in out         # matches this synthetic locked manifest; wrong n is refused below


def test_unseen_is_optional_only_when_all_its_files_are_absent(tmp_path, capsys):
    root = tmp_path / 'tree'
    s1a_tree(root, 400, ADVANCE)
    for path in (root / 'unseen').glob('*.npz'):
        path.unlink()
    run('--stage', 'S1A', '--records', root, '--out', tmp_path / 'out')
    result = json.loads((tmp_path / 'out' / 'summary.json').read_text())
    assert result['unseen'] == dict(status='not run') and result['verdict'] == 'ADVANCE'
    assert 'unseen not run' in capsys.readouterr().out
    assert 'not run' in (tmp_path / 'out' / 'verdict.md').read_text()
    write_set(root / 'unseen', np.arange(400, dtype=np.int64) + 10 ** 6, {'s_re_native': np.zeros(400, bool)})
    refused('unseen/matrix_L1_s17.npz is missing', '--stage', 'S1A', '--records', root, '--out', tmp_path / 'partial')


@pytest.mark.parametrize('name', ['confirmation', 'unseen'])
@pytest.mark.parametrize('label', slot.REFERENCES)
def test_each_reference_is_required_for_each_opened_set(tmp_path, name, label):
    root = tmp_path / 'tree'
    s1a_tree(root, 400, ADVANCE)
    (root / name / f'{label}.npz').unlink()
    refused(f'{name}/{label}.npz is missing', '--stage', 'S1A', '--records', root, '--out', tmp_path / 'out')


@pytest.mark.parametrize('name', ['S0/calibration', 'S0/dev', 'S1A/dev', 'S1A/confirmation', 'S1A/unseen'])
@pytest.mark.parametrize('field', ['count', 'ids_blake2b_128'])
def test_each_analysis_input_set_must_match_the_independent_locked_count_and_ids(tmp_path, name, field):
    root = tmp_path / 'tree'
    if name.startswith('S0/'):
        write_s0(root, s0_metrics())
        argv = ['--stage', 'S0', '--preflight', root / 'preflight.json']
    elif name == 'S1A/dev':
        dev_tree(root)
        argv = ['--stage', 'S1A-DEV']
    else:
        s1a_tree(root, 400, ADVANCE)
        argv = ['--stage', 'S1A']
    path = root / 'manifest.lock.json'
    locked = json.loads(path.read_text())
    ids = locked['manifest']['ids'][f'eval/{name}']
    if field == 'count':
        ids['count'] += 1
        locked['manifest']['sets'][name]['count'] += 1
        match = f'{name}: n .* differs from locked manifest count'
    else:
        ids[field] = '0' * 32
        match = f'{name}: ids digest differs from locked manifest'
    path.write_text(json.dumps(locked))
    refused(match, *argv, '--records', root, '--out', tmp_path / 'out')


def test_analysis_requires_a_locked_manifest_and_rejects_pending_or_inconsistent_locks(tmp_path):
    root = dev_tree(tmp_path / 'tree')
    argv = ['--stage', 'S1A-DEV', '--records', str(root), '--out', str(tmp_path / 'out')]
    with pytest.raises(SystemExit):
        slot.main(argv)  # --manifest has no default
    path = root / 'manifest.lock.json'
    locked = json.loads(path.read_text())
    locked['status'] = 'pending'
    path.write_text(json.dumps(locked))
    refused('not a version 1 locked', *argv)
    locked['status'] = 'locked'
    locked['manifest']['ids']['eval/S1A/dev']['count'] += 1
    path.write_text(json.dumps(locked))
    refused('inconsistent or empty count', *argv)


def test_s1a_refuses_missing_labels_mismatched_ids_and_bad_arrays(tmp_path):
    def tree(name, **kw):
        root = tmp_path / name
        s1a_tree(root, 400, ADVANCE, shift=37, **kw)
        return root

    def attempt(root, match):
        refused(match, '--stage', 'S1A', '--records', root, '--out', tmp_path / (root.name + '-out'))

    root = tree('missing-confirmation')
    (root / 'confirmation' / 'soft_slot_L1_s43.npz').unlink()
    attempt(root, 'confirmation/soft_slot_L1_s43.npz is missing')
    root = tree('missing-unseen')
    (root / 'unseen' / 'matrix_L1_s17.npz').unlink()
    attempt(root, 'unseen/matrix_L1_s17.npz is missing')
    root = tree('ids')                                              # one file of the confirmation set has other episode ids
    z = dict(np.load(root / 'confirmation' / 'matrix_L1_s29.npz'))
    z['episode_id'] = z['episode_id'].copy()
    z['episode_id'][[0, 1]] = z['episode_id'][[1, 0]]
    np.savez(root / 'confirmation' / 'matrix_L1_s29.npz', **z)
    attempt(root, 'confirmation/matrix_L1_s29.npz: episode_id differs')
    root = tree('ids-reference')                                    # a reference file must share the set's ids as well
    z = dict(np.load(root / 'unseen' / 's_re_native.npz'))
    z['episode_id'] = z['episode_id'] + 1
    np.savez(root / 'unseen' / 's_re_native.npz', **z)
    attempt(root, 'unseen/s_re_native.npz: episode_id differs')
    root = tree('length')
    z = {k: v[:-1] for k, v in np.load(root / 'confirmation' / 'soft_slot_L1_s17.npz').items()}
    np.savez(root / 'confirmation' / 'soft_slot_L1_s17.npz', **z)
    attempt(root, 'n differs')
    root = tree('overlap')                                          # the two sets must be different episodes
    for p in (root / 'unseen').glob('*.npz'):
        z = dict(np.load(p))
        z['episode_id'] = np.arange(400, dtype=np.int64) + 1000
        np.savez(p, **z)
    attempt(root, 'share episode ids')
    root = tree('nll')
    z = dict(np.load(root / 'confirmation' / 'soft_slot_L1_s17.npz'))
    z['nll'][3] = np.nan
    np.savez(root / 'confirmation' / 'soft_slot_L1_s17.npz', **z)
    attempt(root, 'nll is not a 1-D finite float array')
    root = tree('dtype')
    z = dict(np.load(root / 'confirmation' / 'soft_slot_L1_s17.npz'))
    z['episode_id'] = z['episode_id'].astype(np.int32)
    np.savez(root / 'confirmation' / 'soft_slot_L1_s17.npz', **z)
    attempt(root, 'episode_id is not a 1-D int64 array')
    root = tree('correct')
    z = dict(np.load(root / 'confirmation' / 'soft_slot_L1_s17.npz'))
    z['correct'] = z['correct'].astype(float)
    np.savez(root / 'confirmation' / 'soft_slot_L1_s17.npz', **z)
    attempt(root, 'correct is not a 1-D bool array')
    root = tree('duplicate-ids')
    for p in (root / 'confirmation').glob('*.npz'):
        z = dict(np.load(p))
        z['episode_id'][1] = z['episode_id'][0]
        np.savez(p, **z)
    attempt(root, 'duplicate episode ids')
    root = tree('garbage')
    (root / 'confirmation' / 'matrix_L1_s17.npz').write_bytes(b'not an archive')
    attempt(root, 'not a readable npz')


def test_the_stage_and_preflight_arguments_are_checked(tmp_path):
    refused('--preflight belongs to stage S0', '--stage', 'S1A', '--records', tmp_path, '--preflight', tmp_path / 'p.json', '--out', tmp_path / 'o')
    refused('--preflight belongs to stage S0', '--stage', 'S0', '--records', tmp_path, '--out', tmp_path / 'o')
    with pytest.raises(SystemExit):
        slot.main(['--stage', 'S2', '--records', str(tmp_path), '--out', str(tmp_path / 'o')])


# ----------------------------------------------------------------------------------------------------------- source index

def read_index(directory):
    return json.loads((directory / 'source_index.json').read_text())


def test_source_index_lists_the_script_and_every_input_digest(tmp_path):
    expected_script = blake(SCRIPT.read_bytes())
    # S1A: every file read, relative to --records
    root = tmp_path / 's1a'
    s1a_tree(root, 400, ADVANCE, shift=37)
    run('--stage', 'S1A', '--records', root, '--out', tmp_path / 'o1')
    index = read_index(tmp_path / 'o1')
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*.npz'))
    assert index['stage'] == 'S1A' and index['script'] == 'analyze_slot_study.py' and index['script_blake2b_128'] == expected_script
    assert sorted(index['inputs']) == sorted(files + [str((root / 'manifest.lock.json').resolve())]) and len(files) == 18
    assert index['manifest_blake2b_128'] == blake((root / 'manifest.lock.json').read_bytes())
    assert all(index['inputs'][name] == blake((root / name).read_bytes()) for name in files)
    # S0: metrics and samples relative, the preflight absolute
    root = write_s0(tmp_path / 's0', s0_metrics())
    run('--stage', 'S0', '--records', root, '--preflight', root / 'preflight.json', '--out', tmp_path / 'o0')
    index = read_index(tmp_path / 'o0')
    assert sorted(index['inputs']) == sorted(['metrics.json', 'samples.npz', str((root / 'preflight.json').resolve()), str((root / 'manifest.lock.json').resolve())])
    assert index['inputs']['metrics.json'] == blake((root / 'metrics.json').read_bytes())
    assert index['inputs']['samples.npz'] == blake((root / 'samples.npz').read_bytes())
    assert index['inputs'][str((root / 'preflight.json').resolve())] == blake((root / 'preflight.json').read_bytes())
    assert index['script_blake2b_128'] == expected_script
    # S1A-DEV: the six records
    root = dev_tree(tmp_path / 'dev')
    run('--stage', 'S1A-DEV', '--records', root, '--out', tmp_path / 'o2')
    index = read_index(tmp_path / 'o2')
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob('record.json'))
    assert sorted(index['inputs']) == sorted(names + [str((root / 'manifest.lock.json').resolve())]) and len(names) == 6 and index['stage'] == 'S1A-DEV'
    assert all(index['inputs'][name] == blake((root / name).read_bytes()) for name in names)


def test_the_script_is_frozen_standard_library_plus_numpy_and_blake2b_only():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module.split('.')[0])
    assert modules <= set(sys.stdlib_module_names) | {'numpy'} and 'numpy' in modules
    assert not modules & {'looplora', 'datetime', 'time', 'random', 'os', 'socket', 'platform', 'subprocess'}
    assert set(re.findall(r'hashlib\.(\w+)', source)) == {'blake2b'} and 'digest_size=16' in source
    assert 'looplora' not in source and '__import__' not in source
    assert not re.search(r'[^\x00-\x7f]', source)
