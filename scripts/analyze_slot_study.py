"""Frozen analysis of SLOT-LOOP-v1 (docs/prereg/SLOT-LOOP-v1.md): the S0 decision, the S1A dev selection with the section-7.4
trigger, and the S1A confirmation verdict. Standard library and numpy only; no clock, no environment: outputs are byte-deterministic.

usage: python scripts/analyze_slot_study.py --stage {S0,S1A-DEV,S1A} --records DIR --out DIR [--preflight FILE]
  S0       DIR/metrics.json, DIR/samples.npz and --preflight FILE  -> decision_s0.json, decision_s0.md
  S1A-DEV  DIR/ARM/sSEED/record.json (2 arms x 3 seeds)            -> s1a-selection.json, s1a-selection.md
  S1A      DIR/confirmation/LABEL.npz, DIR/unseen/LABEL.npz        -> summary.json, verdict.md (stdout: counts, names, digests, verdict)
Every stage also writes source_index.json (blake2b-128 of each input file and of this script). A defect in the inputs ends the run
with SystemExit('refused: ...'); data are never repaired. Thresholds are integer comparisons; ties and interval ends are exact.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

BOOTSTRAP = {'S0': 2079281455337892545, 'S1A': 9056755664197379988}   # numpy default_rng seeds
BLOCKS, BLOCK = 20, 500                                                # 10,000 resamples: 20 sequential blocks of 500 index rows
AV, FO = 'all_valid_question', 'final_only'
CHECKS = ('native_replay', 'matrix_equals_native', 'owner_mask', 'isolation', 'frozen_set', 'bytes', 'capacity',
          'feature_consistency', 'd128_3_replay')
GAINS, TAUS, KINDS = (0.5, 1.0, 2.0), (0.03, 0.1, 0.3, 1.0), ('matrix', 'soft', 'hard')
CALIBRATION_TIE, SELECTION_TIE = Fraction(1, 10 ** 4), Fraction(1, 10 ** 6)   # nats
ROLES = ('native', 'matrix_uncal', 'matrix_cal', 'soft_uncal', 'soft_cal', 'hard_uncal', 'hard_cal', 'soft_rec_uncal', 'soft_rec_cal')
CONDITIONS = tuple((r, AV) for r in ROLES) + tuple((r, FO) for r in ('matrix_cal', 'soft_cal', 'hard_cal', 'clean_final'))
COMPARED = tuple(c for c in CONDITIONS if c[0] != 'matrix_cal')   # each against matrix_cal at its own scope
D128_ROLES = ('native', 'matrix', 'soft', 'hard')
OPTIONAL = ('w_true', 'w_gap', 'w_entropy', 'slot_is_true', 'clean_value_correct', 'read_value_correct', 'seconds')
OLD_RANGE = dict(correct=608, n=1000, loss_float32=2.1291537284851074)
ARMS, SEEDS = ('matrix_L1', 'soft_slot_L1'), (17, 29, 43)
REFERENCES = ('s_re_native', 'matrix_step0', 'soft_step0')
REQUIRED = tuple(f'{a}_s{s}' for a in ARMS for s in SEEDS)
EXPECTED_N = 2000
LABELS = {
    'R0': 'stop, fix the interface, no comparison, no S1A',
    'R1': 'full adaptation',
    'R2': 'full adaptation; soft mixture or temperature suspected, HardSlot stays diagnostic',
    'R3': 'limited adaptation; final_only reported separately; a read-position protocol study is flagged (not authorized here)',
    'R4': 'limited adaptation; addressing, query or role interface flagged as the main bottleneck (S2 priority)',
    'R5': 'limited adaptation: one pre-budgeted pass',
}


# ------------------------------------------------------------------------------------------------------- inputs and checks

def refuse(message):
    raise SystemExit('refused: ' + message)


def digest(data):
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def dumps(value):
    try:
        return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    except ValueError:
        refuse('a non-finite value reached an output file')


def unique(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate key ' + repr(next(k for k in keys if keys.count(k) > 1)))
    return dict(pairs)


class Sources:
    """The input files of one run: each is read once, as bytes, and its blake2b-128 is kept for source_index.json."""

    def __init__(self, root):
        self.root, self.digests = Path(root), {}

    def read(self, path, absolute=False):
        full = Path(path).resolve() if absolute else self.root / path
        try:
            data = full.read_bytes()
        except OSError:
            refuse(f'cannot read {path}')
        self.digests[str(full) if absolute else Path(path).as_posix()] = digest(data)
        return data

    def exists(self, path):
        return (self.root / path).is_file()

    def json(self, path, absolute=False):
        try:
            value = json.loads(self.read(path, absolute), object_pairs_hook=unique)
        except ValueError as error:
            refuse(f'{path} is not valid JSON ({error})')
        if not isinstance(value, dict):
            refuse(f'{path} is not a JSON object')
        return value

    def npz(self, path):
        try:
            with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as z:
                return {name: z[name] for name in z.files}
        except Exception as error:
            refuse(f'{path} is not a readable npz ({type(error).__name__})')

    def index(self, stage):
        script = Path(__file__)
        return dict(stage=stage, script=script.name, script_blake2b_128=digest(script.read_bytes()), inputs=dict(self.digests))


def get(record, key, where):
    if not isinstance(record, dict) or key not in record:
        refuse(f'{where} lacks {key!r}')
    return record[key]


def real(x, where):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        refuse(f'{where} is not a finite number')
    return x


def whole(x, where, high=None):
    if isinstance(x, bool) or not isinstance(x, int) or x < 0 or (high is not None and x > high):
        refuse(f'{where} is not an integer in [0, {high}]')
    return x


def flags(a, where):
    """A 1-D array of 0/1 as int64 (bool, or integers that are all 0 or 1)."""
    if a.ndim != 1 or a.dtype.kind not in 'biu' or (a.dtype.kind != 'b' and ((a < 0) | (a > 1)).any()):
        refuse(f'{where} is not a 1-D bool array')
    return a.astype(np.int64)


def floats(a, where):
    """A 1-D array of finite floats, as float64."""
    if a.ndim != 1 or a.dtype.kind != 'f' or not np.isfinite(a).all():
        refuse(f'{where} is not a 1-D finite float array')
    return a.astype(np.float64)


def mean(values):
    return math.fsum(values.tolist()) / len(values)


# ---------------------------------------------------------------------------------------------------------- bootstrap

def resample_sums(vectors, n, seed):
    """Sorted sums (Python ints) of each int64 vector over 10,000 paired bootstrap resamples of the episode index: one index matrix,
    drawn as 20 sequential blocks of rng.integers(0, n, size=(500, n)), shared by every vector."""
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in vectors}
    for _ in range(BLOCKS):
        index = rng.integers(0, n, size=(BLOCK, n))
        for name, v in vectors.items():
            parts[name].append(v[index].sum(axis=1))
    return {name: np.sort(np.concatenate(p)).tolist() for name, p in parts.items()}


def interval(sums):
    """The percentile 95 % interval of sorted integer sums: numpy's linear interpolation at 2.5 and 97.5, in exact arithmetic."""
    def at(q):
        h = (len(sums) - 1) * q
        i = h.numerator // h.denominator
        j = min(i + 1, len(sums) - 1)
        return sums[i] + (h - i) * (sums[j] - sums[i])
    return at(Fraction(1, 40)), at(Fraction(39, 40))


# ------------------------------------------------------------------------------------------------------------------ S0

def read_preflight(pre):
    """The engineering gate: the required checks that failed (a missing check or a non-boolean `passed` counts as failed)."""
    checks = pre.get('checks', {})
    if not isinstance(checks, dict):
        refuse('the preflight checks are not an object')
    ok = lambda name: isinstance(checks.get(name), dict) and checks[name].get('passed') is True
    passed = pre.get('passed')
    return dict(required={name: ok(name) for name in CHECKS}, failed=[name for name in CHECKS if not ok(name)],
                other_failed=sorted(name for name in checks if name not in CHECKS and not ok(name)),
                overall_passed=passed if isinstance(passed, bool) else None)


def candidates_in_grid_order(raw, n):
    """The 18 registered calibration candidates in grid order (matrix g, soft tau x g, hard g); anything else is refused."""
    grid = ([('matrix', g, None) for g in GAINS] + [('soft', g, t) for t in TAUS for g in GAINS] + [('hard', g, None) for g in GAINS])
    if not isinstance(raw, list):
        refuse('calibration candidates are not a list')
    found = {}
    for c in raw:
        key = (get(c, 'kind', 'a calibration candidate'), get(c, 'gain', 'a calibration candidate'),
               c.get('tau') if c['kind'] == 'soft' else None)
        if key not in grid:
            refuse(f'calibration candidate {key} is not in the registered grid')
        if key in found:
            refuse(f'calibration candidate {key} appears twice')
        found[key] = c
    if len(found) != len(grid):
        refuse(f'calibration has {len(found)} of the {len(grid)} registered candidates')
    out = []
    for key in grid:
        c, nll = found[key], found[key].get('nll')
        if nll is not None and (isinstance(nll, bool) or not isinstance(nll, (int, float))):
            refuse(f'calibration candidate {key} has a non-numeric NLL')
        out.append(dict(kind=key[0], gain=key[1], tau=key[2], nll=float(nll) if nll is not None and math.isfinite(nll) else None,
                        correct=whole(get(c, 'correct', f'calibration candidate {key}'), f'calibration candidate {key} correct', n),
                        seconds=real(c['seconds'], f'calibration candidate {key} seconds') if c.get('seconds') is not None else None))
    return out


def pick(candidates, kind):
    """The registered choice: lowest finite NLL; candidates within 1e-4 nats of it are tied; then the gain nearest 1 (|g - 1|), then
    the larger tau. Exact arithmetic on the recorded doubles."""
    pool = [c for c in candidates if c['kind'] == kind and c['nll'] is not None]
    if not pool:
        refuse(f'calibration has no finite {kind} candidate')
    low = min(Fraction(c['nll']) for c in pool)
    tied = [c for c in pool if Fraction(c['nll']) - low <= CALIBRATION_TIE]
    return min(tied, key=lambda c: (abs(Fraction(c['gain']) - 1), -Fraction(c['tau'] or 0)))


def parse_calibration(raw):
    """The calibration block with `selected` re-derived from the candidate list; a different `selected` is refused."""
    n = whole(get(raw, 'n', 'calibration'), 'calibration n')
    candidates = candidates_in_grid_order(get(raw, 'candidates', 'calibration'), n)
    chosen = get(raw, 'selected', 'calibration')
    selected = {}
    for kind in KINDS:
        best, given = pick(candidates, kind), get(chosen, kind, 'calibration.selected')
        if (get(given, 'gain', f'calibration.selected.{kind}') != best['gain']
                or (kind == 'soft' and get(given, 'tau', 'calibration.selected.soft') != best['tau'])):
            refuse(f'calibration: selected {kind} differs from the registered rule')
        selected[kind] = dict(gain=best['gain'], tau=best['tau']) if kind == 'soft' else dict(gain=best['gain'])
    return dict(n=n, ids_blake2b_128=raw.get('ids_blake2b_128'), candidates=candidates, selected=selected)


def parse_conditions(raw, n, where, wanted, selected):
    """Validated rows of the wanted (role, scope) pairs; their gain and tau must be the registered ones (g = 1, tau = 1 uncalibrated,
    the verified selection otherwise)."""
    one = dict(matrix=(1.0, None), soft=(1.0, 1.0), hard=(1.0, None))
    chosen = {k: (selected[k]['gain'], selected[k].get('tau')) for k in KINDS}
    parameters = {'soft_rec_uncal': one['soft'], 'soft_rec_cal': chosen['soft']}
    for kind in KINDS:
        parameters.update({f'{kind}_uncal': one[kind], f'{kind}_cal': chosen[kind], kind: chosen[kind]})
    out = {}
    for role, scope in wanted:
        key, label = f'{role}/{scope}', f'{where}[{role}/{scope}]'
        row = get(raw, key, where)
        if not isinstance(row, dict) or row.get('role', role) != role or row.get('scope', scope) != scope:
            refuse(f'{label} is not a condition row of that role and scope')
        if get(row, 'n', label) != n:
            refuse(f'{label} has a different n than the record')
        whole(get(row, 'correct', label), f'{label} correct', n)
        for field in ('nll', 'rel_err', 'cos', 'norm_ratio'):
            real(get(row, field, label), f'{label} {field}')
        for field in OPTIONAL:
            if row.get(field) is not None:
                real(row[field], f'{label} {field}')
        if role in parameters:
            gain, tau = parameters[role]
            if row.get('gain') != gain or (tau is not None and row.get('tau') != tau):
                refuse(f'{label} was not run at the registered gain and tau')
        out[key] = dict(row)
    return out


def parse_s0(raw):
    n = get(raw, 'n', 'metrics.json')
    if get(raw, 'version', 'metrics.json') != 1 or get(raw, 'stage', 'metrics.json') != 'S0' or whole(n, 'n') < 1:
        refuse('metrics.json is not a version 1 S0 record')
    calibration = parse_calibration(get(raw, 'calibration', 'metrics.json'))
    selected = calibration['selected']
    common = get(raw, 'common', 'metrics.json')
    d128 = get(raw, 'd128_3', 'metrics.json')
    old = get(d128, 'old_range', 'd128_3')
    whole(get(old, 'n', 'd128_3.old_range'), 'd128_3.old_range n')
    whole(get(old, 'correct', 'd128_3.old_range'), 'd128_3.old_range correct', old['n'])
    real(get(old, 'loss_float32', 'd128_3.old_range'), 'd128_3.old_range loss_float32')
    return dict(n=n, ids=raw.get('ids_blake2b_128'), calibration=calibration,
                conditions=parse_conditions(get(raw, 'conditions', 'metrics.json'), n, 'conditions', CONDITIONS, selected),
                common={k: real(get(common, k, 'common'), f'common {k}') for k in ('key_top1', 'key_margin')},
                d128=parse_conditions(get(d128, 'conditions', 'd128_3'), n, 'd128_3.conditions', [(r, AV) for r in D128_ROLES], selected),
                old_range=dict(correct=old['correct'], n=old['n'], loss_float32=old['loss_float32']))


def parse_samples(z, n, conditions):
    """Episode ids (identical across every condition) and the 0/1 correctness vector of each condition, checked against metrics.json."""
    names = sorted(name for name in z if name.endswith('/episode_id'))
    for key in conditions:
        if f'{key}/episode_id' not in z or f'{key}/correct' not in z:
            refuse(f'samples.npz lacks episode_id or correct of {key}')
    first = z[names[0]]
    for name in names:
        if z[name].dtype != np.int64 or z[name].shape != (n,):
            refuse(f'samples.npz: {name} is not an int64 array of n episodes')
        if not np.array_equal(z[name], first):
            refuse(f'samples.npz: episode_id of {name[:-11]} differs from that of {names[0][:-11]}')
    if len(set(first.tolist())) != n:
        refuse('samples.npz: duplicate episode ids')
    correct = {}
    for key, row in conditions.items():
        correct[key] = flags(z[f'{key}/correct'], f'samples.npz {key}/correct')
        if correct[key].shape != (n,) or int(correct[key].sum()) != row['correct']:
            refuse(f'samples.npz: {key} has another n or correct count than metrics.json')
        for field in ('nll', 'rel_err'):
            if f'{key}/{field}' in z and floats(z[f'{key}/{field}'], f'samples.npz {key}/{field}').shape != (n,):
                refuse(f'samples.npz: {key}/{field} has another n')
    return first, correct


def decide_s0(conditions, n):
    """The registered first-matching row and the comparisons behind it. improves(a over b) = (c_a - c_b) * 20 >= n (+5.0 pp, exact)
    and a lower mean read error, both at the same scope; Clean-final needs only the count."""
    tests = {}
    for role, scope in (('soft_cal', AV), ('hard_cal', AV), ('soft_cal', FO), ('hard_cal', FO), ('clean_final', FO)):
        a, b = conditions[f'{role}/{scope}'], conditions[f'matrix_cal/{scope}']
        gap = a['correct'] - b['correct']
        tests[f'{role}/{scope}'] = t = dict(vs=f'matrix_cal/{scope}', d_correct=gap, five_pp=gap * 20 >= n)
        if role != 'clean_final':
            t.update(read_error_lower=a['rel_err'] < b['rel_err'], improves=t['five_pp'] and a['rel_err'] < b['rel_err'])
    improves = lambda key: tests[key]['improves']
    if improves(f'soft_cal/{AV}'):
        row = 'R1'
    elif improves(f'hard_cal/{AV}'):
        row = 'R2'
    elif improves(f'soft_cal/{FO}') or improves(f'hard_cal/{FO}'):
        row = 'R3'
    elif tests[f'clean_final/{FO}']['five_pp']:
        row = 'R4'
    else:
        row = 'R5'
    return row, tests


def s1a_choices(selected):
    return dict(arms=['Matrix-L1', 'SoftSlot-L1'], matrix=dict(gain=selected['matrix']['gain']),
                soft_slot=dict(gain=selected['soft']['gain'], tau=selected['soft']['tau']))


def differences(conditions, correct, n):
    """Descriptive: each compared condition minus matrix_cal at its scope, in pp, with the paired percentile bootstrap interval
    (one shared index matrix); the decision never reads these."""
    vectors = {f'{r}/{s}': correct[f'{r}/{s}'] - correct[f'matrix_cal/{s}'] for r, s in COMPARED}
    sums = resample_sums(vectors, n, BOOTSTRAP['S0'])
    out = {}
    for r, s in COMPARED:
        key, base = f'{r}/{s}', conditions[f'matrix_cal/{s}']
        low, high = interval(sums[key])
        gap = conditions[key]['correct'] - base['correct']
        out[key] = dict(vs=f'matrix_cal/{s}', d_correct=gap, pp=float(Fraction(100 * gap, n)),
                        ci95_pp=[float(low * 100 / n), float(high * 100 / n)], read_error_diff=conditions[key]['rel_err'] - base['rel_err'])
    return out


def direction(gap):
    return 'higher' if gap > 0 else 'lower' if gap < 0 else 'equal'


def d128_recheck(d128, conditions, n):
    """Descriptive: soft and hard minus matrix on D128-3 (delta store) against the same direction on S-RE."""
    out = {}
    for role in ('soft', 'hard'):
        gap = d128[f'{role}/{AV}']['correct'] - d128[f'matrix/{AV}']['correct']
        out[f'{role}_minus_matrix'] = dict(d_correct=gap, pp=float(Fraction(100 * gap, n)), direction=direction(gap),
                                          read_error_diff=d128[f'{role}/{AV}']['rel_err'] - d128[f'matrix/{AV}']['rel_err'])
    s_re = conditions[f'soft_cal/{AV}']['correct'] - conditions[f'matrix_cal/{AV}']['correct']
    out['s_re_soft_minus_matrix'] = dict(d_correct=s_re, direction=direction(s_re))
    out['same_direction_as_s_re'] = out['soft_minus_matrix']['direction'] == direction(s_re)
    return out


def soft_rec_vs_soft(conditions):
    """Descriptive: the recency-matched SoftSlot against SoftSlot (EM counts at all_valid_question), uncalibrated and calibrated."""
    pairs = {name: dict(soft=conditions[f'soft_{name}/{AV}']['correct'], soft_rec=conditions[f'soft_rec_{name}/{AV}']['correct'])
             for name in ('uncal', 'cal')}
    for p in pairs.values():
        p['d_correct'] = p['soft_rec'] - p['soft']
    return dict(pairs, largest_abs_d_correct=max(abs(p['d_correct']) for p in pairs.values()))


def stage_s0(src, preflight):
    gate = read_preflight(src.json(preflight, absolute=True))
    if gate['failed']:   # the gate failed: no comparison, and no S0 data are opened
        decision = dict(stage='S0', row='R0', label=LABELS['R0'], s1a=None, preflight=gate)
        return {'decision_s0.json': dumps(decision), 'decision_s0.md': render_s0(decision)}, ['stage S0', 'row R0']
    m = parse_s0(src.json('metrics.json'))
    n, conditions = m['n'], m['conditions']
    ids, correct = parse_samples(src.npz('samples.npz'), n, conditions)
    own = digest(ids.astype('<i8').tobytes())
    if own != m['ids']:
        refuse('the episode ids of samples.npz do not match ids_blake2b_128 of metrics.json (blake2b-128 of little-endian int64)')
    row, tests = decide_s0(conditions, n)
    selected, rec, notes = m['calibration']['selected'], soft_rec_vs_soft(conditions), []
    if selected['soft']['tau'] == 0.03:
        notes.append('tau = 0.03 selected: HardSlot is the tau -> 0 limit; the grid is not extended')
    notes.append(f"soft_rec against soft: largest EM-count difference {rec['largest_abs_d_correct']} of {n}; "
                 "S-RE's decay is 0.999999, so near equality is expected")
    if gate['overall_passed'] is not True or gate['other_failed']:
        notes.append(f'preflight file: overall passed = {gate["overall_passed"]}, failed checks outside the required list: '
                     f'{gate["other_failed"]}; every required check passed')
    decision = dict(stage='S0', row=row, label=LABELS[row], s1a=s1a_choices(selected), n=n, ids_blake2b_128=own, preflight=gate,
                    tests=tests, calibration=m['calibration'], conditions=conditions, common=m['common'],
                    differences=differences(conditions, correct, n), soft_rec_vs_soft=rec,
                    d128_3=dict(conditions=m['d128'], old_range=m['old_range'], expected_old_range=OLD_RANGE,
                                recheck=d128_recheck(m['d128'], conditions, n)),
                    notes=notes, bootstrap=dict(seed=BOOTSTRAP['S0'], resamples=BLOCKS * BLOCK, block=BLOCK, descriptive=True))
    return {'decision_s0.json': dumps(decision), 'decision_s0.md': render_s0(decision)}, ['stage S0', f'row {row}', f'n {n}']


# -------------------------------------------------------------------------------------------------------------- S1A-DEV

def select_step(rows):
    """The registered checkpoint choice among the selectable rows in step order: the lowest finite dev NLL; a later row replaces the
    best so far only if lower by MORE than 1e-6 (a tie goes to the earlier)."""
    best = None
    for r in rows:
        if r['selectable'] and r['nll'] is not None and (best is None or Fraction(best['nll']) - Fraction(r['nll']) > SELECTION_TIE):
            best = r
    return best


def parse_run(raw, arm, seed):
    where = f'{arm}/s{seed}/record.json'
    if get(raw, 'status', where) != 'complete':
        reason = raw.get('failure_reason')
        first = reason.strip().splitlines()[0][:120] if isinstance(reason, str) and reason.strip() else ''
        refuse(f'{where} has status {raw["status"]!r}, not complete' + (f' ({first})' if first else ''))
    if get(raw, 'arm', where) != arm or get(raw, 'seed', where) != seed:
        refuse(f'{where} carries another arm or seed')
    steps = whole(get(raw, 'steps', where), f'{where} steps')
    marks = [0] + [steps * k // 10 for k in range(1, 11)]
    dev = get(raw, 'dev', where)
    if steps < 10 or steps % 10 or not isinstance(dev, list) or [r.get('step') if isinstance(r, dict) else None for r in dev] != marks:
        refuse(f'{where}: dev rows are not step 0 and the ten steps k N / 10 of a multiple-of-ten N')
    rows = []
    for r in dev:
        step, label = r['step'], f'{where} dev step {r["step"]}'
        if get(r, 'selectable', label) is not (step >= 1):
            refuse(f'{label}: selectable must be true exactly for step >= 1')
        nll = get(r, 'nll', label)
        if nll is not None and (isinstance(nll, bool) or not isinstance(nll, (int, float))):
            refuse(f'{label}: nll is not a number')
        n = whole(get(r, 'n', label), f'{label} n')
        rows.append(dict(step=step, selectable=step >= 1, n=n, correct=whole(get(r, 'correct', label), f'{label} correct', n),
                         nll=float(nll) if nll is not None and math.isfinite(nll) else None))
    if len({r['n'] for r in rows}) != 1:
        refuse(f'{where}: dev rows have different n')
    if rows[0]['nll'] is None:
        refuse(f'{where}: the step-0 dev NLL is not finite')
    training = get(raw, 'training', where)
    if not isinstance(get(training, 'all_finite', f'{where} training'), bool):
        refuse(f'{where}: training.all_finite is not a boolean')
    best = select_step(rows)
    given = get(raw, 'selected_step', where)
    if given is not None:
        whole(given, f'{where} selected_step')
    if given != (None if best is None else best['step']):
        refuse(f'{where}: selected_step differs from the registered rule')
    checkpoint = raw.get('selected_checkpoint_blake2b_128')
    if best is not None and not (isinstance(checkpoint, str) and len(checkpoint) == 32 and all(c in '0123456789abcdef' for c in checkpoint)):
        refuse(f'{where}: selected_checkpoint_blake2b_128 is not 32 hex digits')
    extra = {k: training[k] for k in ('clipped_fraction', 'grad_norm_max') if isinstance(training.get(k), (int, float))
             and not isinstance(training[k], bool) and math.isfinite(training[k])}
    ids = get(raw, 'dev_ids_blake2b_128', where)
    if not isinstance(ids, str):
        refuse(f'{where}: dev_ids_blake2b_128 is not a string')
    return dict(steps=steps, n=rows[0]['n'], rows=rows, selected=best, checkpoint=checkpoint if best else None, ids=ids,
                all_finite=training['all_finite'], training=dict(extra, all_finite=training['all_finite']))


def stage_dev(src):
    runs = {(arm, seed): parse_run(src.json(f'{arm}/s{seed}/record.json'), arm, seed) for arm in ARMS for seed in SEEDS}
    for field, what in (('ids', 'dev_ids_blake2b_128'), ('n', 'dev n'), ('steps', 'steps')):
        if len({r[field] for r in runs.values()}) != 1:
            refuse(f'the six records differ in {what}')
    n, reasons = next(iter(runs.values()))['n'], []
    for (arm, seed), r in runs.items():
        label = f'{arm}/s{seed}'
        if not r['all_finite']:
            reasons.append(f'{label}: training.all_finite is false')
        if r['selected'] is None:
            reasons.append(f'{label}: no finite selectable dev row')
        elif r['selected']['nll'] > r['rows'][0]['nll']:
            reasons.append(f'{label}: the selected dev NLL (step {r["selected"]["step"]}) is above the step-0 dev NLL')
    gaps = None
    if all(r['selected'] is not None for r in runs.values()):
        per_seed = {s: runs['soft_slot_L1', s]['selected']['correct'] - runs['matrix_L1', s]['selected']['correct'] for s in SEEDS}
        total = sum(per_seed.values())
        gaps = dict(per_seed={str(s): dict(d_correct=d, pp=float(Fraction(100 * d, n))) for s, d in per_seed.items()},
                    sum_d_correct=total, mean_pp=float(Fraction(100 * total, 3 * n)), trigger_below_pp=3.0)
    outcome = 'stop_unstable' if reasons else 'ablation' if gaps['sum_d_correct'] * 100 < 9 * n else 'no_trigger'
    summary = dict(stage='S1A-DEV', outcome=outcome, reasons=reasons, n=n, steps=next(iter(runs.values()))['steps'],
                   dev_ids_blake2b_128=next(iter(runs.values()))['ids'], differences=gaps,
                   runs={f'{a}/s{s}': dict(selected_step=None if r['selected'] is None else r['selected']['step'],
                                           selected_checkpoint_blake2b_128=r['checkpoint'], training=r['training'],
                                           dev=[dict(row, em=row['correct'] / row['n']) for row in r['rows']])
                         for (a, s), r in runs.items()})
    lines = ['stage S1A-DEV', f'outcome {outcome}', 'runs 6'] + [f'reason {x}' for x in reasons]
    return {'s1a-selection.json': dumps(summary), 's1a-selection.md': render_dev(summary)}, lines


# ------------------------------------------------------------------------------------------------------------------ S1A

def read_set(src, name):
    """The per-label arrays of one set (required labels in every set, references where present); ids identical across files."""
    labels, ids, first = {}, None, None
    for label in REQUIRED + REFERENCES:
        path = f'{name}/{label}.npz'
        if not src.exists(path):
            if label in REQUIRED:
                refuse(f'{path} is missing')
            continue
        z = src.npz(path)
        eid = get(z, 'episode_id', path)
        if eid.dtype != np.int64 or eid.ndim != 1:
            refuse(f'{path}: episode_id is not a 1-D int64 array')
        if ids is None:
            ids, first = eid, path
        elif len(eid) != len(ids):
            refuse(f'{path}: n differs from {first}')
        elif not np.array_equal(eid, ids):
            refuse(f'{path}: episode_id differs from {first}')
        row = dict(correct=flags(get(z, 'correct', path), f'{path} correct'), nll=floats(get(z, 'nll', path), f'{path} nll'),
                   rel_err=floats(z['rel_err'], f'{path} rel_err') if 'rel_err' in z else None)
        if any(v is not None and len(v) != len(ids) for v in row.values()):
            refuse(f'{path}: an array has another n than episode_id')
        labels[label] = row
    if len(set(ids.tolist())) != len(ids):
        refuse(f'{name}: duplicate episode ids')
    return ids, labels


def analyze_set(ids, labels):
    """Per-label means and the primary contrast SoftSlot-L1 minus Matrix-L1 (three seeds) with paired bootstrap intervals."""
    n = len(ids)
    rows = {label: dict(correct=int(v['correct'].sum()), em=int(v['correct'].sum()) / n, nll=mean(v['nll']),
                        rel_err=None if v['rel_err'] is None else mean(v['rel_err'])) for label, v in labels.items()}
    diff = {s: labels[f'soft_slot_L1_s{s}']['correct'] - labels[f'matrix_L1_s{s}']['correct'] for s in SEEDS}
    sums = resample_sums({**{str(s): d for s, d in diff.items()}, 'pooled': sum(diff.values())}, n, BOOTSTRAP['S1A'])

    def stats(name, total, scale):
        low, high = interval(sums[name])
        return dict(delta_correct=total, delta_pp=float(Fraction(100 * total, scale)), lower_pp=float(low * 100 / scale),
                    upper_pp=float(high * 100 / scale)), low > 0
    pooled, positive = stats('pooled', int(sum(d.sum() for d in diff.values())), 3 * n)
    pooled['sum_delta_correct'] = pooled.pop('delta_correct')
    arms = {}
    for arm in ARMS:
        seeds = {str(s): rows[f'{arm}_s{s}'] for s in SEEDS}
        errors = [r['rel_err'] for r in seeds.values()]
        arms[arm] = dict(seeds=seeds, em=float(Fraction(sum(r['correct'] for r in seeds.values()), 3 * n)),
                         nll=math.fsum(r['nll'] for r in seeds.values()) / 3, rel_err=None if None in errors else math.fsum(errors) / 3)
    return dict(n=n, expected_n=EXPECTED_N, episode_ids_blake2b_128_le_int64=digest(ids.astype('<i8').tobytes()), arms=arms,
                references={label: rows[label] for label in REFERENCES if label in rows},
                contrast=dict(pooled, lower_end_positive=positive, seeds={str(s): stats(str(s), int(diff[s].sum()), n)[0] for s in SEEDS}))


def stage_s1a(src):
    sets = {name: read_set(src, name) for name in ('confirmation', 'unseen')}
    if np.intersect1d(sets['confirmation'][0], sets['unseen'][0]).size:
        refuse('the confirmation and unseen sets share episode ids')
    results = {name: analyze_set(*sets[name]) for name in sets}
    c = results['confirmation']
    tests = dict(delta_at_least_3pp=c['contrast']['sum_delta_correct'] * 100 >= 9 * c['n'],   # Delta >= +3.0 pp, exact
                 lower_end_positive=c['contrast']['lower_end_positive'],
                 seeds_all_positive=all(s['delta_correct'] > 0 for s in c['contrast']['seeds'].values()))
    verdict = 'ADVANCE' if all(tests.values()) else 'NOT ADVANCED'
    summary = dict(stage='S1A', verdict=verdict, tests=tests, confirmation=c, unseen=results['unseen'],
                   bootstrap=dict(seed=BOOTSTRAP['S1A'], resamples=BLOCKS * BLOCK, block=BLOCK, interval='percentile 95 %'))
    lines = ['stage S1A'] + [f'{name} n {r["n"]} expected {EXPECTED_N} files {len(sets[name][1])}' for name, r in results.items()]
    return {'summary.json': dumps(summary), 'verdict.md': render_s1a(summary)}, lines + [f'verdict {verdict}']


# --------------------------------------------------------------------------------------------------------------- reports

def table(header, rows):
    return (['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def cell(x, digits=4):
    return '' if x is None else f'{x:.{digits}f}'


def render_s0(d):
    lines = [f'# S0 decision: {d["row"]}', '', d['label'], '']
    gate = d['preflight']
    lines += ['## Engineering gate (required preflight checks)', '']
    lines += table(['check', 'passed'], [[k, 'yes' if v else 'NO'] for k, v in gate['required'].items()])
    if gate['failed']:
        return '\n'.join(lines + ['', 'Failed: ' + ', '.join(gate['failed']) + '. No comparison is made, S1A: none; no S0 data were opened.', ''])
    n, s1a = d['n'], d['s1a']
    lines += ['', f'S1A: arms Matrix-L1 and SoftSlot-L1; Matrix g = {s1a["matrix"]["gain"]:g}; SoftSlot g = {s1a["soft_slot"]["gain"]:g}, '
              f'tau = {s1a["soft_slot"]["tau"]:g}.', '', f'n = {n} dev episodes; calibration n = {d["calibration"]["n"]}.', '',
              '## Decision comparisons (against matrix_cal at the same scope; +5.0 pp is (c_a - c_b) * 20 >= n)', '']
    lines += table(['condition', 'vs', 'd correct', '+5.0 pp met', 'read error lower', 'improves'],
                   [[k, t['vs'], t['d_correct'], t['five_pp'], t.get('read_error_lower', ''), t.get('improves', '')] for k, t in d['tests'].items()])
    for scope in (AV, FO):
        lines += ['', f'## Conditions, scope {scope}', '']
        lines += table(['role', 'g', 'tau', 'EM %', 'correct', 'NLL', 'read error', 'cosine', 'norm ratio', 'w_true', 'w_gap', 'w_entropy', 'top slot true'],
                       [[r, cell(c.get('gain'), 2), cell(c.get('tau'), 2), f'{100 * c["correct"] / n:.1f}', c['correct'], cell(c['nll'], 6),
                         cell(c['rel_err']), cell(c['cos']), cell(c['norm_ratio']), cell(c.get('w_true')), cell(c.get('w_gap')),
                         cell(c.get('w_entropy')), cell(c.get('slot_is_true'))]
                        for r, s in CONDITIONS if s == scope for c in [d['conditions'][f'{r}/{s}']]])
    lines += ['', f'Key top-1 {d["common"]["key_top1"]:.4f}, key margin {d["common"]["key_margin"]:.4f} (shared by every condition).', '',
              f'## Calibration ({d["calibration"]["n"]} episodes, mean NLL; g = 1, tau = 1 is the uncalibrated row)', '']
    chosen = d['calibration']['selected']
    lines += table(['kind', 'g', 'tau', 'NLL', 'correct', 'seconds', 'selected'],
                   [[c['kind'], f'{c["gain"]:g}', cell(c['tau'], 2), cell(c['nll'], 6), c['correct'], cell(c['seconds'], 2),
                     'yes' if c['gain'] == chosen[c['kind']]['gain'] and c['tau'] == chosen[c['kind']].get('tau') else '']
                    for c in d['calibration']['candidates']])
    lines += ['', '## Differences against matrix_cal (descriptive: paired 95 % percentile bootstrap, never used in the decision)', '']
    lines += table(['condition', 'vs', 'd correct', 'difference pp', '95 % interval pp', 'd read error'],
                   [[k, v['vs'], v['d_correct'], f'{v["pp"]:+.2f}', f'[{v["ci95_pp"][0]:+.2f}, {v["ci95_pp"][1]:+.2f}]', f'{v["read_error_diff"]:+.4f}']
                    for k, v in d['differences'].items()])
    lines += ['', '## D128-3 recheck (descriptive; S-RE selected g and tau, no recalibration)', '']
    lines += table(['role', 'EM %', 'correct', 'NLL', 'read error'],
                   [[r, f'{100 * c["correct"] / n:.1f}', c['correct'], cell(c['nll'], 6), cell(c['rel_err'])]
                    for r in D128_ROLES for c in [d['d128_3']['conditions'][f'{r}/{AV}']]])
    recheck, old = d['d128_3']['recheck'], d['d128_3']['old_range']
    soft, hard, s_re = recheck['soft_minus_matrix'], recheck['hard_minus_matrix'], recheck['s_re_soft_minus_matrix']
    lines += ['', f'Soft minus matrix on D128-3: {soft["d_correct"]:+d} ({soft["pp"]:+.2f} pp, {soft["direction"]}); '
              f'hard minus matrix: {hard["d_correct"]:+d} ({hard["pp"]:+.2f} pp, {hard["direction"]}); '
              f'on S-RE soft minus matrix is {s_re["direction"]}; same direction: {"yes" if recheck["same_direction_as_s_re"] else "no"}.',
              f'Old-range replay: {old["correct"]}/{old["n"]}, loss {old["loss_float32"]!r} '
              f'(expected {OLD_RANGE["correct"]}/{OLD_RANGE["n"]} and {OLD_RANGE["loss_float32"]!r}; reported, not judged).', '', '## Notes', '']
    return '\n'.join(lines + [f'- {x}' for x in d['notes']] + [''])


def render_dev(s):
    lines = [f'# S1A dev selection: {s["outcome"]}', '',
             f'n = {s["n"]} dev episodes, N = {s["steps"]} steps, dev ids blake2b-128 {s["dev_ids_blake2b_128"]}.', '']
    if s['outcome'] == 'stop_unstable':
        lines += ['STOP for the orchestrator; no ablation trigger.', ''] + [f'- {x}' for x in s['reasons']] + ['']
    elif s['outcome'] == 'ablation':
        lines += ['Stable runs, mean dev gap (SoftSlot minus Matrix) below +3.0 pp: the section-7.4 ablation is triggered.', '']
    else:
        lines += ['Stable runs, mean dev gap (SoftSlot minus Matrix) at or above +3.0 pp: no ablation.', '']
    if s['differences']:
        g = s['differences']
        lines += table(['seed', 'd correct', 'pp'], [[k, v['d_correct'], f'{v["pp"]:+.2f}'] for k, v in g['per_seed'].items()]
                       + [['sum / mean', g['sum_d_correct'], f'{g["mean_pp"]:+.2f}']]) + ['']
    for name, run in s['runs'].items():
        lines += [f'## {name}: selected step {run["selected_step"]}, checkpoint {run["selected_checkpoint_blake2b_128"]}', '']
        lines += table(['step', 'selectable', 'correct', 'EM %', 'NLL'],
                       [[r['step'], r['selectable'], r['correct'], f'{100 * r["em"]:.1f}', cell(r['nll'], 6) or 'not finite'] for r in run['dev']]) + ['']
    return '\n'.join(lines)


def render_s1a(s):
    lines = [f'# S1A verdict: {s["verdict"]}', '', 'Rule: Delta >= +3.0 pp (sum over seeds of C_soft - C_matrix, times 100, >= 9 n), lower interval end L > 0, '
             'and C_soft > C_matrix for each of seeds 17, 29, 43.', '']
    lines += table(['test', 'met'], [[k, v] for k, v in s['tests'].items()]) + ['']
    for name in ('confirmation', 'unseen'):
        r, k = s[name], s[name]['contrast']
        lines += [f'## {name}' + (' (primary)' if name == 'confirmation' else ' (descriptive, no verdict)'), '',
                  f'n = {r["n"]} (expected {r["expected_n"]}); episode ids blake2b-128 {r["episode_ids_blake2b_128_le_int64"]} (little-endian int64).', '']
        lines += table(['contrast', 'd correct', 'difference pp', '95 % lower pp', '95 % upper pp'],
                       [['Delta (mean over seeds)', k['sum_delta_correct'], f'{k["delta_pp"]:+.3f}', f'{k["lower_pp"]:+.3f}', f'{k["upper_pp"]:+.3f}']]
                       + [[f'Delta_s seed {seed}', v['delta_correct'], f'{v["delta_pp"]:+.3f}', f'{v["lower_pp"]:+.3f}', f'{v["upper_pp"]:+.3f}']
                          for seed, v in k['seeds'].items()]) + ['']
        rows = []
        for arm, a in r['arms'].items():
            rows += [[arm, seed, v['correct'], f'{100 * v["em"]:.2f}', cell(v['nll'], 6), cell(v['rel_err'])] for seed, v in a['seeds'].items()]
            rows.append([arm, 'mean', '', f'{100 * a["em"]:.2f}', cell(a['nll'], 6), cell(a['rel_err'])])
        rows += [[label, '', v['correct'], f'{100 * v["em"]:.2f}', cell(v['nll'], 6), cell(v['rel_err'])] for label, v in r['references'].items()]
        lines += table(['arm or reference', 'seed', 'correct', 'EM %', 'mean NLL', 'mean read error'], rows) + ['']
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------------------------------------ main

def emit(out, stage, files, src, lines):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = dict(files, **{'source_index.json': dumps(src.index(stage))})
    digests = {}
    for name in sorted(files):
        data = files[name].encode()
        (out / name).write_bytes(data)
        digests[name] = digest(data)
    print('\n'.join(lines + [f'{name} {digests[name]}' for name in sorted(digests)]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--stage', required=True, choices=('S0', 'S1A-DEV', 'S1A'))
    parser.add_argument('--records', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--preflight')
    args = parser.parse_args(argv)
    if (args.stage == 'S0') != (args.preflight is not None):
        refuse('--preflight belongs to stage S0 and only to it')
    src = Sources(args.records)
    if args.stage == 'S0':
        files, lines = stage_s0(src, args.preflight)
    elif args.stage == 'S1A-DEV':
        files, lines = stage_dev(src)
    else:
        files, lines = stage_s1a(src)
    emit(args.out, args.stage, files, src, lines)
    return 0


if __name__ == '__main__':
    sys.exit(main())
