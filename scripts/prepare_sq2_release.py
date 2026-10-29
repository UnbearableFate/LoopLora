"""Immutable SQ2 schedule release (docs/prereg/SQ2.md): the continuation length N from the timing of the 10% smokes of the four arms.

Standard library only, for the login node. N = min(6,000, the largest multiple of 5 that fits one 1 h job under the DPFP rule at
3,600 s, from the slowest arm's smoke); the same N for every arm. The release also holds the numerics (eta, w_c, the file's
blake2b-128, frozen before any smoke), the parent checkpoint's digests, the passed step-0 check, the science config hashes (compute:
`dpfp_config_hashes.py --study SQ2`, from `--steps-only`'s output) and the registration's blake2b-128. Only identity and timing
fields of the smoke records are read, never a score. An existing release is never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from looplora.llm.release import EVAL_SCALE,MARGIN,RESERVE,largest_schedule,read_json,require,smoke_timing,write_new
from looplora.llm.sq2_release import (ARMS,EVALUATIONS,N_MAX,PARENT,PREREG,SPECS,WALL_SECONDS,file_digest,prereg_digest,
                                      scheduled_steps)

SMOKE_STEPS = 600  # 10 % of N_MAX


def check_release(release):
    """What the runner will do with the release: it must accept the step count of every arm."""
    require(release['study'] == 'SQ2' and release['E'] == 16, 'the release names another study or capacity')
    require(set(release['steps_by_arm']) == set(ARMS), 'the SQ2 release names all four arms')
    for arm, steps in release['steps_by_arm'].items():
        require(scheduled_steps(arm, None, release) == steps, f'{arm}: the runner does not accept this release')


def smoke_row(root, arm, numerics, *, steps):
    record = read_json(Path(root) / arm / 'record.json')
    store, coefficient = SPECS[arm]
    config = record['config']
    require(record['store'] == store and record['rank'] == 128 and record['dpfp_nu'] == 3 and config['sq2']['arm'] == arm
            and config['sq2']['coefficient'] == coefficient and config['sq2']['numerics'] == numerics['file_blake2b_128']
            and record['parent']['config_hash'] == PARENT['config_hash'],
            f'{arm}: the smoke record belongs to another arm or to other numerics')
    require((record['store_eta'] == numerics['eta']) == (store == 'ridge') and
            (record['coefficient_weight'] == numerics['w_c']) == coefficient,
            f'{arm}: the smoke used another eta or w_c than the frozen numerics')
    return smoke_timing(record, arm=arm, steps=steps, E=16)


def continuation_length(rows):
    """N and the sizing record of every arm. The step count is the same for all arms: the smallest of the per-arm rule values
    (DPFP's rule at 3,600 s with six 1,000-episode evaluations: the five checkpoints and step 0) and 6,000."""
    sizing, per_arm = {}, {}
    for arm, row in rows.items():
        step, full_eval = row['steady_step_mean_seconds'], max(row['eval100_seconds']) * EVAL_SCALE
        per_arm[arm] = largest_schedule(step, full_eval, evaluations=EVALUATIONS, wall_seconds=WALL_SECONDS)
        sizing[arm] = dict(step_seconds=step, eval1000_seconds=full_eval, evaluations=EVALUATIONS, margin=MARGIN,
                           setup_reserve_seconds=RESERVE, largest_steps_in_one_hour=per_arm[arm])
    n = 5 * (min(N_MAX, *per_arm.values()) // 5)
    for arm, row in rows.items():
        sizing[arm]['expected_seconds_with_margin'] = MARGIN * (n * sizing[arm]['step_seconds']
            + EVALUATIONS * sizing[arm]['eval1000_seconds']) + RESERVE
        require(sizing[arm]['expected_seconds_with_margin'] <= WALL_SECONDS, f'{arm}: {n} steps do not fit one hour')
        sizing[arm]['smoke_fraction_of_fixed_steps'] = row['steps'] / n
    return n, sizing


def prepare(root, out, *, numerics, step0=None, config_hashes=None, steps=SMOKE_STEPS):
    if out is not None and Path(out).exists():
        raise FileExistsError(f'{out} exists: a frozen release is never overwritten')
    numbers = read_json(numerics)
    frozen = dict(path=str(numerics), file_blake2b_128=file_digest(numerics), eta=numbers['eta'], w_c=numbers['w_c'],
                  eta_curve=numbers['eta_curve'], gradient_ratio_median=numbers['gradient_ratio']['median'],
                  ridge_store_gradient_ratio_median=numbers['ridge_store_gradient_ratio_median'])
    rows = {arm: smoke_row(root, arm, frozen, steps=steps) for arm in ARMS}
    commits = {row['git_commit'] for row in rows.values()}
    require(len(commits) == 1, 'the smokes ran at different commits')
    n, sizing = continuation_length(rows)
    marks = [n * k // 5 for k in range(1, 6)]
    release = dict(study='SQ2', E=16, steps_by_arm={arm: n for arm in ARMS}, primary_eval_steps_by_arm={arm: marks for arm in ARMS},
                   sizing_by_arm=sizing, science_commit=commits.pop(),
                   smoke=dict(steps=steps, evaluation_episodes=100, rows=list(rows.values()), quality_scores_exposed=False),
                   max_job_seconds=WALL_SECONDS, walltime='01:00:00', status='released_after_smoke_timing_gate',
                   scale_up=None, numerics=frozen, prereg=PREREG, prereg_blake2b_128=prereg_digest(),
                   parent=dict(arm=PARENT['arm'], steps=PARENT['steps'], config_hash=PARENT['config_hash'],
                               digests=PARENT['digests']))
    if step0 is not None:
        check = read_json(step0)
        require(check['passed'] is True, 'the step-0 check did not pass: the continuation is void')
        release['step0'] = check
    if config_hashes is not None:
        require(set(config_hashes) == set(ARMS) and all(isinstance(v, str) and len(v) == 32 for v in config_hashes.values()),
                'config_hashes must name all four SQ2 arms')
        release['config_hashes'] = dict(config_hashes)
    if out is not None:
        require(config_hashes is not None and step0 is not None, 'an SQ2 release needs the step-0 check and the science config hashes')
        check_release(release)
        write_new(Path(out), release)
    return release


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='smoke output root: ROOT/ARM/record.json for the four arms')
    parser.add_argument('--numerics', type=Path, required=True, help='the committed numerics file')
    parser.add_argument('--out', type=Path, help='the release file (must not exist)')
    parser.add_argument('--steps', type=int, default=SMOKE_STEPS, help='step count of the smokes')
    parser.add_argument('--steps-only', action='store_true', help='print N, write nothing')
    parser.add_argument('--step0', type=Path, help='JSON of `check_sq2.py step0` (must have passed)')
    parser.add_argument('--config-hashes', type=Path, help='JSON of `dpfp_config_hashes.py --study SQ2`')
    args = parser.parse_args()
    if args.steps_only:
        print(json.dumps(prepare(args.root, None, numerics=args.numerics, steps=args.steps)['steps_by_arm']))
        raise SystemExit(0)
    if args.out is None:
        parser.error('--out is required')
    hashes = None if args.config_hashes is None else json.loads(args.config_hashes.read_text())['config_hashes']
    release = prepare(args.root, args.out, numerics=args.numerics, step0=args.step0, config_hashes=hashes, steps=args.steps)
    print(json.dumps(dict(study=release['study'], steps_by_arm=release['steps_by_arm'], science_commit=release['science_commit'],
                          eta=release['numerics']['eta'], w_c=release['numerics']['w_c'], quality_scores_exposed=False)), flush=True)
