"""The DPFP and LSQ release scripts (stdlib only; no model, no GPU) and the torch-free modules they share with the runners."""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from looplora.infra.seeds import stable_hash
from looplora.llm import dpfp_release, lsq_release, release

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_lsq = load_script('prepare_lsq_release')
prepare_dpfp = load_script('prepare_dpfp_release')

COMMIT = 'a' * 40
EVALS = (0.9, 0.91, 0.89, 1.07, 0.9)  # seconds of the five 100-episode smoke evaluations
SENTINEL = 987654  # a smoke "score": it must never reach a release
STEP = {arm: 0.15 + 0.01 * i for i, arm in enumerate(lsq_release.ARMS + dpfp_release.ARMS)}  # seconds per training step

STORAGE_BOTH = ['L32-K4', 'L32-RG', 'L128-K4', 'L128-RG']
MIXED = ['L32-U', 'L128-K4', 'L128-RG']
RFX_ETA = {'32': 3e-3, '128': 1e-6}  # the second is below the bound: the release clips it to 1e-4
RFX_ARM_SETS = [['L32-K4', 'L32-RG'], ['L128-U'], ['L32-U', 'L128-U'], STORAGE_BOTH, MIXED, ['L32-K4', 'L32-RG', 'L128-U']]


def expected_steps(arm):
    """The registered rule, transcribed: 1.5 margin, 90 s reserve, 7200 s, five evaluations of ten times the smoke's maximum."""
    available = (7200 - 90) / 1.5 - 5 * 10 * max(EVALS)
    return 5 * math.floor(available / STEP[arm] / 5)


def smoke_record(study, arm, *, steps=2400, E=16, commit=COMMIT, tweak=None):
    marks = [round(steps * k / 5) for k in range(1, 6)]
    config = dict(smoke=True, task=dict(E=E), eval=dict(episodes=100))
    record = dict(status='complete', smoke=True, arm=arm, source_dirty=False, git_commit=commit, source_hash='0' * 32,
                  job_id='5.opbs', host='mg0001')
    if study == 'LSQ':
        rank, store = lsq_release.SPECS[arm]
        config['lsq'] = dict(arm=arm)
        record.update(rank=rank, store=store, store_eta=3e-3 if store == 'ridge' else None)
    else:
        rank, nu = dpfp_release.SPECS[arm]
        config['dpfp'] = dict(arm=arm)
        record.update(rank=rank, dpfp_nu=nu)
    record.update(config=config, config_hash=stable_hash(config), training=dict(
        actual_steps=steps, training_episodes=32 * steps, eval_steps=marks, primary_eval_steps=marks, matched_eval_steps=[],
        metrics={str(m): dict(n=100, correct=SENTINEL, em=SENTINEL / 1000, loss=SENTINEL / 7, seconds=s)
                 for m, s in zip(marks, EVALS)},
        seconds=STEP[arm] * steps + sum(EVALS), steady_step_mean_seconds=STEP[arm], step_p95_seconds=STEP[arm] * 1.04))
    if tweak is not None:
        before = record['config_hash']
        tweak(record)
        if record['config_hash'] == before:  # a tweak of the config keeps a consistent hash unless it breaks the hash itself
            record['config_hash'] = stable_hash(record['config'])
    return record


def write_smokes(root, study, arms, *, tweaks=None, **kwargs):
    for arm in arms:
        folder = Path(root) / arm
        folder.mkdir(parents=True)
        (folder / 'record.json').write_text(json.dumps(smoke_record(study, arm, tweak=(tweaks or {}).get(arm), **kwargs)))
    return Path(root)


def rfx_summary(arms, **override):
    ridge = {str(lsq_release.SPECS[a][0]) for a in arms if lsq_release.SPECS[a][1] == 'ridge'}
    rfx_eta = {rank: RFX_ETA[rank] for rank in sorted(ridge)}
    verdicts = {'W5-r32': 'none', **{f'R-r{rank}': 'none' for rank in (32, 128)}}
    for arm in arms:
        rank, store = lsq_release.SPECS[arm]
        verdicts[f'R-r{rank}'] = 'normalization' if store == 'delta' else 'storage'
    summary = dict(smoke=False, source_dirty=False, git_commit='b' * 40, job_id='7.opbs', labels=['R-r128', 'R-r32', 'W5-r32'],
                   verdicts=verdicts, instrumentation_failures={'R-r128': [], 'R-r32': [], 'W5-r32': []},
                   entity_em_guard=dict(slack=.005, passed=True), lsq_arms=list(arms), rfx_eta_by_rank=rfx_eta,
                   eta_by_rank={rank: lsq_release.lsq_eta(eta) for rank, eta in rfx_eta.items()})
    summary.update(override)
    return summary


def lsq_inputs(tmp_path, arms, *, summary=None, tweaks=None, E=16, sized=None):
    """RFX's summary and the smokes of the sized arms (all of RFX's arms unless `sized` names the E8 arm)."""
    summary_path = tmp_path / 'summary.json'
    summary_path.write_text(json.dumps(summary if summary is not None else rfx_summary(arms)))
    root = write_smokes(tmp_path / 'smoke', 'LSQ', arms if sized is None else sized, tweaks=tweaks, E=E)
    return summary_path, root, tmp_path / 'release.json'


@pytest.mark.parametrize('arms', RFX_ARM_SETS)
def test_an_lsq_release_holds_the_rfx_arms_etas_and_the_steps_the_runner_accepts(tmp_path, arms):
    summary_path, root, out = lsq_inputs(tmp_path, arms)
    summary = json.loads(summary_path.read_text())
    made = prepare_lsq.prepare(summary_path, root, out)
    written = json.loads(out.read_text())
    assert written == made and out.read_text().endswith('}\n')
    assert (written['study'], written['E'], written['science_commit']) == ('LSQ', 16, COMMIT)
    assert written['arms'] == summary['lsq_arms'] == arms
    assert written['rfx_eta_by_rank'] == summary['rfx_eta_by_rank'] and written['eta_by_rank'] == summary['eta_by_rank']
    assert written['steps_by_arm'] == {arm: expected_steps(arm) for arm in arms}
    assert written['primary_eval_steps_by_arm'] == {arm: [expected_steps(arm) * k // 5 for k in range(1, 6)] for arm in arms}
    fixed = 1.5 * 5 * 10 * max(EVALS) + 90  # the margin on the five evaluations plus the reserve
    for arm, n in written['steps_by_arm'].items():
        sizing = written['sizing_by_arm'][arm]
        assert sizing['expected_seconds_with_margin'] == pytest.approx(1.5 * n * STEP[arm] + fixed)
        assert 1.5 * n * STEP[arm] + fixed <= 7200 < 1.5 * (n + 5) * STEP[arm] + fixed  # the largest multiple of 5 that fits
        row = next(r for r in written['smoke']['rows'] if r['arm'] == arm)
        assert (row['steady_step_mean_seconds'], row['eval100_seconds'], row['steps']) == (STEP[arm], list(EVALS), 2400)
    # What the runner will do with the release: the same functions it calls.
    assert lsq_release.release_arms(written) == arms
    for arm in arms:
        rank, store = lsq_release.SPECS[arm]
        assert lsq_release.scheduled_steps(arm, None, written) == expected_steps(arm)
        assert lsq_release.release_eta(written, arm) == (summary['eta_by_rank'][str(rank)] if store == 'ridge' else None)
    assert written['smoke']['quality_scores_exposed'] is False and len(written['smoke']['rows']) == len(arms)
    assert written['rfx']['verdicts'] == summary['verdicts'] and len(written['rfx']['summary_blake2b']) == 32


def test_the_release_clips_rfx_eta_and_keeps_both_maps_as_in_the_summary(tmp_path):
    summary_path, root, out = lsq_inputs(tmp_path, STORAGE_BOTH)
    written = prepare_lsq.prepare(summary_path, root, out)
    assert written['rfx_eta_by_rank'] == {'32': 3e-3, '128': 1e-6} and written['eta_by_rank'] == {'32': 3e-3, '128': 1e-4}


def test_a_release_never_contains_a_smoke_score_and_does_not_need_one(tmp_path):
    arms = ['L32-K4', 'L32-RG']
    summary_path, root, out = lsq_inputs(tmp_path, arms)
    with_scores = prepare_lsq.prepare(summary_path, root, out)
    assert str(SENTINEL) not in out.read_text() and str(SENTINEL / 1000) not in out.read_text()
    # Remove every score field from the smoke records: the release is the same.
    for arm in arms:
        path = root / arm / 'record.json'
        record = json.loads(path.read_text())
        for metric in record['training']['metrics'].values():
            for key in ('correct', 'em', 'loss'):
                del metric[key]
        path.write_text(json.dumps(record))
    assert prepare_lsq.prepare(summary_path, root, tmp_path / 'release-without-scores.json') == with_scores


@pytest.mark.parametrize('override, message', [
    (dict(smoke=True), 'smoke summary'),
    (dict(source_dirty=True), 'dirty source'),
    (dict(instrumentation_failures={'R-r128': ['bad'], 'R-r32': [], 'W5-r32': []}), 'instrumentation'),
    (dict(entity_em_guard=dict(slack=.005, passed=False)), 'entity-restricted EM guard'),
    (dict(lsq_arms=[]), 'triggered arms'),
    (dict(lsq_arms=['L32-U', 'L32-K4']), 'rank 32'),
    (dict(lsq_arms=['L128-K4', 'L32-K4', 'L32-RG']), 'candidate order'),
    (dict(rfx_eta_by_rank={}), 'eta maps'),
    (dict(eta_by_rank={}), 'eta maps'),
    (dict(rfx_eta_by_rank={'32': 3e-3, '128': 3e-3}), 'eta maps'),
    (dict(eta_by_rank={'32': 3e-3, '128': 3e-3}), 'eta maps'),
])
def test_an_lsq_release_is_refused_for_a_doubtful_rfx_summary(tmp_path, override, message):
    arms = ['L32-K4', 'L32-RG']
    summary_path, root, out = lsq_inputs(tmp_path, arms, summary=rfx_summary(arms, **override))
    with pytest.raises(ValueError, match=message):
        prepare_lsq.prepare(summary_path, root, out)
    assert not out.exists()


def test_a_summary_without_the_entity_guard_record_is_refused(tmp_path):
    arms = ['L32-K4', 'L32-RG']
    summary = rfx_summary(arms)
    del summary['entity_em_guard']
    summary_path, root, out = lsq_inputs(tmp_path, arms, summary=summary)
    with pytest.raises(ValueError, match='entity-restricted EM guard'):
        prepare_lsq.prepare(summary_path, root, out)
    assert not out.exists()


def test_an_unclipped_eta_in_the_summary_is_refused(tmp_path):
    arms = ['L128-K4', 'L128-RG']
    summary_path, root, out = lsq_inputs(tmp_path, arms, summary=rfx_summary(arms, eta_by_rank={'128': 1e-6}))
    with pytest.raises(ValueError, match='clipped RFX selection'):
        prepare_lsq.prepare(summary_path, root, out)
    assert not out.exists()


def test_the_release_checks_catch_a_release_edited_away_from_the_rfx_summary():
    arms = ['L32-K4', 'L32-RG']
    summary = rfx_summary(arms)
    steps = {arm: expected_steps(arm) for arm in arms}
    good = dict(study='LSQ', E=16, arms=list(arms), steps_by_arm=steps, rfx_eta_by_rank=dict(summary['rfx_eta_by_rank']),
                eta_by_rank=dict(summary['eta_by_rank']))
    prepare_lsq.check_release(good, summary, E=16)
    edits = [
        dict(arms=['L32-RG', 'L32-K4']),  # candidate order
        dict(arms=['L32-K4']),  # an arm dropped
        dict(arms=arms + ['L128-U']),  # an arm added
        dict(arms=['L32-U']),  # the arm of another verdict
        dict(rfx_eta_by_rank={'32': 1e-2}),  # an eta edited
        dict(eta_by_rank={'32': 1e-3}),
        dict(rfx_eta_by_rank={}, eta_by_rank={}),  # the etas dropped
        dict(rfx_eta_by_rank={'32': 3e-3, '128': 3e-3}, eta_by_rank={'32': 3e-3, '128': 3e-3}),  # an eta of a rank without RG
        dict(steps_by_arm={arm: n + 1 for arm, n in steps.items()}),  # not five equal intervals
        dict(steps_by_arm={'L32-K4': steps['L32-K4']}),  # the E16 release names every arm
        dict(steps_by_arm={**steps, 'L128-RG': 15000}),  # a step count for an arm that is not in the release
        dict(study='DPFP'),
    ]
    for edit in edits:
        with pytest.raises(ValueError):
            prepare_lsq.check_release({**good, **edit}, summary, E=16)
    with pytest.raises(ValueError):
        prepare_lsq.check_release(good, summary, E=8)  # an E16 release is not an E8 release
    # Arm U has no eta: neither map may hold one.
    u_summary = rfx_summary(['L128-U'])
    u_release = dict(study='LSQ', E=16, arms=['L128-U'], steps_by_arm={'L128-U': expected_steps('L128-U')},
                     rfx_eta_by_rank={}, eta_by_rank={})
    prepare_lsq.check_release(u_release, u_summary, E=16)
    with pytest.raises(ValueError):
        prepare_lsq.check_release({**u_release, 'rfx_eta_by_rank': {'128': 3e-3}, 'eta_by_rank': {'128': 3e-3}}, u_summary, E=16)


def test_an_existing_release_is_never_overwritten(tmp_path):
    arms = ['L32-U']
    summary_path, root, out = lsq_inputs(tmp_path, arms)
    out.write_text('frozen')
    with pytest.raises(FileExistsError):
        prepare_lsq.prepare(summary_path, root, out)
    with pytest.raises(FileExistsError):  # refused before any input is read
        prepare_lsq.prepare(tmp_path / 'absent-summary.json', root, out)
    assert out.read_text() == 'frozen'
    out.unlink()
    made = prepare_lsq.prepare(summary_path, root, out)
    with pytest.raises(FileExistsError):
        prepare_lsq.prepare(summary_path, root, out)
    assert json.loads(out.read_text()) == made
    with pytest.raises(FileExistsError):
        release.write_new(out, {'other': 1})
    assert json.loads(out.read_text()) == made


COMMON_TWEAKS = [
    (lambda r: r.update(status='failed'), 'complete smoke record'),
    (lambda r: r.update(smoke=False), 'complete smoke record'),
    (lambda r: r.update(arm='other'), 'complete smoke record'),
    (lambda r: r.update(source_dirty=True), 'dirty source'),
    (lambda r: r.update(config_hash='bad'), 'config hash'),
    (lambda r: r.update(git_commit='abc123'), '40-hex commit'),
    (lambda r: r['config'].update(smoke=False), '100-episode smoke'),
    (lambda r: r['config']['task'].update(E=8), '100-episode smoke at E16'),
    (lambda r: r['config']['eval'].update(episodes=1000), '100-episode smoke'),
    (lambda r: r['training'].update(actual_steps=2399), 'five evaluations'),
    (lambda r: r['training'].update(training_episodes=1), 'five evaluations'),
    (lambda r: r['training'].update(matched_eval_steps=[6580]), 'five evaluations'),
    (lambda r: r['training'].update(eval_steps=[480, 960, 1440, 1920]), 'five evaluations'),
    (lambda r: r['training']['metrics'].pop('960'), 'one 100-episode evaluation'),
    (lambda r: r['training']['metrics']['960'].update(n=1000), 'one 100-episode evaluation'),
    (lambda r: r['training'].update(steady_step_mean_seconds=float('nan')), 'positive and finite'),
    (lambda r: r['training'].update(steady_step_mean_seconds=True), 'positive and finite'),
    (lambda r: r['training'].update(step_p95_seconds=0), 'positive and finite'),
    (lambda r: r['training']['metrics']['960'].update(seconds=-1.), 'positive and finite'),
    (lambda r: r['training']['metrics']['960'].update(seconds=1000.), 'no training budget'),
]
LSQ_TWEAKS = [
    (lambda r: r.update(rank=128 if r['rank'] == 32 else 32), 'belongs to another arm'),
    (lambda r: r.update(store='delta' if r['store'] != 'delta' else 'sweep'), 'belongs to another arm'),
    (lambda r: r['config']['lsq'].update(arm='L128-U'), 'belongs to another arm'),
    (lambda r: r.update(store_eta=3e-3 if r['store_eta'] is None else None), 'only the RG arms take an eta'),
]
DPFP_TWEAKS = [
    (lambda r: r.update(rank=128 if r['rank'] == 32 else 32), 'belongs to another arm'),
    (lambda r: r.update(dpfp_nu=2), 'belongs to another arm'),
    (lambda r: r['config']['dpfp'].update(arm='D128-3'), 'belongs to another arm'),
]


@pytest.mark.parametrize('tweak, message', COMMON_TWEAKS + LSQ_TWEAKS)
def test_an_lsq_smoke_record_that_is_not_the_arms_ten_percent_smoke_is_refused(tmp_path, tweak, message):
    arms = ['L32-K4', 'L32-RG']
    summary_path, root, out = lsq_inputs(tmp_path, arms, tweaks={'L32-RG': tweak})
    with pytest.raises(ValueError, match=message):
        prepare_lsq.prepare(summary_path, root, out)
    assert not out.exists()


@pytest.mark.parametrize('tweak, message', COMMON_TWEAKS + DPFP_TWEAKS)
def test_a_dpfp_smoke_record_that_is_not_the_arms_ten_percent_smoke_is_refused(tmp_path, tweak, message):
    root = write_smokes(tmp_path / 'smoke', 'DPFP', dpfp_release.ARMS, tweaks={'D128-1': tweak})
    with pytest.raises(ValueError, match=message):
        prepare_dpfp.prepare(root, tmp_path / 'release.json')
    assert not (tmp_path / 'release.json').exists()


def test_smokes_at_different_commits_or_a_missing_smoke_are_refused(tmp_path):
    arms = ['L32-K4', 'L32-RG']
    summary_path, root, out = lsq_inputs(tmp_path, arms, tweaks={'L32-RG': lambda r: r.update(git_commit='c' * 40)})
    with pytest.raises(ValueError, match='different commits'):
        prepare_lsq.prepare(summary_path, root, out)
    (root / 'L32-RG' / 'record.json').unlink()
    with pytest.raises(FileNotFoundError):
        prepare_lsq.prepare(summary_path, root, out)
    root = write_smokes(tmp_path / 'dpfp', 'DPFP', ['D32-3', 'D128-1'])
    with pytest.raises(FileNotFoundError):
        prepare_dpfp.prepare(root, tmp_path / 'dpfp-release.json')
    assert not out.exists() and not (tmp_path / 'dpfp-release.json').exists()


def test_a_dpfp_release_names_all_three_arms_and_the_runner_accepts_it(tmp_path):
    root = write_smokes(tmp_path / 'smoke', 'DPFP', dpfp_release.ARMS)
    out = tmp_path / 'sub' / 'release.json'
    made = prepare_dpfp.prepare(root, out)
    written = json.loads(out.read_text())
    assert written == made and (written['study'], written['E'], written['science_commit']) == ('DPFP', 16, COMMIT)
    assert written['steps_by_arm'] == {arm: expected_steps(arm) for arm in dpfp_release.ARMS}
    assert written['primary_eval_steps_by_arm']['D32-3'] == [expected_steps('D32-3') * k // 5 for k in range(1, 6)]
    for arm in dpfp_release.ARMS:
        assert dpfp_release.scheduled_steps(arm, None, written) == expected_steps(arm)
    assert str(SENTINEL) not in out.read_text() and written['smoke']['quality_scores_exposed'] is False
    with pytest.raises(FileExistsError):
        prepare_dpfp.prepare(root, out)
    with pytest.raises(ValueError):
        prepare_dpfp.check_release({**written, 'steps_by_arm': {'D32-3': 20000}}, E=16)  # E16 fixes all three arms at once
    with pytest.raises(ValueError):
        prepare_dpfp.check_release(written, E=8)


def lock(arm, action='retry_e8'):
    return dict(choice=dict(action=action, arm=arm if action != 'stop' else None, E=8 if action == 'retry_e8' else None,
                            correct={}))


def test_an_e8_release_holds_only_the_retrained_arm(tmp_path):
    root = write_smokes(tmp_path / 'smoke-e8', 'DPFP', ['D128-3'], E=8)
    lock_path = tmp_path / 'lock.json'
    lock_path.write_text(json.dumps(lock('D128-3')))
    made = prepare_dpfp.prepare(root, tmp_path / 'e8.json', e8_decision=lock_path)
    assert (made['study'], made['E'], list(made['steps_by_arm'])) == ('DPFP', 8, ['D128-3'])
    assert dpfp_release.scheduled_steps('D128-3', None, made, E=8) == expected_steps('D128-3')
    for bad, error in ((lock('D128-3', 'select'), ValueError), (lock(None, 'stop'), ValueError),
                       (lock('D32-3'), FileNotFoundError)):  # no retry; no retry; the arm has no E8 smoke here
        lock_path.write_text(json.dumps(bad))
        with pytest.raises(error):
            prepare_dpfp.prepare(root, tmp_path / 'e8-bad.json', e8_decision=lock_path)
    assert not (tmp_path / 'e8-bad.json').exists()
    lock_path.write_text(json.dumps(lock('D128-3')))
    with pytest.raises(ValueError, match='100-episode smoke at E8'):  # an E16 smoke does not size an E8 retrain
        prepare_dpfp.prepare(write_smokes(tmp_path / 'smoke-e16', 'DPFP', ['D128-3']), tmp_path / 'e8-e16.json',
                             e8_decision=lock_path)


def test_an_lsq_e8_release_keeps_the_rfx_arms_and_etas_and_holds_only_the_retrained_arm(tmp_path):
    summary_path, root, out = lsq_inputs(tmp_path, STORAGE_BOTH, E=8, sized=['L128-RG'])
    lock_path = tmp_path / 'lock.json'
    lock_path.write_text(json.dumps(lock('L128-RG')))
    made = prepare_lsq.prepare(summary_path, root, out, e8_decision=lock_path)
    summary = json.loads(summary_path.read_text())
    assert (made['study'], made['E'], made['arms']) == ('LSQ', 8, STORAGE_BOTH)
    assert list(made['steps_by_arm']) == ['L128-RG'] and made['eta_by_rank'] == summary['eta_by_rank']
    assert lsq_release.scheduled_steps('L128-RG', None, made, E=8) == expected_steps('L128-RG')
    assert lsq_release.release_eta(made, 'L128-RG') == 1e-4
    for bad in (lock('L32-U'), lock('L32-K4'), lock('L128-RG', 'select'), lock(None, 'stop')):  # not RFX's, not this smoke, no retry
        lock_path.write_text(json.dumps(bad))
        with pytest.raises((ValueError, FileNotFoundError)):
            prepare_lsq.prepare(summary_path, root, tmp_path / 'e8-bad.json', e8_decision=lock_path)
    assert not (tmp_path / 'e8-bad.json').exists()


def test_an_e8_arm_that_rfx_did_not_trigger_is_refused_even_with_a_smoke(tmp_path):
    summary_path, root, out = lsq_inputs(tmp_path, ['L32-K4', 'L32-RG'], E=8, sized=['L128-RG'])  # a smoke of another rank's arm
    lock_path = tmp_path / 'lock.json'
    lock_path.write_text(json.dumps(lock('L128-RG')))
    with pytest.raises(ValueError, match='not one of the arms RFX triggered'):
        prepare_lsq.prepare(summary_path, root, out, e8_decision=lock_path)
    assert not out.exists()


def test_prepare_gates_the_release_through_the_checks_the_runner_makes(tmp_path, monkeypatch):
    # A release that the runner would refuse is never written, whatever produced it.
    summary_path, root, out = lsq_inputs(tmp_path, ['L32-K4', 'L32-RG'])
    real = prepare_lsq.schedule_fields
    monkeypatch.setattr(prepare_lsq, 'schedule_fields', lambda rows: {**real(rows), 'steps_by_arm': {a: 20001 for a in rows}})
    with pytest.raises(ValueError, match='five equal integer intervals'):
        prepare_lsq.prepare(summary_path, root, out)
    root = write_smokes(tmp_path / 'dpfp', 'DPFP', dpfp_release.ARMS)
    real = prepare_dpfp.schedule_fields
    monkeypatch.setattr(prepare_dpfp, 'schedule_fields', lambda rows: {**real(rows), 'steps_by_arm': {a: 20001 for a in rows}})
    with pytest.raises(ValueError, match='five equal integer intervals'):
        prepare_dpfp.prepare(root, tmp_path / 'dpfp-release.json')
    assert not out.exists() and not (tmp_path / 'dpfp-release.json').exists()


def test_the_arm_sets_rfx_can_trigger_are_exactly_the_ones_a_release_may_name():
    arms = lsq_release.ARMS
    per_rank = {'none': [], 'normalization': ['U'], 'storage': ['K4', 'RG']}  # an R checkpoint's verdict -> its rank's arms
    triggered = {tuple(f'L{rank}-{name}' for rank, verdict in ((32, v32), (128, v128)) for name in per_rank[verdict])
                 for v32 in per_rank for v128 in per_rank} - {()}
    assert len(triggered) == 8
    for size in range(len(arms) + 1):
        for subset in itertools.combinations(arms, size):  # candidate order
            if subset in triggered:
                assert lsq_release.check_arms(subset) == list(subset)
            else:
                with pytest.raises(ValueError):
                    lsq_release.check_arms(subset)
            if len(subset) > 1:  # any other order is refused
                with pytest.raises(ValueError):
                    lsq_release.check_arms(subset[::-1])
    for bad in (['L32-K4', 'L32-K4', 'L32-RG'], ['L33-K4'], 'L32-U', None):
        with pytest.raises((ValueError, TypeError)):
            lsq_release.check_arms(bad)


@pytest.mark.parametrize('script, study', [('prepare_lsq_release', 'LSQ'), ('prepare_dpfp_release', 'DPFP')])
def test_the_scripts_run_without_torch_or_numpy(tmp_path, script, study):
    # The login node runs them: with torch and numpy made unimportable, the command line must still produce the release.
    if study == 'LSQ':
        arms = ['L32-K4', 'L32-RG']
        summary_path, root, out = lsq_inputs(tmp_path, arms)
        arguments = ['--summary', str(summary_path), '--root', str(root), '--out', str(out)]
    else:
        arms = list(dpfp_release.ARMS)
        root, out = write_smokes(tmp_path / 'smoke', 'DPFP', arms), tmp_path / 'release.json'
        arguments = ['--root', str(root), '--out', str(out)]
    code = ("import runpy, sys\nsys.modules['torch'] = None\nsys.modules['numpy'] = None\n"
            "script = sys.argv[1]\nsys.argv = sys.argv[1:]\nrunpy.run_path(script, run_name='__main__')")
    done = subprocess.run([sys.executable, '-c', code, str(ROOT / 'scripts' / f'{script}.py'), *arguments],
                          env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    printed = json.loads(done.stdout.strip().splitlines()[-1])
    released = json.loads(out.read_text())
    assert printed['study'] == study == released['study'] and printed['steps_by_arm'] == released['steps_by_arm']
    assert printed['quality_scores_exposed'] is False and set(released['steps_by_arm']) == set(arms)
    again = subprocess.run([sys.executable, '-c', code, str(ROOT / 'scripts' / f'{script}.py'), *arguments],
                           env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, capture_output=True, text=True, cwd=tmp_path)
    assert again.returncode != 0 and 'FileExistsError' in again.stderr  # the second run refuses to overwrite


def test_the_sizing_rule_is_wp_l4s():
    from looplora.llm.read_fix_run import largest_schedule  # imports torch: this test needs the compute environment
    for step in (0.05, 0.15, 0.19, 0.2, 0.5, 1.3):
        for evaluation in (0.5, 3.0, 10.7, 40.0):
            assert release.largest_schedule(step, evaluation) == largest_schedule(step, evaluation)
            assert release.largest_schedule(step, evaluation, evaluations=3, wall_seconds=3600, margin=1.2, reserve=30) == \
                largest_schedule(step, evaluation, evaluations=3, wall_seconds=3600, margin=1.2, reserve=30)
    for bad in ((float('nan'), 1.), (0.2, 10000.), (0., 1.), (0.2, float('inf')), (-1., 1.)):
        with pytest.raises(ValueError):
            release.largest_schedule(*bad)
        with pytest.raises(ValueError):
            largest_schedule(*bad)


def test_the_study_registries_are_the_registered_candidate_orders():
    assert dpfp_release.ARMS == ('D32-3', 'D128-1', 'D128-3')
    assert [dpfp_release.SPECS[a] for a in dpfp_release.ARMS] == [(32, 3), (128, 1), (128, 3)]
    assert lsq_release.ARMS == ('L32-U', 'L32-K4', 'L32-RG', 'L128-U', 'L128-K4', 'L128-RG')
    assert [lsq_release.SPECS[a] for a in lsq_release.ARMS] == [
        (32, 'delta'), (32, 'sweep'), (32, 'ridge'), (128, 'delta'), (128, 'sweep'), (128, 'ridge')]
    assert lsq_release.ETA_BOUNDS == (1e-4, 1e-1)
