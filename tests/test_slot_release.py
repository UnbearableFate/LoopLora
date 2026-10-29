"""Release self-checks on synthetic metadata; run through scripts/submit.py on compute."""
import importlib.util
import json
from pathlib import Path

import pytest

from looplora.infra.seeds import stable_hash
from looplora.llm import slot_data, slot_rules

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_slot_release', ROOT / 'scripts' / 'prepare_slot_release.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
COMMIT = 'c' * 40


def dump(path, row):
    path.write_text(json.dumps(row))
    return path


@pytest.fixture
def inputs(tmp_path):
    """The full registered grids, with tie cases and measured times independent of selected scores."""
    ids = {f'eval/{stage}/{purpose}': dict(count=slot_data.SETS[stage, purpose].count, ids_blake2b_128=stable_hash([stage, purpose]))
           for stage, purpose in (('S0', 'calibration'), ('S0', 'dev'), ('S1A', 'pilot'))}
    lock = dict(status='locked', prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128, preflight=dict(passed=True),
                config_hashes={stage: stable_hash(slot_rules.stage_config(stage)) for stage in ('S0', 'S1A')},
                inventory={'S-RE': {'writer.pt': 'a' * 32, 'checkpoint.pt': 'b' * 32}},
                manifest=dict(ids=ids, sets={'S1A/pilot': dict(count=1000)}))
    candidates = [dict(row, nll=1.0) for row in slot_rules.calibration_grid()]
    selected = dict(matrix=dict(gain=1.0), soft=dict(gain=1.0, tau=1.0), hard=dict(gain=1.0))
    s0 = dict(stage='S0', config_hash=lock['config_hashes']['S0'], n=1000,
              ids_blake2b_128=ids['eval/S0/dev']['ids_blake2b_128'],
              calibration=dict(n=2048, ids_blake2b_128=ids['eval/S0/calibration']['ids_blake2b_128'],
                               candidates=candidates, selected=selected))
    runs = [dict(arm=arm, lr=lr, all_finite=True, nll=1.0, step_seconds_mean=0.2,
                 step0=dict(step=0, nll=2.0, n=1000, seconds=6.0),
                 final=dict(step=200, nll=1.0, n=1000, seconds=8.0))
            for arm in slot_rules.ARMS for lr in slot_rules.LR_GRID]
    steps = slot_rules.schedule_steps(0.2, 8.0, 30.0)
    pilot = dict(stage='S1A-pilot', config_hash=lock['config_hashes']['S1A'], stop=None, runs=runs,
                 shard_ids_blake2b_128=ids['eval/S1A/pilot']['ids_blake2b_128'],
                 lr={arm: 3e-4 for arm in slot_rules.ARMS}, step_seconds=0.2, eval_seconds=8.0, load_seconds=30.0,
                 steps=steps, walltime_seconds=slot_rules.job_walltime_seconds(0.2, 8.0, 30.0, steps))
    arms = {}
    for arm, kind in slot_rules.ARM_KIND.items():
        read = dict(kind=kind, gain=1.0, tau=1.0 if kind == 'soft' else None,
                    scope='all_valid_question', recency=False, capacity=16)
        freeze = dict(trainable=dict(trainable=list(slot_rules.TRAINABLE), frozen=list(slot_rules.FROZEN),
                                     buffers=list(slot_rules.BUFFERS), trainable_parameters=180224, frozen_parameters=294914),
                      digests=dict(frozen_unchanged=True, trainable_changed=True,
                                   before={name: 'a' * 32 for name in (*slot_rules.FROZEN, *slot_rules.BUFFERS, *slot_rules.TRAINABLE)},
                                   after={name: ('b' if name in slot_rules.TRAINABLE else 'a') * 32
                                          for name in (*slot_rules.FROZEN, *slot_rules.BUFFERS, *slot_rules.TRAINABLE)}),
                      support=dict(equal=True, before={'keys': 'e' * 32}, after={'keys': 'e' * 32}),
                      gradients=dict(trainable_grad_norms={name: 1.0 for name in slot_rules.TRAINABLE}))
        export_config = dict(arm=arm, read=read, loop_steps=1, query_projection='shared',
                             run=dict(stage='smoke', seed=None), key_dim=768, value_dim=128,
                             feature_map='dpfp-nu3-unit', eps=1e-6, source_commit=COMMIT,
                             trainable=list(slot_rules.TRAINABLE), layout_digest=slot_data.LAYOUT_DIGEST,
                             task=json.loads(json.dumps(slot_data.TASK)), prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128,
                             parent=dict(writer_blake2b_128='a' * 32, checkpoint_blake2b_128='b' * 32))
        arms[arm] = dict(read=read, all_finite=True, freeze=freeze, export_config=export_config,
                         selected_checkpoint_blake2b_128=stable_hash(arm))
    smoke = dict(stage='S1A-smoke', config_hash=lock['config_hashes']['S1A'], steps=20, lr=1e-3, arms=arms)
    rows = dict(manifest=lock, pilot=pilot, s0_metrics=s0, smoke=smoke)
    paths = {name: dump(tmp_path / f'{name}.json', row) for name, row in rows.items()}
    return rows, paths


def build(inputs, **kwargs):
    return builder.prepare(**inputs[1], science_commit=COMMIT, **kwargs)


def reloads(inputs, tmp_path):
    reports = {}
    for arm, row in inputs[0]['smoke']['arms'].items():
        report = dict(passed=True, config_matches_release=True, slot=row['export_config'], read=row['read'],
                      checkpoint_blake2b_128=row['selected_checkpoint_blake2b_128'],
                      equal={key: True for key in ('episode_id', 'prediction', 'correct', 'nll')})
        reports[arm] = dump(tmp_path / f'reload-{arm}.json', report)
    return reports


def test_candidate_and_ready_release_bind_all_eight_science_configs_and_input_digests(inputs, tmp_path):
    candidate = build(inputs, out=tmp_path / 'candidate.json')
    assert candidate['status'] == 'pending_reload' and not candidate['reload_checks']
    assert len(candidate['config_hashes']) == 8
    assert {key for key in candidate['config_hashes'] if key.endswith('/ablation')} == {
        f'{arm}/17/ablation' for arm in slot_rules.ARMS}
    assert candidate['config_hashes'] == {key: slot_rules.science_hash(row) for key, row in candidate['science_configs'].items()}
    assert all(row['steps'] == candidate['steps'] and row['lr'] == candidate['arms'][row['arm']]['lr']
               for row in candidate['science_configs'].values())
    for key, row in candidate['science_configs'].items():
        assert row['ablation'] == key.endswith('/ablation')
        assert row['trainable'] == list(slot_rules.ABLATION_TRAINABLE if row['ablation'] else slot_rules.TRAINABLE)
    assert candidate['manifest_blake2b_128'] == slot_data.file_digest(inputs[1]['manifest'])
    assert candidate['inventory_blake2b_128'] == stable_hash(inputs[0]['manifest']['inventory'])
    assert candidate['analysis_script_blake2b_128'] == slot_data.file_digest(ROOT / slot_data.ANALYSIS_SCRIPT)
    final = build(inputs, reload_checks=reloads(inputs, tmp_path), out=tmp_path / 'ready.json')
    assert final['status'] == 'ready' and set(final['reload_checks']) == set(slot_rules.ARMS)
    assert final['science_configs'] == candidate['science_configs']
    assert json.loads((tmp_path / 'ready.json').read_text())['science_commit'] == COMMIT
    with pytest.raises(ValueError, match='never overwritten'):
        build(inputs, out=tmp_path / 'ready.json')


@pytest.mark.parametrize('defect', ['lock', 'preflight', 'prereg', 'stage', 'ids', 'count', 'calibration', 'grid',
                                   'lr', 'pilot_steps', 'pilot_count', 'pilot_grid', 'timing', 'schedule', 'walltime',
                                   'stopped', 'smoke_steps', 'smoke_lr', 'read', 'frozen', 'support', 'gradient', 'export_parent'])
def test_inconsistent_or_incomplete_inputs_are_refused(inputs, defect):
    rows, paths = inputs
    lock, pilot, s0, smoke = [rows[name] for name in ('manifest', 'pilot', 's0_metrics', 'smoke')]
    arm = smoke['arms']['matrix_L1']
    if defect == 'lock': lock['status'] = 'pending'
    elif defect == 'preflight': lock['preflight']['passed'] = False
    elif defect == 'prereg': lock['prereg_blake2b_128'] = 'f' * 32
    elif defect == 'stage': pilot['config_hash'] = 'f' * 32
    elif defect == 'ids': s0['ids_blake2b_128'] = 'f' * 32
    elif defect == 'count': s0['calibration']['n'] -= 1
    elif defect == 'calibration': s0['calibration']['selected']['soft']['tau'] = 0.03
    elif defect == 'grid': s0['calibration']['candidates'].pop()
    elif defect == 'lr': pilot['lr']['matrix_L1'] = 1e-3
    elif defect == 'pilot_steps': pilot['runs'][0]['final']['step'] -= 1
    elif defect == 'pilot_count': pilot['runs'][0]['final']['n'] -= 1
    elif defect == 'pilot_grid': pilot['runs'][0]['lr'] = 1e-5
    elif defect == 'timing': pilot['step_seconds'] = 0.1
    elif defect == 'schedule': pilot['steps'] -= 10
    elif defect == 'walltime': pilot['walltime_seconds'] += 300
    elif defect == 'stopped': pilot['stop'] = 'no finite candidates'
    elif defect == 'smoke_steps': smoke['steps'] = 19
    elif defect == 'smoke_lr': smoke['lr'] = 3e-3
    elif defect == 'read': arm['read']['scope'] = 'final_only'
    elif defect == 'frozen': arm['freeze']['digests']['frozen_unchanged'] = False
    elif defect == 'support': arm['freeze']['support']['after'] = {}
    elif defect == 'gradient': arm['freeze']['gradients']['trainable_grad_norms']['Q'] = 0
    elif defect == 'export_parent': arm['export_config']['parent']['writer_blake2b_128'] = 'f' * 32
    for name, row in rows.items(): dump(paths[name], row)
    with pytest.raises((ValueError, slot_rules.StopForOrchestrator)):
        build(inputs)


@pytest.mark.parametrize('defect', ['missing_arm', 'passed', 'config_matches_release', 'slot', 'read', 'checkpoint', 'predictions'])
def test_final_release_requires_both_reload_reports_and_exact_smoke_settings(inputs, tmp_path, defect):
    checks = reloads(inputs, tmp_path)
    path = checks['soft_slot_L1']
    report = json.loads(path.read_text())
    if defect == 'missing_arm': checks.pop('matrix_L1')
    elif defect in ('passed', 'config_matches_release'): report[defect] = False
    elif defect == 'slot': report['slot']['query_projection'] = 'copy'
    elif defect == 'read': report['read']['gain'] = 2.0
    elif defect == 'checkpoint': report['checkpoint_blake2b_128'] = 'f' * 32
    elif defect == 'predictions': report['equal']['nll'] = False
    dump(path, report)
    with pytest.raises(ValueError, match='reload'):
        build(inputs, reload_checks=checks)


def test_a_nonfinite_pilot_candidate_is_preserved_but_cannot_win_the_release(inputs):
    rows, paths = inputs
    pilot = rows['pilot']
    pilot['runs'][0] = dict(arm='matrix_L1', lr=3e-4, all_finite=False, nll=None, error='non-finite gradient')
    pilot['lr']['matrix_L1'] = 1e-3
    dump(paths['pilot'], pilot)
    release = build(inputs)
    assert release['arms']['matrix_L1']['lr'] == 1e-3
    assert release['inputs_blake2b_128']['pilot'] == slot_data.file_digest(paths['pilot'])


def test_a_short_commit_and_missing_required_input_are_refused(inputs):
    with pytest.raises(ValueError, match='40-hex'):
        builder.prepare(**inputs[1], science_commit='HEAD')
    inputs[1]['smoke'].unlink()
    with pytest.raises(FileNotFoundError):
        build(inputs)
