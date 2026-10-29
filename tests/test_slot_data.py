"""SLOT-LOOP-v1: the data registry, the purpose-seed guard, the registered rules and the manifest script
(docs/prereg/SLOT-LOOP-v1.md, "Data and manifest", "Calibration", "LR pilot", "Checkpoints and selection").

No model and no GPU. Backend semantics are in `test_slot_backend.py`; the slot read path, the freeze verification, the trainer and
the S0 pipeline are in `test_slot_experiment.py`.
"""
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest

from looplora.infra.seeds import derive_seed
from looplora.llm import slot_data, slot_rules
from looplora.llm.config import CONFIRMATION_WINDOWS
from looplora.llm.slot_rules import StopForOrchestrator

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_slot_manifest', ROOT / 'scripts' / 'prepare_slot_manifest.py')
manifest_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest_script)


# ----------------------------------------------------------------------------------------------------------- the registry

def test_the_registered_integers_are_the_derived_ones():
    slot_data.verify_registry()
    assert slot_data.SETS['S1A', 'dev'].base_seed == derive_seed(20261029, 'slot-loop-v1', 'dev', 'S1A')
    assert slot_data.TRAIN_BASE['S1A', 'pilot'] == derive_seed(20261029, 'slot-loop-v1', 'train', 'S1A', 'pilot')
    assert slot_data.BOOTSTRAP['S1A'] == 9056755664197379988


def test_a_changed_registered_integer_is_refused(monkeypatch):
    monkeypatch.setitem(slot_data.TRAIN_BASE, ('S1A', 17), slot_data.TRAIN_BASE['S1A', 17] + 1)
    with pytest.raises(ValueError):
        slot_data.verify_registry()


def test_every_set_and_train_stream_is_disjoint_from_the_old_protected_ids_and_from_each_other():
    rows = slot_data.check_disjoint()
    assert {k: v['count'] for k, v in rows.items()} == {
        'eval/S0/smoke': 64, 'eval/S0/calibration': 2048, 'eval/S0/dev': 1000, 'eval/S1A/smoke': 64, 'eval/S1A/pilot': 1000,
        'eval/S1A/dev': 1000, 'eval/S1A/confirmation': 2000, 'eval/S1A/unseen': 2000, 'train/S1A/17': 192000,
        'train/S1A/29': 192000, 'train/S1A/43': 192000, 'train/S1A/pilot': 6400}
    assert len({v['ids_blake2b_128'] for v in rows.values()}) == len(rows)


@pytest.mark.parametrize('victim', ['old_train', 'old_window', 'other_set'])
def test_an_overlap_is_found(monkeypatch, victim):
    if victim == 'old_train':  # a train stream on the D128-3 / S-RE training seed
        monkeypatch.setitem(slot_data.TRAIN_BASE, ('S1A', 17), slot_data.OLD_TRAIN[0][0])
    elif victim == 'old_window':  # an evaluation set on the old val seed and window
        monkeypatch.setitem(slot_data.SETS, ('S0', 'dev'), slot_data.EvalSet('S0', 'dev', 'val', 0, 1000, 4242))
        monkeypatch.setattr(slot_data, '_SETS', tuple(slot_data.SETS.values()))
    else:  # two registered sets on one seed
        other = slot_data.SETS['S1A', 'dev']
        monkeypatch.setitem(slot_data.SETS, ('S1A', 'pilot'), slot_data.EvalSet('S1A', 'pilot', other.split, 0, 1000, other.base_seed))
        monkeypatch.setattr(slot_data, '_SETS', tuple(slot_data.SETS.values()))
    with pytest.raises(ValueError):
        slot_data.check_disjoint(max_steps=2)


def test_the_old_windows_are_pinned_and_this_study_adds_none():
    assert CONFIRMATION_WINDOWS == {'p1l': 100000, 'p1l-s': 110000, 'p5l': 104000, 'lsr': 106000, 'b8w': 108000}
    assert slot_data.OLD_WINDOWS == (('val', 4242, 0, 3000), ('val', 4242, 200000, 4000), ('test', 4242, 100000, 12000))
    assert {s.base_seed for s in slot_data.SETS.values()}.isdisjoint(slot_data.OLD_BASE_SEEDS)


def test_the_manifest_digest_is_reproducible_and_depends_on_the_ids():
    ids = slot_data.SETS['S0', 'dev'].episode_ids()
    assert slot_data.ids_digest(ids) == slot_data.ids_digest(list(ids)) == slot_data.manifest()['ids']['eval/S0/dev']['ids_blake2b_128']
    assert len({slot_data.ids_digest(ids), slot_data.ids_digest(ids[:-1]), slot_data.ids_digest(ids[::-1])}) == 3


# ----------------------------------------------------------------------------------------------- the purpose-seed guard

def _sealed():
    return [slot_data.SETS['S1A', 'confirmation'].base_seed, slot_data.SETS['S1A', 'unseen'].base_seed]


def test_the_sealed_seeds_occur_in_the_registry_file_only():
    sealed, found = [str(seed) for seed in _sealed()], []
    for folder in ('src', 'scripts', 'tests', 'configs'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml', '.toml', '.md', '.sh', '.pbs'}:
                text = path.read_text(errors='replace')
                if any(seed in text for seed in sealed):
                    found.append(str(path.relative_to(ROOT)))
    assert found == ['src/looplora/llm/slot_data.py']


def test_the_stage_configs_name_the_sealed_sets_without_their_seeds():
    for stage in ('S0', 'S1A'):
        text = json.dumps(slot_rules.stage_config(stage))
        assert not any(str(seed) in text for seed in _sealed())
    sets = slot_rules.stage_config('S1A')['sealed_sets']
    assert set(sets) == {'confirmation', 'unseen'} and all(v['sealed'] and 'base_seed' not in v for v in sets.values())


def test_a_sealed_set_needs_the_open_token_and_a_named_checkpoint(tmp_path):
    for purpose in ('confirmation', 'unseen'):
        with pytest.raises(slot_data.SealedSetError):
            slot_data.eval_set('S1A', purpose)
        with pytest.raises(slot_data.SealedSetError):
            slot_data.eval_set('S1A', purpose, token=None, checkpoint_digest='a' * 32)
    assert slot_data.eval_set('S1A', 'dev').count == 1000  # an ordinary set needs none

    (tmp_path / 'docs' / 'prereg').mkdir(parents=True)
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'reports' / 'slot-loop-v1').mkdir(parents=True)
    shutil.copy(ROOT / slot_data.PREREG, tmp_path / slot_data.PREREG)
    (tmp_path / slot_data.ANALYSIS_SCRIPT).write_text('print("frozen analysis")\n')
    selected = {f'{arm}/{seed}': f'{i:032x}' for i, (arm, seed) in enumerate((a, s) for a in slot_rules.ARMS for s in slot_data.RUN_SEEDS)}
    record = dict(prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128, selected_checkpoints=selected,
                  analysis_script_blake2b_128=slot_data.file_digest(tmp_path / slot_data.ANALYSIS_SCRIPT))
    release = dict(status="ready", prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128,
                   analysis_script_blake2b_128=record["analysis_script_blake2b_128"],
                   parent=dict(writer_blake2b_128="e" * 32),
                   arms={arm: dict(read={"kind": kind}) for arm, kind in slot_rules.ARM_KIND.items()})
    path = tmp_path / slot_data.OPEN_RECORD
    path.write_text(json.dumps(record))
    token = slot_data.load_open_token(tmp_path, {"S-RE": {"writer.pt": "e" * 32}}, release)
    assert token.selected_checkpoints == selected
    chosen = next(iter(selected.values()))
    assert slot_data.eval_set("S1A", "confirmation", token=token, checkpoint_digest="e" * 32).opened
    assert slot_data.eval_set('S1A', 'confirmation', token=token, checkpoint_digest=chosen).count == 2000
    assert slot_data.eval_set('S1A', 'unseen', token=token, checkpoint_digest=chosen).split == 'test_unseen'
    with pytest.raises(slot_data.SealedSetError):  # a checkpoint the record does not name
        slot_data.eval_set('S1A', 'confirmation', token=token, checkpoint_digest='f' * 32)
    with pytest.raises(slot_data.SealedSetError):
        slot_data.eval_set('S1A', 'confirmation', token=token, checkpoint_digest=None)

    for broken in (dict(record, selected_checkpoints=dict(list(selected.items())[:5])),  # five of six checkpoints
                   dict(record, prereg_blake2b_128='0' * 32),
                   dict(record, analysis_script_blake2b_128='0' * 32),
                   dict(record, selected_checkpoints={**selected, 'matrix_L1/17': 'short'})):
        path.write_text(json.dumps(broken))
        with pytest.raises(slot_data.SealedSetError):
            slot_data.load_open_token(tmp_path, {"S-RE": {"writer.pt": "e" * 32}}, release)
    path.write_text(json.dumps(record))
    (tmp_path / slot_data.ANALYSIS_SCRIPT).write_text('print("edited after the record")\n')
    with pytest.raises(slot_data.SealedSetError):
        slot_data.load_open_token(tmp_path, {"S-RE": {"writer.pt": "e" * 32}}, release)


def test_the_registered_prereg_is_the_file_on_disk():
    assert slot_data.file_digest(ROOT / slot_data.PREREG) == slot_data.PREREG_BLAKE2B_128


@pytest.mark.skipif(not Path(slot_data.PARENT_FOLDERS['S-RE']).exists(), reason='the parents live on the cluster file system')
def test_the_parent_inventory_matches_the_registered_d128_3_digests():
    inventory = slot_data.inventory()
    assert set(inventory) == {'S-RE', 'D128-3'} and all(len(d) == 32 for files in inventory.values() for d in files.values())
    assert {f: inventory['D128-3'][f] for f in slot_data.D128_3['digests']} == dict(slot_data.D128_3['digests'])


# ------------------------------------------------------------------------------------------------------------- the rules

def row(kind, nll, gain=1.0, tau=None, **extra):
    return dict(kind=kind, nll=nll, gain=gain, tau=tau, **extra)


def test_the_calibration_grid_is_the_registered_eighteen():
    grid = slot_rules.calibration_grid()
    assert len(grid) == 18 and [c['kind'] for c in grid].count('soft') == 12
    assert {(c['kind'], c['gain'], c['tau']) for c in grid} == (
        {('matrix', g, None) for g in (0.5, 1.0, 2.0)} | {('hard', g, None) for g in (0.5, 1.0, 2.0)}
        | {('soft', g, t) for g in (0.5, 1.0, 2.0) for t in (0.03, 0.1, 0.3, 1.0)})


def test_the_calibration_choice_is_the_lowest_nll_and_ties_go_to_the_gain_nearest_one_then_the_larger_tau():
    rows = [row('soft', 0.50, 0.5, 0.3), row('soft', 0.52, 1.0, 0.3), row('soft', 0.60, 2.0, 0.3), row('matrix', 0.1)]
    assert slot_rules.choose_candidate(rows, 'soft')['gain'] == 0.5  # clear minimum
    tied = [row('soft', 0.50000, 2.0, 0.3), row('soft', 0.50005, 0.5, 0.3), row('soft', 0.50009, 1.0, 0.1), row('soft', 0.50009, 1.0, 1.0)]
    assert slot_rules.choose_candidate(tied, 'soft') == tied[3]  # all within 1e-4: g = 1 twice, the larger tau of the two
    assert slot_rules.choose_candidate(tied[:2], 'soft') == tied[1]  # both tied: g = 0.5 is nearer to 1 than g = 2
    apart = [row('soft', 0.5, 2.0, 0.3), row('soft', 0.50011, 1.0, 0.3)]  # 1.1e-4 apart: not tied
    assert slot_rules.choose_candidate(apart, 'soft') == apart[0]


def test_the_calibration_choice_excludes_non_finite_candidates_and_stops_without_any():
    rows = [row('hard', math.nan, 1.0), row('hard', math.inf, 2.0), row('hard', 0.9, 0.5)]
    assert slot_rules.choose_candidate(rows, 'hard')['gain'] == 0.5
    with pytest.raises(StopForOrchestrator):
        slot_rules.choose_candidate([row('hard', math.nan)], 'hard')
    with pytest.raises(StopForOrchestrator):
        slot_rules.choose_candidate(rows, 'matrix')


def lr(rate, nll):
    return dict(lr=rate, nll=nll)


def test_the_lr_choice_ties_within_a_hundredth_of_a_nat_go_to_the_lower_rate():
    rows = [lr(3e-4, 1.50), lr(1e-3, 1.40), lr(3e-3, 1.395)]
    assert slot_rules.choose_lr(rows, 2.0)['lr'] == 1e-3  # 1.40 is within 0.01 of the best 1.395
    assert slot_rules.choose_lr([lr(3e-4, 1.50), lr(1e-3, 1.40), lr(3e-3, 1.38)], 2.0)['lr'] == 3e-3  # 0.02 apart: the best wins
    assert slot_rules.choose_lr([lr(3e-4, 1.40), lr(1e-3, 1.40), lr(3e-3, 1.40)], 2.0)['lr'] == 3e-4


def test_the_lr_choice_excludes_non_finite_runs_and_stops_when_nothing_beats_step_zero():
    assert slot_rules.choose_lr([lr(3e-4, 1.4), lr(1e-3, math.nan), lr(3e-3, math.inf)], 2.0)['lr'] == 3e-4
    with pytest.raises(StopForOrchestrator):
        slot_rules.choose_lr([lr(3e-4, 2.1), lr(1e-3, 2.2), lr(3e-3, math.nan)], 2.0)
    with pytest.raises(StopForOrchestrator):
        slot_rules.choose_lr([lr(r, math.nan) for r in (3e-4, 1e-3, 3e-3)], 2.0)
    assert slot_rules.choose_lr([lr(3e-4, 2.0)], 2.0)['lr'] == 3e-4  # equal to step 0 is not worse


def test_the_step_count_is_the_largest_multiple_of_ten_that_fits_one_hour_capped_at_6000():
    # ((3600 - 90) / 1.5 - 11 * 20) / 0.5 = 4240: the load time 60 s is below the 90 s floor
    assert slot_rules.schedule_steps(0.5, 20.0, 60.0) == 4240
    assert slot_rules.schedule_steps(0.5, 20.0, 300.0) == 3960  # load 300 s: (3300 / 1.5 - 220) / 0.5
    assert slot_rules.schedule_steps(0.5, 20.0, 60.0, max_steps=3000) == 3000
    assert slot_rules.schedule_steps(0.1, 20.0, 60.0) == 6000
    assert slot_rules.schedule_steps(0.37, 20.0, 60.0) % 10 == 0
    with pytest.raises(StopForOrchestrator):
        slot_rules.schedule_steps(0.5, 250.0, 60.0)  # eleven evaluations take longer than the budget
    for bad in (0.0, math.nan, -1.0):
        with pytest.raises(ValueError):
            slot_rules.schedule_steps(bad, 20.0, 60.0)


def test_the_job_walltime_is_the_estimate_with_margin_rounded_up_to_five_minutes():
    assert slot_rules.job_walltime_seconds(0.5, 20.0, 60.0, 4240) == math.ceil((60 + 2120 + 220) * 1.5 / 300) * 300 == 3600
    assert slot_rules.job_walltime_seconds(0.5, 20.0, 60.0, 1000) == 1200  # (60 + 500 + 220) * 1.5 = 1170 s
    with pytest.raises(StopForOrchestrator):
        slot_rules.job_walltime_seconds(0.5, 20.0, 60.0, 4300)  # a 2 h scale-up needs the orchestrator's OK


def test_the_checkpoint_choice_is_the_lowest_dev_nll_and_a_tie_goes_to_the_earlier_step():
    assert slot_rules.select_checkpoint([(100, 1.5), (200, 1.4), (300, 1.45)]) == 200
    assert slot_rules.select_checkpoint([(300, 1.4), (100, 1.4), (200, 1.4)]) == 100  # step order, whatever the input order
    assert slot_rules.select_checkpoint([(100, 1.4), (200, 1.4 - 5e-7)]) == 100  # lower, but not by more than 1e-6
    assert slot_rules.select_checkpoint([(100, 1.4), (200, 1.4 - 2e-6)]) == 200
    assert slot_rules.select_checkpoint([(100, math.nan), (200, 1.4), (300, math.inf)]) == 200
    with pytest.raises(StopForOrchestrator):
        slot_rules.select_checkpoint([(100, math.nan)])


def test_the_checkpoints_are_ten_equal_steps_of_a_multiple_of_ten():
    assert slot_rules.checkpoint_steps(4240) == [424 * k for k in range(1, 11)]
    with pytest.raises(ValueError):
        slot_rules.checkpoint_steps(4245)


# ------------------------------------------------------------------------------------------------ run configurations

def run(arm='matrix_L1', seed=17, **override):
    read = dict(kind=slot_rules.ARM_KIND[arm], gain=1.0, tau=None if arm == 'matrix_L1' else 0.3, scope='all_valid_question',
                recency=False, capacity=16)
    return slot_rules.science_config(arm, seed, steps=1000, lr=1e-3, read=read, parent_digest='d' * 32, **override)


def test_a_science_config_carries_the_registered_run_definition():
    config = run('soft_slot_L1', 29)
    assert config['trainable'] == ['Q', 'R', 'U', 'B'] and config['frozen'] == ['A', 'K', 'V', 'a_lambda', 'a_beta']
    assert config['train_base_seed'] == slot_data.TRAIN_BASE['S1A', 29] and config['dev_base_seed'] == slot_data.SETS['S1A', 'dev'].base_seed
    assert config['clip'] == 1.0 and config['warmup'] == 20 and config['batch'] == 32 and config['loop_steps'] == 1
    assert config['optimizer']['betas'] == (0.9, 0.999) and config['optimizer']['weight_decay'] == 0.01
    assert run(ablation=True)['trainable'] == ['A_query', 'Q', 'R', 'U', 'B']
    assert json.loads(json.dumps(config))['read']['tau'] == 0.3


def test_the_science_hash_changes_with_every_defining_field_and_nothing_else():
    base = slot_rules.science_hash(run())
    assert base == slot_rules.science_hash(run())
    assert len({base, slot_rules.science_hash(run(seed=29)), slot_rules.science_hash(run('soft_slot_L1')),
                slot_rules.science_hash(run(ablation=True)),
                slot_rules.science_hash(dict(run(), lr=3e-3)), slot_rules.science_hash(dict(run(), steps=2000)),
                slot_rules.science_hash(dict(run(), parent_writer_blake2b_128='e' * 32))}) == 7


@pytest.mark.parametrize('arm,seed,steps,read', [
    ('matrix_L2', 17, 1000, dict(kind='matrix', scope='all_valid_question')),  # an unregistered arm
    ('matrix_L1', 18, 1000, dict(kind='matrix', scope='all_valid_question')),  # an unregistered seed
    ('matrix_L1', 17, 1005, dict(kind='matrix', scope='all_valid_question')),  # N is a multiple of the number of checkpoints
    ('matrix_L1', 17, 1000, dict(kind='soft', scope='all_valid_question')),  # the arm's own read kind
    ('matrix_L1', 17, 1000, dict(kind='matrix', scope='final_only')),  # the deployed scope
    ('soft_slot_L1', 17, 1000, dict(kind='soft', scope='all_valid_question', recency=True))])  # no recency variant
def test_a_run_that_is_not_one_registered_arm_and_seed_is_refused(arm, seed, steps, read):
    with pytest.raises(ValueError):
        slot_rules.science_config(arm, seed, steps=steps, lr=1e-3, read=read, parent_digest='d' * 32)


def test_the_committed_stage_files_are_the_registered_renderings():
    for stage, name in (('S0', 's0.json'), ('S1A', 's1a.json')):
        assert json.loads((ROOT / 'configs' / 'slot_loop_v1' / name).read_text()) == slot_rules.stage_config(stage)


# ----------------------------------------------------------------------------------------------------- manifest script

def test_the_pending_manifest_carries_the_registry_and_the_config_hashes(tmp_path):
    pending = manifest_script.pending(ROOT / 'configs' / 'slot_loop_v1')
    assert pending['status'] == 'pending' and pending['prereg_blake2b_128'] == slot_data.PREREG_BLAKE2B_128
    assert pending['manifest'] == json.loads(json.dumps(slot_data.manifest())) and set(pending['config_hashes']) == {'S0', 'S1A'}
    manifest_script.write_configs(tmp_path)
    assert manifest_script.pending(tmp_path)['config_hashes'] == pending['config_hashes']
    (tmp_path / 's0.json').write_text(json.dumps(dict(slot_rules.stage_config('S0'), clip=2.0)))
    with pytest.raises(SystemExit):
        manifest_script.pending(tmp_path)


def passed_report(tmp_path, monkeypatch, **override):
    inventory = {'S-RE': {'writer.pt': 'a' * 32}, 'D128-3': {'writer.pt': 'b' * 32}}
    monkeypatch.setattr(slot_data, 'inventory', lambda: inventory)
    required = ['native_replay', 'matrix_equals_native', 'owner_mask', 'isolation', 'frozen_set', 'bytes', 'capacity',
                'feature_consistency', 'd128_3_replay']
    checks = {name: dict(passed=True) for name in required}
    checks['native_replay'].update(replayed=dict(correct=723, loss_float32=1.0, relative_read_error=0.196))
    checks['d128_3_replay'].update(replayed=dict(correct=608, loss_float32=1.1, relative_read_error=0.2))
    checks['matrix_equals_native']['max_abs_difference'] = 0.0
    report = dict(status='complete', passed=True, required=required, checks=checks, inventory=inventory,
                  config_hash=manifest_script.check_configs(ROOT / 'configs' / 'slot_loop_v1')['S0'], job_id='1.opbs',
                  git_commit='c' * 40, source_blake2b_128=slot_data.source_digest())
    report.update(override)
    path = tmp_path / 'preflight.json'
    path.write_text(json.dumps(report))
    return path, report


def test_the_lock_needs_a_passed_preflight_on_the_files_and_configuration_on_disk(tmp_path, monkeypatch):
    path, report = passed_report(tmp_path, monkeypatch)
    locked, table = manifest_script.lock(ROOT / 'configs' / 'slot_loop_v1', path)
    assert locked['status'] == 'locked' and locked['preflight']['passed'] and locked['inventory'] == report['inventory']
    assert locked['preflight']['report_blake2b_128'] == slot_data.file_digest(path) and locked['preflight']['matrix_max_abs_difference'] == 0.0
    assert table['parents']['S-RE']['old_range_replay']['correct'] == 723 and table['parents']['D128-3']['old_range_replay']['correct'] == 608
    # the runner accepts exactly this lock: the manifest in it is the registry now
    assert locked['manifest'] == json.loads(json.dumps(slot_data.manifest()))
    lock_path = tmp_path / 'manifest.lock.json'
    lock_path.write_text(json.dumps(locked))
    from looplora.llm import slot_run  # noqa: E402  (torch is imported here; this module needs a compute node anyway)
    assert slot_run.require_lock(lock_path)['status'] == 'locked'


@pytest.mark.parametrize('defect', ['failed_check', 'missing_check', 'not_passed', 'other_files', 'other_config', 'incomplete', 'other_source'])
def test_the_lock_is_refused_unless_every_required_check_passed_on_these_files(tmp_path, monkeypatch, defect):
    path, report = passed_report(tmp_path, monkeypatch)
    if defect == 'failed_check':
        report['checks']['matrix_equals_native'].update(passed=False, max_abs_difference=1e-3)
    elif defect == 'missing_check':
        del report['checks']['capacity']
    elif defect == 'not_passed':
        report['passed'] = False
    elif defect == 'other_files':
        report['inventory'] = {'S-RE': {'writer.pt': 'f' * 32}, 'D128-3': {'writer.pt': 'b' * 32}}
    elif defect == 'other_config':
        report['config_hash'] = '0' * 32
    elif defect == 'other_source':
        report['source_blake2b_128'] = '0' * 32
    else:
        report['status'] = 'running'
    path.write_text(json.dumps(report))
    with pytest.raises(SystemExit):
        manifest_script.lock(ROOT / 'configs' / 'slot_loop_v1', path)


def test_the_manifest_omits_sealed_seeds_but_keeps_their_counts_and_id_digests():
    manifest = slot_data.manifest()
    for purpose in ("confirmation", "unseen"):
        row = manifest["sets"][f"S1A/{purpose}"]
        assert row["sealed"] and row["count"] == 2000
        assert "base_seed" not in row and "query_seed" not in row
        assert manifest["ids"][f"eval/S1A/{purpose}"]["count"] == row["count"]


def test_source_digest_changes_with_source_paths_and_bytes_but_allows_docs_and_scripts(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "one.py"
    source.write_text("x = 1\n")
    original = slot_data.source_digest(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "state.md").write_text("updated")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "analysis.py").write_text("updated")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "one.pyc").write_bytes(b"cache")
    assert slot_data.source_digest(tmp_path) == original
    source.write_text("x = 2\n")
    changed = slot_data.source_digest(tmp_path)
    assert changed != original
    source.rename(tmp_path / "src" / "two.py")
    assert slot_data.source_digest(tmp_path) not in {original, changed}
