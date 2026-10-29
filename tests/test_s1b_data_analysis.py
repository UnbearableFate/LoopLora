"""S1B registration, protected-data access and whole-episode inference; compute only."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm import s1b_data as d, s1b_rules as r, s1b_analysis as stats, slot_data

ROOT = Path(__file__).resolve().parents[1]


def test_registered_bytes_task_seeds_and_old_state():
    assert slot_data.file_digest(ROOT / d.PREREG) == d.PREREG_DIGEST
    assert slot_data.file_digest(ROOT / slot_data.PREREG) == slot_data.PREREG_BLAKE2B_128
    d.verify_registry()
    assert stable_hash(asdict(d.task_spec())) == d.LAYOUT_DIGEST
    assert stable_hash(asdict(slot_data.task_spec())) == slot_data.LAYOUT_DIGEST
    assert (d.task_spec().E, d.task_spec().chain_lengths) == (12, (8, 4))
    for run, seed in d.TRAIN_BASE.items():
        assert seed == derive_seed(slot_data.BASE_SEED, slot_data.STUDY, "train", "S1B", run)


def test_full_registered_prefixes_are_disjoint_with_original_layouts():
    rows = d.check_disjoint()
    assert rows["train/17"]["count"] == 192000
    assert rows["train/pilot"]["count"] == 6400
    assert rows["old/train/S1A/17"]["ids_blake2b_128"] == slot_data.ids_digest(slot_data.train_ids("S1A", 17, 6000))
    assert rows["dev"]["ids_blake2b_128"] == slot_data.ids_digest(d.SETS["dev"].ids())


def test_overlap_fails_without_generating_graphs(monkeypatch):
    colliding = d.SETS["smoke"].ids()[0]
    monkeypatch.setattr(slot_data, "old_ids", lambda: {colliding})
    with pytest.raises(ValueError, match="overlap"):
        d.check_disjoint()


@pytest.mark.parametrize("purpose", ["confirmation", "unseen"])
def test_sealed_acquisition_and_both_generation_paths_refuse(purpose):
    with pytest.raises(PermissionError):
        d.eval_set(purpose)
    with pytest.raises(PermissionError):
        d.raw_episodes(d.SETS[purpose])
    with pytest.raises(PermissionError):
        d.episodes(d.SETS[purpose])
    old_token = slot_data.OpenToken(slot_data.PREREG_BLAKE2B_128, "a" * 32, {}, "b" * 32, {})
    with pytest.raises(PermissionError):
        d.eval_set(purpose, token=old_token, checkpoint_digest="b" * 32)


def test_training_layout_evaluation_stratification_does_not_change_graphs():
    s = replace(d.SETS["smoke"], count=12)
    raw, evaluation = d.raw_episodes(s), d.episodes(s)
    np.testing.assert_array_equal(raw.facts, evaluation.facts)
    np.testing.assert_array_equal(raw.episode_seed, evaluation.episode_seed)
    assert raw.q_start.shape == (12, 4) and evaluation.q_start.shape == (12, 8)
    assert np.array_equal(evaluation.q_hop_true, np.tile(np.arange(1, 9), (12, 1)))
    assert (raw.facts[:, :, 1] == 0).all() and (evaluation.q_hop_given == -1).all()
    assert np.isin(raw.q_hop_true, [1, 2, 3]).all()


@pytest.mark.parametrize("verdict,outcome,ablation,branch", [
    ("ADVANCE", "no_trigger", None, "advance"),
    ("NOT ADVANCED", "no_trigger", None, "matrix_only"),
    ("NOT ADVANCED", "ablation", False, "matrix_only"),
    ("NOT ADVANCED", "ablation", True, "query_copy"),
])
def test_all_registered_branch_outcomes(verdict, outcome, ablation, branch):
    digest = "c" * 32
    abl = None if ablation is None else dict(stage="S1A-ABL", flag=ablation, release_blake2b_128=digest)
    assert r.resolve_branch(dict(stage="S1A", verdict=verdict, release_blake2b_128=digest),
                            dict(stage="S1A-DEV", outcome=outcome, release_blake2b_128=digest), abl) == branch


def test_invalid_or_missing_branch_evidence_cannot_be_a_null():
    decision = dict(stage="S1A", verdict="NOT ADVANCED")
    with pytest.raises(r.StopForOrchestrator):
        r.resolve_branch(decision, dict(outcome="stop_unstable"))
    with pytest.raises(ValueError):
        r.resolve_branch(decision, dict(stage="S1A-DEV", outcome="ablation"))
    with pytest.raises(ValueError):
        r.resolve_branch(dict(stage="S1A", verdict="pending"), dict(outcome="no_trigger"))


def test_transfer_stop_and_resource_cap_boundaries():
    cal = dict(selected=dict(matrix=dict(per_hop={"1": dict(em=.1)})))
    r.sanity(cal)
    cal["selected"]["matrix"]["per_hop"]["1"]["em"] = .099999
    with pytest.raises(r.StopForOrchestrator):
        r.sanity(cal)
    assert r.walltime(.2, 27.33, 60) == 2400
    with pytest.raises(r.StopForOrchestrator):
        r.walltime(10., 27.33, 60)
    with pytest.raises(ValueError):
        r.walltime(float("nan"), 1, 1)


def samples(values, n=100):
    correct = np.broadcast_to(values, (n, 8)).copy()
    return dict(correct=correct, nll=np.zeros_like(correct, dtype=float), episode_id=np.arange(n, dtype=np.int64), hops=np.arange(1, 9))


def paired_fixture(branch="advance", n=100):
    main = {(arm, s): samples([0, 1, 1, 0, 0, 0, 0, 0] if arm.endswith("4") else [0] * 8, n)
            for arm in r.branch_arms(branch) for s in d.RUN_SEEDS}
    sweeps = {(arm, s, depth): samples([0, 1, 1, 0, 0, 0, 0, 0] if depth >= 4 else [0] * 8, n)
              for arm in r.branch_arms(branch) if arm.endswith("4") for s in d.RUN_SEEDS for depth in (1, 2, 4, 8)}
    return main, sweeps


def test_loop_and_storage_separate_matrix_can_continue_without_slot_advantage():
    main, sweep = paired_fixture()
    result = stats.paired(main, sweep, "advance")
    assert result["s2_primary"] == "matrix"
    assert result["contrasts"]["S4-S1"]["passed"]
    assert result["contrasts"]["M4-M1"]["passed"]
    assert not result["contrasts"]["S4-M4"]["passed"]
    assert result["contrasts"]["M4-M1"]["confidence"] == 1 - .05 / 3
    assert result["resamples"] == 10000


def test_hop1_and_seed_screens_cannot_be_rescued_by_other_hops():
    main, sweep = paired_fixture("matrix_only")
    main["M1", 17]["correct"][:, 0] = 1
    result = stats.paired(main, sweep, "matrix_only")
    assert not result["contrasts"]["M4-M1"]["tests"]["hop1_drop_at_most_two_pp"]
    assert result["s2_primary"] is None
    main, sweep = paired_fixture("matrix_only")
    main["M4", 43]["correct"][:] = 0
    sweep["M4", 43, 4]["correct"][:] = 0
    assert not stats.paired(main, sweep, "matrix_only")["contrasts"]["M4-M1"]["tests"]["seeds_positive"]


def test_descriptive_sweep_cannot_replace_independently_trained_l1_or_rescue_null():
    main, sweep = paired_fixture("matrix_only")
    for s in d.RUN_SEEDS:
        main["M1", s]["correct"][:] = main["M4", s]["correct"]
    result = stats.paired(main, sweep, "matrix_only")
    assert result["continuation"]["M"]["sweep_sign_screen"]
    assert not result["continuation"]["M"]["qualified"]
    main, sweep = paired_fixture("matrix_only")
    for s in d.RUN_SEEDS:
        sweep["M4", s, 1]["correct"][:] = sweep["M4", s, 4]["correct"]
    assert stats.paired(main, sweep, "matrix_only")["s2_primary"] is None


def test_bootstrap_uses_whole_episode_index_once_for_all_seeds_and_hops():
    main, sweep = paired_fixture("matrix_only", n=20)
    for s in d.RUN_SEEDS:
        main["M4", s]["correct"][5:, 1:3] = 0
        sweep["M4", s, 4]["correct"][:] = main["M4", s]["correct"]
    result = stats.paired(main, sweep, "matrix_only")
    rng = np.random.default_rng(d.BOOTSTRAP)
    vector = np.r_[np.ones(5), np.zeros(15)]
    expected = vector[rng.integers(0, 20, size=(10000, 20))].mean(1)
    np.testing.assert_array_equal(result["contrasts"]["M4-M1"]["interval_pp"], np.quantile(expected, [.025, .975]) * 100)
    assert result["contrasts"]["M4-M1"]["delta_pp"] == 25


def test_sample_guard_checks_two_axes_and_exact_hop_id_order():
    sample = samples([1] * 8, 10)
    identity = dict(count=10, ids_blake2b_128=slot_data.ids_digest(sample["episode_id"]))
    stats.validate_samples(sample, identity)
    sample["hops"] = sample["hops"][::-1]
    with pytest.raises(ValueError):
        stats.validate_samples(sample, identity)
    main, sweep = paired_fixture("matrix_only")
    sweep["M4", 17, 2]["episode_id"] = sweep["M4", 17, 2]["episode_id"][::-1]
    with pytest.raises(ValueError, match="unpaired"):
        stats.paired(main, sweep, "matrix_only")


@pytest.fixture
def opening(tmp_path):
    for name in (d.PREREG, slot_data.PREREG, d.ANALYSIS):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, target)
    selected = {f"{a}/{s}": f"{i + 1:032x}" for i, (a, s) in enumerate((a, s) for a in ("M1", "M4") for s in d.RUN_SEEDS)}
    release = dict(branch="matrix_only", status="ready", analysis_script_blake2b_128=slot_data.file_digest(tmp_path / d.ANALYSIS))
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release))
    lock = dict(inventory={"writer.pt": "e" * 32})
    selection = dict(status="selected", selected_checkpoints=selected, release_blake2b_128=slot_data.file_digest(release_path))
    record = dict(stage="S1B", branch="matrix_only", selected_checkpoints=selected,
                  release_blake2b_128=selection["release_blake2b_128"], analysis_script_blake2b_128=release["analysis_script_blake2b_128"],
                  lock_blake2b_128="lock", inventory_blake2b_128=stable_hash(lock["inventory"]), contrasts=r.contrasts("matrix_only"))
    release["lock_blake2b_128"] = "lock"
    release_path.write_text(json.dumps(release))
    selection["release_blake2b_128"] = record["release_blake2b_128"] = slot_data.file_digest(release_path)
    record[d.PREREG], record[slot_data.PREREG] = d.PREREG_DIGEST, slot_data.PREREG_BLAKE2B_128
    record_path = tmp_path / "reports/slot-loop-v1/s1b/open.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(record))
    return tmp_path, release_path, lock, selection, record_path, record


def test_valid_open_token_and_rebuilt_release_refusal(opening):
    root, path, lock, selection, _, _ = opening
    token = d.load_open_token(root, path, lock, selection)
    s = d.eval_set("confirmation", token=token, checkpoint_digest=next(iter(token.selected.values())))
    assert s.opened
    path.write_text(path.read_text() + " ")
    with pytest.raises(PermissionError):
        d.load_open_token(root, path, lock, selection)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "script", "branch", "inventory"])
def test_open_record_mapping_and_provenance_refusals(opening, defect):
    root, release, lock, selection, path, record = opening
    if defect == "missing":
        record["selected_checkpoints"].pop(next(iter(record["selected_checkpoints"])))
    elif defect == "duplicate":
        labels = list(record["selected_checkpoints"])
        record["selected_checkpoints"][labels[1]] = record["selected_checkpoints"][labels[0]]
    elif defect == "script":
        (root / d.ANALYSIS).write_text("changed script")
    elif defect == "branch":
        record["branch"] = "query_copy"
    else:
        record["inventory_blake2b_128"] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(PermissionError):
        d.load_open_token(root, release, lock, selection)
