"""Registered A1 provenance, read transfer and extended training boundaries; compute only."""
import pytest

from looplora.infra.seeds import stable_hash
from looplora.llm import s1b_artifacts as a, s1b_data as d, s1b_rules as r
from test_s1b_artifacts import bundle, prepare
from test_s1b_data_analysis import opening


def test_a1_manifest_covers_complete_main_windows_and_unchanged_pilot():
    manifest = d.manifest()
    assert manifest["amendment_blake2b_128"] == d.AMENDMENT_DIGEST
    for seed in d.RUN_SEEDS:
        assert manifest["train_streams"][str(seed)] == dict(split="train", start=0, count=576000, base_seed=d.TRAIN_BASE[seed])
        assert manifest["ids"][f"train/{seed}"]["count"] == 576000
    assert manifest["train_streams"]["pilot"]["count"] == 6400


def test_a1_overlap_in_new_tail_is_not_missed(monkeypatch):
    # This identity lies beyond the old 192,000-episode window.
    collision = d.episode_id(d.TRAIN_BASE[17], "train", 575999)
    monkeypatch.setattr(d.old, "old_ids", lambda: {collision})
    with pytest.raises(ValueError, match="train/17"):
        d.check_disjoint()


def test_a1_registered_document_mutation_invalidates_lock(bundle, monkeypatch, tmp_path):
    for name in (d.PREREG, d.AMENDMENT, d.old.PREREG):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((a.ROOT / name).read_bytes())
    (tmp_path / d.AMENDMENT).write_text("unregistered amendment")
    monkeypatch.setattr(a, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="registered preregistration"):
        a.require_lock(bundle.paths["lock"])


def test_a1_open_record_must_bind_amendment(opening):
    root, release, lock, selection, path, record = opening
    record.pop(d.AMENDMENT)
    a.dump(path, record)
    with pytest.raises(PermissionError):
        d.load_open_token(root, release, lock, selection)


@pytest.mark.parametrize("defect", ["amendment", "stopped_evidence", "retuned_reads"])
def test_a1_release_refuses_amended_or_retuned_provenance(bundle, defect):
    candidate = prepare(bundle)
    if defect == "amendment":
        candidate["amendment_blake2b_128"] = "0" * 32
    elif defect == "stopped_evidence":
        candidate["stopped_calibration_blake2b_128"] = "0" * 32
    else:
        for arm in ("S1", "S4"):
            candidate["arms"][arm]["read"]["tau"] = .3
    with pytest.raises(ValueError):
        a.require_release(bundle.write("changed-candidate.json", candidate), bundle.paths["lock"], ready=False)


def test_a1_science_hash_binds_budget_schedule_and_transferred_read():
    read = r.read_config("S4").as_dict()
    config = r.science_config("advance", "S4", 17, .001, read)
    assert config["steps"] == 18000 and config["checkpoints"] == list(range(1800, 18001, 1800))
    assert config["amendment_blake2b_128"] == d.AMENDMENT_DIGEST
    old = dict(config, steps=6000, checkpoints=list(range(600, 6001, 600)))
    assert stable_hash(old) != stable_hash(config)
    with pytest.raises(ValueError, match="released S1A"):
        r.science_config("advance", "S4", 17, .001, dict(read, tau=.3))


def test_a1_ready_release_rechecks_transferred_source_after_builder(bundle):
    candidate = prepare(bundle)
    reloads = {arm: bundle.write(f"{arm}-reload.json", dict(passed=True, arm=arm,
        reload_basis=candidate["reload_basis"], metadata=settings["smoke_metadata"],
        original_smoke_samples_blake2b_128=settings["original_smoke_samples_blake2b_128"]))
        for arm, settings in candidate["arms"].items()}
    ready = bundle.write("ready.json", prepare(bundle, reloads))
    source = bundle.paths["s1a"]
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="S1A release"):
        a.require_release(ready, bundle.paths["lock"])
