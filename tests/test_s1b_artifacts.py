"""S1B public artifact boundaries with synthetic metadata/model substitutes; compute only."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from looplora.infra.seeds import stable_hash
from looplora.llm import s1b_artifacts as a, s1b_data as d, s1b_rules as r, s1b_run as run


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    def write(name, value):
        path = tmp_path / name
        a.dump(path, value)
        return path
    decision = write("s1a-summary.json", dict(stage="S1A", verdict="ADVANCE", release_blake2b_128=d.S1A_RELEASE_DIGEST))
    dev = write("s1a-selection.json", dict(stage="S1A-DEV", outcome="no_trigger", release_blake2b_128=d.S1A_RELEASE_DIGEST))
    branch = write("branch.json", a.branch_record(decision, dev))
    inventory = {"writer.pt": "f" * 32}
    manifest = dict(ids={"dev": dict(count=1000, ids_blake2b_128="dev-ids")})
    monkeypatch.setattr(a, "inventory", lambda: inventory)
    monkeypatch.setattr(d, "manifest", lambda: manifest)
    monkeypatch.setattr(d.old, "source_digest", lambda: "source")
    preflight = write("preflight.json", dict(status="complete", passed=True,
        checks={name: dict(passed=True) for name in a.REQUIRED}, inventory=inventory, manifest=manifest,
        source_blake2b_128="source"))
    lock = write("lock.json", a.make_lock(preflight))
    source = write("s1a-release.json", a.read(a.ROOT / "reports/slot-loop-v1/s1a-release.json"))
    # Preserve the exact real release bytes: the digest, not reserialization, is the contract.
    source.write_bytes((a.ROOT / "reports/slot-loop-v1/s1a-release.json").read_bytes())
    stopped = tmp_path / "stopped-calibration.json"
    stopped.write_bytes((a.ROOT / "reports/slot-loop-v1/s1b/diagnosis-6855048/calibration.input.json").read_bytes())
    cal = write("read-settings.json", a.read_settings_record(source, stopped, branch, lock))
    provenance = dict(status="complete", branch="advance", lock_blake2b_128=d.old.file_digest(lock),
                      branch_blake2b_128=d.old.file_digest(branch),
                      read_settings_blake2b_128=d.old.file_digest(cal), source_commit="a" * 40, amendment_blake2b_128=d.AMENDMENT_DIGEST)
    pilot_arms, smoke_arms = {}, {}
    for arm in r.branch_arms("advance"):
        pilot_arms[arm] = dict(candidates=[dict(lr=lr, nll=.9, all_finite=True) for lr in r.LR_GRID],
            step0_nll=1., step_seconds=.2, eval_seconds=27., load_seconds=60., export_seconds=1., overhead_seconds=2.)
        smoke_arms[arm] = dict(steps=20, freeze=dict(digests=dict(frozen_unchanged=True), support=dict(equal=True)),
                              all_finite=True, export_metadata=dict(arm=arm, control="shared",
                                  read=r.read_config(arm).as_dict(), amendment_blake2b_128=d.AMENDMENT_DIGEST),
                              original_smoke_samples_blake2b_128="b" * 32, first_reload_matches_original=True)
    pilot = write("pilot.json", dict(provenance, arms=pilot_arms))
    smoke = write("smoke.json", dict(provenance, arms=smoke_arms))
    paths = dict(lock=lock, branch=branch, cal=cal, pilot=pilot, smoke=smoke, preflight=preflight, s1a=source, stopped=stopped)
    return SimpleNamespace(paths=paths, write=write, inventory=inventory, manifest=manifest)


def prepare(b, reloads=None):
    p = b.paths
    return a.prepare_release(p["lock"], p["branch"], p["cal"], p["pilot"], p["smoke"], "a" * 40, reloads=reloads)


def test_complete_candidate_to_ready_uses_exact_json_reports(bundle):
    b = bundle
    candidate = prepare(b)
    assert candidate["status"] == "pending_reload" and len(candidate["config_hashes"]) == 12
    assert candidate["arms"]["M1"]["read"] == candidate["arms"]["M4"]["read"]
    assert candidate["arms"]["S1"]["read"] == candidate["arms"]["S4"]["read"]
    reloads = {arm: b.write(f"{arm}-reload.json", dict(passed=True, arm=arm, reload_basis=candidate["reload_basis"],
               original_smoke_samples_blake2b_128=candidate["arms"][arm]["original_smoke_samples_blake2b_128"],
               metadata=candidate["arms"][arm]["smoke_metadata"])) for arm in candidate["arms"]}
    release = prepare(b, reloads)
    path = b.write("ready.json", release)
    assert a.require_release(path, b.paths["lock"])[1] == release
    with pytest.raises(PermissionError):
        a.require_approval(b.write("approval.json", dict(status="approved", release_blake2b_128="old")), path)
    a.require_approval(b.write("approval.json", dict(status="approved", release_blake2b_128=d.old.file_digest(path))), path)


@pytest.mark.parametrize("defect", ["missing_preflight", "source", "pilot_lr", "smoke", "read_settings", "branch_source"])
def test_lock_and_release_refuse_incomplete_changed_inputs(bundle, defect, monkeypatch):
    b = bundle
    if defect == "missing_preflight":
        report = a.read(b.paths["preflight"])
        report["checks"].pop("trajectory")
        a.dump(b.paths["preflight"], report)
        with pytest.raises(ValueError):
            a.make_lock(b.paths["preflight"])
        return
    if defect == "source":
        monkeypatch.setattr(d.old, "source_digest", lambda: "changed")
    elif defect == "pilot_lr":
        record = a.read(b.paths["pilot"])
        record["arms"]["M4"]["candidates"].pop()
        a.dump(b.paths["pilot"], record)
    elif defect == "smoke":
        record = a.read(b.paths["smoke"])
        record["arms"]["S4"]["freeze"]["support"]["equal"] = False
        a.dump(b.paths["smoke"], record)
    elif defect == "read_settings":
        record = a.read(b.paths["cal"])
        record["selected"]["matrix"]["gain"] = 2
        a.dump(b.paths["cal"], record)
    else:
        branch = a.read(b.paths["branch"])
        path = Path(branch["files"]["decision"])
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        prepare(b)


def test_reload_report_cannot_be_reused_after_candidate_changes(bundle):
    candidate = prepare(bundle)
    reloads = {arm: bundle.write(f"{arm}.json", dict(passed=True, arm=arm, reload_basis="old-candidate",
                metadata=candidate["arms"][arm]["smoke_metadata"])) for arm in candidate["arms"]}
    with pytest.raises(ValueError, match="reload"):
        prepare(bundle, reloads)


def test_a1_transfers_reads_despite_stopped_hop1_em(bundle):
    p = bundle.paths
    record = a.require_read_settings(p["cal"], "advance", p["lock"])
    assert a.read(p["stopped"])["selected"]["matrix"]["per_hop"]["1"]["em"] < .1
    assert record["selected"]["soft"]["tau"] == .1
    assert record["stopped_calibration_blake2b_128"] == d.STOPPED_CALIBRATION_DIGEST
    assert prepare(bundle)["amendment_blake2b_128"] == d.AMENDMENT_DIGEST


def test_pending_release_cannot_train_even_with_approval(bundle):
    path = bundle.write("candidate.json", prepare(bundle))
    with pytest.raises(ValueError):
        a.require_release(path, bundle.paths["lock"])


def test_cli_refuses_missing_manifest_before_any_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RUN_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        run.main(["--mode", "train", "--out", str(tmp_path / "train")])
    with pytest.raises(ValueError):
        run.output(Path("relative-output"))


def test_mechanism_trigger_is_backend_specific_and_requires_frozen_release_digest(tmp_path):
    release = tmp_path / "release.json"
    release.write_text("{}")
    verdict = tmp_path / "verdict.json"
    a.dump(verdict, dict(stage="S1B", release_blake2b_128=d.old.file_digest(release),
                        contrasts={"M4-M1": dict(passed=True), "S4-S1": dict(passed=False)}))
    assert run.require_loop_gain(verdict, release, "M4")
    with pytest.raises(PermissionError):
        run.require_loop_gain(verdict, release, "S4")
    with pytest.raises(PermissionError):
        run.require_loop_gain(verdict, release, "M1")
