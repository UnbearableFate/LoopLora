"""S1B lock, branch evidence and release provenance (no model imports)."""
from __future__ import annotations

import json
from pathlib import Path

from looplora.infra.seeds import stable_hash
from . import s1b_data as data, s1b_rules as rules

ROOT = Path(__file__).resolve().parents[3]
REQUIRED = ("registry", "legacy", "matrix_L4", "trajectory", "owner_mask", "isolation", "features",
            "capacity", "bytes", "frozen", "reload")


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


def registered():
    for path, expected in ((data.PREREG, data.PREREG_DIGEST), (data.old.PREREG, data.old.PREREG_BLAKE2B_128)):
        if data.old.file_digest(ROOT / path) != expected:
            raise ValueError("registered preregistration bytes differ")
    data.verify_registry()


def branch_record(decision, dev, ablation=None):
    inputs = dict(decision=str(Path(decision).resolve()), dev=str(Path(dev).resolve()))
    if ablation is not None:
        inputs["ablation"] = str(Path(ablation).resolve())
    branch = rules.resolve_branch(read(decision), read(dev), read(ablation) if ablation else None)
    return dict(stage="S1B", branch=branch, files=inputs,
                digests={key: data.old.file_digest(path) for key, path in inputs.items()})


def require_branch(path):
    found = read(path)
    expected = branch_record(found["files"]["decision"], found["files"]["dev"], found["files"].get("ablation"))
    if found != expected:
        raise ValueError("S1A branch evidence changed")
    return found


def inventory():
    expected = dict(writer_pt="3994a5631021de03b2039974ac934e56", record_json="816570a0882ed6952b4f851b67d4ef9f")
    files = {f: data.old.file_digest(Path(data.old.PARENT_FOLDERS["S-RE"]) / f)
             for f in data.old.PARENT_RECORD_FILES["S-RE"]}
    if files["writer.pt"] != expected["writer_pt"] or files["record.json"] != expected["record_json"]:
        raise ValueError("S-RE differs from the registered parent")
    return files


def make_lock(report_path):
    registered()
    report = read(report_path)
    source = data.old.source_digest()
    inv, manifest = inventory(), data.manifest()
    if (report.get("status") != "complete" or report.get("passed") is not True
            or set(report.get("checks", {})) != set(REQUIRED)
            or any(report["checks"][k].get("passed") is not True for k in REQUIRED)
            or report.get("source_blake2b_128") != source or report.get("inventory") != inv
            or report.get("manifest") != manifest):
        raise ValueError("S1B preflight is incomplete/stale; no lock")
    return dict(stage="S1B", status="locked", manifest=manifest, inventory=inv, source_blake2b_128=source,
                prereg_blake2b_128=data.PREREG_DIGEST, preflight=report,
                preflight_blake2b_128=data.old.file_digest(report_path))


def require_lock(path):
    registered()
    lock = read(path)
    if (lock.get("stage") != "S1B" or lock.get("status") != "locked"
            or lock.get("prereg_blake2b_128") != data.PREREG_DIGEST
            or lock.get("source_blake2b_128") != data.old.source_digest()
            or lock.get("inventory") != inventory() or lock.get("manifest") != data.manifest()
            or any(lock.get("preflight", {}).get("checks", {}).get(k, {}).get("passed") is not True for k in REQUIRED)):
        raise ValueError("S1B source/manifest/inventory/preflight differs from its lock")
    return lock


def require_calibration(path, branch, lock_path):
    record = read(path)
    if (record.get("status") != "complete" or record.get("branch") != branch
            or record.get("lock_blake2b_128") != data.old.file_digest(lock_path)):
        raise ValueError("S1B calibration provenance mismatch")
    kinds = ["matrix"] if branch == "matrix_only" else ["matrix", "soft"]
    if len(record["candidates"]) != (3 if len(kinds) == 1 else 15):
        raise ValueError("incomplete S1B calibration grid")
    for kind in kinds:
        chosen = rules.choose_candidate(record["candidates"], kind)
        if chosen != record["selected"][kind]:
            raise ValueError("calibration choice differs from registered rule")
    rules.sanity(record)
    return record


def prepare_release(lock_path, branch_path, calibration_path, pilot_path, smoke_path, science_commit, *, unseen=False, reloads=None):
    lock, branch = require_lock(lock_path), require_branch(branch_path)
    cal = require_calibration(calibration_path, branch["branch"], lock_path)
    pilot, smoke = read(pilot_path), read(smoke_path)
    arms = rules.branch_arms(branch["branch"])
    if len(science_commit) != 40 or any(c not in "0123456789abcdef" for c in science_commit):
        raise ValueError("full science commit required")
    if set(pilot["arms"]) != set(arms) or set(smoke["arms"]) != set(arms):
        raise ValueError("incomplete pilot/smoke")
    for record in (pilot, smoke):
        if (record.get("status") != "complete" or record.get("branch") != branch["branch"]
                or record.get("lock_blake2b_128") != data.old.file_digest(lock_path)
                or record.get("calibration_blake2b_128") != data.old.file_digest(calibration_path)
                or record.get("source_commit") != science_commit):
            raise ValueError("pilot/smoke provenance mismatch")
    released = {}
    for arm in arms:
        p, s = pilot["arms"][arm], smoke["arms"][arm]
        if sorted(r["lr"] for r in p["candidates"]) != list(rules.LR_GRID):
            raise ValueError("pilot LR grid differs")
        chosen = rules.choose_lr(p["candidates"], p["step0_nll"])
        kind, depth = rules.ARM[arm]
        settings = cal["selected"][kind]
        read_spec = dict(kind=kind, gain=settings["gain"], tau=settings["tau"],
                         scope="all_valid_question", recency=False, capacity=12)
        if (s["steps"] != 20 or not s["freeze"]["digests"]["frozen_unchanged"]
                or not s["freeze"]["support"]["equal"] or not s["all_finite"]):
            raise ValueError("S1B smoke/freeze evidence failed")
        released[arm] = dict(lr=chosen["lr"], read=read_spec, trained_depth=depth,
                             walltime_seconds=rules.walltime(p["step_seconds"], p["eval_seconds"], p["load_seconds"], p["overhead_seconds"]),
                             smoke_metadata=s["export_metadata"])
    result = dict(stage="S1B", status="pending_reload", branch=branch["branch"], branch_evidence=branch,
                  lock_blake2b_128=data.old.file_digest(lock_path), inventory_blake2b_128=stable_hash(lock["inventory"]),
                  parent_writer_blake2b_128=lock["inventory"]["writer.pt"], source_commit=science_commit,
                  source_blake2b_128=lock["source_blake2b_128"], prereg_blake2b_128=data.PREREG_DIGEST,
                  parent_prereg_blake2b_128=data.old.PREREG_BLAKE2B_128,
                  analysis_script_blake2b_128=data.old.file_digest(ROOT / data.ANALYSIS),
                  arms=released, steps=data.STEPS, unseen=bool(unseen), contrasts=rules.contrasts(branch["branch"]),
                  calibration_blake2b_128=data.old.file_digest(calibration_path))
    result["config_hashes"] = rules.config_hashes(result)
    result["reload_basis"] = stable_hash(result)
    if reloads is not None:
        if set(reloads) != set(arms):
            raise ValueError("one reload report per active arm required")
        for arm, path in reloads.items():
            r = read(path)
            if (r.get("passed") is not True or r.get("arm") != arm or r.get("reload_basis") != result["reload_basis"]
                    or r.get("metadata") != released[arm]["smoke_metadata"]):
                raise ValueError("reload report differs from exact candidate")
        result.update(status="ready", reload_reports={arm: data.old.file_digest(path) for arm, path in reloads.items()})
    return result


def require_release(path, lock_path, *, ready=True):
    lock, release = require_lock(lock_path), read(path)
    if (release.get("stage") != "S1B" or release.get("status") not in ({"ready"} if ready else {"pending_reload", "ready"})
            or release.get("lock_blake2b_128") != data.old.file_digest(lock_path)
            or release.get("source_blake2b_128") != lock["source_blake2b_128"]
            or release.get("prereg_blake2b_128") != data.PREREG_DIGEST
            or release.get("parent_prereg_blake2b_128") != data.old.PREREG_BLAKE2B_128
            or release.get("analysis_script_blake2b_128") != data.old.file_digest(ROOT / data.ANALYSIS)
            or release.get("inventory_blake2b_128") != stable_hash(lock["inventory"])
            or release.get("parent_writer_blake2b_128") != lock["inventory"]["writer.pt"]
            or release.get("steps") != data.STEPS or release.get("contrasts") != rules.contrasts(release["branch"])
            or release.get("config_hashes") != rules.config_hashes(release)):
        raise ValueError("S1B release/source/lock/config provenance mismatch")
    source = release["branch_evidence"]
    if source != branch_record(source["files"]["decision"], source["files"]["dev"], source["files"].get("ablation")):
        raise ValueError("release branch differs from frozen S1A evidence")
    for kind in {rules.ARM[a][0] for a in rules.branch_arms(release["branch"])}:
        pair = [release["arms"][a]["read"] for a in rules.branch_arms(release["branch"]) if rules.ARM[a][0] == kind]
        if any(p != pair[0] for p in pair):
            raise ValueError("L1/L4 calibration differs within backend")
    return lock, release


def require_approval(path, release_path):
    a = read(path)
    if a.get("status") != "approved" or a.get("release_blake2b_128") != data.old.file_digest(release_path):
        raise PermissionError("orchestrator approval for this exact release is required")
