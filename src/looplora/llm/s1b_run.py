"""S1B compute entry points, with distinct lock/release/open-record gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from looplora.infra.seeds import stable_hash
from . import s1b_data as data, s1b_rules as rules, s1b_model as model, s1b_artifacts as artifacts
from . import s1b_analysis as statistics
from .evaluate import _slice
from .run import source_metadata
from .s1b_train import fit
from .slot_read import bind_batch
from .train import synchronize

RUN_ROOT = Path("/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b")


def output(path):
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("S1B outputs require an absolute path below its persistent root")
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("S1B output must be empty")
    return path


def save_samples(path, samples, **provenance):
    np.savez_compressed(path, **samples, provenance=np.asarray(json.dumps(provenance, sort_keys=True)))


def load_samples(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload if name != "provenance"}, json.loads(str(payload["provenance"].item()))


def common_provenance(lock_path, branch_path, calibration_path=None):
    branch = artifacts.require_branch(branch_path)
    record = dict(stage="S1B", branch=branch["branch"], lock_blake2b_128=data.old.file_digest(lock_path),
                  branch_blake2b_128=data.old.file_digest(branch_path), source_commit=source_metadata()["git_commit"])
    if calibration_path:
        record["calibration_blake2b_128"] = data.old.file_digest(calibration_path)
    return record


def selected_spec(cal, arm):
    choice = cal["selected"][rules.ARM[arm][0]]
    return rules.read_config(arm, choice["gain"], choice["tau"])


def calibration(lock_path, branch_path, out):
    lock = artifacts.require_lock(lock_path)
    provenance = common_provenance(lock_path, branch_path)
    branch = provenance["branch"]
    arrays = data.episodes(data.eval_set("calibration"))
    result, selected = [], {}
    exp = model.load_parent(branch, expected=lock["inventory"])
    try:
        for kind in ["matrix"] + ([] if branch == "matrix_only" else ["soft"]):
            arm = "M1" if kind == "matrix" else "S1"
            for tau in ((None,) if kind == "matrix" else rules.TAUS):
                for gain in rules.GAINS:
                    try:
                        value = model.evaluate(exp, arrays, rules.read_config(arm, gain, tau), 1)
                        result.append(dict(kind=kind, gain=gain, tau=tau, seconds=value["seconds"], **value["summary"]))
                    except FloatingPointError as error:
                        result.append(dict(kind=kind, gain=gain, tau=tau, nll=None, error=str(error), all_finite=False))
            try:
                selected[kind] = rules.choose_calibration(result, kind)
            except rules.StopForOrchestrator:
                artifacts.dump(out / "calibration.json", dict(provenance, status="failed", candidates=result, selected=selected))
                raise
        record = dict(provenance, status="complete", candidates=result, selected=selected)
        artifacts.dump(out / "calibration.json", record)
        rules.sanity(record)
        return record
    finally:
        exp.close()


def smoke(lock_path, branch_path, calibration_path, out):
    lock = artifacts.require_lock(lock_path)
    provenance = common_provenance(lock_path, branch_path, calibration_path)
    branch = provenance["branch"]
    cal = artifacts.require_calibration(calibration_path, branch, lock_path)
    raw = data.raw_episodes(data.eval_set("smoke"))
    dev = data.episodes(data.eval_set("smoke"), [1, 2, 3])
    reports = {}
    for arm in rules.branch_arms(branch):
        exp = model.load_parent(branch, expected=lock["inventory"])
        try:
            spec = selected_spec(cal, arm)
            config = rules.science_config(branch, arm, 17, 1e-3, spec.as_dict())
            source = iter([(_slice(raw, slice((i % 2) * 32, (i % 2 + 1) * 32)), None) for i in range(20)])
            record = fit(exp, spec, arm=arm, seed=17, lr=1e-3, steps=20, out=out / arm, dev=dev,
                         dev_steps=[20], support=raw, parent_digest=lock["inventory"]["writer.pt"],
                         science_hash=stable_hash(config), source=source)
            if record["selected_step"] != 20:
                raise ValueError("smoke must select its sole step-20 checkpoint")
            original = model.evaluate(exp, dev, spec, rules.ARM[arm][1])["samples"]
            save_samples(out / arm / "reload_samples.npz", original,
                         checkpoint_blake2b_128=record["selected_checkpoint_blake2b_128"], step=20)
            reports[arm] = record
        finally:
            exp.close()
        restored = model.load_export(out / arm / "selected.pt")
        try:
            result = model.evaluate(restored, dev, spec, rules.ARM[arm][1])
            if set(result["samples"]) != set(original) or any(not np.array_equal(result["samples"][k], original[k]) for k in original):
                raise ValueError("first S1B reload differs from original in-memory smoke")
            reports[arm]["export_metadata"] = restored.s1b
            reports[arm]["original_smoke_samples_blake2b_128"] = data.old.file_digest(out / arm / "reload_samples.npz")
            reports[arm]["first_reload_matches_original"] = True
        finally:
            restored.close()
    record = dict(provenance, status="complete", arms=reports)
    artifacts.dump(out / "smoke.json", record)
    return record


def pilot(lock_path, branch_path, calibration_path, out):
    lock = artifacts.require_lock(lock_path)
    provenance = common_provenance(lock_path, branch_path, calibration_path)
    branch = provenance["branch"]
    cal = artifacts.require_calibration(calibration_path, branch, lock_path)
    dev, support = data.episodes(data.eval_set("pilot")), data.raw_episodes(data.eval_set("smoke"))
    arms = {}
    for arm in rules.branch_arms(branch):
        rows, load_times = [], []
        for lr in rules.LR_GRID:
            tick = time.perf_counter()
            exp = model.load_parent(branch, expected=lock["inventory"])
            synchronize(exp.device)
            load_times.append(time.perf_counter() - tick)
            try:
                record = fit(exp, selected_spec(cal, arm), arm=arm, seed=17, lr=lr, steps=data.PILOT_STEPS,
                             stream_seed=data.TRAIN_BASE["pilot"], out=out / arm / str(lr), dev=dev,
                             dev_steps=[data.PILOT_STEPS], support=support,
                             parent_digest=lock["inventory"]["writer.pt"], science_hash="pilot")
                rows.append(dict(lr=lr, nll=record["dev"][-1]["nll"], all_finite=record["all_finite"],
                                 step0_nll=record["dev"][0]["nll"], step_seconds=record["step_seconds"],
                                 eval_seconds=max(r["seconds"] for r in record["dev"]), export_seconds=record["export_seconds"],
                                 overhead_seconds=record["setup_seconds"] + 10 * record["checkpoint_overhead_seconds"] + record["finalize_seconds"]))
            except FloatingPointError as error:
                rows.append(dict(lr=lr, nll=None, all_finite=False, error=str(error)))
            finally:
                exp.close()
        finite = [r for r in rows if r["all_finite"]]
        if not finite:
            artifacts.dump(out / "pilot.json", dict(provenance, status="failed", arms=arms, failed_arm=arm, candidates=rows))
            raise rules.StopForOrchestrator("no finite S1B pilot candidate")
        step0 = finite[0]["step0_nll"]
        try:
            chosen = rules.choose_lr(rows, step0)
        except rules.StopForOrchestrator:
            artifacts.dump(out / "pilot.json", dict(provenance, status="failed", arms=arms, failed_arm=arm, candidates=rows))
            raise
        arms[arm] = dict(candidates=rows, step0_nll=step0, selected_lr=chosen["lr"],
                         step_seconds=max(r["step_seconds"] for r in finite), load_seconds=max(load_times),
                         eval_seconds=max(r["eval_seconds"] for r in finite), export_seconds=max(r["export_seconds"] for r in finite),
                         overhead_seconds=max(r["overhead_seconds"] for r in finite))
    result = dict(provenance, status="complete", arms=arms)
    artifacts.dump(out / "pilot.json", result)
    return result


def reload_check(lock_path, release_path, export, predictions, arm, out):
    _, release = artifacts.require_release(release_path, lock_path, ready=False)
    if arm not in release["arms"]:
        raise ValueError("reload arm absent from candidate")
    exp = model.load_export(export)
    try:
        expected = release["arms"][arm]
        if data.old.file_digest(predictions) != expected["original_smoke_samples_blake2b_128"]:
            raise ValueError("original smoke sample file differs from candidate release")
        if exp.s1b != expected["smoke_metadata"] or exp.read_spec.as_dict() != expected["read"]:
            raise ValueError("export metadata differs from candidate release")
        actual = model.evaluate(exp, data.episodes(data.eval_set("smoke"), [1, 2, 3]), exp.read_spec, rules.ARM[arm][1])["samples"]
        saved, _ = load_samples(predictions)
        if set(actual) != set(saved) or any(not np.array_equal(actual[k], saved[k]) for k in actual):
            raise ValueError("S1B export reload failed prediction/NLL/ID check")
        record = dict(passed=True, arm=arm, metadata=exp.s1b, reload_basis=release["reload_basis"],
                      original_smoke_samples_blake2b_128=data.old.file_digest(predictions),
                      export_blake2b_128=data.old.file_digest(export))
        artifacts.dump(out / "reload.json", record)
        return record
    finally:
        exp.close()


def train(lock_path, release_path, approval_path, arm, seed, out):
    lock, release = artifacts.require_release(release_path, lock_path)
    artifacts.require_approval(approval_path, release_path)
    if arm not in rules.branch_arms(release["branch"]) or seed not in data.RUN_SEEDS:
        raise ValueError("unregistered main run")
    settings = release["arms"][arm]
    spec = model.ReadSpec.from_dict(settings["read"])
    config = rules.science_config(release["branch"], arm, seed, settings["lr"], spec.as_dict())
    exp = model.load_parent(release["branch"], expected=lock["inventory"])
    try:
        result = fit(exp, spec, arm=arm, seed=seed, lr=settings["lr"], steps=data.STEPS, out=out,
                     dev=data.episodes(data.eval_set("dev")), dev_steps=rules.checkpoint_steps(data.STEPS),
                     support=data.raw_episodes(data.eval_set("smoke")), parent_digest=lock["inventory"]["writer.pt"],
                     science_hash=stable_hash(config))
        result.update(release_blake2b_128=data.old.file_digest(release_path), source=source_metadata(),
                      lock_blake2b_128=data.old.file_digest(lock_path), read=spec.as_dict())
        artifacts.dump(out / "record.json", result)
        return result
    finally:
        exp.close()


def evaluate_selected(lock_path, release_path, selection_path, entries_path, opening_root, purpose, out):
    lock, release = artifacts.require_release(release_path, lock_path)
    selection = artifacts.read(selection_path)
    if purpose == "dev":
        if selection.get("status") != "selected" or selection.get("release_blake2b_128") != data.old.file_digest(release_path):
            raise ValueError("dev sweep requires completed selection for this release")
        expected = {f"{a}/{s}" for a in rules.branch_arms(release["branch"]) for s in data.RUN_SEEDS}
        if set(selection["selected_checkpoints"]) != expected:
            raise ValueError("incomplete dev selection")
        token = data.OpenToken(selection["selected_checkpoints"], data.old.file_digest(release_path))
    else:
        token = data.load_open_token(opening_root, release_path, lock, selection)
    if purpose not in {"dev", "confirmation", "unseen"} or (purpose == "unseen" and not release["unseen"]):
        raise ValueError("unregistered sealed evaluation")
    entries = artifacts.read(entries_path)
    expected = token.selected
    if not isinstance(entries, dict) or set(entries) != set(expected):
        raise ValueError("one export per selected main label is required")
    index = {}
    for label, export in entries.items():
        arm, seed_text = label.split("/")
        digest = data.old.file_digest(export)
        if digest != expected[label]:
            raise data.old.SealedSetError("checkpoint digest does not match arm/seed label")
        exp = model.load_export(export)
        try:
            if (exp.s1b["arm"] != arm or exp.s1b["seed"] != int(seed_text) or exp.s1b["control"] != "shared"
                    or exp.s1b["branch"] != release["branch"] or exp.s1b["science_hash"] != release["config_hashes"][label]
                    or exp.read_spec.as_dict() != release["arms"][arm]["read"]
                    or exp.s1b["parent_writer_blake2b_128"] != release["parent_writer_blake2b_128"]):
                raise data.old.SealedSetError("checkpoint export/label architecture mismatch")
            arrays = data.episodes(data.eval_set(purpose, token=token, checkpoint_digest=digest), list(range(1, 9)))
            depths = (1, 2, 4, 8) if arm.endswith("4") else (1,)
            for depth in depths:
                result = model.evaluate(exp, arrays, exp.read_spec, depth, diagnostics=True)
                statistics.validate_samples(result["samples"], lock["manifest"]["ids"][purpose])
                statistics.validate_trajectories(result["samples"], depth)
                provenance = dict(label=label, checkpoint_blake2b_128=digest, evaluated_depth=depth,
                                  trained_depth=rules.ARM[arm][1], release_blake2b_128=token.release_digest,
                                  lock_blake2b_128=data.old.file_digest(lock_path), read=exp.read_spec.as_dict(),
                                  task_digest=data.LAYOUT_DIGEST, trace_agrees=True)
                provenance["trajectory_order"] = ["episode", "hop", "step", "feature"]
                filename = f"{arm}-s{seed_text}-L{depth}.npz"
                save_samples(out / filename, result["samples"], **provenance)
                index[f"{label}@{depth}"] = dict(file=filename, blake2b_128=data.old.file_digest(out / filename),
                                                costs=result["costs"], seconds=result["seconds"])
        finally:
            exp.close()
    artifacts.dump(out / "index.json", dict(stage="S1B", purpose=purpose, entries=index,
                    release_blake2b_128=token.release_digest, open_record_blake2b_128=None if purpose == "dev" else data.old.file_digest(
                        Path(opening_root) / "reports/slot-loop-v1/s1b/open.json")))
    return dict(status="complete", entries=len(index))


def analyze_dev(lock_path, release_path, records_root):
    lock, release = artifacts.require_release(release_path, lock_path)
    records = {f"{a}/{s}": artifacts.read(Path(records_root) / a / f"s{s}" / "record.json")
               for a in rules.branch_arms(release["branch"]) for s in data.RUN_SEEDS}
    for label, record in records.items():
        arm, seed = label.split("/")
        for row in record["dev"]:
            path = Path(record["artifact_root"]) / row["checkpoint"]["file"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            meta = payload["s1b"]
            if (meta["step"] != row["step"] or meta["arm"] != arm or meta["seed"] != int(seed)
                    or meta["science_hash"] != release["config_hashes"][label]
                    or meta["read"] != release["arms"][arm]["read"]):
                raise ValueError("dev row export metadata differs from its checkpoint/arm/seed")
    return statistics.select(records, release, lock, data.old.file_digest(release_path))


def analyze_confirmation(lock_path, release_path, selection_path, opening_root, records_root, unseen_root=None):
    lock, release = artifacts.require_release(release_path, lock_path)
    selection = artifacts.read(selection_path)
    token = data.load_open_token(opening_root, release_path, lock, selection)
    root = Path(records_root)
    index = artifacts.read(root / "index.json")
    if index.get("purpose") != "confirmation" or index.get("release_blake2b_128") != token.release_digest:
        raise ValueError("confirmation index belongs to another set/release")
    main, sweeps = {}, {}
    expected = set()
    for arm in rules.branch_arms(release["branch"]):
        for seed in data.RUN_SEEDS:
            depths = (1, 2, 4, 8) if arm.endswith("4") else (1,)
            for depth in depths:
                label = f"{arm}/{seed}"
                key = f"{label}@{depth}"
                expected.add(key)
                entry = index["entries"][key]
                path = root / entry["file"]
                if data.old.file_digest(path) != entry["blake2b_128"]:
                    raise ValueError("edited confirmation sample file")
                samples, provenance = load_samples(path)
                if (provenance.get("label") != label or provenance.get("checkpoint_blake2b_128") != token.selected[label]
                        or provenance.get("release_blake2b_128") != token.release_digest
                        or provenance.get("evaluated_depth") != depth or provenance.get("trained_depth") != rules.ARM[arm][1]
                        or provenance.get("read") != release["arms"][arm]["read"] or provenance.get("trace_agrees") is not True
                        or provenance.get("lock_blake2b_128") != data.old.file_digest(lock_path)):
                    raise ValueError("confirmation row provenance mismatch")
                statistics.validate_samples(samples, lock["manifest"]["ids"]["confirmation"])
                statistics.validate_trajectories(samples, depth)
                if provenance.get("trajectory_order") != ["episode", "hop", "step", "feature"]:
                    raise ValueError("trajectory ordering provenance differs")
                if depth == rules.ARM[arm][1]:
                    main[arm, seed] = samples
                if arm.endswith("4"):
                    sweeps[arm, seed, depth] = samples
    if set(index["entries"]) != expected:
        raise ValueError("extra confirmation rows")
    result = statistics.paired(main, sweeps, release["branch"])
    result.update(release_blake2b_128=token.release_digest,
                  open_record_blake2b_128=data.old.file_digest(Path(opening_root) / "reports/slot-loop-v1/s1b/open.json"),
                  sample_index_blake2b_128=data.old.file_digest(root / "index.json"))
    result["unseen"] = dict(status="not run")
    if release["unseen"]:
        if unseen_root is None:
            raise ValueError("release enabled unseen: complete unseen records are required")
        unseen_root = Path(unseen_root)
        unseen_index = artifacts.read(unseen_root / "index.json")
        if (unseen_index.get("purpose") != "unseen" or set(unseen_index["entries"]) != expected
                or unseen_index.get("release_blake2b_128") != token.release_digest):
            raise ValueError("unseen index incomplete or from another release")
        descriptive = {}
        for key, entry in unseen_index["entries"].items():
            label, depth_text = key.split("@")
            arm, _ = label.split("/")
            path = unseen_root / entry["file"]
            if data.old.file_digest(path) != entry["blake2b_128"]:
                raise ValueError("unseen file digest differs")
            samples, provenance = load_samples(path)
            if (provenance.get("label") != label or provenance.get("checkpoint_blake2b_128") != token.selected[label]
                    or provenance.get("release_blake2b_128") != token.release_digest
                    or provenance.get("read") != release["arms"][arm]["read"]
                    or provenance.get("evaluated_depth") != int(depth_text)):
                raise ValueError("unseen row provenance mismatch")
            statistics.validate_samples(samples, lock["manifest"]["ids"]["unseen"])
            statistics.validate_trajectories(samples, int(depth_text))
            descriptive[key] = model.per_hop(samples)
        result["unseen"] = dict(status="complete", descriptive=descriptive)
    return result


def require_loop_gain(verdict_path, release_path, arm):
    verdict = artifacts.read(verdict_path)
    if (arm not in {"M4", "S4"} or verdict.get("stage") != "S1B"
            or verdict.get("release_blake2b_128") != data.old.file_digest(release_path)
            or verdict.get("contrasts", {}).get(f"{arm}-{arm[0]}1", {}).get("passed") is not True):
        raise PermissionError("mechanism controls require this backend's confirmed loop gain")
    return verdict


def benchmark(lock_path, release_path, arm, out, *, control_kind="shared", verdict_path=None):
    lock, release = artifacts.require_release(release_path, lock_path)
    if arm not in release["arms"]:
        raise ValueError("benchmark arm is not active")
    if control_kind != "shared":
        require_loop_gain(verdict_path, release_path, arm)
    exp = model.load_parent(release["branch"], control=control_kind, expected=lock["inventory"])
    try:
        arrays = data.episodes(data.eval_set("smoke"), [1, 2, 3])
        spec = model.ReadSpec.from_dict(release["arms"][arm]["read"])
        for _ in range(5):
            model.evaluate(exp, arrays, spec, rules.ARM[arm][1])
        uninstrumented = [model.evaluate(exp, arrays, spec, rules.ARM[arm][1]) for _ in range(20)]
        rows = [model.evaluate(exp, arrays, spec, rules.ARM[arm][1], timing=True) for _ in range(20)]
        kernels = [t for r in rows for cost in r["costs"] for t in cost["read_kernel_seconds"]]
        times = [r["seconds"] for r in uninstrumented]
        record = dict(stage="S1B", arm=arm, control=control_kind, samples=20, warmup=5,
                      release_blake2b_128=data.old.file_digest(release_path),
                      kernel_median_seconds=float(np.median(kernels)), kernel_p90_seconds=float(np.quantile(kernels, .9)),
                      forward_median_seconds=float(np.median(times)), forward_p90_seconds=float(np.quantile(times, .9)),
                      costs=rows[-1]["costs"], note="synchronized instrumented reads; end-to-end uninstrumented time is separate")
        artifacts.dump(out / "benchmark.json", record)
        return record
    finally:
        exp.close()


def control(lock_path, release_path, approval_path, extension_path, arm, seed, kind, out):
    lock, release = artifacts.require_release(release_path, lock_path)
    artifacts.require_approval(approval_path, release_path)
    extension = artifacts.read(extension_path)
    require_loop_gain(extension["verdict_path"], release_path, arm)
    settings = release["arms"][arm]
    spec = model.ReadSpec.from_dict(settings["read"])
    config = rules.science_config(release["branch"], arm, seed, settings["lr"], spec.as_dict(), control=kind)
    label = f"{arm}/{seed}/{kind}"
    if (extension.get("main_release_blake2b_128") != data.old.file_digest(release_path)
            or extension.get("verdict_blake2b_128") != data.old.file_digest(extension["verdict_path"])
            or extension.get("source_blake2b_128") != lock["source_blake2b_128"]
            or extension.get("lock_blake2b_128") != data.old.file_digest(lock_path)
            or extension.get("configs", {}).get(label) != stable_hash(config)):
        raise ValueError("control release extension/config/source mismatch")
    exp = model.load_parent(release["branch"], expected=lock["inventory"])
    try:
        support = data.raw_episodes(data.eval_set("smoke"))
        if kind == "untied":
            with torch.no_grad():
                batch = exp.prepare(_slice(support, slice(0, 4)))
                bound = bind_batch(exp, batch, spec)
                shared = exp.score(bound, 4)[0]
                exp.adapter.untie()
                untied = exp.score(bound, 4)[0]
                if not torch.equal(shared, untied):
                    raise RuntimeError("untied warm-start is not bitwise equal")
        else:
            exp.adapter.control = "fixed"
        exp.s1b_control = kind
        result = fit(exp, spec, arm=arm, seed=seed, lr=settings["lr"], steps=data.STEPS, out=out,
                     dev=data.episodes(data.eval_set("dev")), dev_steps=rules.checkpoint_steps(data.STEPS), support=support,
                     parent_digest=lock["inventory"]["writer.pt"], science_hash=stable_hash(config))
        result.update(main_release_blake2b_128=data.old.file_digest(release_path),
                      control_release_blake2b_128=data.old.file_digest(extension_path), descriptive=True)
        artifacts.dump(out / "record.json", result)
        return result
    finally:
        exp.close()


def control_pilot(lock_path, release_path, verdict_path, arm, kind, out):
    lock, release = artifacts.require_release(release_path, lock_path)
    require_loop_gain(verdict_path, release_path, arm)
    started = time.perf_counter()
    exp = model.load_parent(release["branch"], control=kind, expected=lock["inventory"])
    synchronize(exp.device)
    load = time.perf_counter() - started
    try:
        settings = release["arms"][arm]
        result = fit(exp, model.ReadSpec.from_dict(settings["read"]), arm=arm, seed=17, lr=settings["lr"],
            steps=data.PILOT_STEPS, out=out, dev=data.episodes(data.eval_set("pilot")), dev_steps=[data.PILOT_STEPS],
            stream_seed=data.TRAIN_BASE["pilot"], support=data.raw_episodes(data.eval_set("smoke")),
            parent_digest=lock["inventory"]["writer.pt"], science_hash="control-timing")
        timing = dict(arm=arm, control=kind, steps=data.PILOT_STEPS, step_seconds=result["step_seconds"],
                      eval_seconds=max(row["seconds"] for row in result["dev"]), load_seconds=load,
                      export_seconds=result["export_seconds"], release_blake2b_128=data.old.file_digest(release_path),
                      overhead_seconds=result["setup_seconds"] + 10 * result["checkpoint_overhead_seconds"] + result["finalize_seconds"],
                      verdict_blake2b_128=data.old.file_digest(verdict_path))
        artifacts.dump(out / "timing.json", timing)
        return timing
    finally:
        exp.close()


def analysis_main(argv=None):
    p = argparse.ArgumentParser(description="Frozen S1B dev selection and confirmation analysis")
    p.add_argument("--stage", required=True, choices=("DEV", "CONFIRMATION"))
    for name in ("manifest", "release", "records", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--open-root", type=Path)
    p.add_argument("--unseen-records", type=Path)
    a = p.parse_args(argv)
    out = output(a.out)
    if a.stage == "DEV":
        result = analyze_dev(a.manifest, a.release, a.records)
        name = "selection.json"
    else:
        if a.selection is None or a.open_root is None:
            p.error("confirmation requires --selection and --open-root")
        result = analyze_confirmation(a.manifest, a.release, a.selection, a.open_root, a.records, a.unseen_records)
        name = "verdict.json"
        (out / "verdict.md").write_text(statistics.confirmation_markdown(result))
    artifacts.dump(out / name, result)
    artifacts.dump(out / "source_index.json", dict(analysis_script_blake2b_128=data.old.file_digest(artifacts.ROOT / data.ANALYSIS),
        release_blake2b_128=data.old.file_digest(a.release), manifest_blake2b_128=data.old.file_digest(a.manifest),
        output_blake2b_128=data.old.file_digest(out / name)))
    print(json.dumps(dict(stage=a.stage, status="complete", digest=data.old.file_digest(out / name))))
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("preflight", "calibration", "smoke", "pilot", "reload-check", "train", "evaluate", "control", "control-pilot", "benchmark"))
    p.add_argument("--out", type=Path, required=True)
    for name in ("manifest", "branch-record", "calibration", "release", "approval", "export", "predictions", "selection", "entries", "open-root", "control-release", "verdict"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--arm", choices=tuple(rules.ARM))
    p.add_argument("--seed", type=int, choices=data.RUN_SEEDS)
    p.add_argument("--set", dest="purpose", choices=("dev", "confirmation", "unseen"), default="confirmation")
    p.add_argument("--control", choices=("fixed", "untied"))
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    required = dict(preflight=[], calibration=["manifest", "branch_record"],
                    smoke=["manifest", "branch_record", "calibration"], pilot=["manifest", "branch_record", "calibration"],
                    **{"reload-check": ["manifest", "release", "export", "predictions", "arm"]},
                    train=["manifest", "release", "approval", "arm", "seed"],
                    evaluate=["manifest", "release", "selection", "entries"],
                    control=["manifest", "release", "approval", "control_release", "arm", "seed", "control"],
                    benchmark=["manifest", "release", "arm"],
                    **{"control-pilot": ["manifest", "release", "verdict", "arm", "control"]})
    for name in required[a.mode]:
        if getattr(a, name) is None:
            p.error(f"--{name.replace('_', '-')} is required for {a.mode}")
    if a.mode == "evaluate" and a.purpose != "dev" and a.open_root is None:
        p.error("sealed evaluation requires --open-root")
    out = output(a.out)
    if a.mode == "preflight":
        from .s1b_preflight import run
        run(out)
    elif a.mode == "calibration":
        calibration(a.manifest, a.branch_record, out)
    elif a.mode in {"smoke", "pilot"}:
        {"smoke": smoke, "pilot": pilot}[a.mode](a.manifest, a.branch_record, a.calibration, out)
    elif a.mode == "reload-check":
        reload_check(a.manifest, a.release, a.export, a.predictions, a.arm, out)
    elif a.mode == "train":
        train(a.manifest, a.release, a.approval, a.arm, a.seed, out)
    elif a.mode == "evaluate":
        evaluate_selected(a.manifest, a.release, a.selection, a.entries, a.open_root, a.purpose, out)
    elif a.mode == "benchmark":
        benchmark(a.manifest, a.release, a.arm, out, control_kind=a.control or "shared", verdict_path=a.verdict)
    elif a.mode == "control-pilot":
        control_pilot(a.manifest, a.release, a.verdict, a.arm, a.control, out)
    else:
        control(a.manifest, a.release, a.approval, a.control_release, a.arm, a.seed, a.control, out)
    print(json.dumps(dict(mode=a.mode, status="complete", out=str(out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
