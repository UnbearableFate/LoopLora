"""SLOT-LOOP-v1 runner: preflight, S0, and the S1A smoke, pilot, train and evaluate modes (docs/prereg/SLOT-LOOP-v1.md; plan 13.4).

    python -m looplora.llm.slot_run --mode MODE --config configs/slot_loop_v1/{s0,s1a}.json --out ABSOLUTE_DIR [...]

Compute only. Every mode but `preflight` refuses to run without a locked manifest (`--manifest`: the registry, the real parents'
digests and a passed preflight), the stage config must equal the registered rendering (`slot_rules.stage_config`), and sealed
evaluation sets open only with the orchestrator's open record (`slot_data.eval_set`). Outputs go under
/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/. The mode functions take their data and parents as arguments so the tests run the
same code on a tiny model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch

from looplora.infra.seeds import stable_hash
from . import slot_data, slot_preflight, slot_rules
from .evaluate import _slice
from .run import source_metadata
from .slot_experiment import (PARENT_FOLDERS, episodes, evaluate, file_digest, load_export, load_parent, load_value_head,
                              raw_episodes, summarize)
from .slot_read import ReadSpec
from .slot_train import fit
from .sq2_release import PARENT as D128_3

RUNS = Path("/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1")
PRIMARY, DIAGNOSTIC = "all_valid_question", "final_only"
MODES = ("preflight", "s0", "smoke", "pilot", "train", "evaluate", "reload-check")
REGISTERED_COUNTS = dict(trainable_parameters=180_224, frozen_parameters=294_914)


class StopRun(RuntimeError):
    """A registered stop: the record is written, the orchestrator decides."""


def write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


def provenance() -> dict:
    return dict(**source_metadata(), host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


# ------------------------------------------------------------------------------------------ config, inventory and lock

def check_config(path, stage: str) -> dict:
    """The stage config on disk must equal the registered rendering; returns it with its stable hash."""
    found, expected = json.loads(Path(path).read_text()), slot_rules.stage_config(stage)
    if found != expected:
        raise ValueError(f"{path} differs from the registered {stage} configuration")
    return dict(config=expected, hash=stable_hash(expected))


def require_lock(path) -> dict:
    """A locked manifest: registry and disjointness re-derived now equal the lock's, the preflight passed, the parents' files
    have the locked digests."""
    lock = json.loads(Path(path).read_text())
    if lock.get("status") != "locked" or lock.get("prereg_blake2b_128") != slot_data.PREREG_BLAKE2B_128:
        raise StopRun(f"{path} is not a locked manifest of the registered prereg")
    if lock.get("preflight", {}).get("passed") is not True:
        raise StopRun("the preflight did not pass; no science runs")
    if lock.get("preflight", {}).get("source_blake2b_128") != slot_data.source_digest():
        raise StopRun("the source differs from the source that passed the preflight; rerun preflight")
    if lock.get("manifest") != json.loads(json.dumps(slot_data.manifest())):
        raise StopRun("the registry differs from the locked manifest")
    if lock.get("inventory") != slot_data.inventory():
        raise StopRun("a parent file differs from the locked inventory")
    return lock


# ------------------------------------------------------------------------------------------------------------ S0 core

def s0_conditions(selected: dict) -> dict:
    """name -> (ReadSpec or None, meta) of the dev table: roles at all_valid_question, then the final_only table."""
    m, s, h = selected["matrix"], selected["soft"], selected["hard"]
    soft = dict(gain=s["gain"], tau=s["tau"])
    table = [("native", PRIMARY, None), ("matrix_uncal", PRIMARY, ReadSpec("matrix")), ("matrix_cal", PRIMARY, ReadSpec("matrix", m["gain"])),
             ("soft_uncal", PRIMARY, ReadSpec("soft", 1.0, tau=1.0)), ("soft_cal", PRIMARY, ReadSpec("soft", **soft)),
             ("hard_uncal", PRIMARY, ReadSpec("hard")), ("hard_cal", PRIMARY, ReadSpec("hard", h["gain"])),
             ("soft_rec_uncal", PRIMARY, ReadSpec("soft", 1.0, tau=1.0, recency=True)),
             ("soft_rec_cal", PRIMARY, ReadSpec("soft", **soft, recency=True)),
             ("matrix_cal", DIAGNOSTIC, ReadSpec("matrix", m["gain"], scope=DIAGNOSTIC)),
             ("soft_cal", DIAGNOSTIC, ReadSpec("soft", **soft, scope=DIAGNOSTIC)),
             ("hard_cal", DIAGNOSTIC, ReadSpec("hard", h["gain"], scope=DIAGNOSTIC)),
             ("clean_final", DIAGNOSTIC, ReadSpec("clean_final", scope=DIAGNOSTIC))]
    out = {}
    for role, scope, spec in table:
        meta = dict(role=role, scope=scope, kind="native" if spec is None else spec.kind, gain=1.0 if spec is None else spec.gain,
                    tau=None if spec is None else spec.tau, recency=False if spec is None else spec.recency)
        out[f"{role}/{scope}"] = (spec, meta)
    return out


def recheck_conditions(selected: dict) -> dict:
    """D128-3, once, at S-RE's calibrated values (no recalibration): its own read, Matrix, SoftSlot and HardSlot."""
    full = s0_conditions(selected)
    pick = dict(native="native", matrix="matrix_cal", soft="soft_cal", hard="hard_cal")
    return {f"{role}/{PRIMARY}": (full[f"{source}/{PRIMARY}"][0], dict(full[f"{source}/{PRIMARY}"][1], role=role))
            for role, source in pick.items()}


def calibrate(experiment, arrays) -> dict:
    """The 18 registered candidates on the calibration episodes (all_valid_question); NLL is the float64 mean of the per-sample
    full-vocabulary NLL. Returns the candidate rows and the registered choice per kind."""
    grid = slot_rules.calibration_grid()
    specs = {f"{c['kind']}/{c['gain']}/{c['tau']}": ReadSpec(c["kind"], c["gain"], tau=c["tau"]) for c in grid}
    result = evaluate(experiment, arrays, specs, diagnostics=False)
    rows = []
    for candidate, label in zip(grid, specs):
        sample = result["samples"][label]
        rows.append(dict(**candidate, nll=float(sample["nll"].astype(np.float64).mean()), correct=int(sample["correct"].sum()),
                         seconds=result["seconds"][label]))
    selected = {kind: slot_rules.choose_candidate(rows, kind) for kind in ("matrix", "soft", "hard")}
    return dict(n=len(arrays.episode_seed), candidates=rows, prepare_seconds=result["prepare_seconds"],
                selected={k: {f: v[f] for f in ("gain", "tau") if f in v and (f != "tau" or v[f] is not None)} for k, v in selected.items()})


def condition_table(experiment, arrays, conditions: dict, value_head=None) -> dict:
    """Evaluate a `s0_conditions`-style table with diagnostics: {metrics by name, samples by name, common, seconds}."""
    result = evaluate(experiment, arrays, {name: spec for name, (spec, _) in conditions.items()}, diagnostics=True,
                      value_head=value_head)
    metrics = {}
    for name, (_, meta) in conditions.items():
        metrics[name] = dict(**meta, **summarize(result["samples"][name]), seconds=result["seconds"][name])
    common = {k: float(v.astype(np.float64).mean()) for k, v in result["common"].items()}
    return dict(metrics=metrics, samples=result["samples"], common=common, prepare_seconds=result["prepare_seconds"])


def s0_core(experiment, calibration_arrays, dev_arrays, *, value_head=None) -> dict:
    cal = calibrate(experiment, calibration_arrays)
    table = condition_table(experiment, dev_arrays, s0_conditions(cal["selected"]), value_head)
    return dict(calibration=cal, dev=table)


def old_range_record(experiment, name: str) -> dict:
    """The recorded identity of a parent on the old range (a replay, never a selection): counts and the float32 mean NLL."""
    folder = PARENT_FOLDERS[name]
    check = slot_preflight.native_replay(experiment, name, folder)
    return dict(correct=check["replayed"]["correct"], n=1000, loss_float32=check["replayed"]["loss_float32"], passed=check["passed"])


def save_samples(path, samples: dict) -> None:
    flat = {f"{name}/{field}": array for name, fields in samples.items() for field, array in fields.items()}
    np.savez_compressed(path, **flat)


def ids_record(eval_set, arrays) -> str:
    """The set's id digest from the generated arrays, equal to the registered one (a mismatch is a stop)."""
    found = slot_data.ids_digest(arrays.episode_seed.tolist())
    if found != slot_data.ids_digest(eval_set.episode_ids()):
        raise StopRun(f"{eval_set.name}: the generated episode ids differ from the registered ones")
    return found


def run_s0(out, config_path, manifest_path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = check_config(config_path, "S0")
    lock = require_lock(manifest_path)
    digests = lock["inventory"]["S-RE"]
    sets = {p: slot_data.eval_set("S0", p) for p in ("calibration", "dev")}
    arrays = {p: episodes(s) for p, s in sets.items()}
    ids = {p: ids_record(sets[p], arrays[p]) for p in sets}
    tick = time.perf_counter()
    experiment = load_parent("S-RE", expected={f: digests[f] for f in ("writer.pt", "checkpoint.pt")})
    load_seconds = time.perf_counter() - tick
    model, tokenizer = experiment.model, experiment.tokenizer
    try:
        core = s0_core(experiment, arrays["calibration"], arrays["dev"], value_head=load_value_head("S-RE", experiment.device))
    finally:
        experiment.close()
    selected = core["calibration"]["selected"]
    d128 = load_parent("D128-3", model=model, tokenizer=tokenizer, expected=D128_3["digests"])
    try:
        recheck = condition_table(d128, arrays["dev"], recheck_conditions(selected))
        d128_record = dict(conditions=recheck["metrics"], old_range=old_range_record(d128, "D128-3"))
    finally:
        d128.close()
    core["calibration"].update(ids_blake2b_128=ids["calibration"])
    metrics = dict(version=1, stage="S0", n=core["dev"]["metrics"]["native/all_valid_question"]["n"], ids_blake2b_128=ids["dev"],
                   calibration=core["calibration"], conditions=core["dev"]["metrics"], common=core["dev"]["common"], d128_3=d128_record,
                   config_hash=config["hash"], manifest_status=lock["status"], load_seconds=load_seconds, **provenance())
    write_json(out / "metrics.json", metrics)
    write_json(out / "calibration.json", core["calibration"])
    save_samples(out / "samples.npz", core["dev"]["samples"])
    save_samples(out / "d128_3_samples.npz", recheck["samples"])
    return metrics


# ------------------------------------------------------------------------------------------------------------- preflight

def run_preflight(out, config_path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = check_config(config_path, "S0")
    manifest = slot_data.manifest()
    inv = slot_data.inventory()
    smoke = slot_data.eval_set("S0", "smoke")
    smoke_eval, smoke_train = episodes(smoke), raw_episodes(smoke)
    checks, identity = {}, {}
    expected = {f: inv["S-RE"][f] for f in ("writer.pt", "checkpoint.pt")}
    tick = time.perf_counter()
    experiment = load_parent("S-RE", expected=expected)
    load_seconds = time.perf_counter() - tick
    model, tokenizer = experiment.model, experiment.tokenizer
    try:
        identity["S-RE"] = checks["native_replay"] = slot_preflight.native_replay(experiment, "S-RE", PARENT_FOLDERS["S-RE"])
        parts = dict(old_range=slot_preflight.matrix_identity(experiment, slot_preflight.old_range(experiment)),
                     smoke_eval_depths=slot_preflight.matrix_identity(experiment, smoke_eval, depths=(1, 2, 4)),
                     smoke_train_shaped=slot_preflight.matrix_identity(experiment, smoke_train))
        checks["matrix_equals_native"] = dict(passed=all(p["passed"] for p in parts.values()), parts=parts,
                                              max_abs_difference=max(p["max_abs_difference"] for p in parts.values()))
        identity["matrix_gradients"] = slot_preflight.matrix_gradients(experiment, _slice(smoke_train, slice(0, 32)))
        checks["owner_mask"] = slot_preflight.owner_mask_check(experiment.adapter, device=experiment.device)
        checks["isolation"] = slot_preflight.isolation_check(experiment, _slice(smoke_eval, slice(0, 16)))
        checks["feature_consistency"] = slot_preflight.feature_check(experiment, _slice(smoke_train, slice(0, 16)))
        checks["bytes"] = slot_preflight.bytes_check(experiment, _slice(smoke_train, slice(0, 16)))
        checks["capacity"] = slot_preflight.capacity_check(experiment, _slice(smoke_train, slice(0, 16)))
    finally:
        experiment.close()
    d128 = load_parent("D128-3", model=model, tokenizer=tokenizer, expected=D128_3["digests"])
    try:
        identity["D128-3"] = checks["d128_3_replay"] = slot_preflight.native_replay(d128, "D128-3", PARENT_FOLDERS["D128-3"])
    finally:
        d128.close()
    experiment = load_parent("S-RE", model=model, tokenizer=tokenizer, expected=expected)  # fresh weights: the check trains
    try:
        frozen = slot_preflight.frozen_set_check(experiment, smoke_eval, _slice(smoke_train, slice(0, 32)))
        counts = {k: frozen["trainable_report"][k] for k in REGISTERED_COUNTS}
        frozen["registered_counts"], frozen["counts"] = REGISTERED_COUNTS, counts
        frozen["passed"] = frozen["passed"] and counts == REGISTERED_COUNTS
        checks["frozen_set"] = frozen
    finally:
        experiment.close()
    report = dict(version=1, status="complete", passed=all(c["passed"] for c in checks.values()), checks=checks,
                  required=["native_replay", "matrix_equals_native", "owner_mask", "isolation", "frozen_set", "bytes", "capacity",
                            "feature_consistency", "d128_3_replay"], config_hash=config["hash"], inventory=inv,
                  manifest_ids=manifest["ids"], source_blake2b_128=slot_data.source_digest(), load_seconds=load_seconds, **provenance())
    write_json(out / "preflight.json", report)
    write_json(out / "identity.json", identity)
    return report


# ---------------------------------------------------------------------------------------------------- S1A: arms, smoke

def arm_spec(arm: str, selected: dict) -> ReadSpec:
    """The arm's read from S0's calibrated values: Matrix-L1 reads with its g, SoftSlot-L1 with its g and tau."""
    if arm == "matrix_L1":
        return ReadSpec("matrix", selected["matrix"]["gain"])
    if arm == "soft_slot_L1":
        return ReadSpec("soft", selected["soft"]["gain"], tau=selected["soft"]["tau"])
    raise ValueError(f"unknown arm {arm}")


def load_arm(experiment_args: dict, ablation: bool = False):
    experiment = load_parent("S-RE", **experiment_args)
    if ablation:
        experiment.adapter.add_query_projection()
    return experiment


def parent_info(experiment, digests: dict) -> dict:
    return dict(name="S-RE", writer_blake2b_128=digests["writer.pt"], checkpoint_blake2b_128=digests["checkpoint.pt"],
                arm=experiment.read_arm)


def smoke_source(arrays_train):
    """The smoke's batches: the 64 smoke episodes as two 32-episode batches, cycled."""
    halves = [(_slice(arrays_train, slice(0, 32)), None), (_slice(arrays_train, slice(32, 64)), None)]
    step = 0
    while True:
        yield halves[step % 2]
        step += 1


def run_smoke(out, config_path, manifest_path, s0_metrics_path, *, lr: float = 1e-3, steps: int = 20) -> dict:
    """The S1A smoke per arm from the real S-RE: the freeze verification (i) to (iv) over `steps` optimizer steps, an export and
    its predictions on the 64 smoke episodes (the release's reload check compares against them)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = check_config(config_path, "S1A")
    lock = require_lock(manifest_path)
    digests = lock["inventory"]["S-RE"]
    selected = json.loads(Path(s0_metrics_path).read_text())["calibration"]["selected"]
    smoke = slot_data.eval_set("S1A", "smoke")
    arrays_eval, arrays_train = episodes(smoke), raw_episodes(smoke)
    report, model, tokenizer = {}, None, None
    for arm in slot_rules.ARMS:
        spec = arm_spec(arm, selected)
        experiment = load_arm(dict(model=model, tokenizer=tokenizer, expected={f: digests[f] for f in ("writer.pt", "checkpoint.pt")}))
        model, tokenizer = experiment.model, experiment.tokenizer
        try:
            summary = fit(experiment, spec, steps=steps, lr=lr, stream_seed=smoke.base_seed, out=out / arm, dev=arrays_eval,
                          dev_steps=(steps,), support=arrays_eval,
                          export=dict(arm=arm, parent=parent_info(experiment, digests), run=dict(stage="smoke", seed=None)),
                          source=smoke_source(arrays_train))
            samples = evaluate(experiment, arrays_eval, {"read": spec}, diagnostics=False)["samples"]["read"]
            np.savez_compressed(out / arm / "smoke_predictions.npz", **samples)
            report[arm] = dict(read=spec.as_dict(), selected_checkpoint_blake2b_128=summary["selected_checkpoint_blake2b_128"],
                               freeze=summary["freeze"], clipped_fraction=summary["clipped_fraction"], all_finite=summary["all_finite"],
                               dev=summary["dev"],
                               export_config=torch.load(out / arm / "selected.pt", map_location="cpu", weights_only=True)["slot"])
        finally:
            experiment.close()
    result = dict(version=1, stage="S1A-smoke", config_hash=config["hash"], lr=lr, steps=steps, arms=report, **provenance())
    write_json(out / "smoke.json", result)
    return result


def reload_check(export, predictions, arrays, *, release: dict, model=None, tokenizer=None) -> dict:
    """The export reloads (adapter, buffers, backend settings, scope) and a fresh evaluation reproduces the saved predictions."""
    experiment = load_export(export, model=model, tokenizer=tokenizer)
    try:
        arm = experiment.slot["arm"]
        config_matches = (release.get("smoke_configs", {}).get(arm) == experiment.slot
                          and release.get("arms", {}).get(arm, {}).get("read") == experiment.read_spec.as_dict())
        if not config_matches:
            raise StopRun("the reloaded export config differs from the release")
        samples = evaluate(experiment, arrays, {"read": experiment.read_spec}, diagnostics=False)["samples"]["read"]
        saved = np.load(predictions, allow_pickle=False)
        same = {k: bool(np.array_equal(samples[k], saved[k])) for k in ("episode_id", "prediction", "correct", "nll")}
        return dict(passed=all(same.values()), equal=same, read=experiment.read_spec.as_dict(), slot=dict(experiment.slot),
                    config_matches_release=config_matches, checkpoint_blake2b_128=file_digest(export))
    finally:
        experiment.close()


# ---------------------------------------------------------------------------------------------------------- S1A: pilot

def run_pilot(out, config_path, manifest_path, s0_metrics_path) -> dict:
    """The LR and timing pilot: six 200-step runs (3 LRs x 2 arms) from the S-RE state on the pilot stream, each evaluated once
    on the 1,000-episode pilot shard, plus step 0; picks the LR per arm and the step count N (weights are discarded)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = check_config(config_path, "S1A")
    lock = require_lock(manifest_path)
    digests = lock["inventory"]["S-RE"]
    selected = json.loads(Path(s0_metrics_path).read_text())["calibration"]["selected"]
    shard = slot_data.eval_set("S1A", "pilot")
    arrays = episodes(shard)
    ids = ids_record(shard, arrays)
    steps, stream_seed = slot_data.PILOT_STEPS, slot_data.TRAIN_BASE["S1A", "pilot"]
    runs, model, tokenizer, load_seconds = [], None, None, None
    for arm in slot_rules.ARMS:
        spec = arm_spec(arm, selected)
        for lr in slot_rules.LR_GRID:
            tick = time.perf_counter()
            experiment = load_arm(dict(model=model, tokenizer=tokenizer, expected={f: digests[f] for f in ("writer.pt", "checkpoint.pt")}))
            load_seconds = load_seconds or time.perf_counter() - tick  # the first load carries the backbone
            model, tokenizer = experiment.model, experiment.tokenizer
            try:
                summary = fit(experiment, spec, steps=steps, lr=lr, stream_seed=stream_seed, out=out / arm / f"lr{lr:g}", dev=arrays,
                              dev_steps=(steps,))
            except FloatingPointError as error:
                runs.append(dict(arm=arm, lr=lr, nll=None, all_finite=False, error=str(error)))
                continue
            finally:
                experiment.close()
            if not summary["all_finite"] or not math.isfinite(summary["dev"][1]["nll"]):
                runs.append(dict(arm=arm, lr=lr, nll=None, all_finite=False, error="non-finite candidate summary"))
                continue
            runs.append(dict(arm=arm, lr=lr, step0=summary["dev"][0], final=summary["dev"][1], nll=summary["dev"][1]["nll"],
                             step_seconds_mean=summary["step_seconds_mean"], clipped_fraction=summary["clipped_fraction"],
                             all_finite=summary["all_finite"], loss=summary["loss"]))
    choice, stop = {}, None
    for arm in slot_rules.ARMS:
        rows = [r for r in runs if r["arm"] == arm and r["all_finite"]]
        try:
            if not rows:
                raise slot_rules.StopForOrchestrator(f"{arm}: no finite LR candidate")
            choice[arm] = slot_rules.choose_lr(rows, rows[0]["step0"]["nll"])["lr"]
        except slot_rules.StopForOrchestrator as error:
            choice[arm], stop = None, str(error)
    finite = [r for r in runs if r["all_finite"]]
    step_seconds = max((r["step_seconds_mean"] for r in finite), default=None)
    eval_seconds = max((r[k]["seconds"] for r in finite for k in ("step0", "final")), default=None)
    result = dict(version=1, stage="S1A-pilot", config_hash=config["hash"], shard_ids_blake2b_128=ids, runs=runs, lr=choice,
                  step_seconds=step_seconds, eval_seconds=eval_seconds, load_seconds=load_seconds, stop=stop,
                  timing_candidates="finite completed candidates only", **provenance())
    if stop is None:
        try:
            result["steps"] = slot_rules.schedule_steps(step_seconds, eval_seconds, load_seconds)
            result["walltime_seconds"] = slot_rules.job_walltime_seconds(step_seconds, eval_seconds, load_seconds, result["steps"])
            if result["steps"] < slot_rules.N_MIN_OK:
                stop = f"N = {result['steps']} is below {slot_rules.N_MIN_OK}: a 2 h scale-up needs the orchestrator's OK"
        except slot_rules.StopForOrchestrator as error:
            stop = str(error)
    result["stop"] = stop
    write_json(out / "pilot.json", result)
    if stop is not None:
        raise StopRun(stop)
    return result


# ---------------------------------------------------------------------------------------------------- S1A: arm-seed run

def run_train(out, release_path, arm: str, seed: int, *, ablation: bool = False, config_path=None, manifest_path=None) -> dict:
    """One arm-seed run of the release: the matched continuation, dev selection, the freeze verification; `record.json` is the
    record the analysis reads."""
    out = Path(out)
    release = json.loads(Path(release_path).read_text())
    if release.get("status") != "ready":
        raise StopRun("the release has not passed its reload checks")
    config_hash = check_config(config_path, "S1A")["hash"]
    lock = require_lock(manifest_path)
    meta = provenance()
    if release.get("science_commit") != meta["git_commit"] or release.get("prereg_blake2b_128") != slot_data.PREREG_BLAKE2B_128:
        raise StopRun("the code or the registration differs from the release")
    digests = lock["inventory"]["S-RE"]
    if release["parent"]["writer_blake2b_128"] != digests["writer.pt"]:
        raise StopRun("the parent differs from the release")
    read = release["arms"][arm]["read"]
    steps, lr = release["steps"], release["arms"][arm]["lr"]
    science = slot_rules.science_config(arm, seed, steps=steps, lr=lr, read=read, parent_digest=digests["writer.pt"], ablation=ablation)
    key = f"{arm}/{seed}" + ("/ablation" if ablation else "")
    if release["config_hashes"].get(key) != slot_rules.science_hash(science):
        raise StopRun(f"the science config of {key} differs from the release")
    out.mkdir(parents=True, exist_ok=False)
    dev = slot_data.eval_set("S1A", "dev")
    dev_arrays, support = episodes(dev), episodes(slot_data.eval_set("S1A", "smoke"))
    dev_ids = ids_record(dev, dev_arrays)
    record = dict(version=1, status="running", arm=arm, seed=seed, steps=steps, ablation=ablation, config=science,
                  config_hash=slot_rules.science_hash(science), stage_config_hash=config_hash, dev_ids_blake2b_128=dev_ids,
                  release_blake2b_128=file_digest(release_path), confirmation_accessed=False, **meta)
    experiment = None
    try:
        experiment = load_arm(dict(expected={f: digests[f] for f in ("writer.pt", "checkpoint.pt")}), ablation)
        trainable = slot_rules.ABLATION_TRAINABLE if ablation else slot_rules.TRAINABLE
        summary = fit(experiment, ReadSpec.from_dict(read), steps=steps, lr=lr, stream_seed=slot_data.TRAIN_BASE["S1A", seed], out=out,
                      dev=dev_arrays, dev_steps=slot_rules.checkpoint_steps(steps), support=support, trainable=trainable,
                      export=dict(arm=arm, parent=parent_info(experiment, digests), run=dict(stage="S1A", seed=seed, ablation=ablation,
                                                                                            config_hash=record["config_hash"])),
                      time_limit=release.get("train_time_limit"))
        record.update(status="complete", dev=summary["dev"], selected_step=summary["selected_step"],
                      selected_checkpoint_blake2b_128=summary["selected_checkpoint_blake2b_128"],
                      training=dict(all_finite=summary["all_finite"], clipped_fraction=summary["clipped_fraction"],
                                    clipped_steps=summary["clipped_steps"], grad_norm_max=summary["grad_norm"]["max"],
                                    grad_norm=summary["grad_norm"], loss=summary["loss"], step_seconds_mean=summary["step_seconds_mean"],
                                    train_seconds=summary["train_seconds"]),
                      freeze=summary["freeze"], trainable_parameters=summary["trainable_parameters"],
                      frozen_parameters=summary["frozen_parameters"])
    except BaseException as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        if experiment is not None:
            experiment.close()
        write_json(out / "record.json", record)
    return record


# ------------------------------------------------------------------------------------------- S1A: guarded evaluation

def evaluate_entries(set_name: tuple[str, str], entries: list[dict], out, *, token=None, model=None, tokenizer=None) -> dict:
    """Evaluate each entry on one registered set and save its per-sample arrays as `<label>.npz`; prints and returns counts and
    digests only. An entry is {label, export, checkpoint_blake2b_128} (a saved arm checkpoint) or {label,
    parent_writer_blake2b_128, read} (the unmodified S-RE through a read; `read` None = its native read). A sealed set opens
    only for a checkpoint the token names."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report, arrays = {}, None  # model and tokenizer: given, or loaded with the first entry and shared by the rest
    for entry in entries:
        digest = entry.get("checkpoint_blake2b_128") or entry.get("parent_writer_blake2b_128")
        eval_set = slot_data.eval_set(*set_name, token=token, checkpoint_digest=digest)
        if arrays is None:
            arrays = episodes(eval_set)
            ids_record(eval_set, arrays)
        if "export" in entry:
            if file_digest(entry["export"]) != entry["checkpoint_blake2b_128"]:
                raise StopRun(f"{entry['label']}: the checkpoint differs from its recorded digest")
            experiment = load_export(entry["export"], model=model, tokenizer=tokenizer)
            spec = experiment.read_spec
        else:
            experiment = load_parent("S-RE", model=model, tokenizer=tokenizer, expected={"writer.pt": digest})
            spec = None if entry["read"] is None else ReadSpec.from_dict(entry["read"])
        model, tokenizer = experiment.model, experiment.tokenizer
        try:
            samples = evaluate(experiment, arrays, {"read": spec}, diagnostics=True)["samples"]["read"]
        finally:
            experiment.close()
        keep = {k: v for k, v in samples.items() if k in {"episode_id", "answer", "prediction", "correct", "nll", "rel_err"}}
        np.savez_compressed(out / f"{entry['label']}.npz", **keep)
        report[entry["label"]] = dict(n=int(len(samples["correct"])), file_blake2b_128=file_digest(out / f"{entry['label']}.npz"))
        print(json.dumps({entry["label"]: report[entry["label"]]}), flush=True)
    return report


# ------------------------------------------------------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--s0-metrics", type=Path, help="S0 metrics.json: its calibration.selected gives each arm's g and tau")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--arm", choices=slot_rules.ARMS)
    parser.add_argument("--seed", type=int, choices=slot_data.RUN_SEEDS)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--set", dest="set_name", choices=("dev", "confirmation", "unseen"))
    parser.add_argument("--entries", type=Path, help="JSON list of evaluation entries (see evaluate_entries)")
    parser.add_argument("--open-root", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.open_root is not None and not args.open_root.is_absolute():
        parser.error("--open-root must be absolute")
    if args.mode == "evaluate" and args.set_name in {"confirmation", "unseen"} and args.open_root is None:
        parser.error("sealed evaluation requires an absolute --open-root")
    if not args.out.is_absolute() or RUNS not in (args.out, *args.out.parents):
        raise SystemExit(f"--out must be an absolute path under {RUNS}")
    try:
        if args.mode == "preflight":
            report = run_preflight(args.out, args.config)
            print(json.dumps({k: v["passed"] for k, v in report["checks"].items()}), flush=True)
            return 0 if report["passed"] else 2
        if args.mode == "s0":
            run_s0(args.out, args.config, args.manifest)
        elif args.mode == "smoke":
            run_smoke(args.out, args.config, args.manifest, args.s0_metrics)
        elif args.mode == "pilot":
            run_pilot(args.out, args.config, args.manifest, args.s0_metrics)
        elif args.mode == "train":
            run_train(args.out, args.release, args.arm, args.seed, ablation=args.ablation, config_path=args.config,
                      manifest_path=args.manifest)
        elif args.mode == "evaluate":
            lock = require_lock(args.manifest)
            token = slot_data.load_open_token(args.open_root, lock["inventory"]) if args.set_name in {"confirmation", "unseen"} else None
            evaluate_entries(("S1A", args.set_name), json.loads(args.entries.read_text()), args.out, token=token)
        else:
            require_lock(args.manifest)
            check = reload_check(args.export, args.predictions, episodes(slot_data.eval_set("S1A", "smoke")),
                                 release=json.loads(args.release.read_text()))
            write_json(args.out, check)
            return 0 if check["passed"] else 2
    except (StopRun, slot_rules.StopForOrchestrator) as stop:
        print(f"STOP: {stop}", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
