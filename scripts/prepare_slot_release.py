#!/usr/bin/env python3
"""Build the SLOT-LOOP-v1 S1A release without importing torch.

First build a candidate (--out CANDIDATE.json, status pending_reload). Run each
arm's smoke reload check on compute with --release CANDIDATE.json. Then build
the final release with the same inputs and --reload-check ARM=REPORT.json for
both arms; its status is ready. Both outputs must be new paths. Only a ready
release is eligible for the orchestrator's approval; this script never grants it.

Example (use .venv/bin/python or uv run python):
  .venv/bin/python scripts/prepare_slot_release.py --manifest MANIFEST.lock.json \
      --pilot PILOT/pilot.json --s0-metrics S0/metrics.json --smoke SMOKE/smoke.json \
      --science-commit FULL_40_HEX --out CANDIDATE.json
Repeat with --reload-check matrix_L1=MATRIX.json --reload-check
soft_slot_L1=SOFT.json --out s1a-release.json after the compute reload checks.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from looplora.infra.seeds import stable_hash  # noqa: E402
from looplora.llm import slot_data, slot_rules  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_set(lock, name, n, ids):
    registered = lock["manifest"]["ids"][f"eval/{name}"]
    require(n == registered["count"] and ids == registered["ids_blake2b_128"],
            f"{name}: count or episode ids differ from the locked manifest")


def selected_reads(s0, lock):
    require(s0.get("stage") == "S0", "not an S0 metrics record")
    require(s0.get("config_hash") == lock["config_hashes"]["S0"], "S0 stage config differs from the lock")
    check_set(lock, "S0/dev", s0["n"], s0["ids_blake2b_128"])
    calibration = s0["calibration"]
    check_set(lock, "S0/calibration", calibration["n"], calibration["ids_blake2b_128"])
    grid = [{k: row[k] for k in ("kind", "gain", "tau")} for row in calibration["candidates"]]
    require(sorted(map(stable_hash, grid)) == sorted(map(stable_hash, slot_rules.calibration_grid())),
            "calibration candidates differ from the registered grid")
    for kind in ("matrix", "soft", "hard"):
        rows = [dict(row, nll=row["nll"] if finite(row.get("nll")) else math.inf)
                for row in calibration["candidates"]]
        choice = slot_rules.choose_candidate(rows, kind)
        expected = {k: choice[k] for k in ("gain", "tau") if choice[k] is not None}
        require(calibration["selected"][kind] == expected, f"{kind}: calibration selection differs from the rule")
    return {arm: dict(kind=kind, gain=calibration["selected"][kind]["gain"],
                      tau=calibration["selected"][kind].get("tau"), scope="all_valid_question", recency=False, capacity=16)
            for arm, kind in slot_rules.ARM_KIND.items()}


def check_pilot(pilot, lock):
    require(pilot.get("stage") == "S1A-pilot" and pilot.get("stop") is None, "the pilot is incomplete or ended in STOP")
    require(pilot.get("config_hash") == lock["config_hashes"]["S1A"], "pilot stage config differs from the lock")
    check_set(lock, "S1A/pilot", lock["manifest"]["sets"]["S1A/pilot"]["count"], pilot["shard_ids_blake2b_128"])
    runs = pilot["runs"]
    require(len(runs) == 6 and {(r["arm"], r["lr"]) for r in runs} == {
        (a, lr) for a in slot_rules.ARMS for lr in slot_rules.LR_GRID}, "pilot must record all six registered candidates")
    for arm in slot_rules.ARMS:
        rows = [r for r in runs if r["arm"] == arm]
        require(all(r.get("all_finite") is True or (r.get("all_finite") is False and r.get("nll") is None
                    and isinstance(r.get("error"), str) and r["error"]) for r in rows),
                f"{arm}: an excluded pilot candidate needs its non-finite failure record")
        completed = [r for r in rows if r.get("all_finite") is True]
        starts = [r["step0"]["nll"] for r in completed]
        require(starts and all(finite(nll) for nll in starts) and len(set(starts)) == 1, f"{arm}: inconsistent pilot step zero")
        for row in completed:
            require(row["step0"]["step"] == 0 and row["step0"]["n"] == slot_data.SETS["S1A", "pilot"].count,
                    f"{arm}: pilot step zero differs from registration")
            require(row["final"]["step"] == slot_data.PILOT_STEPS and
                    row["final"]["n"] == slot_data.SETS["S1A", "pilot"].count and
                    finite(row.get("nll")) and row["nll"] == row["final"]["nll"], "pilot candidate is incomplete")
        choices = [dict(row, nll=row["nll"] if row.get("all_finite") is True and finite(row.get("nll")) else math.inf)
                   for row in rows]
        require(pilot["lr"][arm] == slot_rules.choose_lr(choices, starts[0])["lr"], f"{arm}: LR selection differs from the rule")
    completed = [r for r in runs if r.get("all_finite") is True]
    measured_steps = [r["step_seconds_mean"] for r in completed if finite(r.get("step_seconds_mean"))]
    measured_evals = [r[key]["seconds"] for r in completed for key in ("step0", "final")
                      if isinstance(r.get(key), dict) and finite(r[key].get("seconds"))]
    require(measured_steps and measured_evals and pilot["step_seconds"] == max(measured_steps)
            and pilot["eval_seconds"] == max(measured_evals), "pilot timing summary differs from the candidate measurements")
    steps = slot_rules.schedule_steps(pilot["step_seconds"], pilot["eval_seconds"], pilot["load_seconds"])
    require(steps >= slot_rules.N_MIN_OK and pilot["steps"] == steps, "pilot N differs from the registered rule or requires STOP")
    walltime = slot_rules.job_walltime_seconds(pilot["step_seconds"], pilot["eval_seconds"], pilot["load_seconds"], steps)
    require(pilot["walltime_seconds"] == walltime, "pilot walltime differs from the registered rule")
    return steps, walltime


def check_smoke(smoke, reads, lock):
    require(smoke.get("stage") == "S1A-smoke" and smoke.get("config_hash") == lock["config_hashes"]["S1A"],
            "smoke stage config differs from the lock")
    require(smoke.get("steps") == 20 and smoke.get("lr") == 1e-3, "release requires the registered 20-step smoke")
    require(set(smoke["arms"]) == set(slot_rules.ARMS), "smoke must verify both arms")
    configs = {}
    for arm, row in smoke["arms"].items():
        require(row.get("all_finite") is True and row["read"] == reads[arm], f"{arm}: smoke read or finiteness differs")
        require(digest(row.get("selected_checkpoint_blake2b_128")), f"{arm}: no smoke checkpoint digest")
        freeze = row["freeze"]
        trainable = freeze["trainable"]
        require(trainable["trainable"] == list(slot_rules.TRAINABLE) and
                set(trainable["frozen"]) == set(slot_rules.FROZEN) and
                set(trainable["buffers"]) == set(slot_rules.BUFFERS) and trainable["trainable_parameters"] == 180224
                and trainable["frozen_parameters"] == 294914, f"{arm}: incorrect frozen or trainable set")
        require(freeze["digests"].get("frozen_unchanged") is True and freeze["digests"].get("trainable_changed") is True
                and freeze["support"].get("equal") is True and freeze["support"]["before"] == freeze["support"]["after"],
                f"{arm}: freeze or support check failed")
        before, after = freeze["digests"]["before"], freeze["digests"]["after"]
        names = {*slot_rules.FROZEN, *slot_rules.BUFFERS, *slot_rules.TRAINABLE}
        require(set(before) == set(after) == names and all(digest(v) for v in [*before.values(), *after.values()])
                and all(before[n] == after[n] for n in (*slot_rules.FROZEN, *slot_rules.BUFFERS))
                and all(before[n] != after[n] for n in slot_rules.TRAINABLE), f"{arm}: freeze digest evidence differs")
        gradients = freeze["gradients"]["trainable_grad_norms"]
        require(set(gradients) == set(slot_rules.TRAINABLE) and all(finite(v) and v > 0 for v in gradients.values()),
                f"{arm}: invalid trainable gradients")
        config = row["export_config"]
        required = {"arm", "read", "parent", "run", "trainable", "query_projection", "loop_steps", "key_dim", "value_dim",
                    "feature_map", "eps", "layout_digest", "task", "source_commit", "prereg_blake2b_128"}
        require(required <= set(config), f"{arm}: smoke export metadata is incomplete")
        require(config["arm"] == arm and config["read"] == reads[arm] and config["loop_steps"] == 1
                and config["query_projection"] == "shared" and config["trainable"] == list(slot_rules.TRAINABLE)
                and config["run"] == dict(stage="smoke", seed=None) and config["key_dim"] == 768 and config["value_dim"] == 128
                and config["feature_map"] == "dpfp-nu3-unit" and finite(config["eps"]) and config["eps"] > 0
                and re.fullmatch(r"[0-9a-f]{40}", config["source_commit"]) is not None
                and config["layout_digest"] == slot_data.LAYOUT_DIGEST and config["task"] == json.loads(json.dumps(slot_data.TASK))
                and config["prereg_blake2b_128"] == slot_data.PREREG_BLAKE2B_128
                and config["parent"]["writer_blake2b_128"] == lock["inventory"]["S-RE"]["writer.pt"]
                and config["parent"]["checkpoint_blake2b_128"] == lock["inventory"]["S-RE"]["checkpoint.pt"],
                f"{arm}: smoke export settings differ from the release")
        configs[arm] = config
    return configs


def prepare(manifest, pilot, s0_metrics, smoke, *, science_commit, out=None, reload_checks=None):
    """Self-check the measured inputs and write a new candidate or final release."""
    require(out is None or not Path(out).exists(), "a release is never overwritten")
    require(isinstance(science_commit, str) and re.fullmatch(r"[0-9a-f]{40}", science_commit), "science_commit must be full 40-hex")
    lock, pilot_row, s0, smoke_row = map(read_json, (manifest, pilot, s0_metrics, smoke))
    require(lock.get("status") == "locked" and lock.get("preflight", {}).get("passed") is True,
            "a passed preflight and locked manifest are required")
    require(lock.get("prereg_blake2b_128") == slot_data.PREREG_BLAKE2B_128 == slot_data.file_digest(ROOT / slot_data.PREREG),
            "the prereg differs from the lock or registered file")
    for stage in ("S0", "S1A"):
        require(lock["config_hashes"][stage] == stable_hash(slot_rules.stage_config(stage)), "locked stage config differs")
    require(all(digest(lock["inventory"]["S-RE"].get(name)) for name in ("writer.pt", "checkpoint.pt")),
            "the locked parent inventory is incomplete")
    reads = selected_reads(s0, lock)
    steps, walltime = check_pilot(pilot_row, lock)
    configs = check_smoke(smoke_row, reads, lock)
    arms = {a: dict(lr=pilot_row["lr"][a], read=reads[a]) for a in slot_rules.ARMS}
    parent = lock["inventory"]["S-RE"]
    science = {f"{arm}/{seed}" + ("/ablation" if ablation else ""): slot_rules.science_config(
        arm, seed, steps=steps, lr=arms[arm]["lr"], read=reads[arm], parent_digest=parent["writer.pt"], ablation=ablation)
        for ablation in (False, True) for arm in slot_rules.ARMS for seed in ((17,) if ablation else slot_data.RUN_SEEDS)}
    reports = {} if reload_checks is None else {arm: read_json(path) for arm, path in reload_checks.items()}
    if reports:
        require(set(reports) == set(slot_rules.ARMS), "reload checks must name both arms")
        for arm, report in reports.items():
            require(report.get("passed") is True and report.get("config_matches_release") is True
                    and report.get("slot") == configs[arm] and report.get("read") == reads[arm]
                    and report.get("checkpoint_blake2b_128") == smoke_row["arms"][arm]["selected_checkpoint_blake2b_128"]
                    and set(report.get("equal", {})) == {"episode_id", "prediction", "correct", "nll"}
                    and all(v is True for v in report["equal"].values()), f"{arm}: reload check failed or differs from the release")
    release = dict(version=1, study=slot_data.STUDY, stage="S1A", status="ready" if reports else "pending_reload",
                   science_commit=science_commit, steps=steps, walltime_seconds=walltime, arms=arms,
                   parent=dict(name="S-RE", writer_blake2b_128=parent["writer.pt"], checkpoint_blake2b_128=parent["checkpoint.pt"]),
                   science_configs=science, config_hashes={key: slot_rules.science_hash(value) for key, value in science.items()},
                   prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128,
                   analysis_script_blake2b_128=slot_data.file_digest(ROOT / slot_data.ANALYSIS_SCRIPT),
                   manifest_blake2b_128=slot_data.file_digest(manifest), inventory_blake2b_128=stable_hash(lock["inventory"]),
                   inputs_blake2b_128={name: slot_data.file_digest(path) for name, path in
                                      (("pilot", pilot), ("s0_metrics", s0_metrics), ("smoke", smoke))},
                   timings={k: pilot_row[k] for k in ("step_seconds", "eval_seconds", "load_seconds")},
                   freeze={arm: smoke_row["arms"][arm]["freeze"] for arm in slot_rules.ARMS},
                   smoke_configs=configs, reload_checks=reports)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with Path(out).open("x") as handle:
            handle.write(json.dumps(release, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("manifest", "pilot", "s0-metrics", "smoke", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--science-commit", required=True)
    parser.add_argument("--reload-check", action="append", default=[], metavar="ARM=PATH")
    args = parser.parse_args(argv)
    checks = {}
    for value in args.reload_check:
        arm, separator, path = value.partition("=")
        if not separator or arm not in slot_rules.ARMS or arm in checks or not path:
            parser.error("--reload-check must name each registered arm once as ARM=PATH")
        checks[arm] = Path(path)
    result = prepare(args.manifest, args.pilot, args.s0_metrics, args.smoke, science_commit=args.science_commit,
                     out=args.out, reload_checks=checks)
    print(json.dumps(dict(status=result["status"], steps=result["steps"], walltime_seconds=result["walltime_seconds"],
                          config_hashes=result["config_hashes"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
