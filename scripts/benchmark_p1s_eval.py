#!/usr/bin/env python3
"""Time selected P1-S checkpoints on 2000 validation episodes, on compute only.

Submit via scripts/submit.py. --out retains large evaluation artifacts;
--report contains only timing, provenance and integrity checks. This does not
train, select learning rates, or evaluate screening/confirmation episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

WRITERS = ("oracle_orth", "t1_frozen")
DEPTHS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
HOPS = list(range(1, 9))
OVERRIDES = {"eval.episodes": 2000, "eval.splits": ["val"], "eval.seed": 4242,
             "eval.counterfactuals": False}
HASH_FIELDS = ("x", "u", "memory", "write_count")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"expected JSON mapping: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _check_artifacts(out: Path, checkpoint: Path, original: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from looplora.infra.seeds import stable_hash
    from looplora.sym.evaluate import aggregate_samples, resolve_evaluation_config
    from looplora.sym.stats import load_samples

    metrics, timing, provenance = (_read(out / name) for name in ("metrics.json", "timing.json", "config.json"))
    expected = resolve_evaluation_config(original, OVERRIDES)
    _require(metrics["checkpoint"] == str(checkpoint), "checkpoint reference differs")
    _require(metrics["checkpoint_config"] == original, "original checkpoint config changed")
    _require(metrics["config"] == expected, "resolved evaluation config differs from approved overrides")
    _require(metrics["eval_overrides"] == OVERRIDES, "recorded overrides differ")
    _require(metrics["config_hash"] == stable_hash(expected), "resolved config hash differs")
    for field in ("checkpoint", "checkpoint_config", "config", "config_hash", "eval_overrides"):
        _require(provenance[field] == metrics[field], f"config/metrics provenance differs: {field}")
    _require(expected["seed"] == 17 and expected["eval"]["seed"] == metrics["evaluation_seed"] == 4242,
             "forbidden training/evaluation seed")
    _require(expected["eval"]["splits"] == ["val"] and not expected["eval"]["counterfactuals"],
             "benchmark must contain original validation episodes only")
    _require(expected["eval"]["episodes"] == 2000 and expected["eval"]["hops"] == HOPS
             and expected["eval"]["depths"] == DEPTHS, "wrong benchmark dimensions")
    samples = load_samples(out / "samples.npz")
    _require(all(value.ndim == 1 and len(value) == 144000 for value in samples.values()),
             "sample arrays must have exactly 2000*8*9 aligned rows")
    _require(set(samples["split"]) == {"val"} and set(samples["condition"]) == {"original"},
             "forbidden sample split/condition")
    _require(np.array_equal(samples["correct"], samples["prediction"] == samples["answer"]),
             "saved correctness differs from predictions")
    _require(np.array_equal(samples["full_correct"], samples["full_prediction"] == samples["answer"]),
             "saved full-vocabulary correctness differs from predictions")
    _require(bool(np.all(samples["candidate_size"] > 0)), "empty sample candidate set")
    rebuilt = aggregate_samples(samples)
    _require(rebuilt == metrics["table"], "samples do not reproduce metric table")
    _require(len(rebuilt) == 72 and all(row["n"] == 2000 for row in rebuilt),
             "benchmark requires 72 complete hop/depth cells of n=2000")
    _require({(row["hop"], row["L"]) for row in rebuilt}
             == {(hop, depth) for hop in HOPS for depth in DEPTHS}, "missing hop/depth cells")
    fingerprint = None
    for depth in DEPTHS:
        selected = np.flatnonzero(samples["L"] == depth)
        _require(len(selected) == 16000 and len(np.unique(samples["episode_id"][selected])) == 2000,
                 f"wrong episode/query count at L={depth}")
        query = samples["query_id"][selected]
        _require(bool(np.all((query >= 0) & (query < 8))), "query id outside 0..7")
        _require(np.array_equal(samples["hop"][selected], np.asarray(HOPS)[query]),
                 "one-query-per-hop protocol differs")
        ordered = selected[np.lexsort((query, samples["episode_id"][selected]))]
        # load_samples rejects duplicate (episode,query,L) keys; the counts above
        # and bounded query ids consequently require all eight queries per episode.
        digest = hashlib.blake2b(digest_size=16)
        for field in ("episode_id", "query_id", "hop", "answer", "candidate_size"):
            value = np.ascontiguousarray(samples[field][ordered])
            digest.update(field.encode() + str(value.dtype).encode() + str(value.shape).encode())
            digest.update(value.tobytes())
        current = digest.hexdigest()
        _require(fingerprint is None or fingerprint == current, "episode/query labels changed across depths")
        fingerprint = current
    _require(metrics.get("gate0_passed") is True, "saved Gate 0 flag failed")
    batches = set(range(0, 2000, expected["eval"]["batch_size"]))
    snapshots, seen = {}, set()
    for row in metrics["memory_hashes"]:
        _require(row["split"] == "val" and row["condition"] == "original", "forbidden memory hash condition")
        key = (row["batch_start"], row["L"])
        _require(key not in seen and key[0] in batches and key[1] in DEPTHS,
                 "duplicate/unexpected batch-depth memory hash")
        seen.add(key)
        hashes = tuple(row[field] for field in HASH_FIELDS)
        _require(all(isinstance(value, str) and len(value) == 32 for value in hashes), "invalid memory hashes")
        _require(key[0] not in snapshots or snapshots[key[0]] == hashes, "depth-dependent x/u/M/write_count")
        snapshots[key[0]] = hashes
    _require(seen == {(batch, depth) for batch in batches for depth in DEPTHS}, "missing depth snapshot hashes")
    runtime = timing["runtime"]
    _require(runtime["device"] == expected["device"] and runtime["threads"] == expected["runtime"]["threads"],
             "runtime device/thread config differs")
    _require(set(timing["reader_seconds"]) == {str(depth) for depth in DEPTHS}, "missing reader timing depths")
    times = [timing["evaluation_seconds"], timing["writer_seconds"], *timing["reader_seconds"].values()]
    _require(all(isinstance(value, (int, float)) and np.isfinite(value) and value >= 0 for value in times),
             "invalid evaluation timings")
    return {"runtime": runtime, "measurement": timing["measurement"],
            "evaluation_seconds": timing["evaluation_seconds"], "writer_seconds": timing["writer_seconds"],
            "reader_seconds": timing["reader_seconds"], "rows": 144000, "episodes": 2000,
            "queries_per_episode": 8, "depths": 9, "metric_cells": 72,
            "data_fingerprint": fingerprint, "effective_config_hash": metrics["config_hash"],
            "eval_overrides": OVERRIDES, "integrity": {
                "validation_only": True, "seed17_eval4242": True, "approved_overrides_only": True,
                "checkpoint_config_preserved": True, "samples_reproduce_metrics": True,
                "complete_2000_by_8_by_9": True, "same_query_labels_across_depths": True,
                "fixed_x_u_memory_write_count_across_depths": True}}


def benchmark(selection_path: Path, out: Path, report: Path) -> dict[str, Any]:
    _require(bool(os.environ.get("PBS_JOBID")), "run this benchmark inside a PBS compute allocation")
    import torch

    selection_path, out, report = selection_path.resolve(), out.resolve(), report.resolve()
    selection = _read(selection_path)
    out.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "scope": "timing only; validation only; no training, LR selection, "
              "scientific contrasts, screening/confirmation evaluation, or gate decisions",
              "selection2": str(selection_path), "artifacts_root": str(out),
              "job_id": os.environ.get("PBS_JOBID"), "runs": [], "failure_reason": None}
    _write(report, result)
    try:
        for writer in WRITERS:
            selected = selection["selected"][writer + "/M1"]
            _require(selected["writer"] == writer and selected["variant"] == "M1", "selected row identity differs")
            checkpoint = (Path(selected["run_dir"]) / "checkpoint.pt").resolve()
            _require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
            original = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
            _require(original["seed"] == 17 and original["writer"] == writer and original["variant"] == "M1",
                     "checkpoint identity/training seed differs")
            _require(original["eval"]["splits"] == ["val"] and original["eval"]["seed"] == 4242,
                     "selected checkpoint is not a validation-only pilot")
            _require(original["task"]["name"] == "t2_chain_end"
                     and original["eval"]["hops"] == HOPS and original["eval"]["depths"] == DEPTHS,
                     "selected checkpoint is not the full Pilot-2 M1 depth protocol")
            destination, log = out / writer, out / (writer + ".log")
            _require(not destination.exists() and not log.exists(), f"benchmark artifacts already exist: {destination}")
            command = [sys.executable, "-m", "looplora.sym.evaluate", "--checkpoint", str(checkpoint),
                       "--out", str(destination)]
            for key, value in OVERRIDES.items():
                command.extend(["--set", key + "=" + json.dumps(value)])
            with log.open("w") as stream:
                started = time.perf_counter()
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                process_seconds = time.perf_counter() - started
            _require(completed.returncode == 0, f"evaluation exited {completed.returncode}; see {log}")
            checked = _check_artifacts(destination, checkpoint, original)
            result["runs"].append({"writer": writer, "variant": "M1", "training_seed": 17,
                                   "selected_name": selected["name"], "checkpoint": str(checkpoint),
                                   "artifacts": str(destination), "log": str(log), "command": command,
                                   "process_seconds": process_seconds, **checked})
            _write(report, result)
        result["status"] = "complete"
        result["process_seconds_total"] = sum(run["process_seconds"] for run in result["runs"])
        _write(report, result)
        return result
    except BaseException as error:
        result.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        _write(report, result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.selection2, args.out, args.report)


if __name__ == "__main__":
    main()
