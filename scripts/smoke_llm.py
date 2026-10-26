#!/usr/bin/env python3
"""Run and audit the authorized P1-L smoke on compute through submit.py.

Use persistent absolute --runs/--reports paths for commit archives. Audit-only
checks existing artifacts and never trains or evaluates another episode.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from looplora.llm.evaluate import aggregate_samples
from looplora.llm.run import compare_suite, load_suite, run_suite
from looplora.sym.stats import load_samples


def audit(suite_path: Path, root: Path, reports: Path) -> dict:
    suite = json.loads((root / "suite.json").read_text())
    runs = Path(suite.get("run_root", root))
    jobs = load_suite(suite_path, runs)
    if suite["status"] != "complete" or set(suite["runs"]) != {job["name"] for job in jobs}:
        raise AssertionError("smoke matrix is incomplete")
    records, timing_rows = [], []
    for job in jobs:
        name, config = job["name"], job["config"]
        output = runs / name
        required = ("config.json", "train_log.jsonl", "metrics.json", "samples.npz",
                    "checkpoint.pt", "timing.json", "record.json")
        if any(not (output / filename).is_file() for filename in required):
            raise AssertionError(f"missing artifacts: {name}")
        record = json.loads((output / "record.json").read_text())
        if record["status"] != "complete" or suite["runs"][name]["status"] != "complete":
            raise AssertionError(f"incomplete run: {name}")
        samples = load_samples(output / "samples.npz")
        metrics = json.loads((output / "metrics.json").read_text())
        if not config["smoke"] or set(samples["split"]) != {"val"}:
            raise AssertionError(f"non-smoke data in {name}")
        if aggregate_samples(samples) != metrics["table"]:
            raise AssertionError(f"sample/metric mismatch: {name}")
        if not np.array_equal(samples["correct"], samples["prediction"] == samples["answer"]):
            raise AssertionError(f"primary metric is not full-vocabulary EM: {name}")
        hashes = {}
        for row in metrics["memory_hashes"]:
            key = row["split"], row["protocol"], row["condition"], row["batch_start"]
            value = {field: row[field] for field in
                     ("x", "u", "memory", "write_count", "attention_mask", "backbone_kv")}
            if key in hashes and hashes[key] != value:
                raise AssertionError(f"depth-dependent input/state in {name}: {key}")
            hashes[key] = value
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        expected_steps = 0 if config["stage"] == "baseline" else config["train"]["steps"]
        if checkpoint["summary"]["steps"] != expected_steps or checkpoint["config"]["seed"] != config["seed"]:
            raise AssertionError(f"training budget or seed mismatch: {name}")
        if config["stage"] == "reader":
            writer = torch.load(config["writer_checkpoint"], map_location="cpu", weights_only=True)["writer"]
            if checkpoint["writer"].keys() != writer.keys() or any(
                not torch.equal(checkpoint["writer"][key], tensor) for key, tensor in writer.items()
            ):
                raise AssertionError(f"frozen writer changed: {name}")
        timing = json.loads((output / "timing.json").read_text())
        records.append(dict(run=name, stage=config["stage"], variant=config["variant"],
                            samples=len(samples["answer"]), git_commit=record["git_commit"],
                            runtime=record["runtime"]))
        for evaluation in timing["eval_sets"]:
            timing_rows.append(dict(run=name, stage=config["stage"], variant=config["variant"],
                train_steps=checkpoint["summary"]["steps"], train_seconds=record["train_seconds"],
                seconds_per_1000_steps=timing["seconds_per_1000_steps"],
                eval_split=evaluation["split"], eval_protocol=evaluation["protocol"],
                eval_episodes=evaluation["episodes"], eval_queries=evaluation["queries"],
                eval_seconds=evaluation["seconds"], total_seconds=timing["total_seconds"],
                peak_memory_bytes=record["peak_memory_bytes"]))
    comparisons = compare_suite(jobs, runs)
    saved = json.loads((runs / "comparisons.json").read_text())
    if comparisons != saved:
        raise AssertionError("paired suite comparisons do not reproduce")
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "p1l_smoke_timing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(timing_rows)
    result = dict(runs=len(jobs), run_root=str(runs), suite_seconds=suite["seconds"],
                  total_node_hours=suite.get("total_node_hours", suite["node_hours"]),
                  procs=suite["procs"], all_sample_tables_reproduced=True,
                  frozen_writers_bitwise_unchanged=True, gate0_passed=True,
                  shared_writer_groups=comparisons["frozen_writer_hashes_verified"], records=records,
                  scope="smoke only; no pilot, screening, confirmation or method selection")
    (reports / "p1l_smoke_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/p1l/smoke.yaml"))
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--procs", choices=("1", "2", "auto"), default="auto")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        code = run_suite(args.suite, args.runs, args.procs)
        if code:
            return code
    result = audit(args.suite, args.runs, args.reports)
    print(json.dumps({key: result[key] for key in ("runs", "suite_seconds", "gate0_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
