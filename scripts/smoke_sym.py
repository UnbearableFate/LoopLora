#!/usr/bin/env python3
"""Run the WP-D smoke once, then audit artifacts and generate paired reports.

Submit through scripts/submit.py. --runs and --reports must point to persistent
/work paths when source is a commit archive. --audit-only never retrains.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from looplora.sym.evaluate import aggregate_samples
from looplora.sym.figures import create_figures
from looplora.sym.run import load_suite, run_suite
from looplora.sym.stats import compare_runs, load_samples


def _snapshot_hashes(metrics: dict) -> dict:
    hashes = {}
    for row in metrics["memory_hashes"]:
        key = row["split"], row["condition"], row["batch_start"]
        value = {field: row[field] for field in ("x", "u", "memory", "write_count")}
        if key in hashes and value != hashes[key]:
            raise AssertionError(f"depth-dependent input/state: {key}")
        hashes[key] = value
    return hashes


def audit(suite_path: Path, runs: Path, reports: Path) -> dict:
    expected = load_suite(suite_path, runs)
    suite = json.loads((runs / "suite.json").read_text())
    if set(suite["runs"]) != {job["name"] for job in expected}:
        raise AssertionError("smoke matrix is incomplete")
    samples, metrics, records, timing_rows = {}, {}, {}, []
    for job in expected:
        name = job["name"]
        output = runs / name
        record = json.loads((output / "record.json").read_text())
        if record["status"] != "complete" or suite["runs"][name]["status"] != "complete":
            raise AssertionError(f"incomplete run: {name}")
        samples[name] = load_samples(output / "samples.npz")
        metrics[name] = json.loads((output / "metrics.json").read_text())
        if aggregate_samples(samples[name]) != metrics[name]["table"]:
            raise AssertionError(f"sample/metric mismatch: {name}")
        _snapshot_hashes(metrics[name])
        records[name] = record
        timing = json.loads((output / "timing.json").read_text())
        timing_rows.append({"run": name, "train_seconds": timing["train_seconds"],
                            "pretrain_seconds": timing.get("pretrain_seconds", 0),
                            "train_step_seconds": timing["train_step_seconds"],
                            "writer_seconds": timing["writer_seconds"],
                            "reader_seconds": sum(timing["reader_seconds"].values()),
                            "total_seconds": timing["total_seconds"],
                            "mean_L": record["mean_L"], "peak_memory_bytes": record["peak_memory_bytes"]})
    comparisons, shared_groups = [], []
    for prefix in ("t1", "t2", "t2-frozen"):
        anchor = prefix + "-M1"
        if anchor not in samples:
            continue
        members = [prefix + "-" + variant for variant in ("B3", "B4", "B5", "M1-lin")]
        expected_hashes = _snapshot_hashes(metrics[anchor])
        for member in members:
            if member not in metrics or _snapshot_hashes(metrics[member]) != expected_hashes:
                raise AssertionError(f"writer/input state differs between {anchor} and {member}")
        shared_groups.append([anchor, *members])
        for variant, depth in (("B3", 4), ("B4", 1), ("B5", 4)):
            result = compare_runs(samples[anchor], samples[prefix + "-" + variant], 4, depth,
                                  label_a="M1", label_b=variant)
            comparisons.extend({"group": prefix, **row} for row in result)
        comparisons.extend({"group": prefix, **row} for row in compare_runs(
            samples[prefix + "-M1-lin"], samples[anchor], 4, 4, label_a="M1-lin", label_b="M1"))
    reports.mkdir(parents=True, exist_ok=True)
    paired_path = reports / "p1s_smoke_paired.json"
    paired_path.write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    with (reports / "p1s_smoke_timing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    artifacts = []
    if "t2-M1" in samples:
        def original_test(name: str) -> dict[str, np.ndarray]:
            arrays = samples[name]
            mask = (arrays["split"] == "test") & (arrays["condition"] == "original")
            return {key: value[mask] for key, value in arrays.items()}
        for variant, depth in (("B3", 4), ("B4", 1), ("B5", 4)):
            artifacts.extend(str(path) for path in create_figures(
                original_test("t2-M1"), reports / "figures/p1s" / f"M1-v-{variant}",
                reference=original_test("t2-" + variant), depth_a=4, depth_b=depth,
                label_a="M1", label_b=variant))
    registry = Path(__file__).with_name("collect_registry.py")
    subprocess.run([sys.executable, str(registry), str(runs),
                    "--out", str(reports / "experiment_registry.csv")], check=True)
    summary = {"runs": len(expected), "suite_seconds": suite["seconds"],
               "all_sample_tables_reproduced": True, "gate0_passed": True,
               "identical_writer_groups": shared_groups, "paired_contrasts": len(comparisons),
               "figures": artifacts, "timing": timing_rows,
               "scope": "smoke validation only; no method-selection or scientific claim"}
    (reports / "p1s_smoke_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/p1s/smoke.yaml"))
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        code = run_suite(args.suite, args.runs, args.procs)
        if code:
            return code
    summary = audit(args.suite, args.runs, args.reports)
    print(json.dumps({key: summary[key] for key in ("runs", "suite_seconds", "gate0_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
