#!/usr/bin/env python3
"""Release the locked main analysis, then append its records without losing prior rows.

Run on Miyabi-G compute through scripts/submit.py. Confirmation analysis is
deliberately single-pass; an interrupted analysis needs inspection before recovery.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import time

from analyze_p1s_main import analyze
from collect_registry import _cell, collect_records
from looplora.sym.main import seal, save
from looplora.sym.run import source_metadata


def append_registry(roots: list[Path], registry: Path) -> dict:
    csv.field_size_limit(10_000_000)
    with registry.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        previous = list(reader)
    keys = {(row["experiment_id"], row["run_dir"]) for row in previous}
    if len(keys) != len(previous):
        raise ValueError("existing registry contains duplicate experiment/path rows")
    incoming = collect_records(roots)
    for row in incoming:
        key = row["experiment_id"], row["run_dir"]
        if key in keys:
            raise ValueError(f"registry already contains run: {key}")
        keys.add(key)
        if row.get("confirmation_samples_written"):
            location = Path(row["run_dir"]) / "confirmation" / "metrics.json"
            metrics = json.loads(location.read_text())
            if not metrics.get("finalized_after_global_barrier"):
                raise ValueError(f"confirmation not released: {location}")
            row.update(confirmation_metrics=str(location),
                       confirmation_scores=metrics["table"],
                       confirmation_evaluation_seed=metrics["evaluation_seed"])
    extra = sorted(set().union(*(row.keys() for row in incoming)) - set(fields))
    temporary = registry.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extra, lineterminator="\n")
        writer.writeheader()
        writer.writerows(previous)
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in incoming)
    temporary.replace(registry)
    return dict(previous_rows=len(previous), appended_rows=len(incoming),
                total_rows=len(previous) + len(incoming), registry=str(registry),
                roots=[str(root) for root in roots], previous_rows_preserved=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--throughput-roots", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    seal(args.plan)
    summary = analyze(args.plan, args.out, args.figures)
    if summary["status"] != "complete" or summary["runs"] != 105 or summary["confirmation_runs"] != 102:
        raise ValueError("incomplete main analysis")
    analysis_seconds = time.perf_counter() - started
    root = Path(json.loads(args.plan.read_text())["root"])
    registry = append_registry([root / "runs", *args.throughput_roots], args.registry)
    save(args.out / "registry_audit.json", registry)
    shutil.copyfile(root / "barrier.json", args.out / "barrier.json")
    save(args.out / "analysis_execution.json", dict(
        **source_metadata(), host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"),
        analysis_seconds=analysis_seconds, total_seconds=time.perf_counter() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        registry=registry, status="complete"))
    print(json.dumps(dict(status="complete", runs=105, confirmation_runs=102,
                          gate_rows=summary["gate_rows"], registry=registry)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
