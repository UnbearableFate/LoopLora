#!/usr/bin/env python3
"""Collect symbolic record.json files into a deterministic CSV using stdlib only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

FIELDS = (
    "experiment_id", "status", "run_dir", "git_commit", "source_dirty", "source_hash",
    "config_hash", "data_seed", "eval_data_seed", "data_version", "model_revision",
    "train_seed", "backbone_seed", "checkpoint", "samples", "writer_checkpoint",
    "consumed_writer_checkpoint", "writer_source_record", "insertion_point", "task",
    "variant", "writer", "parameters", "state_bytes", "rank", "r_mem", "L", "mean_L",
    "train_tokens", "train_seconds", "pretrain_tokens", "pretrain_seconds", "eval_samples", "scores", "latency",
    "peak_memory_bytes", "failure_reason", "created_at", "finished_at",
)


def collect_records(roots: Iterable[str | Path]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            if root.name != "record.json":
                raise ValueError(f"expected record.json: {root}")
            paths.add(root.resolve())
        else:
            paths.update(path.resolve() for path in root.rglob("record.json"))
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or not record.get("experiment_id"):
            raise ValueError(f"invalid experiment record: {path}")
        rows.append({**record, "run_dir": str(path.parent)})
    return sorted(rows, key=lambda row: (str(row["experiment_id"]), row["run_dir"]))


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def write_registry(roots: Iterable[str | Path], out: str | Path) -> int:
    rows = collect_records(roots)
    extra = sorted(set().union(*(row.keys() for row in rows)) - set(FIELDS))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*FIELDS, *extra], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(value) for key, value in row.items()})
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=["runs/p1s"])
    parser.add_argument("--out", default="reports/experiment_registry.csv")
    args = parser.parse_args(argv)
    count = write_registry(args.roots, args.out)
    print(json.dumps({"records": count, "registry": str(Path(args.out).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
