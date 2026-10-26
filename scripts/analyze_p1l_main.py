#!/usr/bin/env python3
"""Compute-only P1-L timing and pilot locks; confirmation analysis follows later."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from run_p1l_main import LABELS, ROOT, digest, save


def read(path: Path):
    return json.loads(path.read_text())


def timings(root: Path) -> list[dict]:
    rows = []
    for label in LABELS:
        directory = root / f"{label}-s17"
        record = read(directory / "record.json")
        cfg = record["config"]
        if record["status"] != "complete" or not cfg["smoke"]:
            raise ValueError(f"incomplete/non-smoke run: {directory}")
        if cfg["eval"]["splits"] != ["val"] or cfg["eval"]["allow_confirmation"]:
            raise ValueError("smoke changed the evaluation boundary")
        with (directory / "train_log.jsonl").open() as stream:
            durations = [row["step_seconds"] for line in stream
                         if (row := json.loads(line))["step"] > 5]
        if len(durations) < 10:
            raise ValueError("insufficient measured production steps")
        mean = statistics.mean(durations)
        overhead = max(0., record["train_seconds"] - cfg["train"]["steps"] * mean)
        rows.append(dict(label=label, steps=cfg["train"]["steps"],
                         step_seconds=mean, overhead_seconds=overhead,
                         peak_memory_bytes=record["peak_memory_bytes"],
                         checkpoint_hash=record["checkpoint_hash"],
                         job_id=record["job_id"], host=record["host"]))
    return rows


def size(args) -> None:
    rows = timings(args.root)
    # Two-hour job /1.5 margin, including a bounded load/export reserve.
    longest = max(row["step_seconds"] for row in rows)
    reserve = 60. + max(row["overhead_seconds"] for row in rows)
    S = math.floor((7200 / 1.5 - reserve) / longest)
    if S < 30:
        raise ValueError("no feasible training budget")
    predicted = (3 * S + 2 * (S // 3)) * sum(row["step_seconds"] for row in rows) / 3600
    result = dict(status="sized", S=S, pilot_steps=S // 3, smoke_steps=math.ceil(S / 10),
                  longest_step_seconds=longest, reserve_seconds=reserve,
                  main_plus_pilot_train_node_hours=predicted,
                  excludes="smoke, process/model startup, evaluation, J-lens and failed jobs",
                  timing_rows=rows, source=str(args.root),
                  time=datetime.now(timezone.utc).isoformat())
    save(args.out, result)
    print(json.dumps(result), flush=True)


def select(args) -> None:
    sizing = read(args.sizing)
    selections, lrs = [], {}
    for label in LABELS:
        candidates = []
        for rate in (.001, .003):
            directory = args.root / label / f"{label}-lr{rate:g}"
            record = read(directory / "record.json")
            cfg = record["config"]
            if (record["status"] != "complete" or cfg["seed"] != 17
                    or cfg["train"]["steps"] != sizing["pilot_steps"]
                    or cfg["eval"]["splits"] != ["val"]
                    or cfg["eval"]["hops"] != [1, 2, 3]):
                raise ValueError(f"pilot contract mismatch: {directory}")
            table = record["validation"]
            protocol = cfg["protocol"]
            selected = [row for row in table if row["condition"] == f"{protocol}/original"]
            if {row["hop"] for row in selected} != {1, 2, 3} or len(selected) != 3:
                raise ValueError("pilot requires exactly one evaluation row per trained hop")
            total = sum(row["n"] for row in selected)
            score = sum(row["accuracy"] * row["n"] for row in selected) / total
            candidates.append(dict(lr=rate, accuracy=score, n=total,
                                   record=str(directory / "record.json"),
                                   checkpoint_hash=record["checkpoint_hash"]))
        winner = max(candidates, key=lambda row: (row["accuracy"], -row["lr"]))
        lrs[label] = winner["lr"]
        selections.append(dict(label=label, selected_lr=winner["lr"], candidates=candidates))
    result = dict(status="locked", S=sizing["S"], lrs=lrs, selection=selections,
                  rule="val full-vocabulary EM, pooled hops1/2/3 at trained L; ties lower lr",
                  sizing_hash=digest(args.sizing), time=datetime.now(timezone.utc).isoformat())
    save(args.out, result)
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("size", "select"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizing", type=Path)
    args = parser.parse_args()
    from audit_p4s_stopping import require_compute
    require_compute()
    if args.stage == "size":
        size(args)
    else:
        if args.sizing is None:
            parser.error("select requires --sizing")
        select(args)


if __name__ == "__main__":
    main()
