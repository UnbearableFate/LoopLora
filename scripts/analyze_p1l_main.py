#!/usr/bin/env python3
"""Compute-only P1-L timing and pilot locks; confirmation analysis follows later."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from run_p1l_main import LABELS, ROOT, STUDIES, digest, save, trained_labels


def read(path: Path):
    return json.loads(path.read_text())


def timings(root: Path, rank: int = 32, study: str = "p1l") -> list[dict]:
    rows = []
    for label in trained_labels(rank, study):
        directory = root / f"{label}-s17"
        if not directory.is_dir():
            if study != "p1l":
                parent = "smoke"
            elif rank == 128:
                parent = "smoke-a" if label in {"B8", "M1-L4"} else "smoke-b"
            else:
                parent = "smoke-context" if label in {"B1", "B2"} else "smoke-oracle"
            directory = root / parent / f"{label}-s17"
        record = read(directory / "record.json")
        cfg = record["config"]
        if cfg["model"]["r"] != rank:
            raise ValueError("timing configuration has the wrong rank")
        if record["status"] != "complete" or not cfg["smoke"]:
            raise ValueError(f"incomplete/non-smoke run: {directory}")
        if cfg["eval"]["splits"] != ["val"] or cfg["eval"]["allow_confirmation"]:
            raise ValueError("smoke changed the evaluation boundary")
        with (directory / "train_log.jsonl").open() as stream:
            durations = [row["step_seconds"] for line in stream
                         if (row := json.loads(line))["step"] > 5]
        # The original20-step preflight logged every second step (8 post-warmup
        # observations). Full smokes log every step and supply hundreds.
        if len(durations) < 5:
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
    rows = timings(args.root, args.rank, args.study)
    # Two-hour job /1.5 margin, including a bounded load/export reserve.
    longest = max(row["step_seconds"] for row in rows)
    reserve = 60. + max(row["overhead_seconds"] for row in rows)
    job_cap = math.floor((7200 / 1.5 - reserve) / longest)
    # Amendment B2 restores per-job sizing; the former total cap is descriptive.
    # Sequential backbone reuse is the measured packing, with no GPU contention.
    overhead = 5 * sum(row["overhead_seconds"] for row in rows) + len(rows) * 5 * 15.
    step_sum = sum(row["step_seconds"] for row in rows)
    S = 17000 if args.rank == 32 else min(17000, job_cap // 500 * 500)
    if S > job_cap:
        raise ValueError("smoke timing no longer supports preregistered S17000 under2h/1.5")
    if S < 30:
        raise ValueError("no feasible training budget")
    predicted = ((3 * S + 2 * (S // 3)) * step_sum + overhead) / 3600
    result = dict(status="sized", rank=args.rank, S=S, pilot_steps=S // 3,
                  reduced_steps_flag=S != 17000, smoke_steps=math.ceil(S / 10),
                  longest_step_seconds=longest, reserve_seconds=reserve,
                  main_plus_pilot_node_hours=predicted,
                  job_cap_steps=job_cap,
                  packing="one GPU process; runs sequentially reuse backbone, prefetch2 per active run",
                  overhead_seconds=overhead,
                  budget_rule="B2: no total cap; record every job's node-hours",
                  excludes="smoke, validation, confirmation, name-gradient probe and failed jobs",
                  timing_rows=rows, source=str(args.root),
                  time=datetime.now(timezone.utc).isoformat())
    if args.study != "p1l":
        result["study"] = args.study
    save(args.out, result)
    print(json.dumps(result), flush=True)


def select(args) -> None:
    sizing = read(args.sizing)
    if sizing.get("study", "p1l") != args.study:
        raise ValueError("the sizing file belongs to another study")
    selections, lrs = [], {}
    for label in trained_labels(args.rank, args.study):
        candidates = []
        for rate in (.001, .003):
            directory = args.root / label / f"{label}-lr{rate:g}"
            record = read(directory / "record.json")
            cfg = record["config"]
            if (record["status"] != "complete" or cfg["seed"] != 17
                    or cfg["train"]["steps"] != sizing["pilot_steps"]
                    or cfg["model"]["r"] != args.rank
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
    result = dict(status="locked", rank=args.rank, S=sizing["S"], lrs=lrs, selection=selections,
                  rule="val full-vocabulary EM, pooled hops1/2/3 at trained L; ties lower lr",
                  sizing_hash=digest(args.sizing), time=datetime.now(timezone.utc).isoformat())
    result["packing"] = sizing["packing"]
    result["projected_main_plus_pilot_node_hours"] = sizing["main_plus_pilot_node_hours"]
    if args.study != "p1l":
        result["study"] = args.study
    save(args.out, result)
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("size", "select"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizing", type=Path)
    parser.add_argument("--rank", type=int, choices=(32, 128), default=32)
    parser.add_argument("--study", choices=tuple(STUDIES), default="p1l")
    args = parser.parse_args()
    if args.study != "p1l" and args.rank != 32:
        parser.error(f"{args.study} runs at r = 32 only")
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
