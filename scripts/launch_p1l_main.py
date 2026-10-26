#!/usr/bin/env python3
"""Small stdlib control-plane launcher; every compute job uses submit.py."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess

from run_p1l_main import LABELS, ROOT, SEEDS, labels_for_rank, save


def walltime(expected: float) -> str:
    seconds = math.ceil(expected * 1.5 / 300) * 300
    if seconds > 7200:
        raise ValueError(f"measured work requires {seconds}s after1.5margin, above2h")
    now = datetime.now(timezone.utc)
    if now + timedelta(seconds=seconds) >= datetime(2026, 10, 28, tzinfo=timezone.utc):
        raise ValueError("requested job would extend beyond the maintenance deadline")
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def submit(name: str, source: str, expected: float, argv: list[str]) -> dict:
    wall = walltime(expected)
    queue = "debug-g" if wall <= "00:30:00" else "regular-g"
    command = [str(ROOT / "scripts/submit.py"), "--name", name, "--queue", queue,
        "--walltime", wall, "--scale-up", "docs/prereg/P1L.md", "--commit", source, "--", *argv]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    receipt = json.loads(result.stdout.strip())
    receipt.update(name=name, requested_walltime=wall, expected_seconds=expected,
                   nodes=1, queue=queue, argv=argv)
    print(json.dumps(receipt), flush=True)
    return receipt


def launch(args):
    source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    sizing = json.loads(args.sizing.read_text())
    if sizing.get("rank", 32) != args.rank:
        raise ValueError("sizing rank differs from the requested condition")
    times = {row["label"]: row["step_seconds"] for row in sizing["timing_rows"]}
    overhead = {row["label"]: row["overhead_seconds"] for row in sizing["timing_rows"]}
    args.root.mkdir(parents=True, exist_ok=True)
    receipt_path = args.root / f"{args.stage}-jobs.json"
    state = json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(stage=args.stage, jobs={})
    if args.stage == "pilot":
        groups = [(label, [label], 17) for label in labels_for_rank(args.rank)]
    else:
        relative_lock = "reports/p1l/lr-S-lock.json" if args.rank == 32 else "reports/p1l/r128/lr-S-lock.json"
        lock_path = ROOT / relative_lock
        lock = json.loads(lock_path.read_text())
        committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{source}:{relative_lock}"], text=True)
        if json.loads(committed) != lock or lock["S"] != sizing["S"]:
            raise ValueError("main requires the committed final S/lr lock")
        groups = []
        for seed in SEEDS:
            candidates = [("B8", ["B8"]), ("M1", ["M1-L4"]),
                          ("B5", ["B5-L4"]), ("B4-B3", ["B4", "B3-L4"])]
            if args.rank == 32:
                candidates.extend([("B1", ["B1"]), ("B2", ["B2"])])
            for tag, labels in candidates:
                expected = sum(lock["S"] * times[label] + overhead[label] for label in labels) + 30.
                if tag == "B4-B3" and math.ceil(expected * 1.5 / 300) * 300 > 7200:
                    groups.extend((f"{label}-s{seed}", [label], seed) for label in labels)
                    continue
                groups.append((f"{tag}-s{seed}", labels, seed))
        if args.rank == 32:
            groups.append(("B0-s17", ["B0"], 17))
    runs = {}
    for tag, labels, seed in groups:
        out = args.root / args.stage / tag
        for label in labels:
            runs[f"{label}-s{seed}"] = str(out / f"{label}-s{seed}")
        if tag in state["jobs"]:
            continue
        if out.exists():
            raise FileExistsError(f"output exists without a submission receipt: {out}")
        if args.stage == "pilot":
            steps = sizing["pilot_steps"]
            expected = 2 * steps * sum(times[label] for label in labels)
            expected += 2 * sum(overhead[label] for label in labels) + 210.
        else:
            steps = lock["S"]
            expected = sum(steps * times[label] + overhead[label] for label in labels if label != "B0") + 30.
        argv = ["scripts/run_p1l_main.py", args.stage, "--out", str(out),
                "--seed", str(seed), "--rank", str(args.rank), "--labels", *labels]
        if args.stage == "pilot":
            argv += ["--steps", str(steps)]
        prefix = "p1l" if args.rank == 32 else "p1l-r128"
        receipt = submit(f"{prefix}-{args.stage[0]}-{tag}", source, expected, argv)
        state["jobs"][tag] = receipt
        state["source"] = source
        save(receipt_path, state)
    if args.stage == "main":
        save(args.root / "training-manifest.json", dict(condition=f"oracle-r{args.rank}", rank=args.rank,
            S=lock["S"], lrs=lock["lrs"], runs=runs,
            lock_commit=source, lock_file_hash=__import__("hashlib").blake2b(lock_path.read_bytes(), digest_size=16).hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "main"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sizing", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(32, 128), default=32)
    args = parser.parse_args()
    launch(args)


if __name__ == "__main__":
    main()
