#!/usr/bin/env python3
"""Verify the 15 P5L trainings and B0 and write the training manifest the confirmation checks (stdlib, login-safe).

Reads each run's record.json (status complete, checkpoint file hash, config as committed, no smoke or confirmation use),
each job's receipt and exit file, and the committed lock. It reads no score: records of finished trainings carry only
timing and hashes (`expose_scores` is off). Refuses an existing output file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/work/xg24i002/x10041/LoopLora/runs/p5l/main")
LABELS = ("B8", "M1-L4", "B4", "B5-L4", "B3-L4")
SEEDS = (17, 29, 43)
BACKBONE = {"id": "Qwen/Qwen3-1.7B-Base", "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"}


def digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def receipts(path: Path) -> dict:
    """The latest receipt per job name (a resubmission after an interruption replaces the earlier one)."""
    found = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = Path(row["log"]).name.rsplit("-2026", 1)[0]
        found[name] = row
    return found


def verify(runs_root: Path, receipts_path: Path, lock_path: Path, source: str) -> dict:
    lock = json.loads(lock_path.read_text())
    if lock["status"] != "locked" or lock["rank"] != 128 or lock["backbone"] != BACKBONE:
        raise ValueError("the lock is not the registered P5L lock")
    jobs, runs = receipts(receipts_path), {}
    names = [f"{label}-s{seed}" for label in LABELS for seed in SEEDS] + ["B0-s17"]
    for name in names:
        label = name.rsplit("-s", 1)[0]
        directory = runs_root / name / name
        record = json.loads((directory / "record.json").read_text())
        cfg = record["config"]
        committed = json.loads((ROOT / "configs/p5l/main-r128" / f"{name}.json").read_text())
        if record["status"] != "complete" or record["checkpoint_hash"] != digest(directory / "checkpoint.pt"):
            raise ValueError(f"incomplete or changed training: {name}")
        if cfg["smoke"] or cfg["eval"]["allow_confirmation"] or cfg["eval"]["splits"] != ["val"] or cfg["backbone"] != BACKBONE:
            raise ValueError(f"training used smoke settings, confirmation data or another backbone: {name}")
        if any(cfg[key] != committed[key] for key in ("name", "seed", "stage", "variant", "writer", "model", "backbone")) \
                or cfg["train"] != {**committed["train"]}:
            raise ValueError(f"the run's config is not the committed config: {name}")
        job = jobs[f"tr-p5l-{name}"]
        exit_file = Path(job["log"] + ".exit")
        if not exit_file.is_file() or int(exit_file.read_text()) != 0:
            raise ValueError(f"job {job['job_id']} of {name} did not exit 0")
        if label != "B0" and job["source"]["ref"] != source:
            raise ValueError(f"{name} ran at another source commit than {source}")
        runs[name] = dict(path=str(directory), job_id=job["job_id"], source=job["source"]["ref"],
                          checkpoint_hash=record["checkpoint_hash"], config_hash=record.get("config_hash"),
                          train_seconds=record.get("train_seconds"))
    return dict(study="p5l", condition="p5l-r128", rank=128, S=lock["S"], lrs=lock["lrs"], backbone=BACKBONE,
                lock=dict(path="reports/p5l/r128/lr-S-lock.json", hash=digest(lock_path)), training_source=source,
                runs={name: row["path"] for name, row in runs.items()}, jobs=runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", required=True, help="full commit the 15 trainings were pinned to")
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--receipts", type=Path, default=RUNS.parent / "main-submissions.jsonl")
    parser.add_argument("--lock", type=Path, default=ROOT / "reports/p5l/r128/lr-S-lock.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = verify(args.runs, args.receipts, args.lock, args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(runs=len(manifest["runs"]), out=str(args.out))))


if __name__ == "__main__":
    main()
