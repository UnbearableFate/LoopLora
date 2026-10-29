#!/usr/bin/env python3
"""Verify the six B8W continuations and write the training manifest and the confirmation authorization (stdlib, login-safe).

Reads each continuation's record.json, train_log.jsonl and the seed's gate.json, each job's receipt and exit file, the committed
lock and the P1-L checkpoints it names. It reads no score and no test episode: the logs carry training-set loss and accuracy
only, and none of them is used. Refuses existing output files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/work/xg24i002/x10041/LoopLora/runs/b8w/main")
SEEDS = (17, 29, 43)
ARMS = ("shared", "untied")
STEPS, LR, DATA_START = 2000, 0.003, 17000
LOCK, PREREG = "reports/b8w/checkpoint-lock.json", "docs/prereg/B8W.md"
VARIANT = {"shared": "M1", "untied": "B8"}


def digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def receipts(path: Path) -> dict:
    found = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        found[Path(row["log"]).name.rsplit("-2026", 1)[0]] = row
    return found


def log_rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(runs_root: Path, receipts_path: Path, lock_path: Path, source: str) -> dict:
    lock = json.loads(lock_path.read_text())
    if lock["status"] != "locked" or lock["study"] != "b8w" or lock["prereg_hash"] != digest(ROOT / PREREG):
        raise ValueError("the lock is not the registered B8W lock")
    jobs, runs, details = receipts(receipts_path), {}, {}
    for seed in SEEDS:
        job = jobs[f"tr-b8w-s{seed}"]
        exit_file = Path(job["log"] + ".exit")
        if not exit_file.is_file() or int(exit_file.read_text()) != 0:
            raise ValueError(f"job {job['job_id']} of seed {seed} did not exit 0")
        if job["source"]["ref"] != source:
            raise ValueError(f"seed {seed} ran at another source commit than {source}")
        directory = runs_root / f"s{seed}"
        gate = json.loads((directory / "gate.json").read_text())
        start = lock["checkpoints"][f"M1-L4-s{seed}"]
        if (not gate["passed"] or gate["max_abs_difference"] != 0.0 or gate["tolerance"] != 0.0 or not gate["control_equal"]
                or gate["start_checkpoint_hash"] != start["hash"]
                or gate["converted_checkpoint_hash"] != digest(directory / "converted.pt")):
            raise ValueError(f"seed {seed}: the equality gate did not pass for the locked checkpoint")
        reference = log_rows(Path(start["path"]).parent / "train_log.jsonl")[0]["data_seed"]
        rows, records = {}, {}
        for arm in ARMS:
            target = directory / arm
            record = json.loads((target / "record.json").read_text())
            cfg = record["config"]
            expected_start = start["hash"] if arm == "shared" else gate["converted_checkpoint_hash"]
            if (record["status"] != "complete" or record["checkpoint_hash"] != digest(target / "checkpoint.pt")
                    or record["steps"] != STEPS or cfg["train"]["steps"] != STEPS or cfg["train"]["lr"] != LR
                    or cfg["train"]["data_start_step"] != DATA_START or cfg["train"]["warmup_steps"] != 20
                    or cfg["train"]["batch_size"] != 64 or cfg["train"]["depths"] != [4]
                    or cfg["variant"] != VARIANT[arm] or cfg["seed"] != seed or cfg["writer"] != "oracle_random"
                    or cfg["smoke"] or cfg["eval"]["allow_confirmation"] or cfg["eval"]["splits"] != ["val"]
                    or record["start_checkpoint_hash"] != expected_start or not record["gate"]["passed"]
                    or record["git_commit"] != source or record["source_dirty"]
                    or record["writer_hash"] != lock["oracle_hash"]):
                raise ValueError(f"seed {seed} {arm}: the continuation record disagrees with the lock or the registration")
            rows[arm] = log_rows(target / "train_log.jsonl")
            if [row["step"] for row in rows[arm]] != [1, *range(100, STEPS + 1, 100)]:
                raise ValueError(f"seed {seed} {arm}: the train log does not end at step {STEPS}")
            if {row["data_seed"] for row in rows[arm]} != {reference}:
                raise ValueError(f"seed {seed} {arm}: not the P1-L training stream")
            records[arm] = record
        if rows["shared"][0]["loss"] != rows["untied"][0]["loss"]:
            raise ValueError(f"seed {seed}: the first-step losses differ (the arms must start from the same function)")
        strip = lambda cfg: {k: v for k, v in cfg.items() if k not in {"name", "variant"}}
        if strip(records["shared"]["config"]) != strip(records["untied"]["config"]):
            raise ValueError(f"seed {seed}: the two arms differ in more than name and variant")
        for arm in ARMS:
            name = f"{arm}-s{seed}"
            runs[name] = str(directory / arm)
            details[name] = dict(job_id=job["job_id"], source=source, checkpoint_hash=records[arm]["checkpoint_hash"],
                                 start_checkpoint_hash=records[arm]["start_checkpoint_hash"],
                                 config_hash=records[arm]["config_hash"], train_seconds=records[arm]["train_seconds"],
                                 converted=str(directory / "converted.pt") if arm == "untied" else None)
        runs[f"start-s{seed}"] = str(Path(start["path"]).parent)
        for label in ("B4", "B8"):
            runs[f"{label}-s{seed}"] = str(Path(lock["checkpoints"][f"{label}-s{seed}"]["path"]).parent)
    return dict(study="b8w", steps=STEPS, lr=LR, data_start_step=DATA_START, window="b8w", training_source=source,
                lock=dict(path=LOCK, hash=digest(lock_path)), runs=runs, jobs=details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="full commit the three continuation jobs were pinned to")
    parser.add_argument("--out-manifest", type=Path, default=ROOT / "reports/b8w/continuation-manifest.json")
    parser.add_argument("--out-authorization", type=Path, default=ROOT / "reports/b8w/confirmation-authorization.json")
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--receipts", type=Path, default=RUNS.parent / "main-submissions.jsonl")
    args = parser.parse_args()
    for path in (args.out_manifest, args.out_authorization):
        if path.exists():
            raise FileExistsError(path)
    manifest = verify(args.runs, args.receipts, ROOT / LOCK, args.source)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    authorization = dict(
        study="b8w", status="allowed", confirmation_window="b8w", gated_conditions=1, confidence=0.95,
        prereg=dict(path=PREREG, hash=digest(ROOT / PREREG)), lock=manifest["lock"],
        manifest=str(args.out_manifest), manifest_hash=digest(args.out_manifest), training_source=args.source,
        orchestrator_decision=("docs/prereg/B8W.md registered by the orchestrator (master 3c9c396, 2026-10-29): after the six "
                               "continuations are verified, evaluate the 15 groups on test [108000, 110000) once. This file is "
                               "committed for the orchestrator's check; no evaluation job is submitted before the "
                               "orchestrator replies."),
        release_rule=("Verify the six continuations (records, logs of 2000 steps, gates, start hashes, one source, exit-0 "
                      "receipts), then create this authorization; no confirmation episode before the orchestrator's check."))
    args.out_authorization.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(runs=len(manifest["runs"]), manifest=str(args.out_manifest), authorization=str(args.out_authorization))))


if __name__ == "__main__":
    main()
