#!/usr/bin/env python3
"""Write reports/b8w/checkpoint-lock.json (docs/prereg/B8W.md, "Order" step 3); stdlib only, login-safe.

Locks the nine registered P1-L r = 128 checkpoints the study uses (M1-L4 start; B4 and B8 references; seeds 17/29/43)
with their blake2b hashes, the registered N, lr and stream start, the confirmation window, and the validation-only smoke
result of the equality gate. It reads no test episode and no score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
LABELS = ("M1-L4", "B8", "B4")
STEPS, LR, DATA_START = 2000, 0.003, 17000


def digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-gate", type=Path, required=True, help="gate.json of the validation-only smoke")
    parser.add_argument("--smoke-job", required=True)
    parser.add_argument("--smoke-steps", type=int, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/b8w/checkpoint-lock.json")
    args = parser.parse_args()
    manifest_path = ROOT / "reports/p1l/r128/run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    p1l_lock = json.loads((ROOT / "reports/p1l/r128/lr-S-lock.json").read_text())
    if (p1l_lock["S"], p1l_lock["lrs"]["M1-L4"], p1l_lock["lrs"]["B8"]) != (DATA_START, LR, LR):
        raise ValueError("the P1-L r = 128 lock disagrees with the registered B8W lr or stream start")
    runs = {row["name"]: row for row in manifest["runs"]}
    checkpoints = {}
    for label in LABELS:
        for seed in SEEDS:
            name = f"{label}-s{seed}"
            run = Path(runs[name]["run"])
            record = json.loads((run / "record.json").read_text())
            actual = digest(run / "checkpoint.pt")
            if actual != runs[name]["checkpoint_hash"] or actual != record["checkpoint_hash"] or record["status"] != "complete":
                raise ValueError(f"{name}: the checkpoint, its record and the P1-L manifest disagree")
            checkpoints[name] = dict(path=str(run / "checkpoint.pt"), hash=actual, config_hash=record["config_hash"],
                                     training_source=record["git_commit"], steps=record["config"]["train"]["steps"],
                                     lr=record["config"]["train"]["lr"])
    gate = json.loads(args.smoke_gate.read_text())
    if not gate["passed"] or gate["max_abs_difference"] != 0.0:
        raise ValueError("the smoke equality gate did not pass; no lock")
    lock = dict(
        status="locked", study="b8w", prereg="docs/prereg/B8W.md", prereg_hash=digest(ROOT / "docs/prereg/B8W.md"),
        steps=STEPS, lr=LR, data_start_step=DATA_START, warmup_steps=20, batch_size=64, depths=[4],
        window="b8w", window_start=108000, episodes=2000, eval_seed=4242, confidence=0.95, noninferiority=0.02,
        decision="untied@4 - shared@4, hops 2 and 3: keeps iff every seed >= -2 pp and the pooled 95% CI lower bound > -2 pp",
        seeds=list(SEEDS), oracle_hash=manifest["oracle_hash"], p1l_run_manifest=str(manifest_path.relative_to(ROOT)),
        p1l_run_manifest_hash=digest(manifest_path), p1l_lr_lock_hash=digest(ROOT / "reports/p1l/r128/lr-S-lock.json"),
        checkpoints=checkpoints,
        smoke=dict(job=args.smoke_job, seed=17, steps_per_arm=args.smoke_steps, gate=gate),
        continuation_source=("the commit that adds this lock, or a later head that changes no file under src/looplora, "
                             "scripts/run_b8w.py or the configs (the job names which)"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status="locked", checkpoints=len(checkpoints), out=str(args.out))))


if __name__ == "__main__":
    main()
