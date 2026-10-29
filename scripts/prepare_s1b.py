#!/usr/bin/env python3
"""Prepare S1B branch evidence, lock or release; no model imports.

Full manifest/ID reconciliation is bounded metadata work and may be run on compute.
All actual experiment execution uses looplora.llm.s1b_run through submit.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from looplora.llm import s1b_artifacts as a, s1b_data as d, s1b_rules as r
from looplora.infra.seeds import stable_hash


def control_release(lock_path, release_path, verdict_path, timings):
    lock, release = a.require_release(release_path, lock_path)
    verdict = a.read(verdict_path)
    if verdict.get("stage") != "S1B" or verdict.get("release_blake2b_128") != d.old.file_digest(release_path):
        raise ValueError("control verdict belongs to another main release")
    configs, jobs = {}, {}
    expected = set()
    for arm in r.branch_arms(release["branch"]):
        if not arm.endswith("4") or verdict["contrasts"][f"{arm}-{arm[0]}1"]["passed"] is not True:
            continue
        settings = release["arms"][arm]
        for kind in ("fixed", "untied"):
            key = f"{arm}/{kind}"
            expected.add(key)
            timing = timings[key]
            if (timing.get("steps") != d.PILOT_STEPS or timing.get("control") != kind or timing.get("arm") != arm
                    or timing.get("release_blake2b_128") != d.old.file_digest(release_path)
                    or timing.get("verdict_blake2b_128") != d.old.file_digest(verdict_path)):
                raise ValueError("control timing must measure the actual 200-step path")
            jobs[key] = r.walltime(timing["step_seconds"], timing["eval_seconds"], timing["load_seconds"], timing["overhead_seconds"])
            for seed in d.RUN_SEEDS:
                configs[f"{arm}/{seed}/{kind}"] = stable_hash(r.science_config(release["branch"], arm, seed, settings["lr"], settings["read"], control=kind))
    if set(timings) != expected or not expected:
        raise ValueError("controls require both controls for every triggered backend, with measured timing")
    return dict(stage="S1B-CONTROLS", main_release_blake2b_128=d.old.file_digest(release_path),
                verdict_path=str(Path(verdict_path).resolve()), verdict_blake2b_128=d.old.file_digest(verdict_path),
                lock_blake2b_128=d.old.file_digest(lock_path), source_blake2b_128=lock["source_blake2b_128"],
                configs=configs, walltime_seconds=jobs, timings=timings)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("branch", "lock", "release", "control-release"))
    p.add_argument("--out", type=Path, required=True)
    for name in ("decision", "dev", "ablation", "preflight", "manifest", "branch-record", "calibration", "pilot", "smoke", "release", "verdict", "timings"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--science-commit")
    p.add_argument("--unseen", action="store_true")
    p.add_argument("--reload", action="append", default=[], metavar="ARM=ABS_REPORT")
    args = p.parse_args(argv)
    needed = dict(branch=["decision", "dev"], lock=["preflight"],
                  release=["manifest", "branch_record", "calibration", "pilot", "smoke", "science_commit"],
                  **{"control-release": ["manifest", "release", "verdict", "timings"]})
    for name in needed[args.mode]:
        if getattr(args, name) is None:
            p.error(f"--{name.replace('_', '-')} is required")
    if args.mode == "branch":
        result = a.branch_record(args.decision, args.dev, args.ablation)
    elif args.mode == "lock":
        result = a.make_lock(args.preflight)
    elif args.mode == "release":
        reloads = dict(v.split("=", 1) for v in args.reload) if args.reload else None
        result = a.prepare_release(args.manifest, args.branch_record, args.calibration, args.pilot, args.smoke,
                                   args.science_commit, unseen=args.unseen, reloads=reloads)
    else:
        result = control_release(args.manifest, args.release, args.verdict, a.read(args.timings))
    a.dump(args.out, result)
    print(f"S1B {args.mode}: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
