#!/usr/bin/env python3
"""Small stdlib control-plane launcher; every compute job uses submit.py."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from run_p1l_main import LABELS, ROOT, SEEDS, STUDIES, config, digest, lock_file, save, trained_labels


def walltime(expected: float) -> str:
    seconds = math.ceil(expected * 1.5 / 300) * 300
    if seconds > 7200:
        raise ValueError(f"measured work requires {seconds}s after1.5margin, above2h")
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def submit(name: str, source: str, expected: float, argv: list[str], scale_up: str = "docs/prereg/P1L.md") -> dict:
    wall = walltime(expected)
    queue = "debug-g" if wall <= "00:30:00" else "regular-g"
    command = [str(ROOT / "scripts/submit.py"), "--name", name, "--queue", queue,
        "--walltime", wall, "--scale-up", scale_up, "--commit", source, "--", *argv]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    receipt = json.loads(result.stdout.strip())
    receipt.update(name=name, requested_walltime=wall, expected_seconds=expected,
                   nodes=1, queue=queue, argv=argv)
    print(json.dumps(receipt), flush=True)
    return receipt


def selected_writer(source: str) -> str:
    """The oracle code WP-L6 selected, read from the decision committed at `source` (P1L-S runs iff it selected one)."""
    decision = STUDIES["p1l-s"]["decision"]
    committed = json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{source}:{decision}"], text=True))
    if committed != json.loads((ROOT / decision).read_text()):
        raise ValueError("the WP-L6 decision in the working tree is not the committed one")
    if committed["status"] != "selected":
        raise ValueError("P1L-S runs only if WP-L6 selected an oracle code")
    for label in trained_labels(32, "p1l-s"):
        for seed in SEEDS:
            if config(label, seed, 32, "p1l-s")["writer"] != committed["selected"]:
                raise ValueError(f"configs/p1l-s/{label}-s{seed} does not use the selected code; "
                                 "run scripts/make_p1ls_configs.py and commit")
    return committed["selected"]


def reused_runs(path: Path, labels: tuple[str, ...]) -> dict:
    """P1-L's r = 32 runs of `labels` that P1L-S evaluates without retraining, with the hashes that identify them."""
    source = json.loads(path.read_text())
    if source.get("condition") != "oracle-r32" or source.get("rank", 32) != 32 or "study" in source:
        raise ValueError("--reuse needs P1-L's r = 32 training-manifest.json")
    return {name: dict(source=source["runs"][name], manifest=str(path), manifest_hash=digest(path),
                       checkpoint_hash=digest(Path(source["runs"][name]) / "checkpoint.pt"))
            for label in labels for name in (f"{label}-s{seed}" for seed in SEEDS)}


def launch(args):
    study = STUDIES[args.study]
    source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    sizing = json.loads(args.sizing.read_text())
    if sizing.get("rank", 32) != args.rank or sizing.get("study", "p1l") != args.study:
        raise ValueError("sizing rank or study differs from the requested condition")
    writer = None if args.study == "p1l" else selected_writer(source)
    reused = reused_runs(args.reuse, study["reused"]) if args.study != "p1l" and args.stage == "main" else {}
    times = {row["label"]: row["step_seconds"] for row in sizing["timing_rows"]}
    overhead = {row["label"]: row["overhead_seconds"] for row in sizing["timing_rows"]}
    args.root.mkdir(parents=True, exist_ok=True)
    receipt_path = args.root / f"{args.stage}-jobs.json"
    state = json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(stage=args.stage, jobs={})
    if args.stage == "pilot":
        groups = [(label, [label], 17) for label in trained_labels(args.rank, args.study)]
    else:
        relative_lock = lock_file(args.rank, args.study)
        lock_path = ROOT / relative_lock
        lock = json.loads(lock_path.read_text())
        committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{source}:{relative_lock}"], text=True)
        if json.loads(committed) != lock or lock["S"] != sizing["S"]:
            raise ValueError("main requires the committed final S/lr lock")
        groups = []
        if args.study != "p1l":
            # One single-run job per retrained reader and seed (the prereg's cost table); B3-L4 is reused.
            groups = [(f"{label}-s{seed}", [label], seed) for seed in SEEDS
                      for label in trained_labels(args.rank, args.study)]
        else:
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
        if args.study != "p1l":
            argv += ["--study", args.study]
        prefix = study["prefix"] if args.rank == 32 else f"{study['prefix']}-r128"
        receipt = submit(f"{prefix}-{args.stage[0]}-{tag}", source, expected, argv, study["prereg"])
        state["jobs"][tag] = receipt
        state["source"] = source
        save(receipt_path, state)
    if args.stage == "main":
        manifest = dict(condition=f"oracle-r{args.rank}", rank=args.rank,
            S=lock["S"], lrs=lock["lrs"], runs=runs,
            lock_commit=source, lock_file_hash=__import__("hashlib").blake2b(lock_path.read_bytes(), digest_size=16).hexdigest())
        if args.study != "p1l":
            manifest.update(study=args.study, condition=study["condition"], writer=writer, reused=reused,
                            runs={**runs, **{name: row["source"] for name, row in reused.items()}})
        save(args.root / "training-manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "main"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sizing", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(32, 128), default=32)
    parser.add_argument("--study", choices=tuple(STUDIES), default="p1l")
    parser.add_argument("--reuse", type=Path, help="P1-L's r = 32 training-manifest.json (p1l-s main)")
    args = parser.parse_args()
    if args.study != "p1l" and args.rank != 32:
        parser.error(f"{args.study} runs at r = 32 only")
    if (args.study != "p1l" and args.stage == "main") != (args.reuse is not None):
        parser.error("--reuse is required by, and only valid for, the main stage of a study that reuses P1-L runs")
    launch(args)


if __name__ == "__main__":
    main()
