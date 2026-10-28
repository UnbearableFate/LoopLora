#!/usr/bin/env python3
"""IVD (F8): key interference versus memory load on frozen r = 32 symbolic checkpoints (docs/prereg/IVD.md).

Three stages, each its own job (scripts/submit.py in a Miyabi-G allocation):

  lock   write the design (settings and the blake2b-sealed checkpoint list) to a new JSON file;
  smoke  evaluate the training-seed-17 checkpoints on validation episodes (seed 4242, 128 episodes);
         records timings, the achieved key overlaps and a walltime suggestion; no confirmation data;
  run    evaluate all twelve checkpoints on the new confirmation episodes (split test, seed 999004,
         2000 episodes); requires the committed design (Git lock) and a passed smoke report.

Nothing is trained and no checkpoint is changed: the reader and writer state hashes and the
checkpoint file hashes are verified after every evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from audit_uns import DEFAULT_F2, DEFAULT_P1S
from s9_common import absolute_path, digest_checkpoint, now, require, require_compute, verify_git_lock, write_json
from s9_torch import load_frozen, state_hash, synchronize
from looplora.data.relations import generate
from looplora.infra.seeds import stable_hash
from looplora.sym import ivd, uns
from looplora.sym.evaluate import stratify_queries
from looplora.sym.model import make_task_spec
from looplora.sym.run import source_metadata

STUDY = "IVD"
CONFIRMATION = dict(split="test", seed=999004, episodes=2000)
SMOKE = dict(split="val", seed=4242, episodes=128, training_seed=17)
HOPS = list(range(1, 9))
BATCH_SIZE = 64
RANK = 32
READERS = ("M1", "B4")
DEPTHS = {"M1": [1, 4], "B4": [1]}
BOOTSTRAP = dict(resamples=2000, seed=2718)
WALLTIME_MARGIN = 1.5


def settings() -> dict:
    """Every registered setting; the lock file is exactly these plus the sealed checkpoint list."""
    return dict(
        study=STUDY, prereg="docs/prereg/IVD.md", confirmation=dict(CONFIRMATION), smoke=dict(SMOKE),
        hops=list(HOPS), batch_size=BATCH_SIZE, depths={reader: list(depths) for reader, depths in DEPTHS.items()},
        loads=dict(ivd.LOADS), overlaps=list(ivd.OVERLAPS), high_cosine=ivd.HIGH_COSINE,
        candidates=ivd.POOL_CANDIDATES, pool="seen",
        decision=dict(reader="M1", load=ivd.DECISION_LOAD, hops=list(ivd.DECISION_HOPS), minimum=ivd.GAIN_MINIMUM,
                      writers=len(ivd.WRITERS), alpha=0.05, confidence=1 - 0.05 / len(ivd.WRITERS), **BOOTSTRAP))


def build_design(p1s_root: str | Path, f2_root: str | Path) -> dict:
    entries = []
    for writer in ivd.WRITERS:
        for reader in READERS:
            for seed in uns.SEEDS:
                path = (uns.run_root(writer, reader, seed, p1s_root=p1s_root, f2_root=f2_root) / "checkpoint.pt").resolve()
                require(path.is_file(), f"missing checkpoint {path}")
                entries.append(dict(name=f"{writer}-{reader}-r{RANK}-s{seed}", writer=writer, variant=reader, r=RANK,
                                    seed=seed, path=str(path), blake2b=digest_checkpoint(path)))
    return {**settings(), "checkpoints": entries}


def verify_design(design: dict) -> list[dict]:
    """The design must be exactly the registered settings and an intact set of the twelve checkpoints."""
    require({key: value for key, value in design.items() if key != "checkpoints"} == settings(),
            "the design differs from the registered settings")
    entries = design["checkpoints"]
    keys = [(entry["writer"], entry["variant"], entry["seed"]) for entry in entries]
    expected = {(writer, reader, seed) for writer in ivd.WRITERS for reader in READERS for seed in uns.SEEDS}
    require(len(keys) == len(set(keys)) and set(keys) == expected and all(entry["r"] == RANK for entry in entries),
            "the design must list exactly the twelve registered checkpoints")
    for entry in entries:
        require(digest_checkpoint(Path(entry["path"])) == entry["blake2b"], f"checkpoint changed: {entry['name']}")
    return entries


def evaluate_checkpoint(entry: dict, design: dict, evaluation: dict, *, device: str) -> tuple[dict, dict]:
    """All IVD cells of one frozen checkpoint: (flat samples, timing row)."""
    path = Path(entry["path"])
    require(digest_checkpoint(path) == entry["blake2b"], f"checkpoint changed: {entry['name']}")
    model = load_frozen(entry, device)
    before = state_hash(model)
    config = copy.deepcopy(model.config)
    config["eval"]["hops"] = list(design["hops"])
    spec = make_task_spec(config, evaluation=True)
    split, seed, episodes = evaluation["split"], evaluation["seed"], evaluation["episodes"]
    arrays = stratify_queries(generate(spec, split, 0, episodes, seed), spec, design["hops"], seed)
    synchronize(device)
    started = time.perf_counter()
    samples = ivd.evaluate_cells(model, arrays, depths=design["depths"][entry["variant"]], seed=seed,
                                 pool=spec.seen, batch_size=design["batch_size"], split=split, loads=design["loads"],
                                 candidates=design["candidates"])
    synchronize(device)
    seconds = time.perf_counter() - started
    require(state_hash(model) == before, f"model state changed during evaluation: {entry['name']}")
    require(digest_checkpoint(path) == entry["blake2b"], f"checkpoint changed during evaluation: {entry['name']}")
    timing = dict(name=entry["name"], writer=entry["writer"], variant=entry["variant"], seed=entry["seed"],
                  split=split, generator_seed=seed, episodes=episodes, queries=int(arrays.answer.size),
                  rows=int(len(samples["correct"])), cells=len(set(samples["condition"].tolist())), seconds=seconds)
    return samples, timing


def save_samples(samples: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **samples)
    return digest_checkpoint(path)


def smoke_report(design: dict, results: list[tuple[dict, dict]]) -> dict:
    """Pass/fail checks and the walltime suggestion from the smoke evaluations."""
    failures, overlaps = [], {}
    per_batch = {}
    smoke = design["smoke"]
    for samples, timing in results:
        key = (timing["writer"], timing["variant"])
        expected_rows = timing["episodes"] * len(design["hops"]) * len(design["depths"][timing["variant"]])
        if timing["variant"] == "M1":
            summary = {row["condition"]: row["overlap_mean"] for row in ivd.overlap_summary(samples)}
            overlaps[timing["writer"]] = summary
            for load in design["loads"]:
                low, nat, high = (summary[f"{load}-{kind}"] for kind in ivd.OVERLAPS)
                if not low < nat < high:
                    failures.append(f"{timing['writer']}: mean overlap not ordered low < nat < high at {load} "
                                    f"({low:.3f}, {nat:.3f}, {high:.3f})")
        if timing["rows"] != expected_rows * timing["cells"] or timing["cells"] != len(ivd.cell_names(timing["writer"])):
            failures.append(f"{timing['name']}: unexpected number of cells or rows")
        per_batch[key] = timing["seconds"] / math.ceil(timing["episodes"] / design["batch_size"])
    batches = math.ceil(design["confirmation"]["episodes"] / design["batch_size"])
    seeds = len(uns.SEEDS)
    main_seconds = sum(per_batch[key] * batches * seeds for key in per_batch)
    return dict(status="failed" if failures else "passed", failures=failures, smoke=smoke, overlaps=overlaps,
                design_hash=stable_hash(design), timings=[timing for _, timing in results],
                main_seconds_estimate=main_seconds,
                suggested_walltime_seconds=int(math.ceil(WALLTIME_MARGIN * main_seconds / 60.0) * 60),
                note="per-batch time of the smoke run includes start-up costs, so the estimate is conservative")


def runtime_metadata(device: str) -> dict:
    cuda = torch.device(device).type == "cuda"
    return dict(host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(device) if cuda else None, device=device, threads=1)


def stage_lock(args: argparse.Namespace) -> int:
    out = Path(args.out)
    require(out.is_absolute() and not out.exists(), "the lock file must be a new absolute path")
    design = build_design(args.p1s_root, args.f2_root)
    write_json(out, design)
    print(json.dumps(dict(stage="lock", checkpoints=len(design["checkpoints"]), design_hash=stable_hash(design),
                          out=str(out))), flush=True)
    return 0


def stage_smoke(args: argparse.Namespace) -> int:
    out = absolute_path(args.out, "--out")
    design_path = Path(args.design)
    design = json.loads(design_path.read_text())
    entries = [entry for entry in verify_design(design) if entry["seed"] == design["smoke"]["training_seed"]]
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for entry in entries:
        samples, timing = evaluate_checkpoint(entry, design, design["smoke"], device=args.device)
        results.append((samples, timing))
        print(json.dumps(dict(stage="smoke", name=entry["name"], seconds=timing["seconds"])), flush=True)
    report = dict(smoke_report(design, results), created=now(), design_path=str(design_path.resolve()),
                  runtime=runtime_metadata(args.device))
    write_json(out / "smoke.json", report)
    print(json.dumps(dict(stage="smoke", status=report["status"], failures=report["failures"],
                          suggested_walltime_seconds=report["suggested_walltime_seconds"])), flush=True)
    return 0 if report["status"] == "passed" else 1


def stage_run(args: argparse.Namespace) -> int:
    out = absolute_path(args.out, "--out")
    design_path = Path(args.design).resolve()
    verify_git_lock(design_path, args.design_commit, "the design")
    design = json.loads(design_path.read_text())
    entries = verify_design(design)
    smoke = json.loads((Path(args.smoke_dir) / "smoke.json").read_text())
    require(smoke["status"] == "passed" and smoke["design_hash"] == stable_hash(design),
            "the smoke report must have passed for exactly this design")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = dict(status="running", smoke=False, design=design, design_hash=stable_hash(design),
                    design_path=str(design_path), design_lock_commit=args.design_commit,
                    evaluation=design["confirmation"], started_at=now(), source=source_metadata(),
                    runtime=runtime_metadata(args.device))
    write_json(out / "manifest.json", manifest)
    runs = []
    for entry in entries:
        samples, timing = evaluate_checkpoint(entry, design, design["confirmation"], device=args.device)
        timing["samples_blake2b"] = save_samples(samples, out / "samples" / f"{entry['name']}.npz")
        runs.append(timing)
        print(json.dumps(dict(stage="run", name=entry["name"], seconds=timing["seconds"])), flush=True)
        del samples
    for entry in entries:
        require(digest_checkpoint(Path(entry["path"])) == entry["blake2b"], f"checkpoint changed: {entry['name']}")
    seconds = time.perf_counter() - started
    manifest.update(status="complete", runs=runs, finished_at=now(), total_seconds=seconds,
                    node_hours=seconds / 3600.0, checkpoints_unchanged=True)
    write_json(out / "manifest.json", manifest)
    print(json.dumps(dict(stage="run", status="complete", seconds=seconds, out=str(out))), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    lock = sub.add_parser("lock", help="write the design and checkpoint seals")
    lock.add_argument("--p1s-root", default=DEFAULT_P1S, help="P1-S main run root")
    lock.add_argument("--f2-root", default=DEFAULT_F2, help="P1-S-F2 root")
    lock.add_argument("--out", required=True, help="new design JSON (absolute path)")
    smoke = sub.add_parser("smoke", help="validation-split smoke run")
    smoke.add_argument("--design", required=True)
    smoke.add_argument("--out", required=True, help="new directory (absolute path)")
    run = sub.add_parser("run", help="confirmation evaluation")
    run.add_argument("--design", required=True, help="the committed design JSON in its repository")
    run.add_argument("--design-commit", required=True, help="full commit hash that locked the design")
    run.add_argument("--smoke-dir", required=True)
    run.add_argument("--out", required=True, help="new directory (absolute path)")
    for stage in (smoke, run):
        stage.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    require_compute()
    torch.set_num_threads(1)
    if args.stage != "lock" and torch.device(args.device).type == "cuda":
        torch.cuda.set_device(torch.device(args.device))
    return {"lock": stage_lock, "smoke": stage_smoke, "run": stage_run}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
