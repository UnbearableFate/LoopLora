#!/usr/bin/env python3
"""UNS Part B (docs/prereg/UNS.md): the M1 trained at L = 4 on the unseen entity pool.

The committed unseen-pool rows (Part A, scripts/audit_uns.py) cover the released M1 (mixed-depth
training) and B4. The P2-S-F1 readers M1-L4 (trained at the fixed depth 4) were never evaluated on the
unseen pool. This script evaluates them, and the B4 references, with the screening protocol on new
episodes: splits test (seen pool) and test_unseen, eval seed 999006, 2000 episodes, hops 1-8, depths 0-16.
Nothing is trained and no checkpoint is changed.

Stages (each its own job; scripts/submit.py in a Miyabi-G allocation):

  lock  write the design (settings and the blake2b-sealed list of 18 checkpoints) to a new JSON file;
  run   evaluate the checkpoints (all of them, or the ``--names`` subset for a timing run); requires the
        committed design (Git lock). A subset run is recorded as partial and cannot be audited.

The analysis is ``scripts/audit_uns.py --extension-root <root>``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from audit_uns import DEFAULT_F2, DEFAULT_P1S
from s9_common import digest_checkpoint, now, read_csv, require, require_compute, verify_git_lock, write_json
from s9_torch import load_frozen, state_hash, synchronize
from looplora.infra.seeds import stable_hash
from looplora.sym import uns
from looplora.sym.config import resolve_config
from looplora.sym.evaluate import evaluate
from looplora.sym.run import source_metadata

STUDY = "UNS"
WRITERS = (*uns.WRITERS, uns.CONTROL)
READERS = ("M1", "B4")
SEEDS = uns.SEEDS
RANK = 32
TRAINED_DEPTHS = {"M1": [4], "B4": [1]}  # the P2-S-F1 M1-L4 recipe; B4 is a single read
P2SF1 = "reports/p2s-f1/checkpoint_manifest.csv"
BOOTSTRAP = dict(resamples=2000, seed=uns.BOOTSTRAP_SEED, confidence=0.95)
DEFAULT_REPO = Path(__file__).resolve().parents[1]


def settings() -> dict:
    return dict(study=STUDY, prereg="docs/prereg/UNS.md", part="B", extension=copy.deepcopy(uns.EXTENSION),
                writers=list(WRITERS), readers=list(READERS), seeds=list(SEEDS), rank=RANK,
                trained_depths={reader: list(depths) for reader, depths in TRAINED_DEPTHS.items()},
                p2sf1_manifest=P2SF1, bootstrap=dict(BOOTSTRAP))


def build_design(p1s_root: str | Path, f2_root: str | Path, repo: str | Path = DEFAULT_REPO) -> dict:
    """Settings plus the sealed checkpoints: M1-L4 from the committed P2-S-F1 manifest, B4 from P1-S and F2."""
    manifest_path = Path(repo) / P2SF1
    rows = {(row["writer"], int(row["seed"])): row for row in read_csv(manifest_path)
            if row["source"] == "p2s-f1" and row["variant"] == "M1" and row["r"] == str(RANK)
            and row["r_mem"] == str(RANK)}
    entries = []
    for writer in WRITERS:
        for seed in SEEDS:
            for reader in READERS:
                if reader == "M1":
                    row = rows.get((writer, seed))
                    require(row is not None, f"the P2-S-F1 manifest lists no M1-L4 run for {writer}/s{seed}")
                    path = Path(row["checkpoint"]).resolve()
                    require(path.is_file(), f"missing checkpoint {path}")
                    require(digest_checkpoint(path) == row["checkpoint_hash"],
                            f"checkpoint {path.parent.name} differs from the P2-S-F1 manifest")
                else:
                    path = (uns.run_root(writer, reader, seed, p1s_root=p1s_root, f2_root=f2_root)
                            / "checkpoint.pt").resolve()
                    require(path.is_file(), f"missing checkpoint {path}")
                config = torch.load(path, map_location="cpu", weights_only=True)["config"]
                identity = (config["writer"], config["variant"], config["seed"], config["model"]["r"])
                require(identity == (writer, reader, seed, RANK)
                        and list(config["train"]["depths"]) == TRAINED_DEPTHS[reader],
                        f"unexpected checkpoint {path}: {identity}, training depths {config['train']['depths']}")
                entries.append(dict(name=uns.extension_name(writer, reader, seed), writer=writer, variant=reader,
                                    r=RANK, seed=seed, trained_depths=list(TRAINED_DEPTHS[reader]), path=str(path),
                                    blake2b=digest_checkpoint(path)))
    return {**settings(), "checkpoints": entries,
            "source": dict(path=str(manifest_path), blake2b=digest_checkpoint(manifest_path))}


def verify_design(design: dict) -> list[dict]:
    """The design must be exactly the registered settings and an intact set of the 18 checkpoints."""
    require({key: value for key, value in design.items() if key not in ("checkpoints", "source")} == settings(),
            "the design differs from the registered settings")
    entries = design["checkpoints"]
    keys = [(entry["writer"], entry["variant"], entry["seed"]) for entry in entries]
    expected = {(writer, reader, seed) for writer in WRITERS for reader in READERS for seed in SEEDS}
    require(len(keys) == len(set(keys)) and set(keys) == expected and all(entry["r"] == RANK for entry in entries),
            "the design must list exactly the registered checkpoints")
    require(all(entry["name"] == uns.extension_name(entry["writer"], entry["variant"], entry["seed"])
                for entry in entries), "a checkpoint name differs from the registered naming")
    for entry in entries:
        require(digest_checkpoint(Path(entry["path"])) == entry["blake2b"], f"checkpoint changed: {entry['name']}")
    return entries


def evaluation_config(model, design: dict, device: str) -> dict:
    """The checkpoint's configuration with the registered evaluation block (screening protocol, new seed)."""
    extension = design["extension"]
    config = copy.deepcopy(model.config)
    config["device"] = device
    config["eval"].update(splits=list(extension["splits"]), seed=extension["seed"], episodes=extension["episodes"],
                          depths=list(extension["depths"]), hops=list(extension["hops"]), counterfactuals=False,
                          swap_successors=False, fixed_depth_sweep=True, batch_size=extension["batch_size"])
    return resolve_config(config)


def evaluate_entry(entry: dict, design: dict, root: Path, *, device: str) -> dict:
    """Evaluate one frozen checkpoint into ``root/evaluation/<name>``; the record holds its seals."""
    path = Path(entry["path"])
    require(digest_checkpoint(path) == entry["blake2b"], f"checkpoint changed: {entry['name']}")
    model = load_frozen(entry, device)
    before = state_hash(model)
    config = evaluation_config(model, design, device)
    out = root / "evaluation" / entry["name"]
    synchronize(device)
    started = time.perf_counter()
    evaluate(model, config, out)
    synchronize(device)
    seconds = time.perf_counter() - started
    require(state_hash(model) == before, f"model state changed during evaluation: {entry['name']}")
    require(digest_checkpoint(path) == entry["blake2b"], f"checkpoint changed during evaluation: {entry['name']}")
    return dict(name=entry["name"], writer=entry["writer"], variant=entry["variant"], seed=entry["seed"],
                directory=str(out), seconds=seconds, config_hash=stable_hash(config),
                samples_blake2b=digest_checkpoint(out / "samples.npz"))


def runtime_metadata(device: str) -> dict:
    cuda = torch.device(device).type == "cuda"
    return dict(host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(device) if cuda else None, device=device, threads=1)


def stage_lock(args: argparse.Namespace) -> int:
    out = Path(args.out)
    require(out.is_absolute() and not out.exists(), "the lock file must be a new absolute path")
    design = build_design(args.p1s_root, args.f2_root, args.repo)
    write_json(out, design)
    print(json.dumps(dict(stage="lock", checkpoints=len(design["checkpoints"]), design_hash=stable_hash(design),
                          out=str(out))), flush=True)
    return 0


def stage_run(args: argparse.Namespace) -> int:
    design_path = Path(args.design).resolve()
    verify_git_lock(design_path, args.design_commit, "the design")
    design = json.loads(design_path.read_text())
    entries = verify_design(design)
    names = None if args.names is None else [name for name in args.names.split(",") if name]
    if names is not None:
        require(names and set(names) <= {entry["name"] for entry in entries}, "--names holds an unregistered name")
        entries = [entry for entry in entries if entry["name"] in names]
    root = Path(args.root)
    require(root.is_absolute() and not any("confirmation" in part.lower() for part in root.parts),
            "the root must be an absolute path without a 'confirmation' component")
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    manifest = dict(status="running", partial=names is not None, design=design, design_hash=stable_hash(design),
                    design_path=str(design_path), design_lock_commit=args.design_commit, started_at=now(),
                    source=source_metadata(), runtime=runtime_metadata(args.device), runs=[])
    write_json(root / "manifest.json", manifest)
    for entry in entries:
        record = evaluate_entry(entry, design, root, device=args.device)
        manifest["runs"].append(record)
        write_json(root / "manifest.json", manifest)
        print(json.dumps(dict(stage="run", name=entry["name"], seconds=record["seconds"])), flush=True)
    seconds = time.perf_counter() - started
    manifest.update(status="partial" if names is not None else "complete", finished_at=now(),
                    total_seconds=seconds, node_hours=seconds / 3600.0, checkpoints_unchanged=True)
    write_json(root / "manifest.json", manifest)
    print(json.dumps(dict(stage="run", status=manifest["status"], seconds=seconds, node_hours=manifest["node_hours"])),
          flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="stage", required=True)
    lock = commands.add_parser("lock", help="write the design JSON")
    lock.add_argument("--p1s-root", default=DEFAULT_P1S)
    lock.add_argument("--f2-root", default=DEFAULT_F2)
    lock.add_argument("--repo", default=str(DEFAULT_REPO), help="repository holding the committed P2-S-F1 manifest")
    lock.add_argument("--out", required=True, help="new design file (absolute path)")
    run = commands.add_parser("run", help="evaluate the checkpoints of a committed design")
    run.add_argument("--design", required=True)
    run.add_argument("--design-commit", required=True, help="full commit hash that holds the design file")
    run.add_argument("--root", required=True, help="new output root (absolute path)")
    run.add_argument("--names", help="comma-separated checkpoint names (timing run); default all")
    run.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    require_compute()  # the lock stage hashes and loads 18 checkpoints, so it also belongs on a compute node
    return stage_lock(args) if args.stage == "lock" else stage_run(args)


if __name__ == "__main__":
    sys.exit(main())
