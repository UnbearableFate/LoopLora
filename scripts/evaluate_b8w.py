#!/usr/bin/env python3
"""B8W confirmation evaluation (docs/prereg/B8W.md); invoke on Miyabi-G only through scripts/submit.py.

One evaluation per run, primary setting only, at the trained depth (L = 4; B4 at L = 1), hops 1-8, on the registered window
test [108000, 110000) (named window `b8w`), eval seed 4242, 2000 episodes. The five groups are `start` (the P1-L M1-L4
checkpoint), `shared` and `untied` (the two continuations) and the references `B4` and `B8` (P1-L checkpoints), seeds 17/29/43.
`check_training` runs before any test episode is generated and checks the manifest, the committed lock, the registered prereg
and the authorization against the checkpoint files. No typed replication, swap, readout, depth scan or timing.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from audit_p4s_stopping import require_compute
from looplora.llm.config import recursive_merge, resolve_config
from looplora.llm.evaluate import evaluate
from looplora.llm.inject import load_model, load_tokenizer
from looplora.llm.train import load_checkpoint

from evaluate_p1l_main import check_run
from run_p1l_main import ROOT, SEEDS, digest, runtime, save

WINDOW = "b8w"
GROUPS = ("start", "shared", "untied", "B4", "B8")
NAMES = tuple(f"{label}-s{seed}" for label in GROUPS for seed in SEEDS)
PREREG = "docs/prereg/B8W.md"
LOCK = "reports/b8w/checkpoint-lock.json"
EXPECTED_VARIANT = {"start": "M1", "shared": "M1", "untied": "B8", "B4": "B4", "B8": "B8"}


def expected_hash(name: str, manifest: dict, lock: dict) -> str:
    """The checkpoint hash the study allows for a run: the lock's for P1-L checkpoints, the manifest's for continuations."""
    label, seed = name.rsplit("-s", 1)
    if label in {"shared", "untied"}:
        return manifest["jobs"][name]["checkpoint_hash"]
    return lock["checkpoints"][f"{'M1-L4' if label == 'start' else label}-s{seed}"]["hash"]


def check_training(manifest: dict, authorization: dict, *, check_all: bool = True) -> None:
    """Before any test episode is generated: the 15 runs, the committed lock, the registered prereg and the authorization
    agree. One gated contrast: k = 1, 95%, the named window `b8w`."""
    if manifest.get("study") != "b8w" or authorization.get("study") != "b8w":
        raise ValueError("a B8W manifest and a B8W authorization are required")
    if authorization.get("status") != "allowed" or not authorization.get("orchestrator_decision"):
        raise ValueError("confirmation is held until the orchestrator releases B8W")
    k = authorization.get("gated_conditions")
    if not isinstance(k, int) or isinstance(k, bool) or k != 1 or authorization.get("confidence") != .95:
        raise ValueError("B8W gates one contrast: k = 1, 95%")
    if authorization.get("confirmation_window") != WINDOW:
        raise ValueError(f"the authorization must name the {WINDOW} confirmation window")
    if authorization.get("prereg") != dict(path=PREREG, hash=digest(ROOT / PREREG)):
        raise ValueError("the authorization does not name the registered prereg")
    if authorization.get("lock") != dict(path=LOCK, hash=digest(ROOT / LOCK)) or manifest.get("lock") != authorization["lock"]:
        raise ValueError("the authorization and the manifest must name the committed checkpoint lock")
    lock = json.loads((ROOT / LOCK).read_text())
    if lock.get("status") != "locked" or lock.get("window") != WINDOW or lock.get("steps") != 2000:
        raise ValueError("the lock is not the registered B8W lock")
    if set(manifest["runs"]) != set(NAMES):
        raise ValueError("confirmation requires the 15 registered runs and no other run")
    if check_all:
        locked = Path(authorization["manifest"])
        if authorization.get("manifest_hash") != digest(locked) or json.loads(locked.read_text()) != manifest:
            raise ValueError("the training manifest is not the one the authorization locked")
    for name, path in manifest["runs"].items():
        label = name.rsplit("-s", 1)[0]
        cfg = check_run(name, path, 128)
        if digest(Path(path) / "checkpoint.pt") != expected_hash(name, manifest, lock):
            raise ValueError(f"{name}: the checkpoint is not the locked one")
        if cfg["variant"] != EXPECTED_VARIANT[label] or cfg["writer"] != "oracle_random" or cfg["oracle_seed"] != 17:
            raise ValueError(f"{name}: not the registered reader or oracle writer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=[])
    parser.add_argument("--check-only", action="store_true", help="run check_training and stop: no model, no episode")
    args = parser.parse_args()
    if not args.check_only and not args.runs:
        parser.error("--runs is required unless --check-only")
    require_compute()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    authorization = json.loads(args.authorization.read_text())
    check_training(manifest, authorization)  # before any test episode is generated
    if args.check_only:
        print(json.dumps(dict(status="checked", runs=len(manifest["runs"]), window=authorization["confirmation_window"])))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(), load_tokenizer()
    for name in args.runs:
        if name not in manifest["runs"]:
            raise ValueError(f"unknown run: {name}")
        run = Path(manifest["runs"][name])
        experiment = load_checkpoint(run / "checkpoint.pt", model=model, tokenizer=tokenizer)
        out = args.out / name
        out.mkdir()
        start = time.perf_counter()
        try:
            template = json.loads((ROOT / "configs/p1l/main/eval-confirmation.json").read_text())
            template = recursive_merge(template, {"eval": {"confirmation": WINDOW}})
            cfg = resolve_config(recursive_merge(copy.deepcopy(experiment.config), template))
            metrics, timing = evaluate(experiment, cfg, out / "primary")
            save(out / "primary" / "timing.json", timing)
            save(out / "primary" / "config.json", cfg)
            save(out / "record.json", dict(status="complete", name=name, checkpoint_hash=digest(run / "checkpoint.pt"),
                seconds=time.perf_counter() - start, authorization_hash=digest(args.authorization),
                manifest_hash=digest(args.manifest), **runtime()))
        finally:
            experiment.close()


if __name__ == "__main__":
    main()
