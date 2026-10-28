#!/usr/bin/env python3
"""P5L confirmation evaluation on Qwen3-1.7B-Base (docs/prereg/P5L.md); invoke on Miyabi-G only through scripts/submit.py.

Same per-run evaluation as evaluate_p1l_main (primary and typed settings, swap_successors for M1-L4 and B8, inference
timing), on the registered window test [104000, 106000). `check_training` runs before any test episode is generated and
checks the whole matrix against the committed lock, the registered prereg and the authorization. The backbone comes from
the training manifest (it must be the registered one and the one every run's config names). No readout (J-lens) here.
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

from evaluate_p1l_main import check_run, inference_timing, swap_successors
from run_p1l_main import ROOT, SEEDS, digest, runtime, save
from run_p5l import BASELINE, LABELS, LOCK, PREREG

WINDOW = "p5l"
CONDITION = "p5l-r128"
BACKBONE = {"id": "Qwen/Qwen3-1.7B-Base", "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"}
NAMES = tuple(f"{label}-s{seed}" for label in LABELS for seed in SEEDS) + (f"{BASELINE}-s17",)


def check_training(manifest: dict, authorization: dict, *, check_all: bool = True) -> None:
    """Before any test episode is generated: the 15 trained runs and B0, the committed lock, the registered prereg and the
    authorization agree. One condition at r = 128, k = 2 (97.5%), the named window `p5l`."""
    if manifest.get("study") != "p5l" or authorization.get("study") != "p5l":
        raise ValueError("a P5L manifest and a P5L authorization are required")
    if authorization.get("status") != "allowed" or not authorization.get("orchestrator_decision"):
        raise ValueError("confirmation is held until the orchestrator releases P5L")
    conditions, k = authorization.get("condition_manifests", {}), authorization.get("gated_conditions")
    if (not isinstance(k, int) or isinstance(k, bool) or k != 2 or authorization.get("confidence") != .975
            or set(conditions) != {CONDITION}):
        raise ValueError("P5L gates one condition with two contrasts: k = 2, 97.5%")
    if authorization.get("confirmation_window") != WINDOW:
        raise ValueError(f"the authorization must name the {WINDOW} confirmation window")
    if authorization.get("prereg") != dict(path=PREREG, hash=digest(ROOT / PREREG)):
        raise ValueError("the authorization does not name the registered prereg")
    if authorization.get("lock") != dict(path=LOCK, hash=digest(ROOT / LOCK)):
        raise ValueError("the authorization does not name the committed lr/S lock")
    lock = json.loads((ROOT / LOCK).read_text())
    if (lock.get("status") != "locked" or lock.get("rank") != 128 or lock.get("backbone") != BACKBONE
            or (manifest.get("S"), manifest.get("lrs")) != (lock["S"], lock["lrs"])):
        raise ValueError("the manifest differs from the committed lock")
    if (manifest.get("condition"), manifest.get("rank"), manifest.get("backbone")) != (CONDITION, 128, BACKBONE):
        raise ValueError("the manifest is not the registered P5L condition")
    if set(manifest["runs"]) != set(NAMES):
        raise ValueError("confirmation requires the 15 trained runs and B0-s17, and no other run")
    if check_all:
        locked = Path(conditions[CONDITION])
        if authorization.get("manifest_hash") != digest(locked) or json.loads(locked.read_text()) != manifest:
            raise ValueError("the training manifest is not the one the authorization locked")
    for name, path in manifest["runs"].items():
        cfg = check_run(name, path, 128)
        if cfg.get("backbone") != BACKBONE:
            raise ValueError(f"{name} was not trained on the registered backbone")
        if name == f"{BASELINE}-s17":
            if cfg["variant"] != BASELINE:
                raise ValueError("B0-s17 is not the baseline")
            continue
        if cfg["writer"] != "oracle_random" or cfg["oracle_seed"] != 17:
            raise ValueError(f"{name} was not trained with the registered oracle writer")
        if cfg["train"]["steps"] != lock["S"] or cfg["train"]["lr"] != lock["lrs"][name.rsplit("-s", 1)[0]]:
            raise ValueError("main training differs from its committed S/lr lock")


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
    model, tokenizer = load_model(backbone=manifest["backbone"]), load_tokenizer(manifest["backbone"])
    for name in args.runs:
        if name not in manifest["runs"]:
            raise ValueError(f"unknown main run: {name}")
        run = Path(manifest["runs"][name])
        experiment = load_checkpoint(run / "checkpoint.pt", model=model, tokenizer=tokenizer)
        out = args.out / name
        out.mkdir()
        start = time.perf_counter()
        try:
            template = json.loads((ROOT / "configs/p1l/main/eval-confirmation.json").read_text())
            template = recursive_merge(template, {"eval": {"confirmation": WINDOW}})
            cfg = resolve_config(recursive_merge(copy.deepcopy(experiment.config), template))
            if cfg["backbone"] != manifest["backbone"]:
                raise ValueError(f"{name}: the checkpoint names another backbone than the manifest")
            if cfg["variant"] == "M1":
                cfg["eval"]["depths"] = list(range(1, 9))
            for task in ("primary", "typed"):
                current = cfg
                if task == "typed":
                    typed = json.loads((ROOT / "configs/p1l/main/eval-typed-replication.json").read_text())
                    current = resolve_config(recursive_merge(cfg, typed))
                metrics, timing = evaluate(experiment, current, out / task)
                save(out / task / "timing.json", timing)
                save(out / task / "config.json", current)
            if cfg["variant"] in {"M1", "B8"}:
                swap_successors(experiment, cfg, out)
            inference_timing(experiment, cfg, out)
            save(out / "record.json", dict(status="complete", name=name,
                checkpoint_hash=digest(run / "checkpoint.pt"), seconds=time.perf_counter() - start,
                authorization_hash=digest(args.authorization), manifest_hash=digest(args.manifest), **runtime()))
        finally:
            experiment.close()


if __name__ == "__main__":
    main()
