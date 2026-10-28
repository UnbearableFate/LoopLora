#!/usr/bin/env python3
"""P5L trainings on Qwen3-1.7B-Base (docs/prereg/P5L.md); invoke on Miyabi-G only through scripts/submit.py.

`smoke` is the timing and memory smoke (no score is computed or exposed). `main` is the registered training of one
seed's readers and refuses unless the committed lock reports/p5l/r128/lr-S-lock.json exists, is `locked` and agrees with
the configs. It reuses P1-L's `fit` (training, record, checkpoint) unchanged; nothing here touches the P1-L path.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from run_p1l_main import ROOT, fit, runtime, save

LABELS = ("B8", "M1-L4", "B4", "B5-L4", "B3-L4")
SEEDS = (17, 29, 43)
CONFIGS = "configs/p5l/main-r128"
LOCK = "reports/p5l/r128/lr-S-lock.json"
PREREG = "docs/prereg/P5L.md"


def config(label: str, seed: int) -> dict:
    return json.loads((ROOT / CONFIGS / f"{label}-s{seed}.json").read_text())


def backbone_of(labels, seed: int) -> dict:
    """The one backbone every selected config names (the models are loaded once and reused sequentially)."""
    pins = {json.dumps(config(label, seed)["backbone"], sort_keys=True) for label in labels}
    if len(pins) != 1:
        raise ValueError("the selected P5L configs name different backbones")
    return json.loads(pins.pop())


def check_lock(lock: dict, labels, seed: int) -> None:
    """The committed lock must agree with the configs it governs (S and lr are carried from P1-L's r = 128 lock)."""
    if lock.get("status") != "locked" or lock.get("rank") != 128:
        raise ValueError("main requires a committed, locked r = 128 lock")
    for label in labels:
        cfg = config(label, seed)
        if cfg["train"]["steps"] != lock["S"] or cfg["train"]["lr"] != lock["lrs"][label]:
            raise ValueError(f"{label}: the config disagrees with the lock (S or lr)")


def train_stage(args) -> None:
    import torch
    from audit_p4s_stopping import require_compute
    from looplora.infra.seeds import derive_seed
    from looplora.llm.inject import load_model, load_tokenizer

    require_compute()
    torch.set_num_threads(1)
    labels = args.labels or LABELS
    backbone = backbone_of(labels, args.seed)
    if args.stage == "main":
        if not (ROOT / PREREG).is_file():
            raise ValueError("main requires the registered prereg in the source tree")
        check_lock(json.loads((ROOT / LOCK).read_text()), labels, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(backbone=backbone), load_tokenizer(backbone)
    records = []
    for label in labels:
        cfg = copy.deepcopy(config(label, args.seed))
        if args.stage == "smoke":
            cfg["seed"] = derive_seed(20261029, "p5l", "timing-smoke")
            cfg["smoke"] = True
            cfg["train"]["steps"] = args.steps
            cfg["train"]["log_every"] = 1
            cfg["eval"]["episodes"] = 100
        record = fit(cfg, args.out / f"{label}-s{args.seed}", model, tokenizer, expose_scores=False)
        records.append(record)
        safe = {k: record[k] for k in ("train_seconds", "mean_step_seconds", "steady_step_seconds", "peak_memory_bytes")}
        print(json.dumps(dict(run=f"{label}-s{args.seed}", **safe)), flush=True)
        save(args.out / "stage.json", dict(stage=args.stage, status="running", runs=records, **runtime()))
    save(args.out / "stage.json", dict(stage=args.stage, status="complete", runs=records, **runtime()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "main"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=17)
    parser.add_argument("--labels", nargs="+", choices=LABELS)
    args = parser.parse_args()
    if args.stage == "smoke" and (args.steps is None or args.steps < 10):
        parser.error("smoke requires --steps >= 10")
    train_stage(args)


if __name__ == "__main__":
    main()
