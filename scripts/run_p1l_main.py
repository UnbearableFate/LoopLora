#!/usr/bin/env python3
"""Staged P1-L runs; invoke on Miyabi-G only through scripts/submit.py."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("B8", "M1-L4", "B4", "B5-L4", "B3-L4", "B1", "B2")


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config(label: str, seed: int = 17) -> dict:
    return json.loads((ROOT / "configs/p1l/main" / f"{label}-s{seed}.json").read_text())


def runtime() -> dict:
    return dict(host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"),
                time=datetime.now(timezone.utc).isoformat())


def fit(cfg: dict, out: Path, model, tokenizer, *, expose_scores: bool) -> dict:
    import torch
    from looplora.llm.config import resolve_config
    from looplora.llm.train import train
    from looplora.sym.evaluate import tensor_hash
    from looplora.infra.seeds import stable_hash
    from looplora.sym.run import source_metadata

    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg = resolve_config(cfg)
    save(out / "config.json", cfg)
    save(out / "record.json", dict(status="running", config=cfg, **runtime(), **source_metadata()))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    experiment = None
    try:
        experiment, summary = train(cfg, out, model=model, tokenizer=tokenizer)
        record = dict(status="complete", config=cfg, **runtime(), **source_metadata(),
                      train_seconds=summary["train_seconds"],
                      mean_step_seconds=summary["train_step_seconds"],
                      p95_step_seconds=summary.get("train_step_p95_seconds"),
                      peak_memory_bytes=torch.cuda.max_memory_allocated(),
                      checkpoint_hash=digest(out / "checkpoint.pt"),
                      writer_hash=stable_hash({k: tensor_hash(v) for k, v in experiment.writer_state().items()}),
                      writer_signature=experiment.writer_signature(),
                      writer_frozen=summary["writer_frozen"],
                      trainable_parameters=summary["trainable_parameters"])
        # Timing smokes retain raw training logs, but expose no quality fields.
        if expose_scores:
            from looplora.llm.evaluate import evaluate

            metrics, timing = evaluate(experiment, cfg, out / "validation")
            record["validation"] = metrics["table"]
            record["validation_seconds"] = timing["evaluation_seconds"]
        record["seconds"] = time.perf_counter() - started
        save(out / "record.json", record)
        return record
    except BaseException as error:
        save(out / "failure.json", dict(error=f"{type(error).__name__}: {error}", **runtime()))
        raise
    finally:
        if experiment is not None:
            experiment.close()


def train_stage(args) -> None:
    import torch
    from audit_p4s_stopping import require_compute
    from looplora.infra.seeds import derive_seed
    from looplora.llm.inject import load_model, load_tokenizer

    require_compute()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(), load_tokenizer()
    records = []
    selected = args.labels or LABELS
    for label in selected:
        base = config(label, args.seed)
        if args.stage in {"preflight", "smoke"}:
            base["seed"] = derive_seed(20261027, "p1l-main", "timing-smoke")
            base["smoke"] = True
            base["train"]["steps"] = args.steps
            base["train"]["log_every"] = max(1, args.steps // 10)
            base["eval"]["episodes"] = 100
        elif args.stage == "pilot":
            if args.steps is None:
                raise ValueError("pilot requires the smoke-locked S/3 steps")
            base["train"]["steps"] = args.steps
            base["eval"].update(episodes=1000, hops=[1, 2, 3], counterfactuals=False)
        elif args.stage == "main":
            lock = json.loads((ROOT / "reports/p1l/lr-S-lock.json").read_text())
            if lock["status"] != "locked":
                raise ValueError("main requires a committed lr/S lock")
            base["train"]["steps"] = lock["S"]
            base["train"]["lr"] = lock["lrs"][label]
        rates = [0.001, 0.003] if args.stage == "pilot" else [base["train"]["lr"]]
        for rate in rates:
            cfg = copy.deepcopy(base)
            cfg["train"]["lr"] = rate
            name = f"{label}-lr{rate:g}" if args.stage == "pilot" else f"{label}-s{args.seed}"
            record = fit(cfg, args.out / name, model, tokenizer, expose_scores=args.stage == "pilot")
            records.append(record)
            safe = {k: record[k] for k in ("train_seconds", "mean_step_seconds", "peak_memory_bytes")}
            print(json.dumps(dict(run=name, **safe)), flush=True)
            save(args.out / "stage.json", dict(stage=args.stage, status="running", runs=records, **runtime()))
    save(args.out / "stage.json", dict(stage=args.stage, status="complete", runs=records, **runtime()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "smoke", "pilot", "main"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--labels", nargs="+", choices=LABELS)
    args = parser.parse_args()
    if args.stage in {"preflight", "smoke"} and (args.steps is None or args.steps < 10):
        parser.error("timing stages require --steps >= 10")
    train_stage(args)


if __name__ == "__main__":
    main()
