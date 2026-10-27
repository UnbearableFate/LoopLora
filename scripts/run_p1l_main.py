#!/usr/bin/env python3
"""Staged P1-L runs; invoke on Miyabi-G only through scripts/submit.py."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("B8", "M1-L4", "B4", "B5-L4", "B3-L4", "B1", "B2")
SEEDS = (17, 29, 43)
# Study profiles. "p1l" is every script's default and holds P1-L's literals. "p1l-s" (docs/prereg/P1L-S.md) runs at
# r = 32 only: B8, M1-L4, B4 and B5-L4 are retrained under the WP-L6 oracle code W; B3-L4 is P1-L's r = 32 run, reused.
STUDIES = {
    "p1l": dict(configs="configs/p1l", reports="reports/p1l", prereg="docs/prereg/P1L.md", prefix="p1l", reused=()),
    "p1l-s": dict(configs="configs/p1l-s", reports="reports/p1l-s", prereg="docs/prereg/P1L-S.md", prefix="ps",
                  labels=("B8", "M1-L4", "B4", "B5-L4", "B3-L4"), reused=("B3-L4",),
                  condition="p1l-s-r32", window="p1l-s", decision="reports/wp-l6/decision.json"),
}


def labels_for_rank(rank: int, study: str = "p1l"):
    """The reader labels of a study's training matrix, reused runs included (B0 is separate)."""
    if study != "p1l":
        return STUDIES[study]["labels"]
    return LABELS if rank == 32 else LABELS[:5]


def trained_labels(rank: int, study: str = "p1l"):
    """The labels a study trains itself: its matrix without the runs it reuses."""
    return tuple(label for label in labels_for_rank(rank, study) if label not in STUDIES[study]["reused"])


def lock_file(rank: int = 32, study: str = "p1l") -> str:
    """Repository-relative path of a study's committed S/lr lock."""
    reports = STUDIES[study]["reports"]
    return f"{reports}/lr-S-lock.json" if rank == 32 else f"{reports}/r128/lr-S-lock.json"


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


def config(label: str, seed: int = 17, rank: int = 32, study: str = "p1l") -> dict:
    directory = "main" if rank == 32 else "main-r128"
    return json.loads((ROOT / STUDIES[study]["configs"] / directory / f"{label}-s{seed}.json").read_text())


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
    save(out / "record.json", dict(status="running", config=cfg, config_hash=stable_hash(cfg),
                                   **runtime(), **source_metadata()))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    experiment = None
    try:
        experiment, summary = train(cfg, out, model=model, tokenizer=tokenizer)
        # Never inspect the timing-smoke loss/accuracy fields.
        with (out / "train_log.jsonl").open() as stream:
            durations = [row["step_seconds"] for line in stream
                         if (row := json.loads(line))["step"] > 5]
        record = dict(status="complete", config=cfg, config_hash=stable_hash(cfg),
                      **runtime(), **source_metadata(),
                      train_seconds=summary["train_seconds"],
                      mean_step_seconds=summary["train_step_seconds"],
                      steady_step_seconds=statistics.mean(durations) if durations else None,
                      p95_step_seconds=summary.get("train_step_p95_seconds"),
                      peak_memory_bytes=torch.cuda.max_memory_allocated(),
                      checkpoint_hash=digest(out / "checkpoint.pt"),
                      writer_hash=stable_hash({k: tensor_hash(v) for k, v in experiment.writer_state().items()}),
                      writer_signature=experiment.writer_signature(),
                      writer_frozen=summary["writer_frozen"],
                      trainable_parameters=summary["trainable_parameters"])
        if cfg["writer"] in ("oracle_rot", "oracle_shared"):
            # WP-L6 codes only (P1-L records keep their fields): oracle_shared's table C counts as a writer
            # parameter, and its drift from the initialization is recorded.
            from looplora.llm.run import _parameter_counts

            record.update(writer_trainable_parameters=summary.get("writer_trainable_parameters", 0),
                          parameters=_parameter_counts(experiment))
            if cfg["writer"] == "oracle_shared":
                from run_wp_l6 import code_drift

                record["code_drift"] = code_drift(experiment.writer, tuple(cfg["task"]["seen"]))
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
    selected = args.labels or trained_labels(args.rank, args.study)
    for label in selected:
        base = config(label, args.seed, args.rank, args.study)
        if args.stage in {"preflight", "smoke"}:
            base["seed"] = derive_seed(20261027, "p1l-main", "timing-smoke")
            base["smoke"] = True
            base["train"]["steps"] = args.steps
            base["train"]["log_every"] = 1
            base["eval"]["episodes"] = 100
        elif args.stage == "pilot":
            if args.steps is None:
                raise ValueError("pilot requires the smoke-locked S/3 steps")
            base["train"]["steps"] = args.steps
            base["eval"].update(episodes=1000, hops=[1, 2, 3], counterfactuals=False)
        elif args.stage == "main":
            lock = json.loads((ROOT / lock_file(args.rank, args.study)).read_text())
            if lock["status"] != "locked":
                raise ValueError("main requires a committed lr/S lock")
            base["train"]["steps"] = lock["S"]
            base["train"]["lr"] = lock["lrs"].get(label, base["train"]["lr"])
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
    parser.add_argument("--rank", type=int, choices=(32, 128), default=32)
    parser.add_argument("--labels", nargs="+", choices=(*LABELS, "B0"))
    parser.add_argument("--study", choices=tuple(STUDIES), default="p1l")
    args = parser.parse_args()
    if args.stage in {"preflight", "smoke"} and (args.steps is None or args.steps < 10):
        parser.error("timing stages require --steps >= 10")
    if args.study != "p1l" and (args.rank != 32 or not set(args.labels or ()) <= set(trained_labels(32, args.study))):
        parser.error(f"{args.study} trains {', '.join(trained_labels(32, args.study))} at r = 32 only")
    train_stage(args)


if __name__ == "__main__":
    main()
