#!/usr/bin/env python3
"""WP-L6 pilot: oracle code designs O-rot and O-shared at r = 32 (docs/prereg/WP-L6.md).

`smoke` and `pilot` train and evaluate on Miyabi-G and run only through scripts/submit.py. `size` and `decide` are
stdlib-only control-plane steps (login-safe). `decide` reads one cell per arm: hop-1 validation full-vocabulary EM
at L = 4 of the final checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from run_p1l_main import digest, runtime, save

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("oracle_rot", "oracle_shared")  # selection order: the fixed code first
STEPS = 5666  # floor(17000 / 3): P1-L's Pilot-2 length
SMOKE_STEPS = math.ceil(STEPS / 10)  # 10% time-only smoke
THRESHOLD = 0.30  # hop-1 val full EM at L = 4, final checkpoint
RESERVE = 90.  # per job: backbone load, checkpoint export and hashing, shutdown
PACK_LIMIT = 3600  # one node x 1 h without --scale-up


def config(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return json.loads((ROOT / "configs" / "wp_l6" / f"{arm}.json").read_text())


def code_drift(writer, seen: tuple[int, int]) -> dict:
    """How far oracle_shared's trained table C moved from its initialization (the global codes), by entity pool."""
    import torch
    from torch.nn import functional as F

    with torch.no_grad():
        table, init = writer.C.detach()[:writer.V], writer.global_codes[:writer.V]
        cosine = F.cosine_similarity(table, init, dim=-1)
        changed = (table != init).any(-1)
    pools = {"seen": slice(*seen), "unseen": slice(seen[1], writer.V)}
    return {name: dict(rows=int(cosine[rows].numel()), mean_cosine_to_init=float(cosine[rows].mean()),
                       min_cosine_to_init=float(cosine[rows].min()), rows_changed=int(changed[rows].sum()))
            for name, rows in pools.items()}


def fit(cfg: dict, out: Path, model, tokenizer, *, evaluation: str) -> dict:
    """Train one arm. evaluation: "scores" keeps the validation table; "timing" evaluates but keeps only its time."""
    import torch
    from looplora.infra.seeds import stable_hash
    from looplora.llm.config import resolve_config
    from looplora.llm.evaluate import evaluate
    from looplora.llm.run import _parameter_counts
    from looplora.llm.train import train
    from looplora.sym.evaluate import tensor_hash
    from looplora.sym.run import source_metadata

    cfg = resolve_config(cfg)
    out.mkdir(parents=True)
    save(out / "config.json", cfg)
    save(out / "record.json", dict(status="running", config=cfg, config_hash=stable_hash(cfg),
                                   **runtime(), **source_metadata()))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    experiment = None
    try:
        experiment, summary = train(cfg, out, model=model, tokenizer=tokenizer)
        with (out / "train_log.jsonl").open() as stream:
            durations = [row["step_seconds"] for line in stream if (row := json.loads(line))["step"] > 5]
        record = dict(status="complete", config=cfg, config_hash=stable_hash(cfg), **runtime(), **source_metadata(),
                      train_seconds=summary["train_seconds"], mean_step_seconds=summary["train_step_seconds"],
                      steady_step_seconds=statistics.mean(durations) if durations else None,
                      p95_step_seconds=summary.get("train_step_p95_seconds"),
                      peak_memory_bytes=torch.cuda.max_memory_allocated(),
                      checkpoint_hash=digest(out / "checkpoint.pt"),
                      writer_hash=stable_hash({k: tensor_hash(v) for k, v in experiment.writer_state().items()}),
                      writer_signature=experiment.writer_signature(), writer_frozen=summary["writer_frozen"],
                      trainable_parameters=summary["trainable_parameters"],
                      writer_trainable_parameters=summary.get("writer_trainable_parameters", 0),
                      parameters=_parameter_counts(experiment))
        if cfg["writer"] == "oracle_shared":
            record["code_drift"] = code_drift(experiment.writer, tuple(cfg["task"]["seen"]))
        if evaluation != "none":
            metrics, timing = evaluate(experiment, cfg, out / "validation")
            record["validation_seconds"] = timing["evaluation_seconds"]
            if evaluation == "scores":
                record["validation"] = metrics["table"]
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
    for arm in args.arms:
        cfg = config(arm)
        if args.stage == "smoke":
            # Timing only: a derived seed and a derived validation seed, so no registered episode is used; the
            # scores of this stage are never read.
            cfg["seed"] = derive_seed(20261027, "wp-l6", "timing-smoke")
            cfg["smoke"] = True
            cfg["train"]["steps"] = SMOKE_STEPS
            cfg["train"]["log_every"] = 1
            del cfg["eval"]["seed"]
        try:
            record = fit(cfg, args.out / arm, model, tokenizer,
                         evaluation="scores" if args.stage == "pilot" else "timing")
        except (FloatingPointError, RuntimeError) as error:
            if not diverged(f"{type(error).__name__}: {error}"):
                raise
            # fit() saved failure.json. A diverged arm is a registered outcome (it fails the pass criterion), and
            # the other arm must still run.
            print(json.dumps(dict(run=arm, failure=f"{type(error).__name__}: {error}")), flush=True)
            continue
        records.append(record)
        print(json.dumps(dict(run=arm, **{key: record[key] for key in
                                          ("train_seconds", "mean_step_seconds", "validation_seconds",
                                           "peak_memory_bytes")})), flush=True)
        save(args.out / "stage.json", dict(stage=args.stage, status="running", runs=records, **runtime()))
    save(args.out / "stage.json", dict(stage=args.stage, status="complete", runs=records, **runtime()))


def diverged(error: str) -> bool:
    """A non-finite loss, evaluation value or gradient norm: a numerical divergence, not an infrastructure failure."""
    return error.startswith("FloatingPointError") or ("RuntimeError" in error.split(":")[0] and "non-finite" in error)


def hhmmss(seconds: float) -> str:
    seconds = math.ceil(seconds / 300) * 300
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def size(smoke: Path, out: Path | None = None) -> dict:
    """Expected pilot times from the time-only smoke; pack both arms in one job only if it fits 1 h with 1.5x."""
    arms = {}
    for arm in ARMS:
        record = json.loads((smoke / arm / "record.json").read_text())
        cfg = record["config"]
        if record["status"] != "complete" or not cfg["smoke"] or cfg["train"]["steps"] != SMOKE_STEPS:
            raise ValueError(f"{arm}: not a complete {SMOKE_STEPS}-step smoke")
        step = record["steady_step_seconds"]
        overhead = max(0., record["train_seconds"] - SMOKE_STEPS * step)
        expected = STEPS * step + overhead + record["validation_seconds"]
        arms[arm] = dict(step_seconds=step, overhead_seconds=overhead, evaluation_seconds=record["validation_seconds"],
                         expected_seconds=expected, walltime=hhmmss(1.5 * (expected + RESERVE)),
                         node_hours=expected / 3600, peak_memory_bytes=record["peak_memory_bytes"])
    packed = sum(row["expected_seconds"] for row in arms.values())
    fits = 1.5 * (packed + RESERVE) <= PACK_LIMIT
    result = dict(smoke_steps=SMOKE_STEPS, pilot_steps=STEPS, reserve_seconds=RESERVE, arms=arms,
                  packed=dict(expected_seconds=packed, walltime=hhmmss(1.5 * (packed + RESERVE)),
                              fits_one_hour_with_margin=fits),
                  recommendation="packed" if fits else "separate",
                  expected_node_hours=packed / 3600,
                  rule="pack the two arms in one 1-node job only if 1.5 x (sum of expected seconds + reserve) <= 1 h")
    if out is not None:
        save(out, result)
    return result


def arm_outcome(directory: Path, arm: str) -> dict:
    """The registered cell of one arm: hop 1, L = 4, memory-only, final checkpoint. No other cell is read."""
    failure = directory / "failure.json"
    if failure.exists():
        error = json.loads(failure.read_text())["error"]
        if not diverged(error):
            raise ValueError(f"{arm}: infrastructure or code failure ({error}); rerun the unchanged config")
        return dict(passed=False, failure=error, record=str(directory / "record.json"))
    record = json.loads((directory / "record.json").read_text())
    cfg = record["config"]
    registered = (record["status"] == "complete" and cfg["writer"] == arm and cfg["variant"] == "M1"
                  and cfg["model"]["r"] == 32 and cfg["seed"] == 17 and cfg["oracle_seed"] == 17
                  and cfg["train"]["steps"] == STEPS and cfg["train"]["lr"] == 0.003 and cfg["train"]["depths"] == [4]
                  and not cfg["smoke"] and cfg["eval"]["splits"] == ["val"] and not cfg["eval"]["allow_confirmation"]
                  and cfg["eval"]["seed"] == 4242 and cfg["eval"]["episodes"] == 1000 and cfg["protocol"] == "memory_only")
    if not registered:
        raise ValueError(f"{arm}: not a complete run of the registered pilot configuration")
    cells = [row for row in record["validation"]
             if row["condition"] == "memory_only/original" and row["hop"] == 1 and row["L"] == 4]
    if len(cells) != 1 or cells[0]["n"] != 1000:
        raise ValueError(f"{arm}: the hop-1 L = 4 validation cell is missing or incomplete")
    value = float(cells[0]["full_accuracy"])
    return dict(hop1_L4_full_em=value, n=cells[0]["n"], passed=value >= THRESHOLD,
                checkpoint_hash=record["checkpoint_hash"], writer_hash=record["writer_hash"],
                job_id=record["job_id"], record=str(directory / "record.json"))


def decide(directories: dict[str, Path], out: Path | None = None) -> dict:
    """O-rot if it passes, else O-shared if it passes, else stop."""
    arms = {arm: arm_outcome(directories[arm], arm) for arm in ARMS}
    selected = next((arm for arm in ARMS if arms[arm]["passed"]), None)
    result = dict(status="selected" if selected else "stop", selected=selected, threshold=THRESHOLD,
                  metric="hop-1 validation full-vocabulary EM, L = 4, final checkpoint (1000 episodes, seed 4242)",
                  rule="O-rot if it passes; else O-shared if it passes; else stop", arms=arms)
    if out is not None:
        save(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("smoke", "pilot"):
        command = commands.add_parser(stage)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    command = commands.add_parser("size")
    command.add_argument("--smoke", type=Path, required=True)
    command.add_argument("--out", type=Path)
    command = commands.add_parser("decide")
    for arm in ARMS:
        command.add_argument(f"--{arm}", type=Path, required=True, help=f"run directory of {arm}")
    command.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.stage == "size":
        print(json.dumps(size(args.smoke, args.out), indent=2))
    elif args.stage == "decide":
        print(json.dumps(decide({arm: getattr(args, arm) for arm in ARMS}, args.out), indent=2))
    else:
        train_stage(args)


if __name__ == "__main__":
    main()
