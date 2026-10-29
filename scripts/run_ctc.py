#!/usr/bin/env python3
"""Study CTC (docs/prereg/CTC.md): Stage A probes, Stage B writer training, evaluation. Runs on Miyabi-G compute only,
through scripts/submit.py --commit <full hash>.

  stage-a   features and probes; writes probes.csv, predictions.npz, feature-stats.pt, lock.json
  train     one Stage B arm (ce or ceans) from a Stage A lock
  evaluate  both arms, the three frozen readers, hard/soft/zero/shuffled/oracle memory; decision when the range is final

Smoke runs pass smaller ranges and steps; registered runs refuse them. A smoke never reads val [0, 2000).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch

from looplora.data.relations import generate
from looplora.llm import ctc
from looplora.llm.run import _write_json, source_metadata


def stage_a(args) -> None:
    train = dict(ctc.STAGE_A_TRAIN, episodes=args.train_episodes)
    select = dict(ctc.STAGE_A_SELECT, start=args.select_start, episodes=args.select_episodes)
    lock = ctc.run_stage_a(Path(args.out), train=train, select=select, steps=args.steps, smoke=args.smoke)
    print(json.dumps({key: lock[key] for key in ("plan", "selection", "selection_joint_accuracy", "seconds")}, indent=1))


def train(args) -> None:
    from looplora.llm.inject import load_model, load_tokenizer
    from looplora.llm.train import make_task_spec

    if not args.smoke and args.steps != ctc.STAGE_B_STEPS:
        raise ValueError("a registered Stage B run uses 8,000 steps")
    lock = ctc.load_lock(Path(args.lock))
    if not args.smoke and lock["smoke"]:
        raise ValueError("a registered Stage B run needs the registered Stage A lock")
    device = torch.device("cuda")
    model, tokenizer = load_model(), load_tokenizer()
    names = ctc.NameTokens(tokenizer)
    spec = make_task_spec(ctc.reader_config())
    experiment = ctc.load_reader("M1-L4", model, tokenizer) if args.arm == "ceans" else None
    try:
        oracle = experiment.writer if experiment is not None else ctc.make_oracle(device)
        if ctc.oracle_hash(oracle) != ctc.ORACLE_HASH:
            raise ValueError("oracle tables differ from P1-L's r = 128 oracle")
        writer = ctc.build_writer(lock, oracle, device)
        summary = ctc.train_writer(writer, args.arm, model=model, tokenizer=tokenizer, names=names, spec=spec,
                                   out=Path(args.out), experiment=experiment, steps=args.steps, workers=args.workers)
    finally:
        if experiment is not None:
            experiment.close()
    summary.update(lock_hash=ctc.digest(Path(args.lock)), smoke=args.smoke)
    _write_json(Path(args.out) / "provenance.json", dict(lock=str(Path(args.lock).resolve()),
                lock_hash=summary["lock_hash"], smoke=args.smoke, **source_metadata()))
    print(json.dumps({key: summary[key] for key in ("arm", "steps", "train_seconds", "train_step_seconds",
                                                    "train_step_p95_seconds", "final")}, indent=1))


def evaluate(args) -> None:
    from looplora.llm.inject import load_model, load_tokenizer
    from looplora.llm.train import make_task_spec
    from looplora.sym.evaluate import stratify_queries

    final = (args.split, args.start, args.episodes) == (ctc.FINAL_EVAL["split"], ctc.FINAL_EVAL["start"],
                                                         ctc.FINAL_EVAL["episodes"])
    if not final and args.split == "val" and args.start < 2000:
        raise ValueError("a non-final evaluation must not read val [0, 2000)")
    arms = dict(item.split("=", 1) for item in args.arm_dir)
    lock = ctc.load_lock(Path(args.lock))
    started = time.perf_counter()
    device = torch.device("cuda")
    model, tokenizer = load_model(), load_tokenizer()
    names = ctc.NameTokens(tokenizer)
    oracle = ctc.make_oracle(device)
    if ctc.oracle_hash(oracle) != ctc.ORACLE_HASH:
        raise ValueError("oracle tables differ from P1-L's r = 128 oracle")
    spec = make_task_spec(ctc.reader_config(), evaluation=True)
    arrays = stratify_queries(generate(spec, args.split, args.start, args.episodes, ctc.EVAL_SEED), spec,
                              list(range(1, 9)), ctc.EVAL_SEED)
    hops = ctc.hop_matrix(arrays)
    facts = torch.as_tensor(arrays.facts, dtype=torch.long, device=device)
    valid = torch.as_tensor(arrays.fact_valid, dtype=torch.bool, device=device)
    states = {"oracle/hard": oracle(facts, valid), "zero": torch.zeros(len(facts), ctc.R_MEM, ctc.R_MEM, device=device)}
    classification, writer_hashes, timing = {}, {}, {}
    for arm, directory in arms.items():
        path = Path(directory) / "writer.pt"
        writer = ctc.load_writer(path, lock, oracle, device)
        writer_hashes[arm] = ctc.digest(path)
        began = time.perf_counter()
        record = ctc.writer_outputs(writer, model, tokenizer, names, arrays)
        timing[f"writer_{arm}"] = time.perf_counter() - began
        classification[arm] = ctc.code_accuracy(record)
        states[f"{arm}/hard"], states[f"{arm}/soft"] = record["hard"], record["soft"]
        states[f"{arm}/shuffled"] = ctc.roll_within_batches(record["hard"])
        classification[arm]["parameters"] = writer.parameter_count()
        # Sanity: perfect predictions reproduce the oracle memory bitwise (the table lookup is exact).
        truth = record["truth"].to(device)
        classification[arm]["perfect_lookup_equals_oracle"] = bool(torch.equal(
            ctc.hard_memory(oracle, truth, valid), states["oracle/hard"]))
        del writer
    results = {}
    for label in ctc.READER_DEPTHS:
        experiment = ctc.load_reader(label, model, tokenizer)
        try:
            names_here = ["oracle/hard", "zero"] if label == "B3-L4" else list(states)
            began = time.perf_counter()
            results[label] = ctc.evaluate_readers(label, experiment, arrays, {c: states[c] for c in names_here})
            timing[f"reader_{label}"] = time.perf_counter() - began
        finally:
            experiment.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "samples.npz", **{f"{r}|{c.replace('/', ':')}|L{d}|{k}": v for r, cs in results.items()
                                                for c, ds in cs.items() for d, rec in ds.items() for k, v in rec.items()})
    metrics = dict(split=args.split, start=args.start, episodes=args.episodes, final=final, seed=ctc.EVAL_SEED,
                   classification=classification, table=ctc.em_rows(results, hops), timing=timing,
                   writer_hashes=writer_hashes, reader_hashes=ctc.READER_HASHES, oracle_hash=ctc.ORACLE_HASH,
                   lock_hash=ctc.digest(Path(args.lock)), host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"),
                   seconds=time.perf_counter() - started, **source_metadata())
    b3 = results["B3-L4"]
    metrics["b3_memory_independent"] = bool(np.array_equal(b3["oracle/hard"][4]["prediction"], b3["zero"][4]["prediction"]))
    metrics["oracle_contrast"] = ctc.contrast(results, "oracle/hard")
    metrics["arms"] = {arm: ctc.arm_metrics(results, arm, classification[arm]["joint"]) for arm in arms}
    for arm in arms:
        for condition in ("hard", "soft", "shuffled"):
            metrics["arms"][arm][f"contrast_{condition}"] = ctc.contrast(results, f"{arm}/{condition}")
    if final and set(arms) == {"ce", "ceans"}:
        metrics["decision"] = ctc.decide(metrics)
    _write_json(out / "metrics.json", metrics)
    print(json.dumps({key: metrics[key] for key in ("arms", "oracle_contrast", "seconds") if key in metrics}, indent=1))
    if "decision" in metrics:
        print(json.dumps(metrics["decision"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("stage-a")
    a.add_argument("--out", required=True)
    a.add_argument("--smoke", action="store_true")
    a.add_argument("--train-episodes", type=int, default=ctc.STAGE_A_TRAIN["episodes"])
    a.add_argument("--select-start", type=int, default=ctc.STAGE_A_SELECT["start"])
    a.add_argument("--select-episodes", type=int, default=ctc.STAGE_A_SELECT["episodes"])
    a.add_argument("--steps", type=int, default=ctc.PROBE_STEPS)
    a.set_defaults(run=stage_a)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=("ce", "ceans"), required=True)
    t.add_argument("--lock", default=str(ctc.ROOT / "reports" / "ctc" / "stage-a-lock.json"))
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=ctc.STAGE_B_STEPS)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--smoke", action="store_true")
    t.set_defaults(run=train)
    e = sub.add_parser("evaluate")
    e.add_argument("--lock", default=str(ctc.ROOT / "reports" / "ctc" / "stage-a-lock.json"))
    e.add_argument("--arm-dir", action="append", required=True, metavar="ARM=DIR")
    e.add_argument("--out", required=True)
    e.add_argument("--split", default=ctc.FINAL_EVAL["split"])
    e.add_argument("--start", type=int, default=ctc.FINAL_EVAL["start"])
    e.add_argument("--episodes", type=int, default=ctc.FINAL_EVAL["episodes"])
    e.set_defaults(run=evaluate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
