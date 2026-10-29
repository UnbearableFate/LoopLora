#!/usr/bin/env python3
"""LSR evaluation: p^0..p^8 and the final-position trajectories of the r = 128 M1-L4 checkpoints (docs/prereg/LSR.md).

Evaluation only. For every checkpoint and episode range it writes ``<out>/<run>/trace.npz`` (pred, change, read_ratio, answer, hop)
and ``record.json``; thresholds and statistics are computed offline by scripts/analyze_lsr.py. Ranges:
  smoke  val [0, 64): timing and audits only, its numbers are never used
  val    val [1000, 3000): the threshold-selection set
  test   test [106000, 108000) (window `lsr`): needs a committed authorization (check_test_authorization) before any episode
         is generated, and an output directory that does not exist yet (one read per checkpoint)
``--latency`` adds the descriptive within-job timing on val [1000, 1128) with thresholds from --thresholds (test) or --tau1/--tau2
(smoke). Invoke on Miyabi-G only through scripts/submit.py.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_p4s_stopping import require_compute
from looplora.data.relations import generate
from looplora.llm import lsr
from looplora.llm.config import CONFIRMATION_WINDOWS, split_window
from looplora.llm.evaluate import _answer_tokens, _slice, _sync
from looplora.llm.inject import load_model, load_tokenizer
from looplora.llm.train import load_checkpoint
from looplora.sym.evaluate import stratify_queries, tensor_hash
from looplora.sym.run import source_metadata

from evaluate_p1l_causal import evaluation_config
from evaluate_p1l_main import check_run
from run_p1l_main import ROOT, digest, runtime, save

PREREG = "docs/prereg/LSR.md"
THRESHOLDS = "reports/lsr/thresholds.json"
CONFIDENCE = 1 - 0.05 / 6
AUDIT_BATCHES = 2
# bf16 answer logits move by up to ~0.3 when the batch composition changes (smoke 2026-10-29: 2 of 128 questions flip in a plain
# fixed-depth pass computed on half the batch); a disagreement is only an error when the grouped pass's top-2 gap exceeds this.
TIE_GAP = 1.0


def check_test_authorization(authorization: dict, thresholds: dict, root: Path = ROOT) -> None:
    """Before any test episode is generated: the committed authorization names this prereg and the locked thresholds."""
    if authorization.get("study") != "lsr" or authorization.get("status") != "allowed":
        raise ValueError("the LSR test range is held until the orchestrator releases it")
    if not authorization.get("orchestrator_decision"):
        raise ValueError("the authorization must record the orchestrator's decision")
    if authorization.get("confirmation_window") != lsr.WINDOW or CONFIRMATION_WINDOWS[lsr.WINDOW] != 106000:
        raise ValueError("the authorization must name the registered `lsr` window")
    if authorization.get("gated_comparisons") != 6 or abs(authorization.get("confidence", 0) - CONFIDENCE) > 1e-12:
        raise ValueError("LSR gates 3 rules x 2 backbones: k = 6, 99.17%")
    if authorization.get("prereg") != dict(path=PREREG, hash=digest(root / PREREG)):
        raise ValueError("the authorization does not name the registered prereg")
    if authorization.get("thresholds") != dict(path=THRESHOLDS, hash=digest(root / THRESHOLDS)):
        raise ValueError("the authorization does not name the committed thresholds")
    if thresholds.get("status") != "locked" or thresholds.get("prereg", {}).get("hash") != digest(root / PREREG):
        raise ValueError("the thresholds are not locked under this prereg")


def manifest_runs(backbone: str, seeds) -> tuple[dict, dict, Path]:
    spec = lsr.BACKBONES[backbone]
    path = (spec["root"] or ROOT) / spec["manifest"]
    manifest = json.loads(path.read_text())
    if manifest.get("rank") != 128 or (spec["backbone"] is not None and manifest.get("backbone") != spec["backbone"]):
        raise ValueError("not the registered r = 128 manifest of this backbone")
    runs = {f"M1-L4-s{seed}": manifest["runs"][f"M1-L4-s{seed}"] for seed in seeds}
    return manifest, runs, path


def make_arrays(experiment, window: dict):
    cfg, spec = evaluation_config(copy.deepcopy(experiment.config))
    arrays = stratify_queries(generate(spec, window["split"], window["start"], window["count"], window["seed"]),
                              spec, cfg["eval"]["hops"], window["seed"])
    if not np.array_equal(arrays.q_hop_true, np.broadcast_to(np.arange(1, 9), arrays.q_hop_true.shape)):
        raise AssertionError("hop columns must be hops 1..8")
    return cfg, spec, arrays


@torch.no_grad()
def scan_range(experiment, spec, arrays, *, batch_size):
    chunks, hashes, seconds = {}, [], 0.0
    for index, begin in enumerate(range(0, len(arrays.episode_seed), batch_size)):
        part = _slice(arrays, slice(begin, min(len(arrays.episode_seed), begin + batch_size)))
        _sync(experiment)
        started = time.perf_counter()
        prepared = experiment.prepare(part)
        if not np.array_equal(prepared.answer_ids.cpu().numpy(), _answer_tokens(part.answer, spec.V)):
            raise AssertionError("prepared token answers differ from generated labels")
        hashes.append(tensor_hash(prepared.snapshot.states[experiment.layer]))
        scan = lsr.scan_batch(experiment, prepared, audit=index < AUDIT_BATCHES)
        _sync(experiment)
        seconds += time.perf_counter() - started
        for key, value in scan.items():
            chunks.setdefault(key, []).append(value)
    return {key: np.concatenate(parts) for key, parts in chunks.items()}, hashes, seconds


@torch.no_grad()
def latency_audit(experiment, spec, arrays, taus: dict, *, batch_size, repeats=5, warmups=2):
    """Wall seconds per question within this job: fixed L = 4 and L = 8, the trace pass (phase 1 as implemented: the full forward
    with the final-position trajectories) and the grouped-depth pass at s* (phase 2) of R1 and R3. Checks phase 2 against p^{s*}."""
    prepared, scans = [], []
    for begin in range(0, len(arrays.episode_seed), batch_size):
        part = _slice(arrays, slice(begin, min(len(arrays.episode_seed), begin + batch_size)))
        prepared.append(experiment.prepare(part))
        scans.append(lsr.scan_batch(experiment, prepared[-1]))
    questions = int(arrays.answer.size)
    depths = {rule: [lsr.stop_depths(rule, scan, tau) for scan in scans]
              for rule, tau in taus.items()}

    def clock(work):
        for _ in range(warmups):
            work()
        values = []
        for _ in range(repeats):
            _sync(experiment)
            started = time.perf_counter()
            work()
            _sync(experiment)
            values.append(time.perf_counter() - started)
        return float(np.mean(values)) / questions, float(np.std(values)) / questions

    def fixed(depth):
        return lambda: [experiment.score(batch, depth) for batch in prepared]

    def trace():
        for batch in prepared:
            with lsr.instrument(experiment.adapter, lsr.p1lc.last_positions(batch.question_mask), trace={}):
                experiment.score(batch, lsr.L_MAX)

    def stopped(rule):
        rows = [torch.as_tensor(depth.reshape(-1), device=experiment.device) for depth in depths[rule]]
        return lambda: [lsr.stopped_logits(experiment, batch, row) for batch, row in zip(prepared, rows)]

    result = dict(questions=questions, repeats=repeats, warmups=warmups, taus=taus, seconds_per_question={})
    for label, work in (("fixed_L4", fixed(4)), ("fixed_L8", fixed(8)), ("phase1_trace_L8", trace),
                        *((f"phase2_{rule}", stopped(rule)) for rule in taus)):
        mean, spread = clock(work)
        result["seconds_per_question"][label] = dict(mean=mean, std=spread)
    mismatches = {}
    for rule in taus:
        count = beyond = 0
        largest = 0.0
        for batch, scan, depth in zip(prepared, scans, depths[rule]):
            row = torch.as_tensor(depth.reshape(-1), device=experiment.device)
            logits = lsr.stopped_logits(experiment, batch, row).float()
            predicted = logits.argmax(-1).cpu().numpy()
            assembled = np.take_along_axis(scan["pred"], depth[..., None], axis=-1)[..., 0]
            wrong = predicted != assembled
            top = logits.topk(2, dim=-1).values
            gap = (top[..., 0] - top[..., 1]).cpu().numpy()[wrong]
            count += int(wrong.sum())
            beyond += int((gap > TIE_GAP).sum())
            largest = max([largest, *gap.tolist()])
        mismatches[rule] = dict(count=count, fraction=count / questions, beyond_tie_gap=beyond, max_top2_gap=largest)
    result["phase2_mismatch"] = mismatches
    result["tie_gap"] = TIE_GAP
    result["mean_depth"] = {rule: float(np.mean([d.mean() for d in depths[rule]])) for rule in taus}
    if any(value["beyond_tie_gap"] for value in mismatches.values()):
        raise AssertionError(f"the grouped stopped pass disagrees with p^s* beyond bf16 near-ties: {mismatches}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=sorted(lsr.BACKBONES), required=True)
    parser.add_argument("--split", choices=("smoke", "val", "test"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(lsr.SEEDS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--tau1", type=float)
    parser.add_argument("--tau2", type=float)
    args = parser.parse_args()
    require_compute()
    torch.set_num_threads(1)
    if set(args.seeds) - set(lsr.SEEDS):
        parser.error("seeds are 17, 29, 43")
    thresholds = None
    if args.split == "test":
        if not args.authorization:
            parser.error("the test range requires --authorization")
        thresholds = json.loads((ROOT / THRESHOLDS).read_text())
        check_test_authorization(json.loads(args.authorization.read_text()), thresholds)  # before any test episode
        start = split_window(dict(confirmation=lsr.WINDOW), "confirm")[1]
        window = dict(split="test", start=start, count=2000, seed=lsr.SEED)
    else:
        if args.authorization:
            parser.error("--authorization is for the test range only")
        window = lsr.SMOKE_RANGE if args.split == "smoke" else lsr.VAL_RANGE
    taus = {}
    if args.latency:
        if thresholds is not None:
            for rule in ("R1", "R3"):
                chosen = thresholds["backbones"][args.backbone]["rules"][rule]["selected"]
                if chosen:
                    taus[rule] = chosen["tau"]
        elif args.tau1 is not None and args.tau2 is not None:
            taus = dict(R1=args.tau1, R3=args.tau2)
        else:
            parser.error("--latency needs the test thresholds or --tau1 and --tau2")
    manifest, runs, manifest_path = manifest_runs(args.backbone, args.seeds)
    backbone = lsr.BACKBONES[args.backbone]["backbone"]
    args.out.mkdir(parents=True, exist_ok=True)
    for path in runs.values():                          # light provenance before any episode: complete, unchanged checkpoints
        cfg = check_run("M1-L4", path, 128)
        if cfg.get("backbone") != backbone:
            raise ValueError("a checkpoint names another backbone than the registered one")
    model, tokenizer = load_model(backbone=backbone), load_tokenizer(backbone)
    for name, path in runs.items():
        run = Path(path)
        out = args.out / name
        out.mkdir(exist_ok=False)                       # a second read of the same range is refused
        started = time.perf_counter()
        experiment = load_checkpoint(run / "checkpoint.pt", model=model, tokenizer=tokenizer)
        try:
            cfg, spec, arrays = make_arrays(experiment, window)
            samples, hashes, scan_seconds = scan_range(experiment, spec, arrays, batch_size=cfg["eval"]["batch_size"])
            np.savez_compressed(out / "trace.npz", episode_index=window["start"] + np.arange(len(arrays.episode_seed)),
                                episode_seed=arrays.episode_seed, hop=arrays.q_hop_true,
                                answer=_answer_tokens(arrays.answer, spec.V), **samples)
            record = {}
            if args.latency:
                latency_window = lsr.LATENCY_RANGE
                _, _, latency_arrays = make_arrays(experiment, latency_window)
                record["latency"] = latency_audit(experiment, spec, latency_arrays, taus,
                                                  batch_size=cfg["eval"]["batch_size"])
                record["latency"]["window"] = latency_window
        finally:
            experiment.close()
        save(out / "record.json", dict(status="complete", name=name, seed=int(name.rsplit("-s", 1)[1]), split=args.split,
            backbone=args.backbone, window=window, episodes=len(arrays.episode_seed), questions=int(arrays.answer.size),
            batch_size=cfg["eval"]["batch_size"], audit_batches=AUDIT_BATCHES, memory_hashes=hashes,
            scan_seconds=scan_seconds, checkpoint_hash=digest(run / "checkpoint.pt"), manifest_hash=digest(manifest_path),
            trace_hash=digest(out / "trace.npz"), prereg_hash=digest(ROOT / PREREG), seconds=time.perf_counter() - started,
            **record, **source_metadata(), **runtime()))
        print(json.dumps(dict(run=name, split=args.split, scan_seconds=scan_seconds,
                              seconds=time.perf_counter() - started)), flush=True)


if __name__ == "__main__":
    main()
