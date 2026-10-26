#!/usr/bin/env python3
"""WP-C3 fixed-checkpoint validation, threshold lock, and screening evaluation."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from analyze_p4s_stopping import select_thresholds, summarize
from benchmark_p4s_stopping import benchmark_reader
from looplora.core.contraction import contraction_upper_bound
from looplora.core.stopping import ReaderBatch, trace_reader
from looplora.data.relations import generate
from looplora.infra.seeds import stable_hash
from looplora.sym.evaluate import _slice, stratify_queries
from looplora.sym.model import make_task_spec
from looplora.sym.run import source_metadata
from looplora.sym.train import load_checkpoint


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checkpoint_manifest(plan: dict) -> list[dict]:
    runs = []
    for row in plan["runs"]:
        if row["variant"] not in {"M1", "M1-lin"}:
            continue
        path = Path(row["training_dir"]) / "checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.blake2b(digest_size=16)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        run = {key: row[key] for key in ("name", "writer", "r", "variant", "seed")}
        run.update(checkpoint=str(path.resolve()), checkpoint_blake2b=digest.hexdigest(),
                   family=f"{row['writer']}/r{row['r']}/{row['variant']}")
        runs.append(run)
    expected = {(writer, rank, variant, seed) for writer in ("oracle_random", "joint")
                for rank in (64, 128) for variant in ("M1", "M1-lin") for seed in (17, 29, 43)}
    actual = {(r["writer"], r["r"], r["variant"], r["seed"]) for r in runs}
    if len(runs) != 24 or actual != expected:
        raise ValueError("P4-S requires exactly the 24 released F1 evolving-query checkpoints")
    return sorted(runs, key=lambda row: row["name"])


def reader_batch(prepared) -> ReaderBatch:
    """Keep the unchanged context candidate set, sorted by global entity id."""
    b, q, r = prepared.u.shape
    vocabulary = prepared.candidate_mask.shape[1]
    count = int(prepared.candidate_mask.sum(-1).max())
    ids = torch.arange(vocabulary, device=prepared.u.device).expand(b, -1)
    ids = ids.masked_fill(~prepared.candidate_mask, vocabulary).sort(dim=-1).values[:, :count]
    ids = torch.where(ids == vocabulary, -1, ids)
    codes = prepared.codes.gather(1, ids.clamp_min(0)[..., None].expand(-1, -1, prepared.codes.shape[-1]))
    return ReaderBatch(prepared.u.reshape(b * q, r), prepared.memory.repeat_interleave(q, dim=0),
                       codes.repeat_interleave(q, dim=0), ids.repeat_interleave(q, dim=0))


@torch.inference_mode()
def collect(model, split: str, design: dict, out: Path) -> tuple[dict, ReaderBatch]:
    config = copy.deepcopy(model.config)
    config["eval"]["hops"] = design["hops"]
    spec = make_task_spec(config, evaluation=True)
    seed, count = design["evaluation_seed"], design["episodes_per_split"]
    started = time.perf_counter()
    arrays = stratify_queries(generate(spec, split, 0, count, seed), spec, design["hops"], seed)
    certificates = design.get("collect_certificates", True)
    chunks = {key: [] for key in ("predictions", "read_norm", "relative_change")}
    if certificates:
        chunks["certified_c"] = []
    first = None
    prepare_seconds = trace_seconds = certificate_seconds = 0.0
    for begin in range(0, count, design["episode_batch_size"]):
        part = _slice(arrays, slice(begin, begin + design["episode_batch_size"]))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        prepared = model.prepare(part)
        batch = reader_batch(prepared)
        torch.cuda.synchronize()
        prepare_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        trace = trace_reader(model.adapter, batch, design["max_steps"])
        torch.cuda.synchronize()
        trace_seconds += time.perf_counter() - tick
        b, q = prepared.u.shape[:2]
        for key in ("predictions", "read_norm", "relative_change"):
            value = getattr(trace, key)
            if key != "predictions" and not torch.isfinite(value[:, 1:]).all():
                raise FloatingPointError(f"non-finite {key} in {split}")
            chunks[key].append(value.reshape(b, q, -1).cpu().numpy())
        if first is None:
            first = batch.take(torch.arange(b, device=model.device) * q + torch.arange(b, device=model.device) % q)
            # Audit candidate-only decoding and trace updates against the existing reader.
            for depth in design["fixed_references"]:
                output = model.read(prepared, depth)["logits"]
                expected = output.masked_fill(~prepared.candidate_mask[:, None], -torch.inf).argmax(-1)
                actual = trace.predictions[:, depth].reshape(b, q)
                if not torch.equal(actual, expected):
                    raise AssertionError(f"fixed-reader/candidate-decoder mismatch at L={depth}")
        if certificates:
            tick = time.perf_counter()
            certificate = contraction_upper_bound(
                model.adapter.R.double(), model.adapter.U.double(), model.adapter.Q.double(),
                prepared.memory.double(), model.adapter.cfg.rho,
            )
            torch.cuda.synchronize()
            certificate_seconds += time.perf_counter() - tick
            chunks["certified_c"].append(certificate.cpu().numpy())
    values = {key: np.concatenate(value) for key, value in chunks.items()}
    values.update(answer=arrays.answer, hop=arrays.q_hop_true,
                  episode_id=np.broadcast_to(arrays.episode_seed[:, None], arrays.answer.shape))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **values)
    timing = dict(split=split, episodes=count, queries=int(arrays.answer.size),
                  prepare_seconds=prepare_seconds, trace_seconds=trace_seconds,
                  certificate_seconds=certificate_seconds, total_seconds=time.perf_counter() - started)
    return timing, first


def load_model(run):
    model = load_checkpoint(run["checkpoint"], device="cuda:0")
    config = model.checkpoint_config
    if (config["writer"], config["model"]["r"], config["variant"], config["seed"]) != (
        run["writer"], run["r"], run["variant"], run["seed"],
    ):
        raise ValueError("checkpoint identity differs from F1 manifest")
    model.requires_grad_(False)
    return model.eval()


def benchmark(model, batch, thresholds, run, design):
    settings = design["latency"]
    rows = []
    # Batch-one timings average eight distinct queries, one per hop.
    choices = [(batch, 0)] + [(batch.take(torch.tensor([i], device=model.device)), i + 1) for i in range(8)]
    for inputs, hop in choices:
        for row in benchmark_reader(model.adapter, inputs, thresholds,
                                    warmup=settings["warmup"], repeats=settings["repeats"]):
            rows.append({**{key: run[key] for key in ("name", "family", "writer", "r", "variant", "seed")},
                         "latency_hop": hop, **row})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("reports/p1s-f1/evaluation_plan.json"))
    parser.add_argument("--design", type=Path, default=Path("reports/p4s/design.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("WP-C3 evaluation requires Miyabi-G compute/GPU")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    design = json.loads(args.design.read_text())
    runs = checkpoint_manifest(json.loads(args.manifest.read_text()))
    if args.smoke:
        runs = [run for run in runs if run["seed"] == 17]
        design["episodes_per_split"] = 200
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = dict(design=design, runs=runs, smoke=args.smoke, **source_metadata(),
                    started_at=datetime.now(timezone.utc).isoformat(),
                    runtime=dict(host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"),
                                 torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                                 dtype="float32; certificate float64", threads=1))
    save(root / "manifest.json", metadata)
    timing, latency = [], []
    for run in runs:
        model = load_model(run)
        row, batch = collect(model, "val", design, root / "traces" / run["name"] / "val.npz")
        timing.append({"name": run["name"], **row})
        if args.smoke:
            latency.extend(benchmark(model, batch, {"read_norm": .1, "relative_change": .1,
                                                     "output_stability": 2}, run, design))
        print(json.dumps({"stage": "val", "name": run["name"], "seconds": row["total_seconds"]}), flush=True)
        del model
    if not args.smoke:
        selection = select_thresholds(root, runs, design)
        selection["locked_at"] = datetime.now(timezone.utc).isoformat()
        save(root / "thresholds.json", selection)
        for run in runs:
            model = load_model(run)
            row, batch = collect(model, "test", design, root / "traces" / run["name"] / "test.npz")
            timing.append({"name": run["name"], **row})
            thresholds = {entry["criterion"]: entry["threshold"] for entry in selection["selected"]
                          if entry["family"] == run["family"]}
            latency.extend(benchmark(model, batch, thresholds, run, design))
            print(json.dumps({"stage": "test", "name": run["name"], "seconds": row["total_seconds"]}), flush=True)
            del model
    save(root / "latency.json", latency)
    save(root / "timing.json", timing)
    if not args.smoke:
        summarize(root, runs, design, selection, args.summary_out or root / "summaries")
    metadata.update(status="complete", total_seconds=time.perf_counter() - started,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    design_hash=stable_hash(design), peak_memory_bytes=torch.cuda.max_memory_allocated())
    save(root / "manifest.json", metadata)
    print(json.dumps({"status": "complete", "smoke": args.smoke,
                      "seconds": metadata["total_seconds"], "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
