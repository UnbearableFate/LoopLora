#!/usr/bin/env python3
"""Compute-only, train-only P1-S throughput measurement and budget projections.

python scripts/benchmark_p1s_main.py --out /work/.../throughput --steps 250 --procs 4 8
Each setting trains the same eight real workloads. No evaluation function is
called; no validation, screening or confirmation samples or scores are produced.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require_compute() -> None:
    host = socket.gethostname().split(".")[0]
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/nonexistent-pbs-nodefile"))
    hosts = {line.strip().split(".")[0] for line in nodefile.read_text().splitlines()} if nodefile.is_file() else set()
    if not os.environ.get("PBS_JOBID") or not host.startswith("mg") or host not in hosts:
        raise RuntimeError("run only inside a confirmed Miyabi-G PBS compute allocation via scripts/submit.py")


def workloads(steps: int) -> list[dict]:
    from looplora.sym.config import resolve_config
    from looplora.sym.pilot import base_config

    cases = [
        ("orth32-relu", "oracle_orth_r32", "oracle_orth", "M1", 32, "t2_chain_end"),
        ("orth32-linear", "oracle_orth_r32", "oracle_orth", "M1-lin", 32, "t2_chain_end"),
        ("orth16-linear", "oracle_orth_r16", "oracle_orth", "M1-lin", 16, "t2_chain_end"),
        ("random32", "oracle_random_r32", "oracle_random", "M1", 32, "t2_chain_end"),
        ("random16", "oracle_random_r16", "oracle_random", "M1-lin", 16, "t2_chain_end"),
        ("joint-chain", "joint_chain_r32", "joint", "M1", 32, "t2_chain_end"),
        ("joint-typed", "joint_typed_r32", "joint", "M1-lin", 32, "t2_typed"),
        ("joint-t1", "joint_t1_r32", "joint", "B4", 32, "t1"),
    ]
    result = []
    for name, cost_class, writer, variant, rank, task in cases:
        typed = task == "t2_typed"
        config = base_config("throughput-" + name, writer, variant, rank,
                             9 if typed else 12, .003,
                             "t2_chain_end" if typed else task,
                             [8, 1] if typed else None if task == "t1" else [8, 4])
        if typed:
            config["task"].update(name="t2_typed", E=12, n_distract=3)
        config["train"].update(steps=steps, batch_size=64, log_every=50)
        config["pretrain"]["steps"] = steps
        config["runtime"]["threads"] = 1
        result.append({"name": name, "cost_class": cost_class, "config": resolve_config(config)})
    return result


def train_trial(config_path: Path, out: Path) -> None:
    """Retain a normal registry record for every real benchmark training run."""
    require_compute()
    from looplora.infra.seeds import derive_seed, stable_hash
    from looplora.sym.run import source_metadata

    config = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    provenance = source_metadata()
    config_hash = stable_hash(config)
    write_json(out / "config.json", {"config": config, "config_hash": config_hash, **provenance})
    record = {
        "experiment_id": config["name"], "status": "running", **provenance,
        "config_hash": config_hash, "data_seed": derive_seed(config["seed"], "train_data"),
        "eval_data_seed": None, "model_revision": "symbolic-v1", "train_seed": config["seed"],
        "backbone_seed": config["backbone_seed"], "checkpoint": str(out / "checkpoint.pt"),
        "samples": None, "insertion_point": "symbolic_projection", "task": config["task"]["name"],
        "writer": config["writer"], "variant": config["variant"], "rank": config["model"]["r"],
        "r_mem": config["model"]["r_mem"], "L": config["train"]["depths"], "mean_L": None,
        "parameters": None, "state_bytes": 64 * config["model"]["r_mem"] ** 2 * 4,
        "train_tokens": None, "train_seconds": None, "pretrain_tokens": 0, "pretrain_seconds": 0.,
        "eval_samples": 0, "scores": None, "latency": None, "peak_memory_bytes": None,
        "failure_reason": None, "scope": "throughput benchmark only; not a main run",
        "evaluation_performed": False,
    }
    write_json(out / "record.json", record)
    started = time.perf_counter()
    try:
        import torch
        from looplora.data.relations import GENERATOR_VERSION
        from looplora.sym.train import train

        torch.set_num_threads(1)
        device = torch.device(config["device"])
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        record["data_version"] = GENERATOR_VERSION
        record["runtime"] = {"host": socket.gethostname(), "job_id": os.environ["PBS_JOBID"],
                             "python": sys.version.split()[0], "torch": torch.__version__,
                             "gpu": torch.cuda.get_device_name(device), "threads": torch.get_num_threads(),
                             "device": str(device), "dtype": "float32"}
        model, summary = train(config, out)
        torch.cuda.synchronize(device)
        summary["process_seconds"] = time.perf_counter() - started
        summary["runtime"] = record["runtime"]
        write_json(out / "summary.json", summary)
        write_json(out / "timing.json", {key: summary[key] for key in
                   ("train_seconds", "train_step_seconds", "train_step_p95_seconds", "process_seconds")})
        record.update(status="complete", train_tokens=summary["train_tokens"],
                      train_seconds=summary["train_seconds"], mean_L=summary["mean_L"],
                      writer_checkpoint=summary["writer_checkpoint"],
                      parameters={"total": sum(p.numel() for p in model.parameters()),
                                  "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
                      latency={"train_step_seconds": summary["train_step_seconds"]},
                      peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)))
    except BaseException as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["process_seconds"] = time.perf_counter() - started
        write_json(out / "record.json", record)


def measure(cases: list[dict], root: Path, procs: int) -> dict:
    root.mkdir()
    (root / "_configs").mkdir()
    (root / "_logs").mkdir()
    pending = list(cases)
    running = {}
    statuses = {}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.perf_counter()
    try:
        while pending or running:
            for name, (process, log) in list(running.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    statuses[name] = code
                    del running[name]
            while pending and len(running) < procs:
                case = pending.pop(0)
                name = case["name"]
                config = dict(case["config"], name=f"throughput-p{procs}-{name}")
                path = root / "_configs" / f"{name}.json"
                write_json(path, config)
                log = (root / "_logs" / f"{name}.log").open("w")
                try:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                                "--worker-config", str(path), "--out", str(root / name)],
                                               env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                running[name] = (process, log)
            if running:
                time.sleep(.1)
    finally:
        for process, log in running.values():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    elapsed = time.perf_counter() - started
    if any(code != 0 for code in statuses.values()):
        write_json(root / "phase.json", {"status": "failed", "returncodes": statuses, "elapsed_seconds": elapsed})
        raise RuntimeError(f"throughput subprocess failed at procs={procs}: {statuses}")
    trials = []
    for case in cases:
        summary = json.loads((root / case["name"] / "summary.json").read_text())
        trials.append({"name": case["name"], "cost_class": case["cost_class"],
                       "variant": case["config"]["variant"], "writer": case["config"]["writer"],
                       "r": case["config"]["model"]["r"], "E": case["config"]["task"]["E"],
                       "steps": case["config"]["train"]["steps"],
                       **{key: summary[key] for key in ("train_seconds", "train_step_seconds",
                                                       "train_step_p95_seconds", "process_seconds")}})
    rates = {}
    for trial in trials:
        rates[trial["cost_class"]] = max(rates.get(trial["cost_class"], 0.), trial["train_step_seconds"])
    work = sum(trial["train_seconds"] for trial in trials)
    phase = {"status": "complete", "procs": procs, "trials": trials,
             "elapsed_seconds": elapsed, "class_step_seconds_max": rates,
             "aggregate_steps_per_node_second": sum(trial["steps"] for trial in trials) / elapsed,
             "packing_factor": max(1., elapsed * procs / work)}
    write_json(root / "phase.json", phase)
    return phase


def project(phase: dict, steps: int, eval_seconds: float, benchmark_seconds: float) -> dict:
    rates = phase["class_step_seconds_max"]
    # Joint training includes writer backprop: a conservative frozen-reader proxy.
    classes = [
        ("auxiliary_T1", 3, "joint_t1_r32", 1.),
        ("primary_oracle_random", 18, "oracle_random_r32", 1.),
        ("primary_oracle_orth", 18, "oracle_orth_r32", 1.),
        ("primary_frozen_proxy_joint", 18, "joint_chain_r32", 1.),
        ("primary_joint", 18, "joint_chain_r32", 1.),
        ("replication_oracle_E18", 6, "oracle_random_r32", 1.5),
        ("replication_frozen_proxy_joint", 6, "joint_typed_r32", 1.),
        ("replication_joint_typed", 6, "joint_typed_r32", 1.),
        ("secondary_oracle_random_r16", 4, "oracle_random_r16", 1.),
        ("secondary_oracle_orth_r16", 4, "oracle_orth_r16", 1.),
        ("secondary_joint_r16_proxy_r32", 4, "joint_chain_r32", 1.),
    ]
    rows = []
    for label, count, cost_class, scale in classes:
        capacity = min(phase["procs"], 2) if cost_class == "oracle_orth_r32" else (
            1 if cost_class == "oracle_orth_r16" else phase["procs"])
        per_stage = steps * rates[cost_class] * scale
        evaluation = 0. if label == "auxiliary_T1" else eval_seconds
        rows.append({"group": label, "stages": count, "cost_class": cost_class, "load_factor": scale,
                     "concurrent_process_seconds": count * per_stage,
                     "conservative_class_capacity": capacity,
                     "single_stage_training_seconds": per_stage,
                     "single_stage_walltime_with_margin_seconds": 1.5 * (per_stage + evaluation),
                     "single_stage_fits_one_hour_with_margin": 1.5 * (per_stage + evaluation) <= 3600})
    train_seconds = sum(row["concurrent_process_seconds"] for row in rows)
    # Serialize cost-class groups and cap orthogonal occupancy at the measured
    # counts. An eight-process batch never measures eight orthogonal trials.
    capacity_adjusted = sum(row["concurrent_process_seconds"] / row["conservative_class_capacity"]
                            for row in rows)
    evaluation_seconds = 102 * eval_seconds
    packed = (capacity_adjusted + evaluation_seconds / phase["procs"]) * phase["packing_factor"]
    return {"steps_every_stage": steps, "training_stages": 105, "reader_runs": 102,
            "groups": rows, "evaluation_seconds_per_reader_assumed": eval_seconds,
            "all_single_stages_fit_one_hour_with_margin": all(
                row["single_stage_fits_one_hour_with_margin"] for row in rows),
            "ideal_all_slots_training_node_seconds": train_seconds / phase["procs"],
            "capacity_adjusted_training_node_seconds": capacity_adjusted,
            "evaluation_concurrent_process_seconds": evaluation_seconds,
            "node_hours_before_margin": packed / 3600,
            "margin": 1.5, "projected_main_node_hours_with_margin": 1.5 * packed / 3600,
            "benchmark_node_hours": benchmark_seconds / 3600,
            "projected_wp_h_node_hours_with_margin": (1.5 * packed + benchmark_seconds) / 3600,
            "packing_factor": phase["packing_factor"],
            "assumptions": ["per-class maximum measured mean step time under this concurrency",
                            "observed eight-trial batch packing/startup factor retained",
                            "orth32 occupancy capped at min(procs,2), orth16 at1; cost-class groups serialized",
                            "do not pack more orthogonal processes than measured; homogeneous scaling is unmeasured",
                            "frozen reader uses measured joint-training upper-bound proxy",
                            "secondary joint r16 uses measured r32 proxy",
                            "oracle_random E18 scales E12 measured cost by1.5",
                            "evaluation is an explicit allowance, not newly measured or executed",
                            "no non-finite retry explicitly scheduled; overall1.5margin remains"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--procs", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--eval-seconds-per-reader", type=float, default=60.)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require_compute()
    if args.worker_config:
        train_trial(args.worker_config, args.out.resolve())
        return 0
    if args.steps < 1 or not args.procs or len(set(args.procs)) != len(args.procs) or any(p < 1 or p > 8 for p in args.procs):
        parser.error("positive steps and distinct procs in1..8 are required")
    if not 0 <= args.eval_seconds_per_reader < float("inf"):
        parser.error("evaluation allowance must be finite and nonnegative")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "job_id": os.environ["PBS_JOBID"], "host": socket.gethostname(),
               "scope": "train-only throughput; not main runs; no evaluation or selection",
               "steps_per_trial": args.steps, "batch_size": 64, "threads_per_child": 1,
               "phases": [], "projections": [], "failure_reason": None}
    started = time.perf_counter()
    try:
        cases = workloads(args.steps)
        for procs in args.procs:
            summary["phases"].append(measure(cases, out / f"procs-{procs}", procs))
            write_json(out / "throughput.json", summary)
        elapsed = time.perf_counter() - started
        summary["elapsed_seconds"] = elapsed
        summary["projections"] = [
            {"procs": phase["procs"], **project(phase, steps, args.eval_seconds_per_reader, elapsed)}
            for phase in summary["phases"] for steps in (20000, 15000)]
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(out / "throughput.json", summary)
    print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"],
                      "report": str(out / "throughput.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
