"""Run P1-L experiments with reusable backbones and measured suite scheduling."""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import re
import socket
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm.config import read_mapping, recursive_merge, resolve_config


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_metadata() -> dict[str, Any]:
    from looplora.sym.run import source_metadata as symbolic_source_metadata

    return symbolic_source_metadata()


def _empty_output(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory must be empty: {out}")


def _load_backbone(device: str):
    from looplora.llm.inject import load_model, load_tokenizer

    started = time.perf_counter()
    tokenizer = load_tokenizer()
    model = load_model(device=device)
    return model, tokenizer, time.perf_counter() - started


def _parameter_counts(experiment) -> dict[str, int]:
    from torch.nn import Parameter
    from looplora.llm.train import WRITER_NAMES

    model = {id(p): p for p in experiment.model.parameters()}
    adapter = ({} if experiment.adapter is None else
               {id(p): p for p in experiment.adapter.parameters()})
    external = {}
    if experiment.writer is not None and experiment.writer is not experiment.adapter:
        for name in WRITER_NAMES:
            parameter = getattr(experiment.writer, name, None)
            if isinstance(parameter, Parameter) and id(parameter) not in model:
                external[id(parameter)] = parameter
    active = {**model, **external}
    return {"total": sum(p.numel() for p in active.values()),
            "trainable": sum(p.numel() for p in active.values() if p.requires_grad),
            "model_total": sum(p.numel() for p in model.values()),
            "backbone": sum(p.numel() for key, p in model.items() if key not in adapter),
            "adapter": sum(p.numel() for p in adapter.values()),
            "external_writer": sum(p.numel() for p in external.values()),
            "active_adapter": sum(p.numel() for p in {**adapter, **external}.values())}


def run_one(config: dict[str, Any], out: str | Path, *, model=None, tokenizer=None) -> dict[str, Any]:
    """Retain the symbolic record envelope plus stage, protocol and LM provenance."""
    config = resolve_config(config)
    if (model is None) != (tokenizer is None):
        raise ValueError("reused model and tokenizer must be supplied together")
    out = Path(out).resolve()
    _empty_output(out)
    provenance = source_metadata()
    config_hash = stable_hash(config)
    _write_json(out / "config.json", {"config": config, "config_hash": config_hash, **provenance})
    from looplora.data.relations import GENERATOR_VERSION

    consumed = config["writer_checkpoint"]
    record: dict[str, Any] = {
        "experiment_id": config["name"], "status": "running", "config_hash": config_hash,
        **provenance, "data_seed": derive_seed(config["seed"], "llm", "writer-data" if config["stage"] == "writer" else "train-data"),
        "eval_data_seed": config["eval"]["seed"], "data_version": GENERATOR_VERSION,
        "train_seed": config["seed"], "backbone_seed": None,
        "checkpoint": str(out / "checkpoint.pt"), "samples": str(out / "samples.npz"),
        "insertion_point": f"model.layers.{config['model']['layer']}.self_attn.o_proj",
        "consumed_writer_checkpoint": consumed,
        "writer_source_record": str(Path(consumed).resolve().parent / "record.json") if consumed else None,
        "writer_checkpoint": None,
        "writer": ("oracle_random" if config["writer"] == "oracle_random" else
                   "t1_frozen" if config["stage"] == "reader" else config["stage"]),
        "stage": config["stage"], "protocol": config["protocol"],
        "eval_protocols": config["eval"]["protocols"], "smoke": config["smoke"],
        "variant": config["variant"], "task": config["task"]["name"],
        "parameters": None, "state_bytes": None, "rank": config["model"]["r"],
        "r_mem": config["model"]["r"], "L": config["eval"]["depths"],
        "mean_L": None, "train_tokens": None, "train_seconds": None,
        "pretrain_tokens": 0, "pretrain_seconds": 0., "eval_samples": None,
        "pretrain_cost_accounting": (
            "Consumed writer training is charged only to writer_source_record; no cost allocated to this reader."
            if consumed else "No separate pretraining; this run's training cost is train_seconds."),
        "scores": None, "latency": None, "peak_memory_bytes": None, "failure_reason": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out / "record.json", record)
    started = time.perf_counter()
    experiment = None
    try:
        import numpy as np
        import torch
        import transformers
        from looplora.llm.evaluate import evaluate
        from looplora.llm.inject import MODEL_ID, MODEL_REVISION
        from looplora.llm.train import train

        device = torch.device(config["device"])
        if device.type == "cuda":
            torch.cuda.init()
            if device.index is not None:
                torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        load_seconds = 0.
        reused = model is not None
        if not reused:
            model, tokenizer, load_seconds = _load_backbone(config["device"])
        record.update(model_id=MODEL_ID, model_revision=MODEL_REVISION,
                      tokenizer_id=MODEL_ID, tokenizer_revision=MODEL_REVISION)
        record["runtime"] = {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "host": socket.gethostname(),
            "job_id": os.environ.get("PBS_JOBID"), "device": str(device),
            "dtype": "bfloat16 backbone / float32 adapter", "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
        experiment, summary = train(config, out, model=model, tokenizer=tokenizer)
        if config["writer"] == "oracle_random":
            from looplora.sym.evaluate import tensor_hash

            record["writer_signature"] = experiment.writer_signature()
            record["writer_hash"] = stable_hash({name: tensor_hash(value)
                                                  for name, value in experiment.writer_state().items()})
        record["data_seed"] = summary.get("data_seed", record["data_seed"])
        metrics, timing = evaluate(experiment, config, out)
        with np.load(out / "samples.npz", allow_pickle=False) as archive:
            samples = {key: archive[key] for key in archive.files}
        _write_json(out / "metrics.json", metrics)
        timing = dict(timing)
        timing.update({key: summary[key] for key in
                       ("train_seconds", "train_step_seconds", "pretrain_seconds") if key in summary})
        train_seconds = summary.get("train_seconds", 0.)
        timing.update(model_load_seconds=load_seconds, backbone_reused=reused,
                      seconds_per_1000_steps=(1000 * train_seconds / config["train"]["steps"]
                                              if config["stage"] != "baseline" else None),
                      total_seconds=time.perf_counter() - started)
        timing["measurement"] = {
            "model_loading": "included in direct run and suite walltime; shared load counted once per worker",
            "suite_cost": "one allocated node; overlapping worker walltimes are not summed as node-hours",
            "train_step_scope": "episode generation, support encoding, answer forward/backward and optimizer",
        }
        _write_json(out / "timing.json", timing)
        record.update(
            status="complete", parameters=_parameter_counts(experiment),
            state_bytes=(0 if config["variant"] in ("B0", "B1", "B2") else
                         config["eval"]["batch_size"] * config["model"]["r"] ** 2 * 4),
            mean_L=summary.get("mean_L"), train_tokens=summary.get("train_tokens"),
            train_seconds=summary.get("train_seconds"), writer_checkpoint=summary.get("writer_checkpoint"),
            eval_samples=int(len(samples["answer"])), scores=metrics["table"], latency=timing,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None)
    except BaseException as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        _write_json(out / "timing.json", {"total_seconds": time.perf_counter() - started})
        raise
    finally:
        try:
            if experiment is not None:
                experiment.close()
        except BaseException as error:
            record.update(status="failed", failure_reason=f"adapter cleanup failed: {type(error).__name__}: {error}")
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(out / "record.json", record)
    return record


def load_suite(path: str | Path, out: str | Path) -> list[dict[str, Any]]:
    """Resolve named artifacts and reject cycles before loading a model."""
    import yaml

    path, out = Path(path).resolve(), Path(out).resolve()
    entries = yaml.safe_load(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("suite must be a nonempty list")
    jobs, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"config", "name", "overrides", "depends_on", "writer_from"}:
            raise ValueError("invalid suite entry fields")
        name = entry.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("suite names must be unique safe directory names")
        names.add(name)
        if not isinstance(entry.get("config"), str) or not isinstance(entry.get("overrides", {}), dict):
            raise ValueError("suite entries require a config path and mapping overrides")
        dependencies = entry.get("depends_on", [])
        if not isinstance(dependencies, list) or any(not isinstance(d, str) for d in dependencies):
            raise ValueError("depends_on must be a list of names")
        dependencies = list(dependencies)
        raw = recursive_merge(read_mapping(path.parent / entry["config"]), entry.get("overrides", {}))
        raw["name"] = name
        if "writer_from" in entry:
            writer = entry["writer_from"]
            if not isinstance(writer, str):
                raise ValueError("writer_from must name a suite run")
            raw["writer_checkpoint"] = str(out / writer / "writer.pt")
            if writer not in dependencies:
                dependencies.append(writer)
        jobs.append({"name": name, "config": resolve_config(raw), "dependencies": dependencies})
    by_name = {job["name"]: job for job in jobs}
    for job in jobs:
        if any(dep not in names or dep == job["name"] for dep in job["dependencies"]):
            raise ValueError("unknown or self-referential suite dependency")
        writer = job["config"]["writer_checkpoint"]
        if writer and Path(writer).parent.name in by_name:
            if by_name[Path(writer).parent.name]["config"]["stage"] != "writer":
                raise ValueError("writer_from must reference a writer-stage run")
    resolved = set()
    while len(resolved) < len(jobs):
        ready = {j["name"] for j in jobs if set(j["dependencies"]) <= resolved} - resolved
        if not ready:
            raise ValueError("cyclic suite dependencies")
        resolved.update(ready)
    if len({job["config"]["device"] for job in jobs}) != 1:
        raise ValueError("a suite must share one device")
    _comparison_groups(jobs)
    return jobs


def _memory_only_samples(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        mask = (archive["protocol"] == "memory_only" if "protocol" in archive.files else
                np.char.startswith(archive["condition"].astype(str), "memory_only/"))
        return {key: archive[key][mask] for key in archive.files}


def _comparison_groups(jobs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for job in jobs:
        config = job["config"]
        if config["stage"] not in ("reader", "joint"):
            continue
        key = stable_hash({"stage": config["stage"], "writer": config["writer_checkpoint"],
                           "writer_kind": config.get("writer", "learned"),
                           "oracle_seed": config.get("oracle_seed", 17) if config.get("writer") == "oracle_random" else None,
                           "seed": config["seed"], "task": config["task"],
                           "model": config["model"], "write_policy": config["write_policy"],
                           "eval": {k: v for k, v in config["eval"].items() if k not in ("protocols", "depths")}})
        variants = groups.setdefault(key, {})
        variant = config["variant"]
        if variant in variants:
            raise ValueError(f"duplicate variant {variant} in comparison group: "
                             f"{variants[variant]['name']} and {job['name']}")
        variants[variant] = job
    return groups


def compare_suite(jobs: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    """Assert frozen-writer Gate 0 and compute paired memory-only variant effects."""
    groups = _comparison_groups(jobs)
    from looplora.sym.stats import compare_runs

    result: dict[str, Any] = {"frozen_writer_hashes_verified": [], "comparisons": []}
    for variants in groups.values():
        frozen = [job for job in variants.values() if job["config"]["stage"] == "reader"]
        reference = None
        for job in frozen:
            metrics = json.loads((out / job["name"] / "metrics.json").read_text())
            fields = ("split", "protocol", "condition", "episode_start", "M", "write_count")
            if job["config"].get("writer", "learned") != "oracle_random":
                fields += ("x", "u")
            hashes = [{k: entry[k] for k in fields}
                      for entry in metrics["invariant_hashes"] if entry["protocol"] == "memory_only"]
            hashes.sort(key=lambda e: (e["split"], e["condition"], e["episode_start"]))
            if not hashes:
                raise AssertionError("frozen-writer comparison requires memory-only invariant hashes")
            if reference is not None and hashes != reference:
                raise AssertionError(f"frozen-writer x/u/M/write_count hashes differ: {job['name']}")
            reference = hashes
        if len(frozen) > 1:
            result["frozen_writer_hashes_verified"].append([j["name"] for j in frozen])
        if "M1" not in variants:
            continue
        main = variants["M1"]
        for variant in ("B4", "B5", "B3", "M1-lin"):
            if variant not in variants:
                continue
            other = variants[variant]
            depth_b = 1 if variant == "B4" else 4
            if 4 not in main["config"]["eval"]["depths"] or depth_b not in other["config"]["eval"]["depths"]:
                continue
            comparisons = compare_runs(_memory_only_samples(out / main["name"] / "samples.npz"),
                                       _memory_only_samples(out / other["name"] / "samples.npz"),
                                       depth_a=4, depth_b=depth_b, label_a=main["name"],
                                       label_b=other["name"], **main["config"]["stats"])
            result["comparisons"].append({"a": main["name"], "b": other["name"],
                                          "depth_a": 4, "depth_b": depth_b, "results": comparisons})
    return result


_WORKER_MODEL = None
_WORKER_TOKENIZER = None
_WORKER_LOAD_SECONDS = 0.


def _worker_initialize(device: str) -> None:
    global _WORKER_MODEL, _WORKER_TOKENIZER, _WORKER_LOAD_SECONDS
    _WORKER_MODEL, _WORKER_TOKENIZER, _WORKER_LOAD_SECONDS = _load_backbone(device)


def _worker_run(job: dict[str, Any], out: str) -> dict[str, Any]:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {job['name']} start", flush=True)
    try:
        record = run_one(job["config"], Path(out) / job["name"],
                         model=_WORKER_MODEL, tokenizer=_WORKER_TOKENIZER)
        print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {job['name']} complete", flush=True)
        return {"status": record["status"], "returncode": 0, "pid": os.getpid(),
                "model_load_seconds": _WORKER_LOAD_SECONDS}
    except Exception as error:
        print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {job['name']} fail", flush=True)
        traceback.print_exc()
        return {"status": "failed", "returncode": 1, "pid": os.getpid(),
                "model_load_seconds": _WORKER_LOAD_SECONDS,
                "failure_reason": f"{type(error).__name__}: {error}"}


def _execute_jobs(jobs: list[dict[str, Any]], out: Path, procs: int) -> dict[str, Any]:
    """Use one persistent model per process, including model loads in walltime."""
    if procs not in (1, 2):
        raise ValueError("procs must be 1 or 2")
    _empty_output(out)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [dict(job, config=recursive_merge(job["config"], {})) for job in jobs]
    for job in jobs:
        if job["config"]["device"].startswith("cuda"):
            job["config"]["device"] = "cuda:0"
    device = jobs[0]["config"]["device"]
    started = time.perf_counter()
    pending, running, results = list(jobs), {}, {}
    executor = None
    try:
        if procs == 1:
            _worker_initialize(device)
        else:
            executor = ProcessPoolExecutor(max_workers=procs, mp_context=multiprocessing.get_context("spawn"),
                                           initializer=_worker_initialize, initargs=(device,))
        while pending or running:
            for job in list(pending):
                dependencies = job["dependencies"]
                if any(d in results and results[d]["status"] != "complete" for d in dependencies):
                    results[job["name"]] = {"status": "skipped", "returncode": 1,
                                            "failure_reason": "dependency failed"}
                    pending.remove(job)
                    continue
                if len(running) >= procs or any(d not in results for d in dependencies):
                    continue
                pending.remove(job)
                if executor is None:
                    results[job["name"]] = _worker_run(job, str(out))
                else:
                    running[executor.submit(_worker_run, job, str(out))] = job["name"]
            if running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    name = running.pop(future)
                    try:
                        results[name] = future.result()
                    except Exception as error:
                        results[name] = {"status": "failed", "returncode": 1,
                                         "failure_reason": f"worker failed: {type(error).__name__}: {error}"}
    except BaseException as error:
        for job in jobs:
            results.setdefault(job["name"], {"status": "failed" if job["name"] in running.values() else "skipped",
                                             "returncode": 1, "failure_reason": f"suite interrupted: {error}"})
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        else:
            global _WORKER_MODEL, _WORKER_TOKENIZER
            _WORKER_MODEL = _WORKER_TOKENIZER = None
            gc.collect()
            import torch
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        seconds = time.perf_counter() - started
        summary = {"runs": results, "seconds": seconds, "node_hours": seconds / 3600,
                   "procs": procs, "visible_gpu": visible,
                   "walltime_scope": "worker startup, model/tokenizer loading, all runs and worker shutdown",
                   "model_load_seconds_by_pid": {str(r["pid"]): r["model_load_seconds"]
                       for r in results.values() if "pid" in r}}
        _write_json(out / "suite.json", summary)
    return summary


def measure_scheduling(out: str | Path, *, device: str = "cuda") -> dict[str, Any]:
    """Bounded system smoke: two untrained B0 runs, 32 val episodes, depth one."""
    out = Path(out).resolve()
    _empty_output(out)
    config = resolve_config({"name": "scheduling", "stage": "baseline", "variant": "B0",
                             "device": device, "smoke": True, "seed": 20261027,
                             "train": {"steps": 100, "warmup_steps": 2},
                             "eval": {"episodes": 32, "depths": [1], "counterfactuals": False}})
    jobs = [{"name": f"baseline-{i}", "config": dict(config, name=f"baseline-{i}"), "dependencies": []}
            for i in range(2)]
    result: dict[str, Any] = {
        "purpose": "system smoke scheduling comparison; no research selection",
        "config": config, "runs_per_mode": 2, "modes": {}, "selected_procs": None,
        "walltime_scope": "includes process startup/shutdown and one model/tokenizer load per worker",
    }
    for procs in (1, 2):
        mode_started = time.perf_counter()
        try:
            summary = _execute_jobs(jobs, out / f"procs-{procs}", procs)
            successful = len(summary["runs"]) == len(jobs) and all(r["status"] == "complete" for r in summary["runs"].values())
            result["modes"][str(procs)] = {"status": "complete" if successful else "failed", **summary}
        except Exception as error:
            mode_summary = out / f"procs-{procs}" / "suite.json"
            seconds = time.perf_counter() - mode_started
            summary = (json.loads(mode_summary.read_text()) if mode_summary.exists() else
                       {"seconds": seconds, "node_hours": seconds / 3600})
            result["modes"][str(procs)] = {**summary, "status": "failed",
                                          "failure_reason": f"{type(error).__name__}: {error}"}
        _write_json(out / "scheduling.json", result)
    complete = {int(p): m["seconds"] for p, m in result["modes"].items() if m["status"] == "complete"}
    if not complete:
        raise RuntimeError(f"both scheduling modes failed; see {out / 'scheduling.json'}")
    result["selected_procs"] = min(complete, key=complete.get)
    result["total_node_hours"] = sum(m.get("node_hours", 0.) for m in result["modes"].values())
    _write_json(out / "scheduling.json", result)
    return result


def run_suite(path: str | Path, out: str | Path, procs: int | str = 1,
              *, measure: bool = False) -> int:
    out = Path(out).resolve()
    jobs = load_suite(path, out / "runs" if procs == "auto" or measure else out)
    scheduling = None
    if procs == "auto" or measure:
        _empty_output(out)
        scheduling = measure_scheduling(out / "scheduling", device=jobs[0]["config"]["device"])
        selected = scheduling["selected_procs"] if procs == "auto" else int(procs)
        run_out = out / "runs"
    else:
        selected, run_out = int(procs), out
    suite_started = time.perf_counter()
    summary = _execute_jobs(jobs, run_out, selected)
    ok = len(summary["runs"]) == len(jobs) and all(r["status"] == "complete" for r in summary["runs"].values())
    try:
        if ok:
            comparisons = compare_suite(jobs, run_out)
            _write_json(run_out / "comparisons.json", comparisons)
        summary["status"] = "complete" if ok else "failed"
    except BaseException as error:
        summary.update(status="failed", failure_reason=f"suite comparison failed: {type(error).__name__}: {error}")
        raise
    finally:
        summary["execution_seconds"] = summary["seconds"]
        summary["seconds"] = time.perf_counter() - suite_started
        summary["node_hours"] = summary["seconds"] / 3600
        summary["walltime_scope"] += "; paired comparisons and Gate 0 checks"
        _write_json(run_out / "suite.json", summary)
        if scheduling is not None:
            _write_json(out / "suite.json", {**summary, "run_root": str(run_out),
                        "scheduling_evidence": str(out / "scheduling" / "scheduling.json"),
                        "total_node_hours": summary["node_hours"] + scheduling["total_node_hours"]})
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config")
    source.add_argument("--suite")
    parser.add_argument("--out", required=True)
    parser.add_argument("--procs", choices=("1", "2", "auto"), default="1")
    parser.add_argument("--measure-scheduling", action="store_true")
    args = parser.parse_args(argv)
    if args.suite:
        return run_suite(args.suite, args.out, args.procs, measure=args.measure_scheduling)
    if args.config:
        if args.measure_scheduling or args.procs != "1":
            parser.error("scheduling options require --suite or standalone --measure-scheduling")
        run_one(read_mapping(args.config), args.out)
        return 0
    if args.measure_scheduling:
        measure_scheduling(args.out)
        return 0
    parser.error("supply --config, --suite, or --measure-scheduling")
    return 2


if __name__ == "__main__":
    sys.exit(main())
