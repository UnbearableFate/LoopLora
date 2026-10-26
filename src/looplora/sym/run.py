"""Run one symbolic experiment or a bounded suite sharing one visible GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from looplora.infra.seeds import derive_seed, stable_hash
from looplora.sym.config import read_mapping, recursive_merge, resolve_config


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def source_metadata() -> dict[str, Any]:
    """Identify either a git worktree or submit.py's exported commit archive."""
    root = Path(os.environ.get("LL_SOURCE_ROOT", Path(__file__).resolve().parents[3])).resolve()
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout
    if (root / ".git").exists():
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal").strip())
        names = set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0"))
    else:
        spec_path = os.environ.get("LL_SPEC")
        if not spec_path:
            raise ValueError("source has no .git; LL_SPEC is required for commit-archive provenance")
        spec = json.loads(Path(spec_path).read_text())
        source = spec.get("source", {})
        commit = source.get("ref")
        if source.get("kind") != "commit" or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("LL_SPEC must identify an exported full commit")
        dirty = False
        names = {name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file()}
        for directory in ("src", "scripts", "tests", "configs"):
            folder = root / directory
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".toml", ".lock"}:
                        names.add(path.relative_to(root).as_posix())
                        if len(names) > 10000:
                            raise ValueError("source archive exceeds the provenance file-count bound")
    digest = hashlib.blake2b(digest_size=16)
    total_bytes = 0
    for name in sorted(names):
        path = root / name
        relative = Path(name)
        if (not name or path.suffix not in {".py", ".yaml", ".yml", ".toml", ".lock"}
                or not (relative.parts[0] in {"src", "scripts", "tests", "configs"}
                        or name in {"pyproject.toml", "uv.lock"})):
            continue
        digest.update(name.encode() + b"\0")
        if path.is_file():
            total_bytes += path.stat().st_size
            if total_bytes > 64 * 1024 * 1024:
                raise ValueError("source exceeds the 64 MiB provenance bound")
            digest.update(path.read_bytes())
        else:
            digest.update(b"<deleted>")
        digest.update(b"\0")
    return {"git_commit": commit, "source_dirty": dirty, "source_hash": digest.hexdigest()}


def _empty_output(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory must be empty: {out}")


def run_one(config: dict[str, Any], out: str | Path) -> dict[str, Any]:
    """Write provenance before training and a failure record if execution fails."""
    config = resolve_config(config)
    out = Path(out).resolve()
    _empty_output(out)
    provenance = source_metadata()
    config_hash = stable_hash(config)
    _write_json(out / "config.json", {"config": config, "config_hash": config_hash, **provenance})
    from looplora.data.relations import GENERATOR_VERSION

    consumed_writer = (config["pretrain"]["writer_checkpoint"]
                       if config["writer"] in {"t1_frozen", "t1_frozen_fact"} else None)
    record: dict[str, Any] = {
        "experiment_id": config["name"], "status": "running", "config_hash": config_hash,
        **provenance, "data_seed": derive_seed(config["seed"], "train_data"),
        "eval_data_seed": config["eval"]["seed"], "data_version": GENERATOR_VERSION,
        "model_revision": "symbolic-v1", "train_seed": config["seed"],
        "backbone_seed": config["backbone_seed"], "checkpoint": str(out / "checkpoint.pt"),
        "samples": str(out / "samples.npz"), "insertion_point": "symbolic_projection",
        "consumed_writer_checkpoint": consumed_writer,
        "writer_source_record": (str(Path(consumed_writer).resolve().parent / "record.json")
                                 if consumed_writer else None),
        "writer_checkpoint": None,
        "writer": config["writer"], "variant": config["variant"], "task": config["task"]["name"],
        "parameters": None, "state_bytes": None, "rank": config["model"]["r"],
        "r_mem": config["model"]["r_mem"], "L": config["eval"]["depths"],
        "mean_L": None, "train_tokens": None, "train_seconds": None,
        "pretrain_tokens": None, "pretrain_seconds": None,
        "eval_samples": None, "scores": None, "latency": None,
        "peak_memory_bytes": None, "failure_reason": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out / "record.json", record)
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        from looplora.sym.evaluate import evaluate
        from looplora.sym.stats import depth_comparisons
        from looplora.sym.train import train

        torch.set_num_threads(config["runtime"]["threads"])
        device = torch.device(config["device"])
        if device.type == "cuda":
            torch.cuda.init()
            if device.index is not None:
                torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        record["runtime"] = {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "host": socket.gethostname(), "job_id": os.environ.get("PBS_JOBID"),
            "device": str(device), "dtype": "float32", "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
        model, summary = train(config, out)
        metrics, timing = evaluate(model, config, out)
        with np.load(out / "samples.npz", allow_pickle=False) as archive:
            samples = {key: archive[key] for key in archive.files}
        comparisons = depth_comparisons(samples, **config["stats"])
        metrics["depth_comparisons"] = comparisons
        _write_json(out / "metrics.json", metrics)
        timing = dict(timing)
        timing.update({key: summary[key] for key in
                       ("train_seconds", "train_step_seconds", "pretrain_seconds") if key in summary})
        timing["total_seconds"] = time.perf_counter() - started
        timing["measurement"] = {
            "synchronization": "CUDA synchronize before and after each timed pass" if device.type == "cuda" else "host clock",
            "writer_scope": "evaluation preparation, including feature projection and codebooks",
            "reader_scope": "core read, branch, vocabulary logits, and first-read diagnostic",
            "train_step_scope": "episode generation, forward, backward, clipping, AdamW; no warmup steps excluded",
        }
        _write_json(out / "timing.json", timing)
        parameters = list(model.parameters())
        record.update({
            "status": "complete", "parameters": {
                "total": sum(p.numel() for p in parameters),
                "trainable": sum(p.numel() for p in parameters if p.requires_grad)},
            "state_bytes": config["eval"]["batch_size"] * model.r_mem ** 2 * 4,
            "mean_L": summary.get("mean_L"), "train_tokens": summary.get("train_tokens"),
            "train_seconds": summary.get("train_seconds"),
            "pretrain_tokens": (summary.get("pretrain") or {}).get("train_tokens", 0),
            "pretrain_seconds": summary.get("pretrain_seconds", 0.),
            "writer_checkpoint": summary.get("writer_checkpoint"),
            "eval_samples": int(len(samples["answer"])), "scores": metrics["table"],
            "latency": timing, "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
        })
    except BaseException as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        _write_json(out / "timing.json", {"total_seconds": time.perf_counter() - started})
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(out / "record.json", record)
    return record


def load_suite(path: str | Path, out: str | Path) -> list[dict[str, Any]]:
    """Resolve configs relative to the suite; writer_from names a shared artifact."""
    import yaml

    path, out = Path(path).resolve(), Path(out).resolve()
    entries = yaml.safe_load(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("suite must be a nonempty list")
    jobs: list[dict[str, Any]] = []
    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"config", "name", "overrides", "depends_on", "writer_from"}:
            raise ValueError("invalid suite entry fields")
        name = entry.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError(f"suite names must be unique safe directory names: {name!r}")
        names.add(name)
        if not isinstance(entry.get("config"), str):
            raise ValueError(f"suite entry {name} requires a config path")
        overrides = entry.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"suite overrides must be a mapping: {name}")
        dependencies = entry.get("depends_on", [])
        if not isinstance(dependencies, list) or any(not isinstance(d, str) for d in dependencies):
            raise ValueError(f"depends_on must be a list of names: {name}")
        dependencies = list(dependencies)
        raw = recursive_merge(read_mapping(path.parent / entry["config"]), overrides)
        raw["name"] = name
        if "writer_from" in entry:
            writer = entry["writer_from"]
            if isinstance(writer, str):
                dependencies.append(writer)
                checkpoint = out / writer / "writer.pt"
            elif isinstance(writer, dict) and set(writer) == {"checkpoint"} and isinstance(writer["checkpoint"], str):
                checkpoint = (path.parent / writer["checkpoint"]).resolve()
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"external writer checkpoint missing: {checkpoint}")
            else:
                raise ValueError("writer_from must name a suite run or contain a checkpoint path")
            raw = recursive_merge(raw, {"pretrain": {"writer_checkpoint": str(checkpoint)}})
        jobs.append({"name": name, "dependencies": list(dict.fromkeys(dependencies)),
                     "config": resolve_config(raw)})
    pending = {job["name"]: set(job["dependencies"]) for job in jobs}
    for name, dependencies in pending.items():
        if dependencies - names or name in dependencies:
            raise ValueError(f"invalid suite dependencies for {name}")
    while pending:
        ready = {name for name, dependencies in pending.items() if not dependencies}
        if not ready:
            raise ValueError("suite dependencies contain a cycle")
        pending = {name: dependencies - ready for name, dependencies in pending.items() if name not in ready}
    return jobs


def run_suite(path: str | Path, out: str | Path, procs: int = 1) -> int:
    """Run independent children concurrently; failed dependencies are skipped."""
    if isinstance(procs, bool) or not isinstance(procs, int) or procs < 1:
        raise ValueError("procs must be a positive integer")
    out = Path(out).resolve()
    jobs = load_suite(path, out)
    _empty_output(out)
    (out / "_configs").mkdir()
    (out / "_logs").mkdir()
    env = dict(os.environ)
    # All children select the first GPU already made visible to this allocation.
    visible = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env["CUDA_VISIBLE_DEVICES"] = visible
    env.setdefault("OMP_NUM_THREADS", "1")
    pending = list(jobs)
    running: dict[str, tuple[subprocess.Popen, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    try:
        while pending or running:
            for name, (process, log) in list(running.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    results[name] = {"status": "complete" if code == 0 else "failed", "returncode": code}
                    del running[name]
            for job in list(pending):
                dependencies = job["dependencies"]
                if any(dep in results and results[dep]["status"] != "complete" for dep in dependencies):
                    results[job["name"]] = {"status": "skipped", "returncode": 1,
                                            "failure_reason": "dependency failed"}
                    pending.remove(job)
                    continue
                if len(running) >= procs or any(dep not in results for dep in dependencies):
                    continue
                name = job["name"]
                config = job["config"]
                if config["device"].startswith("cuda"):
                    config["device"] = "cuda:0"
                config_path = out / "_configs" / f"{name}.json"
                _write_json(config_path, config)
                log = (out / "_logs" / f"{name}.log").open("w")
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "looplora.sym.run", "--config", str(config_path),
                         "--out", str(out / name)], stdout=log, stderr=subprocess.STDOUT, env=env)
                except BaseException:
                    log.close()
                    raise
                running[name] = (process, log)
                pending.remove(job)
            if running:
                time.sleep(.1)
    finally:
        for name, (process, log) in running.items():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            results[name] = {"status": "failed", "returncode": process.returncode,
                             "failure_reason": "suite interrupted"}
        _write_json(out / "suite.json", {"runs": results, "seconds": time.perf_counter() - started,
                                         "procs": procs, "visible_gpu": visible})
    return 0 if all(value["status"] == "complete" for value in results.values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--suite")
    parser.add_argument("--out", required=True)
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.suite:
        return run_suite(args.suite, args.out, args.procs)
    run_one(read_mapping(args.config), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
