"""Locked P1-S main matrix, blind confirmation collection, and global release."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

from looplora.infra.seeds import derive_seed, stable_hash
from .config import resolve_config
from .pilot import base_config, VARIANTS, WRITERS

SEEDS = (17, 29, 43)
DEPTHS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
LRS = {
    "oracle_random": dict(zip(VARIANTS, (.003, .003, .003, .001, .0003, .003))),
    "oracle_orth": dict(zip(VARIANTS, (.003, .003, .001, .001, .001, .001))),
    "t1_frozen": dict(zip(VARIANTS, (.003, .003, .003, .003, .0003, .003))),
    "joint": dict(zip(VARIANTS, (.003, .001, .003, .003, .0003, .003))),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def make_plan(root: Path, steps: int, throughput_decision: dict) -> dict:
    if steps not in (15000, 20000):
        raise ValueError("main budget must be 15000 or 20000")
    if throughput_decision.get("steps") != steps:
        raise ValueError("training steps must match the pre-matrix throughput decision")
    root = root.resolve()
    runs = []

    def add(group: str, writer: str, variant: str, seed: int, rank: int = 32) -> None:
        name = f"{group}-{writer}-{variant}-r{rank}-s{seed}"
        config = base_config(name, writer, variant, rank, 12,
                             .003 if group == "writer" else LRS[writer][variant],
                             "t1" if group == "writer" else "t2_chain_end",
                             None if group == "writer" else [8, 4])
        config["seed"] = seed
        config["train"]["steps"] = config["pretrain"]["steps"] = steps
        config["pretrain"]["lr"] = .003
        config["pretrain"]["task"].update(E=12, n_rel=4, head_sharing=.3)
        config["eval"].update(episodes=2000, splits=["test", "test_unseen"],
                              seed=4242, counterfactuals=True, depths=DEPTHS.copy(),
                              fixed_depth_sweep=True,
                              swap_successors=group in {"primary", "replication"})
        dependencies = []
        if group == "replication":
            if writer == "oracle_random":
                config["task"].update(E=18, chain_lengths=[8, 8, 2])
            else:
                config["task"].update(name="t2_typed", E=12, n_distract=3,
                                      chain_lengths=[8, 1])
        if writer == "t1_frozen":
            source = f"writer-joint-B4-r32-s{seed}"
            config["pretrain"]["writer_checkpoint"] = str(root / "runs" / source / "writer.pt")
            dependencies.append(source)
        if group == "writer":
            config["eval"].update(episodes=1000, splits=["val"], counterfactuals=False,
                                  depths=[1], hops=[1], fixed_depth_sweep=False)
        config = resolve_config(config)
        runs.append(dict(name=name, group=group, writer=writer, variant=variant,
                         seed=seed, r=rank, config=config,
                         run_dir=str(root / "runs" / name), dependencies=dependencies))

    for seed in SEEDS:
        add("writer", "joint", "B4", seed)
    for writer in WRITERS:
        for variant in VARIANTS:
            for seed in SEEDS:
                add("primary", writer, variant, seed)
    for writer in ("t1_frozen", "joint", "oracle_random"):
        for variant in ("M1", "M1-lin"):
            for seed in SEEDS:
                add("replication", writer, variant, seed)
    for writer in ("oracle_random", "oracle_orth", "joint"):
        for variant in ("M1", "M1-lin", "B5", "B5-lin"):
            add("secondary", writer, variant, 17, 16)
    return dict(root=str(root), steps=steps, prereg_commit="c24ecba",
                selection_source="docs/prereg/P1S.md final v1; fixed Pilot-2 LR table",
                throughput_decision=throughput_decision, runs=runs, created_at=now())


def next_lower_lr(lr: float) -> float:
    grid = [.0003, .001, .003]
    if lr not in grid or lr == grid[0]:
        raise ValueError("non-finite loss at lowest/out-of-grid LR requires orchestrator decision")
    return grid[grid.index(lr) - 1]


def run_entry(entry: dict) -> dict:
    import torch
    from looplora.data.relations import GENERATOR_VERSION
    from .evaluate import evaluate
    from .run import source_metadata
    from .train import train

    out = Path(entry["run_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    config = copy.deepcopy(entry["config"])
    source = source_metadata()
    torch.set_num_threads(config["runtime"]["threads"])
    torch.cuda.init()
    torch.cuda.set_device(torch.device(config["device"]))
    torch.cuda.reset_peak_memory_stats()
    record = dict(experiment_id=entry["name"], group=entry["group"], status="running",
                  **source, config=config, config_hash=stable_hash(config),
                  planned_config_hash=stable_hash(entry["config"]),
                  training_complete=False, confirmation_samples_written=False,
                  train_seed=config["seed"], data_seed=derive_seed(config["seed"], "train_data"),
                  eval_data_seed=4242, data_version=GENERATOR_VERSION, model_revision="symbolic-v1",
                  backbone_seed=config["backbone_seed"], writer=config["writer"],
                  variant=config["variant"], task=config["task"]["name"], rank=entry["r"],
                  r_mem=config["model"]["r_mem"], L=config["eval"]["depths"],
                  insertion_point="symbolic_projection", checkpoint=str(out / "checkpoint.pt"),
                  consumed_writer_checkpoint=config["pretrain"]["writer_checkpoint"],
                  writer_source_record=(str(Path(config["pretrain"]["writer_checkpoint"]).parent / "record.json")
                                        if config["pretrain"]["writer_checkpoint"] else None),
                  created_at=now(), attempts=[], failure_reason=None,
                  runtime=dict(host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"),
                               torch=torch.__version__, python=sys.version.split()[0],
                               device=config["device"], threads=torch.get_num_threads(),
                               gpu=torch.cuda.get_device_name(), dtype="float32"))
    started = time.perf_counter()
    save(out / "record.json", record)
    try:
        for attempt in range(2):
            save(out / "config.json", dict(config=config, config_hash=stable_hash(config), **source))
            try:
                model, summary = train(config, out)
                record["attempts"].append(dict(attempt=attempt, lr=config["train"]["lr"], status="complete"))
                break
            except FloatingPointError as error:
                if "loss" not in str(error) or attempt:
                    raise
                record["attempts"].append(dict(attempt=attempt, lr=config["train"]["lr"],
                                               status="nonfinite_loss", error=str(error)))
                lower = next_lower_lr(config["train"]["lr"])
                archive = out / "attempt-0"
                archive.mkdir()
                for path in list(out.iterdir()):
                    if path != archive:
                        shutil.move(str(path), archive / path.name)
                save(archive / "attempt.json", record["attempts"][-1])
                config["train"]["lr"] = lower
                save(out / "record.json", record)
        record.update(training_complete=True, training_finished_at=now(),
                      config=config, config_hash=stable_hash(config), summary=summary,
                      mean_L=summary["mean_L"], train_tokens=summary["train_tokens"],
                      train_seconds=summary["train_seconds"], pretrain_seconds=0., pretrain_tokens=0,
                      writer_checkpoint=summary.get("writer_checkpoint"),
                      parameters={"total":sum(p.numel() for p in model.parameters()),
                                  "trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)},
                      state_bytes=config["eval"]["batch_size"]*entry["r"]**2*4)
        save(out / "record.json", record)
        provenance = dict(checkpoint=str(out / "checkpoint.pt"), checkpoint_config=copy.deepcopy(config),
                          config_hash=stable_hash(config), **source)
        if entry["group"] == "writer":
            metrics, timing = evaluate(model, config, out / "writer_validation", provenance=provenance)
            save(out / "writer_validation/timing.json", timing)
            record.update(scores=metrics["table"], latency=timing,
                          samples=str(out / "writer_validation/samples.npz"),
                          eval_samples=sum(row["n"] for row in metrics["table"]))
        else:
            screen, timing = evaluate(model, config, out / "screening", provenance=provenance)
            save(out / "screening/timing.json", timing)
            save(out / "screening/config.json", dict(config=config, **provenance))
            confirmation = copy.deepcopy(config)
            confirmation["eval"].update(splits=["test"], seed=777001, counterfactuals=False)
            receipt, confirmation_timing = evaluate(
                model, confirmation, out / "confirmation", samples_only=True,
                provenance={**provenance, "config_hash":stable_hash(confirmation),
                            "eval_overrides":{"eval.splits":["test"],"eval.seed":777001,
                                              "eval.counterfactuals":False}})
            record.update(confirmation_samples_written=True, confirmation_written_at=now(),
                          samples=str(out / "screening/samples.npz"),
                          confirmation_samples=str(out / "confirmation/samples.npz"),
                          scores=screen["table"], latency={"screening":timing,"confirmation":confirmation_timing},
                          eval_samples=sum(row["n"] for row in screen["table"]),
                          confirmation_eval_samples=confirmation["eval"]["episodes"]*8*len(DEPTHS))
        record.update(status="complete", total_seconds=time.perf_counter()-started,
                      peak_memory_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        record.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_at"] = now()
        save(out / "record.json", record)
    return record


def run_shard(plan_path: Path, names: list[str], procs: int) -> None:
    if procs < 1:
        raise ValueError("procs must be positive")
    plan = json.loads(plan_path.read_text())
    entries = {row["name"]: row for row in plan["runs"]}
    if not names or len(names) != len(set(names)) or set(names) - entries.keys():
        raise ValueError("invalid or duplicate shard names")
    logs = Path(plan["root"]) / "logs"
    logs.mkdir(exist_ok=True)
    pending, running, completed = list(names), {}, set()
    try:
        while pending or running:
            for name, (process, log) in list(running.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del running[name]
                    if code:
                        raise RuntimeError(f"main run {name} exited {code}; see {logs / (name + '.log')}")
                    completed.add(name)
            for name in list(pending):
                dependencies = entries[name]["dependencies"]
                ready = True
                for dep in dependencies:
                    rec = Path(entries[dep]["run_dir"]) / "record.json"
                    if not rec.exists() or json.loads(rec.read_text()).get("status") != "complete":
                        ready = False
                        break
                if not ready or len(running) >= procs:
                    continue
                log = (logs / (name + ".log")).open("x")
                process = subprocess.Popen([sys.executable, "-m", "looplora.sym.main", "run",
                                            "--plan", str(plan_path), "--name", name],
                                           stdout=log, stderr=subprocess.STDOUT)
                running[name] = process, log
                pending.remove(name)
            if pending and not running:
                raise RuntimeError("unsatisfied writer dependency; submit consumers after writer completion")
            if running:
                time.sleep(1)
    finally:
        for process, log in running.values():
            process.terminate()
            process.wait()
            log.close()
    print(json.dumps(dict(status="complete", names=sorted(completed))))


def seal(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    if len(plan["runs"]) != 105 or len({r["name"] for r in plan["runs"]}) != 105:
        raise ValueError("main matrix must contain 105 distinct runs")
    out = Path(plan["root"]) / "barrier.json"
    if out.exists():
        raise FileExistsError(out)
    confirmations = []
    for entry in plan["runs"]:
        folder = Path(entry["run_dir"])
        record = json.loads((folder / "record.json").read_text())
        if record["status"] != "complete" or not record["training_complete"]:
            raise ValueError(f"training is incomplete: {entry['name']}")
        if record["summary"]["steps"] != plan["steps"] or not (folder / "checkpoint.pt").is_file():
            raise ValueError("wrong training budget or missing checkpoint")
        if record["planned_config_hash"] != stable_hash(entry["config"]):
            raise ValueError("plan changed after training")
        if entry["group"] != "writer":
            sample = folder / "confirmation/samples.npz"
            if not record["confirmation_samples_written"] or not sample.is_file():
                raise ValueError(f"confirmation samples missing: {entry['name']}")
            if (folder / "confirmation/metrics.json").exists():
                raise ValueError("confirmation metrics computed before the global barrier")
            metadata = json.loads((folder / "confirmation/evaluation.json").read_text())
            if "table" in metadata or "diagnostics" in metadata:
                raise ValueError("confirmation metadata contains premature aggregates")
            confirmations.append(str(sample))
    if len(confirmations) != 102:
        raise ValueError("main matrix requires 102 confirmation sample sets")
    barrier = dict(status="passed", created_at=now(), plan_hash=stable_hash(plan),
                   runs=[row["name"] for row in plan["runs"]], confirmation_samples=confirmations)
    save(out, barrier)
    print(json.dumps(dict(status="passed", runs=105, confirmation_sets=102, barrier=str(out))))
    return barrier


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "shard", "seal"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--names", type=Path)
    parser.add_argument("--procs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.action == "plan":
        if args.plan.exists():
            raise FileExistsError(args.plan)
        decision = json.loads(args.decision.read_text())
        plan = make_plan(args.plan.parent, decision["steps"], decision)
        save(args.plan, plan)
        print(json.dumps(dict(runs=len(plan["runs"]), steps=plan["steps"], plan=str(args.plan))))
    elif args.action == "run":
        plan = json.loads(args.plan.read_text())
        entry = next(row for row in plan["runs"] if row["name"] == args.name)
        run_entry(entry)
    elif args.action == "shard":
        run_shard(args.plan, json.loads(args.names.read_text()), args.procs)
    else:
        seal(args.plan)


if __name__ == "__main__":
    main()
