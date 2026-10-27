#!/usr/bin/env python3
"""TSM (F10): terminal-semantics training of the iterative reader (docs/prereg/TSM.md).

Stages (each its own job; scripts/submit.py in a Miyabi-G allocation):

  lock      write the design: settings, the six released M1 training configurations (the template of
            every arm) and the six released B4 reference checkpoints with their blake2b seals;
  smoke     train every (writer, arm) of the seed-17 run for a few steps, concurrently as in the real job,
            then evaluate on validation episodes (seed 4242): timings, memory, gate profile; no confirmation data;
  train     train registered runs (T0 with the unchanged ``train()``, T1/T2 with ``tsm.train_arm``) in ``--procs``
            concurrent child processes; needs the committed design; several jobs may share one root;
            a run that is already complete or a registered training failure is left as it is;
  evaluate  evaluate T0, T1, T2 and the B4 references on the new confirmation episodes (split test, seed 999007).

Training failures (docs/prereg/TSM.md, "Decisions at registration"):

* a non-finite loss in a T1 or T2 run is a registered training failure: no retry, no re-tuning; that arm fails for
  that writer, its completed runs are evaluated and reported descriptively, the other arms are evaluated as registered;
* a non-finite loss in a T0 run stops the study before the confirmation set is opened: ``train`` refuses to start
  anything else and ``evaluate`` refuses to run;
* any other end before completion (walltime, a node or I/O fault) is an infrastructure failure: the run is trained
  again with the identical configuration and code (``train --rerun-incomplete`` once its job has ended), at most twice;
* ``evaluate`` needs every T0 run complete and every T1/T2 run complete or a registered training failure.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from audit_uns import DEFAULT_F2, DEFAULT_P1S
from s9_common import absolute_path, digest_checkpoint, now, require, require_compute, verify_git_lock, write_json
from s9_torch import load_frozen, state_hash, synchronize
from looplora.data.relations import generate
from looplora.infra.seeds import stable_hash
from looplora.sym import tsm, uns
from looplora.sym.config import resolve_config
from looplora.sym.evaluate import stratify_queries
from looplora.sym.model import make_task_spec
from looplora.sym.run import source_metadata
from looplora.sym.train import train

STUDY = "TSM"
WRITERS = ("oracle_random", "joint_fact")
SEEDS = uns.SEEDS
RANK = 32
TRAINING = dict(steps=15000, lr=0.003, batch_size=64, released_depths=[1, 2, 4], t0_depth=4, stay_depth=8,
                hops=[1, 2, 3])
EVALUATION = dict(split="test", seed=999007, episodes=2000, hops=list(range(1, 9)), depths=list(tsm.DEPTHS),
                  batch_size=64)
SMOKE = dict(steps=200, split="val", seed=4242, episodes=200, training_seed=17)
DECISION = dict(stay_tolerance=tsm.STAY_TOLERANCE, far_tolerance=tsm.FAR_TOLERANCE, gain_minimum=tsm.GAIN_MINIMUM,
                comparisons=4, alpha=0.05, confidence=1 - 0.05 / 4, resamples=2000, seed=2718)
SOURCE = Path(__file__).resolve().parents[1] / "src"
WALLTIME_MARGIN = 1.5
POLL_SECONDS = 1.0
MAX_ATTEMPTS = 3  # a run and at most two identical reruns after an infrastructure failure (registered)


def settings() -> dict:
    return dict(study=STUDY, prereg="docs/prereg/TSM.md", writers=list(WRITERS), seeds=list(SEEDS),
                arms=list(tsm.ARMS), training=dict(TRAINING), evaluation=dict(EVALUATION), smoke=dict(SMOKE),
                decision=dict(DECISION), gate_init=list(tsm.GATE_INIT))


def run_name(writer: str, arm: str, seed: int) -> str:
    return f"tsm-{writer}-{arm}-s{seed}"


def reference_name(writer: str, seed: int) -> str:
    return f"B4-{writer}-r{RANK}-s{seed}"


def planned_runs() -> list[dict]:
    return [dict(name=run_name(writer, arm, seed), writer=writer, arm=arm, seed=seed)
            for writer in WRITERS for arm in tsm.ARMS for seed in SEEDS]


def check_template(config: dict, writer: str, seed: int) -> None:
    """A released M1 configuration must carry exactly the registered training settings."""
    train_options, task = config["train"], config["task"]
    identity = (config["writer"], config["variant"], config["seed"], config["model"]["r"])
    require(identity == (writer, "M1", seed, RANK), f"unexpected released configuration {identity}")
    require(train_options["steps"] == TRAINING["steps"] and train_options["lr"] == TRAINING["lr"]
            and train_options["batch_size"] == TRAINING["batch_size"]
            and list(train_options["depths"]) == TRAINING["released_depths"]
            and list(task["hops"]) == TRAINING["hops"] and task["name"] == "t2_chain_end",
            f"the released configuration of {writer}/s{seed} differs from the registered training settings")


def build_design(p1s_root: str | Path, f2_root: str | Path) -> dict:
    templates, references = [], []
    for writer in WRITERS:
        for seed in SEEDS:
            source = (uns.run_root(writer, "M1", seed, p1s_root=p1s_root, f2_root=f2_root) / "checkpoint.pt").resolve()
            require(source.is_file(), f"missing released checkpoint {source}")
            config = torch.load(source, map_location="cpu", weights_only=True)["config"]
            check_template(config, writer, seed)
            templates.append(dict(writer=writer, seed=seed, source=str(source), source_blake2b=digest_checkpoint(source),
                                  config=config, config_hash=stable_hash(config)))
            b4 = (uns.run_root(writer, "B4", seed, p1s_root=p1s_root, f2_root=f2_root) / "checkpoint.pt").resolve()
            require(b4.is_file(), f"missing B4 reference {b4}")
            references.append(dict(name=reference_name(writer, seed), writer=writer, variant="B4", r=RANK, seed=seed,
                                   path=str(b4), blake2b=digest_checkpoint(b4)))
    return {**settings(), "templates": templates, "references": references}


def verify_design(design: dict) -> None:
    keys = ("templates", "references")
    require({key: value for key, value in design.items() if key not in keys} == settings(),
            "the design differs from the registered settings")
    require(sorted((t["writer"], t["seed"]) for t in design["templates"]) == sorted(
        (writer, seed) for writer in WRITERS for seed in SEEDS), "the design must hold six templates")
    require(sorted((r["writer"], r["seed"]) for r in design["references"]) == sorted(
        (writer, seed) for writer in WRITERS for seed in SEEDS), "the design must hold six B4 references")
    for template in design["templates"]:
        check_template(template["config"], template["writer"], template["seed"])
        require(stable_hash(template["config"]) == template["config_hash"], "a template configuration changed")
        require(digest_checkpoint(Path(template["source"])) == template["source_blake2b"],
                f"released M1 checkpoint changed: {template['writer']}/s{template['seed']}")
    for reference in design["references"]:
        require(digest_checkpoint(Path(reference["path"])) == reference["blake2b"],
                f"B4 reference changed: {reference['name']}")


def make_config(design: dict, writer: str, seed: int, arm: str, *, device: str, steps: int | None = None) -> dict:
    """The released M1 configuration with the arm's name, training depth and the registered evaluation block.

    T0 trains at the fixed depth 4 (the recipe of the P2-S-F1 M1-L4 runs: the released recipe with depths [4]);
    T1 and T2 train at the fixed depth 8. Nothing else differs between the arms' configurations.
    """
    template = next(t for t in design["templates"] if t["writer"] == writer and t["seed"] == seed)
    config = copy.deepcopy(template["config"])
    config["name"], config["device"] = run_name(writer, arm, seed), device
    training = design["training"]
    config["train"]["depths"] = [training["t0_depth"] if arm == "T0" else training["stay_depth"]]
    if steps is not None:
        config["train"]["steps"] = steps
    evaluation = design["evaluation"]
    config["eval"].update(splits=[evaluation["split"]], seed=evaluation["seed"], episodes=evaluation["episodes"],
                          depths=list(evaluation["depths"]), hops=list(evaluation["hops"]), counterfactuals=False,
                          swap_successors=False, batch_size=evaluation["batch_size"])
    return resolve_config(config)


def runtime_metadata(device: str) -> dict:
    cuda = torch.device(device).type == "cuda"
    return dict(host=os.uname().nodename, job_id=os.environ.get("PBS_JOBID"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(device) if cuda else None, device=device, threads=1)


def is_numerical_failure(failure: object) -> bool:
    """True for the failure text of a run that stopped with a non-finite loss (or gradient norm).

    ``train`` and ``tsm.fit_tsm`` raise ``FloatingPointError`` for a non-finite loss, and clipping the gradient with
    ``error_if_nonfinite`` raises a ``RuntimeError`` that says "non-finite". An identical rerun would reproduce either.
    """
    kind, _, message = str(failure).partition(": ")
    return kind == "FloatingPointError" or "non-finite" in message


def read_record(root: Path, name: str) -> dict | None:
    """The run's ``record.json``; None when it is absent or was cut off while it was being written."""
    try:
        record = json.loads((root / "runs" / name / "record.json").read_text())
    except (FileNotFoundError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def run_state(root: Path, name: str) -> str:
    """``missing`` (no run directory), ``complete``, ``training_failure`` (stopped with a non-finite loss) or
    ``incomplete`` (still running, ended before completion, or failed for any other reason)."""
    if not (root / "runs" / name).is_dir():
        return "missing"
    record = read_record(root, name)
    if record is None:
        return "incomplete"
    if record.get("status") == "complete":
        return "complete"
    if record.get("status") == "failed" and is_numerical_failure(record.get("failure", "")):
        return "training_failure"
    return "incomplete"


def failure_row(root: Path, name: str) -> dict:
    """The entry of a registered training failure in the manifests and the analysis."""
    record = read_record(root, name)
    return dict(name=name, writer=record["writer"], arm=record["arm"], seed=record["seed"],
                attempt=record.get("attempt"), failure=record["failure"])


def t0_failures(root: Path) -> list[str]:
    """T0 runs that stopped with a non-finite loss: with one of them 'far' has no reference and the study stops."""
    return [row["name"] for row in planned_runs()
            if row["arm"] == "T0" and run_state(root, row["name"]) == "training_failure"]


def study_stops(name: str) -> str:
    return (f"T0 run {name} stopped with a non-finite loss: the study stops before the confirmation set is opened "
            "(docs/prereg/TSM.md); report it to the orchestrator, any continuation needs an explicit amendment")


def moved_attempts(root: Path, name: str) -> list[Path]:
    """Earlier, unfinished attempts of ``name`` that were moved to ``root/attempts``."""
    folder = root / "attempts"
    return sorted(path for path in folder.glob(f"{name}-attempt*") if path.is_dir()) if folder.is_dir() else []


def check_rerun(root: Path, name: str, *, config_hash: str, source: dict) -> None:
    """Refuse a rerun of an unfinished run that is not identical to it, or that exceeds the attempts allowed.

    Identical means the configuration (hence the seeds and the data stream) and the code (``source_hash``) of the
    unfinished attempt; ``MAX_ATTEMPTS`` attempts are allowed in all (the first run and two reruns).
    """
    number = len(moved_attempts(root, name)) + 1  # the attempt that is unfinished
    require(number < MAX_ATTEMPTS, f"run {name} has used {number} of {MAX_ATTEMPTS} attempts: the registered rule "
            "allows two identical reruns; report it to the orchestrator")
    record = read_record(root, name)
    if record is not None:  # a record exists once the run had started; the rerun must not differ from it
        require(record.get("config_hash") == config_hash,
                f"run {name}: the configuration differs from the unfinished attempt; a rerun must be identical")
        require(record.get("source", {}).get("source_hash") == source["source_hash"],
                f"run {name}: the code differs from the unfinished attempt; a rerun must use identical code "
                "(submit it at the same --commit)")


def move_aside(root: Path, name: str) -> Path:
    """Move an unfinished run (its directory and log) to ``root/attempts``; the run can then be trained again."""
    target = root / "attempts" / f"{name}-attempt{len(moved_attempts(root, name)) + 1}"
    target.parent.mkdir(parents=True, exist_ok=True)
    (root / "runs" / name).rename(target)
    log = root / "logs" / f"{name}.log"
    if log.is_file():
        log.rename(target.with_name(target.name + ".log"))
    return target


def retire_stale_log(root: Path, name: str) -> None:
    """A child that died before it created its run directory leaves only its log: keep it, out of the way.

    Nothing was trained, so this is not an attempt (``moved_attempts`` does not count it).
    """
    log = root / "logs" / f"{name}.log"
    if log.is_file():
        folder = root / "attempts"
        folder.mkdir(parents=True, exist_ok=True)
        log.rename(folder / f"{name}-notstarted{len(list(folder.glob(f'{name}-notstarted*.log'))) + 1}.log")


def train_one(design: dict, root: Path, name: str, *, device: str, steps: int | None = None) -> dict:
    """Train one registered run into ``root/runs/<name>``; the record holds its status, attempt and seals."""
    run = next(row for row in planned_runs() if row["name"] == name)
    config = make_config(design, run["writer"], run["seed"], run["arm"], device=device, steps=steps)
    attempt = len(moved_attempts(root, name)) + 1
    require(attempt <= MAX_ATTEMPTS, f"run {name} has used its {MAX_ATTEMPTS} attempts")
    source = source_metadata()
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", dict(config=config, config_hash=stable_hash(config)))
    cuda = torch.device(device).type == "cuda"
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats()
    record = dict(run, status="running", attempt=attempt, config_hash=stable_hash(config), started_at=now(),
                  runtime=runtime_metadata(device), source=source, steps=config["train"]["steps"])
    write_json(out / "record.json", record)
    started = time.perf_counter()
    try:
        if run["arm"] == "T0":
            model, summary = train(config, out)
        else:
            model, _, summary = tsm.train_arm(config, out, run["arm"])
    except BaseException as error:
        record.update(status="failed", failure=f"{type(error).__name__}: {error}", finished_at=now())
        write_json(out / "record.json", record)
        raise
    record.update(status="complete", finished_at=now(), seconds=time.perf_counter() - started, summary=summary,
                  checkpoint=str(out / "checkpoint.pt"), checkpoint_blake2b=digest_checkpoint(out / "checkpoint.pt"),
                  state_hash=state_hash(model),
                  peak_memory_bytes=int(torch.cuda.max_memory_allocated()) if cuda else None)
    write_json(out / "record.json", record)
    return record


class Child:
    """A running ``train-one`` process and its log file."""

    def __init__(self, process: subprocess.Popen, log) -> None:
        self.process, self.log = process, log

    def poll(self) -> int | None:
        return self.process.poll()

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self.log.close()


def start_child(name: str, design_path: Path, root: Path, *, device: str, steps: int | None) -> Child:
    command = [sys.executable, str(Path(__file__).resolve()), "train-one", "--design", str(design_path),
               "--root", str(root), "--name", name, "--device", device] + (["--steps", str(steps)] if steps else [])
    (root / "logs").mkdir(parents=True, exist_ok=True)
    log = (root / "logs" / f"{name}.log").open("x")
    env = dict(os.environ, S9_CHILD="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(SOURCE), os.environ.get("PYTHONPATH")])))
    return Child(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log)


def schedule(names: list[str], procs: int, design_path: Path, root: Path, *, device: str,
             steps: int | None = None, fatal=None) -> dict[str, int]:
    """Run ``names`` with at most ``procs`` concurrent children; returns each child's exit code.

    By default the first child that exits non-zero stops everything (``ValueError``). With ``fatal``, a function
    of the run name that is called after a non-zero exit, only a failure for which it returns true stops everything;
    the other children go on and the caller reads their records.
    """
    require(procs >= 1, "procs must be positive")
    pending, running, codes = list(names), {}, {}
    try:
        while pending or running:
            for name, child in list(running.items()):
                code = child.poll()
                if code is not None:
                    child.stop()
                    del running[name]
                    codes[name] = code
                    require(code == 0 or (fatal is not None and not fatal(name)),
                            f"run {name} exited {code}; see {root / 'logs' / (name + '.log')}")
            while pending and len(running) < procs:
                name = pending.pop(0)
                running[name] = start_child(name, design_path, root, device=device, steps=steps)
            if running:
                time.sleep(POLL_SECONDS)
    finally:
        for child in running.values():
            child.stop()
    return codes


def evaluation_arrays(design: dict, writer: str, evaluation: dict, device: str):
    config = make_config(design, writer, SEEDS[0], "T0", device=device)
    config["eval"]["hops"] = list(evaluation["hops"])
    spec = make_task_spec(config, evaluation=True)
    return stratify_queries(generate(spec, evaluation["split"], 0, evaluation["episodes"], evaluation["seed"]),
                            spec, list(evaluation["hops"]), evaluation["seed"])


def evaluate_runs(design: dict, root: Path, *, writers, seeds, evaluation: dict, device: str,
                  out: Path | None, skip=()) -> tuple[dict, list[dict], dict]:
    """Samples of T0/T1/T2 and the B4 references for the given writers and seeds.

    Returns (samples by name, evaluation rows, gate profiles by run name). With ``out`` set, the samples
    are also saved to ``out/<name>.npz`` and their blake2b seals added to the rows. ``skip`` names the registered
    training failures: they have no checkpoint and are not evaluated.
    """
    samples, rows, profiles = {}, [], {}
    depths = list(design["evaluation"]["depths"])
    for writer in writers:
        arrays = evaluation_arrays(design, writer, evaluation, device)
        for seed in seeds:
            jobs = [(run_name(writer, arm, seed), arm, None) for arm in tsm.ARMS
                    if run_name(writer, arm, seed) not in skip]
            jobs.append((reference_name(writer, seed), "B4",
                         next(r for r in design["references"] if r["writer"] == writer and r["seed"] == seed)))
            for name, arm, reference in jobs:
                started = time.perf_counter()
                if reference is None:
                    record = json.loads((root / "runs" / name / "record.json").read_text())
                    require(record["status"] == "complete", f"run {name} is not complete")
                    path = root / "runs" / name / "checkpoint.pt"
                    require(digest_checkpoint(path) == record["checkpoint_blake2b"], f"checkpoint changed: {name}")
                    model, gate, label = tsm.load_tsm(path, device=device)
                    require(label == arm, f"{name}: the checkpoint is an arm {label} record")
                    use = depths
                else:
                    entry = dict(reference, path=reference["path"])
                    model, gate, use = load_frozen(entry, device), None, [1]
                before = state_hash(model)
                result = tsm.evaluate_arm(model, gate, arrays, use, batch_size=evaluation["batch_size"],
                                          split=evaluation["split"])
                if gate is not None:
                    profiles[name] = tsm.gate_profile(model, gate, arrays, steps=max(depths),
                                                      batch_size=evaluation["batch_size"])
                synchronize(device)
                require(state_hash(model) == before, f"model changed during evaluation: {name}")
                row = dict(name=name, writer=writer, arm=arm, seed=seed, depths=use, rows=int(len(result["correct"])),
                           seconds=time.perf_counter() - started)
                if out is not None:
                    out.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out / f"{name}.npz", **result)
                    row["samples_blake2b"] = digest_checkpoint(out / f"{name}.npz")
                samples[name] = result
                rows.append(row)
                del model
    return samples, rows, profiles


def group_samples(samples: dict, writer: str, arm: str, seeds=SEEDS) -> dict:
    """The mapping ``tsm.arm_endpoints`` expects: T0, the arm and B4 by training seed.

    ``seeds`` are the training seeds the arm completed; T0 and B4 are restricted to the same seeds.
    """
    return {"T0": {s: samples[run_name(writer, "T0", s)] for s in seeds},
            arm: {s: samples[run_name(writer, arm, s)] for s in seeds},
            "B4": {s: samples[reference_name(writer, s)] for s in seeds}}


def final_states(design: dict, root: Path) -> dict[str, str]:
    """State of every registered run for the evaluation: ``complete``, or ``training_failure`` (T1/T2 only).

    Every T0 run must be complete (a T0 failure stops the study: ``far`` has no reference) and every T1/T2 run must
    be complete or a registered training failure; every record must belong to the registered configuration.
    Anything else is refused, and this runs before the confirmation set is opened.
    """
    states = {}
    for row in planned_runs():
        name, state = row["name"], run_state(root, row["name"])
        require(not (row["arm"] == "T0" and state == "training_failure"), study_stops(name))
        require(state in ("complete", "training_failure"),
                f"run {name} is {state}: a run must be complete or a registered training failure (a T1/T2 run with a "
                "non-finite loss); an unfinished run is trained again with `train --rerun-incomplete`, at most twice")
        record = read_record(root, name)
        config = make_config(design, row["writer"], row["seed"], row["arm"], device=record["runtime"]["device"])
        require(record["config_hash"] == stable_hash(config) and record["steps"] == design["training"]["steps"],
                f"run {name} is not a registered run")
        states[name] = state
    return states


def stage_lock(args: argparse.Namespace) -> int:
    out = Path(args.out)
    require(out.is_absolute() and not out.exists(), "the lock file must be a new absolute path")
    design = build_design(args.p1s_root, args.f2_root)
    write_json(out, design)
    print(json.dumps(dict(stage="lock", templates=len(design["templates"]), references=len(design["references"]),
                          design_hash=stable_hash(design), out=str(out))), flush=True)
    return 0


def select_names(args: argparse.Namespace) -> list[str]:
    names = [row["name"] for row in planned_runs() if row["writer"] in args.writers and row["arm"] in args.arms
             and row["seed"] in args.seeds]
    require(bool(names), "no registered run matches the selection")
    return names


def stage_train(args: argparse.Namespace) -> int:
    """Train the selected runs that have no final state; leave complete runs and registered failures as they are.

    A run is final when it is complete or a registered training failure (a non-finite loss: no retry). An unfinished
    run (killed by walltime, a node or I/O fault, or failed for another reason) is trained again, identically, only
    with ``--rerun-incomplete`` (its job must have ended) and at most twice. A T0 run with a non-finite loss stops the
    study: this stage refuses to start anything else. The stage ends with an error unless every selected run is final.
    """
    root = absolute_path(args.root, "--root")
    design_path = Path(args.design).resolve()
    verify_git_lock(design_path, args.design_commit, "the design")
    design = json.loads(design_path.read_text())
    verify_design(design)
    names = select_names(args)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"train_manifest-{args.label}.json"
    require(not manifest_path.exists(), f"{manifest_path.name} exists: every training job needs its own --label")
    stopped = t0_failures(root)
    if stopped:
        raise ValueError(study_stops(stopped[0]))
    source = source_metadata()
    states = {name: run_state(root, name) for name in names}
    reruns = [name for name in names if states[name] == "incomplete"]
    require(not reruns or args.rerun_incomplete,
            f"unfinished runs {reruns}: if their job has ended, train them again, identically, with "
            "--rerun-incomplete (at most twice per run)")
    for name in reruns:
        run = next(row for row in planned_runs() if row["name"] == name)
        config = make_config(design, run["writer"], run["seed"], run["arm"], device=args.device)
        check_rerun(root, name, config_hash=stable_hash(config), source=source)
    started = time.perf_counter()
    for name in reruns:
        move_aside(root, name)
    for name in names:
        if states[name] == "missing":
            retire_stale_log(root, name)
    todo = [name for name in names if states[name] in ("missing", "incomplete")]
    manifest = dict(status="running", design_hash=stable_hash(design), design_path=str(design_path),
                    design_lock_commit=args.design_commit, names=names, started=todo, reruns=reruns,
                    final_before=[name for name in names if name not in todo], procs=args.procs, started_at=now(),
                    source=source, runtime=runtime_metadata(args.device))
    write_json(manifest_path, manifest)
    try:
        schedule(todo, args.procs, design_path, root, device=args.device,
                 fatal=lambda name: name in t0_failures(root))
    except ValueError as error:
        stopped = t0_failures(root)
        reason = study_stops(stopped[0]) if stopped else str(error)
        manifest.update(status="stopped" if stopped else "failed", reason=reason, finished_at=now())
        write_json(manifest_path, manifest)
        raise ValueError(reason) from error
    states = {name: run_state(root, name) for name in names}
    failures = [failure_row(root, name) for name in names if states[name] == "training_failure"]
    unfinished = [name for name in names if states[name] not in ("complete", "training_failure")]
    seconds = time.perf_counter() - started
    status = "incomplete" if unfinished else "complete"
    manifest.update(status=status, finished_at=now(), total_seconds=seconds, node_hours=seconds / 3600.0,
                    run_seconds={name: read_record(root, name)["seconds"] for name in todo if states[name] == "complete"},
                    training_failures=failures, unfinished=unfinished)
    write_json(manifest_path, manifest)
    print(json.dumps(dict(stage="train", status=status, runs=len(names), started=len(todo),
                          training_failures=[row["name"] for row in failures], unfinished=unfinished,
                          seconds=seconds)), flush=True)
    require(not unfinished, f"runs {unfinished} did not finish; see {root / 'logs'}; an infrastructure failure is "
            "trained again with --rerun-incomplete once its job has ended, at most twice")
    return 0


def stage_train_one(args: argparse.Namespace) -> int:
    design = json.loads(Path(args.design).read_text())
    record = train_one(design, Path(args.root), args.name, device=args.device, steps=args.steps)
    print(json.dumps(dict(stage="train-one", name=args.name, status=record["status"], seconds=record["seconds"])),
          flush=True)
    return 0


def stage_smoke(args: argparse.Namespace) -> int:
    """Short training of the seed-17 runs and a validation evaluation: timings and sanity only."""
    root = absolute_path(args.root, "--root")
    design_path = Path(args.design).resolve()
    design = json.loads(design_path.read_text())
    verify_design(design)
    smoke = design["smoke"]
    root.mkdir(parents=True, exist_ok=False)
    names = [row["name"] for row in planned_runs() if row["seed"] == smoke["training_seed"]]
    started = time.perf_counter()
    schedule(names, args.procs, design_path, root, device=args.device, steps=smoke["steps"])
    train_seconds = time.perf_counter() - started
    records = {n: json.loads((root / "runs" / n / "record.json").read_text()) for n in names}
    evaluation = dict(design["evaluation"], split=smoke["split"], seed=smoke["seed"], episodes=smoke["episodes"])
    seed = smoke["training_seed"]
    samples, rows, profiles = evaluate_runs(design, root, writers=WRITERS, seeds=[seed], evaluation=evaluation,
                                            device=args.device, out=None)
    endpoints = {}
    for writer in WRITERS:
        for arm in ("T1", "T2"):
            group = {"T0": {seed: samples[run_name(writer, "T0", seed)]},
                     arm: {seed: samples[run_name(writer, arm, seed)]},
                     "B4": {seed: samples[reference_name(writer, seed)]}}
            result = tsm.arm_endpoints(group, arm, writer=writer, confidence=design["decision"]["confidence"],
                                       split=smoke["split"])
            endpoints[f"{writer}/{arm}"] = {key: result[key]["difference"] for key in ("stay", "far", "gain")}
    per_step = {name: record["summary"]["train_step_seconds"] for name, record in records.items()}
    run_seconds = {name: step * design["training"]["steps"] for name, step in per_step.items()}
    total = sum(run_seconds.values()) * len(SEEDS)  # the smoke trained seed 17 only; the seeds cost the same
    job_seconds = total / args.procs
    report = dict(
        status="passed", created=now(), design_hash=stable_hash(design), procs=args.procs, steps=smoke["steps"],
        train_wall_seconds=train_seconds, train_step_seconds=per_step,
        peak_memory_bytes={n: r["peak_memory_bytes"] for n, r in records.items()},
        final_loss={n: r["summary"]["final_loss"] for n, r in records.items()},
        extrapolated_run_seconds=run_seconds, extrapolated_total_run_seconds=total,
        extrapolated_job_seconds_at_smoke_procs=job_seconds,
        suggested_walltime_seconds=int(-(-WALLTIME_MARGIN * job_seconds // 60) * 60),
        evaluation_rows=rows, gate_profiles=profiles, endpoints=endpoints,
        note="smoke endpoints come from one seed after a few steps and are never used. Per-step times are measured "
             "at the smoke concurrency, so the extrapolations are estimates; extrapolated_run_seconds lets any "
             "subset of runs be costed (a job over a subset S takes about sum(S) / procs).",
        runtime=runtime_metadata(args.device), total_seconds=time.perf_counter() - started)
    write_json(root / "smoke.json", report)
    print(json.dumps(dict(stage="smoke", status="passed", suggested_walltime_seconds=report["suggested_walltime_seconds"],
                          extrapolated_total_run_seconds=total)), flush=True)
    return 0


def stage_evaluate(args: argparse.Namespace) -> int:
    root = absolute_path(args.root, "--root")
    design_path = Path(args.design).resolve()
    verify_git_lock(design_path, args.design_commit, "the design")
    design = json.loads(design_path.read_text())
    verify_design(design)
    out = root / "evaluation"
    require(not (out / "manifest.json").exists(), "the confirmation evaluation has already started")
    states = final_states(design, root)  # before anything touches the confirmation set; refuses a T0 failure
    failures = [failure_row(root, name) for name in sorted(states) if states[name] == "training_failure"]
    started = time.perf_counter()
    manifest = dict(status="running", design=design, design_hash=stable_hash(design), design_path=str(design_path),
                    design_lock_commit=args.design_commit, evaluation=design["evaluation"],
                    training_failures=failures, started_at=now(), source=source_metadata(),
                    runtime=runtime_metadata(args.device))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    _, rows, profiles = evaluate_runs(design, root, writers=WRITERS, seeds=SEEDS, evaluation=design["evaluation"],
                                      device=args.device, out=out / "samples", skip={row["name"] for row in failures})
    for template in design["templates"]:
        require(digest_checkpoint(Path(template["source"])) == template["source_blake2b"], "a released checkpoint changed")
    verify_design(design)
    seconds = time.perf_counter() - started
    write_json(out / "gate_profiles.json", profiles)
    manifest.update(status="complete", evaluations=rows, finished_at=now(), total_seconds=seconds,
                    node_hours=seconds / 3600.0, checkpoints_unchanged=True)
    write_json(out / "manifest.json", manifest)
    print(json.dumps(dict(stage="evaluate", status="complete", evaluated=len(rows),
                          training_failures=[row["name"] for row in failures], seconds=seconds, out=str(out))),
          flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    lock = sub.add_parser("lock", help="write the design and seals")
    lock.add_argument("--p1s-root", default=DEFAULT_P1S, help="P1-S main run root")
    lock.add_argument("--f2-root", default=DEFAULT_F2, help="P1-S-F2 root")
    lock.add_argument("--out", required=True, help="new design JSON (absolute path)")
    smoke = sub.add_parser("smoke", help="short training and validation evaluation")
    smoke.add_argument("--design", required=True)
    smoke.add_argument("--root", required=True, help="new directory (absolute path)")
    smoke.add_argument("--procs", type=int, default=6)
    train_stage = sub.add_parser("train", help="train registered runs")
    train_stage.add_argument("--design", required=True)
    train_stage.add_argument("--design-commit", required=True, help="full commit hash that locked the design")
    train_stage.add_argument("--root", required=True, help="the run directory (shared by the training jobs)")
    train_stage.add_argument("--procs", type=int, default=9)
    train_stage.add_argument("--label", default="main", help="name of this job's manifest (one label per job)")
    train_stage.add_argument("--rerun-incomplete", action="store_true",
                             help="train unfinished runs again, identically (their job must have ended; at most "
                                  "twice per run); without it an unfinished run is an error")
    train_stage.add_argument("--writers", nargs="+", default=list(WRITERS), choices=WRITERS)
    train_stage.add_argument("--arms", nargs="+", default=list(tsm.ARMS), choices=tsm.ARMS)
    train_stage.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    child = sub.add_parser("train-one", help="internal: train one run (started by the train and smoke stages)")
    child.add_argument("--design", required=True)
    child.add_argument("--root", required=True)
    child.add_argument("--name", required=True)
    child.add_argument("--steps", type=int)
    evaluate = sub.add_parser("evaluate", help="confirmation evaluation")
    evaluate.add_argument("--design", required=True)
    evaluate.add_argument("--design-commit", required=True)
    evaluate.add_argument("--root", required=True, help="the training run directory")
    for stage in (smoke, train_stage, child, evaluate):
        stage.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.stage == "train-one":  # started by a stage that already passed the compute check
        require(os.environ.get("S9_CHILD") == "1", "train-one is started by the train and smoke stages")
    else:
        require_compute()
    return {"lock": stage_lock, "smoke": stage_smoke, "train": stage_train, "train-one": stage_train_one,
            "evaluate": stage_evaluate}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
