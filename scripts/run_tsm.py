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

Training failures (docs/prereg/TSM.md, "Decisions at registration" and Clarifications C1 and C2); every run that ends
with an exception is recorded as ``failed`` with a ``failure_kind`` (see ``classify_failure``):

* a non-finite loss or gradient norm in a T1 or T2 run (``numerical``) is a registered training failure: no retry, no
  re-tuning; that arm fails for that writer, its completed runs are evaluated and reported descriptively, the other arms
  are evaluated as registered;
* the same in a T0 run stops the study before the confirmation set is opened: ``train`` refuses to start anything else
  and ``evaluate`` refuses to run;
* an end by walltime or a node fault, an I/O error (an ``OSError``, or a failure of torch's checkpoint serialization)
  and an allocation failure (CUDA out-of-memory, a cuBLAS or cuDNN allocation failure, a failure of torch's CPU
  allocator, a host ``MemoryError``) are infrastructure failures (``infrastructure``): the run is trained again with the
  identical configuration and code (``train --rerun-incomplete`` once its job has ended), at most twice, or more with
  ``--max-attempts N --decision FILE`` on the orchestrator's recorded decision; every attempt is compared with the one
  before it (also when an earlier rerun stopped before its child started). A run that is not final and has used its
  attempts pauses the study: ``train`` starts nothing until a decision grants more;
* any other exception (``code_defect``) stops the study like a T0 failure, whatever the arm: the job starts no pending
  run, lets the runs that are already running finish (their records are kept) and ends with an error, every later
  ``train`` and ``evaluate`` refuses, and the fixed code must retrain every TSM run, in a new ``--root``, before the
  confirmation set is opened (the orchestrator decides);
* all training jobs of a root run at one code: ``train`` refuses a ``source_hash`` that differs from that of any record
  under ``runs/`` or ``attempts/``;
* ``evaluate`` needs every T0 run complete and every T1/T2 run complete or a registered training failure, all of them
  trained with the same code (T0 is "retrained under the same code version") and with their checkpoints unchanged;
  its first attempt runs at that code too;
* an ``evaluate`` job that ended before it completed is restarted from scratch with identical inputs (same design lock,
  code and checkpoints) at most twice, only with ``--restart-interrupted`` once its job has ended: the partial output is
  moved aside to ``<root>/attempts/evaluation-attempt<k>`` without being read, and the new manifest records the restarts
  and the code (``source_hash``) of every attempt. After a defect in the evaluation code, and on the orchestrator's
  decision, ``--new-evaluation-code`` lets the restart run at another code; the training records must still share one
  ``source_hash`` (Clarification C2). A live evaluation touches ``evaluation/heartbeat`` every ``HEARTBEAT_SECONDS``,
  so that the silence rule (``QUIET_SECONDS``) does not mistake it for an ended one.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
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
MAX_ATTEMPTS = 3  # a run (or the evaluation) and two identical reruns (restarts); train --max-attempts raises it
QUIET_SECONDS = 300  # a run or evaluation still marked "running" is started again only after it was silent this long
HEARTBEAT_SECONDS = 30  # a live evaluation writes evaluation/heartbeat this often (at most 60), far below QUIET_SECONDS
NUMERICAL, INFRASTRUCTURE, CODE_DEFECT = "numerical", "infrastructure", "code_defect"  # a failed record's failure_kind
ALLOCATION_STATUSES = (  # the library statuses that say an allocation failed, in lower case (docs/prereg/TSM.md, C2)
    "cublas_status_alloc_failed",  # cuBLAS
    "cudnn_status_alloc_failed",  # cuDNN 8, and a deprecated alias in cuDNN 9
    "cudnn_status_internal_error_host_allocation_failed",  # cuDNN 9 (the pinned one): what its host allocations report
    "cudnn_status_internal_error_device_allocation_failed")  # cuDNN 9: what its device allocations report
CPU_ALLOCATOR_FAILURE = "defaultcpuallocator: not enough memory"  # torch's CPU allocator (c10, C3), in lower case
SERIALIZATION_FAILURES = (  # torch.save and torch.load cannot write or read a file (docs/prereg/TSM.md, C3), lower case
    "pytorchstreamwriter failed", "pytorchstreamreader failed")


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


def is_allocation_failure(error: BaseException) -> bool:
    """A failure to allocate memory (docs/prereg/TSM.md, Clarifications C1 to C3): CUDA out-of-memory, a cuBLAS or
    cuDNN allocation failure, a failure of torch's CPU allocator, or a host ``MemoryError``; any other CUDA runtime
    error is not one.

    CUDA out-of-memory is ``torch.cuda.OutOfMemoryError`` or a ``RuntimeError`` that says so (older PyTorch, and the
    CUDA runtime itself, creating the context on a full device, report "CUDA error: out of memory"). The cuBLAS and
    cuDNN libraries report a failed allocation as a ``RuntimeError`` that names one of ``ALLOCATION_STATUSES``: in
    cuDNN 9 ``CUDNN_STATUS_ALLOC_FAILED`` is a deprecated alias, and its error text names the host or device
    allocation status instead. cuBLAS that cannot create its handle, usually on a full GPU, reports
    ``CUBLAS_STATUS_NOT_INITIALIZED`` "when calling `cublasCreate(handle)`": that status of any other call is not an
    allocation failure. torch's CPU allocator raises a ``RuntimeError`` "DefaultCPUAllocator: not enough memory".
    """
    if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    text = str(error).lower()
    return isinstance(error, RuntimeError) and (("cuda" in text and "out of memory" in text)
                                                or any(status in text for status in ALLOCATION_STATUSES)
                                                or CPU_ALLOCATOR_FAILURE in text
                                                or ("cublas_status_not_initialized" in text and "cublascreate" in text))


def is_serialization_failure(error: BaseException) -> bool:
    """torch's checkpoint serialization could not write or read a file (docs/prereg/TSM.md, Clarification C3): a
    ``RuntimeError`` of the zip stream behind ``torch.save`` and ``torch.load``, "PytorchStreamWriter failed ..." or
    "PytorchStreamReader failed ...", the I/O error of a full disk, a quota or a file system fault."""
    return isinstance(error, RuntimeError) and any(marker in str(error).lower() for marker in SERIALIZATION_FAILURES)


def is_numerical_failure(error: BaseException) -> bool:
    """A non-finite loss or gradient norm: an identical rerun would reproduce either.

    ``train`` and ``tsm.fit_tsm`` raise ``FloatingPointError`` for a non-finite loss, and clipping the gradient with
    ``error_if_nonfinite`` raises a ``RuntimeError`` that says "non-finite".
    """
    return isinstance(error, FloatingPointError) or (isinstance(error, RuntimeError) and "non-finite" in str(error))


def classify_failure(error: BaseException) -> str:
    """How the end of a run is read (docs/prereg/TSM.md, "Decisions at registration" and Clarifications C1 to C3).

    * ``infrastructure``: an I/O error (any ``OSError``, or a failure of torch's checkpoint serialization:
      ``is_serialization_failure``), an allocation failure (``is_allocation_failure``), or an interruption that is not
      an exception of the code (``KeyboardInterrupt``, ``SystemExit``); the run may be trained again, identically;
    * ``numerical``: a non-finite loss or gradient norm; never retried (a T1/T2 arm fails for that writer, T0 stops
      the study);
    * ``code_defect``: any other exception; the study stops. A code defect that is misread as an infrastructure failure
      repeats identically and reaches the pause after the allowed attempts (Clarification C3).
    """
    if (isinstance(error, OSError) or is_serialization_failure(error) or is_allocation_failure(error)
            or not isinstance(error, Exception)):
        return INFRASTRUCTURE
    return NUMERICAL if is_numerical_failure(error) else CODE_DEFECT


def read_json_record(path: Path) -> dict | None:
    """A ``record.json``; None when it is absent or was cut off while it was being written."""
    try:
        record = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def read_record(root: Path, name: str) -> dict | None:
    """The record of run ``name`` in place (see ``read_json_record``)."""
    return read_json_record(root / "runs" / name / "record.json")


def run_state(root: Path, name: str) -> str:
    """``missing`` (no run directory), ``complete``, ``training_failure`` (a non-finite loss or gradient norm),
    ``code_defect`` (failed with an exception that is not a numerical failure, an I/O error or an allocation failure)
    or ``incomplete`` (still running, ended before completion, failed with an I/O error or an allocation failure, or
    has an unreadable record). The kind of a failure is the ``failure_kind`` that ``train_one`` recorded."""
    if not (root / "runs" / name).is_dir():
        return "missing"
    record = read_record(root, name)
    if record is None:
        return "incomplete"
    if record.get("status") == "complete":
        return "complete"
    if record.get("status") == "failed":
        kind = record.get("failure_kind")
        if kind == NUMERICAL:
            return "training_failure"
        if kind != INFRASTRUCTURE:  # also a failed record without a kind: nothing shows that it is not a defect
            return "code_defect"
    return "incomplete"


def failure_row(root: Path, name: str) -> dict:
    """The entry of a registered training failure in the manifests and the analysis."""
    record = read_record(root, name)
    return dict(name=name, writer=record["writer"], arm=record["arm"], seed=record["seed"],
                attempt=record.get("attempt"), failure=record["failure"])


def stops_study(root: Path, name: str) -> bool:
    """True if run ``name`` stopped the study: a code defect (any arm), or a non-finite loss in a T0 run (with it 'far'
    has no reference)."""
    state = run_state(root, name)
    arm = next(row["arm"] for row in planned_runs() if row["name"] == name)
    return state == "code_defect" or (state == "training_failure" and arm == "T0")


def study_stop(root: Path) -> str | None:
    """Why the study has stopped, or None. The message names the first run that stopped it, a code defect before a T0
    failure: a code defect has the code fixed and every TSM run retrained, which supersedes any other result."""
    stops = [row["name"] for row in planned_runs() if stops_study(root, row["name"])]
    defects = [name for name in stops if run_state(root, name) == "code_defect"]
    if defects:
        return (f"run {defects[0]} ended with a code defect ({read_record(root, defects[0]).get('failure')}): the "
                "study stops (docs/prereg/TSM.md, Clarification C1); the fixed code must retrain every TSM run before "
                "the confirmation set is opened; report it to the orchestrator")
    if stops:
        return (f"T0 run {stops[0]} stopped with a non-finite loss or gradient norm: the study stops before the "
                "confirmation set is opened (docs/prereg/TSM.md); report it to the orchestrator, any continuation "
                "needs an explicit amendment")
    return None


def moved_attempts(root: Path, name: str) -> list[Path]:
    """Earlier, unfinished attempts of ``name`` that were moved to ``root/attempts``."""
    folder = root / "attempts"
    return sorted(path for path in folder.glob(f"{name}-attempt*") if path.is_dir()) if folder.is_dir() else []


def earlier_record(root: Path, name: str) -> dict | None:
    """The record of the latest earlier attempt of ``name`` that has one: the run in place, else the last moved aside."""
    for folder in [root / "runs" / name, *reversed(moved_attempts(root, name))]:
        record = read_json_record(folder / "record.json")
        if record is not None:
            return record
    return None


def attempts_used(root: Path, name: str) -> int:
    """The attempts of ``name`` so far: those moved aside and the run in place, if there is one."""
    return len(moved_attempts(root, name)) + int((root / "runs" / name).is_dir())


def check_rerun(root: Path, name: str, *, config_hash: str, source: dict, max_attempts: int = MAX_ATTEMPTS) -> int:
    """Refuse an attempt of ``name`` that exceeds the attempts allowed or is not identical to the one before it.

    Returns the number of the attempt that may start. The attempts used are those moved aside and the run in place, if
    there is one; ``max_attempts`` are allowed in all (by default ``MAX_ATTEMPTS``: the first run and two reruns; more
    only on the orchestrator's recorded decision, ``train --max-attempts``). Identical means the configuration
    (hence the seeds and the data stream) and the code (``source_hash``) of the latest earlier attempt, wherever it is:
    in place, or moved aside by a rerun that stopped before its child had started. By induction every attempt equals
    the first.
    """
    used = attempts_used(root, name)
    require(used < max_attempts, f"run {name} has used {used} of {max_attempts} attempts: report it to the "
            "orchestrator; further identical attempts need its recorded decision (train --max-attempts N "
            "--decision FILE, docs/prereg/TSM.md, Clarifications C1 and C2)")
    record = earlier_record(root, name)
    if record is not None:  # a record exists once an attempt had started; the next one must not differ from it
        require(record.get("config_hash") == config_hash,
                f"run {name}: the configuration differs from the earlier attempt; a rerun must be identical")
        require(record.get("source", {}).get("source_hash") == source["source_hash"],
                f"run {name}: the code differs from the earlier attempt; a rerun must use identical code "
                "(submit it at the same --commit)")
    return used + 1


def seconds_since_written(paths: list[Path]) -> float:
    """Seconds since the newest of ``paths`` was written (a path that does not exist counts as never written)."""
    newest = 0.0
    for path in paths:
        try:
            newest = max(newest, path.stat().st_mtime)
        except FileNotFoundError:
            pass
    return max(0.0, time.time() - newest)


def silent_seconds(root: Path, name: str) -> float:
    """Seconds since anything in the run's directory or its child's log was last written."""
    out = root / "runs" / name
    return seconds_since_written([out, root / "logs" / f"{name}.log", *out.rglob("*")])


def has_ended(root: Path, name: str) -> bool:
    """True if the job of an unfinished run has ended.

    A child that failed has recorded the failure and is gone. A run still marked "running" (or whose record is being
    written) has ended only if nothing touched it for ``QUIET_SECONDS``: a live run appends to ``train_log.jsonl``
    every few seconds.
    """
    record = read_record(root, name)
    return (record is not None and record.get("status") == "failed") or silent_seconds(root, name) >= QUIET_SECONDS


def require_ended(root: Path, name: str) -> None:
    """Refuse to train an unfinished run again while its job may still be running (``has_ended``).

    Moving a live run aside would start a second child while the first one goes on writing into the new directory.
    """
    require(has_ended(root, name), f"run {name} was written to {silent_seconds(root, name):.0f} s ago: its job may "
            f"still be running; check qstat and train it again only after the job has ended (and {QUIET_SECONDS} s of "
            "silence)")


def paused_runs(root: Path, max_attempts: int) -> list[str]:
    """The runs that pause the study (docs/prereg/TSM.md, Clarification C1): not final, every one of the
    ``max_attempts`` attempts used and ended. A run whose last attempt is still alive has not failed yet."""
    paused = []
    for row in planned_runs():
        name, state = row["name"], run_state(root, row["name"])
        if (state in ("missing", "incomplete") and attempts_used(root, name) >= max_attempts
                and (state == "missing" or has_ended(root, name))):
            paused.append(name)
    return paused


def recorded_source_hashes(root: Path) -> set[str]:
    """The ``source_hash`` of every readable run record in ``root``: the runs in place and the attempts moved aside."""
    codes = set()
    for path in [*(root / "runs").glob("*/record.json"), *(root / "attempts").glob("*/record.json")]:
        record = read_json_record(path)
        if record is not None and isinstance(record.get("source", {}).get("source_hash"), str):
            codes.add(record["source"]["source_hash"])
    return codes


def require_one_code(root: Path, source: dict) -> None:
    """Refuse a training job whose code differs from that of any run record in ``root``.

    All 18 runs are trained at one ``source_hash`` (docs/prereg/TSM.md, Clarification C1); ``final_states`` would find a
    mismatch only at the evaluation, after the odd runs had been paid for.
    """
    others = sorted(recorded_source_hashes(root) - {source["source_hash"]})
    require(not others, f"the code differs from the earlier TSM records in {root} (source_hash "
            f"{', '.join(code[:12] for code in others)}; this job {source['source_hash'][:12]}): every TSM training "
            "job runs at one CODE commit (docs/prereg/TSM.md, Clarification C1); a retraining at another code starts "
            "in a new --root")


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


def train_one(design: dict, root: Path, name: str, *, device: str, steps: int | None = None,
              max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Train one registered run into ``root/runs/<name>``; the record holds its status, attempt and seals.

    Whatever ends the run after its directory exists, in the setup, in training or in the seals that follow, is
    recorded before the error propagates: status ``failed``, the failure text and its ``failure_kind``
    (``classify_failure``). Nothing is recorded for an error before the directory exists (the attempt guard, the
    configuration): it has trained nothing.
    """
    run = next(row for row in planned_runs() if row["name"] == name)
    config = make_config(design, run["writer"], run["seed"], run["arm"], device=device, steps=steps)
    source, config_hash = source_metadata(), stable_hash(config)
    require_one_code(root, source)  # both checks run before anything is created
    attempt = check_rerun(root, name, config_hash=config_hash, source=source, max_attempts=max_attempts)
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=False)
    record = dict(run, status="running", attempt=attempt, config_hash=config_hash, started_at=now(), source=source,
                  steps=config["train"]["steps"])
    try:
        write_json(out / "config.json", dict(config=config, config_hash=config_hash))
        cuda = torch.device(device).type == "cuda"
        torch.set_num_threads(1)
        if cuda:
            torch.cuda.set_device(torch.device(device))
            torch.cuda.reset_peak_memory_stats()
        record["runtime"] = runtime_metadata(device)
        write_json(out / "record.json", record)
        started = time.perf_counter()
        if run["arm"] == "T0":
            model, summary = train(config, out)
        else:
            model, _, summary = tsm.train_arm(config, out, run["arm"])
        done = dict(record, status="complete", finished_at=now(), seconds=time.perf_counter() - started,
                    summary=summary, checkpoint=str(out / "checkpoint.pt"),
                    checkpoint_blake2b=digest_checkpoint(out / "checkpoint.pt"), state_hash=state_hash(model),
                    peak_memory_bytes=int(torch.cuda.max_memory_allocated()) if cuda else None)
        write_json(out / "record.json", done)
    except BaseException as error:
        write_json(out / "record.json", dict(record, status="failed", failure=f"{type(error).__name__}: {error}",
                                             failure_kind=classify_failure(error), finished_at=now()))
        raise
    return done


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


def start_child(name: str, design_path: Path, root: Path, *, device: str, steps: int | None,
                max_attempts: int = MAX_ATTEMPTS) -> Child:
    command = [sys.executable, str(Path(__file__).resolve()), "train-one", "--design", str(design_path),
               "--root", str(root), "--name", name, "--device", device,
               "--max-attempts", str(max_attempts)] + (["--steps", str(steps)] if steps else [])
    (root / "logs").mkdir(parents=True, exist_ok=True)
    log = (root / "logs" / f"{name}.log").open("x")
    env = dict(os.environ, S9_CHILD="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(SOURCE), os.environ.get("PYTHONPATH")])))
    return Child(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log)


def schedule(names: list[str], procs: int, design_path: Path, root: Path, *, device: str,
             steps: int | None = None, max_attempts: int = MAX_ATTEMPTS, fatal=None) -> dict[str, int]:
    """Run ``names`` with at most ``procs`` concurrent children; returns each child's exit code.

    By default the first child that exits non-zero stops everything at once (``ValueError``; the running children are
    terminated). With ``fatal``, a function of the run name that is called after a non-zero exit, only a failure for
    which it returns true stops the job, and it stops it as a study stops (docs/prereg/TSM.md, Clarification C1:
    running jobs may finish, and their outputs are kept): no pending run is started, the children that are running
    finish, and then ``ValueError`` names the first such failure. The other failures are only recorded: the other
    children go on and the caller reads their records. A child that cannot be started (``OSError``) stops everything
    at once.
    """
    require(procs >= 1, "procs must be positive")
    pending, running, codes = list(names), {}, {}
    stopped = None  # the first failure that stops the job
    try:
        while pending or running:
            for name, child in list(running.items()):
                code = child.poll()
                if code is not None:
                    child.stop()
                    del running[name]
                    codes[name] = code
                    if code != 0 and (fatal is None or fatal(name)):
                        stopped = stopped or f"run {name} exited {code}; see {root / 'logs' / (name + '.log')}"
                        if fatal is None:
                            raise ValueError(stopped)  # everything stops now: the running children are terminated
                        pending.clear()  # the study stops: nothing more is started, the running children finish
            while pending and len(running) < procs:
                name = pending.pop(0)
                try:
                    running[name] = start_child(name, design_path, root, device=device, steps=steps,
                                                max_attempts=max_attempts)
                except OSError as error:  # for example the log of a run that another job has just started
                    raise ValueError(f"run {name} could not be started: {error}") from error
            if running:
                time.sleep(POLL_SECONDS)
        if stopped:
            raise ValueError(stopped)
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


def registered_record(design: dict, root: Path, row: dict) -> dict:
    """The record of a final run (complete, or a registered failure), after checking it is the registered run."""
    record = read_record(root, row["name"])
    config = make_config(design, row["writer"], row["seed"], row["arm"],
                         device=record.get("runtime", {}).get("device", "cpu"))
    require(record.get("config_hash") == stable_hash(config) and record.get("steps") == design["training"]["steps"],
            f"run {row['name']} is not a registered run: its configuration or steps differ from the locked design")
    return record


def final_states(design: dict, root: Path) -> dict[str, str]:
    """State of every registered run for the evaluation: ``complete``, or ``training_failure`` (T1/T2 only).

    Every T0 run must be complete (a T0 failure stops the study: ``far`` has no reference), no run may have a code
    defect (it stops the study), and every T1/T2 run must be complete or a registered training failure; every record
    must belong to the registered configuration, the checkpoint of a complete run must be unchanged, and all 18 runs
    must have been trained with the same code (the registered T0 is "retrained under the same code version").
    Anything else is refused, and this runs before the confirmation set is opened.
    """
    stopped = study_stop(root)  # first: an unfinished run earlier in the plan must not hide a stop
    if stopped:
        raise ValueError(stopped)
    states, codes = {}, set()
    for row in planned_runs():
        name, state = row["name"], run_state(root, row["name"])
        require(state in ("complete", "training_failure"),
                f"run {name} is {state}: a run must be complete or a registered training failure (a T1/T2 run with a "
                "non-finite loss); an unfinished run is trained again with `train --rerun-incomplete`")
        record = registered_record(design, root, row)
        if state == "complete":
            path = root / "runs" / name / "checkpoint.pt"
            require(path.is_file() and digest_checkpoint(path) == record.get("checkpoint_blake2b"),
                    f"checkpoint missing or changed: {name}")
        codes.add(record.get("source", {}).get("source_hash"))
        states[name] = state
    require(len(codes) == 1 and None not in codes,
            "the registered runs were not all trained with the same code (one source_hash is required): the "
            "registered T0 is 'retrained under the same code version'; report it to the orchestrator")
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


def decision_record(value: str | None) -> dict | None:
    """The orchestrator's recorded decision behind ``train --decision FILE``: its absolute path and blake2b digest.

    The file must exist and not be empty (what it says is the orchestrator's; nothing here reads it). None without
    ``--decision``.
    """
    if value is None:
        return None
    path = absolute_path(value, "--decision")
    require(path.is_file() and path.stat().st_size > 0,
            f"--decision {path} must be an existing, non-empty file: the orchestrator's recorded decision")
    return dict(path=str(path), blake2b=digest_checkpoint(path))


def stage_train(args: argparse.Namespace) -> int:
    """Train the selected runs that have no final state; leave complete runs and registered failures as they are.

    A run is final when it is complete or a registered training failure (a non-finite loss or gradient norm: no retry);
    a final record must be the registered run. An unfinished run (killed by walltime or a node fault, or ended with an
    I/O or serialization error or an allocation failure) is trained again, identically, only with
    ``--rerun-incomplete`` (its job must have ended: a run still marked "running" must have been silent for
    ``QUIET_SECONDS``) and at most twice; ``--max-attempts N`` above 3 grants N attempts in all, and needs
    ``--decision FILE``, the orchestrator's recorded decision (Clarifications C1 and C2): the manifest records the
    file's path and blake2b digest. Each attempt is compared with the one before it.

    The study pauses (Clarification C1) while a run of the study that is not final has used all its attempts and the
    last one has ended: the stage refuses to start anything, whatever it selects, until a decision grants more. It
    refuses too, when the study has stopped, and when a record of the root was written at another code: one root
    holds one ``source_hash`` (a retraining after a code defect starts in a new ``--root``). A T0 run with a non-finite
    loss, and a run with a code defect (any other exception), stops the study: no pending run is started, the runs
    that are already running finish (their records are kept), and the job ends with an error. The stage ends with an
    error unless every selected run is final.
    """
    require(args.max_attempts >= MAX_ATTEMPTS, f"--max-attempts must be at least {MAX_ATTEMPTS}, the registered number "
            "of attempts: more are granted by the orchestrator's recorded decision (docs/prereg/TSM.md, "
            "Clarification C2)")
    require(args.max_attempts == MAX_ATTEMPTS or args.decision,
            f"--max-attempts {args.max_attempts} needs --decision FILE, the orchestrator's recorded decision under "
            "docs/prereg/TSM.md, Clarification C1 (three failed attempts, granted in Clarification C2)")
    decision = decision_record(args.decision)
    root = absolute_path(args.root, "--root")
    design_path = Path(args.design).resolve()
    verify_git_lock(design_path, args.design_commit, "the design")
    design = json.loads(design_path.read_text())
    verify_design(design)
    names = select_names(args)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"train_manifest-{args.label}.json"
    require(not manifest_path.exists(), f"{manifest_path.name} exists: every training job needs its own --label")
    stopped = study_stop(root)
    if stopped:
        raise ValueError(stopped)
    source = source_metadata()
    require_one_code(root, source)
    paused = paused_runs(root, args.max_attempts)
    if paused:  # not require(): the message needs a paused run
        raise ValueError(
            f"run {paused[0]} has used {attempts_used(root, paused[0])} of {args.max_attempts} attempts and is not "
            f"final: the study is paused (docs/prereg/TSM.md, Clarification C1; paused runs: {', '.join(paused)}); "
            "report it to the orchestrator: nothing is trained until its recorded decision grants more attempts "
            "(train --max-attempts N --decision FILE, Clarification C2)")
    rows = {row["name"]: row for row in planned_runs()}
    states = {name: run_state(root, name) for name in names}
    for name in names:
        if states[name] in ("complete", "training_failure"):
            registered_record(design, root, rows[name])  # a final record must be the registered run, as evaluate needs
    unfinished_runs = [name for name in names if states[name] == "incomplete"]
    require(not unfinished_runs or args.rerun_incomplete,
            f"unfinished runs {unfinished_runs}: if their job has ended, train them again, identically, with "
            f"--rerun-incomplete (at most {args.max_attempts - 1} reruns per run)")
    for name in unfinished_runs:
        require_ended(root, name)
    for name in names:  # every attempt after the first is compared with the one before it, also after an aborted rerun
        if states[name] in ("missing", "incomplete"):
            row = rows[name]
            config = make_config(design, row["writer"], row["seed"], row["arm"], device=args.device)
            check_rerun(root, name, config_hash=stable_hash(config), source=source, max_attempts=args.max_attempts)
    started = time.perf_counter()
    for name in unfinished_runs:
        move_aside(root, name)
    for name in names:
        if states[name] == "missing":
            retire_stale_log(root, name)
    todo = [name for name in names if states[name] in ("missing", "incomplete")]
    reruns = [name for name in todo if moved_attempts(root, name)]  # attempt 2 or 3, whoever moved the earlier ones
    manifest = dict(status="running", design_hash=stable_hash(design), design_path=str(design_path),
                    design_lock_commit=args.design_commit, names=names, started=todo, reruns=reruns,
                    final_before=[name for name in names if name not in todo], procs=args.procs,
                    max_attempts=args.max_attempts, decision=decision, started_at=now(), source=source,
                    runtime=runtime_metadata(args.device))
    write_json(manifest_path, manifest)
    try:
        schedule(todo, args.procs, design_path, root, device=args.device, max_attempts=args.max_attempts,
                 fatal=lambda name: stops_study(root, name))
    except ValueError as error:
        stopped = study_stop(root)
        reason = stopped or str(error)
        manifest.update(status="stopped" if stopped else "failed", reason=reason, finished_at=now(),
                        run_states={name: run_state(root, name) for name in names})  # what the running children left
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
            f"trained again with --rerun-incomplete once its job has ended, at most {args.max_attempts - 1} times")
    return 0


def stage_train_one(args: argparse.Namespace) -> int:
    design = json.loads(Path(args.design).read_text())
    record = train_one(design, Path(args.root), args.name, device=args.device, steps=args.steps,
                       max_attempts=args.max_attempts)
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


def moved_evaluations(root: Path) -> list[Path]:
    """Interrupted evaluations that were moved to ``root/attempts``; nothing in them is ever read."""
    folder = root / "attempts"
    return sorted(path for path in folder.glob("evaluation-attempt*") if path.is_dir()) if folder.is_dir() else []


def evaluation_state(root: Path) -> str:
    """``none`` (no evaluation directory in place), ``complete``, ``failed`` (the job recorded its own end) or ``running``
    (still going, or ended before it completed; also a directory without a manifest, or with one that was cut off while
    it was written). Only the manifest's ``status`` is read: nothing else of an evaluation in place."""
    if not (root / "evaluation").exists():
        return "none"
    status = (read_json_record(root / "evaluation" / "manifest.json") or {}).get("status")
    return status if status in ("complete", "failed") else "running"


def earlier_evaluation(root: Path) -> dict | None:
    """The manifest of the latest earlier evaluation attempt that has a readable one: in place, else the last moved
    aside (a restart that stopped before its manifest was written leaves the earlier attempt only there). It holds the
    inputs of the attempt, never a result: the samples are not read."""
    for folder in [root / "evaluation", *reversed(moved_evaluations(root))]:
        manifest = read_json_record(folder / "manifest.json")
        if manifest is not None:
            return manifest
    return None


def checkpoint_seals(root: Path, states: dict[str, str]) -> dict[str, str]:
    """The blake2b seal of the checkpoint of every complete run, from its record (``final_states`` checked the file)."""
    return {name: read_record(root, name)["checkpoint_blake2b"] for name in sorted(states) if states[name] == "complete"}


def training_source_hash(root: Path) -> str:
    """The ``source_hash`` that the 18 runs were trained at (``final_states`` has checked that it is one)."""
    return read_record(root, planned_runs()[0]["name"])["source"]["source_hash"]


def evaluation_inputs(manifest: dict) -> dict:
    """What an evaluation is a function of: the design and its lock, the code and the trained checkpoints (the B4
    references are sealed in the design)."""
    return dict(design=manifest.get("design_hash"), design_lock_commit=manifest.get("design_lock_commit"),
                code=manifest.get("source", {}).get("source_hash"), checkpoints=manifest.get("checkpoints"))


def evaluation_attempt(root: Path, manifest: dict, state: str, *, new_code: bool = False) -> int:
    """The number of the evaluation attempt that may start now: 1, or 2 and 3 after an attempt that did not complete.

    The first attempt runs at the code that the 18 runs were trained at (``training_source_hash``; docs/prereg/TSM.md,
    Clarifications C1 and C2). An evaluation that ended before it completed is started again from scratch with
    identical inputs, at most twice (Clarification C1): ``MAX_ATTEMPTS`` attempts in all. Identical means the design and
    its lock, the code (``source_hash``) and the checkpoints of the latest earlier attempt (``manifest`` is this
    attempt's); with ``new_code`` (the orchestrator's decision after a defect in the evaluation code, Clarification C2)
    the code may differ, nothing else. ``new_code`` needs an earlier attempt, and without one that has a readable
    manifest to compare with, the attempt runs at the training code. An attempt still marked "running" must have been
    silent for ``QUIET_SECONDS`` (a live evaluation writes ``heartbeat``): its job may be alive and would write into the
    new directory; one marked "failed" has recorded its own end.
    """
    used = len(moved_evaluations(root)) + int(state != "none")
    require(used < MAX_ATTEMPTS, f"the confirmation evaluation has used {used} of {MAX_ATTEMPTS} attempts: the "
            "registered rule allows two restarts; report it to the orchestrator")
    require(used > 0 or not new_code, "--new-evaluation-code applies to the restart of an evaluation that has an "
            "earlier attempt: the first attempt runs at the training code (docs/prereg/TSM.md, Clarifications C1 and "
            "C2)")
    if state == "running":
        out = root / "evaluation"
        silent = seconds_since_written([out, *out.rglob("*")])
        require(silent >= QUIET_SECONDS, f"the evaluation was written to {silent:.0f} s ago: its job may still be "
                f"running; check qstat and restart it only after the job has ended (and {QUIET_SECONDS} s of silence)")
    earlier = earlier_evaluation(root)
    if earlier is not None:
        old, new = evaluation_inputs(earlier), evaluation_inputs(manifest)
        differing = [key for key in new if old[key] != new[key] and not (key == "code" and new_code)]
        hint = ("; a fix of the evaluation code is restarted with --new-evaluation-code, on the orchestrator's "
                "decision (docs/prereg/TSM.md, Clarification C2)" if "code" in differing else "")
        require(not differing, f"the restart differs from the interrupted evaluation in {', '.join(differing)}: a "
                f"restart must have identical inputs (the same design lock, --commit and trained runs){hint}")
    else:  # the first attempt, or a restart with no manifest to compare with: the code the runs were trained at
        code, trained = manifest["source"]["source_hash"], manifest["training_source_hash"]
        require(code == trained or new_code, f"the evaluation runs at source_hash {code[:12]} but the 18 runs were "
                f"trained at {trained[:12]}: an evaluation attempt with no earlier manifest to compare with runs at "
                "the training code (docs/prereg/TSM.md, Clarifications C1 and C2; submit it at the training --commit)"
                + ("; a fix of the evaluation code is restarted with --new-evaluation-code, on the orchestrator's "
                   "decision" if used else ""))
    return used + 1


def restart_rows(root: Path, attempt: int) -> list[dict]:
    """One row per evaluation that was moved aside before attempt ``attempt``: where it is and the code it ran at.

    The code of the interrupted attempt is in the manifest of the latest earlier attempt that has a readable one, which
    carries the rows of its own predecessors; no other manifest is opened (``None``: not known).
    """
    earlier = earlier_evaluation(root)
    codes = {row.get("attempt"): row.get("source_hash") for row in (earlier or {}).get("restarts", [])}
    if earlier is not None:
        codes[earlier.get("attempt")] = earlier.get("source", {}).get("source_hash")
    return [dict(attempt=number, moved_to=f"attempts/evaluation-attempt{number}", source_hash=codes.get(number))
            for number in range(1, attempt)]


def move_evaluation_aside(root: Path) -> Path:
    """Move the interrupted evaluation, unread, to ``root/attempts/evaluation-attempt<k>``."""
    target = root / "attempts" / f"evaluation-attempt{len(moved_evaluations(root)) + 1}"
    target.parent.mkdir(parents=True, exist_ok=True)
    (root / "evaluation").rename(target)
    return target


@contextlib.contextmanager
def heartbeat(path: Path):
    """Write ``path`` now and again every ``HEARTBEAT_SECONDS`` while the block runs, from a thread of its own.

    An evaluation writes nothing for as long as a run takes to evaluate, and ``evaluation_attempt`` reads "nothing
    written for ``QUIET_SECONDS``" as "its job has ended": the beat is the sign of life. A beat that cannot be written
    (a full disk) is skipped: it is not a result and must not end the evaluation.
    """
    stop = threading.Event()

    def beat() -> None:
        try:
            path.write_text(now() + "\n")
        except OSError:
            pass

    def beats() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            beat()

    thread = threading.Thread(target=beats, name="evaluation-heartbeat", daemon=True)
    beat()  # from the start, not only after the first interval
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def stage_evaluate(args: argparse.Namespace) -> int:
    """Evaluate T0, T1, T2 and the B4 references on the confirmation episodes, once.

    An evaluation that has started is not repeated (``complete``), and one that ended before it completed is started
    again from scratch only with ``--restart-interrupted``, once its job has ended and with identical inputs, at most
    twice (``evaluation_attempt``): the partial output is moved aside unread, and the manifest records the attempt, the
    restarts, the code each attempt ran at and the code the runs were trained at. The first attempt runs at the training
    code. ``--new-evaluation-code`` lets a restart (not the first attempt) run at another code (the orchestrator's
    decision, Clarification C2); the 18 training records must still share one ``source_hash`` (``final_states``). All
    of this is checked before the confirmation episodes are generated. While it runs, the evaluation writes
    ``evaluation/heartbeat`` every ``HEARTBEAT_SECONDS`` (``heartbeat``).
    """
    require(args.restart_interrupted or not args.new_evaluation_code,
            "--new-evaluation-code applies to a restart: it needs --restart-interrupted")
    root = absolute_path(args.root, "--root")
    design_path = Path(args.design).resolve()
    verify_git_lock(design_path, args.design_commit, "the design")
    design = json.loads(design_path.read_text())
    verify_design(design)
    out = root / "evaluation"
    state = evaluation_state(root)
    require(state != "complete", "the confirmation evaluation has already started and is complete: it is not repeated")
    require(state == "none" or args.restart_interrupted,
            "the confirmation evaluation has already started and did not complete: if its job has ended, restart it "
            "from scratch with --restart-interrupted (at most twice; its partial output is moved aside unread)")
    states = final_states(design, root)  # before anything touches the confirmation set; refuses a stop
    failures = [failure_row(root, name) for name in sorted(states) if states[name] == "training_failure"]
    started = time.perf_counter()
    manifest = dict(status="running", design=design, design_hash=stable_hash(design), design_path=str(design_path),
                    design_lock_commit=args.design_commit, evaluation=design["evaluation"],
                    training_failures=failures, checkpoints=checkpoint_seals(root, states),
                    training_source_hash=training_source_hash(root), started_at=now(),
                    source=source_metadata(), runtime=runtime_metadata(args.device))
    attempt = evaluation_attempt(root, manifest, state, new_code=args.new_evaluation_code)
    manifest.update(attempt=attempt, restarts=restart_rows(root, attempt))
    if state != "none":
        move_evaluation_aside(root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    with heartbeat(out / "heartbeat"):
        try:
            _, rows, profiles = evaluate_runs(design, root, writers=WRITERS, seeds=SEEDS,
                                              evaluation=design["evaluation"], device=args.device,
                                              out=out / "samples", skip={row["name"] for row in failures})
            for template in design["templates"]:
                require(digest_checkpoint(Path(template["source"])) == template["source_blake2b"],
                        "a released checkpoint changed")
            verify_design(design)
            write_json(out / "gate_profiles.json", profiles)
        except BaseException as error:  # the job has ended: say so, the manifest would otherwise stay "running"
            manifest.update(status="failed", failure=f"{type(error).__name__}: {error}", finished_at=now())
            write_json(out / "manifest.json", manifest)
            raise
        seconds = time.perf_counter() - started
        manifest.update(status="complete", evaluations=rows, finished_at=now(), total_seconds=seconds,
                        node_hours=seconds / 3600.0, checkpoints_unchanged=True)
        write_json(out / "manifest.json", manifest)
    print(json.dumps(dict(stage="evaluate", status="complete", attempt=attempt, evaluated=len(rows),
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
    train_stage.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                             help="attempts allowed per run, the first one included (default 3, the registered "
                                  "number); more only on the orchestrator's recorded decision, which needs "
                                  "--decision (docs/prereg/TSM.md, Clarifications C1 and C2); every attempt stays "
                                  "identical to the one before it")
    train_stage.add_argument("--decision", help="the orchestrator's recorded decision (an absolute path to a "
                                                "non-empty file); required with --max-attempts above 3; the manifest "
                                                "records its path and blake2b digest")
    train_stage.add_argument("--rerun-incomplete", action="store_true",
                             help="train unfinished runs again (walltime or a node fault, an I/O or serialization "
                                  "error, an allocation failure), identically (their job must have ended: a run still "
                                  "marked running must have been silent for 300 s; at most twice per run, or "
                                  "--max-attempts attempts in all); without it an unfinished run is an error")
    train_stage.add_argument("--writers", nargs="+", default=list(WRITERS), choices=WRITERS)
    train_stage.add_argument("--arms", nargs="+", default=list(tsm.ARMS), choices=tsm.ARMS)
    train_stage.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    child = sub.add_parser("train-one", help="internal: train one run (started by the train and smoke stages)")
    child.add_argument("--design", required=True)
    child.add_argument("--root", required=True)
    child.add_argument("--name", required=True)
    child.add_argument("--steps", type=int)
    child.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    evaluate = sub.add_parser("evaluate", help="confirmation evaluation")
    evaluate.add_argument("--design", required=True)
    evaluate.add_argument("--design-commit", required=True)
    evaluate.add_argument("--root", required=True, help="the training run directory")
    evaluate.add_argument("--restart-interrupted", action="store_true",
                          help="start an evaluation that ended before it completed again from scratch, with identical "
                               "inputs (its job must have ended: one still marked running must have been silent for "
                               "300 s; at most twice); its partial output is moved aside unread")
    evaluate.add_argument("--new-evaluation-code", action="store_true",
                          help="with --restart-interrupted, on the orchestrator's decision after a defect in the "
                               "evaluation code (docs/prereg/TSM.md, Clarification C2): the restart may run at another "
                               "source_hash than the interrupted attempt; refused without an earlier attempt (the "
                               "first attempt runs at the training code); the 18 training records must still share "
                               "one source_hash, and the manifest records the code of every attempt")
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
