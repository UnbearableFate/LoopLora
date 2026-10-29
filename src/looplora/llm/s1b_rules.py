"""Registered S1B branch, schedule, budgets and science configurations."""
from __future__ import annotations

import math

from looplora.infra.seeds import stable_hash
from . import s1b_data as data
from .slot_rules import (OPTIMIZER, TRAINABLE, LR_GRID, GAINS, TAUS, choose_candidate,
                         choose_lr, checkpoint_steps, select_checkpoint, StopForOrchestrator)

BRANCHES = ("advance", "matrix_only", "query_copy")
ARM = {"M1": ("matrix", 1), "M4": ("matrix", 4), "S1": ("soft", 1), "S4": ("soft", 4)}


def branch_arms(branch):
    if branch not in BRANCHES:
        raise ValueError("unknown registered S1B branch")
    return ("M1", "M4") if branch == "matrix_only" else tuple(ARM)


def contrasts(branch):
    return ["M4-M1"] if branch == "matrix_only" else ["S4-S1", "M4-M1", "S4-M4"]


def resolve_branch(decision, dev, ablation=None):
    """Consume the frozen S1A script's actual summary/dev/ablation schemas."""
    if dev.get("outcome") == "stop_unstable" or dev.get("reasons"):
        raise StopForOrchestrator("invalid/unstable S1A requires orchestrator disposition")
    if decision.get("stage") != "S1A" or decision.get("verdict") not in {"ADVANCE", "NOT ADVANCED"}:
        raise ValueError("S1A frozen decision record is required")
    if dev.get("stage") != "S1A-DEV" or dev.get("outcome") not in {"ablation", "no_trigger"}:
        raise ValueError("S1A frozen dev analysis is required")
    if dev["outcome"] == "ablation" and ablation is None:
        raise ValueError("triggered S1A ablation record is required")
    if ablation is not None and dev["outcome"] != "ablation":
        raise ValueError("untriggered S1A ablation record is forbidden")
    if ablation is not None and (ablation.get("stage") != "S1A-ABL" or not isinstance(ablation.get("flag"), bool)):
        raise ValueError("malformed S1A ablation record")
    digest = decision.get("release_blake2b_128")
    records = [decision, dev] + ([] if ablation is None else [ablation])
    if (not isinstance(digest, str) or len(digest) != 32 or any(c not in "0123456789abcdef" for c in digest)
            or any(record.get("release_blake2b_128") != digest for record in records)):
        raise ValueError("S1A branch records must share one release digest")
    if decision["verdict"] == "ADVANCE":
        return "advance"
    return "query_copy" if ablation is not None and ablation["flag"] else "matrix_only"


def trainable(branch, control="shared"):
    names = list(TRAINABLE)
    if control == "untied":
        names += [f"{name}_steps.{i}" for name in ("Q", "R", "U") for i in range(3)]
    elif control not in {"shared", "fixed"}:
        raise ValueError("unknown S1B control")
    return (["A_query"] if branch == "query_copy" else []) + names


def read_config(arm, gain=1., tau=None):
    from .slot_read import ReadSpec
    kind, _ = ARM[arm]
    return ReadSpec(kind, gain=gain, tau=(.1 if tau is None else tau) if kind == "soft" else None, capacity=12)


def science_config(branch, arm, seed, lr, read, *, control="shared"):
    if arm not in branch_arms(branch) or seed not in data.RUN_SEEDS or lr not in LR_GRID:
        raise ValueError("unregistered arm, seed or learning rate")
    if read["kind"] != ARM[arm][0] or read["capacity"] != 12 or read["scope"] != "all_valid_question" or read["recency"]:
        raise ValueError("S1B read differs from registered protocol")
    if read["gain"] not in GAINS or (read["kind"] == "soft" and read["tau"] not in TAUS):
        raise ValueError("S1B gain/temperature outside calibration grid")
    return dict(stage="S1B", branch=branch, arm=arm, seed=seed, lr=lr, read=read,
                control=control, loop_depth=ARM[arm][1], task_digest=data.LAYOUT_DIGEST,
                steps=data.STEPS, batch=data.BATCH, train_base=data.TRAIN_BASE[seed],
                trainable=trainable(branch, control), optimizer=OPTIMIZER, warmup=20, clip=1.,
                checkpoints=checkpoint_steps(data.STEPS), prereg_blake2b_128=data.PREREG_DIGEST)


def walltime(step, evaluation, load, io=0., steps=data.STEPS):
    times = (step, evaluation, load, io)
    if any(not math.isfinite(v) or v < 0 for v in times) or step == 0 or evaluation == 0:
        raise ValueError("missing/non-finite measured cost")
    seconds = math.ceil(1.5 * (load + steps * step + 11 * evaluation + io) / 600) * 600
    if seconds > 16 * 3600:
        raise StopForOrchestrator("indivisible S1B run exceeds 16 node-hours")
    return seconds


def sanity(calibration):
    if calibration["selected"]["matrix"]["per_hop"]["1"]["em"] < .10:
        raise StopForOrchestrator("calibrated Matrix parent hop-1 calibration EM below 10%")


def choose_calibration(rows, kind):
    """The inherited selection rule after explicitly excluding failed/nonfinite candidates."""
    finite = [row for row in rows if row.get("nll") is not None and math.isfinite(row["nll"])]
    return choose_candidate(finite, kind)


def config_hashes(release):
    return {f"{arm}/{seed}": stable_hash(science_config(release["branch"], arm, seed,
                release["arms"][arm]["lr"], release["arms"][arm]["read"]))
            for arm in branch_arms(release["branch"]) for seed in data.RUN_SEEDS}
