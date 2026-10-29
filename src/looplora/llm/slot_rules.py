"""SLOT-LOOP-v1 registered selection rules that the runners apply online (torch-free; docs/prereg/SLOT-LOOP-v1.md).

The decision of S0 (rows R0 to R5) and the S1A verdict live in `scripts/analyze_slot_study.py`, the frozen analysis script; the
rules here are the ones a job needs while it runs: the calibration choice, the LR choice, the step count and the checkpoint
selection.
"""
from __future__ import annotations

import json
import math

from looplora.infra.seeds import stable_hash
from . import slot_data

ARMS = ("matrix_L1", "soft_slot_L1")
ARM_KIND = {"matrix_L1": "matrix", "soft_slot_L1": "soft"}
TRAINABLE = ("Q", "R", "U", "B")
FROZEN = ("A", "K", "V", "a_lambda", "a_beta")
BUFFERS = ("input_mean", "input_std")
ABLATION_TRAINABLE = ("A_query", *TRAINABLE)
OPTIMIZER = dict(name="adamw", betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
CLIP, WARMUP, CHECKPOINTS, LOG_EVERY = 1.0, 20, 10, 50
CALIBRATION_TIE = 1e-4   # nats: candidates within this of the minimum NLL are tied
LR_TIE = 0.01            # nats on the pilot shard
SELECTION_TIE = 1e-6     # nats on dev: a later checkpoint must beat the best so far by more than this
LR_GRID = (3e-4, 1e-3, 3e-3)
GAINS = (0.5, 1.0, 2.0)
TAUS = (0.03, 0.1, 0.3, 1.0)
WALL_SECONDS, MARGIN, RESERVE, EVALUATIONS = 3600, 1.5, 90, 11
N_MIN_OK = 3000          # below this the orchestrator decides (a 2 h scale-up or another plan)


class StopForOrchestrator(RuntimeError):
    """A registered rule ends in a STOP: the orchestrator decides, the worker does not improvise."""


def calibration_grid() -> list[dict]:
    """The 18 registered candidates: Matrix g (3), SoftSlot (tau, g) (12), HardSlot g (3)."""
    return ([dict(kind="matrix", gain=g, tau=None) for g in GAINS]
            + [dict(kind="soft", gain=g, tau=t) for t in TAUS for g in GAINS]
            + [dict(kind="hard", gain=g, tau=None) for g in GAINS])


def choose_candidate(rows: list[dict], kind: str) -> dict:
    """Lowest calibration NLL among the candidates of `kind`; within CALIBRATION_TIE of the minimum, the gain nearest to 1
    (absolute distance), then the larger temperature. Non-finite NLLs are excluded."""
    pool = [r for r in rows if r["kind"] == kind and math.isfinite(r["nll"])]
    if not pool:
        raise StopForOrchestrator(f"no finite {kind} calibration candidate")
    best = min(r["nll"] for r in pool)
    tied = [r for r in pool if r["nll"] <= best + CALIBRATION_TIE]
    return min(tied, key=lambda r: (abs(r["gain"] - 1.0), -(r["tau"] or 0.0)))


def choose_lr(rows: list[dict], step0_nll: float) -> dict:
    """Lowest pilot-shard NLL; within LR_TIE of the best, the lower LR. Non-finite candidates are excluded; if every candidate
    is worse than the arm's step 0 the orchestrator decides (the grid is not extended)."""
    pool = [r for r in rows if r["nll"] is not None and math.isfinite(r["nll"]) and r.get("all_finite", True)]
    if not pool or all(r["nll"] > step0_nll for r in pool):
        raise StopForOrchestrator("every LR candidate is non-finite or worse than step 0")
    best = min(r["nll"] for r in pool)
    return min((r for r in pool if r["nll"] <= best + LR_TIE), key=lambda r: r["lr"])


def schedule_steps(step_seconds: float, eval_seconds: float, load_seconds: float, *, max_steps: int = 6000) -> int:
    """N = min(6000, 10 * floor(((3600 - max(90, Ld)) / 1.5 - 11 Ev) / S / 10)): the largest multiple of ten whose estimated
    job time (load, N steps, eleven 1,000-episode evaluations, margin 1.5) fits one hour."""
    if not all(math.isfinite(v) and v > 0 for v in (step_seconds, eval_seconds, load_seconds)):
        raise ValueError("schedule requires positive finite timings")
    available = (WALL_SECONDS - max(RESERVE, load_seconds)) / MARGIN - EVALUATIONS * eval_seconds
    steps = 10 * math.floor(available / step_seconds / 10)
    if steps < 10:
        raise StopForOrchestrator("evaluations and setup leave no training budget")
    return min(max_steps, steps)


def job_walltime_seconds(step_seconds: float, eval_seconds: float, load_seconds: float, steps: int) -> int:
    """(Ld + N S + 11 Ev) x 1.5 rounded up to five minutes; at most one hour."""
    seconds = math.ceil((load_seconds + steps * step_seconds + EVALUATIONS * eval_seconds) * MARGIN / 300) * 300
    if seconds > WALL_SECONDS:
        raise StopForOrchestrator("the job does not fit one hour; a 2 h scale-up needs the orchestrator's OK")
    return seconds


def select_checkpoint(evaluations: list[tuple[int, float]]) -> int:
    """The step of the selected checkpoint among the selectable evaluations [(step, dev NLL)] in step order: the lowest NLL;
    a later checkpoint replaces the best so far only if it is lower by more than SELECTION_TIE (a tie goes to the earlier)."""
    best_step, best = None, math.inf
    for step, nll in sorted(evaluations):
        if math.isfinite(nll) and nll < best - SELECTION_TIE:
            best_step, best = step, nll
    if best_step is None:
        raise StopForOrchestrator("no finite checkpoint evaluation")
    return best_step


def checkpoint_steps(steps: int, count: int = 10) -> list[int]:
    """k N / 10 for k = 1..10 (N is a multiple of ten, so all are integers)."""
    if steps % count:
        raise ValueError("N must be a multiple of the number of checkpoints")
    return [steps * k // count for k in range(1, count + 1)]


def science_config(arm: str, seed: int, *, steps: int, lr: float, read: dict, parent_digest: str, ablation: bool = False,
                   stage: str = "S1A") -> dict:
    """Everything that defines one arm-seed run except the commit: the release stores its `stable_hash`, and the run
    recomputes it from the release's values before it trains."""
    if arm not in ARMS or seed not in slot_data.RUN_SEEDS or steps % CHECKPOINTS or steps < CHECKPOINTS:
        raise ValueError("a run is one registered arm and seed with N a multiple of the number of checkpoints")
    if read["kind"] != ARM_KIND[arm] or read.get("scope") != "all_valid_question" or read.get("recency"):
        raise ValueError("the arm's read must be its own kind at all_valid_question without recency")
    return dict(study=slot_data.STUDY, stage=stage, arm=arm, seed=seed, steps=steps, lr=lr, read=dict(read), ablation=ablation,
                loop_steps=1, optimizer=dict(OPTIMIZER), clip=CLIP, warmup=WARMUP, batch=slot_data.BATCH, checkpoints=CHECKPOINTS,
                trainable=list(ABLATION_TRAINABLE if ablation else TRAINABLE), frozen=list(FROZEN), buffers=list(BUFFERS),
                train_base_seed=slot_data.TRAIN_BASE[stage, seed], dev=slot_data.SETS[stage, "dev"].name,
                dev_base_seed=slot_data.SETS[stage, "dev"].base_seed, parent_writer_blake2b_128=parent_digest,
                prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128, layout_digest=slot_data.LAYOUT_DIGEST, task=dict(slot_data.TASK))


def science_hash(config: dict) -> str:
    return stable_hash(config)


def _set_rows(stage: str, *, sealed: bool = False) -> dict:
    """The registered evaluation sets of a stage; sealed ones are named only (their seeds live in slot_data alone)."""
    rows = {}
    for (name, purpose), s in slot_data.SETS.items():
        if name == stage and s.sealed == sealed:
            rows[purpose] = dict(sealed=True, split=s.split, start=s.start, count=s.count) if sealed else dict(
                split=s.split, start=s.start, count=s.count, base_seed=s.base_seed)
    return rows


def stage_config(stage: str) -> dict:
    """The frozen stage configuration (`configs/slot_loop_v1/<stage>.json`): the registered values the runner executes, as the
    JSON it is written in. The runner refuses a file that differs from this rendering, so an edited config cannot change a
    stage silently."""
    return json.loads(json.dumps(_stage_config(stage)))


def _stage_config(stage: str) -> dict:
    common = dict(study=slot_data.STUDY, prereg=slot_data.PREREG, prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128,
                  layout_digest=slot_data.LAYOUT_DIGEST, task=dict(slot_data.TASK), loop_steps=1, capacity=16, chunk=16,
                  parent=dict(name="S-RE", files=["writer.pt", "checkpoint.pt"]))
    if stage == "S0":
        return dict(common, stage="S0", recheck=dict(name="D128-3", files=["writer.pt", "checkpoint.pt"]),
                    sets=_set_rows("S0"), scopes=dict(primary="all_valid_question", diagnostic="final_only"),
                    calibration=dict(grid=calibration_grid(), tie_nats=CALIBRATION_TIE,
                                     choice="lowest NLL; ties: gain nearest 1 by |g - 1|, then the larger tau"),
                    conditions=["native", "matrix_uncal", "matrix_cal", "soft_uncal", "soft_cal", "hard_uncal", "hard_cal",
                                "soft_rec_uncal", "soft_rec_cal", "clean_final"],
                    decision=dict(em_threshold_pp=5.0, rule="(c_a - c_b) * 20 >= n and lower mean read error"),
                    bootstrap_seed=slot_data.BOOTSTRAP["S0"])
    if stage == "S1A":
        return dict(common, stage="S1A", arms=list(ARMS), seeds=list(slot_data.RUN_SEEDS), trainable=list(TRAINABLE),
                    frozen=list(FROZEN), buffers=list(BUFFERS), optimizer=dict(OPTIMIZER), clip=CLIP, warmup=WARMUP,
                    batch=slot_data.BATCH, checkpoints=CHECKPOINTS, log_every=LOG_EVERY, lr_grid=list(LR_GRID),
                    pilot=dict(steps=slot_data.PILOT_STEPS, stream_seed=slot_data.TRAIN_BASE["S1A", "pilot"], tie_nats=LR_TIE),
                    steps_rule=dict(max_steps=slot_data.MAX_STEPS, wall_seconds=WALL_SECONDS, margin=MARGIN, reserve=RESERVE,
                                    evaluations=EVALUATIONS, minimum=N_MIN_OK),
                    selection=dict(tie_nats=SELECTION_TIE), sets=_set_rows("S1A"), sealed_sets=_set_rows("S1A", sealed=True),
                    train_base={str(seed): slot_data.TRAIN_BASE["S1A", seed] for seed in slot_data.RUN_SEEDS},
                    ablation=dict(trainable=list(ABLATION_TRAINABLE), seed=17), bootstrap_seed=slot_data.BOOTSTRAP["S1A"])
    raise ValueError(f"no registered stage {stage}")
