"""Fixed configuration, selection and step-budget rules for P1-L Pilot-1."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import floor, isfinite
from numbers import Integral, Real
from typing import Any

POLICIES = ("fact_final", "all_tokens")
LEARNING_RATES = (.0003, .001, .003)
CAPACITIES = (8, 12, 16, 24)
EVAL_EPISODES = 1000
RUN_SECONDS = 1800
TRAIN_MARGIN = 1.5
RESERVE_SECONDS = 90


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def validate_writer_config(config: Mapping[str, Any]) -> None:
    """Reject any configuration outside the authorized Pilot-1 writer scope."""
    expected = {"stage": "writer", "variant": "M1", "protocol": "memory_only",
                "seed": 17, "smoke": False, "writer_checkpoint": None}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Pilot-1 requires writer/M1, memory_only, seed 17, and no consumed writer")
    model, task, train, evaluation = (config[group] for group in ("model", "task", "train", "eval"))
    if model["r"] != 32 or model["layer"] != 14:
        raise ValueError("Pilot-1 fixes rank 32 and layer 14")
    if (task["name"] != "t1" or task["Q"] != 4 or task["n_rel"] != 4
            or task["head_sharing"] != .3 or task["hops"] != [1]
            or task["chain_lengths"] is not None or task["E"] not in CAPACITIES):
        raise ValueError("Pilot-1 requires T1 Q=4/n_rel=4/head_sharing=.3 and E in 8/12/16/24")
    if (evaluation["splits"] != ["val"] or evaluation["episodes"] != EVAL_EPISODES
            or evaluation["seed"] != 4242 or evaluation["protocols"] != ["memory_only"]
            or evaluation["depths"] != [1] or evaluation["hops"] != [1]
            or evaluation["counterfactuals"] is not False or evaluation["allow_confirmation"] is not False):
        raise ValueError("Pilot-1 permits only 1000 val episodes, seed 4242, memory_only L=1, no counterfactuals")
    if train["depths"] != [1] or config["write_policy"] not in POLICIES or train["lr"] not in LEARNING_RATES:
        raise ValueError("Pilot-1 fixes L1 and the registered policy/lr grid")
    if _integer(train["batch_size"], "batch_size") not in (8, 16, 32):
        raise ValueError("Pilot-1 batch must be 8, 16, or 32")
    _integer(train["steps"], "steps")


def writer_config(name: str, E: int, policy: str, lr: float, batch: int, steps: int) -> dict[str, Any]:
    """Build a resolved writer-only config without accepting scope overrides.

    T1 has no chain layout. The E16 (8,4,2,2) chain layout applies to later T2;
    the writer here has direct one-hop facts with the binding head sharing.
    """
    from looplora.llm.config import resolve_config

    if _integer(E, "E") not in CAPACITIES:
        raise ValueError("Pilot-1 capacity must be 8, 12, 16, or 24")
    if policy not in POLICIES or _positive(lr, "lr") not in LEARNING_RATES:
        raise ValueError("Pilot-1 policy/lr must be in the registered grid")
    if _integer(batch, "batch") not in (8, 16, 32):
        raise ValueError("Pilot-1 batch must be 8, 16, or 32")
    _integer(steps, "steps")
    config = resolve_config({
        "name": name, "seed": 17, "stage": "writer", "variant": "M1", "smoke": False,
        "protocol": "memory_only", "write_policy": policy,
        "model": {"r": 32, "layer": 14},
        "task": {"name": "t1", "E": int(E), "Q": 4, "chain_lengths": None},
        "train": {"steps": int(steps), "batch_size": int(batch), "lr": float(lr), "depths": [1]},
        "eval": {"episodes": EVAL_EPISODES, "splits": ["val"], "seed": 4242,
                 "depths": [1], "hops": [1], "protocols": ["memory_only"],
                 "counterfactuals": False, "allow_confirmation": False},
    })
    validate_writer_config(config)
    return config


def _row(row: Mapping[str, Any]) -> tuple[int, str, float, int]:
    if not isinstance(row, Mapping):
        raise ValueError("selection rows must be mappings")
    E = _integer(row["E"], "E")
    policy, lr = row["write_policy"], _positive(row["lr"], "lr")
    correct, n = _integer(row["correct"], "correct", 0), _integer(row["n"], "n")
    if E not in CAPACITIES or policy not in POLICIES or lr not in LEARNING_RATES:
        raise ValueError("selection row is outside the registered capacity/policy/lr grid")
    if n != EVAL_EPISODES or correct > n:
        raise ValueError("selection requires integer full-vocabulary EM counts out of exactly 1000")
    return E, policy, lr, correct


def select_policy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose from all six E16 cells; tie tolerance is five correct answers."""
    parsed = [_row(row) for row in rows]
    required = {(policy, lr) for policy in POLICIES for lr in LEARNING_RATES}
    if (len(parsed) != len(required) or any(E != 16 for E, _, _, _ in parsed)
            or {(policy, lr) for _, policy, lr, _ in parsed} != required):
        raise ValueError("policy selection requires all six distinct E16 policy/lr cells")
    best = max(correct for _, _, _, correct in parsed)
    eligible = [i for i, (_, _, _, correct) in enumerate(parsed) if best - correct <= 5]
    selected = min(eligible, key=lambda i: (parsed[i][1] != "fact_final", parsed[i][2]))
    return dict(rows[selected])


def select_capacity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered largest-capacity rule to one selected policy/lr."""
    parsed = [_row(row) for row in rows]
    if len(parsed) != 4 or {E for E, _, _, _ in parsed} != set(CAPACITIES):
        raise ValueError("capacity selection requires exactly one row at each E8/12/16/24")
    if len({(policy, lr) for _, policy, lr, _ in parsed}) != 1:
        raise ValueError("capacity rows must use one selected policy/lr pair")
    correct = {E: score for E, _, _, score in parsed}
    large = [E for E in (16, 24) if correct[E] >= 950]
    selected = max(large) if large else 12 if correct[12] >= 950 else 8 if correct[8] >= 800 else None
    return {"E_star": selected, "flagged": selected == 8, "diagnosis_required": selected is None}


def choose_steps(throughput: Mapping[str, Any]) -> int:
    """Reserve 90 seconds, margin training by 1.5, then round the step budget down to 100."""
    if throughput["E"] != 24 or throughput["write_policy"] != "all_tokens":
        raise ValueError("step budget requires the worst-case E=24/all_tokens benchmark")
    if _integer(throughput["batch_size"], "batch_size") not in (8, 16, 32):
        raise ValueError("benchmark batch_size must be 8, 16, or 32")
    if _integer(throughput["procs"], "procs") not in (1, 2):
        raise ValueError("benchmark procs must be 1 or 2")
    p95 = _positive(throughput["worst_worker_p95_step_seconds"], "worst_worker_p95_step_seconds")
    steps = floor((RUN_SECONDS - RESERVE_SECONDS) / (TRAIN_MARGIN * p95) / 100) * 100
    if steps < 100:
        raise ValueError("benchmark cannot fit 100 training steps within the 30-minute pilot budget")
    return steps
