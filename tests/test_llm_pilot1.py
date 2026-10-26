"""Binding Pilot-1 selection boundaries and writer-only configuration."""
from __future__ import annotations

from copy import deepcopy

import pytest

from looplora.llm.pilot1_select import (
    LEARNING_RATES, POLICIES, choose_steps, select_capacity, select_policy,
    validate_writer_config, writer_config,
)


def policy_rows(scores):
    return [{"E": 16, "write_policy": policy, "lr": lr, "correct": score, "n": 1000}
            for (policy, lr), score in zip([(p, lr) for p in POLICIES for lr in LEARNING_RATES], scores, strict=True)]


def capacity_rows(scores):
    return [{"E": E, "write_policy": "fact_final", "lr": .0003, "correct": score, "n": 1000}
            for E, score in zip((8, 12, 16, 24), scores, strict=True)]


def test_writer_config_fixes_scope_and_keeps_training_q4():
    config = writer_config("policy-fact-final", 16, "fact_final", .001, 16, 1200)
    assert config["stage"] == "writer" and config["variant"] == "M1" and config["seed"] == 17
    assert config["model"]["r"] == 32 and config["model"]["layer"] == 14
    assert config["task"]["name"] == "t1" and config["task"]["E"] == 16
    assert config["task"]["Q"] == 4 and config["task"]["chain_lengths"] is None
    assert config["task"]["n_rel"] == 4 and config["task"]["head_sharing"] == .3
    assert config["train"]["steps"] == 1200 and config["train"]["batch_size"] == 16
    assert config["train"]["depths"] == [1]
    assert config["eval"]["episodes"] == 1000 and config["eval"]["seed"] == 4242
    assert config["eval"]["splits"] == ["val"] and config["eval"]["protocols"] == ["memory_only"]
    assert not config["eval"]["counterfactuals"] and not config["eval"]["allow_confirmation"]


@pytest.mark.parametrize("group,key,value", [
    (None, "stage", "joint"), (None, "variant", "B3"), (None, "seed", 29),
    (None, "protocol", "context_retained"), (None, "smoke", True),
    (None, "writer_checkpoint", "writer.pt"),
    ("task", "Q", 1), ("task", "head_sharing", 0.), ("task", "E", 32),
    ("model", "r", 16), ("model", "layer", 13),
    ("eval", "splits", ["test"]), ("eval", "splits", ["confirm"]),
    ("eval", "episodes", 200), ("eval", "seed", 20261027),
    ("eval", "counterfactuals", True), ("eval", "allow_confirmation", True),
    ("eval", "protocols", ["memory_only", "context_retained"]), ("train", "depths", [1, 2, 4]),
])
def test_scope_guard_rejects_overrides_before_execution(group, key, value):
    config = writer_config("pilot", 16, "fact_final", .0003, 8, 100)
    (config if group is None else config[group])[key] = value
    with pytest.raises(ValueError):
        validate_writer_config(config)


@pytest.mark.parametrize("kwargs", [
    {"E": 32}, {"E": True}, {"policy": "chosen-later"}, {"lr": .01},
    {"lr": float("nan")}, {"batch": 4}, {"batch": True}, {"steps": 0},
])
def test_writer_config_rejects_values_outside_registered_design(kwargs):
    arguments = dict(name="pilot", E=16, policy="fact_final", lr=.0003, batch=8, steps=100)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        writer_config(**arguments)


def test_policy_tolerance_uses_global_best_then_policy_then_lr():
    rows = policy_rows([944, 945, 949, 950, 949, 948])
    before = deepcopy(rows)
    chosen = select_policy(rows)
    assert chosen["write_policy"] == "fact_final" and chosen["lr"] == .001 and chosen["correct"] == 945
    assert select_policy(list(reversed(rows))) == chosen
    assert rows == before and all(chosen is not row for row in rows)


def test_policy_six_correct_is_outside_tolerance_and_lower_lr_breaks_policy_ties():
    rows = policy_rows([949, 949, 949, 950, 949, 955])
    chosen = select_policy(rows)
    assert chosen["write_policy"] == "all_tokens" and chosen["lr"] == .0003
    assert select_policy(policy_rows([950, 950, 950, 950, 950, 950]))["lr"] == .0003


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_E", "wrong_n", "fractional", "too_many_correct"])
def test_policy_selection_requires_complete_valid_primary_counts(mutation):
    rows = policy_rows([950] * 6)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    elif mutation == "wrong_E":
        rows[0]["E"] = 24
    elif mutation == "wrong_n":
        rows[0]["n"] = 4000
    elif mutation == "fractional":
        rows[0]["correct"] = .95
    else:
        rows[0]["correct"] = 1001
    with pytest.raises(ValueError):
        select_policy(rows)


@pytest.mark.parametrize("scores,expected", [
    ([1000, 1000, 950, 950], {"E_star": 24, "flagged": False, "diagnosis_required": False}),
    ([1000, 1000, 950, 949], {"E_star": 16, "flagged": False, "diagnosis_required": False}),
    ([1000, 950, 949, 949], {"E_star": 12, "flagged": False, "diagnosis_required": False}),
    ([800, 949, 949, 949], {"E_star": 8, "flagged": True, "diagnosis_required": False}),
    ([799, 949, 949, 949], {"E_star": None, "flagged": False, "diagnosis_required": True}),
])
def test_capacity_uses_exact_thresholds_without_tuning(scores, expected):
    assert select_capacity(capacity_rows(scores)) == expected


def test_capacity_rejects_duplicate_E_and_mixed_selected_pairs():
    rows = capacity_rows([950] * 4)
    duplicate = [*rows[:3], rows[0]]
    with pytest.raises(ValueError, match="exactly one row"):
        select_capacity(duplicate)
    rows[-1]["write_policy"] = "all_tokens"
    with pytest.raises(ValueError, match="one selected policy/lr"):
        select_capacity(rows)


def throughput_row(**updates):
    return {"E": 24, "write_policy": "all_tokens", "batch_size": 16, "procs": 2,
            "worst_worker_p95_step_seconds": .5, **updates}


def test_step_budget_uses_p95_margin_reserve_and_rounds_down_100():
    row = throughput_row()
    steps = choose_steps(row)
    assert steps == 2200
    assert 1.5 * row["worst_worker_p95_step_seconds"] * steps + 90 <= 1800
    assert 1.5 * row["worst_worker_p95_step_seconds"] * (steps + 100) + 90 > 1800
    assert choose_steps(throughput_row(worst_worker_p95_step_seconds=1.)) == 1100


@pytest.mark.parametrize("updates", [
    {"E": 16}, {"write_policy": "fact_final"}, {"batch_size": 4}, {"procs": 3},
    {"worst_worker_p95_step_seconds": 0}, {"worst_worker_p95_step_seconds": True},
    {"worst_worker_p95_step_seconds": float("nan")}, {"worst_worker_p95_step_seconds": float("inf")},
    {"worst_worker_p95_step_seconds": 20},
])
def test_step_budget_rejects_wrong_benchmark_or_infeasible_budget(updates):
    with pytest.raises(ValueError):
        choose_steps(throughput_row(**updates))
