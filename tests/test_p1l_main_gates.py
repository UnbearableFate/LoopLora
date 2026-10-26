"""Registered P1-L decision failures and held-confirmation behavior."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_p1l_main import check_training
from finalize_p1l_main import endpoints
from launch_p1l_main import walltime
from run_p1l_main import LABELS, SEEDS


def sample(depths, *, good=False, fail_seed=False, hop1_failure=False):
    rows = []
    for depth in depths:
        for episode in range(8):
            for hop in range(1, 9):
                correct = good and not fail_seed
                if hop1_failure and hop == 1:
                    correct = False
                rows.append(("confirm", "memory_only/original", episode, hop-1, hop, depth, 7, correct))
    names = ("split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct")
    return {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(names)}


def cohorts():
    result = {}
    for label in LABELS:
        depths = range(1, 9) if label == "M1-L4" else [1] if label in {"B4", "B1", "B2"} else [4]
        result[label] = {seed: sample(depths, good=label in {"B8", "M1-L4"}) for seed in SEEDS}
    return result


def test_global_training_gate_requires_orchestrator_decision_before_opening_runs():
    manifest = {"runs": {f"{label}-s{seed}": "/does-not-exist" for label in LABELS for seed in SEEDS}}
    manifest["runs"]["B0-s17"] = "/does-not-exist"
    with pytest.raises(ValueError, match="orchestrator"):
        check_training(manifest, dict(status="held", writer_conditions=1))
    manifest["runs"].pop("B2-s43")
    with pytest.raises(ValueError, match="all21"):
        check_training(manifest, dict(status="allowed", writer_conditions=1, orchestrator_decision="yes"))


def test_global_gate_rejects_missing_rank_or_an_incomplete_other_condition(tmp_path):
    manifest = {"rank": 128, "runs": {f"{label}-s{seed}": "/does-not-exist"
                for label in LABELS[:5] for seed in SEEDS}}
    with pytest.raises(ValueError, match="B3"):
        check_training(manifest, dict(status="allowed", orchestrator_decision="none selected", gated_conditions=1,
                                       condition_manifests={"oracle-r128": "/missing"}))
    import json
    current = tmp_path / "r128.json"
    current.write_text(json.dumps(manifest))
    other = tmp_path / "r32.json"
    other.write_text(json.dumps(dict(rank=32, runs={})))
    authorization = dict(status="allowed", orchestrator_decision="none selected", gated_conditions=2,
                         condition_manifests={"oracle-r32": str(other), "oracle-r128": str(current)})
    with pytest.raises(ValueError, match="all21"):
        check_training(manifest, authorization)


def test_job_margin_is_enforced_before_submission():
    assert walltime(3600) == "01:30:00"
    with pytest.raises(ValueError, match="above2h"):
        walltime(4801)


def test_g1_requires_positive_each_seed_and_typed_replication():
    primary, typed = cohorts(), cohorts()
    assert endpoints(primary, typed, .975)["passes"]
    primary["B8"][43] = sample([4], good=False)
    # Strong positive pooled effect cannot substitute for a positive effect in seed43.
    assert not endpoints(primary, typed, .975)["G1_L"]
    primary = cohorts()
    typed["B8"] = {seed: sample([4], good=False) for seed in SEEDS}
    assert not endpoints(primary, typed, .975)["G1_L"]


def test_g1_hop1_guard_and_g2_comparator_remain_separate():
    primary, typed = cohorts(), cohorts()
    primary["B4"] = {seed: sample([1], good=True, hop1_failure=True) for seed in SEEDS}
    primary["B8"] = {seed: sample([4], good=True, hop1_failure=True) for seed in SEEDS}
    # Give B4 hop1 correct, but keep its reach queries incorrect.
    for seed in SEEDS:
        primary["B4"][seed] = sample([1], good=False)
        primary["B4"][seed]["correct"][primary["B4"][seed]["hop"] == 1] = True
    result = endpoints(primary, typed, .95)
    assert not result["G1_L"] and result["G2_L"]
    primary = cohorts()
    primary["B5-L4"] = {seed: sample([4], good=True) for seed in SEEDS}
    result = endpoints(primary, typed, .95)
    assert result["G1_L"] and not result["G2_L"]
