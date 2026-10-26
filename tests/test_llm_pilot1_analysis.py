"""Protect the pre-capacity policy decision and bounded shard discovery."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pilot1_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_p1l_pilot1.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def rows():
    return [dict(E=16, write_policy=policy, lr=lr, correct=900, n=1000,
                 candidate_correct=920, run_path=f"/runs/{policy}-{lr}",
                 git_commit="fixed-source", config_hash=f"config-{policy}-{lr}")
            for policy in ("fact_final", "all_tokens") for lr in (.0003, .001, .003)]


def test_policy_lock_rejects_changed_budget_and_source_even_when_winner_matches(tmp_path):
    candidates = rows()
    chosen = analysis.select_policy(candidates)
    throughput = dict(batch_size=32, procs=2)
    context = analysis._policy_context(candidates, throughput, 1500)
    locked = dict(selected=chosen, context_hash=context, created_at="2026-10-26T12:00:00+00:00")
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(locked))
    assert analysis._policy_lock(path, chosen, context) == locked
    changed_source = deepcopy(candidates)
    changed_source[-1]["git_commit"] = "changed-source"
    for changed in (analysis._policy_context(candidates, dict(batch_size=16, procs=2), 1500),
                    analysis._policy_context(candidates, throughput, 1600),
                    analysis._policy_context(changed_source, throughput, 1500)):
        with pytest.raises(ValueError, match="budget or run provenance"):
            analysis._policy_lock(path, chosen, changed)
    assert json.loads(path.read_text()) == locked


def test_capacity_accepts_three_single_run_shards_and_rejects_duplicate_names(tmp_path):
    for E in (8, 12, 24):
        shard = tmp_path / f"shard-{E}"
        leaf = shard / f"E{E}"
        leaf.mkdir(parents=True)
        (leaf / "config.json").write_text("{}")
        (shard / "suite.json").write_text(json.dumps(
            dict(status="complete", procs=2, runs={leaf.name: dict(status="complete")})))
    assert len(analysis._run_paths(tmp_path, 3, 2)) == 3
    shard = tmp_path / "shard-24"
    (shard / "E24").rename(shard / "E8")
    (shard / "suite.json").write_text(json.dumps(
        dict(status="complete", procs=2, runs={"E8": dict(status="complete")})))
    with pytest.raises(ValueError, match="duplicate writer directory names"):
        analysis._run_paths(tmp_path, 3, 2)


def test_missing_policy_lock_fails_before_capacity_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "_audit_policy", lambda *args: (rows(), [], dict(batch_size=32, procs=2), 1500))
    def unexpected_read(*args):
        raise AssertionError("capacity was opened without a prior policy decision")
    monkeypatch.setattr(analysis, "_run_paths", unexpected_read)
    with pytest.raises(ValueError, match="persisted pre-capacity policy selection"):
        analysis.analyze(tmp_path, tmp_path / "capacity", tmp_path / "throughput.json", tmp_path)


def test_naive_policy_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        analysis._timestamp("2026-10-26T12:00:00")
