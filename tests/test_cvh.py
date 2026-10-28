"""CVH (F9): continuation labels, policy accounting, constrained theta selection, shared machinery.

Run only inside a Miyabi-G compute allocation, including the CPU cases.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from looplora.core.learned_halting import (
    HaltingHead, learned_adaptive_read, learned_halting_steps, trace_frozen_reader,
)
from looplora.data.relations import generate
from looplora.sym import cvh
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.evaluate import stratify_queries
from looplora.sym.model import SymbolicModel, make_task_spec

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_p4s_learned as p4l  # noqa: E402
import run_p4s_stopping as p4s  # noqa: E402


def _config(writer, **updates):
    base = {
        "name": "cvh-tiny", "device": "cpu", "seed": 17, "writer": writer, "variant": "M1",
        "model": {"d_e": 8, "r": 32, "r_mem": 32, "max_hop": 8},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 96, "n_rel": 4,
                 "seen": [0, 72], "unseen": [72, 96], "chain_lengths": [4, 4], "hops": [1, 2, 3, 4]},
        "train": {"steps": 1, "batch_size": 2, "warmup_steps": 0, "depths": [1, 4], "log_every": 1},
        "eval": {"episodes": 5, "batch_size": 3, "splits": ["val"], "depths": [1, 4],
                 "hops": [1, 2, 3, 4], "counterfactuals": False},
    }
    return resolve_config(recursive_merge(base, updates))


def _model(writer):
    model = SymbolicModel(_config(writer))
    model.requires_grad_(False)
    return model.eval()


def _arrays(model, split="val", episodes=5, seed=11):
    config = model.config
    spec = make_task_spec(config, evaluation=True)
    return stratify_queries(generate(spec, split, 0, episodes, seed), spec, config["eval"]["hops"], seed)


def test_stop_labels_follow_the_margin_rule_and_the_last_step_always_stops():
    losses = torch.tensor([[2.0, 1.0, 1.9, 0.5, 0.45],
                           [1.0, 1.1, 1.2, 1.3, 1.4],
                           [3.0, 2.0, 1.0, 0.0, 0.1]])
    expected = torch.tensor([[False, False, False, True, True],
                             [True, True, True, True, True],
                             [False, False, False, True, True]])
    assert torch.equal(cvh.stop_labels(losses, 0.25), expected)
    # An improvement of exactly the margin counts as an improvement.
    assert torch.equal(cvh.stop_labels(torch.tensor([[1.0, 0.75]]), 0.25), torch.tensor([[False, True]]))
    assert torch.equal(cvh.stop_labels(torch.tensor([[1.0, 0.76]]), 0.25), torch.tensor([[True, True]]))
    with pytest.raises(ValueError):
        cvh.stop_labels(losses[0], 0.25)


def test_policy_steps_match_the_core_halting_rule_and_fallback_executes_all_updates():
    generator = np.random.default_rng(0)
    probabilities = generator.random((5, 4, 16)).astype(np.float32)
    probabilities[0, 0] = 0.1
    probabilities[0, 1, 5] = 0.97
    for theta in (0.05, 0.3, 0.5, 0.9, 0.97, 1.0):
        selected, executed = cvh.policy_steps(probabilities, theta)
        expected = learned_halting_steps(torch.as_tensor(probabilities.reshape(-1, 16)), theta)
        assert np.array_equal(selected.reshape(-1), expected.numpy())
        found = (probabilities >= theta).any(axis=-1)
        assert np.array_equal(executed, np.where(found, selected, 16))
        assert selected.min() >= 1 and executed.max() <= 16
    selected, executed = cvh.policy_steps(np.full((1, 1, 16), 0.2, dtype=np.float32), 0.5)
    assert selected.item() == 1 and executed.item() == 16
    with pytest.raises(ValueError):
        cvh.policy_steps(probabilities[..., :15], 0.5)
    with pytest.raises(ValueError):
        cvh.policy_steps(probabilities + 1, 0.5)


def _synthetic_trace(hop1_peak_step):
    """Two queries per episode: hop 1 (correct only at step 1) and hop 2 (correct from step 2)."""
    episodes = 3
    answer = np.tile(np.array([5, 6]), (episodes, 1))
    hop = np.tile(np.array([1, 2]), (episodes, 1))
    predictions = np.zeros((episodes, 2, 17), dtype=np.int64)
    predictions[:, 0, 1] = 5
    predictions[:, 1, 2:] = 6
    probabilities = np.zeros((episodes, 2, 16), dtype=np.float32)
    probabilities[:, 0, hop1_peak_step - 1] = 0.9
    probabilities[:, 1, 0] = 0.1
    probabilities[:, 1, 1] = 0.8
    return dict(predictions=predictions, answer=answer, hop=hop, p=probabilities)


def test_threshold_table_counts_selected_and_executed_steps_and_hop1_delta():
    traces = {17: _synthetic_trace(1), 29: _synthetic_trace(1)}
    table = cvh.threshold_table(traces, "p", thetas=[0.05, 0.5, 0.95])
    assert [row["theta"] for row in table] == [0.05, 0.5, 0.95]
    expected = [(0.5, 0.0, 1.0, 1.0), (1.0, 0.0, 1.5, 1.5), (1.0, 0.0, 1.5, 16.0)]
    for row, (accuracy, delta, steps, executed) in zip(table, expected):
        assert row["validation_accuracy"] == pytest.approx(accuracy)
        assert row["hop1_delta"] == pytest.approx(delta)
        assert row["validation_mean_steps"] == pytest.approx(steps)
        assert row["validation_mean_executed_steps"] == pytest.approx(executed)
    # Ties on accuracy prefer fewer chosen steps, then the smaller theta.
    assert cvh.select_theta(table)["theta"] == 0.5
    assert cvh.select_theta(table, tolerance=None)["theta"] == 0.5


def test_constrained_selection_rejects_thresholds_that_lose_hop_one_and_can_be_infeasible():
    broken = {17: _synthetic_trace(4), 29: _synthetic_trace(4)}
    table = cvh.threshold_table(broken, "p", thetas=[0.05, 0.5, 0.95])
    # The hop-1 peak sits at step 4 (wrong for hop 1), so every theta loses hop 1.
    assert [round(row["hop1_delta"], 6) for row in table] == [-1.0, -1.0, -1.0]
    assert cvh.select_theta(table) is None
    assert cvh.select_theta(table, tolerance=None) is not None
    rows = [dict(theta=0.2, validation_accuracy=0.60, hop1_delta=-0.2, validation_mean_steps=3.0,
                 validation_mean_executed_steps=5.0),
            dict(theta=0.4, validation_accuracy=0.50, hop1_delta=-0.0099, validation_mean_steps=4.0,
                 validation_mean_executed_steps=5.0),
            dict(theta=0.6, validation_accuracy=0.50, hop1_delta=-0.0101, validation_mean_steps=2.0,
                 validation_mean_executed_steps=5.0),
            dict(theta=0.8, validation_accuracy=0.50, hop1_delta=0.0, validation_mean_steps=4.0,
                 validation_mean_executed_steps=5.0)]
    assert cvh.select_theta(rows)["theta"] == 0.4
    assert cvh.select_theta(rows, tolerance=None)["theta"] == 0.2
    assert cvh.select_theta([]) is None


def _features(episodes=11, seed=5):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(episodes, 8, 16, 51, generator=generator)
    labels = torch.randn(episodes, 8, 16, generator=generator) > 0
    return features, labels


def test_fit_head_reproduces_the_p4s_l_fit_only_under_its_seed_label(monkeypatch):
    monkeypatch.setattr(p4l.torch.cuda, "synchronize", lambda: None)
    features, targets = _features()
    run = dict(name="synthetic", seed=29, family="oracle_random/r32/M1")
    design = dict(training=dict(lr=0.001, batch_episodes=256))
    reference, record = p4l.fit_head(features, targets, run, design, steps=3)
    same, same_record = cvh.fit_head(features, targets, seed=29, family=run["family"], label="p4s-l", steps=3)
    other, other_record = cvh.fit_head(features, targets, seed=29, family=run["family"], label="cvh", steps=3)
    for name, value in reference.state_dict().items():
        assert torch.equal(value, same.state_dict()[name]), name
    assert same_record["logs"] == record["logs"]
    assert same_record["init_seed"] == record["init_seed"] and same_record["sampler_seed"] == record["sampler_seed"]
    assert other_record["init_seed"] != record["init_seed"]
    assert any(not torch.equal(value, other.state_dict()[name]) for name, value in reference.state_dict().items())
    assert same_record["parameters"] == other_record["parameters"] == 3393
    assert not same.training


@pytest.mark.parametrize("writer", ["oracle_random", "joint_fact"])
def test_reader_batch_equals_the_p4s_reader_batch(writer):
    model = _model(writer)
    prepared = model.prepare(_arrays(model))
    ours, theirs = cvh.reader_batch(prepared), p4s.reader_batch(prepared)
    for name in ("u", "memory", "codes", "candidate_ids"):
        assert torch.equal(getattr(ours, name), getattr(theirs, name)), name


@pytest.mark.parametrize("writer", ["oracle_random", "joint_fact"])
def test_step_losses_equal_the_candidate_cross_entropy_of_the_model_read(writer):
    model = _model(writer)
    arrays = _arrays(model)
    prepared = model.prepare(arrays)
    batch = cvh.reader_batch(prepared)
    trace = trace_frozen_reader(model.adapter, batch)
    answer = prepared.answer.reshape(-1)
    losses = cvh.step_losses(model.adapter, batch, trace.states, answer)
    assert losses.shape == (answer.numel(), 16)
    for depth in (1, 4, 16):
        logits = model.read(prepared, depth)["logits"]
        logits = logits.masked_fill(~prepared.candidate_mask[:, None, :], -torch.inf)
        expected = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer, reduction="none")
        torch.testing.assert_close(losses[:, depth - 1], expected, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(losses).all() and (losses >= 0).all()
    wrong = answer.clone()
    wrong[0] = -5
    with pytest.raises(ValueError):
        cvh.step_losses(model.adapter, batch, trace.states, wrong)


@pytest.mark.parametrize("theta", [0.3, 0.5, 0.7])
def test_streaming_halting_matches_the_policy_steps_and_counters(theta):
    model = _model("joint_fact")
    prepared = model.prepare(_arrays(model))
    batch = cvh.reader_batch(prepared)
    head = HaltingHead(seed=3).eval().requires_grad_(False)
    probabilities = head(trace_frozen_reader(model.adapter, batch).features)
    selected, executed = cvh.policy_steps(probabilities.numpy()[None], theta)
    streamed = learned_adaptive_read(model.adapter, batch, head, theta)
    assert np.array_equal(streamed.steps.numpy(), selected[0])
    assert np.array_equal(streamed.executed_updates.numpy(), executed[0])
    assert np.array_equal(streamed.read_calls.numpy(), executed[0] + 1)


@pytest.mark.parametrize("writer", ["oracle_random", "joint_fact"])
def test_collect_traces_are_deterministic_and_independent_of_the_batch_size(writer):
    model = _model(writer)
    head = HaltingHead(seed=3).eval().requires_grad_(False)
    kwargs = dict(heads={"h": head})
    first = cvh.collect_traces(model, [1, 2, 3, 4], "val", 11, 5, batch_episodes=2, **kwargs)
    second = cvh.collect_traces(model, [1, 2, 3, 4], "val", 11, 5, batch_episodes=5, **kwargs)
    assert first["predictions"].shape == (5, 4, 17) and first["losses"].shape == (5, 4, 16)
    assert first["probabilities"]["h"].shape == (5, 4, 16)
    assert np.array_equal(first["predictions"], second["predictions"])
    np.testing.assert_allclose(first["probabilities"]["h"], second["probabilities"]["h"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(first["losses"], second["losses"], rtol=1e-4, atol=1e-5)
    assert np.array_equal(first["hop"], np.broadcast_to(np.arange(1, 5), (5, 4)))
    assert np.all(first["episode_id"] == first["episode_id"][:, :1]) and len(np.unique(first["episode_id"][:, 0])) == 5
    assert len(first["first"].u) == 2
    # Training mode returns the cached features and both label kinds for the same episodes.
    training = cvh.collect_traces(model, [1, 2, 3, 4], "val", 11, 5, training_margin=cvh.MARGIN, batch_episodes=2)
    assert training["features"].shape == (5, 4, 16, 51)
    assert training["targets"].shape == training["correct"].shape == (5, 4, 16)
    assert training["targets"][..., -1].all()
    assert not training["features"].requires_grad
    at_one = cvh.correct_at(first["predictions"], first["answer"], np.ones((5, 4), dtype=np.int64))
    assert np.array_equal(training["correct"].numpy()[..., 0], at_one)
    assert np.array_equal(at_one, first["predictions"][..., 1] == first["answer"])


def _confirmation_trace(index=0, episodes=100, late_gain=40, hop1_late_correct=True):
    """Three queries (hops 1, 2, 3) per episode; the leading episodes are right at L1 and from step 4 on."""
    hop = np.tile(np.array([1, 2, 3]), (episodes, 1))
    answer = np.full((episodes, 3), 5)
    episode_id = np.tile(np.arange(episodes)[:, None] + 1000, (1, 3))
    predictions = np.zeros((episodes, 3, 17), dtype=np.int64)
    at_one = np.arange(episodes) < 20 + index
    late = np.arange(episodes) < 20 + late_gain + index
    predictions[:, :, 1] = np.where(at_one[:, None], 5, 0)
    predictions[:, :, 4:] = np.where(late[:, None, None], 5, 0)
    if not hop1_late_correct:
        predictions[:, 0, 2:] = 0
    return dict(predictions=predictions, answer=answer, hop=hop, episode_id=episode_id)


def _traces(**options):
    return {seed: _confirmation_trace(index, **options) for index, seed in enumerate((17, 29, 43))}


def _steps(traces, step_by_hop, executed=None):
    out = {}
    for seed, trace in traces.items():
        selected = np.vectorize(step_by_hop.get)(trace["hop"]).astype(np.int64)
        out[seed] = (selected, selected if executed is None else np.full_like(selected, executed))
    return out


def test_arm_endpoints_pass_only_when_gain_hop_one_and_depth_all_hold():
    traces = _traces()
    good = cvh.arm_endpoints(traces, _steps(traces, {1: 1, 2: 4, 3: 4}))
    assert good["reach_gain"]["difference"] == pytest.approx(0.40, abs=1e-12)
    assert good["reach_gain"]["ci_low"] > 0 and good["reach_gain"]["confidence"] == pytest.approx(1 - 0.05 / 6)
    assert good["hop1_delta"]["difference"] == pytest.approx(0.0, abs=1e-12)
    assert good["reach_vs_l4"]["difference"] == pytest.approx(0.0, abs=1e-12)
    assert good["reach_vs_l16"]["difference"] == pytest.approx(0.0, abs=1e-12)
    assert good["mean_selected"] == pytest.approx(3.0) and good["mean_executed"] == pytest.approx(3.0)
    assert good["passes_gain"] and good["passes_hop1"] and good["passes_depth"] and good["passes"]
    flat = cvh.arm_endpoints(traces, _steps(traces, {1: 1, 2: 1, 3: 1}))
    assert flat["reach_gain"]["difference"] == pytest.approx(0.0, abs=1e-12)
    assert not flat["passes_gain"] and not flat["passes"] and flat["passes_hop1"] and flat["passes_depth"]
    deep = cvh.arm_endpoints(traces, _steps(traces, {1: 1, 2: 4, 3: 4}, executed=16))
    assert deep["mean_executed"] == pytest.approx(16.0) and deep["mean_selected"] == pytest.approx(3.0)
    assert not deep["passes_depth"] and not deep["passes"] and deep["passes_gain"]
    small = _traces(late_gain=2)  # +2 pp on every hop: below the registered 3 pp
    low = cvh.arm_endpoints(small, _steps(small, {1: 1, 2: 4, 3: 4}))
    assert low["reach_gain"]["difference"] == pytest.approx(0.02, abs=1e-12) and not low["passes_gain"]
    just = _traces(late_gain=4)
    assert cvh.arm_endpoints(just, _steps(just, {1: 1, 2: 4, 3: 4}))["reach_gain"]["difference"] == pytest.approx(0.04)


def test_arm_endpoints_detect_a_hop_one_loss_and_ignore_gains_on_other_hops():
    traces = _traces(hop1_late_correct=False)
    lossy = cvh.arm_endpoints(traces, _steps(traces, {1: 4, 2: 4, 3: 4}))
    assert lossy["hop1_delta"]["difference"] == pytest.approx(-0.21, abs=1e-12)  # L1 accuracy 0.20, 0.21, 0.22
    assert not lossy["passes_hop1"] and lossy["passes_gain"] and not lossy["passes"]
    kept = cvh.arm_endpoints(traces, _steps(traces, {1: 1, 2: 4, 3: 4}))
    assert kept["passes_hop1"] and kept["passes"]
    # Only hops 2-3 enter the gain: moving hop 1 to step 4 leaves reach_gain unchanged.
    assert lossy["reach_gain"]["difference"] == pytest.approx(kept["reach_gain"]["difference"], abs=1e-12)


def test_arm_endpoints_and_accuracy_rows_reject_misaligned_seeds_and_report_exact_accuracies():
    traces = _traces()
    broken = {seed: dict(trace) for seed, trace in traces.items()}
    broken[29]["episode_id"] = broken[29]["episode_id"] + 1
    with pytest.raises(ValueError, match="seed 29 disagree on episode_id"):
        cvh.arm_endpoints(broken, _steps(broken, {1: 1, 2: 4, 3: 4}))
    steps = _steps(traces, {1: 1, 2: 4, 3: 4})
    methods = {"L1": cvh.fixed_correct(traces, 1), "L4": cvh.fixed_correct(traces, 4),
               "arm": cvh.arm_correct(traces, {seed: value[0] for seed, value in steps.items()})}
    rows = {(r["method"], r["hop_group"]): r for r in cvh.accuracy_rows(traces, methods)}
    assert set(g for _, g in rows) == {"hop1", "reach", "all"}  # groups without queries are skipped
    assert rows["L1", "all"]["accuracy"] == pytest.approx(0.21, abs=1e-12)
    assert rows["L4", "reach"]["accuracy"] == pytest.approx(0.61, abs=1e-12)
    assert rows["arm", "hop1"]["accuracy"] == pytest.approx(0.21, abs=1e-12)
    assert rows["arm", "reach"]["accuracy"] == pytest.approx(0.61, abs=1e-12)
    assert all(r["ci_low"] <= r["accuracy"] <= r["ci_high"] for r in rows.values())
