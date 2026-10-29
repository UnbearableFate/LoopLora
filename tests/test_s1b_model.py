"""S1B real adapter/T2 paths on a tiny backbone; no real parent or sealed data."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import slot_golden
from looplora.core.memory_backend import state_digest
from looplora.llm import s1b_data as d, s1b_rules as r, s1b_model as m, s1b_preflight
from looplora.llm.slot_experiment import adopt, freeze_for_s1a, gradient_report, tensor_digests, support_digest
from looplora.llm.slot_read import bind_batch
from looplora.llm.s1b_train import fit
from looplora.llm.inject import load_tokenizer
from looplora.llm.evaluate import _slice


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def backbone():
    model = slot_golden.tiny_model()
    yield model
    assert not hasattr(model, "looplora_adapters")


@pytest.fixture
def factory(backbone, tokenizer):
    experiments = []
    def make(branch="advance", control="shared"):
        if any(not exp._closed for exp in experiments):
            raise RuntimeError("close previous test adapter before sharing its backbone")
        exp = m.configure(adopt(slot_golden.tiny_experiment(backbone, tokenizer)), branch, control)
        experiments.append(exp)
        return exp
    yield make
    for exp in experiments:
        exp.close()


def raw(count=4):
    return d.raw_episodes(replace(d.SETS["smoke"], count=count))


def evaluation(count=4, hops=None):
    return d.episodes(replace(d.SETS["smoke"], count=count), hops)


@pytest.mark.parametrize("branch", ["advance", "query_copy"])
def test_matrix_l4_logits_and_gradients_replay_native_tensor_loop(factory, branch):
    exp = factory(branch)
    report = s1b_preflight.matrix_identity(exp, raw())
    assert report["passed"] and report["max_abs_difference"] == 0
    assert all(value == 0 for value in report["gradients"].values())
    if branch == "query_copy":
        assert torch.equal(exp.adapter.A_query, exp.adapter.A)


@pytest.mark.parametrize("arm", ["M4", "S4"])
def test_actual_forward_depth_traces_match_and_do_not_change_memory_or_logits(factory, arm):
    exp = factory()
    arrays = evaluation()
    with torch.no_grad():
        batch = exp.prepare(arrays)
        bound = bind_batch(exp, batch, r.read_config(arm))
        before = state_digest(bound.snapshot.states[exp.layer].state)
        for depth in (1, 2, 4, 8):
            exp.adapter.reset_observation()
            plain = exp.score(bound, depth)[0]
            exp.adapter.reset_observation(capture=True)
            captured = exp.score(bound, depth)[0]
            assert torch.equal(plain, captured)
            assert m.verify_trace(exp.adapter)
            assert state_digest(bound.snapshot.states[exp.layer].state) == before
        # Corrupting a captured value must be detected instead of claiming trace agreement.
        exp.adapter.trace[0]["_z"].add_(1.)
        with pytest.raises(RuntimeError, match="trajectory"):
            m.verify_trace(exp.adapter)


def test_multiquery_metrics_use_path_targets_not_the_final_fact_every_loop(factory):
    exp = factory()
    arrays = evaluation(3)
    result = m.evaluate(exp, arrays, r.read_config("S4"), 4, diagnostics=True)
    samples = result["samples"]
    assert samples["correct"].shape == (3, 8)
    assert samples["nll"].shape == (3, 8)
    assert np.isnan(samples["path_error_s2"][:, 0]).all()
    assert np.isfinite(samples["path_error_s2"][:, 1:]).all()
    assert np.isnan(samples["post_completion_drift_s2"][:, 1:]).all()
    assert np.isfinite(samples["post_completion_drift_s2"][:, 0]).all()
    costs = result["costs"][0]
    assert costs["logical_reads"] == 4 * costs["valid_read_positions"]
    assert costs["backend_rows"] == costs["logical_reads"]


def test_same_checkpoint_depths_use_same_graphs_and_no_gold_label_reaches_forward(factory):
    exp = factory()
    arrays = evaluation(2)
    with torch.no_grad():
        batch = exp.prepare(arrays)
        reference = exp.score(bind_batch(exp, batch, r.read_config("S4")), 4)[0]
        changed = replace(arrays, q_hop_true=np.ones_like(arrays.q_hop_true), answer=arrays.answer[:, ::-1].copy())
        # Keep tokenized questions fixed; only offline labels on the prepared batch change.
        alternate = replace(batch, arrays=changed)
        logits = exp.score(bind_batch(exp, alternate, r.read_config("S4")), 4)[0]
        assert torch.equal(reference, logits)


@pytest.mark.parametrize("kind", ["fixed", "untied"])
def test_control_read_counts_and_warm_start(factory, kind):
    exp = factory("query_copy")
    arrays = raw(3)
    with torch.no_grad():
        batch = exp.prepare(arrays)
        bound = bind_batch(exp, batch, r.read_config("S4"))
        shared = exp.score(bound, 4)[0]
        total_before = sum(p.numel() for p in exp.adapter.parameters())
        if kind == "untied":
            exp.adapter.untie()
            assert sum(p.numel() for p in exp.adapter.parameters()) - total_before == 147456
            assert torch.equal(shared, exp.score(bound, 4)[0])
        else:
            exp.adapter.control = "fixed"
        exp.s1b_control = kind
        exp.adapter.reset_observation()
        exp.score(bound, 4)
        cost = exp.adapter.cost
        assert cost["logical_reads"] == cost["valid_read_positions"] * (1 if kind == "fixed" else 4)
        assert cost["recurrent_updates"] == cost["valid_read_positions"] * 4
    freeze_for_s1a(exp, r.trainable("query_copy", kind))
    exp.adapter.zero_grad(set_to_none=True)
    logits = exp.score(bound, 4)[0]
    F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten()).backward()
    assert gradient_report(exp, r.trainable("query_copy", kind))["trainable_grad_norms"]


@pytest.mark.parametrize("branch,arm", [("advance", "M4"), ("query_copy", "S4")])
def test_matched_training_freezes_writer_and_json_export_round_trip(factory, backbone, tokenizer, tmp_path, branch, arm):
    exp = factory(branch)
    support, dev = raw(3), evaluation(3, [1, 2, 3])
    spec = r.read_config(arm)
    frozen = tensor_digests(exp.adapter)
    original_support = support_digest(exp, support)
    science = r.science_config(branch, arm, 17, 1e-3, spec.as_dict())
    # Two deterministic updates exercise the actual trainer without running a scientific schedule.
    record = fit(exp, spec, arm=arm, seed=17, lr=1e-3, steps=2, out=tmp_path, dev=dev, dev_steps=[1, 2],
                 support=support, parent_digest="f" * 32, science_hash="test", source=iter([(support, None)] * 2))
    assert record["all_finite"] and record["freeze"]["support"]["equal"]
    assert support_digest(exp, support) == original_support
    after = tensor_digests(exp.adapter)
    assert all(after[name] == digest for name, digest in frozen.items() if name not in r.trainable(branch))
    assert record["costs"]["logical_reads"] == 4 * record["costs"]["valid_read_positions"]
    exp.close()
    reloaded = m.load_export(tmp_path / "selected.pt", model=backbone, tokenizer=tokenizer)
    try:
        assert reloaded.s1b == json.loads(json.dumps(reloaded.s1b))
        assert reloaded.s1b["trained_depth"] == 4 and reloaded.s1b["task_digest"] == d.LAYOUT_DIGEST
        assert (reloaded.adapter.A_query is not None) == (branch == "query_copy")
        original = m.evaluate(reloaded, dev, spec, 4)["samples"]
    finally:
        reloaded.close()
    again = m.load_export(tmp_path / "selected.pt", model=backbone, tokenizer=tokenizer)
    try:
        actual = m.evaluate(again, dev, spec, 4)["samples"]
        assert all(np.array_equal(actual[name], original[name]) for name in original)
    finally:
        again.close()


def test_export_requires_buffers_and_explicit_task(factory, backbone, tokenizer, tmp_path):
    exp = factory()
    spec = r.read_config("M4")
    meta = m.metadata(exp, "M4", 17, spec, 20, "f" * 32, "test")
    path = tmp_path / "export.pt"
    m.save_export(exp, path, meta)
    exp.close()
    payload = torch.load(path, weights_only=True)
    payload["adapter"].pop("input_std")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incomplete"):
        m.load_export(path, model=backbone, tokenizer=tokenizer)
