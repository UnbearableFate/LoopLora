"""P4-S-L frozen traces and streaming halts; execute in a compute allocation."""
from dataclasses import fields, replace
import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.learned_halting import (
    FEATURE_DIM, HaltingHead, LearnedAdaptiveResult, LearnedReaderTrace,
    learned_adaptive_read, learned_halting_steps, trace_frozen_reader,
)
from looplora.core.stopping import ReaderBatch, fixed_read, trace_reader
from looplora.infra.seeds import derive_seed

DTYPE = torch.float64


def _random_case(activation="identity", count=4):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(derive_seed(29, "P4L", "random-reader", activation, count))
        adapter = make_adapter(AdapterConfig(d_in=3, d_out=5, r=32, r_mem=7,
                                             variant="iterq", activation=activation, rho=.25)).double()
        with torch.no_grad():
            adapter.B.normal_(std=.3)
            adapter.R.normal_(std=.06)
            adapter.U.normal_(std=.1)
            adapter.Q.normal_(std=.1)
        batch = ReaderBatch(torch.randn(count, 32, dtype=DTYPE),
                            torch.randn(count, 7, 7, dtype=DTYPE) * .2,
                            torch.randn(count, 4, 5, dtype=DTYPE),
                            torch.tensor([[100 * i + 7, 100 * i + 31, 100 * i + 62, -1]
                                          for i in range(count)], dtype=torch.long).reshape(count, 4))
    return adapter, batch


def _toy(count=5):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(derive_seed(29, "P4L", "toy-reader"))
        adapter = make_adapter(AdapterConfig(d_in=1, d_out=1, r=32, r_mem=1,
                                             variant="iterq", activation="identity", rho=1.)).double()
    with torch.no_grad():
        adapter.R.zero_()
        adapter.Q.zero_()
        adapter.Q[0, 1] = 1
        adapter.U.zero_()
        adapter.U[1, 0] = 1
        adapter.B.zero_()
        adapter.B[0, 1], adapter.B[0, 2] = 1, -1.6
    u = torch.zeros(count, 32, dtype=DTYPE)
    u[:, 0] = torch.arange(1, count + 1, dtype=DTYPE)
    u[:, 1:3] = 1
    ids = torch.tensor([[100 + 10 * i, 101 + 10 * i, -1] for i in range(count)], dtype=torch.long).reshape(count, 3)
    batch = ReaderBatch(u, torch.full((count, 1, 1), .5, dtype=DTYPE),
                        torch.tensor([[[1.], [-1.], [1e6]]], dtype=DTYPE).repeat(count, 1, 1), ids)
    return adapter, batch


class _ScheduledHead(nn.Module):
    """Controlled probabilities identify queries by a constant state coordinate."""

    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer("probabilities", probabilities)
        self.rows = []

    def forward(self, features):
        rows = features[..., 0].long() - 1
        step = features[..., 35:51].argmax(-1)
        if features.ndim == 2:
            self.rows.append(rows.tolist())
        return self.probabilities[rows, step]


@pytest.mark.parametrize("activation", ["relu", "identity"])
def test_trace_features_states_and_predictions_match_fixed_and_label_free_readers(activation):
    adapter, batch = _random_case(activation)
    trace = trace_frozen_reader(adapter, batch)
    baseline = trace_reader(adapter, batch)
    assert isinstance(trace, LearnedReaderTrace)
    assert trace.states.shape == (4, 17, 32)
    assert trace.features.shape == (4, 16, 51)
    assert not trace.states.requires_grad and not trace.features.requires_grad
    assert not trace.states.is_inference() and not trace.features.is_inference()
    torch.testing.assert_close(trace.predictions, baseline.predictions)
    torch.testing.assert_close(trace.read_norm, baseline.read_norm, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(trace.relative_change, baseline.relative_change,
                               rtol=1e-12, atol=1e-13, equal_nan=True)
    torch.testing.assert_close(trace.features[..., :32], trace.states[:, 1:])
    torch.testing.assert_close(trace.features[..., 32], trace.states[:, 1:].norm(dim=-1))
    change = ((trace.states[:, 1:] - trace.states[:, :-1]).norm(dim=-1)
              / trace.states[:, :-1].norm(dim=-1).clamp_min(1e-12))
    torch.testing.assert_close(trace.features[..., 33], change)
    query = trace.states @ adapter.Q.T
    read = torch.matmul(batch.memory[:, None], query.unsqueeze(-1)).squeeze(-1) @ adapter.U.T
    ratio = read.norm(dim=-1) / read[:, :1].norm(dim=-1).clamp_min(1e-12)
    torch.testing.assert_close(trace.features[..., 34], ratio[:, 1:])
    torch.testing.assert_close(trace.features[..., 35:], torch.eye(16, dtype=DTYPE)[None].expand(4, -1, -1))
    for depth in range(17):
        torch.testing.assert_close(trace.states[:, depth], adapter.read(batch.u, batch.memory, depth))
        torch.testing.assert_close(trace.predictions[:, depth], fixed_read(adapter, batch, depth))


def test_head_has_preregistered_architecture_deterministic_init_and_gradients_only_in_head():
    adapter, batch = _random_case()
    before = torch.random.get_rng_state().clone()
    head = HaltingHead(seed=43).double()
    assert torch.equal(torch.random.get_rng_state(), before)
    same = HaltingHead(seed=43).double()
    for left, right in zip(head.parameters(), same.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(43)
        expected = nn.Sequential(nn.Linear(51, 64), nn.ReLU(), nn.Linear(64, 1)).double()
    for actual, standard in zip(head.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, standard, rtol=0, atol=0)
    assert [(layer.in_features, layer.out_features) for layer in head.network if isinstance(layer, nn.Linear)] == [(51, 64), (64, 1)]
    assert isinstance(head.network[1], nn.ReLU)
    trace = trace_frozen_reader(adapter, batch)
    logits = head.logits(trace.features)
    probabilities = head(trace.features)
    assert logits.shape == probabilities.shape == (4, 16)
    torch.testing.assert_close(probabilities, logits.sigmoid())
    targets = (trace.predictions[:, 1:] == batch.candidate_ids[:, :1]).to(DTYPE)
    F.binary_cross_entropy_with_logits(logits, targets).backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in head.parameters())
    assert all(parameter.grad is None for parameter in adapter.parameters())


def test_first_threshold_hit_is_inclusive_and_fallback_uses_earliest_maximum():
    probabilities = torch.tensor([[.5, .9, .1, .9], [.1, .5, .9, .1],
                                  [.1, .4, .4, .3], [.2, .2, .2, .2]], dtype=DTYPE)
    assert learned_halting_steps(probabilities, .5).tolist() == [1, 2, 2, 1]
    assert learned_halting_steps(probabilities, 0).tolist() == [1, 1, 1, 1]
    assert learned_halting_steps(probabilities, 1).tolist() == [2, 3, 2, 1]


def test_mixed_streaming_compaction_fallback_state_retention_and_real_read_accounting(monkeypatch):
    adapter, batch = _toy()
    table = torch.full((5, 16), .1, dtype=DTYPE)
    table[0, 0], table[0, 8] = .5, .9
    table[1, 3] = .6
    table[2, 1:3] = .4
    table[3, 0], table[3, 15] = .4, .4
    table[4, 15] = .5
    head = _ScheduledHead(table)
    trace = trace_frozen_reader(adapter, batch)
    expected_steps = learned_halting_steps(head(trace.features), .5)
    calls, original = [], adapter._query

    def counted(z, memory, q, u):
        calls.append(z.shape[0])
        return original(z, memory, q, u)

    monkeypatch.setattr(adapter, "_query", counted)
    result = learned_adaptive_read(adapter, batch, head, .5)
    assert isinstance(result, LearnedAdaptiveResult)
    assert expected_steps.tolist() == result.steps.tolist() == [1, 4, 2, 1, 16]
    assert result.executed_updates.tolist() == [1, 4, 16, 16, 16]
    assert result.read_calls.tolist() == [2, 5, 17, 17, 17]
    assert result.threshold_hit.tolist() == [True, True, False, False, True]
    assert sum(calls) == int(result.read_calls.sum())
    assert calls == [5, 5, 4, 4, 4] + [3] * 12
    assert [sum(row in active for active in head.rows) for row in range(5)] == result.executed_updates.tolist()
    rows = torch.arange(5)
    torch.testing.assert_close(result.states, trace.states[rows, expected_steps], rtol=0, atol=0)
    torch.testing.assert_close(result.predictions, trace.predictions[rows, expected_steps])
    assert result.predictions[3] != trace.predictions[3, 16]
    assert not result.states.requires_grad


def test_every_first_hit_depth_including_cap_is_selected_with_exact_read_counts(monkeypatch):
    adapter, batch = _toy(16)
    table = torch.full((16, 16), .1, dtype=DTYPE)
    table[torch.arange(16), torch.arange(16)] = .75
    calls, original = [], adapter._query

    def counted(z, memory, q, u):
        calls.append(z.shape[0])
        return original(z, memory, q, u)

    monkeypatch.setattr(adapter, "_query", counted)
    result = learned_adaptive_read(adapter, batch, _ScheduledHead(table), .75)
    expected = torch.arange(1, 17)
    torch.testing.assert_close(result.steps, expected)
    torch.testing.assert_close(result.executed_updates, expected)
    torch.testing.assert_close(result.read_calls, expected + 1)
    assert bool(result.threshold_hit.all())
    assert calls == [16] + list(range(16, 0, -1))


@pytest.mark.parametrize("activation", ["relu", "identity"])
@pytest.mark.parametrize("theta", [0., .5, 1.])
def test_real_head_streaming_matches_offline_trace_states_and_decisions(activation, theta):
    adapter, batch = _random_case(activation)
    head = HaltingHead(17).double()
    trace = trace_frozen_reader(adapter, batch)
    probabilities = head(trace.features)
    selected = learned_halting_steps(probabilities, theta)
    result = learned_adaptive_read(adapter, batch, head, theta)
    rows = torch.arange(len(batch.u))
    torch.testing.assert_close(result.steps, selected)
    torch.testing.assert_close(result.states, trace.states[rows, selected], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(result.predictions, trace.predictions[rows, selected])
    hit = (probabilities >= theta).any(-1)
    expected_executed = torch.where(hit, selected, 16)
    torch.testing.assert_close(result.executed_updates, expected_executed)
    torch.testing.assert_close(result.read_calls, expected_executed + 1)
    torch.testing.assert_close(result.threshold_hit, hit)


def test_zero_denominators_use_fixed_epsilon_and_do_not_generate_nonfinite_features():
    adapter, batch = _toy(1)
    batch.u.zero_()
    trace = trace_frozen_reader(adapter, batch)
    assert bool(torch.isfinite(trace.features).all())
    torch.testing.assert_close(trace.features[..., :35], torch.zeros(1, 16, 35, dtype=DTYPE))
    with torch.no_grad():
        batch.u[0, 1] = 1
        batch.memory.fill_(-1)
    trace = trace_frozen_reader(adapter, batch, 2)
    torch.testing.assert_close(trace.features[0, :, 33], torch.tensor([1., 1e12], dtype=DTYPE))
    torch.testing.assert_close(trace.features[0, :, 34], torch.tensor([0., 1.], dtype=DTYPE))


def test_zero_initial_read_and_nonzero_post_state_read_use_epsilon_ratio():
    adapter, batch = _toy(1)
    batch.u[:, 1] = 0
    with torch.no_grad():
        adapter.R[1, 2] = 1
    trace = trace_frozen_reader(adapter, batch, 1)
    assert trace.read_norm[0, 0].item() == 0
    torch.testing.assert_close(trace.features[0, 0, 34], torch.tensor(5e11, dtype=DTYPE))
    assert bool(torch.isfinite(trace.features).all())


def test_trace_reads_initial_and_every_post_update_state_once(monkeypatch):
    adapter, batch = _toy(2)
    calls, original = [], adapter._query

    def counted(z, memory, q, u):
        calls.append(z.clone())
        return original(z, memory, q, u)

    monkeypatch.setattr(adapter, "_query", counted)
    trace = trace_frozen_reader(adapter, batch)
    assert len(calls) == 17
    for step, state in enumerate(calls):
        torch.testing.assert_close(state, trace.states[:, step], rtol=0, atol=0)


def test_trace_and_streaming_do_not_change_reader_inputs_parameters_or_modes():
    adapter, batch = _random_case()
    head = HaltingHead(29).double()
    batch.u.requires_grad_(True)
    before_batch = {field.name: getattr(batch, field.name).detach().clone() for field in fields(batch)}
    before_parameters = {name: value.detach().clone() for name, value in adapter.named_parameters()}
    adapter.train()
    head.train()
    trace_frozen_reader(adapter, batch)
    learned_adaptive_read(adapter, batch, head, .9)
    assert adapter.training and head.training
    for name, value in before_batch.items():
        torch.testing.assert_close(getattr(batch, name), value, rtol=0, atol=0)
    for name, value in adapter.named_parameters():
        torch.testing.assert_close(value, before_parameters[name], rtol=0, atol=0)
        assert value.grad is None
    assert all(parameter.grad is None for parameter in head.parameters())


def test_decoder_changes_cannot_change_the_halting_features():
    adapter, batch = _toy(1)
    before = trace_frozen_reader(adapter, batch)
    changed = replace(batch, codes=-batch.codes, candidate_ids=batch.candidate_ids + 500)
    changed.candidate_ids[:, -1] = -1
    after = trace_frozen_reader(adapter, changed)
    torch.testing.assert_close(before.features, after.features, rtol=0, atol=0)
    assert not torch.equal(before.predictions, after.predictions)
    forbidden = {"answer", "answers", "label", "labels", "target", "targets", "hop", "hops"}
    for function in (trace_frozen_reader, learned_halting_steps, learned_adaptive_read, HaltingHead.forward, HaltingHead.logits):
        assert not forbidden.intersection(inspect.signature(function).parameters)
    for cls in (LearnedReaderTrace, LearnedAdaptiveResult):
        assert not forbidden.intersection(field.name for field in fields(cls))


def test_empty_batches_and_short_caps_keep_fixed_feature_layout():
    adapter, batch = _random_case(count=0)
    trace = trace_frozen_reader(adapter, batch, 3)
    assert trace.predictions.shape == (0, 4)
    assert trace.states.shape == (0, 4, 32)
    assert trace.features.shape == (0, 3, FEATURE_DIM)
    assert learned_halting_steps(torch.empty(0, 3), .5).shape == (0,)
    result = learned_adaptive_read(adapter, batch, HaltingHead().double(), .5, 3)
    assert result.states.shape == (0, 32)
    assert result.executed_updates.shape == result.read_calls.shape == (0,)
    adapter, batch = _toy(1)
    head = _ScheduledHead(torch.full((1, 16), .2, dtype=DTYPE))
    result = learned_adaptive_read(adapter, batch, head, .5, 3)
    assert result.steps.tolist() == [1]
    assert result.executed_updates.tolist() == [3]
    assert result.read_calls.tolist() == [4]


@pytest.mark.parametrize("theta", [-.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_thresholds_fail_in_offline_and_streaming(theta):
    adapter, batch = _toy(1)
    with pytest.raises(ValueError, match="theta"):
        learned_halting_steps(torch.ones(1, 16), theta)
    with pytest.raises(ValueError, match="theta"):
        learned_adaptive_read(adapter, batch, HaltingHead().double(), theta)


@pytest.mark.parametrize("maximum", [0, -1, 17, 1.5, True])
def test_invalid_caps_are_rejected(maximum):
    adapter, batch = _toy(1)
    with pytest.raises(ValueError):
        trace_frozen_reader(adapter, batch, maximum)
    with pytest.raises(ValueError):
        learned_adaptive_read(adapter, batch, HaltingHead().double(), .5, maximum)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1, 1.1])
def test_nonfinite_or_out_of_range_probabilities_fail_explicitly(bad):
    adapter, batch = _toy(1)
    table = torch.full((1, 16), bad, dtype=DTYPE)
    error = FloatingPointError if not bool(torch.isfinite(table).all()) else ValueError
    with pytest.raises(error):
        learned_halting_steps(table, .5)
    with pytest.raises(error):
        learned_adaptive_read(adapter, batch, _ScheduledHead(table), .5)


@pytest.mark.parametrize("field", ["u", "memory", "codes"])
def test_nonfinite_reader_data_is_rejected(field):
    adapter, batch = _toy(1)
    getattr(batch, field).reshape(-1)[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        trace_frozen_reader(adapter, batch)
    with pytest.raises(FloatingPointError, match="non-finite"):
        learned_adaptive_read(adapter, batch, HaltingHead().double(), .5)


def test_reader_shape_rank_variant_head_shape_and_overflow_errors_are_explicit():
    adapter, batch = _toy(1)
    for malformed in (replace(batch, u=batch.u[0]), replace(batch, memory=batch.memory[:, :, :0]),
                      replace(batch, candidate_ids=torch.full_like(batch.candidate_ids, -1))):
        with pytest.raises(ValueError):
            trace_frozen_reader(adapter, malformed)
    for rank, variant in ((16, "iterq"), (32, "fixedq")):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(derive_seed(29, "P4L", "invalid-reader", rank, variant))
            other = make_adapter(AdapterConfig(d_in=1, d_out=1, r=rank, r_mem=1, variant=variant)).double()
        bad_batch = replace(batch, u=batch.u[:, :rank])
        with pytest.raises(ValueError):
            trace_frozen_reader(other, bad_batch)
    with pytest.raises(ValueError, match="one probability"):
        learned_adaptive_read(adapter, batch, nn.Identity(), .5)
    with torch.no_grad():
        adapter.U[1, 0] = 1e308
        batch.memory.fill_(1e308)
    with pytest.raises(FloatingPointError, match="non-finite"):
        trace_frozen_reader(adapter, batch)
    with pytest.raises(FloatingPointError, match="non-finite"):
        learned_adaptive_read(adapter, batch, HaltingHead().double(), .5)


def test_head_rejects_nonfinite_or_wrong_size_features_and_nonfinite_weights():
    head = HaltingHead()
    with pytest.raises(ValueError, match="final dimension"):
        head(torch.zeros(2, 50))
    with pytest.raises(FloatingPointError, match="features"):
        head(torch.full((2, 51), float("nan")))
    with torch.no_grad():
        head.network[-1].bias.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="logits"):
        head(torch.zeros(2, 51))
