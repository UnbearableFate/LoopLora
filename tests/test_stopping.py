"""Label-free reader stopping, checked against independent FP64 mathematics.

Run on a Miyabi compute node. The chain example uses an ideal construction;
convergence and prediction stability do not imply correctness in general.
"""

from dataclasses import fields
import inspect

import pytest
import torch

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.stopping import (
    AdaptiveResult, ReaderBatch, ReaderTrace, adaptive_read, fixed_read,
    halting_steps, predict, trace_reader,
)
from looplora.infra.seeds import derive_seed


DTYPE = torch.float64
EPS = 1e-12


def _random_case(activation, r_mem):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(17, "WP-C3", "stopping-tests", activation, r_mem))
        adapter = make_adapter(AdapterConfig(
            d_in=4, d_out=5, r=3, r_mem=r_mem, variant="iterq",
            activation=activation, alpha=.7, rho=.43,
        )).double()
        with torch.no_grad():
            adapter.B.normal_(std=.6)
            adapter.R.normal_(std=.25)
            adapter.U.normal_(std=.4)
            adapter.Q.normal_(std=.5)
        batch = ReaderBatch(
            u=torch.randn(4, 3, dtype=DTYPE),
            memory=torch.randn(4, r_mem, r_mem, dtype=DTYPE) * .6,
            codes=torch.randn(4, 4, 5, dtype=DTYPE),
            candidate_ids=torch.tensor([[31, 7, -1, -1], [9, 80, 2, -1],
                                        [21, 6, 55, 4], [101, -1, -1, -1]]),
        )
    return adapter, batch


def _manual_prediction(adapter, batch, z):
    output = adapter.cfg.alpha * (z @ adapter.B.T)
    scores = torch.einsum("ncd,nd->nc", batch.codes, output)
    scores = scores.masked_fill(batch.candidate_ids < 0, -torch.inf)
    return batch.candidate_ids.gather(1, scores.argmax(-1)[:, None]).squeeze(1)


@torch.no_grad()
def _manual_trace(adapter, batch, maximum):
    states, readouts = [batch.u], []
    for step in range(maximum + 1):
        z = states[step]
        q = z @ adapter.Q.T
        memory_read = torch.matmul(batch.memory, q.unsqueeze(-1)).squeeze(-1)
        readout = memory_read @ adapter.U.T
        readouts.append(readout)
        if step < maximum:
            preactivation = z @ adapter.R.T + readout
            activated = torch.relu(preactivation) if adapter.cfg.activation == "relu" else preactivation
            states.append(batch.u + adapter.cfg.rho * activated)
    norms = torch.stack([value.norm(dim=-1) for value in readouts], dim=1)
    read_norm = norms / norms[:, :1].clamp_min(EPS)
    change = torch.full_like(read_norm, torch.nan)
    for step in range(1, maximum + 1):
        change[:, step] = ((states[step] - states[step - 1]).norm(dim=-1)
                          / states[step - 1].norm(dim=-1).clamp_min(EPS))
    predictions = torch.stack([_manual_prediction(adapter, batch, z) for z in states], dim=1)
    return states, ReaderTrace(predictions, read_norm, change)


def _scalar_case(memories):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(17, "WP-C3", "scalar-stopping-tests"))
        adapter = make_adapter(AdapterConfig(
            d_in=1, d_out=1, r=1, r_mem=1, variant="iterq",
            activation="identity", rho=1.,
        )).double()
    with torch.no_grad():
        adapter.R.zero_()
        adapter.Q.fill_(1)
        adapter.U.fill_(1)
        adapter.B.fill_(1)
    count = len(memories)
    batch = ReaderBatch(
        u=torch.ones(count, 1, dtype=DTYPE),
        memory=torch.tensor(memories, dtype=DTYPE).reshape(count, 1, 1),
        codes=torch.tensor([[[1.], [-1.], [1e6]]], dtype=DTYPE).repeat(count, 1, 1),
        candidate_ids=torch.tensor([[71 + 10 * i, 72 + 10 * i, -1] for i in range(count)]),
    )
    return adapter, batch


@pytest.mark.parametrize("criterion", ["read_norm", "relative_change"])
def test_norm_criteria_use_first_inclusive_hit_after_at_least_one_update(criterion):
    metric = torch.tensor([[0., .9, .2, .1, .8, 0.],
                           [0., .2, .9, .1, .8, 0.],
                           [0., .9, .8, .7, .6, .5]], dtype=DTYPE)
    trace = ReaderTrace(torch.zeros(3, 6, dtype=torch.long), metric, metric.clone())
    actual = halting_steps(trace, criterion, .2)
    assert actual.dtype == torch.int64
    assert actual.tolist() == [2, 1, 5]


@pytest.mark.parametrize("patience,expected", [(1, [1, 1, 5]), (2, [4, 2, 5]), (8, [5, 5, 5])])
def test_stability_requires_consecutive_unchanged_pairs_and_resets(patience, expected):
    predictions = torch.tensor([[4, 4, 5, 5, 5, 8], [6, 6, 6, 5, 6, 6], [1, 2, 1, 2, 1, 2]])
    unused = torch.full(predictions.shape, torch.nan, dtype=DTYPE)
    trace = ReaderTrace(predictions, unused, unused.clone())
    assert halting_steps(trace, "output_stability", patience).tolist() == expected


@pytest.mark.parametrize("activation", ["relu", "identity"])
@pytest.mark.parametrize("r_mem", [2, 5])
def test_trace_metrics_and_predictions_match_independent_matmul(activation, r_mem):
    adapter, batch = _random_case(activation, r_mem)
    _, expected = _manual_trace(adapter, batch, 6)
    actual = trace_reader(adapter, batch, max_steps=6)
    torch.testing.assert_close(actual.predictions, expected.predictions)
    torch.testing.assert_close(actual.read_norm, expected.read_norm, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(actual.relative_change, expected.relative_change,
                               rtol=1e-12, atol=1e-13, equal_nan=True)
    assert bool(torch.isnan(actual.relative_change[:, 0]).all())


@pytest.mark.parametrize("activation", ["relu", "identity"])
@pytest.mark.parametrize("r_mem", [2, 5])
@pytest.mark.parametrize("criterion,threshold", [("read_norm", .8), ("relative_change", .05),
                                                ("output_stability", 2)])
def test_each_adaptive_prediction_and_state_equals_existing_fixed_reader(
        activation, r_mem, criterion, threshold):
    adapter, batch = _random_case(activation, r_mem)
    trace = trace_reader(adapter, batch, max_steps=7)
    selected = halting_steps(trace, criterion, threshold)
    result = adaptive_read(adapter, batch, criterion, threshold, max_steps=7)
    torch.testing.assert_close(result.steps, selected)
    expected_predictions = trace.predictions.gather(1, selected[:, None]).squeeze(1)
    torch.testing.assert_close(result.predictions, expected_predictions)
    torch.testing.assert_close(result.read_calls, selected + int(criterion == "read_norm"))
    assert result.steps.dtype == result.read_calls.dtype == torch.int64
    assert not result.states.requires_grad
    for row, depth in enumerate(selected.tolist()):
        single = batch.take(torch.tensor([row]))
        expected_z = adapter.read(single.u, single.memory, depth)
        torch.testing.assert_close(result.states[row:row + 1], expected_z, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(result.predictions[row:row + 1], fixed_read(adapter, single, depth))


@pytest.mark.parametrize("threshold,expected_steps", [(.55, 1), (.4, 4)])
def test_read_norm_looks_ahead_but_returns_the_current_state_and_caches_reads(
        monkeypatch, threshold, expected_steps):
    adapter, batch = _scalar_case([-.5])
    calls = []
    original = adapter._query

    def counted_query(z, memory, q, u):
        calls.append(z.detach().clone())
        return original(z, memory, q, u)

    monkeypatch.setattr(adapter, "_query", counted_query)
    result = adaptive_read(adapter, batch, "read_norm", threshold, max_steps=4)
    assert result.steps.item() == expected_steps
    assert result.read_calls.item() == expected_steps + 1
    assert len(calls) == expected_steps + 1
    # Diagnostic read at z^1=.5 must not accidentally advance to z^2=.75.
    expected_z = torch.tensor([[.5 if expected_steps == 1 else .6875]], dtype=DTYPE)
    torch.testing.assert_close(result.states, expected_z, rtol=0, atol=0)
    torch.testing.assert_close(calls[-1], expected_z, rtol=0, atol=0)


@pytest.mark.parametrize("criterion,threshold,memories,expected", [
    ("read_norm", .55, [0., .5, -.5], [1, 4, 1]),
    ("relative_change", .2, [0., -1., -.5], [1, 4, 3]),
    ("output_stability", 2, [0., -2., -.5], [2, 4, 2]),
])
def test_compaction_freezes_halted_queries_and_preserves_each_querys_inputs(
        monkeypatch, criterion, threshold, memories, expected):
    adapter, batch = _scalar_case(memories)
    counts = []
    original = adapter._query

    def counted_query(z, memory, q, u):
        counts.append(z.shape[0])
        return original(z, memory, q, u)

    monkeypatch.setattr(adapter, "_query", counted_query)
    result = adaptive_read(adapter, batch, criterion, threshold, max_steps=4)
    assert result.steps.tolist() == expected
    assert sum(counts) == result.read_calls.sum().item()
    assert counts[0] == 3 and min(counts) < 3
    for row, depth in enumerate(expected):
        single = batch.take(torch.tensor([row]))
        alone = adaptive_read(adapter, single, criterion, threshold, max_steps=4)
        fixed_z = adapter.read(single.u, single.memory, depth)
        torch.testing.assert_close(result.states[row:row + 1], fixed_z, rtol=0, atol=0)
        torch.testing.assert_close(result.states[row:row + 1], alone.states, rtol=0, atol=0)
        torch.testing.assert_close(result.predictions[row:row + 1], alone.predictions)
        # Distinct global IDs per row expose accidental candidate-row mixing.
        assert result.predictions[row].item() in single.candidate_ids[0, :2].tolist()


def test_zero_norm_denominators_use_the_declared_epsilon():
    adapter, batch = _scalar_case([-1.])
    trace = trace_reader(adapter, batch, max_steps=2)
    torch.testing.assert_close(trace.relative_change[:, 1:], torch.tensor([[1., 1e12]], dtype=DTYPE))
    batch.u.zero_()
    zero_trace = trace_reader(adapter, batch, max_steps=3)
    torch.testing.assert_close(zero_trace.read_norm, torch.zeros(1, 4, dtype=DTYPE))
    torch.testing.assert_close(zero_trace.relative_change[:, 1:], torch.zeros(1, 3, dtype=DTYPE))
    for criterion in ("read_norm", "relative_change"):
        result = adaptive_read(adapter, batch, criterion, 0., max_steps=3)
        assert result.steps.tolist() == [1]
        assert bool(torch.isfinite(result.states).all())


def test_zero_initial_read_with_nonzero_later_read_has_finite_epsilon_ratio():
    adapter, _ = _random_case("identity", 1)
    with torch.no_grad():
        adapter.R.zero_()
        adapter.R[1, 0] = 1 / adapter.cfg.rho
        adapter.U.zero_()
        adapter.U[0, 0] = 1
        adapter.Q.zero_()
        adapter.Q[0, 1] = 1
    batch = ReaderBatch(
        torch.tensor([[1., 0., 0.]], dtype=DTYPE), torch.ones(1, 1, 1, dtype=DTYPE),
        torch.ones(1, 1, 5, dtype=DTYPE), torch.tensor([[42]]),
    )
    trace = trace_reader(adapter, batch, max_steps=1)
    torch.testing.assert_close(trace.read_norm, torch.tensor([[0., 1e12]], dtype=DTYPE))
    assert adaptive_read(adapter, batch, "read_norm", 1., max_steps=1).read_calls.tolist() == [2]


@pytest.mark.parametrize("depth", [0, 1, 4])
def test_fixed_read_preserves_existing_reader_and_candidate_decoder(depth):
    adapter, batch = _random_case("identity", 2)
    z = adapter.read(batch.u, batch.memory, depth)
    expected = _manual_prediction(adapter, batch, z)
    torch.testing.assert_close(fixed_read(adapter, batch, depth), expected)
    torch.testing.assert_close(predict(adapter, batch, z), expected)


def test_padding_cannot_win_even_with_largest_logit_and_ids_are_global():
    adapter, batch = _scalar_case([0., 0.])
    batch.u[1, 0] = -1
    batch.codes[1, 2, 0] = -1e6
    expected = torch.tensor([71, 82])
    torch.testing.assert_close(predict(adapter, batch, batch.u), expected)
    torch.testing.assert_close(fixed_read(adapter, batch, 0), expected)
    torch.testing.assert_close(trace_reader(adapter, batch, 1).predictions,
                               expected[:, None].expand(-1, 2))
    for criterion, threshold in (("read_norm", 0.), ("relative_change", 0.), ("output_stability", 1)):
        torch.testing.assert_close(adaptive_read(adapter, batch, criterion, threshold, 1).predictions,
                                   expected)


def test_reader_operations_do_not_mutate_inputs_or_adapter_parameters():
    adapter, batch = _random_case("relu", 2)
    before_batch = {field.name: getattr(batch, field.name).clone() for field in fields(batch)}
    before_parameters = {name: value.detach().clone() for name, value in adapter.named_parameters()}
    reordered = batch.take(torch.tensor([3, 0, 2, 0]))
    for field in fields(batch):
        torch.testing.assert_close(getattr(reordered, field.name),
                                   before_batch[field.name][[3, 0, 2, 0]], rtol=0, atol=0)
    fixed_read(adapter, batch, 3)
    predict(adapter, batch, batch.u)
    trace_reader(adapter, batch, 4)
    for criterion, threshold in (("read_norm", .8), ("relative_change", .05), ("output_stability", 2)):
        adaptive_read(adapter, batch, criterion, threshold, 4)
    for name, expected in before_batch.items():
        torch.testing.assert_close(getattr(batch, name), expected, rtol=0, atol=0)
    for name, value in adapter.named_parameters():
        torch.testing.assert_close(value, before_parameters[name], rtol=0, atol=0)
        assert value.grad is None


@pytest.mark.parametrize("criterion,threshold", [("read_norm", -1.), ("relative_change", float("nan")),
                                                ("read_norm", float("inf")), ("output_stability", 0),
                                                ("output_stability", 1.5), ("read_norm", True),
                                                ("unknown", .1)])
def test_invalid_criteria_and_thresholds_are_rejected(criterion, threshold):
    adapter, batch = _scalar_case([0.])
    trace = trace_reader(adapter, batch, 2)
    with pytest.raises(ValueError):
        halting_steps(trace, criterion, threshold)
    with pytest.raises(ValueError):
        adaptive_read(adapter, batch, criterion, threshold, 2)


@pytest.mark.parametrize("variant", ["fixedq", "loop", "lora"])
def test_stopping_rejects_other_reader_variants(variant):
    _, batch = _scalar_case([0.])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(17, "WP-C3", "unsupported-reader", variant))
        adapter = make_adapter(AdapterConfig(d_in=1, d_out=1, r=1, variant=variant)).double()
    with pytest.raises(ValueError):
        trace_reader(adapter, batch, 2)
    with pytest.raises(ValueError):
        adaptive_read(adapter, batch, "read_norm", 0., 2)


def test_runtime_api_contains_no_ground_truth_or_hop_inputs():
    forbidden = {"answer", "answers", "label", "labels", "target", "targets", "hop", "hops"}
    for cls in (ReaderBatch, ReaderTrace, AdaptiveResult):
        assert not forbidden.intersection(field.name for field in fields(cls))
    for function in (trace_reader, halting_steps, adaptive_read, fixed_read, predict):
        parameters = inspect.signature(function).parameters
        assert not forbidden.intersection(parameters)
        assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


@pytest.mark.parametrize("hop", [1, 2, 4])
@pytest.mark.parametrize("activation", ["identity", "relu"])
def test_constructed_orthogonal_chain_stops_when_terminal_prediction_stabilizes(hop, activation):
    # Proposition 3's ideal gain-two chain, not a guarantee for learned readers.
    size = hop + 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(17, "WP-C3", "ideal-chain", hop, activation))
        adapter = make_adapter(AdapterConfig(d_in=size, d_out=size, r=size,
                                             variant="iterq", activation=activation, rho=.25)).double()
    with torch.no_grad():
        adapter.R.zero_()
        adapter.Q.copy_(torch.eye(size, dtype=DTYPE))
        adapter.U.copy_(8 * torch.eye(size, dtype=DTYPE))
        adapter.B.copy_(torch.eye(size, dtype=DTYPE))
    memory = torch.zeros(1, size, size, dtype=DTYPE)
    for head in range(hop):
        memory[0, head + 1, head] = 1
    ids = torch.arange(size, dtype=torch.long)[None] + 100
    batch = ReaderBatch(torch.eye(size, dtype=DTYPE)[:1], memory,
                        torch.eye(size, dtype=DTYPE)[None], ids)
    trace = trace_reader(adapter, batch, max_steps=hop + 3)
    assert trace.predictions[0, :hop + 1].tolist() == list(range(100, 101 + hop))
    for criterion, threshold in (("relative_change", 0.), ("output_stability", 1)):
        result = adaptive_read(adapter, batch, criterion, threshold, max_steps=hop + 3)
        assert result.steps.item() == hop + 1
        assert result.predictions.item() == 100 + hop
    # UMQ z remains nonzero at this fixed point because z retains earlier nodes.
    assert bool((trace.read_norm > 0).all())
    assert adaptive_read(adapter, batch, "read_norm", 0., hop + 3).steps.item() == hop + 3
