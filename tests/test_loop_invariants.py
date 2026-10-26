"""Reader definitions and wrapper work independent of reader depth."""

import copy

import pytest
import torch
from torch import nn

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.wrapper import AdaptedLinear, MemoryContext
from looplora.infra.seeds import derive_seed


@pytest.mark.parametrize("variant", ["lora", "parallel", "loop", "iterq", "fixedq", "double_ba"])
def test_loop_depth_does_not_repeat_base_or_writes(variant):
    torch.manual_seed(derive_seed(47, "loop-counts", variant))
    adapter = make_adapter(AdapterConfig(d_in=4, d_out=4, r=3, r_mem=2, variant=variant)).double()
    wrapped = AdaptedLinear(nn.Linear(4, 4, dtype=torch.float64), adapter)
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    valid = torch.tensor([[True, True, False, True, True], [True, False, True, True, False]])
    write = torch.tensor([[True, False, True, True, True], [False, True, True, False, True]])
    memory_variant = variant in {"iterq", "fixedq"}
    expected_writes = (valid & write).sum(dim=1) if memory_variant else torch.zeros(2, dtype=torch.int64)
    previous_state = None
    for loops in (0, 1, 2, 4):
        context = MemoryContext(loop_steps=loops, valid=valid, write=write)
        wrapped.set_context(context)
        before_calls, before_tokens = wrapped.base_call_count, wrapped.base_token_count
        wrapped(x)
        assert wrapped.base_call_count - before_calls == 1
        assert wrapped.base_token_count - before_tokens == x.shape[0] * x.shape[1]
        torch.testing.assert_close(context.stats["writes"], expected_writes)
        reads_per_token = loops if variant == "iterq" else int(loops > 0) if variant == "fixedq" else 0
        torch.testing.assert_close(context.stats["reads"], valid.sum(dim=1) * reads_per_token)
        if memory_variant:
            if previous_state is not None:
                torch.testing.assert_close(context.state, previous_state, rtol=0, atol=0)
            previous_state = context.state.detach().clone()


@pytest.mark.parametrize("loops", [0, 1, 2, 4])
def test_unshared_counts_use_exact_configured_depth(loops):
    torch.manual_seed(derive_seed(47, "unshared-counts", loops))
    adapter = make_adapter(AdapterConfig(d_in=3, d_out=4, r=2, variant="unshared", unshared_steps=loops)).double()
    ctx = MemoryContext(loop_steps=loops)
    wrapped = AdaptedLinear(nn.Linear(3, 4, dtype=torch.float64), adapter, ctx)
    wrapped(torch.randn(2, 3, 3, dtype=torch.float64))
    assert wrapped.base_call_count == 1
    assert wrapped.base_token_count == 6
    torch.testing.assert_close(ctx.stats["reads"], torch.full((2,), 3 * loops, dtype=torch.int64))
    torch.testing.assert_close(ctx.stats["writes"], torch.full((2,), 3, dtype=torch.int64))


@pytest.mark.parametrize("variant", ["lora", "loop", "iterq", "fixedq", "unshared"])
def test_zero_reader_steps_are_exactly_lora(variant):
    torch.manual_seed(derive_seed(47, "zero-steps", variant))
    adapter = make_adapter(AdapterConfig(d_in=4, d_out=5, r=3, r_mem=2, variant=variant, alpha=0.7, unshared_steps=0)).double()
    with torch.no_grad():
        adapter.B.normal_()
    x = torch.randn(2, 4, 4, dtype=torch.float64)
    state = torch.randn(2, 2, 2, dtype=torch.float64)
    branch, _, stats = adapter.process_sequence(x, state, loop_steps=0)
    expected = adapter.cfg.alpha * ((x @ adapter.A.T) @ adapter.B.T)
    torch.testing.assert_close(branch, expected, rtol=1e-13, atol=1e-13)
    assert stats["reads"].sum().item() == 0


@pytest.mark.parametrize("loops", [0, 1, 2, 4])
@pytest.mark.parametrize("sequence_shape", [False, True])
def test_fixed_query_matches_cached_query_definition(loops, sequence_shape):
    torch.manual_seed(derive_seed(47, "fixed-query", loops, sequence_shape))
    adapter = make_adapter(AdapterConfig(d_in=4, d_out=5, r=3, r_mem=2, variant="fixedq", rho=0.4)).double()
    shape = (2, 4) if sequence_shape else (2,)
    u = torch.randn(*shape, 3, dtype=torch.float64)
    memory = torch.randn(*shape, 2, 2, dtype=torch.float64)
    cached_read = (memory @ (u @ adapter.Q.T).unsqueeze(-1)).squeeze(-1)
    z = u
    for _ in range(loops):
        z = u + adapter.cfg.rho * torch.relu(z @ adapter.R.T + cached_read @ adapter.U.T)
    torch.testing.assert_close(adapter.read(u, memory, loops), z, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("loops", [0, 1, 2, 4])
@pytest.mark.parametrize("method", ["process_sequence", "process_sequence_fast"])
def test_tied_unshared_reader_equals_iterative_query(loops, method):
    torch.manual_seed(derive_seed(47, "tied-unshared", loops, method))
    common = dict(d_in=4, d_out=5, r=3, r_mem=2, gate="input")
    tied = make_adapter(AdapterConfig(**common, variant="iterq")).double()
    unshared = make_adapter(AdapterConfig(**common, variant="unshared", unshared_steps=loops)).double()
    with torch.no_grad():
        tied.B.normal_()
        for name in ("A", "B", "K", "V", "a_lambda", "a_beta", "w_lambda", "w_beta"):
            getattr(unshared, name).copy_(getattr(tied, name))
        for name in ("R", "U", "Q"):
            for parameter in getattr(unshared, name):
                parameter.copy_(getattr(tied, name))
    x = torch.randn(2, 4, 4, dtype=torch.float64)
    initial = torch.randn(2, 2, 2, dtype=torch.float64)
    valid = torch.tensor([[True, False, True, True], [True, True, True, False]])
    write = torch.tensor([[True, True, False, True], [False, True, True, True]])
    reset = torch.tensor([[False, False, True, False], [False, True, False, True]])
    args = dict(loop_steps=loops, valid=valid, write=write, reset=reset)
    expected, state, stats = getattr(tied, method)(x, initial, **args)
    actual, actual_state, actual_stats = getattr(unshared, method)(x, initial, **args)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual_state, state, rtol=1e-13, atol=1e-13)
    assert actual_stats.keys() == stats.keys()
    for name in stats:
        torch.testing.assert_close(actual_stats[name], stats[name])


@pytest.mark.parametrize("form", ["skip", "gated"])
def test_double_ba_startup_and_definition(form):
    torch.manual_seed(derive_seed(47, "double-ba", form))
    adapter = make_adapter(AdapterConfig(d_in=3, d_out=3, r=2, variant="double_ba", double_ba_form=form, alpha=0.7, eta=0.4)).double()
    with torch.no_grad():
        adapter.A.uniform_(0.1, 0.4)
        if form == "gated":
            assert torch.count_nonzero(adapter.B).item() > 0
            adapter.B.uniform_(0.1, 0.4)
            assert adapter.g.item() == 0
        else:
            assert torch.count_nonzero(adapter.B).item() == 0
    x = 0.2 + torch.rand(2, 3, 3, dtype=torch.float64)
    output, _, _ = adapter.process_sequence(x, loop_steps=4)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    output.sum().backward()
    starter = "B" if form == "skip" else "g"
    for name, parameter in adapter.named_parameters():
        if name == starter:
            assert parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        else:
            assert parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0, name
    with torch.no_grad():
        adapter.B.uniform_(0.1, 0.4)
        if form == "gated":
            adapter.g.fill_(0.3)
    u = x @ adapter.A.T
    double = (torch.relu(u @ adapter.B.T) @ adapter.A.T) @ adapter.B.T
    expected = adapter.cfg.alpha * (u @ adapter.B.T + adapter.cfg.eta * double) if form == "skip" else adapter.g * adapter.cfg.alpha * double
    actual, _, _ = adapter.process_sequence(x, loop_steps=0)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("variant", ["iterq", "fixedq", "unshared"])
@pytest.mark.parametrize("mode", ["online", "read_only", "write_only"])
def test_wrapper_full_sequence_equals_incremental_and_honors_modes(variant, mode):
    torch.manual_seed(derive_seed(47, "wrapper-incremental", variant, mode))
    adapter = make_adapter(AdapterConfig(d_in=4, d_out=5, r=3, r_mem=2, variant=variant, unshared_steps=2)).double()
    with torch.no_grad():
        adapter.B.normal_()
    base = nn.Linear(4, 5, dtype=torch.float64)
    x = torch.randn(2, 4, 4, dtype=torch.float64)
    initial = torch.randn(2, 2, 2, dtype=torch.float64)
    valid = torch.tensor([[True, False, True, True], [True, True, True, False]])
    write = torch.tensor([[True, True, False, True], [False, True, True, True]])
    reset = torch.tensor([[False, False, True, False], [False, True, False, True]])
    full_context = MemoryContext(state=initial.clone(), loop_steps=2, valid=valid, write=write, reset=reset, mode=mode)
    full = AdaptedLinear(base, adapter, full_context)
    expected = full(x)
    incremental_context = MemoryContext(state=initial.clone(), loop_steps=2, mode=mode)
    incremental = AdaptedLinear(copy.deepcopy(base), copy.deepcopy(adapter), incremental_context)
    outputs = []
    counts = {name: torch.zeros(2, dtype=torch.int64) for name in ("reads", "writes")}
    for token in range(x.shape[1]):
        incremental_context.valid = valid[:, token]
        incremental_context.write = write[:, token]
        incremental_context.reset = reset[:, token]
        outputs.append(incremental(x[:, token]))
        for name in counts:
            counts[name] += incremental_context.stats[name]
    torch.testing.assert_close(torch.stack(outputs, dim=1), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(incremental_context.state, full_context.state, rtol=1e-12, atol=1e-12)
    for name in counts:
        torch.testing.assert_close(counts[name], full_context.stats[name])
    assert full.base_call_count == 1
    assert incremental.base_call_count == x.shape[1]
    assert full.base_token_count == incremental.base_token_count == 8
    expected_reads = torch.zeros(2, dtype=torch.int64) if mode == "write_only" else valid.sum(dim=1) * (1 if variant == "fixedq" else 2)
    expected_writes = torch.zeros(2, dtype=torch.int64) if mode == "read_only" else (valid & write).sum(dim=1)
    torch.testing.assert_close(full_context.stats["reads"], expected_reads)
    torch.testing.assert_close(full_context.stats["writes"], expected_writes)
    if mode == "write_only":
        torch.testing.assert_close(expected, base(x), rtol=0, atol=0)
    if mode == "read_only":
        # Each sequence has a reset, so read-only still clears its state.
        torch.testing.assert_close(full_context.state, torch.zeros_like(initial), rtol=0, atol=0)
