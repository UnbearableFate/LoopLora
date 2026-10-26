"""A fixed, active writer/reader instance isolates bf16 memory error."""

import pytest
import torch

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed


@pytest.mark.parametrize("method", ["process_sequence", "process_sequence_fast"])
def test_bf16_state_against_fp32_reference(method):
    torch.manual_seed(derive_seed(53, "bf16-state-fixed-instance"))
    cfg = AdapterConfig(d_in=12, d_out=10, r=6, r_mem=5, rho=0.2, lambda_init=0.97, beta_init=0.15)
    adapter = make_adapter(cfg).float()
    with torch.no_grad():
        for parameter in (adapter.A, adapter.B, adapter.R, adapter.U, adapter.Q, adapter.K, adapter.V):
            parameter.uniform_(0.02, 0.15)
    x = 0.1 + 0.9 * torch.rand(2, 64, 12, dtype=torch.float32)
    initial = 0.02 * torch.randn(2, 5, 5, dtype=torch.float32)
    run = getattr(adapter, method)
    with torch.no_grad():
        reference, reference_state, reference_stats = run(x, initial.clone(), loop_steps=3)
        actual, actual_state, actual_stats = run(x, initial.to(torch.bfloat16), loop_steps=3)
        no_reads, _, _ = run(x, initial.clone(), loop_steps=0)
        empty_memory, _, _ = run(
            x, torch.zeros_like(initial), loop_steps=3,
            write=torch.zeros(x.shape[:2], dtype=torch.bool),
        )
    assert reference_state.dtype == torch.float32
    assert actual_state.dtype == torch.bfloat16
    assert actual.dtype == reference.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
    assert (reference_state - initial).abs().max().item() > 0.01
    assert (reference[:, 1:] - no_reads[:, 1:]).abs().max().item() > 1e-4
    assert (reference[:, 1:] - empty_memory[:, 1:]).abs().max().item() > 1e-4
    assert reference.abs().min().item() > 1e-3
    for name in reference_stats:
        torch.testing.assert_close(actual_stats[name], reference_stats[name])
    torch.testing.assert_close(actual_stats["writes"], torch.full((2,), 64, dtype=torch.int64))
    torch.testing.assert_close(actual_stats["reads"], torch.full((2,), 192, dtype=torch.int64))

    difference = (actual - reference).abs()
    max_abs = difference.max().item()
    max_rel = (difference / reference.abs().clamp_min(1e-8)).max().item()
    state_abs = (actual_state.float() - reference_state).abs().max().item()
    print(f"{method}: bf16-state output max_abs={max_abs:.6g}, max_rel={max_rel:.6g}, state max_abs={state_abs:.6g}")
    # bf16's unit roundoff is ~0.004. These 64 damped delta updates and three
    # contractive reader steps permit accumulated rounding; 5% relative and
    # 0.02 absolute are deliberately loose regression bounds for this fixed
    # bounded instance, rather than a claim about arbitrary learned dynamics.
    assert max_abs < 0.02
    assert max_rel < 0.05


def test_bf16_base_with_fp32_adapter_preserves_dtype_and_gradients():
    from copy import deepcopy
    from torch import nn

    from looplora.core.wrapper import AdaptedLinear, MemoryContext

    torch.manual_seed(derive_seed(53, "bf16-base-fp32-adapter"))
    adapter = make_adapter(AdapterConfig(d_in=4, d_out=4, r=3, r_mem=2, gate="input")).float()
    with torch.no_grad():
        # Activate every reader path and let the recurrent writer affect the loss.
        for name, parameter in adapter.named_parameters():
            if parameter.ndim == 2:
                parameter.uniform_(0.05, 0.3)
    base = nn.Linear(4, 4, dtype=torch.bfloat16)
    initial = 0.1 * torch.rand(2, 2, 2)
    reference_context = MemoryContext(state=initial.clone(), loop_steps=2)
    reference = AdaptedLinear(deepcopy(base).float(), deepcopy(adapter), reference_context)
    context = MemoryContext(state=initial.clone(), loop_steps=2)
    mixed = AdaptedLinear(base, adapter, context)
    x = (0.2 + torch.rand(2, 4, 4)).to(torch.bfloat16).requires_grad_()
    expected = reference(x.float())
    actual = mixed(x)
    assert actual.dtype == torch.bfloat16
    assert context.state.dtype == torch.float32
    torch.testing.assert_close(context.state, reference_context.state, rtol=0, atol=0)
    # Only the base matmul, branch cast, and final addition round to bf16.
    # Two percent plus 0.01 absolute allows those roundings near cancellation.
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
    actual.float().square().sum().backward()
    for name, parameter in adapter.named_parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad).item() > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in base.parameters())

    # An omitted state starts in adapter precision, including the write-only path.
    for mode in ("online", "write_only"):
        fresh_context = MemoryContext(loop_steps=2, mode=mode)
        mixed.set_context(fresh_context)
        assert mixed(x).dtype == torch.bfloat16
        assert fresh_context.state.dtype == torch.float32
