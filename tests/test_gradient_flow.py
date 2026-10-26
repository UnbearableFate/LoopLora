"""Gradients through readers, recurrent writes, and the frozen base."""

import pytest
import torch
from torch import nn
from torch.func import functional_call

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.wrapper import AdaptedLinear, MemoryContext
from looplora.infra.seeds import derive_seed


def _positive_internal_parameters(adapter):
    # Keep ReLUs away from zero without suppressing writer/key derivatives.
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if parameter.ndim == 2 and name != "B":
                parameter.uniform_(0.05, 0.3)


def _assert_nonzero_finite_gradient(parameter, name):
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(parameter.grad).item() > 0, name


@pytest.mark.parametrize("gate", ["scalar", "input"])
def test_zero_b_startup_and_all_v2_parameters_after_b_step(gate):
    torch.manual_seed(derive_seed(41, "gradient-startup", gate))
    cfg = AdapterConfig(d_in=3, d_out=4, r=3, r_mem=2, gate=gate)
    adapter = make_adapter(cfg).double()
    _positive_internal_parameters(adapter)
    assert torch.count_nonzero(adapter.B).item() == 0
    base = nn.Linear(3, 4, dtype=torch.float64)
    context = MemoryContext(loop_steps=2)
    wrapped = AdaptedLinear(base, adapter, context)
    x = (0.2 + torch.rand(2, 4, 3, dtype=torch.float64)).requires_grad_()
    weights = 0.3 + torch.rand(2, 4, 4, dtype=torch.float64)

    y = wrapped(x)
    torch.testing.assert_close(y, base(x), rtol=0, atol=0)
    (y * weights).sum().backward()
    _assert_nonzero_finite_gradient(adapter.B, "B")
    for name, parameter in adapter.named_parameters():
        if name != "B":
            assert parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0, name

    # Frozen W must retain the Jacobian to upstream layers.
    expected_input_gradient = weights @ base.weight
    torch.testing.assert_close(x.grad, expected_input_gradient, rtol=1e-13, atol=1e-13)
    assert torch.count_nonzero(x.grad).item() > 0
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())

    torch.optim.SGD([adapter.B], lr=0.05).step()
    adapter.zero_grad(set_to_none=True)
    x.grad = None
    context.state = None
    (wrapped(x) * weights).sum().backward()
    expected = {"A", "B", "R", "U", "Q", "K", "V", "a_lambda", "a_beta"}
    if gate == "input":
        expected |= {"w_lambda", "w_beta"}
    parameters = dict(adapter.named_parameters())
    assert expected <= parameters.keys()
    for name, parameter in parameters.items():
        _assert_nonzero_finite_gradient(parameter, name)
    assert all(p.grad is None for p in base.parameters())


class _SequenceHarness(nn.Module):
    def __init__(self, adapter, method):
        super().__init__()
        self.adapter = adapter
        self.method = method

    def forward(self, x, state):
        output, final, _ = getattr(self.adapter, self.method)(x, state, loop_steps=2)
        return output, final


@pytest.mark.parametrize("method", ["process_sequence", "process_sequence_fast"])
def test_tiny_v2_gradcheck_includes_writer_parameters(method):
    torch.manual_seed(derive_seed(41, "writer-gradcheck", method))
    adapter = make_adapter(AdapterConfig(d_in=2, d_out=2, r=2, r_mem=2, gate="input")).double()
    _positive_internal_parameters(adapter)
    with torch.no_grad():
        adapter.B.uniform_(0.05, 0.3)
    harness = _SequenceHarness(adapter, method)
    x = (0.3 + torch.rand(1, 3, 2, dtype=torch.float64)).requires_grad_()
    state = (0.1 + 0.1 * torch.rand(1, 2, 2, dtype=torch.float64)).requires_grad_()
    names, parameters = zip(*harness.named_parameters())

    def run(sequence, initial, *values):
        return functional_call(harness, dict(zip(names, values)), (sequence, initial))

    # Both later reader outputs and the final state depend on the normalized
    # key, value, scalar logits, and input-gate vectors passed to gradcheck.
    assert torch.autograd.gradcheck(
        run,
        (x, state, *parameters),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=True,
    )
