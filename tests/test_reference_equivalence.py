"""Compare both sequence paths to independent single-token mathematics."""
from types import SimpleNamespace

import pytest
import torch

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.reference import recurrent_adapter_step
from looplora.infra.seeds import derive_seed


@pytest.mark.parametrize("variant", ["iterq", "fixedq", "unshared"])
@pytest.mark.parametrize("loop_steps", [0, 1, 2, 4])
@pytest.mark.parametrize("gate", ["scalar", "input"])
@pytest.mark.parametrize("r_mem", [2, 5])
@pytest.mark.parametrize("activation", ["relu", "identity"])
def test_sequence_paths_match_oracle_at_every_token(variant, loop_steps, gate, r_mem, activation):
    torch.manual_seed(derive_seed(710, "oracle", variant, loop_steps, gate, r_mem, activation))
    cfg = AdapterConfig(
        d_in=4, d_out=6, r=3, r_mem=r_mem, variant=variant,
        gate=gate, unshared_steps=loop_steps if variant == "unshared" else 1,
        alpha=0.7, rho=0.31, lambda_init=0.81, beta_init=0.23,
        activation=activation,
    )
    adapter = make_adapter(cfg).double()
    with torch.no_grad():
        adapter.B.normal_(std=0.4)
        if gate == "input":
            adapter.w_lambda.normal_(std=0.3)
            adapter.w_beta.normal_(std=0.3)
    p = SimpleNamespace(
        W=torch.randn(6, 4, dtype=torch.float64),
        bias=torch.randn(6, dtype=torch.float64),
        variant=variant, gate=gate, activation=activation,
        alpha=cfg.alpha, rho=cfg.rho, eps=cfg.eps,
        **{name: getattr(adapter, name) for name in
           ("A", "B", "R", "U", "Q", "K", "V", "lam", "beta", "a_lambda", "a_beta")},
    )
    if gate == "input":
        p.w_lambda, p.w_beta = adapter.w_lambda, adapter.w_beta
    x = torch.randn(3, 7, 4, dtype=torch.float64)
    initial = torch.randn(3, r_mem, r_mem, dtype=torch.float64) * 0.4
    initial_copy = initial.clone()
    valid = torch.rand(3, 7) > 0.3
    write = torch.rand(3, 7) > 0.35
    reset = torch.rand(3, 7) > 0.7
    # Include reset-on-padding and a valid read-only token in every case.
    valid[0, 1], reset[0, 1] = False, True
    valid[1, 2], write[1, 2] = True, False
    oracle_state = list(initial.unbind())
    reads = torch.zeros(3, dtype=torch.int64)
    writes = torch.zeros(3, dtype=torch.int64)
    expected_branches = []

    for t in range(x.shape[1]):
        branches = []
        for b in range(x.shape[0]):
            y, oracle_state[b], stats = recurrent_adapter_step(
                x[b, t], oracle_state[b], p, loop_steps,
                valid=bool(valid[b, t]), reset=bool(reset[b, t]),
                write_enabled=bool(write[b, t]),
            )
            branches.append(y - (p.W @ x[b, t] + p.bias))
            reads[b] += stats["reads"]
            writes[b] += stats["writes"]
        expected_branches.append(torch.stack(branches))
        expected = torch.stack(expected_branches, dim=1)
        states = torch.stack(oracle_state)
        # Prefixes verify intermediate states, including the vectorised path.
        for process in (adapter.process_sequence, adapter.process_sequence_fast):
            actual, state, stats = process(
                x[:, :t + 1], initial, loop_steps,
                valid=valid[:, :t + 1], write=write[:, :t + 1], reset=reset[:, :t + 1],
            )
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
            torch.testing.assert_close(state, states, rtol=1e-11, atol=1e-12)
            assert stats["reads"].dtype == torch.int64
            assert stats["writes"].dtype == torch.int64
            torch.testing.assert_close(stats["reads"], reads)
            torch.testing.assert_close(stats["writes"], writes)
    torch.testing.assert_close(initial, initial_copy, rtol=0, atol=0)


def test_oracle_defaults_to_appendix_a_iterative_reader():
    """Hand-computable scalar case checks orientation and decay-before-error."""
    t = lambda value: torch.tensor(value, dtype=torch.float64)
    p = SimpleNamespace(
        W=t([[2.0]]), bias=t([0.5]), A=t([[1.0]]), B=t([[3.0]]),
        R=t([[0.2]]), U=t([[0.4]]), Q=t([[0.5]]), K=t([[2.0]]), V=t([[1.5]]),
        rho=0.25, alpha=0.7, eps=1e-6, lam=t(0.8), beta=t(0.3),
    )
    y, state, stats = recurrent_adapter_step(t([2.0]), t([[4.0]]), p, 2)
    # z1=2+.25*(.2*2+.4*4*.5*2)=2.5; z2=2.625.
    torch.testing.assert_close(y, t([4.5 + 0.7 * 3 * 2.625]))
    torch.testing.assert_close(state, t([[3.2 + 0.3 * (3.0 - 3.2)]]))
    assert stats == {"reads": 2, "writes": 1}
