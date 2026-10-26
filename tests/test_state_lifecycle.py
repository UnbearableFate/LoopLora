import pytest
import torch

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.memory import delta_update, fork
from looplora.infra.seeds import derive_seed


@pytest.fixture(params=["iterq", "fixedq", "unshared"])
def adapter(request):
    torch.manual_seed(derive_seed(712, "lifecycle", request.node.name, request.param))
    module = make_adapter(AdapterConfig(
        d_in=4, d_out=5, r=3, r_mem=2, variant=request.param,
        unshared_steps=2, lambda_init=0.65, beta_init=0.35,
    )).double()
    with torch.no_grad():
        module.B.normal_(std=0.5)
    return module


@pytest.mark.parametrize("fast", [False, True])
def test_padding_and_disabled_writes_never_decay(adapter, fast):
    process = adapter.process_sequence_fast if fast else adapter.process_sequence
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    initial = torch.randn(2, 2, 2, dtype=torch.float64)
    for valid_value, write_value in ((False, True), (True, False)):
        out, final, stats = process(
            x, initial, 2,
            valid=torch.full((2, 5), valid_value),
            write=torch.full((2, 5), write_value),
        )
        torch.testing.assert_close(final, initial, rtol=0, atol=0)
        torch.testing.assert_close(stats["writes"], torch.zeros(2, dtype=torch.int64))
        read_count = (1 if adapter.cfg.variant == "fixedq" else 2) * 5 if valid_value else 0
        torch.testing.assert_close(stats["reads"], torch.full((2,), read_count, dtype=torch.int64))
        if not valid_value:
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("valid_value", [False, True])
def test_reset_precedes_padding_reads_and_writes(adapter, fast, valid_value):
    process = adapter.process_sequence_fast if fast else adapter.process_sequence
    x = torch.randn(2, 1, 4, dtype=torch.float64)
    initial = torch.randn(2, 2, 2, dtype=torch.float64)
    valid = torch.full((2, 1), valid_value)
    # Reset must remove prior state even when this token is padding.
    out, final, stats = process(x, initial, 2, valid=valid, reset=torch.ones(2, 1, dtype=torch.bool))
    expected, expected_state, expected_stats = process(x, torch.zeros_like(initial), 2, valid=valid)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_state, rtol=0, atol=0)
    for name in ("reads", "writes"):
        torch.testing.assert_close(stats[name], expected_stats[name])
    if not valid_value:
        torch.testing.assert_close(final, torch.zeros_like(final), rtol=0, atol=0)


@pytest.mark.parametrize("fast", [False, True])
def test_packed_sequences_equal_independent_runs(adapter, fast):
    process = adapter.process_sequence_fast if fast else adapter.process_sequence
    x = torch.randn(2, 7, 4, dtype=torch.float64)
    initial = torch.randn(2, 2, 2, dtype=torch.float64)
    valid = torch.tensor([[True, False, True, True, True, False, True],
                          [True, True, True, False, True, True, True]])
    write = torch.tensor([[True, True, False, True, False, True, True],
                          [False, True, True, True, True, False, True]])
    reset = torch.zeros(2, 7, dtype=torch.bool)
    reset[:, 3] = True  # Row 1 resets on padding; no prior state may survive.
    together, final, stats = process(x, initial, 2, valid=valid, write=write, reset=reset)
    first, _, first_stats = process(x[:, :3], initial, 2, valid=valid[:, :3], write=write[:, :3])
    second, second_final, second_stats = process(
        x[:, 3:], torch.zeros_like(initial), 2,
        valid=valid[:, 3:], write=write[:, 3:],
    )
    torch.testing.assert_close(together, torch.cat((first, second), dim=1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(final, second_final, rtol=1e-12, atol=1e-12)
    for name in ("reads", "writes"):
        torch.testing.assert_close(stats[name], first_stats[name] + second_stats[name])


@pytest.mark.parametrize("variant", ["iterq", "fixedq"])
def test_questions_and_loop_counts_fork_the_same_snapshot(variant):
    torch.manual_seed(derive_seed(712, "fork-questions", variant))
    adapter = make_adapter(AdapterConfig(d_in=4, d_out=5, r=3, r_mem=2, variant=variant)).double()
    with torch.no_grad():
        adapter.B.normal_()
    context = torch.randn(2, 3, 4, dtype=torch.float64)
    _, snapshot, _ = adapter.process_sequence(context, None, 1)
    original = snapshot.clone()
    states = fork(snapshot, 3)
    torch.testing.assert_close(states, snapshot.repeat_interleave(3, dim=0))
    questions = torch.randn(6, 2, 4, dtype=torch.float64)
    for steps in (0, 1, 4):
        branches, final, stats = adapter.process_sequence_fast(
            questions, states, steps, write=torch.zeros(6, 2, dtype=torch.bool),
        )
        for q in range(6):
            isolated, isolated_state, _ = adapter.process_sequence(
                questions[q:q + 1], snapshot[q // 3:q // 3 + 1], steps,
                write=torch.zeros(1, 2, dtype=torch.bool),
            )
            torch.testing.assert_close(branches[q:q + 1], isolated, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(final[q:q + 1], isolated_state, rtol=0, atol=0)
        torch.testing.assert_close(stats["writes"], torch.zeros(6, dtype=torch.int64))
    # Writing one fork cannot change another fork or the context snapshot.
    updated = adapter.write(states, adapter.u(questions[:, 0]), torch.tensor([True, False, False, False, False, False]))
    assert not torch.allclose(updated[0], states[0])
    torch.testing.assert_close(updated[1:], states[1:], rtol=0, atol=0)
    torch.testing.assert_close(snapshot, original, rtol=0, atol=0)
    # Independent storage also protects against a caller editing one fork.
    states[0].add_(1)
    torch.testing.assert_close(states[1], original[0], rtol=0, atol=0)
    torch.testing.assert_close(snapshot, original, rtol=0, atol=0)


def test_delta_primitive_decays_before_computing_prediction_error():
    state = torch.tensor([[[2., 3.], [5., 7.]]], dtype=torch.float64)
    key = torch.tensor([[1., 0.]], dtype=torch.float64)
    value = torch.tensor([[11., 13.]], dtype=torch.float64)
    updated = delta_update(state, key, value, torch.tensor([True]), lam=0.5, beta=0.25)
    # Mbar=[[1,1.5],[2.5,3.5]], prediction=[1,2.5], error=[10,10.5].
    expected = torch.tensor([[[3.5, 1.5], [5.125, 3.5]]], dtype=torch.float64)
    torch.testing.assert_close(updated, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, torch.tensor([[[2., 3.], [5., 7.]]], dtype=torch.float64), rtol=0, atol=0)


def test_delta_primitive_masks_decay_and_zero_key_is_finite():
    state = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]], dtype=torch.float64)
    key = torch.tensor([[0., 0.], [0., 1.], [0., 0.]], dtype=torch.float64)
    value = torch.tensor([[2., 3.], [4., 5.], [6., 7.]], dtype=torch.float64)
    lam = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64)
    beta = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    updated = delta_update(state, key, value, torch.tensor([True, False, False]), lam=lam, beta=beta)
    assert torch.isfinite(updated).all()
    # A zero key still decays a written row; masked rows never decay.
    torch.testing.assert_close(updated[0], state[0] * 0.4, rtol=0, atol=0)
    torch.testing.assert_close(updated[1:], state[1:], rtol=0, atol=0)
