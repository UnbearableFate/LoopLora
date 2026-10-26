"""Small deterministic witnesses for docs/theory.md; run on a compute node."""

import pytest
import torch

from looplora.core import AdapterConfig, make_adapter
from looplora.core.memory import delta_update
from looplora.infra.seeds import derive_seed


DTYPE = torch.float64


def _reader(r, variant="iterq", activation="identity"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(17, "theory-examples", r, variant, activation))
        model = make_adapter(AdapterConfig(
            d_in=r, d_out=r, r=r, variant=variant, activation=activation,
            rho=.25, lambda_init=.5, beta_init=.5,
        )).double()
    with torch.no_grad():
        for name in ("A", "B", "K", "V", "Q"):
            getattr(model, name).copy_(torch.eye(r, dtype=DTYPE))
        model.U.copy_(8 * torch.eye(r, dtype=DTYPE))
        model.R.zero_()
    return model


def _store(r, edges):
    codes = torch.eye(r, dtype=DTYPE)
    memory = torch.zeros(1, r, r, dtype=DTYPE)
    for head, tail in edges:
        memory = delta_update(memory, codes[head:head + 1], codes[tail:tail + 1],
                              None, lam=1., beta=1.)
    return memory


@pytest.mark.parametrize("h", [1, 2, 4, 8])
@pytest.mark.parametrize("reverse_writes", [False, True])
def test_orthogonal_chain_terminal_and_default_gain(h, reverse_writes):
    edges = [(j, j + 1) for j in range(h)]
    memory = _store(h + 1, edges[::-1] if reverse_writes else edges)
    expected_memory = torch.zeros_like(memory)
    for head, tail in edges:
        expected_memory[0, tail, head] = 1
    torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)
    model = _reader(h + 1)
    u = torch.eye(h + 1, dtype=DTYPE)[:1]
    for depth in range(h + 3):
        actual = model.branch(model.read(u, memory, depth))
        expected = torch.zeros_like(u)
        for j in range(min(depth, h) + 1):
            expected[0, j] = 2 ** j
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.argmax(-1).item() == min(depth, h)
    with torch.no_grad():
        model.U.copy_(torch.eye(h + 1, dtype=DTYPE))
    for depth in range(1, h + 3):
        assert model.read(u, memory, depth).argmax(-1).item() == 0


@pytest.mark.parametrize("activation", ["identity", "relu"])
def test_same_first_hop_hides_terminal_from_b4_and_fixed_query(activation):
    first = _store(6, [(0, 1), (1, 2), (3, 4), (4, 5)])
    second = _store(6, [(0, 1), (1, 5), (3, 4), (4, 2)])
    u = torch.eye(6, dtype=DTYPE)[:1]
    torch.testing.assert_close(first @ u.unsqueeze(-1), second @ u.unsqueeze(-1))
    # Both episodes share the same six candidate codes, but terminals are 2/5.
    for variant in ("iterq", "fixedq"):
        model = _reader(6, variant, activation)
        with torch.no_grad():
            model.R.copy_(.5 * torch.eye(6, dtype=DTYPE))
        for depth in ([1] if variant == "iterq" else [1, 2, 4, 8]):
            torch.testing.assert_close(model.branch(model.read(u, first, depth)),
                                       model.branch(model.read(u, second, depth)),
                                       rtol=0, atol=0)
        if variant == "iterq":
            assert model.read(u, first, 2).argmax(-1).item() == 2
            assert model.read(u, second, 2).argmax(-1).item() == 5


def test_sequence_memory_path_jacobian_can_exceed_module_rank():
    model = _reader(1, activation="relu")
    with torch.no_grad():
        model.U.fill_(1)
    x = torch.tensor([1., 2., 3.], dtype=DTYPE, requires_grad=True)

    def sequence(value):
        output, _, _ = model.process_sequence(value.reshape(1, 3, 1), loop_steps=1)
        return output.flatten()

    jacobian = torch.autograd.functional.jacobian(sequence, x)
    history = torch.tril(jacobian, diagonal=-1)
    expected = .25 * .5 * torch.tensor([[0., 0., 0.], [2., 0., 0.],
                                        [3. / 4, 3., 0.]], dtype=DTYPE)
    torch.testing.assert_close(history, expected, rtol=0, atol=0)
    assert torch.linalg.matrix_rank(history).item() == 2 > model.cfg.r
    assert torch.linalg.matrix_rank(jacobian).item() == 3


def test_noncommuting_power_identity_and_polynomial_perturbation_bound():
    model = _reader(2)
    with torch.no_grad():
        model.R.copy_(torch.tensor([[.2, -.1], [.3, .1]], dtype=DTYPE))
        model.U.copy_(torch.tensor([[1., .2], [0., .8]], dtype=DTYPE))
        model.Q.copy_(torch.tensor([[.7, 0.], [-.2, 1.]], dtype=DTYPE))
    memory = torch.tensor([[.1, .8], [0., -.2]], dtype=DTYPE)
    reference = torch.tensor([[.3, 0.], [.4, -.1]], dtype=DTYPE)
    X = model.cfg.rho * (model.R + model.U @ memory @ model.Q)
    Y = model.cfg.rho * (model.R + model.U @ reference @ model.Q)
    assert torch.linalg.matrix_norm(X @ Y - Y @ X).item() > 1e-3
    norm = lambda value: torch.linalg.matrix_norm(value, ord=2)
    u = torch.tensor([[.7, -.4]], dtype=DTYPE)
    d = abs(model.cfg.rho) * norm(model.U) * norm(memory - reference) * norm(model.Q)
    polynomial_x = torch.eye(2, dtype=DTYPE)
    polynomial_y = polynomial_x.clone()
    series_bound = torch.zeros((), dtype=DTYPE)
    for depth in range(1, 7):
        telescoped = sum(torch.linalg.matrix_power(X, depth - 1 - k) @ (X - Y)
                         @ torch.linalg.matrix_power(Y, k) for k in range(depth))
        power_x, power_y = (torch.linalg.matrix_power(Z, depth) for Z in (X, Y))
        torch.testing.assert_close(power_x - power_y, telescoped, rtol=1e-12, atol=1e-14)
        scalar_sum = sum(norm(X) ** (depth - 1 - k) * norm(Y) ** k
                         for k in range(depth))
        assert norm(power_x - power_y) <= norm(X - Y) * scalar_sum + 1e-14
        polynomial_x, polynomial_y = polynomial_x + power_x, polynomial_y + power_y
        actual = model.read(u, memory[None], depth)
        baseline = model.read(u, reference[None], depth)
        torch.testing.assert_close(actual, u @ polynomial_x.T, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(baseline, u @ polynomial_y.T, rtol=1e-12, atol=1e-14)
        series_bound = series_bound + scalar_sum
        assert torch.linalg.vector_norm(actual - baseline) <= d * u.norm() * series_bound + 1e-14


def test_relu_cannot_be_moved_through_input_projection():
    A = torch.tensor([[1., 1.], [0., 1.]], dtype=DTYPE)
    B = torch.eye(2, dtype=DTYPE)
    z = torch.tensor([1., -1.], dtype=DTYPE)
    before = A @ torch.relu(B @ z)
    after = torch.relu(A @ B @ z)
    torch.testing.assert_close(before, torch.tensor([1., 0.], dtype=DTYPE))
    torch.testing.assert_close(after, torch.tensor([0., 0.], dtype=DTYPE))
    assert not torch.equal(before, after)


def test_signed_codes_need_more_than_a_hidden_bias_for_identity():
    codes = torch.cat((torch.eye(2, dtype=DTYPE), -torch.eye(2, dtype=DTYPE)))
    hidden_only = torch.relu(codes + 2)
    assert bool((hidden_only >= 0).all())
    assert not torch.equal(hidden_only, codes)
    torch.testing.assert_close(hidden_only - 2, codes, rtol=0, atol=0)
    U = torch.cat((torch.eye(2, dtype=DTYPE), -torch.eye(2, dtype=DTYPE)))
    Q = U.T
    torch.testing.assert_close(torch.relu(codes @ U.T) @ Q.T, codes, rtol=0, atol=0)


def test_exact_elimination_preserves_state_dependent_router():
    A = torch.tensor([[.2, -.1, .4], [.3, .2, -.1]], dtype=DTYPE)
    B = torch.tensor([[.2, -.1], [.1, .3], [-.2, .1]], dtype=DTYPE)
    x = torch.tensor([.4, -.2, .7], dtype=DTYPE)
    w = torch.tensor([.1, -.3, .2], dtype=DTYPE)
    h, z = x.clone(), torch.zeros(2, dtype=DTYPE)
    for _ in range(8):
        gate_full = torch.sigmoid(w @ h + .1)
        gate_low = torch.sigmoid(w @ x + (w @ B) @ z + .1)
        torch.testing.assert_close(gate_full, gate_low, rtol=1e-12, atol=1e-14)
        h = h + gate_full * B @ torch.relu(A @ h)
        z = z + gate_low * torch.relu(A @ x + (A @ B) @ z)
        torch.testing.assert_close(h, x + B @ z, rtol=1e-12, atol=1e-14)


def test_double_ba_zero_first_derivative_but_not_a_two_sided_hessian():
    def original(a, b):
        return b * a * torch.relu(b * a)  # x=1

    a = torch.tensor(1., dtype=DTYPE, requires_grad=True)
    b = torch.tensor(0., dtype=DTYPE, requires_grad=True)
    gradient = torch.autograd.grad(original(a, b), (a, b))
    assert all(value.item() == 0 for value in gradient)
    # Homogeneity gives distinct directional quadratic coefficients at zero.
    for size in (1e-2, 1e-4):
        for sign, coefficient in ((1, 1.), (-1, 0.)):
            point = torch.tensor(sign * size, dtype=DTYPE, requires_grad=True)
            value = original(a, point)
            assert (value / size ** 2).item() == pytest.approx(coefficient)
            slope = torch.autograd.grad(value, point, create_graph=True)[0]
            curvature = torch.autograd.grad(slope, point)[0]
            assert curvature.item() == 2 * coefficient
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(17, "theory-examples", "double-ba-skip"))
        skip = make_adapter(AdapterConfig(d_in=1, d_out=1, r=1,
                                         variant="double_ba", double_ba_form="skip")).double()
    with torch.no_grad():
        skip.A.fill_(1)
    output, _, _ = skip.process_sequence(torch.ones(1, 1, 1, dtype=DTYPE))
    assert output.item() == 0
    assert torch.autograd.grad(output.sum(), skip.B)[0].item() == 1
