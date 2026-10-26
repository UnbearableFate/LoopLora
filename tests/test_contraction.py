import pytest
import torch

from looplora.core.contraction import contraction_upper_bound, fixed_point_error_bound, residual
from looplora.infra.seeds import derive_seed


DTYPE = torch.float64


def test_spectral_norm_bound_uses_exact_matrix_norms_and_absolute_rho():
    R = torch.diag(torch.tensor([0.3, -0.4], dtype=DTYPE))
    U = torch.diag(torch.tensor([0.5, 0.2], dtype=DTYPE))
    Q = torch.diag(torch.tensor([0.7, -0.1], dtype=DTYPE))
    M = torch.stack([torch.eye(2, dtype=DTYPE), torch.eye(2, dtype=DTYPE) * 2])
    c = contraction_upper_bound(R, U, Q, M, rho=-0.25, lip=2.0)
    torch.testing.assert_close(c, torch.tensor([0.375, 0.55], dtype=DTYPE))
    actual = 0.5 * torch.linalg.matrix_norm(R + U @ M @ Q, ord=2)
    assert bool((actual <= c).all())


def test_iterates_obey_actual_fixed_point_error_bound_at_each_step():
    generator = torch.Generator().manual_seed(derive_seed(17, "contraction", "fixed-point"))
    R = 0.12 * torch.randn(3, 3, dtype=DTYPE, generator=generator)
    U = 0.1 * torch.randn(3, 2, dtype=DTYPE, generator=generator)
    Q = 0.1 * torch.randn(2, 3, dtype=DTYPE, generator=generator)
    M = 0.2 * torch.randn(4, 2, 2, dtype=DTYPE, generator=generator)
    R = 0.3 * R / torch.linalg.matrix_norm(R, ord=2)
    U = 0.2 * U / torch.linalg.matrix_norm(U, ord=2)
    Q = 0.2 * Q / torch.linalg.matrix_norm(Q, ord=2)
    M = 0.3 * M / torch.linalg.matrix_norm(M, ord=2)[..., None, None]
    rho = 0.7
    c = contraction_upper_bound(R, U, Q, M, rho)
    assert bool(((c > 0) & (c < 1)).all())
    H = R + U @ M @ Q
    fixed_point = torch.randn(4, 3, dtype=DTYPE, generator=generator)
    u = fixed_point - rho * torch.relu(torch.einsum("bij,bj->bi", H, fixed_point))

    def F(z):
        return u + rho * torch.relu(torch.einsum("bij,bj->bi", H, z))

    torch.testing.assert_close(F(fixed_point), fixed_point)
    z = torch.randn(4, 3, dtype=DTYPE, generator=generator)
    for _ in range(16):
        true_error = torch.linalg.vector_norm(z - fixed_point, dim=-1)
        bound = fixed_point_error_bound(residual(F, z), c)
        assert bound.shape == (4,)
        assert bool((true_error <= bound + 2e-14).all())
        z = F(z)


def test_noncontractive_instance_does_not_admit_the_bound():
    R = 2 * torch.eye(2, dtype=DTYPE)
    zero = torch.zeros(2, 2, dtype=DTYPE)
    c = contraction_upper_bound(R, zero, zero, zero, rho=0.6)
    assert c.item() == pytest.approx(1.2)
    with pytest.raises(ValueError, match="0 <= c < 1"):
        fixed_point_error_bound(torch.tensor(0.5, dtype=DTYPE), c)


def test_residual_retains_all_batch_dimensions_and_autograd():
    z = torch.arange(24, dtype=DTYPE).reshape(2, 4, 3).requires_grad_()
    e = residual(lambda v: 0.5 * v, z)
    torch.testing.assert_close(e, 0.5 * torch.linalg.vector_norm(z, dim=-1))
    assert e.shape == (2, 4)
    bound = fixed_point_error_bound(e, 0.5)
    assert bound.dtype == DTYPE
    bound.sum().backward()
    assert z.grad is not None and bool(torch.isfinite(z.grad).all())


@pytest.mark.parametrize("c", [-0.1, 1.0, 1.2, float("nan"), float("inf"), -float("inf")])
def test_error_bound_rejects_invalid_contraction_coefficients(c):
    with pytest.raises(ValueError):
        fixed_point_error_bound(torch.ones(2, dtype=DTYPE), c)


@pytest.mark.parametrize("e", [-1.0, float("nan"), float("inf")])
def test_error_bound_rejects_invalid_residuals(e):
    with pytest.raises(ValueError):
        fixed_point_error_bound(e, 0.5)


def test_error_bound_broadcasts_and_handles_zero_coefficient():
    c = torch.tensor([0.0, 0.5], dtype=DTYPE)
    result = fixed_point_error_bound(2.0, c)
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([2.0, 4.0], dtype=DTYPE))
    with pytest.raises(ValueError):
        fixed_point_error_bound(1.0, torch.tensor([0.5, 1.0]))


@pytest.mark.parametrize("rho, lip", [(float("nan"), 1.0), (0.5, -1.0), (0.5, float("inf"))])
def test_upper_bound_rejects_invalid_scalars(rho, lip):
    eye = torch.eye(2, dtype=DTYPE)
    with pytest.raises(ValueError):
        contraction_upper_bound(eye, eye, eye, eye, rho, lip)


def test_shape_errors_are_reported_before_computing_norms():
    eye = torch.eye(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="expected R"):
        contraction_upper_bound(eye, torch.ones(3, 2, dtype=DTYPE), eye, eye, 0.5)
    with pytest.raises(ValueError, match="same shape"):
        residual(lambda z: z[0], eye)
    with pytest.raises(ValueError, match="vector dimension"):
        residual(lambda z: z, torch.tensor(1.0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_matrices_and_residuals_are_rejected(bad):
    eye = torch.eye(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="finite"):
        contraction_upper_bound(eye * bad, eye, eye, eye, 0.5)
    with pytest.raises(ValueError, match="finite"):
        residual(lambda z: z * bad, torch.ones(2, dtype=DTYPE))
