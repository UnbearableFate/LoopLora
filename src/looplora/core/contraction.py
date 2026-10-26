"""Contraction certificates and a posteriori fixed-point error bounds."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def _finite_real(value: Tensor, name: str) -> None:
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be real and finite")


def _like(value: Tensor | float, reference: Tensor) -> Tensor:
    dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
    if isinstance(value, Tensor):
        _finite_real(value, "value")
    elif isinstance(value, complex):
        raise ValueError("value must be real and finite")
    tensor = torch.as_tensor(value, dtype=dtype, device=reference.device)
    _finite_real(tensor, "value")
    return tensor


def contraction_upper_bound(
    R: Tensor,
    U: Tensor,
    Q: Tensor,
    M: Tensor,
    rho: float | Tensor,
    lip: float | Tensor = 1.0,
) -> Tensor:
    """Return |rho| lip (||R||₂ + ||U||₂ ||M||₂ ||Q||₂).

    Spectral norms are exact SVD norms over the last two dimensions; batch
    dimensions broadcast. The composite matrix U M Q is never constructed.
    A value below one certifies contraction; a larger value is inconclusive.
    """
    for name, matrix in (("R", R), ("U", U), ("Q", Q), ("M", M)):
        if matrix.ndim < 2 or min(matrix.shape[-2:]) < 1:
            raise ValueError(f"{name} must have nonempty matrix dimensions")
        if not matrix.is_floating_point():
            raise ValueError(f"{name} must have a real floating dtype")
        _finite_real(matrix, name)
    r, r_mem = R.shape[-1], M.shape[-1]
    if (
        R.shape[-2] != r
        or M.shape[-2] != r_mem
        or U.shape[-2:] != (r, r_mem)
        or Q.shape[-2:] != (r_mem, r)
    ):
        raise ValueError("expected R[r,r], U[r,r_mem], M[r_mem,r_mem], Q[r_mem,r]")
    rho_t, lip_t = _like(rho, R), _like(lip, R)
    _finite_real(rho_t, "rho")
    _finite_real(lip_t, "lip")
    if bool((lip_t < 0).any()):
        raise ValueError("lip must be nonnegative")
    norm_r, norm_u, norm_m, norm_q = (
        torch.linalg.matrix_norm(matrix, ord=2) for matrix in (R, U, M, Q)
    )
    return rho_t.abs() * lip_t * (norm_r + norm_u * norm_m * norm_q)


def residual(F: Callable[[Tensor], Tensor], z: Tensor) -> Tensor:
    """Return ||F(z)-z||₂ over the final vector dimension, retaining batches."""
    if z.ndim == 0:
        raise ValueError("z must have a vector dimension")
    _finite_real(z, "z")
    next_z = F(z)
    if next_z.shape != z.shape:
        raise ValueError("F(z) must have the same shape as z")
    _finite_real(next_z, "F(z)")
    return torch.linalg.vector_norm(next_z - z, dim=-1)


def fixed_point_error_bound(e_s: Tensor | float, c: Tensor | float) -> Tensor:
    """Return e_s/(1-c), requiring nonnegative residuals and 0 <= c < 1."""
    reference = e_s if isinstance(e_s, Tensor) else c if isinstance(c, Tensor) else torch.tensor(0.0)
    error, coefficient = _like(e_s, reference), _like(c, reference)
    _finite_real(error, "e_s")
    _finite_real(coefficient, "c")
    if bool((error < 0).any()):
        raise ValueError("e_s must be nonnegative")
    if bool(((coefficient < 0) | (coefficient >= 1)).any()):
        raise ValueError("c must satisfy 0 <= c < 1")
    return error / (1 - coefficient)
