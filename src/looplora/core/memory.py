"""Functional decay-first memory writes and pre-write snapshots.

The state is [B, r_mem, key_dim]: key_dim = r_mem, or 2 * r_mem * nu when keys and
queries are mapped through DPFP features (``feature_map``, off by default).
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

FeatureMap = Callable[[Tensor], Tensor]


def dpfp_features(k: Tensor, nu: int) -> Tensor:
    """DPFP of Schlag, Irie & Schmidhuber (2021, sec. 5.4): [r * roll(r, j)]_{j=1..nu}, r = relu([k; -k]).

    Maps [..., d] to [..., 2 * d * nu]; nonnegative, and homogeneous of degree two for scalars c >= 0.
    """
    if isinstance(nu, bool) or not isinstance(nu, int) or nu < 1:
        raise ValueError("nu must be a positive integer")
    r = torch.relu(torch.cat((k, -k), dim=-1))
    return torch.cat([r * torch.roll(r, shifts=j, dims=-1) for j in range(1, nu + 1)], dim=-1)


def zeros_state(
    batch: int, r_mem: int, dtype: torch.dtype, device: torch.device | str, key_dim: int | None = None,
) -> Tensor:
    if batch < 0 or r_mem <= 0 or (key_dim is not None and key_dim <= 0):
        raise ValueError("batch must be nonnegative and r_mem and key_dim positive")
    return torch.zeros(batch, r_mem, r_mem if key_dim is None else key_dim, dtype=dtype, device=device)


def fork(M: Tensor, n: int) -> Tensor:
    """Return [B*n, r_mem, key_dim], grouping n independent copies per sample."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    _check_state(M)
    return M.repeat_interleave(n, dim=0)


def _check_state(M: Tensor) -> None:
    if M.ndim != 3 or not M.is_floating_point():
        raise ValueError("state must be a floating tensor [B, r_mem, key_dim]")


def _check_key_width(M0: Tensor, K: Tensor, feature_map: FeatureMap | None) -> None:
    """The state's key width must be the width of the projected key after the optional feature map."""
    width = K.shape[0] if feature_map is None else feature_map(K.new_zeros(1, K.shape[0])).shape[-1]
    if M0.shape[2] != width:
        raise ValueError(f"state key width {M0.shape[2]} does not match the written key width {width}")


def _mask(value: Tensor | None, shape: tuple[int, ...], device: torch.device, default: bool) -> Tensor:
    if value is None:
        return torch.full(shape, default, dtype=torch.bool, device=device)
    if value.shape != shape or value.device != device:
        raise ValueError(f"mask must have shape {shape} and device {device}")
    return value.to(dtype=torch.bool)


def _gate(value: Tensor | float, M: Tensor, indices: Tensor) -> Tensor:
    gate = torch.as_tensor(value, dtype=M.dtype, device=M.device)
    if gate.numel() == 1:
        return gate.reshape(1, 1, 1)
    if gate.shape not in {(M.shape[0],), (M.shape[0], 1), (M.shape[0], 1, 1)}:
        raise ValueError("gate must be scalar or have one value per batch row")
    return gate.reshape(M.shape[0]).index_select(0, indices)[:, None, None]


def delta_update(
    M: Tensor, k: Tensor, v: Tensor, write_mask: Tensor | None,
    lam: Tensor | float, beta: Tensor | float,
) -> Tensor:
    """Apply the decay-first rule to already-normalized keys; never mutate M.

    Gate values are probabilities supplied by the caller. State arithmetic and
    the returned state use M.dtype; casts preserve gradients.
    """
    _check_state(M)
    if (k.shape != (M.shape[0], M.shape[2]) or v.shape != M.shape[:2]
            or k.device != M.device or v.device != M.device):
        raise ValueError("k must have shape [B, key_dim] and v [B, r_mem] on the state device")
    mask = _mask(write_mask, (M.shape[0],), M.device, True)
    indices = mask.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return M
    key = k.index_select(0, indices).to(M.dtype)
    value = v.index_select(0, indices).to(M.dtype)
    decayed = _gate(lam, M, indices) * M.index_select(0, indices)
    error = value - torch.bmm(decayed, key.unsqueeze(-1)).squeeze(-1)
    updated = decayed + _gate(beta, M, indices) * error.unsqueeze(-1) * key.unsqueeze(-2)
    return M.index_copy(0, indices, updated)


def write(
    M: Tensor, u: Tensor, write_mask: Tensor | None, *, K: Tensor, V: Tensor,
    lam: Tensor | float, beta: Tensor | float, eps: float = 1e-6,
    feature_map: FeatureMap | None = None,
) -> Tensor:
    """Project and normalize one token's writer codes, then apply delta_update.

    A feature_map is applied to the projected key before its unit normalization.
    """
    _check_state(M)
    if u.ndim != 2 or u.shape[0] != M.shape[0] or u.device != M.device:
        raise ValueError("u must have shape [B, r] on the state device")
    if K.shape != (M.shape[1], u.shape[-1]) or V.shape != K.shape:
        raise ValueError("K and V must have shape [r_mem, r]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    mask = _mask(write_mask, (M.shape[0],), M.device, True)
    # Exclude inactive rows from projection, including undefined padding values.
    indices = mask.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return M
    active_u = u.index_select(0, indices)
    raw = F.linear(active_u, K)
    if feature_map is not None:
        raw = feature_map(raw)
    key = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(eps)
    value = F.linear(active_u, V)
    active_lam = _gate(lam, M, indices).flatten()
    active_beta = _gate(beta, M, indices).flatten()
    updated = delta_update(
        M.index_select(0, indices), key, value, None, active_lam, active_beta,
    )
    return M.index_copy(0, indices, updated)


def scan_prewrite(
    M0: Tensor, u: Tensor, valid: Tensor | None = None, write: Tensor | None = None,
    reset: Tensor | None = None, *, K: Tensor, V: Tensor, lam: Tensor | float,
    beta: Tensor | float, eps: float = 1e-6, feature_map: FeatureMap | None = None,
) -> tuple[Tensor, Tensor]:
    """Sequential functional scan; output t is after reset, before token t writes.

    lam/beta can be scalars or [B,T] input-dependent gate probabilities. This
    reference scan stores O(B*T*r_mem**2) state, without dense affine products.
    """
    _check_state(M0)
    if u.ndim != 3 or u.shape[0] != M0.shape[0] or u.device != M0.device:
        raise ValueError("u must have shape [B,T,r] on the state device")
    _check_key_width(M0, K, feature_map)
    shape = tuple(u.shape[:2])
    valid_mask = _mask(valid, shape, u.device, True)
    write_mask = _mask(write, shape, u.device, True)
    reset_mask = _mask(reset, shape, u.device, False)
    for gate in (lam, beta):
        if isinstance(gate, Tensor) and gate.numel() != 1 and gate.shape != shape:
            raise ValueError("scan gates must be scalar or [B,T]")
    state = M0
    snapshots = []
    for t in range(u.shape[1]):
        state = torch.where(reset_mask[:, t, None, None], torch.zeros_like(state), state)
        snapshots.append(state)
        lam_t = lam[:, t] if isinstance(lam, Tensor) and lam.numel() != 1 else lam
        beta_t = beta[:, t] if isinstance(beta, Tensor) and beta.numel() != 1 else beta
        state = _write_token(
            state, u[:, t], valid_mask[:, t] & write_mask[:, t],
            K=K, V=V, lam=lam_t, beta=beta_t, eps=eps, feature_map=feature_map,
        )
    if not snapshots:
        return M0.new_empty((M0.shape[0], 0, *M0.shape[1:])), state
    return torch.stack(snapshots, dim=1), state


def scan_final(
    M0: Tensor, u: Tensor, valid: Tensor | None = None, write: Tensor | None = None,
    reset: Tensor | None = None, *, K: Tensor, V: Tensor, lam: Tensor | float,
    beta: Tensor | float, eps: float = 1e-6, feature_map: FeatureMap | None = None,
) -> Tensor:
    """Final-only scan over packed writes, preserving each row's token order.

    Two sequence-level decisions synchronize with the host (interior resets and
    maximum write count). The write loop uses only dense masked tensor updates.
    Interior resets use the reference; a reset at token zero stays on this path.
    """
    _check_state(M0)
    if u.ndim != 3 or u.shape[0] != M0.shape[0] or u.device != M0.device:
        raise ValueError("u must have shape [B,T,r] on the state device")
    shape = tuple(u.shape[:2])
    valid_mask = _mask(valid, shape, u.device, True)
    write_mask = _mask(write, shape, u.device, True)
    reset_mask = _mask(reset, shape, u.device, False)
    for gate in (lam, beta):
        if isinstance(gate, Tensor) and gate.numel() != 1 and gate.shape != shape:
            raise ValueError("scan gates must be scalar or [B,T]")
    if u.shape[1] == 0:
        return M0
    if K.shape != (M0.shape[1], u.shape[-1]) or V.shape != K.shape:
        raise ValueError("K and V must have shape [r_mem, r]")
    _check_key_width(M0, K, feature_map)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if reset is not None and bool(reset_mask[:, 1:].any()):
        return scan_prewrite(M0, u, valid, write, reset, K=K, V=V,
                             lam=lam, beta=beta, eps=eps, feature_map=feature_map)[1]
    state = torch.where(reset_mask[:, 0, None, None], torch.zeros_like(M0), M0)
    if u.shape[0] == 0:
        return state
    permitted = valid_mask & write_mask
    counts = permitted.sum(dim=1)
    maximum = int(counts.max().item())
    if maximum == 0:
        return state
    positions = torch.arange(u.shape[1], device=u.device).expand(shape)
    positions = positions.masked_fill(~permitted, u.shape[1]).sort(dim=1).values[:, :maximum]
    positions = positions.clamp_max(u.shape[1] - 1)
    enabled = torch.arange(maximum, device=u.device)[None] < counts[:, None]
    packed = u.gather(1, positions[..., None].expand(-1, -1, u.shape[-1]))
    packed = torch.where(enabled[..., None], packed, torch.zeros_like(packed))

    def pack_gate(value: Tensor | float) -> Tensor:
        gate = torch.as_tensor(value, dtype=M0.dtype, device=M0.device)
        if gate.numel() == 1:
            return gate.reshape(1, 1).expand(u.shape[0], maximum)
        # Undefined non-write gates must not enter arithmetic or gradients.
        return torch.where(enabled, gate.gather(1, positions), torch.zeros_like(enabled, dtype=M0.dtype))

    lambdas, betas = pack_gate(lam), pack_gate(beta)
    for step in range(maximum):
        mask = enabled[:, step, None, None]
        # Keep the reference's per-token projections. A single larger GEMM
        # changes FP32 rounding enough to affect a subsequent BF16 backbone.
        token = packed[:, step].contiguous()
        raw = F.linear(token, K)
        if feature_map is not None:
            raw = feature_map(raw)
        key = (raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(eps)).to(M0.dtype)
        value = F.linear(token, V).to(M0.dtype)
        active_state = torch.where(mask, state, torch.zeros_like(state))
        decayed = lambdas[:, step, None, None] * active_state
        error = value - torch.bmm(decayed, key.unsqueeze(-1)).squeeze(-1)
        updated = decayed + betas[:, step, None, None] * error.unsqueeze(-1) * key.unsqueeze(-2)
        state = torch.where(mask, updated, state)
    return state


# The public scan's `write` mask intentionally has the task-specified name.
_write_token = write


# --- Least-squares stores (WP-L7 LSQ). Final-only: the support is written as one batch, never token by token. ---

def _scalar_gate(value: Tensor | float, M0: Tensor, name: str) -> Tensor:
    gate = torch.as_tensor(value, dtype=M0.dtype, device=M0.device)
    if gate.numel() != 1:
        raise ValueError(f"the least-squares stores require a scalar {name} gate")
    return gate.reshape(())


def ridge_final(
    M0: Tensor, u: Tensor, valid: Tensor | None = None, write: Tensor | None = None,
    reset: Tensor | None = None, *, K: Tensor, V: Tensor, lam: Tensor | float, eta: float,
    eps: float = 1e-6, feature_map: FeatureMap | None = None,
) -> Tensor:
    """Recency-weighted ridge least-squares store of a row's n packed writes (unit keys k_j, values v_j).

    The state minimizes  sum_j w_j |M k_j - v_j|^2 + eta |M - lam^n M0|_F^2  with w_j = lam^(n-1-j) (the last write has
    weight one; lam is the scalar decay gate), i.e.  M = Mb + (V - Mb K) W K^T (K W K^T + eta I)^-1  with Mb = lam^n M0.
    For M0 = 0 and eta -> 0 it is the interpolant M k_j = v_j whenever the keys are independent (n <= r_mem), whatever
    lam. It is solved in float64 through the n x n system (K^T W K + eta I; a Cholesky solve, differentiable). A reset at
    token zero clears M0; interior resets are unsupported. Keys and values use the per-token projections of scan_final.
    With a `feature_map` (DPFP, SQ2) the unit keys are the normalized features of the projected keys and the state is
    [B, r_mem, key_dim] with key_dim the feature width; the solve is the same n x n system (key_dim need not equal r_mem).
    """
    _check_state(M0)
    if u.ndim != 3 or u.shape[0] != M0.shape[0] or u.device != M0.device:
        raise ValueError("u must have shape [B,T,r] on the state device")
    shape = tuple(u.shape[:2])
    valid_mask = _mask(valid, shape, u.device, True)
    write_mask = _mask(write, shape, u.device, True)
    reset_mask = _mask(reset, shape, u.device, False)
    lam_t = _scalar_gate(lam, M0, "decay")
    if isinstance(eta, bool) or not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    if u.shape[1] == 0:
        return M0
    if K.shape != (M0.shape[1], u.shape[-1]) or V.shape != K.shape:
        raise ValueError("K and V must have shape [r_mem, r]")
    _check_key_width(M0, K, feature_map)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if bool(reset_mask[:, 1:].any()):
        raise ValueError("interior resets are not supported by the least-squares stores")
    base = torch.where(reset_mask[:, 0, None, None], torch.zeros_like(M0), M0)
    if u.shape[0] == 0:
        return base
    permitted = valid_mask & write_mask
    counts = permitted.sum(dim=1)
    maximum = int(counts.max().item())
    if maximum == 0:
        return base
    positions = torch.arange(u.shape[1], device=u.device).expand(shape)
    positions = positions.masked_fill(~permitted, u.shape[1]).sort(dim=1).values[:, :maximum]
    positions = positions.clamp_max(u.shape[1] - 1)
    enabled = torch.arange(maximum, device=u.device)[None] < counts[:, None]
    packed = u.gather(1, positions[..., None].expand(-1, -1, u.shape[-1]))
    packed = torch.where(enabled[..., None], packed, torch.zeros_like(packed))
    keys, values = [], []
    for step in range(maximum):
        token = packed[:, step].contiguous()
        raw = F.linear(token, K)
        if feature_map is not None:
            raw = feature_map(raw)
        keys.append(raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(eps))
        values.append(F.linear(token, V))
    keys, values = torch.stack(keys, 1).double(), torch.stack(values, 1).double()
    lam64 = lam_t.double()
    ages = (counts[:, None] - 1 - torch.arange(maximum, device=u.device)[None]).clamp_min(0).double()
    # sqrt(w_j) = lam^(age/2); disabled slots get exactly zero (no sqrt at zero, so no NaN gradients).
    root = torch.where(enabled, lam64 ** (0.5 * ages), torch.zeros_like(ages))
    prior = (lam64 ** counts.double())[:, None, None] * base.double()
    residual = values - torch.einsum("bij,bnj->bni", prior, keys)
    scaled = keys * root[..., None]
    system = scaled @ scaled.transpose(1, 2)
    system = system + eta * torch.eye(maximum, dtype=system.dtype, device=system.device)
    solved = torch.cholesky_solve(scaled, torch.linalg.cholesky(system))
    update = (residual * root[..., None]).transpose(1, 2) @ solved
    return (prior + update).to(M0.dtype)


def sweep_final(
    M0: Tensor, u: Tensor, valid: Tensor | None = None, write: Tensor | None = None,
    reset: Tensor | None = None, *, K: Tensor, V: Tensor, lam: Tensor | float, eps: float = 1e-6,
    sweeps: int = 4,
) -> Tensor:
    """K-sweep store: the final-only beta = 1 delta scan repeated `sweeps` times over the same ordered writes.

    Each pass is the unchanged scan_final with unit keys and beta = 1, so every write overwrites exactly along its key
    (a Kaczmarz step); a reset at token zero clears M0 once, before the first pass, and interior resets are unsupported.
    With lam = 1 and independent keys the states converge to the interpolant as `sweeps` grows; sweeps = 1 is
    scan_final(beta = 1).
    """
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 1:
        raise ValueError("sweeps must be a positive integer")
    if reset is not None and bool(reset[:, 1:].any()):
        raise ValueError("interior resets are not supported by the least-squares stores")
    state = scan_final(M0, u, valid, write, reset, K=K, V=V, lam=lam, beta=1.0, eps=eps)
    for _ in range(sweeps - 1):
        state = scan_final(state, u, valid, write, None, K=K, V=V, lam=lam, beta=1.0, eps=eps)
    return state


def ridge_coefficients(keys: Tensor, q: Tensor, lam: Tensor | float, eta: float) -> dict[str, Tensor]:
    """Float64 coefficients of the ridge_final read m = sum_j d_j v_j for M0 = 0 and n valid writes.

    keys [B,n,r] are the unit written keys, q [B,Q,r] the queries. With W = diag(lam^(n-1-j)):
    a = (W^1/2 K^T K W^1/2 + eta I)^-1 W^1/2 (K^T q) and d = W^1/2 a (b = K^T q, gram = K^T K).
    """
    keys, q = keys.double(), q.double()
    n = keys.shape[1]
    lam = torch.as_tensor(lam, dtype=torch.float64, device=keys.device)
    root = (lam ** torch.arange(n - 1, -1, -1, device=keys.device, dtype=torch.float64)).sqrt()
    gram = keys @ keys.transpose(1, 2)
    b = torch.einsum("bnr,bqr->bqn", keys, q)
    system = root[:, None] * gram * root[None, :] + eta * torch.eye(n, dtype=gram.dtype, device=gram.device)
    a = torch.linalg.solve(system[:, None], (root * b)[..., None]).squeeze(-1)
    return dict(b=b, a=a, d=root * a, gram=gram)


def sweep_coefficients(keys: Tensor, q: Tensor, lam: Tensor | float, sweeps: int) -> dict[str, Tensor]:
    """Float64 coefficients d of the sweep_final read m = sum_j d_j v_j, by the dual form of the delta rule.

    The state is M = sum_i v_i c_i^T; one beta = 1 write of (k, v_j) maps every c_i to  lam c_i - (lam c_i . k) k  and adds
    k to c_j. There is no separate pre-decay coefficient: a = d.
    """
    keys, q = keys.double(), q.double()
    batch, n, width = keys.shape
    lam = torch.as_tensor(lam, dtype=torch.float64, device=keys.device)
    eye = torch.eye(n, dtype=keys.dtype, device=keys.device)
    coefficients = torch.zeros(batch, n, width, dtype=keys.dtype, device=keys.device)
    for _ in range(sweeps):
        for j in range(n):
            coefficients = lam * coefficients
            overlap = torch.einsum("bir,br->bi", coefficients, keys[:, j])
            coefficients = coefficients + (eye[j] - overlap)[..., None] * keys[:, j, None, :]
    d = torch.einsum("bnr,bqr->bqn", coefficients, q)
    return dict(b=torch.einsum("bnr,bqr->bqn", keys, q), a=d, d=d, gram=keys @ keys.transpose(1, 2))


def gram_conditions(keys: Tensor, lam: Tensor | float | None = None, eta: float | None = None,
                    floor: float = 1e-30) -> dict[str, Tensor]:
    """Finite float64 condition numbers (largest / smallest eigenvalue, the smallest floored) per episode.

    keys [B,n,r]. 'gram' is the condition of K^T K; with eta, 'ridge' is that of the ridge system
    W^1/2 K^T K W^1/2 + eta I (W = I when lam is None).
    """
    keys = keys.double()
    n = keys.shape[1]
    gram = keys @ keys.transpose(1, 2)

    def condition(matrix: Tensor) -> Tensor:
        eigenvalues = torch.linalg.eigvalsh(matrix)
        return eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(floor)

    result = dict(gram=condition(gram))
    if eta is not None:
        root = torch.ones(n, dtype=keys.dtype, device=keys.device)
        if lam is not None:
            lam = torch.as_tensor(lam, dtype=torch.float64, device=keys.device)
            root = (lam ** torch.arange(n - 1, -1, -1, device=keys.device, dtype=torch.float64)).sqrt()
        system = root[:, None] * gram * root[None, :] + eta * torch.eye(n, dtype=keys.dtype, device=keys.device)
        result["ridge"] = condition(system)
    return result
