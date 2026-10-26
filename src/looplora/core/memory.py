"""Functional decay-first memory writes and pre-write snapshots."""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F


def zeros_state(batch: int, r_mem: int, dtype: torch.dtype, device: torch.device | str) -> Tensor:
    if batch < 0 or r_mem <= 0:
        raise ValueError("batch must be nonnegative and r_mem positive")
    return torch.zeros(batch, r_mem, r_mem, dtype=dtype, device=device)


def fork(M: Tensor, n: int) -> Tensor:
    """Return [B*n, r_mem, r_mem], grouping n independent copies per sample."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    _check_state(M)
    return M.repeat_interleave(n, dim=0)


def _check_state(M: Tensor) -> None:
    if M.ndim != 3 or M.shape[-1] != M.shape[-2] or not M.is_floating_point():
        raise ValueError("state must be a floating tensor [B, r_mem, r_mem]")


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
    if k.shape != M.shape[:2] or v.shape != k.shape or k.device != M.device or v.device != M.device:
        raise ValueError("k and v must have shape [B, r_mem] on the state device")
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
) -> Tensor:
    """Project and normalize one token's writer codes, then apply delta_update."""
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
    beta: Tensor | float, eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Sequential functional scan; output t is after reset, before token t writes.

    lam/beta can be scalars or [B,T] input-dependent gate probabilities. This
    reference scan stores O(B*T*r_mem**2) state, without dense affine products.
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
    state = M0
    snapshots = []
    for t in range(u.shape[1]):
        state = torch.where(reset_mask[:, t, None, None], torch.zeros_like(state), state)
        snapshots.append(state)
        lam_t = lam[:, t] if isinstance(lam, Tensor) and lam.numel() != 1 else lam
        beta_t = beta[:, t] if isinstance(beta, Tensor) and beta.numel() != 1 else beta
        state = _write_token(
            state, u[:, t], valid_mask[:, t] & write_mask[:, t],
            K=K, V=V, lam=lam_t, beta=beta_t, eps=eps,
        )
    if not snapshots:
        return M0.new_empty((M0.shape[0], 0, *M0.shape[1:])), state
    return torch.stack(snapshots, dim=1), state


def scan_final(
    M0: Tensor, u: Tensor, valid: Tensor | None = None, write: Tensor | None = None,
    reset: Tensor | None = None, *, K: Tensor, V: Tensor, lam: Tensor | float,
    beta: Tensor | float, eps: float = 1e-6,
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
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if reset is not None and bool(reset_mask[:, 1:].any()):
        return scan_prewrite(M0, u, valid, write, reset, K=K, V=V,
                             lam=lam, beta=beta, eps=eps)[1]
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
    raw = F.linear(packed, K)
    keys = (raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(eps)).to(M0.dtype)
    values = F.linear(packed, V).to(M0.dtype)

    def pack_gate(value: Tensor | float) -> Tensor:
        gate = torch.as_tensor(value, dtype=M0.dtype, device=M0.device)
        if gate.numel() == 1:
            return gate.reshape(1, 1).expand(u.shape[0], maximum)
        # Undefined non-write gates must not enter arithmetic or gradients.
        return torch.where(enabled, gate.gather(1, positions), torch.zeros_like(enabled, dtype=M0.dtype))

    lambdas, betas = pack_gate(lam), pack_gate(beta)
    for step in range(maximum):
        mask = enabled[:, step, None, None]
        key, value = keys[:, step], values[:, step]
        active_state = torch.where(mask, state, torch.zeros_like(state))
        decayed = lambdas[:, step, None, None] * active_state
        error = value - torch.bmm(decayed, key.unsqueeze(-1)).squeeze(-1)
        updated = decayed + betas[:, step, None, None] * error.unsqueeze(-1) * key.unsqueeze(-2)
        state = torch.where(mask, updated, state)
    return state


# The public scan's `write` mask intentionally has the task-specified name.
_write_token = write
