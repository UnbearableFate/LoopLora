"""Independent, unbatched token semantics from the research plan's Appendix A."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def recurrent_adapter_step(
    x: Tensor,
    M_prev: Tensor | None,
    p: Any,
    loop_steps: int,
    *,
    valid: bool = True,
    reset: bool = False,
    write_enabled: bool = True,
) -> tuple[Tensor, Tensor | None, dict[str, int]]:
    """Read a pre-write snapshot, then optionally apply one decay-first write.

    ``p`` supplies mathematical-shape matrices and a frozen ``W``/optional
    ``bias``. It need not be an adapter module. No batched adapter or memory
    helpers are used, so this remains an independent oracle for those paths.
    """
    if isinstance(loop_steps, bool) or not isinstance(loop_steps, int) or loop_steps < 0:
        raise ValueError("loop_steps must be a nonnegative integer")
    variant = getattr(p, "variant", "iterq")
    if variant not in {"lora", "parallel", "loop", "iterq", "fixedq", "unshared", "double_ba"}:
        raise ValueError(f"unknown adapter variant: {variant}")
    activation = getattr(p, "activation", "relu")
    if activation not in {"relu", "identity"}:
        raise ValueError("activation must be 'relu' or 'identity'")
    phi = torch.relu if activation == "relu" else lambda value: value
    has_memory = variant in {"iterq", "fixedq", "unshared"}
    if variant == "unshared" and loop_steps != len(p.R):
        raise ValueError("unshared reader requires its configured number of steps")

    old = M_prev
    if old is None and has_memory:
        old = x.new_zeros((p.K.shape[0], p.K.shape[0]))
    if reset and old is not None:
        old = torch.zeros_like(old)

    base = p.W @ x
    bias = getattr(p, "bias", None)
    if bias is not None:
        base = base + bias
    if not valid:
        return base, old, {"reads": 0, "writes": 0}

    u = p.A @ x
    z = u
    reads = 0
    if variant == "parallel":
        z = phi(u)
    elif variant == "double_ba":
        extra = p.A @ phi(p.B @ u)
        z = u + getattr(p, "eta", 1.0) * extra if getattr(p, "double_ba_form", "skip") == "skip" else extra
    elif variant in {"loop", "iterq", "fixedq", "unshared"}:
        cached = None
        if variant == "fixedq" and loop_steps:
            cached = p.U @ (old @ (p.Q @ u).to(old.dtype)).to(u.dtype)
            reads = 1
        for step in range(loop_steps):
            R = p.R[step] if variant == "unshared" else p.R
            h = R @ z
            if variant in {"iterq", "unshared"}:
                Q = p.Q[step] if variant == "unshared" else p.Q
                U = p.U[step] if variant == "unshared" else p.U
                m = old @ (Q @ z).to(old.dtype)
                h = h + U @ m.to(u.dtype)
                reads += 1
            elif variant == "fixedq":
                h = h + cached
            z = u + p.rho * phi(h)

    branch = p.alpha * (p.B @ z)
    if variant == "double_ba" and getattr(p, "double_ba_form", "skip") == "gated":
        branch = p.g * branch
    y = base + branch
    if not has_memory or not write_enabled:
        return y, old, {"reads": reads, "writes": 0}

    if getattr(p, "gate", "scalar") == "input":
        lam = torch.sigmoid(p.w_lambda @ u + p.a_lambda)
        beta = torch.sigmoid(p.w_beta @ u + p.a_beta)
    else:
        lam, beta = p.lam, p.beta
    k_raw = p.K @ u
    k = (k_raw / torch.linalg.vector_norm(k_raw).clamp_min(p.eps)).to(old.dtype)
    v = (p.V @ u).to(old.dtype)
    lam = torch.as_tensor(lam, dtype=old.dtype, device=old.device)
    beta = torch.as_tensor(beta, dtype=old.dtype, device=old.device)
    decayed = lam * old
    error = v - decayed @ k
    state = decayed + beta * torch.outer(error, k)
    return y, state, {"reads": reads, "writes": 1}
