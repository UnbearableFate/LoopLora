"""Dominant operation, trainable parameter, and persistent state accounting."""
from __future__ import annotations

import torch

from .config import AdapterConfig


_MEMORY_VARIANTS = {"iterq", "fixedq", "unshared"}


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def param_count(cfg: AdapterConfig) -> int:
    """Count adapter parameters, including trainable gates but excluding the base."""
    total = cfg.r * (cfg.d_in + cfg.d_out)
    if cfg.variant == "loop":
        total += cfg.r**2
    elif cfg.variant in _MEMORY_VARIANTS:
        steps = cfg.unshared_steps if cfg.variant == "unshared" else 1
        # R/U/Q belong to each unrolled step; K/V and gates are shared.
        total += steps * (cfg.r**2 + 2 * cfg.r * cfg.r_mem)
        total += 2 * cfg.r * cfg.r_mem + 2
        if cfg.gate == "input":
            total += 2 * cfg.r
    elif cfg.variant == "double_ba" and cfg.double_ba_form == "gated":
        total += 1
    return total


def macs_per_token(
    cfg: AdapterConfig, loop_steps: int, writes: bool | int = True
) -> int:
    """Count dominant MACs (one MAC = two FLOPs), excluding the frozen base.

    ``writes`` is a nonnegative write multiplicity; bools select zero or one.
    Normalization, nonlinearities, scalar gates and elementwise additions are
    omitted. Input gates contribute their two length-r dot products per write.
    """
    steps = _nonnegative_int(loop_steps, "loop_steps")
    n_writes = int(writes) if isinstance(writes, bool) else _nonnegative_int(writes, "writes")
    if cfg.variant == "unshared" and steps != cfg.unshared_steps:
        raise ValueError("unshared loop_steps must equal cfg.unshared_steps")
    base = cfg.r * (cfg.d_in + cfg.d_out)
    if cfg.variant == "double_ba":
        return 2 * base
    if cfg.variant == "loop":
        return base + steps * cfg.r**2
    if cfg.variant not in _MEMORY_VARIANTS:
        return base
    read = 2 * cfg.r * cfg.r_mem + cfg.r_mem * cfg.key_dim
    if cfg.variant == "fixedq":
        reader = steps * cfg.r**2 + (read if steps else 0)
    else:
        reader = steps * (cfg.r**2 + read)
    writer = 2 * cfg.r * cfg.r_mem + 2 * cfg.r_mem * cfg.key_dim
    if cfg.gate == "input":
        writer += 2 * cfg.r
    return base + reader + n_writes * writer


def state_bytes(
    cfg: AdapterConfig, batch: int, n_modules: int, dtype: torch.dtype
) -> int:
    """Persistent delta-memory bytes; excludes working vectors and backbone KV."""
    batch = _nonnegative_int(batch, "batch")
    n_modules = _nonnegative_int(n_modules, "n_modules")
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    if cfg.variant not in _MEMORY_VARIANTS:
        return 0
    return n_modules * batch * cfg.r_mem * cfg.key_dim * dtype.itemsize
