"""Binding adapter shapes and initialization from docs/notation.md."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

Variant = Literal["lora", "parallel", "loop", "iterq", "fixedq", "unshared", "double_ba"]
MEMORY_VARIANTS = frozenset({"iterq", "fixedq", "unshared"})


@dataclass(frozen=True)
class AdapterConfig:
    d_in: int
    d_out: int
    r: int
    variant: Variant = "iterq"
    r_mem: int | None = None
    rho: float = 0.25
    alpha: float = 1.0
    gate: Literal["scalar", "input"] = "scalar"
    lambda_init: float = 0.99
    beta_init: float = 0.1
    eps: float = 1e-6
    sigma_r: float = 0.5
    sigma_u: float = 0.5
    sigma_q: float = 1.0
    sigma_k: float = 1.0
    sigma_v: float = 1.0
    eta: float = 1.0
    unshared_steps: int = 1
    double_ba_form: Literal["skip", "gated"] = "skip"
    activation: Literal["relu", "identity"] = "relu"
    dpfp_nu: int = 0
    store: Literal["delta", "ridge", "sweep"] = "delta"
    store_eta: float = 1e-3
    store_sweeps: int = 4
    unit_query: bool = False

    def __post_init__(self) -> None:
        if self.r_mem is None:
            object.__setattr__(self, "r_mem", self.r)
        for name in ("d_in", "d_out", "r", "r_mem"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.variant not in {"lora", "parallel", "loop", *MEMORY_VARIANTS, "double_ba"}:
            raise ValueError(f"unknown variant: {self.variant!r}")
        if self.gate not in {"scalar", "input"}:
            raise ValueError("gate must be 'scalar' or 'input'")
        if self.activation not in {"relu", "identity"}:
            raise ValueError("activation must be 'relu' or 'identity'")
        if self.double_ba_form not in {"skip", "gated"}:
            raise ValueError("double_ba_form must be 'skip' or 'gated'")
        if self.variant == "double_ba" and self.d_in != self.d_out:
            raise ValueError("double_ba requires d_in == d_out")
        validate_steps(self.unshared_steps)
        if isinstance(self.dpfp_nu, bool) or not isinstance(self.dpfp_nu, int) or self.dpfp_nu < 0:
            raise ValueError("dpfp_nu must be a nonnegative integer")
        if self.dpfp_nu and self.variant not in MEMORY_VARIANTS:
            raise ValueError("dpfp_nu requires a memory variant")
        if self.store not in {"delta", "ridge", "sweep"}:
            raise ValueError("store must be 'delta', 'ridge' or 'sweep'")
        if not isinstance(self.unit_query, bool):
            raise ValueError("unit_query must be a bool")
        if isinstance(self.store_sweeps, bool) or not isinstance(self.store_sweeps, int) or self.store_sweeps < 1:
            raise ValueError("store_sweeps must be a positive integer")
        if isinstance(self.store_eta, bool) or not math.isfinite(self.store_eta) or self.store_eta <= 0:
            raise ValueError("store_eta must be finite and positive")
        if (self.store != "delta" or self.unit_query) and self.variant not in MEMORY_VARIANTS:
            raise ValueError("store and unit_query require a memory variant")
        if self.store != "delta" and (self.gate != "scalar" or self.dpfp_nu):
            raise ValueError("least-squares stores require scalar gates and no DPFP features")
        if self.unit_query and self.dpfp_nu:
            raise ValueError("unit_query is part of the DPFP query map; enable one of them")
        for name in ("lambda_init", "beta_init"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        for name in ("rho", "alpha", "eta", "eps", "sigma_r", "sigma_u", "sigma_q", "sigma_k", "sigma_v"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if any(getattr(self, name) < 0 for name in ("sigma_r", "sigma_u", "sigma_q", "sigma_k", "sigma_v")):
            raise ValueError("initialization scales must be nonnegative")

    @property
    def key_dim(self) -> int:
        """Columns of the memory M: r_mem, or 2 * r_mem * dpfp_nu with DPFP address features."""
        return 2 * self.r_mem * self.dpfp_nu if self.dpfp_nu else self.r_mem


def validate_steps(loop_steps: int) -> None:
    if isinstance(loop_steps, bool) or not isinstance(loop_steps, int) or loop_steps < 0:
        raise ValueError("loop_steps must be a nonnegative integer")
