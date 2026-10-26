"""Low-rank adapters with functional, causal per-sequence memory semantics."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import AdapterConfig, validate_steps
from . import memory


class AdapterBase(nn.Module):
    """Common projection and sequence interface; initialization uses caller RNG."""

    has_memory = False

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.A = nn.Parameter(torch.empty(cfg.r, cfg.d_in))
        self.B = nn.Parameter(torch.zeros(cfg.d_out, cfg.r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def _normal(self, rows: int, cols: int, sigma: float) -> nn.Parameter:
        parameter = nn.Parameter(self.A.new_empty(rows, cols))
        nn.init.normal_(parameter, std=sigma / math.sqrt(cols))
        return parameter

    def _phi(self, z: Tensor) -> Tensor:
        return torch.relu(z) if self.cfg.activation == "relu" else z

    def _check_steps(self, loop_steps: int) -> None:
        validate_steps(loop_steps)

    def u(self, x: Tensor) -> Tensor:
        return F.linear(x, self.A)

    def read(self, u: Tensor, M_prev: Tensor | None, loop_steps: int) -> Tensor:
        self._check_steps(loop_steps)
        if u.ndim not in {2, 3} or u.shape[-1] != self.cfg.r:
            raise ValueError("reader u must have shape [B,r] or [B,T,r]")
        if self.has_memory and loop_steps:
            shape = (*u.shape[:-1], self.cfg.r_mem, self.cfg.r_mem)
            if M_prev is None or M_prev.shape != shape or M_prev.device != u.device:
                raise ValueError(f"reader state must have shape {shape} on the input device")
        return self._read(u, M_prev, loop_steps)

    def _read(self, u: Tensor, M_prev: Tensor | None, loop_steps: int) -> Tensor:
        return u

    def branch(self, z: Tensor) -> Tensor:
        return self.cfg.alpha * F.linear(z, self.B)

    def _reads_per_token(self, loop_steps: int) -> int:
        return loop_steps if self.has_memory else 0

    def _prepare(
        self, x: Tensor, M0: Tensor | None, loop_steps: int,
        valid: Tensor | None, write: Tensor | None, reset: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor, Tensor]:
        self._check_steps(loop_steps)
        if x.ndim != 3 or x.shape[-1] != self.cfg.d_in:
            raise ValueError("sequence input must have shape [B,T,d_in]")
        shape = tuple(x.shape[:2])
        valid_mask = memory._mask(valid, shape, x.device, True)
        write_mask = memory._mask(write, shape, x.device, True)
        reset_mask = memory._mask(reset, shape, x.device, False)
        if M0 is None and self.has_memory:
            M0 = memory.zeros_state(x.shape[0], self.cfg.r_mem, x.dtype, x.device)
        if M0 is not None:
            memory._check_state(M0)
            if M0.shape != (x.shape[0], self.cfg.r_mem, self.cfg.r_mem) or M0.device != x.device:
                raise ValueError("M0 shape/device does not match this sequence and adapter")
        # Undefined padding values must not enter parameter-gradient products.
        projected = self.u(torch.where(valid_mask[..., None], x, torch.zeros_like(x)))
        return projected, M0, valid_mask, write_mask, reset_mask

    def _masked_branch(
        self, u: Tensor, state: Tensor | None, loop_steps: int, valid: Tensor,
    ) -> Tensor:
        flat_u = u.reshape(-1, self.cfg.r)
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        output = u.new_zeros((flat_u.shape[0], self.cfg.d_out))
        if indices.numel():
            selected_state = None
            if state is not None:
                selected_state = state.reshape(-1, self.cfg.r_mem, self.cfg.r_mem).index_select(0, indices)
            z = self.read(flat_u.index_select(0, indices), selected_state, loop_steps)
            values = self.branch(z)
            output = output.to(values.dtype).index_copy(0, indices, values)
        return output.reshape(*u.shape[:-1], self.cfg.d_out)

    def _stats(self, valid: Tensor, write: Tensor, loop_steps: int) -> dict[str, Tensor]:
        reads = valid.sum(dim=1) * self._reads_per_token(loop_steps)
        writes = (valid & write).sum(dim=1) if self.has_memory else torch.zeros_like(reads)
        return {"reads": reads, "writes": writes}

    def process_sequence(
        self, x: Tensor, M0: Tensor | None = None, loop_steps: int = 1,
        valid: Tensor | None = None, write: Tensor | None = None, reset: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        """Read each token from its post-reset, pre-write snapshot, then write once."""
        u, state, valid, write, reset = self._prepare(x, M0, loop_steps, valid, write, reset)
        outputs = []
        for t in range(x.shape[1]):
            if state is not None:
                state = torch.where(reset[:, t, None, None], torch.zeros_like(state), state)
            outputs.append(self._masked_branch(u[:, t], state, loop_steps, valid[:, t]))
            if self.has_memory:
                state = self.write(state, u[:, t], valid[:, t] & write[:, t])
        branch = torch.stack(outputs, dim=1) if outputs else u.new_empty((x.shape[0], 0, self.cfg.d_out))
        return branch, state, self._stats(valid, write, loop_steps)

    def process_sequence_fast(
        self, x: Tensor, M0: Tensor | None = None, loop_steps: int = 1,
        valid: Tensor | None = None, write: Tensor | None = None, reset: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        """Scan writer states, then batch all valid-token readers in one call.

        The reference scan is sequential and stores every prefix state; this
        method promises equivalence, not a fused scan or a speed improvement.
        """
        u, state, valid, write, reset = self._prepare(x, M0, loop_steps, valid, write, reset)
        snapshots = None
        if self.has_memory:
            snapshots, state = self.scan_prewrite(state, u, valid, write, reset)
        elif state is not None:
            state = torch.where(reset.any(dim=1)[:, None, None], torch.zeros_like(state), state)
        branch = self._masked_branch(u, snapshots, loop_steps, valid)
        return branch, state, self._stats(valid, write, loop_steps)


class LoRAAdapter(AdapterBase):
    pass


class ParallelAdapter(AdapterBase):
    def _read(self, u: Tensor, M_prev: Tensor | None, loop_steps: int) -> Tensor:
        return self._phi(u)


class LoopAdapter(AdapterBase):
    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)
        self.R = self._normal(cfg.r, cfg.r, cfg.sigma_r)

    def _read(self, u: Tensor, M_prev: Tensor | None, loop_steps: int) -> Tensor:
        z = u
        for _ in range(loop_steps):
            z = u + self.cfg.rho * self._phi(F.linear(z, self.R))
        return z


class MemoryAdapter(AdapterBase):
    has_memory = True

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)
        self.K = self._normal(cfg.r_mem, cfg.r, cfg.sigma_k)
        self.V = self._normal(cfg.r_mem, cfg.r, cfg.sigma_v)
        self.a_lambda = nn.Parameter(self.A.new_tensor(math.log(cfg.lambda_init / (1 - cfg.lambda_init))))
        self.a_beta = nn.Parameter(self.A.new_tensor(math.log(cfg.beta_init / (1 - cfg.beta_init))))
        if cfg.gate == "input":
            self.w_lambda = nn.Parameter(self.A.new_zeros(cfg.r))
            self.w_beta = nn.Parameter(self.A.new_zeros(cfg.r))

    @property
    def lam(self) -> Tensor:
        return self.a_lambda.sigmoid()

    @property
    def beta(self) -> Tensor:
        return self.a_beta.sigmoid()

    def gates(self, u: Tensor) -> tuple[Tensor, Tensor]:
        if self.cfg.gate == "input":
            return (torch.sigmoid(u @ self.w_lambda + self.a_lambda),
                    torch.sigmoid(u @ self.w_beta + self.a_beta))
        return self.lam, self.beta

    def write(self, M_prev: Tensor, u: Tensor, write_mask: Tensor | None) -> Tensor:
        lam, beta = self.gates(u)
        return memory.write(M_prev, u, write_mask, K=self.K, V=self.V, lam=lam, beta=beta, eps=self.cfg.eps)

    def scan_prewrite(
        self, M0: Tensor, u: Tensor, valid: Tensor | None = None,
        write: Tensor | None = None, reset: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        lam, beta = self.gates(u)
        return memory.scan_prewrite(M0, u, valid, write, reset, K=self.K, V=self.V,
                                    lam=lam, beta=beta, eps=self.cfg.eps)

    def scan_final(
        self, M0: Tensor, u: Tensor, valid: Tensor | None = None,
        write: Tensor | None = None, reset: Tensor | None = None,
    ) -> Tensor:
        lam, beta = self.gates(u)
        return memory.scan_final(M0, u, valid, write, reset, K=self.K, V=self.V,
                                 lam=lam, beta=beta, eps=self.cfg.eps)

    @staticmethod
    def _query(u: Tensor, M: Tensor, Q: Tensor, U: Tensor) -> Tensor:
        q = F.linear(u, Q)
        m = torch.matmul(M, q.to(M.dtype).unsqueeze(-1)).squeeze(-1)
        return F.linear(m.to(u.dtype), U)


class IterativeQueryAdapter(MemoryAdapter):
    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)
        self.R = self._normal(cfg.r, cfg.r, cfg.sigma_r)
        self.U = self._normal(cfg.r, cfg.r_mem, cfg.sigma_u)
        self.Q = self._normal(cfg.r_mem, cfg.r, cfg.sigma_q)

    def _read(self, u: Tensor, M_prev: Tensor | None, loop_steps: int) -> Tensor:
        z = u
        for _ in range(loop_steps):
            z = u + self.cfg.rho * self._phi(F.linear(z, self.R) + self._query(z, M_prev, self.Q, self.U))
        return z


class FixedQueryAdapter(IterativeQueryAdapter):
    def _read(self, u: Tensor, M_prev: Tensor | None, loop_steps: int) -> Tensor:
        z = u
        if loop_steps:
            readout = self._query(u, M_prev, self.Q, self.U)
            for _ in range(loop_steps):
                z = u + self.cfg.rho * self._phi(F.linear(z, self.R) + readout)
        return z

    def _reads_per_token(self, loop_steps: int) -> int:
        return int(loop_steps > 0)


class UnsharedAdapter(MemoryAdapter):
    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)
        self.R = nn.ParameterList([self._normal(cfg.r, cfg.r, cfg.sigma_r) for _ in range(cfg.unshared_steps)])
        self.U = nn.ParameterList([self._normal(cfg.r, cfg.r_mem, cfg.sigma_u) for _ in range(cfg.unshared_steps)])
        self.Q = nn.ParameterList([self._normal(cfg.r_mem, cfg.r, cfg.sigma_q) for _ in range(cfg.unshared_steps)])

    def _check_steps(self, loop_steps: int) -> None:
        super()._check_steps(loop_steps)
        if loop_steps != self.cfg.unshared_steps:
            raise ValueError("unshared reader requires loop_steps == unshared_steps")

    def _read(self, u: Tensor, M_prev: Tensor | None, loop_steps: int) -> Tensor:
        z = u
        for R, U, Q in zip(self.R, self.U, self.Q):
            z = u + self.cfg.rho * self._phi(F.linear(z, R) + self._query(z, M_prev, Q, U))
        return z


class DoubleBAAdapter(AdapterBase):
    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)
        if cfg.d_in != cfg.d_out:
            raise ValueError("double_ba requires d_in == d_out")
        if cfg.double_ba_form == "gated":
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
            self.g = nn.Parameter(self.A.new_zeros(()))

    def _read(self, u: Tensor, M_prev: Tensor | None, loop_steps: int) -> Tensor:
        inner = F.linear(self._phi(F.linear(u, self.B)), self.A)
        return u + self.cfg.eta * inner if self.cfg.double_ba_form == "skip" else inner

    def branch(self, z: Tensor) -> Tensor:
        result = super().branch(z)
        return self.g * result if self.cfg.double_ba_form == "gated" else result


def make_adapter(cfg: AdapterConfig) -> AdapterBase:
    constructors = {
        "lora": LoRAAdapter, "parallel": ParallelAdapter, "loop": LoopAdapter,
        "iterq": IterativeQueryAdapter, "fixedq": FixedQueryAdapter,
        "unshared": UnsharedAdapter, "double_ba": DoubleBAAdapter,
    }
    return constructors[cfg.variant](cfg)
