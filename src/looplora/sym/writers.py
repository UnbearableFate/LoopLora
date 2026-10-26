"""Fact-only oracle and learned writers using the core decay-first rule."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.core.memory import delta_update, write, zeros_state
from looplora.infra.seeds import derive_seed


class OracleWriter(nn.Module):
    def __init__(self, V: int, r_mem: int, seed: int, orthogonal: bool = False) -> None:
        super().__init__()
        self.V, self.r_mem, self.seed, self.orthogonal = V, r_mem, seed, orthogonal
        generator = torch.Generator().manual_seed(derive_seed(seed, "oracle", "global_codes"))
        codes = F.normalize(torch.randn(V + 1, r_mem, generator=generator), dim=-1)
        self.register_buffer("global_codes", codes)

    def codes(self, facts: Tensor, valid: Tensor, episode_seed: np.ndarray,
              include_insufficient: bool = False,
              extra_entities: list[list[int]] | None = None) -> Tensor:
        if not self.orthogonal:
            return self.global_codes.unsqueeze(0).expand(facts.shape[0], -1, -1)
        # Stage each batch once: per-episode CUDA indexing/unique and basis
        # assignment otherwise force many small host/device round trips.
        facts_cpu, valid_cpu = facts.cpu(), valid.cpu()
        global_codes_cpu = self.global_codes.cpu()
        result = []
        # Keep the original per-episode RNG/QR order and entity slot ordering.
        for i, episode in enumerate(episode_seed):
            entities = facts_cpu[i, valid_cpu[i]][:, (0, 2)].unique()
            if include_insufficient:
                entities = torch.cat((entities, torch.tensor([self.V], dtype=entities.dtype))).unique()
            if extra_entities is not None:
                original = set(entities.tolist())
                added = sorted(set(extra_entities[i]) - original)
                entities = torch.cat((entities, torch.tensor(added, dtype=entities.dtype)))
            if entities.numel() > self.r_mem:
                raise ValueError("oracle_orth requires episode entities <= r_mem")
            generator = torch.Generator().manual_seed(derive_seed(self.seed, "oracle_orth", int(episode)))
            basis, _ = torch.linalg.qr(torch.randn(self.r_mem, self.r_mem, generator=generator))
            table = global_codes_cpu.clone()
            table[entities] = basis[:, :len(entities)].T
            result.append(table)
        return torch.stack(result).to(self.global_codes.device)

    def forward(self, facts: Tensor, valid: Tensor, codes: Tensor) -> Tensor:
        state = zeros_state(len(facts), self.r_mem, codes.dtype, codes.device)
        rows = torch.arange(len(facts), device=facts.device)
        for t in range(facts.shape[1]):
            key = codes[rows, facts[:, t, 0]]
            value = codes[rows, facts[:, t, 2]]
            state = delta_update(state, key, value, valid[:, t], lam=1.0, beta=1.0)
        return state


class LearnedWriter(nn.Module):
    def __init__(self, d_in: int, r: int, r_mem: int, seed: int,
                 lambda_init: float = .99, beta_init: float = .1) -> None:
        super().__init__()
        self.r_mem = r_mem
        generator = torch.Generator().manual_seed(derive_seed(seed, "learned_writer"))
        self.A = nn.Parameter(torch.empty(r, d_in))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)
        self.K = nn.Parameter(torch.randn(r_mem, r, generator=generator) / r**.5)
        self.V = nn.Parameter(torch.randn(r_mem, r, generator=generator) / r**.5)
        self.a_lambda = nn.Parameter(torch.tensor(math.log(lambda_init / (1 - lambda_init))))
        self.a_beta = nn.Parameter(torch.tensor(math.log(beta_init / (1 - beta_init))))

    def u(self, x: Tensor) -> Tensor:
        return F.linear(x, self.A)

    def forward(self, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        u = self.u(x)
        state = zeros_state(len(x), self.r_mem, x.dtype, x.device)
        for t in range(x.shape[1]):
            state = write(state, u[:, t], valid[:, t], K=self.K, V=self.V,
                          lam=self.a_lambda.sigmoid(), beta=self.a_beta.sigmoid(), eps=1e-6)
        return state, F.linear(u, self.V)


class FactorizedWriter(nn.Module):
    """Tail-blind keys/queries and head/relation-blind values, by construction."""

    def __init__(self, d_in: int, r: int, r_mem: int, seed: int,
                 lambda_init: float = .99, beta_init: float = .1) -> None:
        super().__init__()
        if d_in < 3 or d_in % 3:
            raise ValueError("factorized writer requires d_in = 3 * d_e")
        self.d_e, self.r_mem = d_in // 3, r_mem
        # Reuse the exact seeded initialization stream, including K/V and gates.
        initial = LearnedWriter(d_in, r, r_mem, seed, lambda_init, beta_init)
        self.A_k = nn.Parameter(initial.A[:, :2 * self.d_e].detach().clone())
        self.A_v = nn.Parameter(initial.A[:, 2 * self.d_e:].detach().clone())
        self.K, self.V = initial.K, initial.V
        self.a_lambda, self.a_beta = initial.a_lambda, initial.a_beta

    def u(self, x: Tensor) -> Tensor:
        return F.linear(x[..., :2 * self.d_e], self.A_k)

    def forward(self, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        key_raw = F.linear(self.u(x), self.K)
        keys = key_raw / torch.linalg.vector_norm(key_raw, dim=-1, keepdim=True).clamp_min(1e-6)
        values = F.linear(F.linear(x[..., 2 * self.d_e:], self.A_v), self.V)
        state = zeros_state(len(x), self.r_mem, x.dtype, x.device)
        lam, beta = self.a_lambda.sigmoid(), self.a_beta.sigmoid()
        for t in range(x.shape[1]):
            state = delta_update(state, keys[:, t], values[:, t], valid[:, t], lam, beta)
        return state, values
