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
              include_insufficient: bool = False) -> Tensor:
        if not self.orthogonal:
            return self.global_codes.unsqueeze(0).expand(facts.shape[0], -1, -1)
        result = []
        # QR on CPU makes the fixed episode codebook independent of GPU lanes.
        for i, episode in enumerate(episode_seed):
            entities = facts[i, valid[i]][:, (0, 2)].unique().cpu()
            if include_insufficient:
                entities = torch.cat((entities, torch.tensor([self.V], dtype=entities.dtype))).unique()
            if entities.numel() > self.r_mem:
                raise ValueError("oracle_orth requires episode entities <= r_mem")
            generator = torch.Generator().manual_seed(derive_seed(self.seed, "oracle_orth", int(episode)))
            basis, _ = torch.linalg.qr(torch.randn(self.r_mem, len(entities), generator=generator))
            table = self.global_codes.clone()
            table[entities.to(table.device)] = basis.T.to(table.device)
            result.append(table)
        return torch.stack(result)

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
