"""One fixed symbolic backbone, a fact writer, and one core reader."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.core import AdapterConfig, make_adapter
from looplora.data.relations import EpisodeArrays, TaskSpec
from looplora.infra.seeds import derive_seed
from .backbone import SymbolicBackbone
from .writers import LearnedWriter, OracleWriter


@dataclass
class PreparedBatch:
    x: Tensor
    u: Tensor
    memory: Tensor
    answer: Tensor
    candidate_mask: Tensor
    codes: Tensor
    true_value: Tensor
    retrieval_valid: Tensor
    write_count: Tensor


def make_task_spec(config: dict[str, Any], evaluation: bool = False) -> TaskSpec:
    fields = dict(config["task"])
    if evaluation and fields["name"] != "t1" and fields.get("base") != "t1":
        fields["hops"] = tuple(config["eval"]["hops"])
        fields["Q"] = len(fields["hops"])
    return TaskSpec(**fields)


class SymbolicModel(nn.Module):
    def __init__(self, config: dict[str, Any], *, reader_label: str = "reader") -> None:
        super().__init__()
        self.config = config
        self.writer_condition = config["writer"]
        self.variant = config["variant"]
        m = config["model"]
        spec = make_task_spec(config)
        self.r_mem, self.r = m["r_mem"], m["r"]
        self.backbone = SymbolicBackbone(spec.V, spec.n_rel, m["d_e"], m["max_hop"],
                                         config["backbone_seed"])
        oracle = self.writer_condition.startswith("oracle_")
        if oracle and self.r != self.r_mem:
            raise ValueError("oracle query codes require r == r_mem")
        if oracle:
            self.writer = OracleWriter(spec.V, self.r_mem, config["backbone_seed"],
                                       orthogonal=self.writer_condition == "oracle_orth")
        else:
            self.writer = LearnedWriter(3 * m["d_e"], self.r, self.r_mem, config["seed"],
                                        m["lambda_init"], m["beta_init"])
        core_variant = {"B3": "loop", "B5": "fixedq", "B8": "unshared"}.get(self.variant, "iterq")
        cfg = AdapterConfig(d_in=self.r if oracle else 3 * m["d_e"],
                            d_out=self.r_mem if oracle else m["d_e"], r=self.r, r_mem=self.r_mem,
                            variant=core_variant, rho=m["rho"], alpha=m["alpha"],
                            activation=config["activation"], unshared_steps=4,
                            lambda_init=m["lambda_init"], beta_init=m["beta_init"])
        # Core initializes from the torch RNG. Preserve the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed(config["seed"], reader_label, self.variant, config["activation"]))
            self.adapter = make_adapter(cfg)
        for name in ("A", "K", "V", "a_lambda", "a_beta"):
            if not hasattr(self.adapter, name):
                continue
            if oracle:
                getattr(self.adapter, name).requires_grad_(False)
            else:
                setattr(self.adapter, name, getattr(self.writer, name))
        self.to(config["device"])

    @property
    def device(self) -> torch.device:
        return self.adapter.B.device

    def freeze_writer(self) -> None:
        self.writer.requires_grad_(False)

    def writer_signature(self) -> dict[str, Any]:
        return {"r": self.r, "r_mem": self.r_mem, "d_e": self.backbone.d_e,
                "V": self.backbone.V, "n_rel": len(self.backbone.relations),
                "backbone_seed": self.config["backbone_seed"]}

    def prepare(self, arrays: EpisodeArrays) -> PreparedBatch:
        facts = torch.as_tensor(arrays.facts, dtype=torch.long, device=self.device)
        valid = torch.as_tensor(arrays.fact_valid, dtype=torch.bool, device=self.device)
        facts = torch.where(valid[..., None], facts, torch.zeros_like(facts))
        start = torch.as_tensor(arrays.q_start, dtype=torch.long, device=self.device)
        rel = torch.as_tensor(arrays.q_rel, dtype=torch.long, device=self.device)
        hop = torch.as_tensor(arrays.q_hop_given, dtype=torch.long, device=self.device)
        x = self.backbone.queries(start, rel, hop)
        answer = torch.as_tensor(arrays.answer, dtype=torch.long, device=self.device)
        B, Q = start.shape
        rows = torch.arange(B, device=self.device)
        insufficient = self.config["task"]["name"] == "t3_missing"
        candidate = torch.zeros(B, self.backbone.V + 1, dtype=torch.bool, device=self.device)
        # scatter_add excludes padding without accidentally clearing repeated ids.
        counts = torch.zeros_like(candidate, dtype=torch.long)
        for column in (0, 2):
            counts.scatter_add_(1, facts[..., column], valid.long())
        candidate = counts > 0
        if insufficient:
            candidate[:, self.backbone.V] = True
        if isinstance(self.writer, OracleWriter):
            codes = self.writer.codes(facts, valid, arrays.episode_seed, insufficient)
            memory = self.writer(facts, valid, codes)
            u = codes[rows[:, None], start]
            values = codes[rows[:, None], facts[..., 2]]
        else:
            fact_x = self.backbone.facts(facts)
            fact_x = torch.where(valid[..., None], fact_x, torch.zeros_like(fact_x))
            memory, values = self.writer(fact_x, valid)
            u = self.writer.u(x)
            codes = self.backbone.entities.unsqueeze(0).expand(B, -1, -1)
        matches = ((facts[:, None, :, 0] == start[..., None]) &
                   (facts[:, None, :, 1] == rel[..., None]) & valid[:, None, :])
        positions = torch.arange(facts.shape[1], device=self.device)
        last = torch.where(matches, positions, -1).amax(dim=-1)
        true_value = values[rows[:, None], last.clamp_min(0)]
        return PreparedBatch(x, u, memory, answer, candidate, codes, true_value,
                             last >= 0, valid.sum(dim=1))

    def read(self, batch: PreparedBatch, loop_steps: int) -> dict[str, Tensor]:
        memory = batch.memory[:, None].expand(-1, batch.u.shape[1], -1, -1)
        z = self.adapter.read(batch.u, memory, loop_steps)
        output = self.adapter.branch(z)
        logits = torch.einsum("bqd,bvd->bqv", output, batch.codes)
        cosine = torch.full_like(batch.answer, float("nan"), dtype=z.dtype)
        if self.adapter.has_memory and loop_steps:
            Q = self.adapter.Q[0] if self.variant == "B8" else self.adapter.Q
            query = F.linear(batch.u, Q)
            first_read = torch.einsum("bij,bqj->bqi", batch.memory, query)
            value = F.cosine_similarity(first_read, batch.true_value, dim=-1)
            cosine = torch.where(batch.retrieval_valid, value, cosine)
        return {"logits": logits, "z": z, "first_read_cosine": cosine}

    def forward(self, arrays: EpisodeArrays, loop_steps: int) -> tuple[Tensor, PreparedBatch]:
        batch = self.prepare(arrays)
        return self.read(batch, loop_steps)["logits"], batch
