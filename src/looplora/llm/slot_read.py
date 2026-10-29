"""SLOT-LOOP-v1: question-token reads through an explicit memory backend (docs/prereg/SLOT-LOOP-v1.md, plan 11).

`SlotFixAdapter` is the S-RE / D128-3 adapter (`FixAdapter`) with one addition: when the memory it is handed is a `MemoryRead`
(a backend, the memory of the B episodes and an owner per question row) it reads through that backend; a tensor state takes the
unchanged inherited path. The reads happen at the same places as before (every valid question token, L loops of the same
`_read`), the query code is the adapter's own `query_code`, and nothing is written: online writing, resets and any write mask
refuse. `ReadSpec` names one read condition; `bind_batch` swaps a prepared batch's matrix snapshot for the matching `MemoryRead`,
built from the keys and values the writer actually wrote (`FixBatch.keys/values`), never from `base_keys`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.core import memory
from looplora.core.memory_backend import (HardSlotBackend, MatrixBackend, MatrixState, MemoryRead, SoftSlotBackend,
                                          build_slots)
from .writer_fix import FixAdapter, FixBatch, true_fact_indices

KINDS = ("matrix", "soft", "hard", "clean_final")
SCOPES = ("all_valid_question", "final_only")


@dataclass(frozen=True)
class ScopedRead(MemoryRead):
    """Non-deployable two-state diagnostic of S0's localisation table.

    The replacement (``backend``, ``state``) serves only the final valid question token of each row, or, for the oracle
    (which uses the label), the true value of the row's fact; every other position reads the native matrix state in
    ``native``. A deployed backend never holds a second state: this exists for S0 only.
    """

    native: MemoryRead | None = None
    oracle: Tensor | None = None  # [B, dv]: the true value of each episode's single question (one question per episode)
    final: Tensor | None = None   # [n] bool, set per token by the adapter loop

    def check_fp32(self) -> None:
        super().check_fp32()
        self.native.check_fp32()
        if self.oracle is not None and self.oracle.dtype != torch.float32:
            raise ValueError("LLM adapter memory must be fp32")

    def expand(self, queries: int) -> "ScopedRead":
        if self.oracle is not None and queries != 1:
            raise ValueError("the oracle diagnostic is defined for one question per episode")
        return replace(self, owner=self.owner.repeat_interleave(queries), native=self.native.expand(queries))

    def select(self, rows: Tensor, final: Tensor | None = None) -> "ScopedRead":
        return replace(self, owner=self.owner.index_select(0, rows), native=self.native.select(rows), final=final)

    def read(self, query: Tensor) -> Tensor:
        if self.final is None:
            raise RuntimeError("a scoped read needs the final-token flag of its rows")
        native = self.native.read(query)
        replaced = (self.oracle.index_select(0, self.owner) if self.oracle is not None
                    else self.backend.read(self.state, query, self.owner))
        return torch.where(self.final[:, None], replaced.to(native.dtype), native)

    def clone(self) -> "ScopedRead":
        return replace(self, state=self.backend.clone_state(self.state), owner=self.owner.clone(),
                       native=self.native.clone(), oracle=None if self.oracle is None else self.oracle.clone())


class SlotReadMixin:
    """Read-only memory-QA through a MemoryRead; every other call is the inherited one."""

    def read(self, u: Tensor, M_prev, loop_steps: int) -> Tensor:
        if not isinstance(M_prev, MemoryRead):
            return super().read(u, M_prev, loop_steps)
        self._check_steps(loop_steps)
        if u.ndim != 2 or u.shape[-1] != self.cfg.r:
            raise ValueError("reader u must have shape [N,r]")
        if M_prev.rows != u.shape[0]:
            raise ValueError("a memory read needs one owner entry per reader row")
        return self._read(u, M_prev, loop_steps)

    def _query(self, u: Tensor, M, Q: Tensor, U: Tensor) -> Tensor:
        if not isinstance(M, MemoryRead):
            return super()._query(u, M, Q, U)
        q = self.query_code(F.linear(u, Q))
        return F.linear(M.read(q).to(u.dtype), U)

    def u_read(self, x: Tensor) -> Tensor:
        """Reader-side input projection: the shared A, or the ablation's own copy ``A_query`` (SLOT-LOOP-v1 S1A, 7.4)."""
        if self.A_query is None:
            return self.u(x)
        if self.arm in {"W1", "W5"}:
            x = (x - self.input_mean) / self.input_std
        return F.linear(x, self.A_query)

    def _masked_view_branch(self, u: Tensor, view: MemoryRead, loop_steps: int, valid: Tensor, final: Tensor) -> Tensor:
        """`_masked_branch` for one token of a MemoryRead, operation for operation."""
        flat_u = u.reshape(-1, self.cfg.r)
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        output = u.new_zeros((flat_u.shape[0], self.cfg.d_out))
        if indices.numel():
            z = self.read(flat_u.index_select(0, indices), view.select(indices, final.index_select(0, indices)), loop_steps)
            values = self.branch(z)
            output = output.to(values.dtype).index_copy(0, indices, values)
        return output.reshape(*u.shape[:-1], self.cfg.d_out)

    def process_sequence(self, x, M0=None, loop_steps=1, valid=None, write=None, reset=None):
        if not isinstance(M0, MemoryRead):
            return super().process_sequence(x, M0, loop_steps, valid, write, reset)
        self._check_steps(loop_steps)
        if x.ndim != 3 or x.shape[-1] != self.cfg.d_in:
            raise ValueError("sequence input must have shape [B,T,d_in]")
        if write is not None and bool(write.any()):
            raise NotImplementedError("online_write is not supported by a memory backend")
        if reset is not None and bool(reset.any()):
            raise NotImplementedError("resets are not supported by a memory backend")
        if M0.rows != x.shape[0]:
            raise ValueError("a memory read needs one owner entry per sequence row")
        shape = tuple(x.shape[:2])
        valid_mask = memory._mask(valid, shape, x.device, True)
        u = self.u_read(torch.where(valid_mask[..., None], x, torch.zeros_like(x)))
        positions = torch.arange(1, shape[1] + 1, device=x.device)[None]
        last = (valid_mask * positions).amax(dim=1) - 1 if shape[1] else positions.new_zeros(shape[0])
        outputs = [self._masked_view_branch(u[:, t], M0, loop_steps, valid_mask[:, t], last == t) for t in range(shape[1])]
        branch = torch.stack(outputs, dim=1) if outputs else u.new_empty((x.shape[0], 0, self.cfg.d_out))
        return branch, M0, self._stats(valid_mask, torch.zeros_like(valid_mask), loop_steps)


class SlotFixAdapter(SlotReadMixin, FixAdapter):
    """The WP-L3 fix adapter (S-RE, D128-3) with memory-backend reads and an optional separate reader projection."""

    def __init__(self, initial, arm: str) -> None:
        super().__init__(initial, arm)
        self.register_parameter("A_query", None)

    def add_query_projection(self) -> None:
        """The S1A ablation: the reader gets its own copy of A (trainable, initialised to A); the writer keeps A."""
        if self.A_query is not None:
            raise ValueError("the query projection copy already exists")
        self.A_query = nn.Parameter(self.A.detach().clone())


def attach_slot_reader(experiment) -> SlotFixAdapter:
    """Replace the experiment's adapter by a SlotFixAdapter with identical weights and buffers (a loaded S-RE or D128-3)."""
    old = experiment.adapter
    if isinstance(old, SlotFixAdapter):
        return old
    if type(old) is not FixAdapter:
        raise TypeError("the slot reader replaces a FixAdapter (the S-RE / D128-3 adapter)")
    new = SlotFixAdapter(old, experiment.arm).to(device=experiment.device, dtype=torch.float32)
    experiment.adapter = experiment.writer = experiment.bank.modules[experiment.layer].adapter = new
    return new


@dataclass(frozen=True)
class ReadSpec:
    """One read condition: the operator, its global gain and temperature, and where it replaces the native read.

    kind: matrix (g M q), soft (g sum softmax(k.q/tau) v), hard (g v at the largest k.q), clean_final (the oracle: the
    true value at the final valid question token, g = 1, uses the label). scope: all_valid_question (every valid
    question token, the deployed protocol) or final_only (S0's localisation table: only the final valid token is replaced,
    the others keep the native matrix read). recency adds log(lam^age) to the soft logits (descriptive control).
    """

    kind: str
    gain: float = 1.0
    tau: float | None = None
    scope: str = "all_valid_question"
    recency: bool = False
    capacity: int = 16

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.scope not in SCOPES:
            raise ValueError("unknown read kind or scope")
        if (self.kind == "soft") != (self.tau is not None):
            raise ValueError("tau belongs to the soft read and to no other")
        if self.recency and self.kind != "soft":
            raise ValueError("recency is a variant of the soft read")
        if self.kind == "clean_final" and (self.scope != "final_only" or self.gain != 1.0):
            raise ValueError("the clean-final oracle reads the true value at the final token only, with g = 1")
        if not (math.isfinite(self.gain) and self.gain > 0) or self.capacity < 1:
            raise ValueError("gain must be positive and capacity at least one")

    def as_dict(self) -> dict:
        return dict(kind=self.kind, gain=self.gain, tau=self.tau, scope=self.scope, recency=self.recency,
                    capacity=self.capacity)

    @classmethod
    def from_dict(cls, value: dict) -> "ReadSpec":
        return cls(**value)

    def label(self) -> str:
        tau = "" if self.tau is None else f"/t{self.tau:g}"
        return f"{self.kind}{'-rec' if self.recency else ''}/g{self.gain:g}{tau}/{self.scope}"


def _decay(experiment) -> float:
    return float(experiment.adapter.lam.detach())


def read_parts(experiment, batch: FixBatch, spec: ReadSpec):
    """(backend, state) the spec reads in this batch: the matrix state, or slots of the written keys and values."""
    matrix = batch.snapshot.states[experiment.layer]
    if spec.kind == "matrix":
        return MatrixBackend(spec.gain), MatrixState(matrix)
    slots = build_slots(batch.keys, batch.values, spec.capacity)
    if spec.kind == "soft":
        return SoftSlotBackend(spec.tau, spec.gain, _decay(experiment) if spec.recency else None), slots
    if spec.kind == "hard":
        return HardSlotBackend(spec.gain), slots
    return MatrixBackend(), MatrixState(matrix)  # clean_final: the replacement is the oracle value, set below


def true_values(batch: FixBatch, device) -> Tensor:
    """[B, Q, dv] the written raw value of each question's fact (labels: metrics and the oracle only)."""
    target = true_fact_indices(batch.arrays, device)
    return batch.values.gather(1, target[..., None].expand(-1, -1, batch.values.shape[-1]))


def bind_batch(experiment, batch: FixBatch, spec: ReadSpec) -> FixBatch:
    """The batch with its tensor matrix snapshot replaced by the MemoryRead of `spec` (Native is the unbound batch)."""
    matrix = batch.snapshot.states[experiment.layer]
    if not isinstance(matrix, Tensor):
        raise ValueError("the batch is already bound to a memory read")
    backend, state = read_parts(experiment, batch, spec)
    read = MemoryRead.bind(backend, state)
    if spec.scope == "final_only":
        oracle = None
        if spec.kind == "clean_final":
            if batch.question_ids.shape[1] != 1:
                raise ValueError("the oracle diagnostic is defined for one question per episode")
            oracle = true_values(batch, experiment.device)[:, 0]
        native = MemoryRead.bind(MatrixBackend(), MatrixState(matrix))
        read = ScopedRead(read.backend, read.state, read.owner, native=native, oracle=oracle)
    return replace(batch, snapshot=replace(batch.snapshot, states={experiment.layer: read}))


def final_query(experiment, batch: FixBatch, x: Tensor) -> Tensor:
    """[B, Q, dk] the adapter's own query code at each question's final valid token (the first read, z0 = u)."""
    last = batch.question_mask.long().sum(-1) - 1
    B, Q = last.shape
    final = x[torch.arange(B, device=x.device)[:, None], torch.arange(Q, device=x.device)[None, :], last]
    adapter = experiment.adapter
    u = adapter.u_read(final.float()) if hasattr(adapter, "u_read") else adapter.u(final.float())
    return adapter.query_code(F.linear(u, adapter.Q))


def final_read(experiment, batch: FixBatch, x: Tensor, spec: ReadSpec | None, eps: float = 1e-6) -> dict[str, Tensor]:
    """The read m [B, Q, dv] of `spec` at each question's final valid token and its statistics against the written true
    value (None: the native read, g = 1). The query is the one the forward used; the read is the backend's own.

    rel_err is the per-sample |m - v|^2 / (|v|^2 + eps) of the SQ2 report (`read_fix.diagnostics`); cos and norm_ratio are
    the cosine and |m| / |v|. A soft read adds the weight on the true slot, the gap between the two largest weights and the
    entropy; a hard read adds whether the selected slot is the true fact's.
    """
    q = final_query(experiment, batch, x)
    B, Q = q.shape[:2]
    clean = true_values(batch, experiment.device)
    target = true_fact_indices(batch.arrays, experiment.device)
    owner = torch.arange(B, device=q.device).repeat_interleave(Q)
    flat = q.flatten(0, 1)
    out: dict[str, Tensor] = {}
    spec = ReadSpec("matrix") if spec is None else spec
    if spec.kind == "clean_final":
        m = clean
    elif spec.kind == "matrix":
        # The SQ2 report's own form (`read_fix.question_terms`: one einsum over the episode's questions), so that the replay of
        # its read error is exact; the deployed per-token read is the backend's matmul, checked bitwise elsewhere.
        stored = torch.einsum("bij,bqj->bqi", batch.snapshot.states[experiment.layer], q)
        m = stored if spec.gain == 1.0 else stored * spec.gain
    else:
        backend, state = read_parts(experiment, batch, spec)
        m = backend.read(state, flat, owner).reshape(B, Q, -1)
        if spec.kind == "soft":
            w = backend.weights(state, flat, owner).reshape(B, Q, -1)
            top = w.topk(2, dim=-1).values
            out.update(w_true=w.gather(-1, target[..., None]).squeeze(-1), w_gap=top[..., 0] - top[..., 1],
                       w_entropy=-torch.xlogy(w, w).sum(-1))
        elif spec.kind == "hard":
            chosen, _ = backend.index(state, flat, owner)
            out["slot_is_true"] = chosen.reshape(B, Q) == target
    norm = lambda a: a.norm(dim=-1)
    out.update(m=m, clean=clean,
               rel_err=(m - clean).square().sum(-1) / (clean.square().sum(-1) + eps),
               cos=(m * clean).sum(-1) / (norm(m) * norm(clean)).clamp_min(1e-12),
               norm_ratio=norm(m) / norm(clean).clamp_min(1e-12))
    return out
