"""Evaluation-only, label-free stopping for the evolving-query reader."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters import AdapterBase
from .config import validate_steps

EPS = 1e-12
CRITERIA = ("read_norm", "relative_change", "output_stability")


@dataclass
class ReaderBatch:
    u: Tensor
    memory: Tensor
    codes: Tensor
    candidate_ids: Tensor

    def take(self, indices: Tensor) -> ReaderBatch:
        return ReaderBatch(*(value.index_select(0, indices) for value in
                             (self.u, self.memory, self.codes, self.candidate_ids)))


@dataclass
class ReaderTrace:
    predictions: Tensor
    read_norm: Tensor
    relative_change: Tensor


@dataclass
class AdaptiveResult:
    predictions: Tensor
    steps: Tensor
    states: Tensor
    read_calls: Tensor


def _check(adapter: AdapterBase, batch: ReaderBatch, max_steps: int) -> None:
    validate_steps(max_steps)
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if adapter.cfg.variant != "iterq":
        raise ValueError("stopping requires the evolving-query (iterq) reader")
    n = batch.u.shape[0]
    if batch.u.shape != (n, adapter.cfg.r):
        raise ValueError("u must have shape [N,r]")
    if batch.memory.shape != (n, adapter.cfg.r_mem, adapter.cfg.r_mem):
        raise ValueError("memory must have shape [N,r_mem,r_mem]")
    if (batch.codes.ndim != 3 or batch.codes.shape[0] != n
            or batch.codes.shape[-1] != adapter.cfg.d_out
            or batch.candidate_ids.shape != batch.codes.shape[:2]
            or batch.codes.shape[1] == 0):
        raise ValueError("codes and candidate_ids must have matching nonempty candidate dimensions")
    if not bool((batch.candidate_ids >= 0).any(dim=-1).all()):
        raise ValueError("each query needs a valid candidate")


def _criterion(criterion: str, threshold: float | int) -> None:
    if criterion not in CRITERIA:
        raise ValueError(f"unknown stopping criterion: {criterion}")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if criterion == "output_stability" and (int(threshold) != threshold or threshold < 1):
        raise ValueError("output_stability threshold must be a positive integer patience")


def _read(adapter: AdapterBase, batch: ReaderBatch, z: Tensor) -> Tensor:
    return adapter._query(z, batch.memory, adapter.Q, adapter.U)


def _update(adapter: AdapterBase, u: Tensor, z: Tensor, readout: Tensor) -> Tensor:
    h = F.linear(z, adapter.R) + readout
    return u + adapter.cfg.rho * (torch.relu(h) if adapter.cfg.activation == "relu" else h)


@torch.no_grad()
def predict(adapter: AdapterBase, batch: ReaderBatch, z: Tensor) -> Tensor:
    output = adapter.branch(z)
    logits = (batch.codes * output[:, None]).sum(dim=-1)
    logits = logits.masked_fill(batch.candidate_ids < 0, -torch.inf)
    return batch.candidate_ids.gather(1, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)


@torch.no_grad()
def fixed_read(adapter: AdapterBase, batch: ReaderBatch, loop_steps: int) -> Tensor:
    validate_steps(loop_steps)
    return predict(adapter, batch, adapter.read(batch.u, batch.memory, loop_steps))


@torch.no_grad()
def trace_reader(adapter: AdapterBase, batch: ReaderBatch, max_steps: int = 16) -> ReaderTrace:
    """Record z^0..z^L decisions, including the read needed at each state."""
    _check(adapter, batch, max_steps)
    z = batch.u
    readout = _read(adapter, batch, z)
    initial_norm = torch.linalg.vector_norm(readout, dim=-1).clamp_min(EPS)
    predictions = [predict(adapter, batch, z)]
    reads = [torch.linalg.vector_norm(readout, dim=-1) / initial_norm]
    changes = [torch.full_like(initial_norm, float("nan"))]
    for _ in range(max_steps):
        previous = z
        z = _update(adapter, batch.u, z, readout)
        readout = _read(adapter, batch, z)
        predictions.append(predict(adapter, batch, z))
        reads.append(torch.linalg.vector_norm(readout, dim=-1) / initial_norm)
        changes.append(torch.linalg.vector_norm(z - previous, dim=-1)
                       / torch.linalg.vector_norm(previous, dim=-1).clamp_min(EPS))
    return ReaderTrace(torch.stack(predictions, -1), torch.stack(reads, -1), torch.stack(changes, -1))


@torch.no_grad()
def halting_steps(trace: ReaderTrace, criterion: str, threshold: float | int) -> Tensor:
    """Choose the first qualifying post-update state; never inspect labels."""
    _criterion(criterion, threshold)
    if trace.predictions.ndim != 2 or trace.predictions.shape[1] < 2:
        raise ValueError("trace must include an initial state and at least one update")
    shape = trace.predictions.shape
    if trace.read_norm.shape != shape or trace.relative_change.shape != shape:
        raise ValueError("trace fields must have matching shapes")
    maximum = shape[1] - 1
    steps = torch.full((shape[0],), maximum, dtype=torch.long, device=trace.predictions.device)
    stopped = torch.zeros(shape[0], dtype=torch.bool, device=steps.device)
    patience = torch.zeros_like(steps)
    for step in range(1, maximum + 1):
        if criterion == "output_stability":
            equal = trace.predictions[:, step] == trace.predictions[:, step - 1]
            patience = torch.where(equal, patience + 1, torch.zeros_like(patience))
            hit = patience >= threshold
        else:
            hit = getattr(trace, criterion)[:, step] <= threshold
        first = hit & ~stopped
        steps = torch.where(first, step, steps)
        stopped = stopped | hit
    return steps


@torch.no_grad()
def adaptive_read(
    adapter: AdapterBase, batch: ReaderBatch, criterion: str,
    threshold: float | int, max_steps: int = 16,
) -> AdaptiveResult:
    """Compact active queries and freeze each state immediately at its halt.

    The read-norm diagnostic at z^L is cached for update L+1 if needed. It
    incurs exactly one additional memory read per query, including at the cap.
    """
    _check(adapter, batch, max_steps)
    _criterion(criterion, threshold)
    n = batch.u.shape[0]
    active = torch.arange(n, device=batch.u.device)
    active_batch, z = batch, batch.u
    states = torch.zeros_like(z)
    predictions = torch.full((n,), -1, dtype=torch.long, device=z.device)
    steps = torch.zeros(n, dtype=torch.long, device=z.device)
    patience = torch.zeros_like(steps)
    previous_prediction = predict(adapter, batch, z) if criterion == "output_stability" else None
    cached = _read(adapter, batch, z) if criterion == "read_norm" else None
    initial_norm = torch.linalg.vector_norm(cached, dim=-1).clamp_min(EPS) if cached is not None else None
    for step in range(1, max_steps + 1):
        if active.numel() == 0:
            break
        readout = cached if cached is not None else _read(adapter, active_batch, z)
        next_z = _update(adapter, active_batch.u, z, readout)
        next_prediction = None
        if criterion == "read_norm":
            cached = _read(adapter, active_batch, next_z)
            hit = torch.linalg.vector_norm(cached, dim=-1) / initial_norm <= threshold
        elif criterion == "relative_change":
            ratio = (torch.linalg.vector_norm(next_z - z, dim=-1)
                     / torch.linalg.vector_norm(z, dim=-1).clamp_min(EPS))
            hit = ratio <= threshold
        else:
            next_prediction = predict(adapter, active_batch, next_z)
            equal = next_prediction == previous_prediction
            patience = torch.where(equal, patience + 1, torch.zeros_like(patience))
            hit = patience >= threshold
        if step == max_steps:
            hit = torch.ones_like(hit)
        halted = hit.nonzero(as_tuple=True)[0]
        if halted.numel():
            ids = active.index_select(0, halted)
            values = (next_prediction.index_select(0, halted) if next_prediction is not None else
                      predict(adapter, active_batch.take(halted), next_z.index_select(0, halted)))
            predictions = predictions.index_copy(0, ids, values)
            states = states.index_copy(0, ids, next_z.index_select(0, halted))
            steps = steps.index_fill(0, ids, step)
        continuing = (~hit).nonzero(as_tuple=True)[0]
        active = active.index_select(0, continuing)
        active_batch = active_batch.take(continuing)
        z = next_z.index_select(0, continuing)
        if cached is not None:
            cached = cached.index_select(0, continuing)
            initial_norm = initial_norm.index_select(0, continuing)
        if next_prediction is not None:
            previous_prediction = next_prediction.index_select(0, continuing)
            patience = patience.index_select(0, continuing)
    return AdaptiveResult(predictions, steps, states, steps + int(criterion == "read_norm"))
