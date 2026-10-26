"""P4-S-L: label-free features and a learned halt on a frozen r32 reader."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor, nn

from .adapters import AdapterBase
from .stopping import AdaptiveResult, EPS, ReaderBatch, ReaderTrace, _check, _read, _update

RANK = 32
MAX_STEPS = 16
FEATURE_DIM = RANK + 3 + MAX_STEPS


@dataclass
class LearnedReaderTrace(ReaderTrace):
    states: Tensor
    features: Tensor


@dataclass
class LearnedAdaptiveResult(AdaptiveResult):
    executed_updates: Tensor
    threshold_hit: Tensor


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite {name}")


def _validate_features(features: Tensor) -> None:
    if (not isinstance(features, Tensor) or features.ndim < 1
            or features.shape[-1] != FEATURE_DIM or not features.is_floating_point()):
        raise ValueError(f"features must be floating point with final dimension {FEATURE_DIM}")
    _finite(features, "halting features")


class HaltingHead(nn.Module):
    """51→64 ReLU→1 logit; forward returns the sigmoid probability."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.network = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.ReLU(), nn.Linear(64, 1))

    def logits(self, features: Tensor) -> Tensor:
        _validate_features(features)
        parameter = self.network[0].weight
        if features.device != parameter.device or features.dtype != parameter.dtype:
            raise ValueError("features must match the halting head device and dtype")
        result = self.network(features).squeeze(-1)
        _finite(result, "halting logits")
        return result

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(features))


def _validate_reader(adapter: AdapterBase, batch: ReaderBatch, max_steps: int) -> None:
    if batch.u.ndim != 2:
        raise ValueError("u must have shape [N,32]")
    _check(adapter, batch, max_steps)
    if adapter.cfg.r != RANK:
        raise ValueError("learned halting requires reader rank 32")
    if max_steps > MAX_STEPS:
        raise ValueError("max_steps must lie in 1..16")
    for name in ("u", "memory", "codes"):
        value = getattr(batch, name)
        if not value.is_floating_point() or value.device != batch.u.device or value.dtype != batch.u.dtype:
            raise ValueError("reader tensors must share a floating-point dtype and device")
        _finite(value, name)
    if batch.candidate_ids.device != batch.u.device or batch.candidate_ids.dtype != torch.long:
        raise ValueError("candidate_ids must be int64 on the reader device")
    for name in ("R", "U", "Q", "B"):
        value = getattr(adapter, name)
        if value.device != batch.u.device or value.dtype != batch.u.dtype:
            raise ValueError("adapter and reader batch must share dtype and device")
        _finite(value, f"adapter {name}")


def _threshold(theta: float) -> None:
    if isinstance(theta, bool) or not isinstance(theta, Real) or not math.isfinite(theta) or not 0 <= theta <= 1:
        raise ValueError("theta must be finite and lie in [0,1]")


def _probabilities(probabilities: Tensor) -> None:
    if not probabilities.is_floating_point():
        raise ValueError("halting probabilities must be floating point")
    _finite(probabilities, "halting probabilities")
    if not bool(((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("halting probabilities must lie in [0,1]")


def _features(z: Tensor, previous: Tensor, readout: Tensor,
              initial_norm: Tensor, step: int) -> tuple[Tensor, Tensor, Tensor]:
    norm = torch.linalg.vector_norm(z, dim=-1)
    change = (torch.linalg.vector_norm(z - previous, dim=-1)
              / torch.linalg.vector_norm(previous, dim=-1).clamp_min(EPS))
    ratio = torch.linalg.vector_norm(readout, dim=-1) / initial_norm
    one_hot = z.new_zeros((z.shape[0], MAX_STEPS))
    one_hot[:, step - 1] = 1
    features = torch.cat((z, norm[:, None], change[:, None], ratio[:, None], one_hot), dim=-1)
    _finite(features, "reader features")
    return features, ratio, change


def _prediction(adapter: AdapterBase, batch: ReaderBatch, z: Tensor) -> Tensor:
    # Same candidate decoder as stopping.predict, with explicit overflow checks.
    output = adapter.branch(z)
    _finite(output, "reader output")
    logits = (batch.codes * output[:, None]).sum(dim=-1)
    _finite(logits, "candidate logits")
    logits = logits.masked_fill(batch.candidate_ids < 0, -torch.inf)
    return batch.candidate_ids.gather(1, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)


@torch.no_grad()
def trace_frozen_reader(adapter: AdapterBase, batch: ReaderBatch,
                        max_steps: int = MAX_STEPS) -> LearnedReaderTrace:
    """Cache normal no-grad tensors usable later in head-only autograd training."""
    _validate_reader(adapter, batch, max_steps)
    z = batch.u
    readout = _read(adapter, batch, z)
    _finite(readout, "initial readout")
    initial_norm = torch.linalg.vector_norm(readout, dim=-1).clamp_min(EPS)
    _finite(initial_norm, "initial read norm")
    states, predictions = [z], [_prediction(adapter, batch, z)]
    ratios = [torch.linalg.vector_norm(readout, dim=-1) / initial_norm]
    changes = [torch.full_like(initial_norm, float("nan"))]
    features = []
    for step in range(1, max_steps + 1):
        previous = z
        z = _update(adapter, batch.u, z, readout)
        readout = _read(adapter, batch, z)
        _finite(readout, "post-state readout")
        feature, ratio, change = _features(z, previous, readout, initial_norm, step)
        states.append(z)
        predictions.append(_prediction(adapter, batch, z))
        features.append(feature)
        ratios.append(ratio)
        changes.append(change)
    return LearnedReaderTrace(torch.stack(predictions, 1), torch.stack(ratios, 1),
                              torch.stack(changes, 1), torch.stack(states, 1), torch.stack(features, 1))


@torch.no_grad()
def learned_halting_steps(probabilities: Tensor, theta: float) -> Tensor:
    """First inclusive threshold hit, otherwise earliest maximum, in steps 1..S."""
    _threshold(theta)
    if (not isinstance(probabilities, Tensor) or probabilities.ndim != 2
            or not 1 <= probabilities.shape[1] <= MAX_STEPS):
        raise ValueError("probabilities must have shape [N,S] with 1<=S<=16")
    _probabilities(probabilities)
    hits = probabilities >= theta
    first = hits.to(torch.long).argmax(dim=-1) + 1
    best = probabilities.argmax(dim=-1) + 1
    return torch.where(hits.any(dim=-1), first, best)


@torch.no_grad()
def learned_adaptive_read(adapter: AdapterBase, batch: ReaderBatch, head: nn.Module,
                          theta: float, max_steps: int = MAX_STEPS) -> LearnedAdaptiveResult:
    """Compact hits immediately; retain only each active query's best fallback."""
    _validate_reader(adapter, batch, max_steps)
    _threshold(theta)
    if not callable(head):
        raise ValueError("head must map feature rows to probabilities")
    count = batch.u.shape[0]
    device = batch.u.device
    steps = torch.zeros(count, dtype=torch.long, device=device)
    executed = torch.zeros_like(steps)
    hit_result = torch.zeros(count, dtype=torch.bool, device=device)
    states = torch.zeros_like(batch.u)
    predictions = torch.full_like(steps, -1)
    if count == 0:
        return LearnedAdaptiveResult(predictions, steps, states, executed.clone(), executed, hit_result)
    active, active_batch, z = torch.arange(count, device=device), batch, batch.u
    readout = _read(adapter, active_batch, z)
    _finite(readout, "initial readout")
    initial_norm = torch.linalg.vector_norm(readout, dim=-1).clamp_min(EPS)
    _finite(initial_norm, "initial read norm")
    best_probability = z.new_full((count,), -torch.inf)
    best_states, best_steps = torch.zeros_like(z), torch.zeros_like(steps)
    for step in range(1, max_steps + 1):
        next_z = _update(adapter, active_batch.u, z, readout)
        next_readout = _read(adapter, active_batch, next_z)
        _finite(next_readout, "post-state readout")
        features, _, _ = _features(next_z, z, next_readout, initial_norm, step)
        probability = head(features)
        if not isinstance(probability, Tensor) or probability.shape != (active.numel(),):
            raise ValueError("head must return one probability per feature row")
        if probability.device != device:
            raise ValueError("head probabilities must remain on the reader device")
        _probabilities(probability)
        better = probability > best_probability
        best_probability = torch.where(better, probability, best_probability)
        best_states = torch.where(better[:, None], next_z, best_states)
        best_steps = torch.where(better, step, best_steps)
        hit = probability >= theta
        finish = hit if step < max_steps else torch.ones_like(hit)
        halted = finish.nonzero(as_tuple=True)[0]
        if halted.numel():
            ids = active.index_select(0, halted)
            chosen = torch.where(hit[:, None], next_z, best_states).index_select(0, halted)
            chosen_steps = torch.where(hit, step, best_steps).index_select(0, halted)
            prediction = _prediction(adapter, active_batch.take(halted), chosen)
            predictions = predictions.index_copy(0, ids, prediction)
            states = states.index_copy(0, ids, chosen)
            steps = steps.index_copy(0, ids, chosen_steps)
            executed = executed.index_fill(0, ids, step)
            hit_result = hit_result.index_copy(0, ids, hit.index_select(0, halted))
        continuing = (~finish).nonzero(as_tuple=True)[0]
        if continuing.numel() == 0:
            break
        active = active.index_select(0, continuing)
        active_batch = active_batch.take(continuing)
        z = next_z.index_select(0, continuing)
        readout = next_readout.index_select(0, continuing)
        initial_norm = initial_norm.index_select(0, continuing)
        best_probability = best_probability.index_select(0, continuing)
        best_states = best_states.index_select(0, continuing)
        best_steps = best_steps.index_select(0, continuing)
    return LearnedAdaptiveResult(predictions, steps, states, executed + 1, executed, hit_result)
