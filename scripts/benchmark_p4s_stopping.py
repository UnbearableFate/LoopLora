"""Prepared-memory CUDA reader benchmarks for the WP-C3 stopping pilot.

The caller supplies queries and validation-selected thresholds. No labels,
writer preparation, checkpoint loading, or threshold fitting enter this API.
"""
from __future__ import annotations

from collections.abc import Callable
from statistics import fmean, median
from time import perf_counter
from typing import Any

import torch

from looplora.core.adapters import IterativeQueryAdapter
from looplora.core.stopping import (
    AdaptiveResult,
    ReaderBatch,
    adaptive_read,
    fixed_read,
    halting_steps,
    trace_reader,
)


_FIXED_STEPS = (1, 2, 4, 8, 16)
_CRITERIA = ("read_norm", "relative_change", "output_stability")
_MAX_STEPS = 16


def _p95(values: list[float]) -> float:
    """Linearly interpolated 95th percentile, including for a single repeat."""
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _audit_adaptive(
    adapter: IterativeQueryAdapter,
    batch: ReaderBatch,
    thresholds: dict[str, float | int],
) -> dict[str, tuple[float, float]]:
    """Check the actual masked reader against untimed, full-depth references."""
    trace = trace_reader(adapter, batch, max_steps=_MAX_STEPS)
    counters = {}
    for criterion in _CRITERIA:
        expected_steps = halting_steps(trace, criterion, thresholds[criterion])
        expected_predictions = trace.predictions.gather(1, expected_steps[:, None]).squeeze(1)
        actual = adaptive_read(
            adapter, batch, criterion, thresholds[criterion], max_steps=_MAX_STEPS,
        )
        if not torch.equal(actual.steps, expected_steps):
            raise AssertionError(f"{criterion}: masked-reader steps differ from offline halting")
        if not torch.equal(actual.predictions, expected_predictions):
            raise AssertionError(f"{criterion}: masked-reader predictions differ from offline halting")
        expected_reads = expected_steps + int(criterion == "read_norm")
        if not torch.equal(actual.read_calls, expected_reads):
            raise AssertionError(f"{criterion}: memory-read counters do not match the stopping convention")
        # Compare terminal states as well as argmax predictions. Different active
        # batch shapes can change floating-point GEMM rounding slightly.
        for step in expected_steps.unique().tolist():
            indices = (expected_steps == step).nonzero(as_tuple=True)[0]
            selected = batch.take(indices)
            expected_states = adapter.read(selected.u, selected.memory, int(step))
            torch.testing.assert_close(
                actual.states.index_select(0, indices), expected_states,
                rtol=1e-4, atol=1e-6,
                msg=f"{criterion}: terminal states differ from fixed-depth reader at L={step}",
            )
        counters[criterion] = (
            float(actual.steps.double().mean().item()),
            float(actual.read_calls.double().mean().item()),
        )
    # Audit the fixed baseline's complete prediction path, too.
    for step in _FIXED_STEPS:
        if not torch.equal(fixed_read(adapter, batch, step), trace.predictions[:, step]):
            raise AssertionError(f"fixed_{step}: predictions differ from full-depth trace")
    return counters


def _measure(
    operation: Callable[[], torch.Tensor | AdaptiveResult],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    stream = torch.cuda.current_stream(device)
    # Allocate event objects outside the timed intervals. Prime their lazy CUDA
    # initialization before timing so the first sample has the same setup cost.
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    for begin, end in events:
        begin.record(stream)
        end.record(stream)
    torch.cuda.synchronize(device)
    cuda_ms = []
    wall_ms = []
    for begin, end in events:
        torch.cuda.synchronize(device)
        wall_begin = perf_counter()
        begin.record(stream)
        result = operation()
        end.record(stream)
        torch.cuda.synchronize(device)
        wall_ms.append(1000.0 * (perf_counter() - wall_begin))
        cuda_ms.append(float(begin.elapsed_time(end)))
        # Release outputs after both clocks stop, before the next repeat.
        del result
    return {
        "cuda_ms_mean": fmean(cuda_ms),
        "cuda_ms_median": median(cuda_ms),
        "cuda_ms_p95": _p95(cuda_ms),
        "wall_ms_mean": fmean(wall_ms),
        "wall_ms_median": median(wall_ms),
        "wall_ms_p95": _p95(wall_ms),
    }


@torch.inference_mode()
def benchmark_reader(
    adapter: IterativeQueryAdapter,
    batch: ReaderBatch,
    thresholds: dict[str, float | int],
    warmup: int = 5,
    repeats: int = 20,
) -> list[dict[str, Any]]:
    """Benchmark five fixed depths and three actual active-query readers.

    Production batches contain 64 queries or one query, already projected and
    paired with their memory and candidate codes. CUDA events measure the reader
    call; synchronized wall time also includes Python dispatch, active-query
    gather/masking, stopping bookkeeping, event recording, and final device
    synchronization. Each method includes its required candidate prediction.
    Audits and counter extraction happen before all timed calls. The adapter
    must already be in evaluation mode; caller-owned input/model state is kept.
    """
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if batch.u.ndim != 2 or batch.u.shape[0] < 1:
        raise ValueError("benchmark queries must have nonempty shape [N,r]")
    if batch.u.device.type != "cuda":
        raise ValueError("benchmark_reader requires prepared CUDA inputs")
    if adapter.training:
        raise ValueError("benchmark_reader requires an adapter in evaluation mode")
    missing = set(_CRITERIA).difference(thresholds)
    if missing:
        raise ValueError(f"missing selected stopping thresholds: {sorted(missing)}")
    device = batch.u.device
    with torch.cuda.device(device):
        counters = _audit_adaptive(adapter, batch, thresholds)
        methods: list[tuple[str, Callable[[], torch.Tensor | AdaptiveResult], float, float]] = []
        for step in _FIXED_STEPS:
            methods.append((
                f"fixed_{step}",
                lambda step=step: fixed_read(adapter, batch, step),
                float(step), float(step),
            ))
        for criterion in _CRITERIA:
            mean_steps, mean_reads = counters[criterion]
            methods.append((
                criterion,
                lambda criterion=criterion: adaptive_read(
                    adapter, batch, criterion, thresholds[criterion], max_steps=_MAX_STEPS,
                ),
                mean_steps, mean_reads,
            ))
        rows = []
        for method, operation, mean_steps, mean_reads in methods:
            measurements = _measure(operation, device, warmup, repeats)
            rows.append({
                "method": method,
                "batch_size": int(batch.u.shape[0]),
                "mean_steps": mean_steps,
                "mean_reads": mean_reads,
                **measurements,
                "warmup": warmup,
                "repeats": repeats,
            })
    return rows
