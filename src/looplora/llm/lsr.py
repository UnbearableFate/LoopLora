"""LSR: label-free stop rules on the r = 128 M1-L4 checkpoints (evaluation only; docs/prereg/LSR.md).

Terms as in p1lc. A row is one scored question (episode-major B*Q flattening); its final position is its last real question
token. For a row, z^0 = u, z^s (s = 1..8) are the reader states of ``p1lc.read_states`` at that position and
r^s = ``adapter._query(z^s, M, Q, U)`` = U M Q z^s. p^s is the full-vocabulary answer-position argmax when every question position
reads s times (s = 0..8). The stop depth s* of a rule is decided from the final position's trajectory and the question is then
scored at fixed depth s*, so its stopped prediction is p^{s*}.

Nothing in training, the P1-L evaluation or p1lc imports this module. ``instrument`` wraps ONE adapter instance for the duration
of one scoring call (as ``p1lc.read_control``) and restores it on exit. The rules, the selection and the statistics are numpy.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from looplora.llm import p1lc
from looplora.llm.lsr_rules import *  # noqa: F401,F403  (constants, rules and selection stay reachable as lsr.<name>)
from looplora.llm.lsr_rules import EPS, L_MAX


# ------------------------------------------------------------------------------------ instrumentation

@contextmanager
def instrument(adapter, last, *, trace=None, depth_rows=None):
    """Instrument the adapter's memory reads during ONE scoring forward of the rows ``last`` describes.

    trace       a dict. The scoring pass must read at L = 8; for every row it receives, from the row's final position,
                ``change`` [rows, 8] = ||z^s - z^{s-1}|| / max(||z^{s-1}||, EPS), s = 1..8, and ``read_ratio`` [rows, 8] =
                ||r^s|| / max(||r^0||, EPS), s = 1..7 (column 7 is NaN: the rule is forced at s = 8). Deployed outputs are
                unchanged (the state used is ``p1lc.read_states(...)[-1]``, equal to ``adapter.read`` bit for bit).
    depth_rows  a long tensor [rows]: every position of row i reads depth_rows[i] times (rows grouped by depth, so rows that
                stop early run fewer reads). The per-question stopped read.
    Entering the context with neither option only counts positions. On exit the adapter has no instance attributes left over.
    """
    if (trace is not None) and (depth_rows is not None):
        raise ValueError("trace and depth_rows are separate passes")
    rows_total = len(last)
    cursor = SimpleNamespace(t=-1, final=None, rows=None)
    process, masked, read = adapter.process_sequence, adapter._masked_branch, adapter._read
    if depth_rows is not None and len(depth_rows) != rows_total:
        raise ValueError("depth_rows must have one entry per question row")
    if trace is not None:
        trace.clear()
        trace["seen"] = torch.zeros(rows_total, dtype=torch.long, device=last.device)

    def process_sequence(x, M0=None, loop_steps=1, valid=None, write=None, reset=None):
        if x.shape[0] != rows_total:
            raise ValueError("instrument was built for a different number of question rows")
        cursor.t = -1
        return process(x, M0, loop_steps, valid, write, reset)

    def masked_branch(u, memory, loop_steps, valid):
        if u.ndim != 2:
            raise ValueError("instrument needs the per-position reader path (process_sequence)")
        cursor.t += 1
        cursor.final = last == cursor.t
        cursor.rows = valid.reshape(-1).nonzero(as_tuple=True)[0]
        return masked(u, memory, loop_steps, valid)

    def traced_read(u, memory, loop_steps):
        if loop_steps != L_MAX:
            raise ValueError(f"the trace pass reads at L = {L_MAX}")
        hit = cursor.final[cursor.rows]
        states = p1lc.read_states(adapter, u, memory, loop_steps)
        if hit.any():
            if "change" not in trace:
                trace["change"] = torch.full((rows_total, L_MAX), float("nan"), device=u.device)
                trace["read_ratio"] = torch.full((rows_total, L_MAX), float("nan"), device=u.device)
            chosen = cursor.rows[hit]
            zs = [u[hit]] + [state[hit] for state in states]
            norms = torch.stack([z.float().norm(dim=-1) for z in zs], -1)                 # [h, 9]
            steps = torch.stack([(zs[s] - zs[s - 1]).float().norm(dim=-1) for s in range(1, L_MAX + 1)], -1)
            reads = torch.stack([adapter._query(zs[s], memory[hit], adapter.Q, adapter.U).float().norm(dim=-1)
                                 for s in range(L_MAX)], -1)                              # [h, 8], s = 0..7
            trace["change"][chosen] = steps / norms[:, :L_MAX].clamp_min(EPS)
            trace["read_ratio"][chosen, : L_MAX - 1] = reads[:, 1:] / reads[:, :1].clamp_min(EPS)
            trace["seen"][chosen] += 1
        return states[-1]

    def graded_read(u, memory, loop_steps):
        depth = depth_rows[cursor.rows]
        out = torch.empty_like(u)
        for value in depth.unique().tolist():
            index = (depth == value).nonzero(as_tuple=True)[0]
            out[index] = read(u[index], memory[index], int(value))
        return out

    with ExitStack() as stack:
        stack.enter_context(patch.object(adapter, "process_sequence", process_sequence))
        stack.enter_context(patch.object(adapter, "_masked_branch", masked_branch))
        if trace is not None:
            stack.enter_context(patch.object(adapter, "_read", traced_read))
        elif depth_rows is not None:
            stack.enter_context(patch.object(adapter, "_read", graded_read))
        yield
    if trace is not None and not bool((trace["seen"] == 1).all()):
        raise AssertionError("every question row must be traced exactly once, at its final position")


# ------------------------------------------------------------------------------------- batch scans

@torch.no_grad()
def scan_batch(experiment, prepared, *, audit=False):
    """p^0..p^8 [B, Q, 9] (token ids), change and read_ratio [B, Q, 8] for one prepared batch.

    p^8 comes from the traced L = 8 pass. With ``audit`` the traced logits must equal a plain L = 8 pass exactly and the
    projection inputs x (hence u) and the memory must not change with the depth (Gate 0 of llm.evaluate).
    """
    B, Q, _ = prepared.question_ids.shape
    last = p1lc.last_positions(prepared.question_mask)
    layer = experiment.layer
    memory = prepared.snapshot.states[layer].clone() if audit else None
    predictions, inputs = [], None
    for depth in range(L_MAX):
        logits, x = experiment.score(prepared, depth, capture_inputs=audit)
        predictions.append(logits.argmax(-1))
        if audit:
            inputs = x if inputs is None else inputs
            if not torch.equal(x, inputs):
                raise AssertionError(f"Gate 0: question inputs changed at L={depth}")
    trace = {}
    with instrument(experiment.adapter, last, trace=trace):
        logits8, x = experiment.score(prepared, L_MAX, capture_inputs=audit)
    predictions.append(logits8.argmax(-1))
    if audit:
        plain, _ = experiment.score(prepared, L_MAX)
        if not torch.equal(plain, logits8) or not torch.equal(x, inputs):
            raise AssertionError("the traced L = 8 pass differs from the plain L = 8 pass")
        if not torch.equal(memory, prepared.snapshot.states[layer]):
            raise AssertionError("the memory snapshot changed during the scan")
    pred = torch.stack(predictions, -1)
    return dict(pred=pred.cpu().numpy().astype(np.int32),
                change=trace["change"].reshape(B, Q, L_MAX).cpu().numpy().astype(np.float32),
                read_ratio=trace["read_ratio"].reshape(B, Q, L_MAX).cpu().numpy().astype(np.float32))


def stopped_logits(experiment, prepared, depth_rows):
    """Answer logits [B, Q, V] with every question read at its own depth (rows in B*Q order)."""
    last = p1lc.last_positions(prepared.question_mask)
    with instrument(experiment.adapter, last, depth_rows=depth_rows):
        return experiment.score(prepared, 1)[0]
