"""Task-aware S1B reader, actual-forward traces and complete model-only exports.

No existing adapter, S1A loader, global task or numerical path is modified.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from looplora.core.memory_backend import MemoryRead
from looplora.infra.seeds import stable_hash
from . import s1b_data as data, s1b_rules as rules, slot_experiment as parent, sq2
from .evaluate import _slice
from .slot_read import SlotFixAdapter, ReadSpec, bind_batch
from .train import synchronize


class LoopAdapter(SlotFixAdapter):
    def __init__(self, initial, arm):
        super().__init__(initial, arm)
        self.control = "shared"
        self.capture = self.time_reads = False
        self.trace = []
        self.cost = dict(backend_calls=0, backend_rows=0, valid_read_positions=0, recurrent_updates=0,
                         logical_reads=0, read_seconds=[], read_kernel_seconds=[])
        self._final_mask = None

    def untie(self):
        if self.control != "shared":
            raise ValueError("control already configured")
        for name in ("Q", "R", "U"):
            original = getattr(self, name)
            setattr(self, name + "_steps", nn.ParameterList([nn.Parameter(original.detach().clone()) for _ in range(3)]))
        self.control = "untied"

    def reset_observation(self, *, capture=False, timing=False):
        self.capture, self.time_reads = capture, timing
        self.trace = []
        self.cost = dict(backend_calls=0, backend_rows=0, valid_read_positions=0, recurrent_updates=0,
                         logical_reads=0, read_seconds=[], read_kernel_seconds=[])

    def _masked_view_branch(self, u, view, loop_steps, valid, final):
        previous = self._final_mask
        self._final_mask = final.reshape(-1)[valid.reshape(-1)]
        self._active_rows = valid.reshape(-1).nonzero(as_tuple=True)[0]
        try:
            return super()._masked_view_branch(u, view, loop_steps, valid, final)
        finally:
            self._final_mask = previous

    def _read(self, u, state, loop_steps):
        if not isinstance(state, MemoryRead):
            return super()._read(u, state, loop_steps)
        z, cached = u, None
        self.cost["valid_read_positions"] += len(u)
        self.cost["recurrent_updates"] += len(u) * loop_steps
        for step in range(loop_steps):
            z_before = z
            if self.control == "untied" and loop_steps != 4:
                raise ValueError("untied control is registered at depth four only")
            Q, R, U = ((getattr(self, n) if step == 0 else getattr(self, n + "_steps")[step - 1]
                         for n in ("Q", "R", "U")) if self.control == "untied" else (self.Q, self.R, self.U))
            if cached is None or self.control != "fixed":
                q = self.query_code(F.linear(u if self.control == "fixed" else z, Q))
                if self.time_reads:
                    synchronize(u.device)
                    tick = time.perf_counter()
                    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if u.is_cuda else None
                    if events is not None:
                        events[0].record()
                m = state.read(q)
                if self.time_reads:
                    if events is not None:
                        events[1].record()
                    synchronize(u.device)
                    elapsed = time.perf_counter() - tick
                    self.cost["read_seconds"].append(elapsed)
                    self.cost["read_kernel_seconds"].append(events[0].elapsed_time(events[1]) / 1000 if events is not None else elapsed)
                self.cost["backend_calls"] += 1
                self.cost["backend_rows"] += len(u)
                self.cost["logical_reads"] += len(u)
                cached = (q, m)
            else:
                q, m = cached
            z = u + self.cfg.rho * self._phi(F.linear(z, R) + F.linear(m.to(u.dtype), U))
            if self.capture:
                mask = self._final_mask
                if mask is None:
                    mask = torch.ones(len(u), dtype=torch.bool, device=u.device)
                self.trace.append(dict(step=step + 1, owner=state.owner[mask].detach().clone(),
                                       question_row=self._active_rows[mask].detach().clone(),
                                       q=q[mask].detach().clone(), m=m[mask].detach().clone(),
                                       z=z[mask].detach().clone(), _u=u.detach().clone(),
                                       _before=z_before.detach().clone(), _q=q.detach().clone(),
                                       _m=m.detach().clone(), _z=z.detach().clone(), _view=state))
        return z


def configure(experiment, branch="advance", control="shared"):
    """Adopt only after strict original-parent loading. Keep original config for SQ2 validation/export."""
    rules.branch_arms(branch)
    initial = experiment.adapter
    adapter = LoopAdapter(initial, experiment.arm).to(device=experiment.device, dtype=torch.float32)
    if branch == "query_copy":
        adapter.add_query_projection()
    if control == "untied":
        adapter.untie()
    elif control in {"shared", "fixed"}:
        adapter.control = control
    else:
        raise ValueError("unknown control")
    experiment.adapter = experiment.writer = experiment.bank.modules[experiment.layer].adapter = adapter
    experiment.value_head = None
    experiment.standardize = parent._refuse_standardize
    experiment.s1b_branch, experiment.s1b_control = branch, control
    return experiment


def load_parent(branch="advance", *, control="shared", expected=None, model=None, tokenizer=None):
    experiment = parent.load_parent("S-RE", expected=expected, model=model, tokenizer=tokenizer)
    try:
        return configure(experiment, branch, control)
    except BaseException:
        experiment.close()
        raise


def per_hop(samples):
    correct, nll, hops = samples["correct"], samples["nll"], samples["hops"]
    if correct.ndim != 2 or nll.shape != correct.shape or len(hops) != correct.shape[1]:
        raise ValueError("S1B samples must retain episode and hop axes")
    if not np.isfinite(nll).all():
        raise FloatingPointError("nonfinite S1B NLL")
    return dict(n=len(correct), em=float(correct.mean()), nll=float(nll.astype(np.float64).mean()),
                per_hop={str(int(h)): dict(correct=int(correct[:, j].sum()), em=float(correct[:, j].mean()),
                                           nll=float(nll[:, j].astype(np.float64).mean())) for j, h in enumerate(hops)})


def path_indices(arrays, device):
    """Offline labels only: ordered gold edges, -1 after termination; never an argument of forward."""
    labels = np.full((*arrays.q_start.shape, 8), -1, dtype=np.int64)
    for i in range(len(labels)):
        graph = {(int(h), int(r)): (j, int(t)) for j, (h, r, t) in enumerate(arrays.facts[i]) if arrays.fact_valid[i, j]}
        for j, start in enumerate(arrays.q_start[i]):
            node = int(start)
            for s in range(int(arrays.q_hop_true[i, j])):
                fact, node = graph[node, 0]
                labels[i, j, s] = fact
    return torch.as_tensor(labels, device=device)


def trace_diagnostics(experiment, batch, arrays, spec, depth):
    """Group actual final-token reads by episode/query; no recomputed first-query surrogate."""
    B, Q = arrays.q_start.shape
    terms = {}
    labels = path_indices(arrays, experiment.device)
    adapter = experiment.adapter
    for s in range(1, depth + 1):
        events = [t for t in adapter.trace if t["step"] == s and len(t["owner"])]
        q = torch.cat([t["q"] for t in events])
        m = torch.cat([t["m"] for t in events])
        owner = torch.cat([t["owner"] for t in events])
        # Final token positions may vary; pair each event with its flattened question index.
        rows = torch.cat([t["question_row"] for t in events])
        order = rows.argsort()
        if not torch.equal(rows[order], torch.arange(B * Q, device=rows.device)):
            raise RuntimeError("final-token trajectory is incomplete or duplicated")
        q, m, owner = q[order], m[order], owner[order]
        scores = (batch.keys.index_select(0, owner) * q[:, None]).sum(-1)
        target = labels[..., min(s - 1, 7)].flatten() if s <= 8 else labels[..., -1].flatten().new_full((B * Q,), -1)
        valid = target >= 0
        clean = batch.values[owner, target.clamp_min(0)]
        error = (m - clean).square().sum(-1) / (clean.square().sum(-1) + 1e-6)
        for name, value in (("path_error", error), ("key_top1", (scores.argmax(-1) == target).float()),
                            ("cosine", F.cosine_similarity(m, clean)),
                            ("norm_ratio", m.norm(dim=-1) / clean.norm(dim=-1).clamp_min(1e-12))):
            terms[f"{name}_s{s}"] = torch.where(valid, value, torch.nan).reshape(B, Q).cpu().numpy()
        z = torch.cat([t["z"] for t in events])[order]
        terms[f"state_norm_s{s}"] = z.norm(dim=-1).reshape(B, Q).cpu().numpy()
        if s > 1:
            earlier = [t for t in adapter.trace if t["step"] == s - 1 and len(t["owner"])]
            prior_rows = torch.cat([t["question_row"] for t in earlier])
            prior_z = torch.cat([t["z"] for t in earlier])[prior_rows.argsort()]
            drift = (z - prior_z).norm(dim=-1).reshape(B, Q)
            ended = torch.as_tensor(arrays.q_hop_true < s, device=z.device)
            terms[f"post_completion_drift_s{s}"] = torch.where(ended, drift, torch.nan).cpu().numpy()
        if spec.kind == "soft":
            w = (scores / spec.tau).softmax(-1)
            terms[f"path_weight_s{s}"] = torch.where(valid, w.gather(1, target.clamp_min(0)[:, None]).squeeze(1), torch.nan).reshape(B, Q).cpu().numpy()
            terms[f"entropy_s{s}"] = (-torch.xlogy(w, w).sum(-1)).reshape(B, Q).cpu().numpy()
            top = w.topk(2).values
            terms[f"gap_s{s}"] = (top[:, 0] - top[:, 1]).reshape(B, Q).cpu().numpy()
    return terms


def verify_trace(adapter):
    """Re-evaluate full original read batches, retaining the exact operation shapes/dtypes."""
    for event in adapter.trace:
        step = event["step"] - 1
        Q, R, U = ((getattr(adapter, n) if step == 0 else getattr(adapter, n + "_steps")[step - 1]
                     for n in ("Q", "R", "U")) if adapter.control == "untied" else (adapter.Q, adapter.R, adapter.U))
        u, before = event["_u"], event["_before"]
        q = adapter.query_code(F.linear(u if adapter.control == "fixed" else before, Q))
        m = event["_view"].read(q)
        z = u + adapter.cfg.rho * adapter._phi(F.linear(before, R) + F.linear(m.to(u.dtype), U))
        if any(not torch.equal(a, b) for a, b in ((q, event["_q"]), (m, event["_m"]), (z, event["_z"]))):
            raise RuntimeError("captured S1B trajectory disagrees with recurrence/backend")
    return True


@torch.no_grad()
def evaluate(experiment, arrays, spec, depth, *, chunk=16, diagnostics=False, timing=False):
    values, prep_seconds, score_seconds = {}, 0., 0.
    costs = []
    for begin in range(0, len(arrays.episode_seed), chunk):
        part = _slice(arrays, slice(begin, begin + chunk))
        synchronize(experiment.device)
        tick = time.perf_counter()
        batch = experiment.prepare(part)
        synchronize(experiment.device)
        prep_seconds += time.perf_counter() - tick
        adapter = experiment.adapter
        adapter.reset_observation(capture=diagnostics, timing=timing)
        # This owner-free mapping identifies final question rows, including variable padding.
        adapter._question_rows = torch.arange(batch.question_mask.shape[0] * batch.question_mask.shape[1], device=experiment.device)
        tick = time.perf_counter()
        logits, _ = experiment.score(bind_batch(experiment, batch, spec), depth)
        synchronize(experiment.device)
        score_seconds += time.perf_counter() - tick
        nll = F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten(), reduction="none").reshape_as(batch.answer_ids)
        row = dict(correct=(logits.argmax(-1) == batch.answer_ids).cpu().numpy(), nll=nll.cpu().numpy(),
                   prediction=logits.argmax(-1).cpu().numpy(), answer=batch.answer_ids.cpu().numpy())
        if diagnostics:
            verify_trace(adapter)
            row.update(trace_diagnostics(experiment, batch, part, spec, depth))
        for name, value in row.items():
            values.setdefault(name, []).append(value)
        costs.append(copy.deepcopy(adapter.cost))
        adapter.reset_observation()
    sample = {name: np.concatenate(parts) for name, parts in values.items()}
    sample.update(episode_id=arrays.episode_seed.copy(), hops=arrays.q_hop_true[0].copy())
    if not np.all(arrays.q_hop_true == sample["hops"][None]):
        raise ValueError("evaluation questions are not hop-stratified")
    return dict(samples=sample, summary=per_hop(sample), prepare_seconds=prep_seconds, score_seconds=score_seconds,
                costs=costs, seconds=prep_seconds + score_seconds)


def metadata(experiment, arm, seed, spec, step, parent_digest, config_hash):
    from .run import source_metadata
    return dict(stage="S1B", arm=arm, seed=seed, branch=experiment.s1b_branch, control=experiment.s1b_control,
                trained_depth=rules.ARM[arm][1], read=spec.as_dict(), task_digest=data.LAYOUT_DIGEST,
                task=json.loads(json.dumps(asdict(data.task_spec()))), step=step,
                parent_writer_blake2b_128=parent_digest, science_hash=config_hash,
                prereg_blake2b_128=data.PREREG_DIGEST, source_commit=source_metadata()["git_commit"],
                key_dim=experiment.adapter.cfg.key_dim, value_dim=experiment.adapter.cfg.r_mem,
                feature_map=f"dpfp-nu{experiment.adapter.cfg.dpfp_nu}-unit", eps=experiment.adapter.cfg.eps,
                trainable=rules.trainable(experiment.s1b_branch, experiment.s1b_control))


def save_export(experiment, path, meta):
    torch.save(dict(s1b_version=1, config=experiment.config, arm=experiment.read_arm,
                    signature=experiment.writer_signature(), adapter=experiment.adapter.state_dict(), s1b=meta), path)
    return parent.file_digest(path)


def load_export(path, *, model=None, tokenizer=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload.get("s1b", {})
    if (payload.get("s1b_version") != 1 or meta.get("prereg_blake2b_128") != data.PREREG_DIGEST
            or meta.get("task_digest") != data.LAYOUT_DIGEST or meta.get("task") != json.loads(json.dumps(asdict(data.task_spec())))
            or any(n not in payload.get("adapter", {}) for n in ("input_mean", "input_std"))
            or set(payload) != {"s1b_version", "config", "arm", "signature", "adapter", "s1b"}):
        raise ValueError("incomplete S1B model export")
    experiment = sq2.SQ2Experiment(payload["config"], payload["arm"], model=model, tokenizer=tokenizer)
    try:
        if payload["signature"] != experiment.writer_signature():
            raise ValueError("S1B parent signature mismatch")
        configure(experiment, meta["branch"], meta["control"])
        if meta["trainable"] != rules.trainable(meta["branch"], meta["control"]) or meta["trained_depth"] != rules.ARM[meta["arm"]][1]:
            raise ValueError("S1B export architecture mismatch")
        experiment.adapter.load_state_dict(payload["adapter"], strict=True)
        experiment.s1b = meta
        experiment.read_spec = ReadSpec.from_dict(meta["read"])
    except BaseException:
        experiment.close()
        raise
    return experiment
