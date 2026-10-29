"""SLOT-LOOP-v1 preflight: the engineering gate of S0 (docs/prereg/SLOT-LOOP-v1.md, "Verification list" and decision R0).

Compute only. Each check returns a dict with `passed` and its evidence; `run_preflight` (in `slot_run.py`) runs them on the real
parents. The functions take the experiment (or the adapter) as an argument, so the tests run the same code on a tiny model.
Nothing here reads S0 calibration or dev data: the old range replays the recorded identity, the rest uses smoke sets.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from looplora.core import memory
from looplora.core.memory_backend import (CapacityError, HardSlotBackend, MatrixBackend, MatrixState, MemoryRead, SoftSlotBackend,
                                          build_slots, persistent_bytes, select_rows, state_digest)
from looplora.data.relations import generate
from looplora.sym.evaluate import stratify_queries
from .evaluate import _slice
from .read_fix import store_read
from .slot_experiment import (CHUNK, digest_report, evaluate, freeze_for_s1a, gradient_report, support_digest, tensor_digests,
                              trainable_report)
from .slot_read import ReadSpec, bind_batch, final_query
from .slot_rules import TRAINABLE
from .train import make_task_spec
from .writer_fix import gather_positions

# Recorded identity of the parents on the old val[0,1000) at seed 4242 (docs/prereg/SLOT-LOOP-v1.md, "S0": engineering gate R0).
RECORDED = {
    "S-RE": dict(correct=723, loss_float32=1.642127275466919, relative_read_error=0.19612866640090942, samples="val-6000/samples.npz"),
    "D128-3": dict(correct=608, loss_float32=2.1291537284851074, relative_read_error=0.36268818378448486,
                   samples="val-22370/samples.npz"),
}
SLOT_BYTES, MATRIX_BYTES = 57_368, 393_216  # per episode at S = 16, dk 768, dv 128, float32 (plan 10.1)
KEY_TOLERANCE, RECONSTRUCTION_TOLERANCE = 1e-5, 1e-6


def old_range(experiment):
    """`read_fix_run.validate`'s episodes: val [0, 1000) at seed 4242, one question each (the old protected range, identity use only)."""
    spec = make_task_spec(experiment.config)
    return stratify_queries(generate(spec, "val", 0, 1000, 4242), spec, [1], 4242)


@torch.no_grad()
def native_replay(experiment, name: str, folder) -> dict:
    """The parent's own read on the old range against its recorded run: counts, NLL, per-sample predictions, NLLs, read errors."""
    recorded, folder = RECORDED[name], Path(folder)
    old = np.load(folder / recorded["samples"], allow_pickle=False)
    result = evaluate(experiment, old_range(experiment), {"native": None}, diagnostics=True)
    sample = result["samples"]["native"]
    equal = lambda a, b: bool(np.array_equal(sample[a], old[b]))
    same = dict(episode_id=equal("episode_id", "episode_id"), prediction=equal("prediction", "prediction"),
                correct=equal("correct", "correct"), nll=equal("nll", "nll"), relative_read_error=equal("rel_err", "relative_read_error"))
    if "key_top1" in old.files:
        same.update(key_top1=bool(np.array_equal(result["common"]["key_top1"], old["key_top1"])),
                    key_margin=bool(np.array_equal(result["common"]["key_margin"], old["key_margin"])))
    summary = dict(correct=int(sample["correct"].sum()), loss_float32=float(sample["nll"].mean()),
                   relative_read_error=float(sample["rel_err"].mean()))
    expected = {k: recorded[k] for k in ("correct", "loss_float32", "relative_read_error")}
    differences = {k: (summary[k], expected[k]) for k in expected if summary[k] != expected[k]}
    return dict(passed=not differences and all(same.values()), replayed=summary, recorded=expected,
                per_sample_bitwise=same, differences=differences, seconds=result["seconds"]["native"] + result["prepare_seconds"])


@torch.no_grad()
def matrix_identity(experiment, arrays, *, depths=(1,), chunk: int = CHUNK) -> dict:
    """Matrix (g = 1) logits against the unbound native logits on the same prepared batches: bitwise, or the measured maximum."""
    spec = ReadSpec("matrix")
    worst, unequal, compared, chunks = 0.0, 0, 0, 0
    for begin in range(0, len(arrays.episode_seed), chunk):
        batch = experiment.prepare(_slice(arrays, slice(begin, min(begin + chunk, len(arrays.episode_seed)))))
        bound = bind_batch(experiment, batch, spec)
        for depth in depths:
            native, _ = experiment.score(batch, depth)
            matrix, _ = experiment.score(bound, depth)
            chunks += 1
            compared += native.numel()
            if not torch.equal(native, matrix):
                unequal += int((native != matrix).sum())
                worst = max(worst, float((native.float() - matrix.float()).abs().max()))
    return dict(passed=unequal == 0, max_abs_difference=worst, unequal_elements=unequal, elements=compared, chunks=chunks,
                depths=list(depths), episodes=len(arrays.episode_seed))


def matrix_gradients(experiment, arrays) -> dict:
    """Descriptive: the gradients of one training-shaped loss through the unbound native path and through Matrix (g = 1)."""
    spec = ReadSpec("matrix")
    names = freeze_for_s1a(experiment)
    adapter = experiment.adapter
    out = {}
    for label, bind in (("native", False), ("matrix", True)):
        for p in names:
            p.grad = None
        with torch.no_grad():
            batch = experiment.prepare(arrays)
        logits, _ = experiment.score(bind_batch(experiment, batch, spec) if bind else batch, 1)
        F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten()).backward()
        out[label] = {n: getattr(adapter, n).grad.clone() for n in TRAINABLE}
    for p in names:
        p.grad = None
    worst = {n: float((out["native"][n] - out["matrix"][n]).abs().max()) for n in TRAINABLE}
    return dict(max_abs_difference=worst, bitwise=all(torch.equal(out["native"][n], out["matrix"][n]) for n in TRAINABLE))


@torch.no_grad()
def owner_mask_check(adapter, *, device, episodes: int = 3, questions: int = 4, length: int = 6, facts: int = 4, seed: int = 0) -> dict:
    """Distinguishable sentinel memories, rows expanded per question, padded (left, right, hole) and reordered: only the owner's
    rows react to a change of its memory, rows are independent of their batch position, padding positions are exactly zero."""
    g = torch.Generator().manual_seed(seed)
    cfg = adapter.cfg
    dk, dv = cfg.key_dim, cfg.r_mem
    unit = lambda t: t / t.norm(dim=-1, keepdim=True)
    keys = unit(torch.rand(episodes, facts, dk, generator=g)).to(device)
    sentinel = (torch.arange(episodes, dtype=torch.float32)[:, None, None] + 1) * 100
    values = (sentinel + torch.randn(episodes, facts, dv, generator=g)).to(device)
    matrix = torch.einsum("bnv,bnk->bvk", values, keys)
    rows = episodes * questions
    x = torch.randn(rows, length, cfg.d_in, generator=g).to(device)
    valid = torch.ones(rows, length, dtype=torch.bool, device=device)
    valid[1, length - 2:], valid[2, :2], valid[5, 3] = False, False, False
    owner = torch.arange(episodes, device=device).repeat_interleave(questions)
    perm = torch.randperm(rows, generator=g).to(device)
    backends = {"matrix": (MatrixBackend(), MatrixState(matrix)), "soft": (SoftSlotBackend(0.3, 1.5), build_slots(keys, values, facts)),
                "hard": (HardSlotBackend(), build_slots(keys, values, facts))}

    def run(backend, state, x_rows, valid_rows, owners):
        return adapter.process_sequence(x_rows, MemoryRead(backend, state, owners), 1, valid_rows)[0]

    report, passed = {}, True
    for name, (backend, state) in backends.items():
        base = run(backend, state, x, valid, owner)
        scale = float(base.abs().max()) or 1.0
        reference = torch.stack([run(backend, select_rows(state, owner[r:r + 1]), x[r:r + 1], valid[r:r + 1],
                                     torch.zeros(1, dtype=torch.long, device=device))[0] for r in range(rows)])
        reference_gap = float((base - reference).abs().max()) / scale
        reorder_gap = float((run(backend, state, x[perm], valid[perm], owner[perm]) - base[perm]).abs().max()) / scale
        padding_zero = bool((base[~valid] == 0).all())
        untouched = changed = True
        for j in range(episodes):
            tensors = {k: v.clone() for k, v in state.tensors().items()}
            tensors["matrix" if name == "matrix" else "values"][j] += 500.0
            other = replace(state, **tensors)
            moved = run(backend, other, x, valid, owner)
            untouched &= bool(torch.equal(moved[owner != j], base[owner != j]))
            changed &= bool(((moved[owner == j] != base[owner == j]) & valid[owner == j][..., None]).any())
        ok = reference_gap < 1e-4 and reorder_gap < 1e-4 and padding_zero and untouched and changed
        passed &= ok
        report[name] = dict(passed=ok, reference_gap=reference_gap, reorder_gap=reorder_gap, padding_zero=padding_zero,
                            other_episodes_bitwise_untouched=untouched, owner_rows_react=changed)
    return dict(passed=passed, backends=report, rows=rows, episodes=episodes, questions=questions)


@torch.no_grad()
def isolation_check(experiment, arrays) -> dict:
    """Read-only isolation: the state, keys and values keep their digests over several questions, loop depths and read
    conditions; every backend refuses `online_write`."""
    batch = experiment.prepare(arrays)
    layer = experiment.layer
    matrix = batch.snapshot.states[layer]
    slots = build_slots(batch.keys, batch.values, 16)
    held = lambda: dict(matrix=state_digest(MatrixState(matrix)), slots=state_digest(slots), keys=state_digest(MatrixState(batch.keys)),
                        values=state_digest(MatrixState(batch.values)))
    before = held()
    specs = [ReadSpec("matrix", 1.5), ReadSpec("soft", tau=0.1), ReadSpec("soft", 2.0, tau=1.0, recency=True), ReadSpec("hard"),
             ReadSpec("soft", tau=0.3, scope="final_only"), ReadSpec("clean_final", scope="final_only")]
    reads = 0
    for spec in specs:
        if spec.kind == "clean_final" and batch.question_ids.shape[1] != 1:
            continue
        bound = bind_batch(experiment, batch, spec)
        for depth in (1, 2, 4):
            experiment.score(bound, depth)
            reads += 1
    refused = []
    for backend in (MatrixBackend(), SoftSlotBackend(1.0), HardSlotBackend()):
        try:
            backend.online_write(None, None, None)
        except NotImplementedError:
            refused.append(backend.name)
    after = held()
    return dict(passed=before == after and len(refused) == 3, digests_equal=before == after, online_write_refused=refused,
                scored=reads, before=before, after=after)


@torch.no_grad()
def feature_check(experiment, arrays) -> dict:
    """The captured keys are the unit DPFP features of the recomputed K u (768-d), `M q` is the stored coefficient sum
    (`gram_relative_error`), the values are the raw written ones, and the reads never touch `base_keys`."""
    adapter, layer = experiment.adapter, experiment.layer
    batch = experiment.prepare(arrays)
    fact_x = gather_positions(batch.raw_support, batch.positions[:, :, 0])
    raw = F.linear(adapter.u(fact_x), adapter.K)
    unit = lambda k: k / torch.linalg.vector_norm(k, dim=-1, keepdim=True).clamp_min(adapter.cfg.eps)
    recomputed = unit(memory.dpfp_features(raw, adapter.cfg.dpfp_nu)) if adapter.cfg.dpfp_nu else unit(raw)
    key_gap = float((recomputed - batch.keys).abs().max())
    _, x = experiment.score(batch, 1, capture_inputs=True)
    q = final_query(experiment, batch, x)
    exact = store_read(adapter, batch.keys, batch.values, q)
    stored = torch.einsum("bij,bqj->bqi", batch.snapshot.states[layer], q)
    reconstruction = float((exact["reconstructed"] - stored).norm(dim=-1).div(stored.norm(dim=-1).clamp_min(1e-8)).max())
    specs = [ReadSpec("matrix"), ReadSpec("soft", tau=0.3), ReadSpec("hard")]
    poisoned = replace(batch, base_keys=None if batch.base_keys is None else torch.full_like(batch.base_keys, float("nan")))
    base_unused = True
    for spec in specs:
        a, _ = experiment.score(bind_batch(experiment, batch, spec), 1)
        b, _ = experiment.score(bind_batch(experiment, poisoned, spec), 1)
        base_unused &= bool(torch.equal(a, b)) and bool(torch.isfinite(b).all())
    value_gap = float((F.linear(adapter.u(fact_x), adapter.V) - batch.values).abs().max() / batch.values.abs().max())
    dims = dict(key_dim=int(batch.keys.shape[-1]), value_dim=int(batch.values.shape[-1]), keys_unit=bool(
        torch.allclose(batch.keys.norm(dim=-1), torch.ones_like(batch.keys[..., 0]), atol=1e-5)))
    passed = (key_gap <= KEY_TOLERANCE and value_gap <= KEY_TOLERANCE and reconstruction <= RECONSTRUCTION_TOLERANCE and base_unused
              and dims["keys_unit"])
    return dict(passed=passed, key_max_abs_gap=key_gap, value_max_relative_gap=value_gap, reconstruction_max_relative_error=reconstruction,
                base_keys_unused=base_unused, **dims, key_tolerance=KEY_TOLERANCE, reconstruction_tolerance=RECONSTRUCTION_TOLERANCE)


@torch.no_grad()
def bytes_check(experiment, arrays, capacity: int = 16) -> dict:
    """`numel * element_size` over the slot state (keys, values, valid mask, count) and the matrix state, per episode."""
    batch = experiment.prepare(arrays)
    rows = batch.keys.shape[0]
    slot = persistent_bytes(build_slots(batch.keys, batch.values, capacity)) // rows
    matrix = persistent_bytes(MatrixState(batch.snapshot.states[experiment.layer])) // rows
    dk, dv = batch.keys.shape[-1], batch.values.shape[-1]
    formula = 4 * capacity * (dk + dv) + capacity + 8
    registered = capacity == 16 and (dk, dv) == (768, 128)
    return dict(passed=slot == formula and matrix == 4 * dk * dv and (not registered or (slot, matrix) == (SLOT_BYTES, MATRIX_BYTES)),
                slot_bytes_per_episode=slot, matrix_bytes_per_episode=matrix, formula_slot_bytes=formula, capacity=capacity)


@torch.no_grad()
def capacity_check(experiment, arrays, capacity: int = 16) -> dict:
    """E = S passes and keeps every fact; E > S (fewer slots, or one more fact) raises and never truncates."""
    batch = experiment.prepare(arrays)
    facts = batch.keys.shape[1]
    exact = build_slots(batch.keys, batch.values, facts)
    fits = bool((exact.count == facts).all()) and torch.equal(exact.keys, batch.keys) and torch.equal(exact.values, batch.values)
    raised = []
    for label, keys, values, slots in (("S = E - 1", batch.keys, batch.values, facts - 1),
                                       ("E + 1 facts, S = E", torch.cat((batch.keys, batch.keys[:, :1]), 1),
                                        torch.cat((batch.values, batch.values[:, :1]), 1), facts)):
        try:
            build_slots(keys, values, slots)
        except CapacityError:
            raised.append(label)
    return dict(passed=fits and len(raised) == 2, facts=facts, exact_fit_keeps_all=fits, raised=raised)


def frozen_set_check(experiment, support, training, spec: ReadSpec | None = None, steps: int = 2, lr: float = 1e-3) -> dict:
    """Checks (i) to (iv) of the S1A freeze verification, with a few optimizer steps. Destructive: the weights move, so the
    caller discards the experiment."""
    spec = ReadSpec("soft", tau=0.3) if spec is None else spec
    params = freeze_for_s1a(experiment)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    report = trainable_report(experiment, optimizer)
    reference = tensor_digests(experiment.adapter)
    before = support_digest(experiment, support)
    gradients = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            batch = experiment.prepare(training)
        logits, _ = experiment.score(bind_batch(experiment, batch, spec), 1)
        F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten()).backward()
        gradients = gradients or gradient_report(experiment)
        optimizer.step()
    digests = digest_report(experiment.adapter, reference, trained=True)
    after = support_digest(experiment, support)
    return dict(passed=before == after, trainable_report=report, gradients=gradients, support=dict(before=before, after=after),
                frozen_unchanged=digests["frozen_unchanged"], trainable_changed=digests["trainable_changed"], steps=steps)
