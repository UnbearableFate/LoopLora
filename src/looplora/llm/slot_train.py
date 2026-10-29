"""SLOT-LOOP-v1 S1A: the matched continuation of the reader Q, R, U, B from the S-RE state (docs/prereg/SLOT-LOOP-v1.md, "S1A").

Compute only. One `fit` serves the arm-seed runs, the LR pilot, the ablation and the smoke: answer-only full-vocabulary
cross-entropy at L = 1 on all valid question tokens, a fresh AdamW, a 20-step linear warm-up and then a constant learning rate,
global gradient-norm clipping at 1.0 with `error_if_nonfinite`, the dev evaluation and the checkpoint selection of the registered
rule, and the freeze verification (i) to (iv) of the registration. Every step's loss and pre-clip gradient norm are kept
(`train_steps.npz`); the log rows carry the pre-clip norm and the clipped flag, the summary the clipped-step fraction.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import slot_data
from .prefetch import BatchStream
from .slot_experiment import (digest_report, evaluate, freeze_for_s1a, gradient_report, save_export, summarize,
                              support_digest, tensor_digests, trainable_report)
from .slot_read import ReadSpec, bind_batch
from .slot_rules import (CLIP, LOG_EVERY, OPTIMIZER, SELECTION_TIE, TRAINABLE, WARMUP, select_checkpoint)
from .train import synchronize


def dev_row(experiment, spec: ReadSpec, arrays, step: int, selectable: bool) -> dict:
    """One evaluation of the arm's read on `arrays` (one question per episode): the registered dev row."""
    synchronize(experiment.device)
    tick = time.perf_counter()
    result = evaluate(experiment, arrays, {"read": spec}, diagnostics=False)
    synchronize(experiment.device)
    summary = summarize(result["samples"]["read"])
    return dict(step=step, selectable=selectable, n=summary["n"], correct=summary["correct"], em=summary["em"],
                nll=summary["nll"], loss_float32=summary["loss_float32"], seconds=time.perf_counter() - tick)


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def fit(experiment, spec: ReadSpec, *, steps: int, lr: float, stream_seed: int, out, dev=None, dev_steps=(), support=None,
        export: dict | None = None, trainable=TRAINABLE, workers: int = 2, time_limit: float | None = None,
        source=None) -> dict:
    """Train `trainable` adapter tensors for `steps` steps and return the run summary (also written as `training.json`).

    dev: evaluation arrays (one question per episode); evaluated at step 0 (recorded, not selectable) and at `dev_steps`.
    support: arrays of the fixed support batch of check (iii). export: {arm, parent, run} enables `selected.pt` (the best
    selectable checkpoint so far; a later one replaces it only if lower by more than the registered 1e-6). source: an iterator
    of (arrays, tokens) replacing the training stream (the smoke); stream_seed is then only recorded.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = experiment.device
    params = freeze_for_s1a(experiment, trainable)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=tuple(OPTIMIZER["betas"]), eps=OPTIMIZER["eps"],
                                  weight_decay=OPTIMIZER["weight_decay"])
    freeze = dict(trainable=trainable_report(experiment, optimizer, trainable))
    reference = tensor_digests(experiment.adapter)
    support_before = None if support is None else support_digest(experiment, support)
    evaluations: list[dict] = []
    best_nll, best_step, selected_digest = math.inf, None, None
    if dev is not None:
        evaluations.append(dev_row(experiment, spec, dev, 0, False))
    losses, norms = torch.zeros(steps, device=device), torch.zeros(steps, device=device)
    durations: list[float] = []
    stream = (BatchStream(slot_data.task_spec(), "train", slot_data.BATCH, stream_seed, steps, tokenizer=experiment.tokenizer,
                          workers=workers) if source is None else contextlib.nullcontext(source))
    synchronize(device)
    started = time.perf_counter()
    with stream as batches, (out / "train_log.jsonl").open("w") as log:
        for step in range(1, steps + 1):
            tick = time.perf_counter()
            arrays, tokens = next(batches)
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, step / WARMUP)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():  # the writer is frozen: the support is a constant of the step
                batch = experiment.prepare(arrays, tokens=tokens)
            logits, _ = experiment.score(bind_batch(experiment, batch, spec), 1)
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            if step == 1:
                freeze["gradients"] = gradient_report(experiment, trainable)
            grad = torch.nn.utils.clip_grad_norm_(params, CLIP, error_if_nonfinite=True)  # the pre-clip global norm
            optimizer.step()
            synchronize(device)
            duration = time.perf_counter() - tick
            durations.append(duration)
            losses[step - 1], norms[step - 1] = loss.detach(), grad.detach()
            checkpoint = dev is not None and step in dev_steps
            if step == 1 or step % LOG_EVERY == 0 or step == steps or checkpoint:
                row = dict(step=step, loss=float(loss.detach()), accuracy=float((logits.argmax(-1) == batch.answer_ids).float().mean()),
                           grad_norm=float(grad), clipped=bool(grad > CLIP), lr=optimizer.param_groups[0]["lr"],
                           step_seconds=duration, data_seed=stream_seed)
                if not _finite(row["loss"], row["grad_norm"]):
                    raise FloatingPointError(f"non-finite logged value at step {step}")
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
            if checkpoint:
                row = dev_row(experiment, spec, dev, step, True)
                digest_report(experiment.adapter, reference, trainable, trained=True)
                evaluations.append(row)
                if export is not None and _finite(row["nll"]) and row["nll"] < best_nll - SELECTION_TIE:
                    best_nll, best_step = row["nll"], step
                    selected_digest = save_export(experiment, out / "selected.pt", step=step, spec=spec, trainable=trainable,
                                                  **export)
            if time_limit is not None and step != steps and time.perf_counter() - started > time_limit:
                raise RuntimeError("the run exceeded its measured allowance")
    synchronize(device)
    train_seconds = time.perf_counter() - started
    final = digest_report(experiment.adapter, reference, trainable, trained=True)
    freeze["digests"] = dict(frozen_unchanged=final["frozen_unchanged"], trainable_changed=final["trainable_changed"],
                             before=reference, after=final["digests"])
    if support is not None:
        after = support_digest(experiment, support)
        if after != support_before:
            raise RuntimeError("the support keys, values or matrix state changed during training")
        freeze["support"] = dict(before=support_before, after=after, equal=True)
    selectable = [(r["step"], r["nll"]) for r in evaluations if r["selectable"]]
    if export is not None:
        chosen = select_checkpoint(selectable)
        if chosen != best_step:
            raise RuntimeError("the incremental checkpoint choice differs from the registered selection rule")
    clipped = norms > CLIP
    steady = durations[10:] or durations
    host_norms, host_losses = norms.cpu().numpy(), losses.cpu().numpy()
    np.savez_compressed(out / "train_steps.npz", loss=host_losses, grad_norm=host_norms)
    summary = dict(
        steps=steps, lr=lr, stream_seed=stream_seed, warmup=WARMUP, clip=CLIP, batch=slot_data.BATCH, optimizer=dict(OPTIMIZER),
        trainable=list(trainable), trainable_parameters=int(sum(p.numel() for p in params)),
        frozen_parameters=freeze["trainable"]["frozen_parameters"],
        all_finite=bool(np.isfinite(host_losses).all() and np.isfinite(host_norms).all()),
        clipped_steps=int(clipped.sum()), clipped_fraction=float(clipped.float().mean()),
        grad_norm=dict(max=float(host_norms.max()), mean=float(host_norms.mean()), median=float(np.median(host_norms)),
                       p95=float(np.quantile(host_norms, 0.95))),
        loss=dict(first=float(host_losses[:10].mean()), last=float(host_losses[-10:].mean())),
        step_seconds_mean=float(np.mean(steady)), step_seconds_p95=float(np.quantile(durations, 0.95)), train_seconds=train_seconds,
        dev=evaluations, selected_step=best_step, selected_checkpoint_blake2b_128=selected_digest, freeze=freeze)
    (out / "training.json").write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return summary
