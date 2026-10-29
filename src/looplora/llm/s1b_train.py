"""Matched S1B depth training. Only the runner can authorize formal configurations."""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import s1b_data as data, s1b_rules as rules, s1b_model as model
from .prefetch import BatchStream
from .slot_experiment import (freeze_for_s1a, trainable_report, tensor_digests, digest_report,
                              gradient_report, support_digest)
from .slot_read import bind_batch
from .train import synchronize


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


def fit(experiment, spec, *, arm, seed, lr, steps, out, dev, dev_steps, support, parent_digest,
        science_hash, stream_seed=None, workers=2, source=None):
    """Private primitive shared by registered smoke, pilot, main and gated controls."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    setup_tick = time.perf_counter()
    depth = rules.ARM[arm][1]
    names = rules.trainable(experiment.s1b_branch, experiment.s1b_control)
    params = freeze_for_s1a(experiment, names)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    freeze = dict(trainable=trainable_report(experiment, optimizer, names))
    reference = tensor_digests(experiment.adapter)
    before = support_digest(experiment, support)
    rows, durations, losses, norms, costs = [], [], [], [], []
    export_seconds = []
    checkpoint_overhead = []
    best, best_step, selected_digest = math.inf, None, None

    def evaluate_dev(step):
        result = model.evaluate(experiment, dev, spec, depth)
        row = dict(result["summary"], step=step, seconds=result["seconds"], selectable=step != 0)
        if not math.isfinite(row["nll"]):
            raise FloatingPointError("nonfinite dev evaluation")
        meta = model.metadata(experiment, arm, seed, spec, step, parent_digest, science_hash)
        export_tick = time.perf_counter()
        candidate = out / "checkpoints" / f"step-{step:06d}.pt"
        digest = model.save_export(experiment, candidate, meta)
        sample_file = candidate.with_suffix(".npz")
        np.savez_compressed(sample_file, **result["samples"], checkpoint_blake2b_128=np.asarray(digest), step=np.asarray(step))
        row["checkpoint"] = dict(step=step, file=str(candidate.relative_to(out)), blake2b_128=digest,
                                 samples=str(sample_file.relative_to(out)), samples_blake2b_128=data.old.file_digest(sample_file))
        row["export_seconds"] = time.perf_counter() - export_tick
        export_seconds.append(row["export_seconds"])
        rows.append(row)
        return row

    setup_seconds = time.perf_counter() - setup_tick
    initial_row = evaluate_dev(0)
    setup_seconds += initial_row["export_seconds"]
    stream_seed = data.TRAIN_BASE[seed] if stream_seed is None else stream_seed
    stream = (BatchStream(data.task_spec(), "train", data.BATCH, stream_seed, steps,
                          tokenizer=experiment.tokenizer, workers=workers) if source is None else contextlib.nullcontext(source))
    synchronize(experiment.device)
    tick_all = time.perf_counter()
    with stream as batches, (out / "train_log.jsonl").open("w") as log:
        for step in range(1, steps + 1):
            tick = time.perf_counter()
            arrays, tokens = next(batches)
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]["lr"] = lr * min(1., step / 20)
            with torch.no_grad():
                batch = experiment.prepare(arrays, tokens=tokens)
            experiment.adapter.reset_observation()
            logits, _ = experiment.score(bind_batch(experiment, batch, spec), depth)
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError("nonfinite training gradient")
            if step == 1:
                freeze["gradients"] = gradient_report(experiment, names)
            try:
                grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            except RuntimeError as error:
                if "non-finite" not in str(error):
                    raise
                raise FloatingPointError("nonfinite global gradient norm") from error
            optimizer.step()
            synchronize(experiment.device)
            durations.append(time.perf_counter() - tick)
            losses.append(float(loss.detach()))
            norms.append(float(grad))
            costs.append(dict(experiment.adapter.cost))
            if step in dev_steps:
                row = evaluate_dev(step)
                overhead_tick = time.perf_counter()
                digest_report(experiment.adapter, reference, names, trained=True)
                if support_digest(experiment, support) != before:
                    raise RuntimeError("writer/support changed at checkpoint")
                if row["nll"] < best - 1e-6:
                    best, best_step = row["nll"], step
                    shutil.copyfile(out / row["checkpoint"]["file"], out / "selected.pt")
                    selected_digest = row["checkpoint"]["blake2b_128"]
                checkpoint_overhead.append(time.perf_counter() - overhead_tick + row["export_seconds"])
            if step == 1 or step % 50 == 0 or step == steps or step in dev_steps:
                log.write(json.dumps(dict(step=step, loss=losses[-1], grad_norm=norms[-1],
                    clipped=norms[-1] > 1., lr=optimizer.param_groups[0]["lr"], seconds=durations[-1])) + "\n")
                log.flush()
    finalize_tick = time.perf_counter()
    freeze["digests"] = digest_report(experiment.adapter, reference, names, trained=True)
    after = support_digest(experiment, support)
    freeze["support"] = dict(before=before, after=after, equal=before == after)
    if before != after:
        raise RuntimeError("writer/support changed during S1B training")
    chosen = rules.select_checkpoint([(r["step"], r["nll"]) for r in rows if r["selectable"]])
    if chosen != best_step:
        raise RuntimeError("online checkpoint selection disagrees with registered rule")
    if data.old.file_digest(out / "selected.pt") != selected_digest:
        raise RuntimeError("selected export bytes do not match the winning dev row")
    np.savez_compressed(out / "train_steps.npz", loss=np.asarray(losses), grad_norm=np.asarray(norms),
                        step_seconds=np.asarray(durations))
    record = dict(status="complete", stage="S1B", arm=arm, seed=seed, steps=steps, lr=lr,
                  artifact_root=str(out.resolve()),
                  branch=experiment.s1b_branch, control=experiment.s1b_control, trained_depth=depth,
                  stream_seed=stream_seed, train_episodes=steps * data.BATCH, train_questions=steps * data.BATCH * 4,
                  all_finite=bool(np.isfinite(losses).all() and np.isfinite(norms).all()), dev=rows,
                  selected_step=best_step, selected_checkpoint_blake2b_128=selected_digest,
                  science_hash=science_hash, freeze=freeze, clipped_fraction=float(np.mean(np.asarray(norms) > 1.)),
                  step_seconds=float(np.mean(durations[10:] or durations)),
                  step_p90_seconds=float(np.quantile(durations, .9)), train_seconds=time.perf_counter() - tick_all,
                  setup_seconds=setup_seconds, checkpoint_overhead_seconds=max(checkpoint_overhead, default=0.),
                  finalize_seconds=time.perf_counter() - finalize_tick,
                  export_seconds=sum(export_seconds), dev_ids_blake2b_128=data.old.ids_digest(dev.episode_seed),
                  costs={k: sum(c[k] for c in costs) for k in ("valid_read_positions", "logical_reads", "backend_calls", "backend_rows", "recurrent_updates")})
    dump(out / "record.json", record)
    return record
