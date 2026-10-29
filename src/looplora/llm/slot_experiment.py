"""SLOT-LOOP-v1: the parents, the evaluation of read conditions, the freeze checks and the S1A export (docs/prereg/SLOT-LOOP-v1.md).

Compute only (imports torch). S-RE and D128-3 are loaded through their own loaders (`sq2.load_inference`,
`dpfp_run.load_inference`: strict, buffers included) and then given the slot-capable adapter with identical weights; nothing
here calls `WriterFixExperiment.standardize()`, which would overwrite `input_mean` and `input_std`.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.sym.evaluate import stratify_queries
from . import dpfp_run, slot_data, sq2
from .evaluate import _slice
from .run import source_metadata
from .slot_read import ReadSpec, attach_slot_reader, bind_batch, final_query, final_read
from .slot_rules import BUFFERS, TRAINABLE
from .train import make_task_spec, synchronize
from .writer_fix import true_fact_indices

PARENT_FOLDERS = slot_data.PARENT_FOLDERS
CHUNK = 16  # `read_fix_run.validate`'s evaluation batch: the identity replays use the same batches
EXPORT_VERSION = 1


def file_digest(path) -> str:
    return slot_data.file_digest(path)


def _refuse_standardize():
    raise RuntimeError("standardize() would overwrite input_mean/input_std; the slot study keeps the parent's buffers")


def adopt(experiment):
    """Give a loaded S-RE / D128-3 experiment the slot reader and forbid `standardize()` (it would overwrite the buffers)."""
    attach_slot_reader(experiment)
    experiment.value_head = None
    experiment.standardize = _refuse_standardize
    experiment.read_spec = None
    return experiment


def load_parent(name: str, *, model=None, tokenizer=None, expected: dict | None = None):
    """The parent as an experiment with the slot-capable adapter. `expected` maps file name to a blake2b-128 digest."""
    folder = Path(PARENT_FOLDERS[name])
    for file, digest in (expected or {}).items():
        found = file_digest(folder / file)
        if found != digest:
            raise ValueError(f"{name}/{file}: blake2b-128 {found} differs from the expected {digest}")
    loader = {"S-RE": sq2.load_inference, "D128-3": dpfp_run.load_inference}[name]
    experiment = loader(folder / "writer.pt", model=model, tokenizer=tokenizer)
    try:
        if make_task_spec(experiment.config) != slot_data.task_spec():
            raise ValueError(f"{name}: its task is not the registered T1/E16 layout")
        adopt(experiment)
    except BaseException:
        experiment.close()
        raise
    return experiment


def load_value_head(name: str, device):
    """The parent's training-only value head from its checkpoint, in a separate module (descriptive audit, never a read path)."""
    payload = torch.load(Path(PARENT_FOLDERS[name]) / "checkpoint.pt", map_location="cpu", weights_only=True)
    head = torch.nn.Linear(*reversed(payload["value_head"]["weight"].shape))
    head.load_state_dict(payload["value_head"])
    return head.to(device).eval()


def raw_episodes(eval_set: slot_data.EvalSet):
    """The set's episodes with the generator's four questions each (the training format)."""
    if eval_set.sealed and not eval_set.opened:
        raise slot_data.SealedSetError(f"{eval_set.name} has not been opened through eval_set")
    return generate(slot_data.task_spec(), eval_set.split, eval_set.start, eval_set.count, eval_set.base_seed)


def episodes(eval_set: slot_data.EvalSet):
    """The set's episodes with one question each (T1, hop 1); the query seed is the set's own base seed."""
    return stratify_queries(raw_episodes(eval_set), slot_data.task_spec(), [1], eval_set.base_seed)


@torch.no_grad()
def evaluate(experiment, arrays, conditions: dict, *, chunk: int = CHUNK, diagnostics: bool = True, value_head=None) -> dict:
    """Per-sample records of every condition on the same prepared batches (support encoded once per chunk).

    `conditions` maps a name to a ReadSpec, or to None for the unbound native read. Returns {"samples": name -> field -> array,
    "common": the fields that do not depend on the read (key top-1 and margin), "seconds": name -> bind and score time,
    "prepare_seconds": the support encoding time}.
    """
    if arrays.q_start.shape[1] != 1:
        raise ValueError("evaluation has one question per episode")
    device = experiment.device
    count = len(arrays.episode_seed)
    samples = {name: {} for name in conditions}
    common: dict[str, list] = {}
    seconds, prepare_seconds = dict.fromkeys(conditions, 0.0), 0.0

    def keep(store: dict, field: str, value) -> None:
        store.setdefault(field, []).append(np.asarray(value.detach().cpu() if torch.is_tensor(value) else value).ravel())

    for begin in range(0, count, chunk):
        part = _slice(arrays, slice(begin, min(begin + chunk, count)))
        synchronize(device)
        tick = time.perf_counter()
        batch = experiment.prepare(part)
        synchronize(device)
        prepare_seconds += time.perf_counter() - tick
        answers = torch.as_tensor(part.answer[:, 0], device=device, dtype=torch.long)
        target = true_fact_indices(part, device)
        for index, (name, spec) in enumerate(conditions.items()):
            synchronize(device)
            tick = time.perf_counter()
            bound = batch if spec is None else bind_batch(experiment, batch, spec)
            logits, x = experiment.score(bound, 1, capture_inputs=diagnostics)
            synchronize(device)
            seconds[name] += time.perf_counter() - tick
            store = samples[name]
            prediction = logits.argmax(-1)
            keep(store, "episode_id", part.episode_seed)
            keep(store, "answer", batch.answer_ids)
            keep(store, "prediction", prediction)
            keep(store, "correct", prediction == batch.answer_ids)
            keep(store, "nll", F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten(), reduction="none"))
            if not diagnostics:
                continue
            terms = final_read(experiment, batch, x, spec)
            for field in ("rel_err", "cos", "norm_ratio", "w_true", "w_gap", "w_entropy", "slot_is_true"):
                if field in terms:
                    keep(store, field, terms[field])
            if value_head is not None:
                keep(store, "clean_value_correct", value_head(terms["clean"]).argmax(-1) == answers[:, None])
                keep(store, "read_value_correct", value_head(terms["m"]).argmax(-1) == answers[:, None])
            if index == 0:
                scores = torch.einsum("bqk,bek->bqe", final_query(experiment, batch, x), batch.keys)
                actual = scores.gather(-1, target[..., None]).squeeze(-1)
                other = scores.masked_fill(F.one_hot(target, scores.shape[-1]).bool(), -torch.inf).amax(-1)
                keep(common, "key_top1", scores.argmax(-1) == target)
                keep(common, "key_margin", actual - other)
    pack = lambda store: {field: np.concatenate(parts) for field, parts in store.items()}
    return dict(samples={name: pack(store) for name, store in samples.items()}, common=pack(common), seconds=seconds,
                prepare_seconds=prepare_seconds, count=count)


def summarize(samples: dict) -> dict:
    """Means of one condition's per-sample fields; `nll` is the float64 mean, `loss_float32` the float32 mean that
    `read_fix_run.validate` reports (the identity replays compare against it)."""
    out = dict(n=int(len(samples["correct"])), correct=int(samples["correct"].sum()), em=float(samples["correct"].mean()),
               nll=float(samples["nll"].astype(np.float64).mean()), loss_float32=float(samples["nll"].mean()))
    for field, values in samples.items():
        if field not in {"episode_id", "answer", "prediction", "correct", "nll"}:
            out[field] = float(values.astype(np.float64).mean())
    return out


# ------------------------------------------------------------------------------------------------------ freeze checks

def freeze_for_s1a(experiment, trainable=TRAINABLE) -> list:
    """Only `trainable` adapter parameters keep requires_grad (in the given order); returns them."""
    named = dict(experiment.adapter.named_parameters())
    if set(trainable) - set(named):
        raise ValueError(f"unknown trainable parameters: {sorted(set(trainable) - set(named))}")
    for name, parameter in named.items():
        parameter.requires_grad_(name in trainable)
    return [named[name] for name in trainable]


def tensor_digests(adapter) -> dict[str, str]:
    """blake2b-128 of the raw bytes of every parameter and buffer, by name."""
    out = {}
    for name, tensor in adapter.state_dict().items():
        host = tensor.detach().cpu().contiguous()
        out[name] = hashlib.blake2b(host.reshape(-1).view(torch.uint8).numpy().tobytes(), digest_size=16).hexdigest()
    return out


def _backbone(experiment) -> list:
    """The model's parameters outside the adapter (the adapter sits inside the model's module tree)."""
    own = {id(p) for p in experiment.adapter.parameters()}
    return [p for p in experiment.model.parameters() if id(p) not in own]


def trainable_report(experiment, optimizer=None, trainable=TRAINABLE) -> dict:
    """Check (i): the tensors with requires_grad are exactly `trainable` and the optimizer holds exactly them; the backbone and
    the value head hold none. Raises on a mismatch; returns names, counts and the frozen/trainable split."""
    adapter = experiment.adapter
    named = dict(adapter.named_parameters())
    active = sorted(name for name, p in named.items() if p.requires_grad)
    if active != sorted(trainable):
        raise RuntimeError(f"trainable set {active} differs from {sorted(trainable)}")
    if any(p.requires_grad for p in _backbone(experiment)):
        raise RuntimeError("the backbone holds a parameter with requires_grad")
    if getattr(experiment, "value_head", None) is not None:
        raise RuntimeError("the slot study has no value head in the experiment")
    if optimizer is not None:
        held = [p for group in optimizer.param_groups for p in group["params"]]
        if len(optimizer.param_groups) != 1 or {id(p) for p in held} != {id(named[n]) for n in trainable} or len(held) != len(trainable):
            raise RuntimeError("the optimizer does not hold exactly the trainable parameters")
    count = lambda names: sum(named[n].numel() for n in names)
    frozen = sorted(set(named) - set(trainable))
    return dict(trainable=list(trainable), frozen=frozen, buffers=sorted(n for n, _ in adapter.named_buffers()),
                trainable_parameters=count(trainable), frozen_parameters=count(frozen))


def digest_report(adapter, reference: dict[str, str], trainable=TRAINABLE, *, trained: bool) -> dict:
    """Check (ii): every frozen tensor and buffer equals its reference digest; after training the trainable ones differ."""
    found = tensor_digests(adapter)
    frozen = [n for n in found if n not in trainable]
    unchanged = all(found[n] == reference[n] for n in frozen)
    changed = all(found[n] != reference[n] for n in trainable)
    if not unchanged or (trained and not changed):
        raise RuntimeError("frozen tensors changed or trainable tensors did not move: "
                           f"{[n for n in frozen if found[n] != reference[n]]} / {[n for n in trainable if found[n] == reference[n]]}")
    return dict(frozen_unchanged=unchanged, trainable_changed=changed, digests=found)


@torch.no_grad()
def support_digest(experiment, arrays) -> str:
    """Check (iii): blake2b-128 of the written keys, values and matrix state of a fixed support batch (same precision)."""
    digest = hashlib.blake2b(digest_size=16)
    for begin in range(0, len(arrays.episode_seed), CHUNK):
        batch = experiment.prepare(_slice(arrays, slice(begin, min(begin + CHUNK, len(arrays.episode_seed)))))
        for tensor in (batch.keys, batch.values, batch.snapshot.states[experiment.layer]):
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_report(experiment, trainable=TRAINABLE) -> dict:
    """Check (iv), right after one backward pass: the trainable tensors have finite non-zero gradients, the frozen adapter
    tensors and the backbone have none."""
    named = dict(experiment.adapter.named_parameters())
    norms = {}
    for name in trainable:
        grad = named[name].grad
        if grad is None or not bool(torch.isfinite(grad).all()) or not bool((grad != 0).any()):
            raise RuntimeError(f"{name}: no finite non-zero gradient (the read path must stay in the graph)")
        norms[name] = float(grad.norm())
    if any(p.grad is not None for n, p in named.items() if n not in trainable):
        raise RuntimeError("a frozen adapter tensor received a gradient")
    if any(p.grad is not None for p in _backbone(experiment)):
        raise RuntimeError("the backbone received a gradient")
    return dict(trainable_grad_norms=norms)


# ----------------------------------------------------------------------------------------------------- export and reload

def save_export(experiment, path, *, arm: str, spec: ReadSpec, step: int, parent: dict, run: dict,
                trainable=TRAINABLE) -> str:
    """A model-weights export (the episode state is never saved): the adapter with its buffers and everything a reload needs."""
    adapter = experiment.adapter
    payload = dict(
        slot_version=EXPORT_VERSION, arm=experiment.read_arm, config=experiment.config, step=step,
        signature=experiment.writer_signature(), adapter=experiment.inference_state(),
        slot=dict(arm=arm, read=spec.as_dict(), parent=parent, run=run, trainable=list(trainable),
                  query_projection="copy" if adapter.A_query is not None else "shared", loop_steps=1,
                  key_dim=adapter.cfg.key_dim, value_dim=adapter.cfg.r_mem, feature_map=f"dpfp-nu{adapter.cfg.dpfp_nu}-unit",
                  eps=adapter.cfg.eps, layout_digest=slot_data.LAYOUT_DIGEST, task=dict(slot_data.TASK),
                  source_commit=source_metadata()["git_commit"], prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128))
    payload["slot"] = json.loads(json.dumps(payload["slot"], allow_nan=False))
    torch.save(payload, path)
    return file_digest(path)


REQUIRED_SLOT_FIELDS = ("arm", "read", "parent", "run", "trainable", "query_projection", "loop_steps", "key_dim", "value_dim",
                        "feature_map", "eps", "layout_digest", "task", "source_commit", "prereg_blake2b_128")


def load_export(path, *, model=None, tokenizer=None):
    """Reload an S1A export: adapter, buffers, backend settings and scope. An export without `input_mean`/`input_std` or
    without the backend settings does not load."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    slot, adapter = payload.get("slot", {}), payload.get("adapter", {})
    if (payload.get("slot_version") != EXPORT_VERSION or "optimizer" in payload or "value_head" in payload
            or any(field not in slot for field in REQUIRED_SLOT_FIELDS) or any(name not in adapter for name in BUFFERS)):
        raise ValueError("not a slot export (version, backend settings or input_mean/input_std missing)")
    experiment = sq2.SQ2Experiment(payload["config"], payload["arm"], model=model, tokenizer=tokenizer)
    try:
        if payload["signature"] != experiment.writer_signature():
            raise ValueError("slot export signature mismatch")
        adopt(experiment)
        if slot["query_projection"] == "copy":
            experiment.adapter.add_query_projection()
        experiment.adapter.load_state_dict(payload["adapter"], strict=True)
        experiment.read_spec = ReadSpec.from_dict(slot["read"])
        experiment.slot = slot
    except BaseException:
        experiment.close()
        raise
    return experiment
