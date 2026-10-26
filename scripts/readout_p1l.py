#!/usr/bin/env python3
"""J-lens-inspired name-gradient probes and exact reader-state diagnostics."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed
from looplora.llm.batching import tokenize_batch
from looplora.llm.evaluate import _slice
from looplora.llm.inject import MODEL_ID, MODEL_REVISION, load_model, load_tokenizer
from looplora.llm.train import load_checkpoint, make_task_spec
from looplora.sym.evaluate import stratify_queries, tensor_hash

from run_p1l_main import config, digest, save


def reader_states(adapter, u, memory, steps=4):
    """Return genuine prefixes without changing B8's fixed-four-step contract."""
    if adapter.cfg.variant not in {"iterq", "unshared"} or not 1 <= steps <= 4:
        raise ValueError("per-step readout requires M1/B8 and steps1..4")
    states, z = [], u
    for step in range(steps):
        R, U, Q = ((adapter.R[step], adapter.U[step], adapter.Q[step])
                   if adapter.cfg.variant == "unshared" else (adapter.R, adapter.U, adapter.Q))
        z = u + adapter.cfg.rho * adapter._phi(F.linear(z, R) + adapter._query(z, memory, Q, U))
        states.append(z)
    return states


def path_entities(arrays, steps=4):
    """Latest-fact paths; -1 denotes a position after the terminal."""
    result = np.full((*arrays.answer.shape, steps), -1, dtype=np.int64)
    for b in range(len(arrays.facts)):
        edges = {(int(h), int(r)): int(t) for (h, r, t), valid
                 in zip(arrays.facts[b], arrays.fact_valid[b], strict=True) if valid}
        for q, (head, relation) in enumerate(zip(arrays.q_start[b], arrays.q_rel[b], strict=True)):
            node = int(head)
            for step in range(steps):
                if (node, int(relation)) not in edges:
                    break
                node = edges[node, int(relation)]
                result[b, q, step] = node
    return result


def normalize_directions(raw):
    centered = raw - raw.mean(dim=0, keepdim=True)
    if not torch.isfinite(raw).all() or (raw.norm(dim=-1) == 0).any() or (centered.norm(dim=-1) == 0).any():
        raise ValueError("non-finite or zero name-gradient probe direction")
    return F.normalize(centered, dim=-1), F.normalize(raw, dim=-1)


def name_gradients(model, ids, valid, names, *, layer=14, chunk=4):
    """Sum per-question name-logit gradients at the scored attention residual."""
    if hasattr(model, "looplora_adapters"):
        raise ValueError("name-gradient probe requires the frozen backbone without adapters")
    last = torch.where(valid, torch.arange(ids.shape[1], device=ids.device), -1).amax(-1)
    perturb = torch.zeros((len(ids), model.config.hidden_size), device=ids.device,
                          dtype=torch.float32, requires_grad=True)
    mask = F.one_hot(last, ids.shape[1]).to(model.dtype)

    def inject(module, inputs, output):
        return output + mask[..., None] * perturb[:, None, :].to(output.dtype)

    hook = model.model.layers[layer].self_attn.o_proj.register_forward_hook(inject)
    try:
        with torch.enable_grad():
            positions = valid.long().cumsum(-1).sub(1).clamp_min(0)
            hidden = model.model(input_ids=ids, attention_mask=valid, position_ids=positions,
                                 use_cache=False).last_hidden_state
            final = hidden[torch.arange(len(ids), device=ids.device), last]
            logits = F.linear(final, model.lm_head.weight[names]).float()
            result = torch.empty((len(names), model.config.hidden_size), device=ids.device)
            for begin in range(0, len(names), chunk):
                stop = min(len(names), begin + chunk)
                selected = logits[:, begin:stop].sum(0)
                gradient, = torch.autograd.grad(selected, perturb,
                    grad_outputs=torch.eye(stop - begin, device=ids.device),
                    is_grads_batched=True, retain_graph=stop < len(names))
                result[begin:stop] = gradient.sum(dim=1)
            return result.detach()
    finally:
        hook.remove()


def cache_directions(args):
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(), load_tokenizer()
    cfg = config("B8")
    spec = make_task_spec(cfg, evaluation=True)
    # Fixed64 episodes x8 questions =512; timing trials use a prefix only.
    arrays = stratify_queries(generate(spec, "val", 0, 64, 4242), spec, list(range(1, 9)), 4242)
    tokens = tokenize_batch(arrays, tokenizer, device="cuda", include_support=False)
    ids, valid = tokens.question_ids.flatten(0, 1), tokens.question_mask.flatten(0, 1)
    count = 8 if args.trial else 512
    names = torch.tensor(load_name_pool().token_ids[:8 if args.trial else 1024], device="cuda")
    accumulated = torch.zeros((len(names), model.config.hidden_size), device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for begin in range(0, count, args.batch):
        end = min(count, begin + args.batch)
        accumulated += name_gradients(model, ids[begin:end], valid[begin:end], names,
                                      chunk=args.chunk)
        torch.cuda.synchronize()
        print(json.dumps(dict(questions=end, names=len(names), seconds=time.perf_counter()-start)), flush=True)
    raw = accumulated / count
    contrastive, secondary = normalize_directions(raw)
    generator = torch.Generator(device="cuda").manual_seed(
        derive_seed(2718, "p1l", "readout-random-directions"))
    random = F.normalize(torch.randn(raw.shape, generator=generator, device="cuda"), dim=-1)
    tensor = dict(raw_mean=raw.cpu(), contrastive=contrastive.cpu(),
                  raw_logit=secondary.cpu(), random=random.cpu(), names=names.cpu())
    torch.save(tensor, args.out / "directions.pt")
    save(args.out / "record.json", dict(status="complete", trial=args.trial, questions=count,
        model=MODEL_ID, revision=MODEL_REVISION, layer=14, names=len(names),
        question_ids_hash=tensor_hash(ids[:count]), question_mask_hash=tensor_hash(valid[:count]),
        tensor_hashes={k: tensor_hash(v) for k, v in tensor.items()},
        file_hash=digest(args.out / "directions.pt"), seconds=time.perf_counter()-start,
        host=socket.gethostname(), job_id=os.environ["PBS_JOBID"],
        time=datetime.now(timezone.utc).isoformat()))


@contextmanager
def prefix_reader(adapter, step):
    def read(u, memory, requested):
        if requested != 4:
            raise ValueError("readout always starts from a four-step checkpoint")
        return reader_states(adapter, u, memory, step)[-1]
    with patch.object(adapter, "_read", side_effect=read):
        yield


@torch.no_grad()
def score_answer_branches(experiment, prepared):
    """Capture the actual alpha*B*z at each scored question token."""
    frames = []
    original = experiment.adapter.branch

    def capture(z):
        value = original(z)
        frames.append(value)
        return value

    with patch.object(experiment.adapter, "branch", side_effect=capture):
        logits, _ = experiment.score(prepared, 4)
    valid = prepared.question_mask.flatten(0, 1)
    b, t = valid.shape
    last = torch.where(valid, torch.arange(t, device=valid.device), -1).amax(-1)
    positions = valid.any(0).nonzero(as_tuple=True)[0]
    if len(frames) != len(positions):
        raise AssertionError("readout branch calls do not match valid token positions")
    branches = frames[0].new_zeros((b, experiment.adapter.cfg.d_out))
    for position, frame in zip(positions, frames, strict=True):
        rows = valid[:, position].nonzero(as_tuple=True)[0]
        keep = last[rows] == position
        branches[rows[keep]] = frame[keep]
    return logits, branches


@torch.no_grad()
def read_batch(experiment, prepared, directions):
    actual, _ = experiment.score(prepared, 4)
    names = directions["names"].to(actual.device)
    actual_names = actual.flatten(0, 1)[:, names].argmax(-1)
    outputs = []
    for step in range(1, 5):
        with prefix_reader(experiment.adapter, step):
            logits, branch = score_answer_branches(experiment, prepared)
        if step == 4 and not torch.equal(logits, actual):
            raise AssertionError("final logit-lens replay differs from deployed model")
        scores = {"logit_lens": logits.flatten(0, 1)[:, names].float()}
        for kind in ("contrastive", "raw_logit", "random"):
            scores[kind] = branch @ directions[kind].to(branch.device).T
        outputs.append({kind: value.cpu().numpy() for kind, value in scores.items()})
    return outputs, actual_names.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=4)
    args = parser.parse_args()
    from audit_p4s_stopping import require_compute
    require_compute()
    cache_directions(args)


if __name__ == "__main__":
    main()
