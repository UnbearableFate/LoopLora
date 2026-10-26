#!/usr/bin/env python3
"""P4-S-L head-only training, validation lock, and separate confirmation release."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_p4s_stopping import digest_checkpoint, require, require_compute
from benchmark_p4s_stopping import _measure
from looplora.core.learned_halting import (
    HaltingHead, learned_adaptive_read, learned_halting_steps, trace_frozen_reader,
)
from looplora.core.stopping import fixed_read
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed, stable_hash
from looplora.sym.evaluate import _slice, stratify_queries
from looplora.sym.model import make_task_spec
from looplora.sym.run import source_metadata
from run_p4s_stopping import load_model, reader_batch, save


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_hash(model) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def verify_seals(runs: list[dict]) -> None:
    for run in runs:
        require(digest_checkpoint(Path(run['checkpoint'])) == run['checkpoint_blake2b'],
                f"Frozen reader changed: {run['name']}")
        if 'head_checkpoint' in run:
            require(digest_checkpoint(Path(run['head_checkpoint'])) == run['head_blake2b'],
                    f"Halting head changed: {run['name']}")


@torch.no_grad()
def collect(model, design: dict, split: str, generator_seed: int, episodes: int,
            *, head=None, theta: float | None = None, training=False):
    """Cache training features on GPU; retain only compact traces for evaluation."""
    config = copy.deepcopy(model.config)
    config['eval']['hops'] = design['hops']
    spec = make_task_spec(config, evaluation=True)
    started = time.perf_counter()
    arrays = stratify_queries(generate(spec, split, 0, episodes, generator_seed),
                              spec, design['hops'], generator_seed)
    fields = {key: [] for key in ('predictions', 'read_norm', 'relative_change')}
    if head is not None:
        fields['probabilities'] = []
    features, targets, first = [], [], None
    prepare_seconds = trace_seconds = 0.
    for begin in range(0, episodes, 64):
        part = _slice(arrays, slice(begin, begin + 64))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        prepared = model.prepare(part)
        batch = reader_batch(prepared)
        torch.cuda.synchronize()
        prepare_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        trace = trace_frozen_reader(model.adapter, batch)
        b, q = prepared.u.shape[:2]
        if training:
            features.append(trace.features.reshape(b, q, 16, 51))
            answer = torch.as_tensor(part.answer, device=model.device)
            targets.append(trace.predictions[:, 1:].reshape(b, q, 16) == answer[..., None])
        else:
            for key in ('predictions', 'read_norm', 'relative_change'):
                fields[key].append(getattr(trace, key).reshape(b, q, 17).cpu().numpy())
            if head is not None:
                fields['probabilities'].append(head(trace.features).reshape(b, q, 16).cpu().numpy())
            if first is None:
                indices = torch.arange(b, device=model.device) * q + torch.arange(b, device=model.device) % q
                first = batch.take(indices)
                for depth in (1, 4, 16):
                    output = model.read(prepared, depth)['logits']
                    expected = output.masked_fill(~prepared.candidate_mask[:, None], -torch.inf).argmax(-1)
                    require(torch.equal(trace.predictions[:, depth].reshape(b, q), expected),
                            f'Candidate decoding changed at depth {depth}')
        torch.cuda.synchronize()
        trace_seconds += time.perf_counter() - tick
    timing = dict(split=split, generator_seed=generator_seed, episodes=episodes,
                  queries=episodes * 8, prepare_seconds=prepare_seconds,
                  trace_seconds=trace_seconds, total_seconds=time.perf_counter() - started)
    if training:
        return (torch.cat(features), torch.cat(targets)), timing
    result = {key: np.concatenate(parts) for key, parts in fields.items()}
    result.update(answer=arrays.answer, hop=arrays.q_hop_true,
                  episode_id=np.broadcast_to(arrays.episode_seed[:, None], arrays.answer.shape))
    if theta is not None:
        probabilities = torch.as_tensor(result['probabilities'])
        selected = learned_halting_steps(probabilities.reshape(-1, 16), theta).reshape(episodes, 8)
        hit = (probabilities >= theta).any(-1)
        executed = torch.where(hit, selected, 16)
        result.update(selected_steps=selected.numpy(), executed_steps=executed.numpy(),
                      read_calls=(executed + 1).numpy())
    return result, timing, first


def fit_head(features, targets, run: dict, design: dict, steps: int):
    init_seed = derive_seed(run['seed'], 'p4s-l', run['family'], 'head_init')
    sampler_seed = derive_seed(run['seed'], 'p4s-l', run['family'], 'episode_sampler')
    head = HaltingHead(seed=init_seed).to(features.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=design['training']['lr'])
    generator = torch.Generator(device=features.device).manual_seed(sampler_seed)
    logs = []
    head.train()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(features.shape[0], (design['training']['batch_episodes'],),
                                generator=generator, device=features.device)
        x, y = features[indices].reshape(-1, 51), targets[indices].reshape(-1).float()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head.logits(x), y)
        require(bool(torch.isfinite(loss)), f"Non-finite head loss: {run['name']}/{step + 1}")
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            logs.append(dict(step=step + 1, loss=float(loss.detach())))
    torch.cuda.synchronize()
    head.eval()
    return head, dict(steps=steps, seconds=time.perf_counter() - started,
                      init_seed=init_seed, sampler_seed=sampler_seed, logs=logs,
                      parameters=sum(p.numel() for p in head.parameters()))


def train_all(root: Path, design: dict, design_commit: str, *, design_path: Path,
              smoke=False) -> dict:
    from analyze_p4s_learned import _git_locked_file, select_theta

    design_path = design_path.resolve()
    require(_git_locked_file(design_path, design_commit) == design_path.read_bytes()
            and json.loads(design_path.read_text()) == design,
            'Design differs from its pre-head Git lock')
    root.mkdir(parents=True, exist_ok=False)
    runs = copy.deepcopy(design['runs'])
    verify_seals(runs)
    started = time.perf_counter()
    manifest = dict(status='running', smoke=smoke, design=design, design_hash=stable_hash(design),
                    design_lock_commit=design_commit, design_path=str(design_path),
                    runs=runs, started_at=now(), **source_metadata(),
                    runtime=dict(host=socket.gethostname(), job_id=os.environ['PBS_JOBID'],
                                 torch=torch.__version__, gpu=torch.cuda.get_device_name(), threads=1))
    save(root / 'train_manifest.json', manifest)
    timings = []
    for run in runs:
        torch.cuda.reset_peak_memory_stats()
        model = load_model(run)
        before = state_hash(model)
        episodes = design['training']['episodes'] // (10 if smoke else 1)
        steps = design['training']['steps'] // (10 if smoke else 1)
        (features, targets), trace_timing = collect(model, design, 'train', 31337, episodes, training=True)
        require(not features.requires_grad and not targets.requires_grad,
                'Frozen-reader cache unexpectedly tracks gradients')
        head, training = fit_head(features, targets, run, design, steps)
        del features, targets
        val, val_timing, _ = collect(model, design, 'val', 4242,
                                     design['validation']['episodes'] // (10 if smoke else 1), head=head)
        require(state_hash(model) == before, f"Reader parameters changed: {run['name']}")
        peak = torch.cuda.max_memory_allocated()
        timings.append(dict(name=run['name'], training_trace=trace_timing,
                            head_training_seconds=training['seconds'], validation=val_timing,
                            peak_memory_bytes=peak))
        if not smoke:
            out = root / 'heads' / f"{run['name']}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(head={k: v.detach().cpu() for k, v in head.state_dict().items()},
                            reader_checkpoint=run['checkpoint'], reader_blake2b=run['checkpoint_blake2b'],
                            design_hash=stable_hash(design), training=training,
                            training_episodes=episodes, feature_dim=51, reader_state_hash=before), out)
            run.update(head_checkpoint=str(out), head_blake2b=digest_checkpoint(out))
            folder = root / 'traces' / run['name']
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'val.npz', **val)
            save(root / 'head_records' / f"{run['name']}.json",
                 dict(**run, training=training, training_trace=trace_timing, validation=val_timing,
                      peak_memory_bytes=peak, reader_state_hash=before, reader_unchanged=True,
                      design_hash=stable_hash(design), finished_at=now()))
        # Smoke emits system measurements only; its heads and predictions are discarded.
        print(json.dumps(dict(stage='smoke' if smoke else 'train', name=run['name'],
                              seconds=trace_timing['total_seconds'] + training['seconds'] + val_timing['total_seconds'],
                              peak_memory_bytes=peak)), flush=True)
        del model, head, val
    verify_seals(runs)
    manifest.update(status='complete', runs=runs, finished_at=now(),
                    total_seconds=time.perf_counter() - started, checkpoints_unchanged=True)
    save(root / 'timing.json', timings)
    save(root / 'train_manifest.json', manifest)
    if not smoke:
        theta = select_theta(root, runs, design)
        theta.update(locked_at=now())
        save(root / 'theta.json', theta)
    return manifest


@torch.no_grad()
def benchmark(model, batch, head, theta, run, design):
    expected = trace_frozen_reader(model.adapter, batch)
    probabilities = head(expected.features)
    chosen = learned_halting_steps(probabilities, theta)
    actual = learned_adaptive_read(model.adapter, batch, head, theta)
    require(torch.equal(actual.steps, chosen), 'Streaming/offline halting steps differ')
    require(torch.equal(actual.predictions, expected.predictions.gather(1, chosen[:, None]).squeeze(1)),
            'Streaming/offline halting predictions differ')
    expected_states = expected.states[torch.arange(len(chosen), device=chosen.device), chosen]
    error = torch.linalg.vector_norm(actual.states - expected_states, dim=-1)
    norm = torch.linalg.vector_norm(expected_states, dim=-1)
    require(bool((error <= 1e-6 + 1e-4 * norm).all()), 'Streaming state error exceeds C3 tolerance')
    hit = (probabilities >= theta).any(-1)
    executed = torch.where(hit, chosen, 16)
    require(torch.equal(actual.executed_updates, executed) and torch.equal(actual.read_calls, executed + 1),
            'Streaming physical execution counters differ')
    settings = design['latency']
    operations = [('learned', lambda: learned_adaptive_read(model.adapter, batch, head, theta)),
                  ('fixed_4', lambda: fixed_read(model.adapter, batch, 4)),
                  ('fixed_16', lambda: fixed_read(model.adapter, batch, 16))]
    rows = []
    for method, operation in operations:
        timing = _measure(operation, batch.u.device, settings['warmup'], settings['repeats'])
        n = 4 if method == 'fixed_4' else 16
        rows.append(dict(**{k: run[k] for k in ('name', 'writer', 'r', 'variant', 'seed', 'family')},
                         method=method, batch_size=len(batch.u), warmup=settings['warmup'],
                         repeats=settings['repeats'], **timing,
                         mean_steps=float(actual.steps.float().mean()) if method == 'learned' else n,
                         mean_executed_steps=float(executed.float().mean()) if method == 'learned' else n,
                         mean_reads=float(actual.read_calls.float().mean()) if method == 'learned' else n,
                         audit_prediction_steps_exact=True,
                         audit_state_rel_l2_max=float((error / norm.clamp_min(1e-12)).max()),
                         audit_state_abs_max=float((actual.states - expected_states).abs().max())))
    return rows


def confirm(root: Path, design: dict, theta_path: Path, theta_commit: str) -> dict:
    from analyze_p4s_learned import _git_locked_file, validate_theta

    require(not (root / 'manifest.json').exists(), 'Confirmation already started')
    trained = json.loads((root / 'train_manifest.json').read_text())
    require(trained['status'] == 'complete' and not trained['smoke'], 'Training is incomplete')
    require(trained['design_hash'] == stable_hash(design), 'Locked design changed after training')
    design_path = Path(trained['design_path'])
    require(_git_locked_file(design_path, trained['design_lock_commit']) == design_path.read_bytes()
            and json.loads(design_path.read_text()) == design, 'Pre-head Git lock changed')
    require(_git_locked_file(theta_path, theta_commit) == theta_path.read_bytes(),
            'Theta file differs from its pre-confirmation Git lock')
    selection = json.loads(theta_path.read_text())
    thresholds = validate_theta(selection, trained['runs'], design)
    require(selection == json.loads((root / 'theta.json').read_text()), 'Committed theta differs from training selection')
    verify_seals(trained['runs'])
    started = time.perf_counter()
    manifest = dict(trained, status='running', theta_path=str(theta_path.resolve()),
                    theta_blake2b=digest_checkpoint(theta_path), theta_lock_commit=theta_commit,
                    confirmation_started_at=now(), confirmation_source=source_metadata())
    save(root / 'manifest.json', manifest)
    timing, latency = [], []
    for run in trained['runs']:
        model = load_model(run)
        before = state_hash(model)
        checkpoint = torch.load(run['head_checkpoint'], map_location='cpu', weights_only=True)
        require(checkpoint['design_hash'] == stable_hash(design)
                and checkpoint['reader_blake2b'] == run['checkpoint_blake2b'], 'Head provenance differs')
        head = HaltingHead().to(model.device)
        head.load_state_dict(checkpoint['head'], strict=True)
        head.eval().requires_grad_(False)
        theta = thresholds[run['family']]
        arrays, measured, batch = collect(model, design, 'test', 999002, 2000, head=head, theta=theta)
        np.savez_compressed(root / 'traces' / run['name'] / 'test.npz', **arrays)
        latency.extend(benchmark(model, batch, head, theta, run, design))
        require(state_hash(model) == before, 'Reader changed during confirmation')
        timing.append(dict(name=run['name'], **measured))
        print(json.dumps(dict(stage='confirmation', name=run['name'], seconds=measured['total_seconds'])), flush=True)
        del model, head, arrays, batch
    verify_seals(trained['runs'])
    save(root / 'latency.json', latency)
    save(root / 'confirmation_timing.json', timing)
    manifest.update(status='complete', confirmation_finished_at=now(),
                    confirmation_seconds=time.perf_counter() - started, checkpoints_unchanged=True)
    save(root / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('smoke', 'train', 'confirm'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True, help='Persistent committed design JSON path')
    parser.add_argument('--design-commit')
    parser.add_argument('--theta', type=Path)
    parser.add_argument('--theta-commit')
    args = parser.parse_args()
    require_compute()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    design = json.loads(args.design.read_text())
    if args.action == 'confirm':
        require(args.theta is not None and args.theta_commit, 'Confirmation requires the committed theta lock')
        result = confirm(args.root.resolve(), design, args.theta, args.theta_commit)
    else:
        require(bool(args.design_commit), 'Head training requires the committed design lock')
        result = train_all(args.root.resolve(), design, args.design_commit,
                           design_path=args.design, smoke=args.action == 'smoke')
    print(json.dumps(dict(status=result['status'], action=args.action,
                          seconds=result.get('confirmation_seconds', result.get('total_seconds')))), flush=True)


if __name__ == '__main__':
    main()
