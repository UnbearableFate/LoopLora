"""TSM (F10): terminal-semantics training of the iterative (iterq) reader.

Arms (all M1, r = 32, hops 1-3, one shared optimizer recipe and data stream):

* T0: the existing ``looplora.sym.train.train`` at depth 4 (not reimplemented here);
* T1: depth 8, answer cross-entropy at every step ``s >= hop`` ("stay"), averaged
  per query over its steps;
* T2: T1 plus the content-gated sticky update
  ``z^{s+1} = z^s + g_s (f(z^s) - z^s)``, ``g_s = sigmoid(a * |U M Q z^s| / |u| + b)``
  with two scalar parameters ``(a, b)``, initialised to ``(0, 4)`` (a nearly plain
  update at the start of training).

``fit_tsm(stay=False, gate=None)`` is op-for-op the existing ``_fit`` loop and is
tested to be bitwise equal to it, so T1/T2 differ from T0 only in depth, loss and
gate. Nothing here edits an existing code path.
"""
from __future__ import annotations

from collections import defaultdict
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.data.relations import EpisodeArrays, generate
from looplora.infra.seeds import derive_seed
from . import followup_stats as fs
from .evaluate import _slice
from .model import PreparedBatch, SymbolicModel, make_task_spec
from .train import load_checkpoint, synchronize

ARMS = ("T0", "T1", "T2")
DEPTHS = (1, 4, 8, 16)
GATE_INIT = (0.0, 4.0)
STAY_TOLERANCE = 0.01
FAR_TOLERANCE = 0.01
GAIN_MINIMUM = 0.03


class StickyGate(nn.Module):
    """Scalar content gate g = sigmoid(a * ||readout|| / ||u|| + b)."""

    def __init__(self, a: float = GATE_INIT[0], b: float = GATE_INIT[1], eps: float = 1e-12) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))
        self.eps = eps

    def forward(self, readout: Tensor, u: Tensor) -> Tensor:
        ratio = torch.linalg.vector_norm(readout, dim=-1) / torch.linalg.vector_norm(u, dim=-1).clamp_min(self.eps)
        return torch.sigmoid(self.a * ratio + self.b)


def _check_reader(model: SymbolicModel) -> None:
    if model.adapter.cfg.variant != "iterq":
        raise ValueError("TSM requires the iterative-query reader (M1, M1-lin or B4)")


def read_states(model: SymbolicModel, batch: PreparedBatch, steps: int, gate: StickyGate | None = None,
                gates: list[Tensor] | None = None) -> list[Tensor]:
    """The states z^0..z^steps ([B, Q, r] each); ``gate=None`` is exactly ``adapter.read``.

    When ``gates`` is a list, the gate value of every step ([B, Q]) is appended.
    """
    _check_reader(model)
    adapter = model.adapter
    memory = batch.memory[:, None].expand(-1, batch.u.shape[1], -1, -1)
    u = batch.u
    z = u
    states = [z]
    for _ in range(steps):
        readout = adapter._query(z, memory, adapter.Q, adapter.U)
        update = u + adapter.cfg.rho * adapter._phi(F.linear(z, adapter.R) + readout)
        if gate is None:
            z = update
        else:
            g = gate(readout, u)
            if gates is not None:
                gates.append(g.detach())
            z = z + g[..., None] * (update - z)
        states.append(z)
    return states


def decode(model: SymbolicModel, batch: PreparedBatch, z: Tensor) -> Tensor:
    """Full-vocabulary logits of a state, as ``SymbolicModel.read``."""
    return torch.einsum("bqd,bvd->bqv", model.adapter.branch(z), batch.codes)


def stay_loss(logits: Tensor, candidate_mask: Tensor, answer: Tensor, hop: Tensor) -> Tensor:
    """Per-query mean over steps ``s >= hop`` of the candidate cross-entropy; mean over queries.

    ``logits`` is [B, Q, S, V + 1] for the steps 1..S.
    """
    steps = logits.shape[2]
    restricted = logits.masked_fill(~candidate_mask[:, None, None, :], -torch.inf)
    log_probability = torch.log_softmax(restricted, dim=-1)
    nll = -log_probability.gather(-1, answer[:, :, None, None].expand(-1, -1, steps, 1)).squeeze(-1)
    active = torch.arange(1, steps + 1, device=logits.device)[None, None, :] >= hop[:, :, None]
    weight = active.to(nll.dtype) / active.sum(dim=-1, keepdim=True).clamp_min(1)
    return (nll * weight).sum(dim=-1).mean()


def fit_tsm(model: SymbolicModel, config: dict[str, Any], out: Path, *, stay: bool,
            gate: StickyGate | None = None, stage: str = "train") -> dict[str, Any]:
    """The existing ``_fit`` loop with an optional per-step stay loss and sticky gate."""
    _check_reader(model)
    options = config["train"]
    spec = make_task_spec(config)
    depths = options["depths"]
    if stay and len(depths) != 1:
        raise ValueError("stay training uses one fixed depth")
    depth_rng = np.random.default_rng(derive_seed(config["seed"], stage, "depths"))
    data_seed = derive_seed(config["seed"], "train_data" if stage == "train" else "pretrain_data")
    params = [p for p in model.parameters() if p.requires_grad]
    if gate is not None:
        params = params + list(gate.parameters())
    optimizer = torch.optim.AdamW(params, lr=options["lr"], weight_decay=options["weight_decay"])
    model.train()
    synchronize(model.device)
    started = time.perf_counter()
    elapsed_steps, chosen_depths, final_loss = [], [], None
    out = Path(out)
    with (out / "train_log.jsonl").open("a") as log:
        for step in range(options["steps"]):
            synchronize(model.device)
            step_start = time.perf_counter()
            arrays = generate(spec, "train", step * options["batch_size"], options["batch_size"], data_seed)
            L = int(depth_rng.choice(depths))
            warmup = min(1.0, (step + 1) / max(1, options["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = options["lr"] * warmup
            optimizer.zero_grad(set_to_none=True)
            batch = model.prepare(arrays)
            if not batch.candidate_mask.gather(1, batch.answer).all():
                raise ValueError("training answer absent from episode candidate set")
            states = read_states(model, batch, L, gate)
            if stay:
                hop = torch.as_tensor(arrays.q_hop_true, dtype=torch.long, device=model.device)
                per_step = torch.stack([decode(model, batch, z) for z in states[1:]], dim=2)
                loss = stay_loss(per_step, batch.candidate_mask, batch.answer, hop)
                logits = per_step[:, :, -1]
            else:
                logits = decode(model, batch, states[L])
            restricted = logits.masked_fill(~batch.candidate_mask[:, None, :], -torch.inf)
            if not stay:
                loss = F.cross_entropy(restricted.flatten(0, 1), batch.answer.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {stage} loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            synchronize(model.device)
            step_seconds = time.perf_counter() - step_start
            elapsed_steps.append(step_seconds)
            chosen_depths.append(L)
            if (step + 1) % options["log_every"] == 0 or step == 0 or step + 1 == options["steps"]:
                final_loss = float(loss.detach())
                row = {"stage": stage, "step": step + 1, "L": L,
                       "mean_L": float(np.mean(chosen_depths)), "loss": final_loss,
                       "accuracy": float((restricted.argmax(-1) == batch.answer).float().mean()),
                       "lr": options["lr"] * warmup, "grad_norm": float(grad_norm),
                       "step_seconds": step_seconds, "data_seed": data_seed}
                if gate is not None:
                    row.update(gate_a=float(gate.a.detach()), gate_b=float(gate.b.detach()))
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
    synchronize(model.device)
    return {"train_seconds": time.perf_counter() - started,
            "train_step_seconds": float(np.mean(elapsed_steps)) if elapsed_steps else 0.0,
            "train_step_p95_seconds": float(np.quantile(elapsed_steps, .95)) if elapsed_steps else 0.0,
            "mean_L": float(np.mean(chosen_depths)) if chosen_depths else None,
            "steps": options["steps"], "final_loss": final_loss, "data_seed": data_seed,
            "train_tokens": options["steps"] * options["batch_size"] * (spec.E + spec.Q),
            "train_queries": options["steps"] * options["batch_size"] * spec.Q}


def save_checkpoint(path: Path, config: dict[str, Any], model: SymbolicModel, summary: dict[str, Any],
                    arm: str, gate: StickyGate | None) -> None:
    """``load_checkpoint`` compatible record with the arm label and the gate parameters."""
    torch.save({"config": config, "model": model.state_dict(), "summary": summary, "arm": arm,
                "gate": None if gate is None else {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}},
               path)


def train_arm(config: dict[str, Any], out: Path, arm: str) -> tuple[SymbolicModel, StickyGate | None, dict[str, Any]]:
    """Train T1 or T2 (T0 is the unchanged ``looplora.sym.train.train``)."""
    if arm not in ("T1", "T2"):
        raise ValueError("train_arm trains T1 or T2; T0 uses looplora.sym.train.train")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint.pt").exists() or (out / "train_log.jsonl").exists():
        raise FileExistsError(f"training output already exists: {out}")
    model = SymbolicModel(config)
    gate = StickyGate().to(model.device) if arm == "T2" else None
    summary = fit_tsm(model, config, out, stay=True, gate=gate)
    summary.update(arm=arm, stay_loss=True, sticky_gate=arm == "T2", gate_init=list(GATE_INIT) if gate is not None else None,
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
                   + (sum(p.numel() for p in gate.parameters()) if gate is not None else 0),
                   registered_parameters=sum(p.numel() for p in model.parameters())
                   + (sum(p.numel() for p in gate.parameters()) if gate is not None else 0))
    save_checkpoint(out / "checkpoint.pt", config, model, summary, arm, gate)
    return model, gate, summary


def load_tsm(path: str | Path, device: str | None = None) -> tuple[SymbolicModel, StickyGate | None, str]:
    """Reader, gate (or None) and arm label of any T0/T1/T2 checkpoint."""
    record = torch.load(path, map_location="cpu", weights_only=True)
    model = load_checkpoint(path, device=device)
    model.requires_grad_(False)
    gate = None
    if record.get("gate") is not None:
        gate = StickyGate().to(model.device)
        gate.load_state_dict(record["gate"], strict=True)
        gate.requires_grad_(False)
        gate.eval()
    return model, gate, str(record.get("arm", "T0"))


@torch.no_grad()
def evaluate_arm(model: SymbolicModel, gate: StickyGate | None, arrays: EpisodeArrays, depths: Sequence[int], *,
                 batch_size: int = 64, split: str = "test", condition: str = "original") -> dict[str, np.ndarray]:
    """Flat evaluation samples (``evaluate`` fields) of one reader on prepared episodes."""
    depths = [int(d) for d in depths]
    if not depths or len(set(depths)) != len(depths) or min(depths) < 0:
        raise ValueError("depths must be distinct nonnegative integers")
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    previous = model.training
    model.eval()
    try:
        for begin in range(0, len(arrays.episode_seed), batch_size):
            part = _slice(arrays, slice(begin, begin + batch_size))
            prepared = model.prepare(part)
            states = read_states(model, prepared, max(depths), gate)
            candidate_size = prepared.candidate_mask.sum(dim=-1).cpu().numpy()
            shape = part.answer.shape
            for depth in depths:
                logits = decode(model, prepared, states[depth])
                restricted = logits.masked_fill(~prepared.candidate_mask[:, None, :], -torch.inf)
                prediction = restricted.argmax(-1).cpu().numpy()
                full = logits.argmax(-1).cpu().numpy()
                values = dict(split=np.full(shape, split), condition=np.full(shape, condition),
                              episode_id=np.broadcast_to(part.episode_seed[:, None], shape),
                              query_id=np.broadcast_to(np.arange(shape[1]), shape), hop=part.q_hop_true,
                              L=np.full(shape, depth, dtype=np.int64), prediction=prediction, answer=part.answer,
                              correct=prediction == part.answer, full_correct=full == part.answer,
                              candidate_size=np.broadcast_to(candidate_size[:, None], shape))
                for name, value in values.items():
                    chunks[name].append(np.asarray(value).ravel().copy())
    finally:
        model.train(previous)
    return {name: np.concatenate(parts) for name, parts in chunks.items()}


@torch.no_grad()
def gate_profile(model: SymbolicModel, gate: StickyGate, arrays: EpisodeArrays, steps: int = 16, *,
                 batch_size: int = 64) -> dict[str, Any]:
    """Mean gate value per (hop, step): rows are the queries' true hops, columns steps 1..steps."""
    hops = np.unique(arrays.q_hop_true)
    total = np.zeros((len(hops), steps))
    count = np.zeros(len(hops))
    for begin in range(0, len(arrays.episode_seed), batch_size):
        part = _slice(arrays, slice(begin, begin + batch_size))
        prepared = model.prepare(part)
        recorded: list[Tensor] = []
        read_states(model, prepared, steps, gate, gates=recorded)
        values = torch.stack(recorded, dim=-1).cpu().numpy()
        for i, hop in enumerate(hops):
            selected = part.q_hop_true == hop
            total[i] += values[selected].sum(axis=0)
            count[i] += selected.sum()
    return {"hops": hops.tolist(), "steps": list(range(1, steps + 1)),
            "mean_gate": (total / count[:, None]).tolist()}


def _contrast(left: Mapping[int, Mapping[str, np.ndarray]], left_depth: int,
              right: Mapping[int, Mapping[str, np.ndarray]], right_depth: int, hops: Sequence[int], *,
              confidence: float, label: Sequence[Any], resamples: int = 2000, split: str = "test") -> dict[str, Any]:
    """Mean over seeds of (left at depth) minus (right at depth) accuracy on the given hops."""
    seeds = sorted(left)
    if sorted(right) != seeds:
        raise ValueError("both sides need the same training seeds")
    values: dict[int, np.ndarray] = {}
    episode_id = None
    for seed in seeds:
        a = fs.aligned_terms({seed: left[seed]}, [(split, "original", left_depth)])
        b = fs.aligned_terms({seed: right[seed]}, [(split, "original", right_depth)])
        for name in ("episode_id", "query_id", "hop", "answer"):
            if not np.array_equal(a[name], b[name]):
                raise ValueError(f"the two sides disagree on {name}")
        keep = np.isin(a["hop"], list(hops))
        values[seed] = (a["values"][seed][0] - b["values"][seed][0])[keep]
        if episode_id is None:
            episode_id = a["episode_id"][keep]
        elif not np.array_equal(episode_id, a["episode_id"][keep]):
            raise ValueError("seeds disagree on the evaluation episodes")
    return fs.pooled_mean_contrast(values, episode_id, resamples=resamples, confidence=confidence, label=label)


def arm_endpoints(samples: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]], arm: str, *, writer: str,
                  confidence: float, resamples: int = 2000, split: str = "test") -> dict[str, Any]:
    """Registered F10 contrasts of one arm (T1 or T2) for one writer.

    ``samples[name][seed]`` are ``evaluate_arm`` outputs for T0, the arm and B4 (B4 at depth 1).

    * stay: arm@16 - arm@4 on hops 1-3, passes if |difference| <= 1 pp;
    * far:  arm@8 - T0@8 on hops 4-8, passes if the difference is >= -1 pp;
    * gain: arm@4 - B4@1 on hops 2-3, passes if >= +3 pp with CI lower bound > 0.
    """
    if arm not in ("T1", "T2"):
        raise ValueError("endpoints are defined for T1 and T2")
    options = dict(confidence=confidence, resamples=resamples, split=split)
    own, base = samples[arm], samples["T0"]
    far_hops = (4, 5, 6, 7, 8)
    stay = _contrast(own, 16, own, 4, (1, 2, 3), label=(writer, arm, "stay"), **options)
    far = _contrast(own, 8, base, 8, far_hops, label=(writer, arm, "far"), **options)
    gain = _contrast(own, 4, samples["B4"], 1, (2, 3), label=(writer, arm, "gain"), **options)
    checks = dict(stay=abs(stay["difference"]) <= STAY_TOLERANCE,
                  far=far["difference"] >= -FAR_TOLERANCE,
                  gain=fs.passes_gain(gain, GAIN_MINIMUM))
    descriptive = dict(
        hop1_l8_vs_l1=_contrast(own, 8, own, 1, (1,), label=(writer, arm, "hop1_l8"), **options),
        hop1_l16_vs_l1=_contrast(own, 16, own, 1, (1,), label=(writer, arm, "hop1_l16"), **options),
        reach_l4_vs_t0_l4=_contrast(own, 4, base, 4, (2, 3), label=(writer, arm, "reach_t0"), **options),
        far_l16_vs_t0_l16=_contrast(own, 16, base, 16, far_hops, label=(writer, arm, "far16_t0"), **options))
    return dict(writer=writer, arm=arm, stay=stay, far=far, gain=gain, descriptive=descriptive, checks=checks,
                passes=all(checks.values()))
