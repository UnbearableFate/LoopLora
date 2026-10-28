"""CVH (F9): continuation-value halting heads on frozen r = 32 readers.

A head sees only features of the current post-update state (the P4-S-L 51
features) and has the P4-S-L architecture (51 -> 64 -> 1, 3393 parameters). It is
fit on the frozen reader's own traces with the P4-S-L optimizer settings, so two
heads that differ only in their label are comparable. The continuation label of
step ``s`` is "stop": no later state ``s' > s`` has a candidate cross-entropy of the
answer lower by at least ``MARGIN`` nats; step 16 always stops. Inference and
cost accounting are the P4-S-L ones (first step with p >= theta, otherwise the
earliest argmax; the fallback executes all 16 updates).

Nothing in this module trains or changes a reader.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.core.learned_halting import FEATURE_DIM, MAX_STEPS, HaltingHead, trace_frozen_reader
from looplora.core.stopping import ReaderBatch
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from . import followup_stats as fs
from .evaluate import _slice, stratify_queries
from .model import SymbolicModel, make_task_spec

MARGIN = 0.25
THETAS = tuple(i / 20 for i in range(1, 20))
HOP1_TOLERANCE = 0.01
FAMILIES = ("oracle_random/r32/M1", "joint_fact/r32/M1", "t1_frozen_fact/r32/M1")
SEEDS = (17, 29, 43)
ARMS = ("A0", "A1", "A2")
COMPARISONS = 6  # three families x the two new arms (A1, A2)
CONFIDENCE = 1 - 0.05 / COMPARISONS
GAIN_MINIMUM = 0.03
HOP1_LOSS = 0.01
MEAN_DEPTH_LIMIT = 8.0
FIXED_DEPTHS = (1, 4, 16)


def reader_batch(prepared) -> ReaderBatch:
    """The context candidate set sorted by entity id, one row per query (mirrors P4-S-L)."""
    b, q, r = prepared.u.shape
    vocabulary = prepared.candidate_mask.shape[1]
    count = int(prepared.candidate_mask.sum(-1).max())
    ids = torch.arange(vocabulary, device=prepared.u.device).expand(b, -1)
    ids = ids.masked_fill(~prepared.candidate_mask, vocabulary).sort(dim=-1).values[:, :count]
    ids = torch.where(ids == vocabulary, -1, ids)
    codes = prepared.codes.gather(1, ids.clamp_min(0)[..., None].expand(-1, -1, prepared.codes.shape[-1]))
    return ReaderBatch(prepared.u.reshape(b * q, r), prepared.memory.repeat_interleave(q, dim=0),
                       codes.repeat_interleave(q, dim=0), ids.repeat_interleave(q, dim=0))


@torch.no_grad()
def step_losses(adapter, batch: ReaderBatch, states: Tensor, answer: Tensor) -> Tensor:
    """Candidate cross-entropy of the answer at the states z^1..z^S: [N, S].

    ``states`` is ``trace.states`` ([N, S + 1, r]); ``answer`` holds one entity id
    per row and must appear among that row's candidates.
    """
    output = adapter.branch(states[:, 1:])
    logits = torch.einsum("ncd,nsd->nsc", batch.codes, output)
    logits = logits.masked_fill((batch.candidate_ids < 0)[:, None, :], -torch.inf)
    is_answer = batch.candidate_ids == answer[:, None]
    if not bool(is_answer.any(dim=-1).all()):
        raise ValueError("answer is missing from a candidate row")
    position = is_answer.long().argmax(dim=-1)
    log_probability = torch.log_softmax(logits, dim=-1)
    index = position[:, None, None].expand(-1, logits.shape[1], 1)
    return -log_probability.gather(2, index).squeeze(-1)


def stop_labels(losses: Tensor, margin: float = MARGIN) -> Tensor:
    """True where no later step has a loss at least ``margin`` below the current one: [N, S]."""
    if losses.ndim != 2 or not margin >= 0:
        raise ValueError("losses must be [N, S] and the margin nonnegative")
    suffix_min = torch.flip(torch.cummin(torch.flip(losses, dims=(1,)), dim=1).values, dims=(1,))
    best_later = torch.cat((suffix_min[:, 1:], torch.full_like(losses[:, :1], torch.inf)), dim=1)
    return ~(best_later <= losses - margin)


def policy_steps(probabilities: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """First inclusive hit, else the earliest best step; the fallback executes all 16 updates."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3 or probabilities.shape[-1] != MAX_STEPS:
        raise ValueError("probabilities must be [E, Q, 16]")
    if not (np.all(np.isfinite(probabilities)) and np.all((probabilities >= 0) & (probabilities <= 1))):
        raise ValueError("halting probabilities must be finite and lie in [0, 1]")
    if not (np.isfinite(theta) and 0 <= theta <= 1):
        raise ValueError("theta must lie in [0, 1]")
    hits = probabilities >= theta
    found = hits.any(axis=-1)
    selected = np.where(found, hits.argmax(axis=-1) + 1, probabilities.argmax(axis=-1) + 1).astype(np.int64)
    executed = np.where(found, selected, MAX_STEPS).astype(np.int64)
    return selected, executed


def correct_at(predictions: np.ndarray, answer: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Correctness of the candidate decision after ``steps`` updates (0..16): [E, Q] bool."""
    return np.take_along_axis(predictions, steps[..., None], axis=-1)[..., 0] == answer


def threshold_table(traces: Mapping[int, Mapping[str, np.ndarray]], key: str,
                    thetas: Sequence[float] = THETAS) -> list[dict[str, Any]]:
    """Per-theta validation statistics of one head, pooled equally over training seeds.

    ``traces[seed]`` holds predictions [E, Q, 17], answer, hop and ``key`` (the
    head's probabilities [E, Q, 16]). ``hop1_delta`` is learned minus fixed L1 on
    hop 1; ``accuracy`` is the mean over all queries (every hop has one query per
    episode, so hops are equally weighted).
    """
    seeds = sorted(traces)
    if not seeds:
        raise ValueError("at least one training seed is required")
    l1 = {s: correct_at(traces[s]["predictions"], traces[s]["answer"],
                        np.ones(traces[s]["answer"].shape, dtype=np.int64)) for s in seeds}
    rows = []
    for theta in thetas:
        per_seed = []
        for s in seeds:
            trace = traces[s]
            selected, executed = policy_steps(trace[key], theta)
            correct = correct_at(trace["predictions"], trace["answer"], selected)
            first = trace["hop"] == 1
            per_seed.append((correct.mean(), float(correct[first].mean() - l1[s][first].mean()),
                             selected.mean(), executed.mean()))
        values = np.asarray(per_seed, dtype=np.float64)
        rows.append(dict(theta=float(theta), validation_accuracy=float(values[:, 0].mean()),
                         hop1_delta=float(values[:, 1].mean()), validation_mean_steps=float(values[:, 2].mean()),
                         validation_mean_executed_steps=float(values[:, 3].mean())))
    return rows


def select_theta(table: Sequence[Mapping[str, Any]], *, tolerance: float | None = HOP1_TOLERANCE) -> dict[str, Any] | None:
    """Validation choice: highest accuracy, then fewer chosen steps, then smaller theta.

    With ``tolerance`` set, only thresholds whose hop-1 accuracy is at least
    ``tolerance`` below fixed L1 are feasible; ``None`` is returned when no
    threshold is feasible (the registered failure, decided on validation alone).
    ``tolerance=None`` is the unconstrained P4-S-L rule.
    """
    candidates = [row for row in table if tolerance is None or row["hop1_delta"] >= -tolerance]
    if not candidates:
        return None
    return dict(min(candidates, key=lambda row: (-row["validation_accuracy"], row["validation_mean_steps"], row["theta"])))


def fit_head(features: Tensor, targets: Tensor, *, seed: int, family: str, label: str, steps: int,
             lr: float = 0.001, batch_episodes: int = 256) -> tuple[HaltingHead, dict[str, Any]]:
    """P4-S-L head fit with a configurable seed label (``label='p4s-l'`` reproduces it exactly)."""
    init_seed = derive_seed(seed, label, family, "head_init")
    sampler_seed = derive_seed(seed, label, family, "episode_sampler")
    head = HaltingHead(seed=init_seed).to(features.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    generator = torch.Generator(device=features.device).manual_seed(sampler_seed)
    logs = []
    head.train()
    for step in range(steps):
        indices = torch.randint(features.shape[0], (batch_episodes,), generator=generator, device=features.device)
        x, y = features[indices].reshape(-1, FEATURE_DIM), targets[indices].reshape(-1).float()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head.logits(x), y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite head loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            logs.append(dict(step=step + 1, loss=float(loss.detach())))
    head.eval()
    return head, dict(steps=steps, init_seed=init_seed, sampler_seed=sampler_seed, logs=logs,
                      parameters=sum(p.numel() for p in head.parameters()), label=label)


def _sync(device: torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def collect_traces(model: SymbolicModel, hops: Sequence[int], split: str, seed: int, episodes: int, *,
                   heads: Mapping[str, nn.Module] | None = None, training_margin: float | None = None,
                   batch_episodes: int = 64) -> dict[str, Any]:
    """Trace a frozen reader on ``episodes`` stratified episodes (one query per hop).

    With ``training_margin`` set, returns ``features`` [E, Q, 16, 51] and the
    continuation stop labels ``targets`` [E, Q, 16] (and the correctness targets
    ``correct`` used by P4-S-L). Otherwise returns predictions [E, Q, 17], the
    answer/hop/episode ids, per-step candidate losses [E, Q, 16] and, for every
    head in ``heads``, its probabilities [E, Q, 16]; ``first`` is one query per
    episode of the first batch (for latency audits).
    """
    config = copy.deepcopy(model.config)
    config["eval"]["hops"] = list(hops)
    spec = make_task_spec(config, evaluation=True)
    arrays = stratify_queries(generate(spec, split, 0, episodes, seed), spec, list(hops), seed)
    device = model.device
    fields: dict[str, list] = {name: [] for name in ("predictions", "losses", "features", "targets", "correct")}
    probabilities = {name: [] for name in (heads or {})}
    first = None
    for begin in range(0, episodes, batch_episodes):
        part = _slice(arrays, slice(begin, begin + batch_episodes))
        prepared = model.prepare(part)
        batch = reader_batch(prepared)
        trace = trace_frozen_reader(model.adapter, batch)
        b, q = prepared.u.shape[:2]
        answer = torch.as_tensor(part.answer, device=device).reshape(-1)
        losses = step_losses(model.adapter, batch, trace.states, answer)
        if training_margin is not None:
            fields["features"].append(trace.features.reshape(b, q, MAX_STEPS, FEATURE_DIM))
            fields["targets"].append(stop_labels(losses, training_margin).reshape(b, q, MAX_STEPS))
            fields["correct"].append((trace.predictions[:, 1:] == answer[:, None]).reshape(b, q, MAX_STEPS))
            continue
        fields["predictions"].append(trace.predictions.reshape(b, q, MAX_STEPS + 1).cpu().numpy())
        fields["losses"].append(losses.reshape(b, q, MAX_STEPS).cpu().numpy())
        for name, head in (heads or {}).items():
            probabilities[name].append(head(trace.features).reshape(b, q, MAX_STEPS).cpu().numpy())
        if first is None:
            index = torch.arange(b, device=device) * q + torch.arange(b, device=device) % q
            first = batch.take(index)
            for depth in (1, 4, MAX_STEPS):
                logits = model.read(prepared, depth)["logits"]
                expected = logits.masked_fill(~prepared.candidate_mask[:, None], -torch.inf).argmax(-1)
                if not torch.equal(trace.predictions[:, depth].reshape(b, q), expected):
                    raise AssertionError(f"candidate decoding changed at depth {depth}")
    _sync(device)
    if training_margin is not None:
        return dict(features=torch.cat(fields["features"]), targets=torch.cat(fields["targets"]),
                    correct=torch.cat(fields["correct"]))
    return dict(predictions=np.concatenate(fields["predictions"]), losses=np.concatenate(fields["losses"]),
                probabilities={name: np.concatenate(parts) for name, parts in probabilities.items()},
                answer=arrays.answer, hop=arrays.q_hop_true,
                episode_id=np.broadcast_to(arrays.episode_seed[:, None], arrays.answer.shape).copy(),
                first=first)


def _labels(traces_by_seed: Mapping[int, Mapping[str, np.ndarray]]) -> tuple[list[int], Mapping[str, np.ndarray]]:
    """Training seeds and shared labels; every seed must have traced exactly the same queries."""
    seeds = sorted(traces_by_seed)
    if not seeds:
        raise ValueError("at least one training seed is required")
    first = traces_by_seed[seeds[0]]
    for seed in seeds[1:]:
        for name in ("episode_id", "hop", "answer"):
            if not np.array_equal(traces_by_seed[seed][name], first[name]):
                raise ValueError(f"traces of seed {seed} disagree on {name}")
    return seeds, first


def fixed_correct(traces_by_seed: Mapping[int, Mapping[str, np.ndarray]], depth: int) -> dict[int, np.ndarray]:
    """Per-seed correctness [E, Q] of the fixed-depth read after ``depth`` updates."""
    return {seed: correct_at(trace["predictions"], trace["answer"], np.full(trace["answer"].shape, depth, np.int64))
            for seed, trace in traces_by_seed.items()}


def arm_correct(traces_by_seed: Mapping[int, Mapping[str, np.ndarray]],
                selected_by_seed: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-seed correctness [E, Q] of an arm that reads after its selected step."""
    return {seed: correct_at(traces_by_seed[seed]["predictions"], traces_by_seed[seed]["answer"],
                             np.asarray(selected_by_seed[seed], dtype=np.int64)) for seed in traces_by_seed}


def _contrast(correct_a: Mapping[int, np.ndarray], correct_b: Mapping[int, np.ndarray], hops: Sequence[int],
              labels: Mapping[str, np.ndarray], seeds: Sequence[int], *, resamples: int, confidence: float,
              label: Sequence[Any]) -> dict[str, Any]:
    keep = np.isin(labels["hop"], hops)
    values = {seed: correct_a[seed][keep].astype(np.float64) - correct_b[seed][keep].astype(np.float64)
              for seed in seeds}
    return fs.pooled_mean_contrast(values, np.asarray(labels["episode_id"])[keep], resamples=resamples,
                                   confidence=confidence, label=("cvh", *label))


def arm_endpoints(traces_by_seed: Mapping[int, Mapping[str, np.ndarray]],
                  steps_by_seed: Mapping[int, tuple[np.ndarray, np.ndarray]], *, resamples: int = 2000,
                  confidence: float = CONFIDENCE, label: Sequence[Any] = ()) -> dict[str, Any]:
    """Registered CVH endpoints of one arm of one family on confirmation traces.

    ``steps_by_seed[seed]`` is the arm's (selected, executed) steps [E, Q] from ``policy_steps``.
    ``reach_gain`` and ``hop1_delta`` are paired differences to the fixed single read
    (L1) on hops 2-3 and hop 1; ``mean_executed`` is the mean number of executed updates
    over all queries and seeds. The arm passes if hops 2-3 gain at least ``GAIN_MINIMUM`` with
    the interval above zero, hop 1 loses at most ``HOP1_LOSS`` and ``mean_executed`` is at
    most ``MEAN_DEPTH_LIMIT``.
    """
    seeds, labels = _labels(traces_by_seed)
    arm = arm_correct(traces_by_seed, {seed: steps_by_seed[seed][0] for seed in seeds})
    fixed = {depth: fixed_correct(traces_by_seed, depth) for depth in FIXED_DEPTHS}
    options = dict(resamples=resamples, confidence=confidence)
    result: dict[str, Any] = dict(
        reach_gain=_contrast(arm, fixed[1], (2, 3), labels, seeds, label=(*label, "reach_gain"), **options),
        hop1_delta=_contrast(arm, fixed[1], (1,), labels, seeds, label=(*label, "hop1_delta"), **options),
        reach_vs_l4=_contrast(arm, fixed[4], (2, 3), labels, seeds, label=(*label, "reach_vs_l4"), **options),
        reach_vs_l16=_contrast(arm, fixed[16], (2, 3), labels, seeds, label=(*label, "reach_vs_l16"), **options),
        mean_selected=float(np.mean([np.asarray(steps_by_seed[s][0]).mean() for s in seeds])),
        mean_executed=float(np.mean([np.asarray(steps_by_seed[s][1]).mean() for s in seeds])))
    result["passes_gain"] = fs.passes_gain(result["reach_gain"], GAIN_MINIMUM)
    result["passes_hop1"] = bool(result["hop1_delta"]["difference"] >= -HOP1_LOSS)
    result["passes_depth"] = bool(result["mean_executed"] <= MEAN_DEPTH_LIMIT)
    result["passes"] = bool(result["passes_gain"] and result["passes_hop1"] and result["passes_depth"])
    return result


def accuracy_rows(traces_by_seed: Mapping[int, Mapping[str, np.ndarray]],
                  methods: Mapping[str, Mapping[int, np.ndarray]], *, resamples: int = 2000,
                  confidence: float = 0.95, label: Sequence[Any] = ()) -> list[dict[str, Any]]:
    """Descriptive pooled accuracy of each method (per-seed correctness [E, Q]) by hop group and overall."""
    seeds, labels = _labels(traces_by_seed)
    rows = []
    groups = {**fs.HOP_GROUPS, "all": tuple(sorted({int(h) for h in np.unique(labels["hop"])}))}
    for method, correct in methods.items():
        for group, hops in groups.items():
            keep = np.isin(labels["hop"], hops)
            if not keep.any():
                continue
            values = {seed: correct[seed][keep].astype(np.float64) for seed in seeds}
            result = fs.pooled_mean_contrast(values, np.asarray(labels["episode_id"])[keep], resamples=resamples,
                                             confidence=confidence, label=("cvh", "accuracy", *label, method, group))
            rows.append(dict(method=method, hop_group=group, hops=",".join(str(h) for h in hops),
                             accuracy=result["difference"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                             confidence=confidence,
                             **{f"accuracy_seed_{row['seed']}": row["difference"] for row in result["per_seed"]}))
    return rows
