"""Paired one-token LLM evaluation on fixed episode and hop sets."""
from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from looplora.data.relations import EpisodeArrays, counterfactual, generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm.config import split_window
from looplora.sym.evaluate import stratify_queries, tensor_hash
from looplora.sym.stats import depth_comparisons, load_samples


def _slice(arrays: EpisodeArrays, index: slice | np.ndarray) -> EpisodeArrays:
    return EpisodeArrays(**{
        field.name: (getattr(arrays, field.name) if field.name == "meta"
                     else getattr(arrays, field.name)[index])
        for field in fields(arrays)
    })


def _sync(experiment: Any) -> None:
    if torch.device(experiment.device).type == "cuda":
        torch.cuda.synchronize(experiment.device)


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _finite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite {label} during LLM evaluation")


def _answer_tokens(answers: np.ndarray, vocabulary: int) -> np.ndarray:
    pool = load_name_pool()
    token_ids = np.asarray((*pool.token_ids[:vocabulary], pool.insufficient_token_id), dtype=np.int64)
    return token_ids[answers]


def _snapshot_hashes(prepared: Any) -> dict[str, str]:
    snapshot = prepared.snapshot
    for layer, memory in snapshot.states.items():
        _finite(memory, f"memory at layer {layer}")
    return {
        "memory": stable_hash({str(layer): tensor_hash(value)
                               for layer, value in sorted(snapshot.states.items())}),
        "write_count": tensor_hash(prepared.write_count),
        "attention_mask": tensor_hash(snapshot.attention_mask),
        "backbone_kv": stable_hash(None if snapshot.past_key_values is None else [
            [tensor_hash(key), tensor_hash(value)] for key, value in snapshot.past_key_values
        ]),
    }


def aggregate_samples(samples: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Reproduce the public hop × depth table solely from flat sample records."""
    samples = load_samples(samples)
    keys = sorted(set(zip(samples["split"].tolist(), samples["condition"].tolist(),
                          samples["hop"].tolist(), samples["L"].tolist())))
    table = []
    for split, condition, hop, depth in keys:
        mask = ((samples["split"] == split) & (samples["condition"] == condition)
                & (samples["hop"] == hop) & (samples["L"] == depth))
        row = dict(split=split, condition=condition, protocol=str(samples["protocol"][mask][0]),
                   hop=int(hop), L=int(depth), n=int(mask.sum()),
                   accuracy=float(samples["correct"][mask].mean()),
                   full_accuracy=float(samples["full_correct"][mask].mean()),
                   candidate_accuracy=float(samples["candidate_correct"][mask].mean()),
                   chance_accuracy=float((1.0 / samples["candidate_size"][mask]).mean()))
        if condition.endswith("/replace_key_fact"):
            row["prediction_changed_rate"] = float((samples["prediction"][mask]
                                                      != samples["original_prediction"][mask]).mean())
            previously_correct = samples["original_prediction"][mask] == samples["original_answer"][mask]
            row["n_original_correct"] = int(previously_correct.sum())
            row["new_answer_accuracy_given_original_correct"] = (
                float(samples["correct"][mask][previously_correct].mean()) if previously_correct.any() else None)
        table.append(row)
    return table


def _settings(config: dict[str, Any]) -> tuple[list[int], list[str], list[str], int, int]:
    settings = config["eval"]
    depths = list(settings["depths"])
    if config["variant"] == "B4":
        depths = [1]
    elif config["variant"] == "B8":
        depths = [4]
    if (not depths or any(isinstance(depth, bool) or not isinstance(depth, int) or depth < 0
                          for depth in depths) or len(set(depths)) != len(depths)):
        raise ValueError("evaluation depths must be distinct nonnegative integers")
    splits = list(settings["splits"])
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"val", "test", "confirm", "test_unseen"}:
        raise ValueError("evaluation splits must be distinct val/test/confirm/test_unseen labels")
    if "confirm" in splits and (config.get("smoke", False) or not settings.get("allow_confirmation", False)):
        raise ValueError("confirmation evaluation requires allow_confirmation and a non-smoke run")
    protocols = list(settings.get("protocols", [config["protocol"]]))
    if (not protocols or len(set(protocols)) != len(protocols)
            or set(protocols) - {"memory_only", "context_retained"}):
        raise ValueError("evaluation protocols must be distinct memory_only/context_retained labels")
    count, batch_size = settings["episodes"], settings["batch_size"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (count, batch_size)) or count > 2000:
        raise ValueError("evaluation episodes must be in [1,2000], and batch_size must be positive")
    if settings.get("counterfactuals", True) and count < 2:
        raise ValueError("swap_memory requires at least two episodes")
    return depths, splits, protocols, count, batch_size


@torch.no_grad()
def evaluate(experiment: Any, config: dict[str, Any], out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write full-vocabulary EM, secondary candidate accuracy and exact Gate 0 hashes.

    Swap changes only adapter memory. In context-retained mode the original
    support KV remains available, so that control cannot establish memory-only
    mechanism claims. Every condition retains its source episode/query labels.
    """
    from looplora.llm.train import make_task_spec

    depths, splits, protocols, count, batch_size = _settings(config)
    settings = config["eval"]
    windows = {split: split_window(settings, split) for split in splits}
    spec = make_task_spec(config, evaluation=True)
    seed = int(settings["seed"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chunks: dict[str, list[np.ndarray]] = {}
    hashes: list[dict[str, Any]] = []
    invariant_hashes: list[dict[str, Any]] = []
    timing: dict[str, Any] = dict(writer_seconds=0.0,
                                  reader_seconds={str(depth): 0.0 for depth in depths}, eval_sets=[])
    previous_mode = experiment.model.training
    experiment.model.eval()
    _sync(experiment)
    started = time.perf_counter()
    try:
        for split in splits:
            generator_split, start = windows[split]
            arrays = stratify_queries(generate(spec, generator_split, start, count, seed),
                                     spec, settings["hops"], seed)
            n, q = arrays.answer.shape
            original_answers = _answer_tokens(arrays.answer, spec.V)
            conditions = [("original", arrays, np.arange(n), None)]
            if settings.get("counterfactuals", True):
                replaced, _ = counterfactual(arrays, "replace_key_fact", derive_seed(seed, "eval-replace", split))
                swapped, _ = counterfactual(arrays, "swap_memory", derive_seed(seed, "eval-swap", split))
                conditions.extend([
                    ("replace_key_fact", replaced, np.asarray(replaced.meta["source_episode"]),
                     np.asarray(replaced.meta["source_query"])),
                    ("swap_memory", arrays, np.arange(n), np.asarray(swapped.meta["pairing"])),
                ])
            for protocol in protocols:
                _sync(experiment)
                set_started = time.perf_counter()
                original_predictions: dict[tuple[int, int, int], int] = {}
                for condition_name, dataset, sources, query_or_pairing in conditions:
                    condition = f"{protocol}/{condition_name}"
                    for begin in range(0, len(dataset.episode_seed), batch_size):
                        end = min(begin + batch_size, len(dataset.episode_seed))
                        part = _slice(dataset, slice(begin, end))
                        _sync(experiment)
                        before = time.perf_counter()
                        prepared = experiment.prepare(part, protocol=protocol)
                        donor_ids = part.episode_seed
                        if condition_name == "swap_memory":
                            donor_indices = query_or_pairing[begin:end]
                            donor = experiment.prepare(_slice(arrays, donor_indices), protocol=protocol)
                            prepared = replace(prepared,
                                               snapshot=replace(prepared.snapshot, states=donor.snapshot.states),
                                               write_count=donor.write_count)
                            donor_ids = arrays.episode_seed[donor_indices]
                        _sync(experiment)
                        timing["writer_seconds"] += time.perf_counter() - before
                        baseline = _snapshot_hashes(prepared)
                        shape = part.answer.shape
                        source_i = np.broadcast_to(sources[begin:end, None], shape)
                        source_q = (query_or_pairing[begin:end, None] if condition_name == "replace_key_fact"
                                    else np.broadcast_to(np.arange(q), shape))
                        episode_ids = arrays.episode_seed[source_i]
                        answers = _numpy(prepared.answer_ids)
                        if not np.array_equal(answers, _answer_tokens(part.answer, spec.V)):
                            raise AssertionError("prepared token answers differ from generated labels")
                        candidates, valid = prepared.candidate_ids, prepared.candidate_mask
                        if candidates.ndim != 2 or valid.shape != candidates.shape or not bool(valid.any(-1).all()):
                            raise AssertionError("each episode needs a nonempty candidate set")
                        if not bool(((prepared.answer_ids[:, :, None] == candidates[:, None, :])
                                     & valid[:, None, :]).any(-1).all()):
                            raise AssertionError("answer token absent from episode candidate set")
                        candidate_size = np.broadcast_to(_numpy(valid.sum(-1))[:, None], shape)
                        first_inputs = None
                        for depth in depths:
                            _sync(experiment)
                            before = time.perf_counter()
                            logits, x = experiment.score(prepared, depth, capture_inputs=True)
                            _sync(experiment)
                            timing["reader_seconds"][str(depth)] += time.perf_counter() - before
                            _finite(logits, "answer logits")
                            if x is None:
                                raise AssertionError("Gate 0 requires captured question-token adapter inputs")
                            _finite(x, "question inputs")
                            u = x.float()[..., :0] if experiment.adapter is None else experiment.adapter.u(x.float())
                            _finite(u, "projected question inputs")
                            inputs = dict(x=tensor_hash(x), u=tensor_hash(u))
                            if first_inputs is None:
                                first_inputs = inputs
                                invariant_hashes.append(dict(split=split, protocol=protocol,
                                                             condition=condition, episode_start=begin,
                                                             M=baseline["memory"],
                                                             write_count=baseline["write_count"], **inputs))
                            if inputs != first_inputs or _snapshot_hashes(prepared) != baseline:
                                raise AssertionError(f"Gate 0: x/u/M/write_count/KV changed at L={depth}")
                            hashes.append(dict(split=split, protocol=protocol, condition=condition,
                                               batch_start=begin, L=depth, **baseline, **inputs))
                            prediction = _numpy(logits.argmax(-1))
                            restricted = logits.gather(-1, candidates[:, None, :].expand(-1, shape[1], -1))
                            choice = restricted.masked_fill(~valid[:, None, :], -torch.inf).argmax(-1)
                            candidate_prediction = _numpy(candidates.gather(1, choice))
                            if condition_name == "original":
                                for ei, qi, pred in zip(episode_ids.ravel(), source_q.ravel(), prediction.ravel()):
                                    original_predictions[(int(ei), int(qi), depth)] = int(pred)
                            old_prediction = np.asarray([
                                original_predictions[(int(ei), int(qi), depth)]
                                for ei, qi in zip(episode_ids.ravel(), source_q.ravel())
                            ], dtype=np.int64).reshape(shape)
                            values = dict(split=np.full(shape, split), condition=np.full(shape, condition),
                                          protocol=np.full(shape, protocol), episode_id=episode_ids,
                                          query_id=source_q, hop=arrays.q_hop_true[source_i, source_q],
                                          L=np.full(shape, depth, dtype=np.int64), answer=answers,
                                          prediction=prediction, correct=prediction == answers,
                                          full_prediction=prediction, full_correct=prediction == answers,
                                          candidate_prediction=candidate_prediction,
                                          candidate_correct=candidate_prediction == answers,
                                          candidate_size=candidate_size,
                                          original_answer=original_answers[source_i, source_q],
                                          original_prediction=old_prediction,
                                          memory_episode_id=np.broadcast_to(donor_ids[:, None], shape))
                            for name, value in values.items():
                                chunks.setdefault(name, []).append(np.asarray(value).ravel().copy())
                _sync(experiment)
                timing["eval_sets"].append(dict(split=split, protocol=protocol, episodes=n, queries=n*q,
                                                  seconds=time.perf_counter() - set_started))
        samples = {name: np.concatenate(parts) for name, parts in chunks.items()}
        stats = config.get("stats", {})
        metrics = dict(table=aggregate_samples(samples), memory_hashes=hashes,
                       invariant_hashes=invariant_hashes, gate0_passed=True,
                       depth_comparisons=depth_comparisons(samples, resamples=int(stats.get("resamples", 2000)),
                                                           seed=int(stats.get("seed", 2718))),
                       primary_metric="full_vocabulary_first_token_exact_match",
                       secondary_metric="episode_candidate_token_accuracy",
                       chance_metric="mean_inverse_episode_candidate_count_for_secondary_accuracy",
                       query_protocol="one_query_per_hop_from_support_v1", evaluation_seed=seed,
                       episode_ranges={split: dict(split=generator_split, start=start, count=count)
                                       for split, (generator_split, start) in windows.items()},
                       counterfactual_hop="original_query_hop",
                       swap_candidate_set="original_episode", swap_labels="original_answer",
                       swap_context_retained_kv="original_episode")
        np.savez_compressed(out / "samples.npz", **samples)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        _sync(experiment)
        timing["evaluation_seconds"] = time.perf_counter() - started
        return metrics, timing
    finally:
        experiment.model.train(previous_mode)
