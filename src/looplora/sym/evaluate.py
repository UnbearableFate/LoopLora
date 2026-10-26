"""Paired depth evaluation on fixed, explicitly stratified episode queries."""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np
import torch

from looplora.data.relations import EpisodeArrays, TaskSpec, counterfactual, generate
from looplora.infra.seeds import derive_seed, stable_hash


def stratify_queries(arrays: EpisodeArrays, spec: TaskSpec, hops: list[int], seed: int) -> EpisodeArrays:
    """Choose one query per hop from each unchanged support graph.

    This is an evaluation query design, independent of training query sampling.
    Latest facts win for repeated keys. Missing-data tasks retain their generated
    queries to preserve the generator's missingness event and require coverage.
    """
    hops = [1] if spec.kind == "t1" else list(hops)
    if not hops or len(set(hops)) != len(hops) or any(h < 1 for h in hops):
        raise ValueError("evaluation hops must be distinct positive integers")
    n = len(arrays.episode_seed)
    starts = np.empty((n, len(hops)), dtype=np.int32)
    rels = np.zeros_like(starts)
    given = np.full_like(starts, -1)
    answers = np.empty_like(starts)
    for i, episode_seed in enumerate(arrays.episode_seed):
        graph = {(int(head), int(rel)): int(tail)
                 for (head, rel, tail), valid in zip(arrays.facts[i], arrays.fact_valid[i]) if valid}
        for j, hop in enumerate(hops):
            rng = np.random.default_rng(derive_seed(seed, "eval-query", int(episode_seed), hop))
            if spec.name == "t3_missing":
                matches = np.flatnonzero(arrays.q_hop_true[i] == hop)
                if not len(matches):
                    raise ValueError("t3_missing evaluation needs generated query coverage for every hop")
                q = int(rng.choice(matches))
                starts[i, j], rels[i, j] = arrays.q_start[i, q], arrays.q_rel[i, q]
                given[i, j], answers[i, j] = arrays.q_hop_given[i, q], arrays.answer[i, q]
                continue
            if spec.kind == "t1":
                choices = [(head, rel, tail) for (head, rel), tail in sorted(graph.items())]
            else:
                choices = []
                for start, rel in sorted(graph):
                    if rel != 0:
                        continue
                    node, visited = start, set()
                    for _ in range(hop):
                        if node in visited or (node, 0) not in graph:
                            break
                        visited.add(node)
                        node = graph[node, 0]
                    else:
                        terminal = (node, 0) not in graph
                        if node not in visited and (spec.kind == "t2_walk" or terminal):
                            choices.append((start, 0, node))
            if not choices:
                raise ValueError(f"episode {int(episode_seed)} has no valid query at hop {hop}")
            starts[i, j], rels[i, j], answers[i, j] = choices[int(rng.integers(len(choices)))]
            if spec.kind in ("t1", "t2_walk"):
                given[i, j] = hop
    return replace(arrays, q_start=starts, q_rel=rels, q_hop_given=given,
                   q_hop_true=np.broadcast_to(np.asarray(hops, dtype=np.int32), starts.shape).copy(),
                   answer=answers, meta={**arrays.meta, "spec": {**arrays.meta["spec"], "Q": len(hops)},
                                         "eval_query_protocol": "one_query_per_hop_from_support_v1"})


def _slice(arrays: EpisodeArrays, index: slice | np.ndarray) -> EpisodeArrays:
    return EpisodeArrays(**{f.name: (getattr(arrays, f.name) if f.name == "meta"
                                    else getattr(arrays, f.name)[index]) for f in fields(arrays)})


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash exact tensor storage, including dtype and dimensions, with blake2b."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _snapshot(prepared: Any) -> dict[str, str]:
    return {name: tensor_hash(getattr(prepared, name)) for name in ("x", "u", "memory", "write_count")}


def _sync(model: Any) -> None:
    if torch.device(model.device).type == "cuda":
        torch.cuda.synchronize(model.device)


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _require_finite(tensor: torch.Tensor, name: str, context: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"non-finite {name} during evaluation ({context})")


_EVALUATION_OVERRIDE_KEYS = frozenset({"eval.seed", "eval.episodes", "eval.splits",
                                       "eval.batch_size", "eval.counterfactuals"})


def parse_evaluation_overrides(assignments: list[str]) -> dict[str, Any]:
    """Parse allowlisted YAML/JSON values; repeated keys use their last value."""
    import yaml

    overrides = {}
    for assignment in assignments:
        key, separator, text = assignment.partition("=")
        key = key.strip()
        if not separator:
            raise ValueError(f"invalid evaluation override {assignment!r}; expected key=value")
        if key not in _EVALUATION_OVERRIDE_KEYS:
            allowed = ", ".join(sorted(_EVALUATION_OVERRIDE_KEYS))
            raise ValueError(f"evaluation override {key!r} is not allowed; allowed keys: {allowed}")
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML/JSON value for {key}: {text!r}") from error
        if key in {"eval.seed", "eval.episodes", "eval.batch_size"}:
            minimum = 0 if key == "eval.seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} override must be an integer >= {minimum}")
        elif key == "eval.counterfactuals":
            if not isinstance(value, bool):
                raise ValueError(f"{key} override must be a boolean")
        elif not isinstance(value, list) or not value or any(not isinstance(split, str) for split in value):
            raise ValueError("eval.splits override must be a nonempty YAML/JSON list of split names")
        overrides[key] = value
    return overrides


def resolve_evaluation_config(checkpoint_config: dict[str, Any], overrides: dict[str, Any],
                              device: str | None = None) -> dict[str, Any]:
    """Apply evaluation-only values to a copy, then validate the whole config."""
    from looplora.sym.config import resolve_config

    if set(overrides) - _EVALUATION_OVERRIDE_KEYS:
        raise ValueError("evaluation overrides contain a non-allowlisted key")
    config = copy.deepcopy(checkpoint_config)
    for key, value in overrides.items():
        config.setdefault("eval", {})[key.removeprefix("eval.")] = copy.deepcopy(value)
    if device is not None:
        config["device"] = device
    return resolve_config(config)


def aggregate_samples(samples: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Reproduce the public hop × depth accuracy table from aligned samples."""
    keys = sorted(set(zip(samples["split"].tolist(), samples["condition"].tolist(),
                          samples["hop"].tolist(), samples["L"].tolist())))
    table = []
    for split, condition, hop, depth in keys:
        mask = ((samples["split"] == split) & (samples["condition"] == condition)
                & (samples["hop"] == hop) & (samples["L"] == depth))
        row = dict(split=split, condition=condition, hop=int(hop), L=int(depth),
                   n=int(mask.sum()), accuracy=float(samples["correct"][mask].mean()),
                   full_accuracy=float(samples["full_correct"][mask].mean()),
                   chance_accuracy=float((1.0 / samples["candidate_size"][mask]).mean()))
        if condition == "replace_key_fact":
            moved = samples["prediction"][mask] != samples["original_prediction"][mask]
            row["prediction_changed_rate"] = float(moved.mean())
            previously_correct = samples["original_prediction"][mask] == samples["original_answer"][mask]
            row["n_original_correct"] = int(previously_correct.sum())
            row["new_answer_accuracy_given_original_correct"] = (
                float(samples["correct"][mask][previously_correct].mean()) if previously_correct.any() else None)
        table.append(row)
    return table


def _aggregate_diagnostics(samples: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    diagnostics = []
    for split, condition, depth in sorted(set(zip(samples["split"], samples["condition"], samples["L"]))):
        mask = (samples["split"] == split) & (samples["condition"] == condition) & (samples["L"] == depth)
        cosine = samples["first_read_cosine"][mask]
        valid = np.isfinite(cosine)
        diagnostics.append(dict(split=str(split), condition=str(condition), L=int(depth),
                                z_norm=float(samples["z_norm"][mask].mean()),
                                memory_frobenius=float(samples["memory_norm"][mask].mean()),
                                first_read_cosine=float(cosine[valid].mean()) if valid.any() else None,
                                first_read_count=int(valid.sum())))
    return diagnostics


def _file_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_evaluation(out: Path, *, barrier_passed: bool = False) -> dict[str, Any]:
    """Aggregate saved samples only after the caller attests the global barrier."""
    if barrier_passed is not True:
        raise ValueError("finalization requires explicit global training-and-confirmation-samples barrier clearance")
    out = Path(out)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"evaluation metrics already exist: {out}")
    metadata = json.loads((out / "evaluation.json").read_text())
    if not metadata.get("aggregation_deferred"):
        raise ValueError("evaluation was not generated in samples-only mode")
    if _file_hash(out / "samples.npz") != metadata["samples_hash"]:
        raise ValueError("saved evaluation samples changed after generation")
    with np.load(out / "samples.npz", allow_pickle=False) as archive:
        samples = {key: archive[key] for key in archive.files}
    metrics = {**metadata, "aggregation_deferred": False, "finalized_after_global_barrier": True,
               "table": aggregate_samples(samples), "diagnostics": _aggregate_diagnostics(samples)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics


@torch.no_grad()
def evaluate(model: Any, config: dict[str, Any], out: Path, *, samples_only: bool = False,
             provenance: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write immutable-depth paired samples, metrics, diagnostics, and Gate 0 hashes."""
    from looplora.sym.model import make_task_spec

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if samples_only and ((out / "metrics.json").exists() or (out / "samples.npz").exists()):
        raise FileExistsError(f"samples-only evaluation output already exists: {out}")
    settings = config["eval"]
    depths = [int(depth) for depth in settings["depths"]]
    fixed_depth = {"B4": 1, "B6": 1, "B7": 1, "B8": 4}.get(config["variant"])
    if fixed_depth is not None and (config["variant"] == "B8" or not settings.get("fixed_depth_sweep", False)):
        depths = [fixed_depth]
    if not depths or len(set(depths)) != len(depths) or any(depth < 0 for depth in depths):
        raise ValueError("evaluation depths must be distinct nonnegative integers")
    count, batch_size = int(settings["episodes"]), int(settings["batch_size"])
    if count < 1 or batch_size < 1:
        raise ValueError("evaluation episodes and batch_size must be positive")
    if settings.get("counterfactuals", True) and "test" in settings["splits"] and count < 2:
        raise ValueError("swap_memory requires at least two evaluation episodes")
    if (settings.get("swap_successors", False) and settings.get("counterfactuals", True)
            and "test" in settings["splits"] and not {1, 4, 8}.issubset(depths)):
        raise ValueError("swap_successors requires evaluation depths 1, 4, 8")
    spec = make_task_spec(config, evaluation=True)
    seed = int(settings["seed"])
    chunks: dict[str, list[np.ndarray]] = {}
    hashes = []
    timing: dict[str, Any] = {"writer_seconds": 0.0, "reader_seconds": {str(depth): 0.0 for depth in depths}}
    previous_mode = model.training
    model.eval()
    _sync(model)
    started = time.perf_counter()
    try:
        for split in settings["splits"]:
            arrays = stratify_queries(generate(spec, split, 0, count, seed), spec, settings["hops"], seed)
            n, q = arrays.answer.shape
            original_predictions: dict[tuple[int, int, int], int] = {}
            conditions = [("original", arrays, np.arange(n), None)]
            if split == "test" and settings.get("counterfactuals", True):
                replaced, _ = counterfactual(arrays, "replace_key_fact", derive_seed(seed, "eval-replace", split))
                removed, _ = counterfactual(arrays, "remove_required_fact", derive_seed(seed, "eval-remove", split))
                swapped, _ = counterfactual(arrays, "swap_memory", derive_seed(seed, "eval-swap", split))
                conditions.extend([
                    ("replace_key_fact", replaced, np.asarray(replaced.meta["source_episode"]),
                     np.asarray(replaced.meta["source_query"])),
                    ("remove_required_fact", removed, np.asarray(removed.meta["source_episode"]),
                     np.asarray(removed.meta["source_query"])),
                    ("swap_memory", arrays, np.arange(n), np.asarray(swapped.meta["pairing"])),
                ])
                if settings.get("swap_successors", False):
                    successors, _ = counterfactual(arrays, "swap_successors", seed)
                    sources = np.asarray(successors.meta["source_episode"])
                    questions = np.asarray(successors.meta["source_query"])
                    if not np.all(successors.answer[:, 0] != arrays.answer[sources, questions]):
                        raise AssertionError("swap_successors did not change every terminal answer")
                    conditions.append(("swap_successors", successors, sources, questions))
            codebooks = None
            if config.get("writer") == "oracle_orth":
                _sync(model)
                before_codes = time.perf_counter()
                codebooks = model.orthogonal_codebooks(arrays)
                _sync(model)
                timing["writer_seconds"] += time.perf_counter() - before_codes
            for condition, dataset, source_episodes, query_or_pairing in conditions:
                condition_depths = [1, 4, 8] if condition == "swap_successors" else depths
                for begin in range(0, len(dataset.episode_seed), batch_size):
                    end = min(begin + batch_size, len(dataset.episode_seed))
                    part = _slice(dataset, slice(begin, end))
                    src = source_episodes[begin:end]
                    path_condition = condition in {"replace_key_fact", "remove_required_fact"}
                    fork_condition = path_condition or condition == "swap_successors"
                    insufficient = condition == "remove_required_fact" and spec.kind in {"t1", "t2_walk"}
                    _sync(model)
                    before_prepare = time.perf_counter()
                    options = ({} if codebooks is None else {"oracle_codes": codebooks[
                        torch.as_tensor(src, dtype=torch.long, device=codebooks.device)]})
                    if codebooks is not None and path_condition:
                        # Give each query fork its own fresh-entity slots, keeping
                        # the original episode's basis and entity slots unchanged.
                        extras = replace(part, meta={**part.meta, "source_episode": list(range(end - begin))})
                        options = {"oracle_codes": model.orthogonal_codebooks(
                            _slice(arrays, src), extras, include_insufficient=insufficient)}
                    prepared = model.prepare(part, **options)
                    if condition == "remove_required_fact":
                        # A chain cut may return the query start, whose only fact
                        # was removed. Keep source entities as candidates as well.
                        candidate = prepared.candidate_mask.clone()
                        for row, source in enumerate(src):
                            entities = arrays.facts[source, arrays.fact_valid[source]][:, (0, 2)].ravel()
                            candidate[row, torch.as_tensor(entities, device=candidate.device, dtype=torch.long)] = True
                        if insufficient:
                            candidate[:, spec.V] = True
                        prepared = replace(prepared, candidate_mask=candidate)
                    if condition == "swap_memory":
                        donor_options = ({} if codebooks is None else {"oracle_codes": codebooks[
                            torch.as_tensor(query_or_pairing[begin:end], dtype=torch.long, device=codebooks.device)]})
                        donor = model.prepare(_slice(arrays, query_or_pairing[begin:end]), **donor_options)
                        prepared = replace(prepared, memory=donor.memory, write_count=donor.write_count)
                    _sync(model)
                    timing["writer_seconds"] += time.perf_counter() - before_prepare
                    context = f"split={split}, condition={condition}, batch_start={begin}"
                    for name in ("x", "u", "memory"):
                        _require_finite(getattr(prepared, name), name, context)
                    snapshot = _snapshot(prepared)
                    shape = part.answer.shape
                    source_q = (query_or_pairing[begin:end, None] if fork_condition
                                else np.broadcast_to(np.arange(q), shape))
                    source_i = np.broadcast_to(src[:, None], shape)
                    episode_ids = np.broadcast_to(arrays.episode_seed[src, None], shape)
                    hops = arrays.q_hop_true[source_i, source_q]
                    answers = _numpy(prepared.answer)
                    if not np.array_equal(answers, part.answer):
                        raise AssertionError("prepared answers differ from episode labels")
                    if not prepared.candidate_mask.gather(1, prepared.answer).all():
                        raise AssertionError("evaluation answer absent from episode candidate set")
                    candidate_size = _numpy(prepared.candidate_mask.sum(dim=-1))
                    if np.any(candidate_size < 1):
                        raise AssertionError("empty candidate set")
                    candidate_size = np.broadcast_to(candidate_size[:, None], shape)
                    for depth in condition_depths:
                        _sync(model)
                        before_read = time.perf_counter()
                        result = model.read(prepared, depth)
                        _sync(model)
                        timing["reader_seconds"][str(depth)] += time.perf_counter() - before_read
                        after = _snapshot(prepared)
                        if snapshot != after:
                            raise AssertionError(f"Gate 0: read at L={depth} mutated prepared x/u/M/write_count")
                        hashes.append(dict(split=split, condition=condition, batch_start=begin, L=depth, **after))
                        logits = result["logits"]
                        _require_finite(logits, "logits", f"{context}, L={depth}")
                        _require_finite(result["z"], "reader state", f"{context}, L={depth}")
                        prediction = _numpy(logits.masked_fill(~prepared.candidate_mask[:, None, :], -torch.inf).argmax(-1))
                        full_prediction = _numpy(logits.argmax(-1))
                        original_answer = arrays.answer[source_i, source_q]
                        if condition == "original":
                            for ei, qi, pred in zip(episode_ids.ravel(), source_q.ravel(), prediction.ravel()):
                                original_predictions[(int(ei), int(qi), depth)] = int(pred)
                        original_prediction = np.asarray([
                            original_predictions[(int(ei), int(qi), depth)]
                            for ei, qi in zip(episode_ids.ravel(), source_q.ravel())], dtype=np.int64).reshape(shape)
                        z_norm = _numpy(torch.linalg.vector_norm(result["z"], dim=-1))
                        memory_norm = _numpy(torch.linalg.matrix_norm(prepared.memory, ord="fro"))
                        cosine = _numpy(result["first_read_cosine"])
                        if not np.isfinite(z_norm).all() or not np.isfinite(memory_norm).all() or np.isinf(cosine).any():
                            raise FloatingPointError(f"non-finite evaluation diagnostic ({context}, L={depth})")
                        values = dict(split=np.full(shape, split), condition=np.full(shape, condition),
                                      episode_id=episode_ids, query_id=source_q, hop=hops,
                                      counterfactual_hop_true=part.q_hop_true,
                                      L=np.full(shape, depth, dtype=np.int64), prediction=prediction,
                                      answer=answers, correct=prediction == answers,
                                      full_prediction=full_prediction, full_correct=full_prediction == answers,
                                      candidate_size=candidate_size, original_answer=original_answer,
                                      original_prediction=original_prediction, z_norm=z_norm,
                                      memory_norm=np.broadcast_to(memory_norm[:, None], shape), first_read_cosine=cosine)
                        if condition == "swap_memory":
                            donor_ids = arrays.episode_seed[query_or_pairing[begin:end]]
                        else:
                            donor_ids = part.episode_seed
                        values["memory_episode_id"] = np.broadcast_to(donor_ids[:, None], shape)
                        for name, value in values.items():
                            chunks.setdefault(name, []).append(np.asarray(value).ravel().copy())
        samples = {name: np.concatenate(value) for name, value in chunks.items()}
        metadata = dict(memory_hashes=hashes,
                       gate0_passed=True, query_protocol="one_query_per_hop_from_support_v1",
                       counterfactual_hop="original_query_hop", swap_candidate_set="original_episode",
                       swap_code_space="donor_memory_in_original_query_code_space",
                       remove_candidate_set="original_source_and_modified_fact_entities_plus_insufficient_when_applicable",
                       codebook_policy=("fixed_source_episode_basis_with_replacement_entity_slots"
                                        if config.get("writer") == "oracle_orth" else "fixed_global_codes_or_backbone"),
                       counterfactual_codebook_allocation="per_query_fork",
                       swap_successors_depths=[1, 4, 8], swap_successors_diagnostic_only=True,
                       swap_successors_seed=seed,
                       sample_hop_fields={"hop": "original source-query hop",
                                          "counterfactual_hop_true": "generated part.q_hop_true; swap_memory retains original queries"},
                       evaluation_seed=seed, aggregation_deferred=samples_only,
                       **(provenance or {}))
        metadata.update(config=copy.deepcopy(config), config_hash=stable_hash(config))
        if "checkpoint_config" in metadata:
            metadata["checkpoint_config_hash"] = stable_hash(metadata["checkpoint_config"])
        np.savez_compressed(out / "samples.npz", **samples)
        metadata["samples_hash"] = _file_hash(out / "samples.npz")
        if samples_only:
            (out / "evaluation.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            config_record = {key: metadata[key] for key in
                             ("checkpoint", "checkpoint_config", "checkpoint_config_hash", "config", "config_hash", "eval_overrides")
                             if key in metadata}
            (out / "config.json").write_text(json.dumps(config_record, indent=2, allow_nan=False) + "\n")
            result = metadata
        else:
            result = {**metadata, "table": aggregate_samples(samples), "diagnostics": _aggregate_diagnostics(samples)}
            (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        _sync(model)
        timing["evaluation_seconds"] = time.perf_counter() - started
        if samples_only:
            (out / "timing.json").write_text(json.dumps(timing, indent=2, allow_nan=False) + "\n")
        return result, timing
    finally:
        model.train(previous_mode)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved symbolic checkpoint without training.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New or empty evaluation output directory")
    parser.add_argument("--device", help="Override the checkpoint device, e.g. cpu or cuda:0")
    parser.add_argument("--samples-only", action="store_true",
                        help="Save samples/provenance only; defer all aggregates until the global barrier")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Evaluation override (YAML/JSON); repeat keys for last-value-wins. "
                             "Allowed: eval.seed, eval.episodes, eval.splits (e.g. '[test]'), "
                             "eval.batch_size, eval.counterfactuals")
    args = parser.parse_args(argv)
    try:
        overrides = parse_evaluation_overrides(args.set)
    except ValueError as error:
        parser.error(str(error))
    checkpoint, out = args.checkpoint.resolve(), args.out.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if out == checkpoint.parent or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise FileExistsError(f"evaluation requires a new or empty output directory: {out}")
    from looplora.sym.train import load_checkpoint

    model = load_checkpoint(checkpoint, device=args.device)
    checkpoint_config = copy.deepcopy(getattr(model, "checkpoint_config", model.config))
    try:
        config = resolve_evaluation_config(checkpoint_config, overrides, device=args.device)
    except ValueError as error:
        parser.error(f"invalid evaluation configuration: {error}")
    torch.set_num_threads(config["runtime"]["threads"])
    device = torch.device(config["device"])
    provenance = dict(checkpoint=str(checkpoint), checkpoint_config=checkpoint_config,
                      config=config, config_hash=stable_hash(config), eval_overrides=overrides)
    if args.samples_only:
        metrics, timing = evaluate(model, config, out, samples_only=True, provenance=provenance)
    else:
        metrics, timing = evaluate(model, config, out)
    timing["runtime"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "host": socket.gethostname(), "job_id": os.environ.get("PBS_JOBID"),
        "device": str(device), "dtype": "float32", "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    timing["measurement"] = {
        "synchronization": "CUDA synchronize before and after each timed pass" if device.type == "cuda" else "host clock",
        "writer_scope": "evaluation preparation, including feature projection and codebooks",
        "reader_scope": "core read, branch, vocabulary logits, and first-read diagnostic",
        "evaluation_scope": "episode generation, preparation, all readers, validation, hashing, and sample/metric output",
    }
    if not args.samples_only:
        metrics.update(provenance)
        (out / "config.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (out / "timing.json").write_text(json.dumps(timing, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
