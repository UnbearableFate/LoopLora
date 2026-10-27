"""NCR: neighbour-confusion rate of single-read errors (descriptive, existing samples only).

For a hop-h query on a chain the answer is the chain end ``succ^h(start)``. A
single read (``L = 1``) can at best return ``succ(start)``: the first-hop
neighbour. This module classifies every prediction of a saved ``samples.npz``
(condition ``original``) for a hop ``h >= 2`` into

* ``first_hop``: the prediction is ``succ(start)`` (the neighbour);
* ``one_short``: the node one step before the answer (distinct from
  ``first_hop`` only when ``h >= 3``);
* ``start``: the query entity itself;
* ``terminal_other``: another chain end (an entity without an outgoing fact);
* ``other``: any other candidate,

and reports, per category, the rate over all queries and the share of the
errors next to the values expected when the guess (respectively the error) is
uniform over the episode's candidates. The episode graphs are recovered by
regenerating the evaluation episodes from the evaluation config recorded next to
the samples and are verified against the saved answers, hops and candidate
counts row by row. Nothing is trained or re-evaluated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from looplora.data.relations import EpisodeArrays, generate
from .evaluate import stratify_queries
from .model import make_task_spec
from .stats import load_samples
from . import followup_stats as fs

CATEGORIES = ("first_hop", "one_short", "start", "terminal_other", "other")


class UndefinedShares(ValueError):
    """A training seed has no errors in the group: its error shares, and so their pooling, are undefined."""


def load_directory(directory: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Samples and the recorded evaluation config of one evaluation directory."""
    directory = Path(directory)
    for name in ("config.json", "metrics.json", "evaluation.json"):
        record = directory / name
        if record.exists():
            config = json.loads(record.read_text())["config"]
            break
    else:
        raise FileNotFoundError(f"no config.json, metrics.json or evaluation.json in {directory}")
    samples = load_samples(directory)
    for name in ("prediction", "candidate_size"):
        if name not in samples:
            raise ValueError(f"samples in {directory} lack the {name!r} array")
    return samples, config


def regenerate(config: Mapping[str, Any], split: str) -> EpisodeArrays:
    """The evaluation episodes of a recorded evaluation config (stratified, one query per hop)."""
    spec = make_task_spec(dict(config), evaluation=True)
    settings = config["eval"]
    seed = int(settings["seed"])
    return stratify_queries(generate(spec, split, 0, int(settings["episodes"]), seed), spec,
                            list(settings["hops"]), seed)


def category_sizes(terminals: set[int], n_candidates: int, hop: int, answer: int) -> dict[str, int]:
    """Number of wrong candidates in each category (they partition the C - 1 wrong candidates)."""
    if hop < 2:
        raise ValueError("neighbour confusion is defined for hops of at least 2")
    sizes = {"first_hop": 1, "one_short": 1 if hop >= 3 else 0, "start": 1,
             "terminal_other": len(terminals - {answer})}
    sizes["other"] = n_candidates - 1 - sum(sizes.values())
    if sizes["other"] < 0:
        raise ValueError("category sizes exceed the candidate set")
    return sizes


def classify(graph: Mapping[int, int], terminals: set[int], n_candidates: int, start: int, hop: int,
             answer: int, prediction: int) -> tuple[str | None, dict[str, int]]:
    """(category of a wrong prediction or None when correct, size of every category)."""
    sizes = category_sizes(terminals, n_candidates, hop, answer)
    path = [start]
    for _ in range(hop):
        if path[-1] not in graph:
            raise ValueError("query path leaves the episode graph")
        path.append(graph[path[-1]])
    if path[-1] != answer or len(set(path)) != hop + 1:
        raise ValueError("regenerated path does not end in the saved answer or revisits a node")
    if prediction == answer:
        return None, sizes
    if prediction == path[1]:
        return "first_hop", sizes
    if hop >= 3 and prediction == path[hop - 1]:
        return "one_short", sizes
    if prediction == start:
        return "start", sizes
    if prediction in terminals:
        return "terminal_other", sizes
    return "other", sizes


def episode_tables(arrays: EpisodeArrays) -> list[tuple[dict[int, int], set[int], int]]:
    """Per episode: successor map, chain ends and candidate count (entities in valid facts)."""
    tables = []
    for facts, valid in zip(arrays.facts, arrays.fact_valid):
        valid_facts = [(int(h), int(r), int(t)) for (h, r, t), ok in zip(facts, valid) if ok]
        graph = {h: t for h, r, t in valid_facts if r == 0}
        entities = {e for h, _, t in valid_facts for e in (h, t)}
        terminals = {t for _, t in graph.items()} - set(graph)
        tables.append((graph, terminals, len(entities)))
    return tables


def confusion_rows(samples: Mapping[str, np.ndarray], arrays: EpisodeArrays, *, hop: int, depth: int = 1,
                   split: str, condition: str = "original") -> dict[str, Any]:
    """Per-query indicators of one (split, condition, depth, hop) cell, ordered by (episode, query).

    ``episode_id``, ``correct``, ``error`` and ``uniform_correct`` (1 / C) are
    1-D. ``category[name]`` marks errors in the category, ``uniform_rate[name]``
    is the category's probability under a uniform guess over the C candidates
    and ``uniform_error[name]`` its probability under a uniform wrong guess
    (zero on correct rows).
    """
    if hop < 2:
        raise ValueError("neighbour confusion is defined for hops of at least 2")
    index = fs.select_rows(samples, split, condition, depth)
    index = index[np.asarray(samples["hop"])[index] == hop]
    if not index.size:
        raise ValueError(f"no samples for hop {hop} at L={depth}")
    position = {int(seed): i for i, seed in enumerate(arrays.episode_seed)}
    tables = episode_tables(arrays)
    plain = {name: [] for name in ("episode_id", "correct", "error", "uniform_correct")}
    category = {name: [] for name in CATEGORIES}
    uniform_rate = {name: [] for name in CATEGORIES}
    uniform_error = {name: [] for name in CATEGORIES}
    for row in index:
        episode = position.get(int(samples["episode_id"][row]))
        query = int(samples["query_id"][row])
        if episode is None:
            raise ValueError("saved episode is absent from the regenerated evaluation set")
        if (int(arrays.answer[episode, query]) != int(samples["answer"][row])
                or int(arrays.q_hop_true[episode, query]) != int(samples["hop"][row])):
            raise ValueError("regenerated episodes differ from the saved samples")
        graph, terminals, n_candidates = tables[episode]
        if int(samples["candidate_size"][row]) != n_candidates:
            raise ValueError("saved candidate count differs from the regenerated episode")
        label, sizes = classify(graph, terminals, n_candidates, int(arrays.q_start[episode, query]), hop,
                                int(samples["answer"][row]), int(samples["prediction"][row]))
        if (label is None) != bool(samples["correct"][row]):
            raise ValueError("saved correctness disagrees with the saved prediction and answer")
        plain["episode_id"].append(int(samples["episode_id"][row]))
        plain["correct"].append(label is None)
        plain["error"].append(label is not None)
        plain["uniform_correct"].append(1.0 / n_candidates)
        for name in CATEGORIES:
            category[name].append(label == name)
            uniform_rate[name].append(sizes[name] / n_candidates)
            uniform_error[name].append(0.0 if label is None else sizes[name] / (n_candidates - 1))
    result: dict[str, Any] = {name: np.asarray(values) for name, values in plain.items()}
    result["category"] = {name: np.asarray(values) for name, values in category.items()}
    result["uniform_rate"] = {name: np.asarray(values) for name, values in uniform_rate.items()}
    result["uniform_error"] = {name: np.asarray(values) for name, values in uniform_error.items()}
    return result


def summarize(per_seed: Mapping[int, Mapping[str, Any]], *, confidence: float = 0.95, resamples: int = 2000,
              label: Sequence[Any] = ()) -> dict[str, Any]:
    """Pooled accuracy, category rates and error shares, with the uniform-guess reference values.

    Intervals are percentile intervals of a joint whole-episode bootstrap
    (the resampling of the registered S9 studies); they are descriptive.
    """
    seeds = sorted(int(s) for s in per_seed)
    episode_id = np.asarray(per_seed[seeds[0]]["episode_id"])
    for s in seeds:
        if not np.array_equal(per_seed[s]["episode_id"], episode_id):
            raise ValueError("seeds disagree on the evaluation episodes")
    options = dict(confidence=confidence, resamples=resamples)
    accuracy = fs.pooled_mean_contrast({s: per_seed[s]["correct"].astype(float) for s in seeds}, episode_id,
                                       label=(*label, "accuracy"), **options)
    errors = {s: per_seed[s]["error"].astype(float) for s in seeds}
    if any(errors[s].sum() == 0 for s in seeds):
        raise UndefinedShares("a training seed has no errors: error shares are undefined")
    categories = {}
    for name in CATEGORIES:
        rate = fs.pooled_mean_contrast({s: per_seed[s]["category"][name].astype(float) for s in seeds},
                                       episode_id, label=(*label, "rate", name), **options)
        share = fs.pooled_ratio({s: per_seed[s]["category"][name].astype(float) for s in seeds}, errors,
                                episode_id, label=(*label, "share", name), **options)
        categories[name] = dict(
            rate=rate["difference"], rate_ci_low=rate["ci_low"], rate_ci_high=rate["ci_high"],
            uniform_rate=float(np.mean([per_seed[s]["uniform_rate"][name].mean() for s in seeds])),
            share=share["ratio"], share_ci_low=share["ci_low"], share_ci_high=share["ci_high"],
            uniform_share=float(np.mean([per_seed[s]["uniform_error"][name].sum() / errors[s].sum()
                                         for s in seeds])),
            per_seed_rate=[row["difference"] for row in rate["per_seed"]],
            per_seed_share=[row["ratio"] for row in share["per_seed"]])
    chance = float(np.mean([per_seed[s]["uniform_correct"].mean() for s in seeds]))
    return dict(accuracy=accuracy["difference"], accuracy_ci_low=accuracy["ci_low"],
                accuracy_ci_high=accuracy["ci_high"], chance_accuracy=chance,
                below_chance=bool(accuracy["difference"] < chance), confidence=float(confidence),
                n_episodes=accuracy["n_episodes"], n_queries_per_seed=accuracy["n_queries_per_seed"],
                training_seeds=seeds,
                errors_per_seed={s: int(errors[s].sum()) for s in seeds}, categories=categories)
