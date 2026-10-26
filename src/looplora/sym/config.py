"""Resolved, JSON-compatible configuration for the symbolic mechanism study."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

VARIANTS = ("B3", "B4", "B5", "M1", "M1-lin", "B6", "B7", "B8")
WRITERS = ("oracle_random", "oracle_orth", "t1_frozen", "joint")
DEFAULTS: dict[str, Any] = {
    "name": "symbolic", "seed": 17, "backbone_seed": 31415, "device": "cuda",
    "writer": "oracle_random", "variant": "M1", "activation": "relu",
    "model": {"d_e": 64, "r": 64, "r_mem": 64, "rho": .25, "alpha": 1.,
              "lambda_init": .99, "beta_init": .1, "max_hop": 16},
    "task": {"name": "t2_chain_end", "E": 24, "Q": 4, "hops": [1, 2, 3]},
    "train": {"steps": 3000, "batch_size": 32, "lr": .003, "weight_decay": .01,
              "grad_clip": 1., "warmup_steps": 20, "depths": [1, 2, 4], "log_every": 10},
    "pretrain": {"steps": 3000, "batch_size": 32, "lr": .003,
                 "warmup_steps": 20, "writer_checkpoint": None},
    "eval": {"episodes": 64, "batch_size": 32, "depths": [0, 1, 2, 3, 4, 6, 8, 12, 16],
             "splits": ["val", "test", "test_unseen"], "hops": list(range(1, 9)),
             "seed": 4242, "counterfactuals": True},
    "stats": {"resamples": 2000, "seed": 2718},
}


def recursive_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested mappings without mutating either input; lists replace lists."""
    result = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = recursive_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_mapping(path: str | Path) -> dict[str, Any]:
    """Read YAML or JSON without importing the training runtime."""
    import yaml

    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _positive_int(value: Any, label: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _depths(value: Any, label: str, minimum: int) -> None:
    if not isinstance(value, list) or not value or len(set(value)) != len(value):
        raise ValueError(f"{label} must be a nonempty list of distinct integers")
    for depth in value:
        _positive_int(depth, label, minimum)


def resolve_config(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Apply defaults, enforce variant constraints, and resolve the data spec."""
    if not isinstance(mapping, Mapping):
        raise ValueError("configuration must be a mapping")
    unknown = set(mapping) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    for group in ("model", "task", "train", "pretrain", "eval", "stats"):
        if group in mapping and not isinstance(mapping[group], Mapping):
            raise ValueError(f"{group} must be a mapping")
        if group != "task" and group in mapping:
            unknown = set(mapping[group]) - set(DEFAULTS[group])
            if unknown:
                raise ValueError(f"unknown {group} fields: {sorted(unknown)}")
    config = recursive_merge(DEFAULTS, mapping)
    variant = config["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if config["writer"] not in WRITERS:
        raise ValueError(f"unknown writer: {config['writer']}")
    if config["activation"] not in ("relu", "identity"):
        raise ValueError("activation must be relu or identity")
    config["activation"] = "identity" if variant == "M1-lin" else config["activation"]
    model = config["model"]
    explicit_model = mapping.get("model", {})
    if variant == "B6" and "r" not in explicit_model:
        model["r"] = 96
        if "r_mem" not in explicit_model:
            model["r_mem"] = 96
    if variant == "B7" and "r_mem" not in explicit_model:
        model["r_mem"] = 96
    for key in ("d_e", "r", "r_mem", "max_hop"):
        _positive_int(model[key], f"model.{key}")
    if variant == "B6" and model["r"] <= 64:
        raise ValueError("B6 requires a larger rank than the default 64")
    if variant == "B7" and model["r_mem"] <= model["r"]:
        raise ValueError("B7 requires r_mem > r")
    if config["writer"].startswith("oracle_") and model["r"] != model["r_mem"]:
        raise ValueError("oracle writers require r == r_mem")
    for key in ("rho", "alpha", "lambda_init", "beta_init"):
        value = model[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"model.{key} must be finite")
    for key in ("lambda_init", "beta_init"):
        if not 0 < model[key] < 1:
            raise ValueError(f"model.{key} must be strictly between zero and one")
    for label in ("seed", "backbone_seed"):
        _positive_int(config[label], label, 0)
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("name must be a nonempty string")
    if not isinstance(config["device"], str) or not config["device"]:
        raise ValueError("device must be a nonempty string")
    for group in ("train", "pretrain"):
        for key in ("steps", "batch_size"):
            _positive_int(config[group][key], f"{group}.{key}")
        _positive_int(config[group]["warmup_steps"], f"{group}.warmup_steps", 0)
        if not isinstance(config[group]["lr"], (int, float)) or not math.isfinite(config[group]["lr"]) or config[group]["lr"] <= 0:
            raise ValueError(f"{group}.lr must be finite and positive")
    for key in ("weight_decay", "grad_clip"):
        value = config["train"][key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"train.{key} must be finite and nonnegative")
    _positive_int(config["train"]["log_every"], "train.log_every")
    _depths(config["train"]["depths"], "train.depths", 1)
    _depths(config["eval"]["depths"], "eval.depths", 0)
    _depths(config["eval"]["hops"], "eval.hops", 1)
    for key in ("episodes", "batch_size"):
        _positive_int(config["eval"][key], f"eval.{key}")
    _positive_int(config["eval"]["seed"], "eval.seed", 0)
    if not isinstance(config["eval"]["counterfactuals"], bool):
        raise ValueError("eval.counterfactuals must be boolean")
    splits = config["eval"]["splits"]
    if not isinstance(splits, list) or not splits or len(set(splits)) != len(splits) or set(splits) - {"val", "test", "test_unseen"}:
        raise ValueError("eval.splits must contain distinct evaluation splits")
    _positive_int(config["stats"]["resamples"], "stats.resamples", 2000)
    _positive_int(config["stats"]["seed"], "stats.seed", 0)
    if variant in ("B4", "B6", "B7"):
        config["train"]["depths"] = config["eval"]["depths"] = [1]
    elif variant == "B8":
        config["train"]["depths"] = config["eval"]["depths"] = [4]
    from looplora.data.relations import TaskSpec

    # A task switch must not inherit chain-specific default fields from T2.
    task = dict(mapping.get("task", {}))
    task.setdefault("name", DEFAULTS["task"]["name"])
    if task["name"] == "t1" or task.get("base") == "t1":
        task.setdefault("hops", [1])
        config["eval"]["hops"] = [1]
    else:
        task.setdefault("hops", DEFAULTS["task"]["hops"])
    spec = TaskSpec(**task)
    config["task"] = asdict(spec)
    if max(config["eval"]["hops"]) > model["max_hop"] or max(spec.hops) > model["max_hop"]:
        raise ValueError("requested hop exceeds model.max_hop")
    checkpoint = config["pretrain"]["writer_checkpoint"]
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("pretrain.writer_checkpoint must be a path string or null")
    return json.loads(json.dumps(config, allow_nan=False))


def load_config(path: str | Path) -> dict[str, Any]:
    return resolve_config(read_mapping(path))
