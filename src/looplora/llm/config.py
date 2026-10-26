"""Resolved configuration and data-boundary guards for the P1-L pipeline."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from looplora.infra.seeds import derive_seed

VARIANTS = ("M1", "M1-lin", "B5", "B3", "B4", "B8", "B1", "B2", "B0")
STAGES = ("writer", "reader", "joint", "lora_ctx", "baseline")
PROTOCOLS = ("memory_only", "context_retained")
DEFAULTS: dict[str, Any] = {
    "name": "llm", "seed": 17, "device": "cuda", "stage": "joint", "variant": "M1",
    "protocol": "memory_only", "write_policy": "fact_final", "activation": "relu",
    "smoke": False, "writer_checkpoint": None,
    "model": {"r": 32, "layer": 14, "rho": .25, "alpha": 1., "input_site": "o_proj"},
    "task": {"name": "t2_chain_end", "E": 24, "Q": 4, "V": 1024, "n_rel": 4,
             "chain_lengths": [8, 8, 8], "hops": [1, 2, 3]},
    "train": {"steps": 1000, "batch_size": 8, "lr": .0003, "weight_decay": .01,
              "grad_clip": 1., "warmup_steps": 20, "depths": [1, 2, 4], "log_every": 10,
              "prefetch_workers": 0},
    "eval": {"episodes": 2000, "batch_size": 16, "depths": [0, 1, 2, 3, 4, 6, 8],
             "splits": ["val"], "hops": list(range(1, 9)), "seed": 4242,
             "counterfactuals": True, "protocols": ["memory_only"],
             "allow_confirmation": False},
    "stats": {"resamples": 2000, "seed": 2718},
}


def recursive_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in updates.items():
        result[key] = (recursive_merge(result[key], value)
                       if isinstance(value, Mapping) and isinstance(result.get(key), Mapping)
                       else copy.deepcopy(value))
    return result


def read_mapping(path: str | Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _integer(value: Any, label: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _distinct(value: Any, label: str, minimum: int) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    for item in value:
        _integer(item, label, minimum)
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must contain distinct values")


def resolve_config(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve stage contracts; smoke never permits screening or confirmation data."""
    if not isinstance(mapping, Mapping):
        raise ValueError("configuration must be a mapping")
    unknown = set(mapping) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    for group in ("model", "task", "train", "eval", "stats"):
        if group in mapping and not isinstance(mapping[group], Mapping):
            raise ValueError(f"{group} must be a mapping")
        if group != "task" and group in mapping:
            unknown = set(mapping[group]) - set(DEFAULTS[group])
            if unknown:
                raise ValueError(f"unknown {group} fields: {sorted(unknown)}")
    config = recursive_merge(DEFAULTS, mapping)
    stage, variant = config["stage"], config["variant"]
    if stage not in STAGES or variant not in VARIANTS:
        raise ValueError("unknown stage or variant")
    if ((stage == "writer" and variant not in {"M1", "B8"})
            or ((stage == "lora_ctx") != (variant in ("B1", "B2")))
            or ((stage == "baseline") != (variant == "B0"))):
        raise ValueError("writer requires M1/B8, lora_ctx requires B1/B2, baseline requires B0")
    if config["protocol"] not in PROTOCOLS:
        raise ValueError("unknown protocol")
    expected = "context_retained" if stage == "lora_ctx" else "memory_only"
    if stage != "baseline" and config["protocol"] != expected:
        raise ValueError(f"{stage} trains with protocol={expected}")
    if config["write_policy"] not in ("fact_final", "all_tokens"):
        raise ValueError("unknown write_policy")
    if config["activation"] not in ("relu", "identity"):
        raise ValueError("activation must be relu or identity")
    if variant == "M1-lin":
        config["activation"] = "identity"
    for key in ("smoke",):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("name", "device"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty string")
    _integer(config["seed"], "seed", 0)
    _integer(config["model"]["r"], "model.r")
    _integer(config["model"]["layer"], "model.layer", 0)
    if config["model"]["input_site"] not in {"o_proj", "residual"}:
        raise ValueError("model.input_site must be o_proj or residual")
    for group, keys in (("model", ("rho", "alpha")),
                        ("train", ("lr", "weight_decay", "grad_clip"))):
        for key in keys:
            value = config[group][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{group}.{key} must be finite")
    if config["train"]["lr"] <= 0 or min(config["train"]["weight_decay"], config["train"]["grad_clip"]) < 0:
        raise ValueError("invalid optimizer settings")
    for key in ("steps", "batch_size", "log_every"):
        _integer(config["train"][key], f"train.{key}")
    _integer(config["train"]["warmup_steps"], "train.warmup_steps", 0)
    _integer(config["train"]["prefetch_workers"], "train.prefetch_workers", 0)
    _distinct(config["train"]["depths"], "train.depths", 1)
    if stage in ("writer", "lora_ctx") or variant == "B4":
        config["train"]["depths"] = [1]
    if stage in ("writer", "lora_ctx", "baseline") or variant == "B4":
        config["eval"]["depths"] = [1]
    if variant == "B8":
        config["train"]["depths"] = config["eval"]["depths"] = [4]
    checkpoint = config["writer_checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ValueError("writer_checkpoint must be a path string or null")
    if stage == "reader" and checkpoint is None:
        raise ValueError("reader requires writer_checkpoint")
    if stage != "reader" and checkpoint is not None:
        raise ValueError("only the reader stage consumes writer_checkpoint")
    evaluation = config["eval"]
    for key in ("episodes", "batch_size"):
        _integer(evaluation[key], f"eval.{key}")
    # All ordinary sets are [0, episodes); confirm alone maps to [100000,100000+episodes).
    if evaluation["episodes"] > 2000:
        raise ValueError("eval.episodes exceeds the preregistered 2000-episode set")
    _integer(evaluation["seed"], "eval.seed", 0)
    _distinct(evaluation["depths"], "eval.depths", 0)
    _distinct(evaluation["hops"], "eval.hops", 1)
    if max(evaluation["hops"]) > 8:
        raise ValueError("P1-L evaluation only defines hops 1..8")
    for key in ("counterfactuals", "allow_confirmation"):
        if not isinstance(evaluation[key], bool):
            raise ValueError(f"eval.{key} must be boolean")
    splits = evaluation["splits"]
    if (not isinstance(splits, list) or not splits or any(not isinstance(s, str) for s in splits)
            or len(set(splits)) != len(splits)
            or set(splits) - {"val", "test", "confirm", "test_unseen"}):
        raise ValueError("invalid eval.splits")
    if "confirm" in splits and not evaluation["allow_confirmation"]:
        raise ValueError("confirmation requires eval.allow_confirmation=true")
    if config["smoke"] and (splits != ["val"] or evaluation["allow_confirmation"]):
        raise ValueError("smoke is validation-only and cannot authorize confirmation")
    if config["smoke"] and "seed" not in mapping.get("eval", {}):
        evaluation["seed"] = derive_seed(20261027, "p1l-smoke-eval")
    if "protocols" not in mapping.get("eval", {}):
        if variant in ("B0", "B1", "B2") or (variant in {"M1", "B8"} and stage in ("reader", "joint")):
            evaluation["protocols"] = list(PROTOCOLS)
    protocols = evaluation["protocols"]
    if (not isinstance(protocols, list) or not protocols
            or any(p not in PROTOCOLS for p in protocols) or len(set(protocols)) != len(protocols)):
        raise ValueError("invalid eval.protocols")
    if "context_retained" in protocols and variant not in ("M1", "B8", "B0", "B1", "B2"):
        raise ValueError("secondary context-retained evaluation is defined for M1/B8/B0/B1/B2")
    _integer(config["stats"]["resamples"], "stats.resamples", 2000)
    _integer(config["stats"]["seed"], "stats.seed", 0)
    from looplora.data.relations import TaskSpec

    task = dict(config["task"])
    if stage == "writer":
        task.update(name="t1", n_rel=4, head_sharing=.3, hops=[1], chain_lengths=None)
        evaluation["hops"] = [1]
    elif task["name"] == "t1" or task.get("base") == "t1":
        if "chain_lengths" not in mapping.get("task", {}):
            task["chain_lengths"] = None
        task["hops"] = [1]
        evaluation["hops"] = [1]
    elif max(task["hops"]) > 3:
        raise ValueError("P1-L training is restricted to hops 1..3")
    config["task"] = asdict(TaskSpec(**task))
    return json.loads(json.dumps(config, allow_nan=False))


def load_config(path: str | Path) -> dict[str, Any]:
    return resolve_config(read_mapping(path))
