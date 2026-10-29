"""B8W (docs/prereg/B8W.md): copy a trained M1-L4 reader into B8, check the copy is the same function, decide.

Pure helpers: no file or model is touched here except through the experiment objects handed in.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import torch

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed

from .config import ORACLE_WRITERS, resolve_config
from .train import LLMExperiment, make_task_spec

STEPS = 4                                   # B8 has four unshared steps; the deployed depth of both arms
COPIED = ("A", "B", "K", "V", "a_lambda", "a_beta")   # equal in both readers (K, V, gates are frozen under the oracle)
SHARED = ("R", "U", "Q")                    # one shared tensor in M1-L4, one per step in B8
NONINFERIORITY = 0.02                       # registered margin: at most 2 pp loss (docs/prereg/B8W.md, decision rule)
# Accuracies are ratios of counts over 4000 queries; the bar is on them directly. This only absorbs float rounding.
EPS = 1e-12
GATE_EPISODES, GATE_BATCH, GATE_EVAL_SEED = 64, 16, 4242


def convert_m1_to_b8(record: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """The format-1 checkpoint of a B8 whose every step uses the M1-L4 reader's R, U and Q.

    The input is not modified. Copied: A, B, K, V and the gates; R, U, Q are cloned into R.i, U.i, Q.i (i < 4), the
    parameter names of `UnsharedAdapter`. The config is the M1-L4 config with variant B8; the writer record is kept
    (the oracle writer does not depend on the reader).
    """
    if record.get("format_version") != 1:
        raise ValueError("unsupported LLM checkpoint format")
    config = record["config"]
    if config["variant"] != "M1" or config["stage"] != "reader" or config.get("writer") not in ORACLE_WRITERS:
        raise ValueError("conversion starts from an oracle-writer M1 reader checkpoint")
    state = record["adapter"]
    if set(state) != set(COPIED) | set(SHARED):
        raise ValueError(f"unexpected M1 adapter keys: {sorted(set(state) ^ (set(COPIED) | set(SHARED)))}")
    r = r_mem = config["model"]["r"]    # AdapterConfig: r_mem defaults to r (all P1-L oracle readers)
    for key, shape in (("R", (r, r)), ("U", (r, r_mem)), ("Q", (r_mem, r))):
        if tuple(state[key].shape) != shape:
            raise ValueError(f"{key} has shape {tuple(state[key].shape)}, expected {shape}")
    adapter = {key: state[key].detach().clone() for key in COPIED}
    for step in range(STEPS):
        for key in SHARED:
            adapter[f"{key}.{step}"] = state[key].detach().clone()
    new_config = copy.deepcopy(config)
    new_config.update(variant="B8", name=name)
    new_config = resolve_config(new_config)
    if new_config["train"]["depths"] != [STEPS] or new_config["eval"]["depths"] != [STEPS]:
        raise ValueError("a B8 config must train and evaluate at depth 4")
    summary = copy.deepcopy(record.get("summary", {}))
    summary["converted_from"] = dict(variant="M1", name=config["name"])
    return {**{key: copy.deepcopy(value) for key, value in record.items() if key not in {"config", "adapter", "summary"}},
            "format_version": 1, "config": new_config, "adapter": adapter, "summary": summary}


@torch.no_grad()
def gate_signature(experiment: LLMExperiment) -> dict[str, Any]:
    """Answer-position logits (CPU, in the dtype `score` returns) on the validation batch of the equality gate, and the
    fp32 reader output of the adapter on a fixed random input and memory state. Validation split only; no test episode."""
    config = experiment.config
    spec = make_task_spec(config, evaluation=True)
    logits = []
    for start in range(0, GATE_EPISODES, GATE_BATCH):
        arrays = generate(spec, "val", start, GATE_BATCH, GATE_EVAL_SEED)
        prepared = experiment.prepare(arrays)
        scores, _ = experiment.score(prepared, STEPS)
        logits.append(scores.detach().cpu())
    adapter, cfg = experiment.adapter, experiment.adapter.cfg
    generator = torch.Generator().manual_seed(derive_seed(17, "b8w", "gate-read"))
    u = torch.randn(32, cfg.r, generator=generator).to(experiment.device)
    memory = torch.randn(32, cfg.r_mem, cfg.key_dim, generator=generator).to(experiment.device)
    return {"logits": logits, "read": adapter.read(u, memory, STEPS).detach().cpu()}


def compare_signatures(reference: Mapping[str, Any], control: Mapping[str, Any], converted: Mapping[str, Any]) -> dict[str, Any]:
    """Gate result: tolerance 0. Passed iff the reference is bitwise reproducible (control), and the converted reader's
    logits, predicted tokens and adapter output equal the reference's exactly."""
    def differences(left, right):
        pairs = [*zip(left["logits"], right["logits"]), (left["read"], right["read"])]
        return (all(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b) for a, b in pairs),
                max(float((a.float() - b.float()).abs().max()) if a.shape == b.shape else float("inf") for a, b in pairs))
    control_equal, control_max = differences(reference, control)
    equal, maximum = differences(reference, converted)
    argmax_equal = all(torch.equal(a.argmax(-1), b.argmax(-1))
                       for a, b in zip(reference["logits"], converted["logits"]))
    return dict(passed=bool(control_equal and equal and argmax_equal and maximum == 0.0),
                logits_equal=bool(equal), argmax_equal=bool(argmax_equal), control_equal=bool(control_equal),
                max_abs_difference=maximum, control_max_abs_difference=control_max,
                logits_dtype=str(reference["logits"][0].dtype), read_dtype=str(reference["read"].dtype),
                tolerance=0.0, episodes=GATE_EPISODES, split="val", depth=STEPS)


def keeps_solution(contrast: Mapping[str, Any]) -> dict[str, Any]:
    """Registered B8W decision on Delta = untied@4 - shared@4 (hops 2, 3; a `pooled_compare_runs` row at 95%).

    Untying keeps the solution iff the loss is at most 2 pp in every seed (Delta_seed >= -0.02) and the pooled CI lower
    bound is above -0.02. Otherwise untying loses the solution (fragility of the solution under untied updates, not
    lack of expressivity).
    """
    per_seed = {int(item["seed"]): float(item["difference"]) for item in contrast["per_seed"]}
    seeds_ok = bool(per_seed) and all(value >= -NONINFERIORITY - EPS for value in per_seed.values())
    interval_ok = float(contrast["ci_low"]) > -NONINFERIORITY + EPS
    return dict(keeps=bool(seeds_ok and interval_ok), every_seed_within_2pp=seeds_ok, interval_excludes_worse_than_2pp=interval_ok,
                difference=float(contrast["difference"]), ci_low=float(contrast["ci_low"]), ci_high=float(contrast["ci_high"]),
                per_seed=per_seed)


def decision_text(result: Mapping[str, Any]) -> str:
    if result["keeps"]:
        return ("untying keeps the solution: the copied B8 stays within 2 pp of continued M1-L4 on hops 2 and 3 in every seed "
                "and the pooled interval excludes a loss worse than 2 pp (no equal-parameter status; G1-L stays failed; "
                "nothing about why B8 failed from a random start)")
    return ("untying loses the solution: fragility of the solution under untied updates, not lack of expressivity "
            "(no equal-parameter status; G1-L stays failed; nothing about why B8 failed from a random start)")
