"""Pure P1-S final-v1 gate decisions from audited, precomputed statistics.

``evaluate_gate_table(endpoints, evaluation_seed=777001)`` takes exactly eight
``(writer, reader)`` keys. Each value contains reach, far, specificity, hop1,
replication_reach, replication_far, replication_specificity, vs_fixed, vs_loop.
Each endpoint is ``{'report': stats95, 'gate': stats975}`` from sym.stats.
The caller must audit the manifest's evaluation seed and wait for confirmation
release before loading artifacts. This module performs no I/O or selection.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping

WRITERS = ("oracle_random", "oracle_orth", "t1_frozen", "joint")
READERS = ("M1", "M1-lin")
ENDPOINTS = ("reach", "far", "specificity", "hop1", "replication_reach", "replication_far",
             "replication_specificity", "vs_fixed", "vs_loop")
_SEEDS = [17, 29, 43]


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite real data")
    return float(value)


def _validate_endpoint(name: str, pair: Mapping[str, Any]) -> None:
    if not isinstance(pair, Mapping) or set(pair) != {"report", "gate"}:
        raise ValueError(f"{name} requires report and gate statistics")
    expected_hops = [6, 7, 8] if name in {"far", "replication_far"} else [1] if name == "hop1" else [2, 3]
    specificity = name in {"specificity", "replication_specificity"}
    for role, confidence in (("report", .95), ("gate", .975)):
        stat = pair[role]
        if not isinstance(stat, Mapping):
            raise ValueError(f"{name}/{role} must be a statistics mapping")
        required = {"confidence", "difference", "ci_low", "ci_high", "training_seeds", "per_seed",
                    "n_seeds", "n_episodes", "n_queries_per_seed", "resamples", "split", "condition",
                    "depth_a", "depth_b", "bootstrap_seed", "seed", "cluster_unit",
                    "joint_episode_resampling", "statistic"}
        if not required <= stat.keys():
            raise ValueError(f"{name}/{role} has incomplete pooled-statistics metadata")
        if (_number(stat["confidence"], "confidence") != confidence
                or stat["training_seeds"] != _SEEDS or stat["n_seeds"] != 3
                or stat["n_episodes"] != 2000 or stat["resamples"] != 2000
                or stat["split"] != "test" or stat["condition"] != "original"
                or stat["depth_a"] != 4 or stat["depth_b"] != (4 if name.startswith("vs_") else 1)
                or stat["cluster_unit"] != "episode" or stat["joint_episode_resampling"] is not True):
            raise ValueError(f"{name}/{role} does not match preregistered gate metadata")
        expected_statistic = ("mean_per_seed_episode_walk_specificity" if specificity
                              else "mean_per_seed_paired_accuracy_difference")
        if stat["statistic"] != expected_statistic:
            raise ValueError(f"{name}/{role} has the wrong statistic")
        if specificity:
            if stat.get("reach_hops") != [2, 3] or stat.get("far_hops") != [6, 7, 8]:
                raise ValueError(f"{name}/{role} has incorrect specificity hop subsets")
        elif stat.get("hops") != expected_hops:
            raise ValueError(f"{name}/{role} has incorrect hop subset")
        expected_queries = 10000 if specificity else 2000 * len(expected_hops)
        if stat["n_queries_per_seed"] != expected_queries:
            raise ValueError(f"{name}/{role} must retain one query per requested hop per episode")
        difference = _number(stat["difference"], "difference")
        low, high = _number(stat["ci_low"], "ci_low"), _number(stat["ci_high"], "ci_high")
        if low > high:
            raise ValueError(f"{name}/{role} has reversed CI bounds")
        rows = stat["per_seed"]
        if not isinstance(rows, list) or len(rows) != 3 or sorted(row["seed"] for row in rows) != _SEEDS:
            raise ValueError(f"{name}/{role} requires each preregistered seed exactly once")
        values = [_number(row["difference"], "per-seed difference") for row in rows]
        if not math.isclose(sum(values) / 3, difference, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{name}/{role} point estimate does not equal the mean over seeds")
    report, gate = pair["report"], pair["gate"]
    for field in ("difference", "per_seed", "bootstrap_seed", "seed", "n_queries_per_seed"):
        if report[field] != gate[field]:
            raise ValueError(f"{name} confidence levels disagree on {field}")
    if gate["ci_low"] > report["ci_low"] or gate["ci_high"] < report["ci_high"]:
        raise ValueError(f"{name} higher-confidence interval must contain its report interval")


def evaluate_gate_table(
    endpoints: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    *,
    evaluation_seed: int,
) -> list[dict[str, Any]]:
    """Return exactly eight gate rows using raw values and final-v1 thresholds.

    The evaluation seed is an explicit assertion from the caller's audited
    manifest, since sample arrays do not encode it. Reach must be >= .03;
    hop1 must be >= -.01; CI lower bounds and each primary reach seed effect
    must be strictly > 0. Replication does not inherit the .03 or each-seed
    positivity requirement. Far effects are reported, without separate gates.
    """
    if evaluation_seed != 777001:
        raise ValueError("Gate decisions require the audited confirmation evaluation seed 777001")
    expected = {(writer, reader) for writer in WRITERS for reader in READERS}
    if not isinstance(endpoints, Mapping) or set(endpoints) != expected:
        raise ValueError("Gate table requires exactly eight preregistered writer/reader rows")
    output = []
    for writer in WRITERS:
        for reader in READERS:
            stats = endpoints[writer, reader]
            if not isinstance(stats, Mapping) or set(stats) != set(ENDPOINTS):
                raise ValueError(f"{writer}/{reader} requires all nine named endpoints")
            for name in ENDPOINTS:
                _validate_endpoint(name, stats[name])
            criteria = {}

            def criterion(name: str, value: float, threshold: float, *, inclusive: bool = False) -> None:
                criteria[name] = {"value": value, "threshold": threshold,
                                  "operator": ">=" if inclusive else ">",
                                  "passed": bool(value >= threshold if inclusive else value > threshold)}

            criterion("reach_at_least_3pp", stats["reach"]["gate"]["difference"], .03, inclusive=True)
            criterion("reach_ci_positive", stats["reach"]["gate"]["ci_low"], 0)
            seed_values = [row["difference"] for row in stats["reach"]["gate"]["per_seed"]]
            criterion("reach_each_seed_positive", min(seed_values), 0)
            criteria["reach_each_seed_positive"]["per_seed"] = stats["reach"]["gate"]["per_seed"]
            criterion("specificity_ci_positive", stats["specificity"]["gate"]["ci_low"], 0)
            criterion("hop1_degradation_at_most_1pp", stats["hop1"]["gate"]["difference"], -.01, inclusive=True)
            criterion("replication_reach_ci_positive", stats["replication_reach"]["gate"]["ci_low"], 0)
            criterion("replication_specificity_ci_positive", stats["replication_specificity"]["gate"]["ci_low"], 0)
            g1 = all(row["passed"] for row in criteria.values())
            criterion("vs_fixed_ci_positive", stats["vs_fixed"]["gate"]["ci_low"], 0)
            criterion("vs_loop_ci_positive", stats["vs_loop"]["gate"]["ci_low"], 0)
            g2 = criteria["vs_fixed_ci_positive"]["passed"] and criteria["vs_loop_ci_positive"]["passed"]
            output.append({"writer": writer, "reader": reader,
                           "fixed_query_comparator": "B5" if reader == "M1" else "B5-lin",
                           "evaluation_seed": evaluation_seed, "gate_confidence": .975,
                           "report_confidence": .95, "criteria": criteria,
                           "g1_passed": g1, "g2_passed": g2, "passed": g1 and g2,
                           "statistics": dict(stats)})
    return output
