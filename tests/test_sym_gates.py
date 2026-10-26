"""Final-v1 gate boundaries; synthetic summaries contain no experiment data."""
from __future__ import annotations

import copy

import pytest

from looplora.sym.gates import ENDPOINTS, READERS, WRITERS, evaluate_gate_table


def _table():
    endpoints = {}
    for writer in WRITERS:
        for reader in READERS:
            row = {}
            for name in ENDPOINTS:
                specificity = "specificity" in name
                stat = {"confidence": .95, "difference": .04, "ci_low": .02, "ci_high": .08,
                        "training_seeds": [17, 29, 43], "n_seeds": 3,
                        "per_seed": [{"seed": s, "difference": .04} for s in (17, 29, 43)],
                        "n_episodes": 2000, "resamples": 2000, "split": "test", "condition": "original",
                        "depth_a": 4, "depth_b": 4 if name.startswith("vs_") else 1,
                        "bootstrap_seed": 123, "seed": 2718, "cluster_unit": "episode",
                        "joint_episode_resampling": True,
                        "statistic": "mean_per_seed_episode_walk_specificity" if specificity
                        else "mean_per_seed_paired_accuracy_difference"}
                if specificity:
                    stat.update(reach_hops=[2, 3], far_hops=[6, 7, 8], n_queries_per_seed=10000)
                else:
                    stat["hops"] = [6, 7, 8] if name in ("far", "replication_far") else [1] if name == "hop1" else [2, 3]
                    stat["n_queries_per_seed"] = 2000 * len(stat["hops"])
                gate = dict(copy.deepcopy(stat), confidence=.975, ci_low=.01, ci_high=.09)
                row[name] = {"report": stat, "gate": gate}
            endpoints[writer, reader] = row
    return endpoints


def _set_point(endpoints, name, value):
    for stats in endpoints.values():
        for role in ("report", "gate"):
            stats[name][role]["difference"] = value
            for seed in stats[name][role]["per_seed"]:
                seed["difference"] = value


def test_exactly_eight_rows_with_correct_comparators_and_all_criteria():
    source = _table()
    before = copy.deepcopy(source)
    rows = evaluate_gate_table(source, evaluation_seed=777001)
    assert len(rows) == 8
    assert {(r["writer"], r["reader"]) for r in rows} == set(source)
    assert all(row["passed"] and row["g1_passed"] and row["g2_passed"] for row in rows)
    assert all(len(row["criteria"]) == 9 for row in rows)
    assert all(row["fixed_query_comparator"] == ("B5" if row["reader"] == "M1" else "B5-lin") for row in rows)
    assert source == before


def test_point_threshold_equalities_pass_and_far_components_do_not_gate():
    table = _table()
    _set_point(table, "reach", .03)
    _set_point(table, "hop1", -.01)
    for endpoint in ("far", "replication_far"):
        _set_point(table, endpoint, -.04)
        for stats in table.values():
            stats[endpoint]["report"].update(ci_low=-.08, ci_high=-.02)
            stats[endpoint]["gate"].update(ci_low=-.09, ci_high=-.01)
    assert all(row["passed"] for row in evaluate_gate_table(table, evaluation_seed=777001))


@pytest.mark.parametrize("name,value,criterion", [
    ("reach", .029999, "reach_at_least_3pp"),
    ("hop1", -.010001, "hop1_degradation_at_most_1pp"),
])
def test_point_threshold_failures(name, value, criterion):
    table = _table()
    _set_point(table, name, value)
    for row in evaluate_gate_table(table, evaluation_seed=777001):
        assert not row["criteria"][criterion]["passed"]
        assert not row["g1_passed"] and not row["passed"]
        assert row["g2_passed"]


@pytest.mark.parametrize("name", ["reach", "specificity", "replication_reach", "replication_specificity",
                                  "vs_fixed", "vs_loop"])
def test_zero_975_ci_lower_bound_fails_even_with_positive_95_bound(name):
    table = _table()
    for stats in table.values():
        stats[name]["gate"]["ci_low"] = 0
    rows = evaluate_gate_table(table, evaluation_seed=777001)
    assert not any(row["passed"] for row in rows)
    assert all(row["g1_passed"] == name.startswith("vs_") for row in rows)
    assert all(row["g2_passed"] != name.startswith("vs_") for row in rows)


@pytest.mark.parametrize("seed_effect", [0, -.01])
def test_nonpositive_primary_seed_effect_fails_but_replication_has_no_each_seed_or_3pp_rule(seed_effect):
    table = _table()
    for stats in table.values():
        for role in ("report", "gate"):
            stats["reach"][role]["per_seed"] = [
                {"seed": 17, "difference": seed_effect},
                {"seed": 29, "difference": (.12 - seed_effect) / 2},
                {"seed": 43, "difference": (.12 - seed_effect) / 2}]
    assert all(not row["criteria"]["reach_each_seed_positive"]["passed"]
               for row in evaluate_gate_table(table, evaluation_seed=777001))
    table = _table()
    _set_point(table, "replication_reach", .01)
    for stats in table.values():
        for role in ("report", "gate"):
            stats["replication_reach"][role]["per_seed"] = [
                {"seed": 17, "difference": -.01}, {"seed": 29, "difference": .02}, {"seed": 43, "difference": .02}]
    assert all(row["passed"] for row in evaluate_gate_table(table, evaluation_seed=777001))


@pytest.mark.parametrize("field,value", [("confidence", .95), ("training_seeds", [17]), ("resamples", 1999),
                                       ("n_episodes", 1999), ("split", "val"), ("condition", "swap_memory"),
                                       ("depth_b", 4), ("hops", [1, 2, 3]), ("ci_low", float("nan")),
                                       ("difference", float("inf")), ("joint_episode_resampling", False),
                                       ("n_queries_per_seed", 3999)])
def test_invalid_gate_metadata_is_rejected(field, value):
    table = _table()
    table["joint", "M1"]["reach"]["gate"][field] = value
    with pytest.raises(ValueError):
        evaluate_gate_table(table, evaluation_seed=777001)


def test_mismatched_dualconfidence_statistics_are_rejected():
    table = _table()
    table["joint", "M1"]["reach"]["gate"]["bootstrap_seed"] = 999
    with pytest.raises(ValueError, match="confidence levels disagree"):
        evaluate_gate_table(table, evaluation_seed=777001)


def test_wrong_confirmation_seed_missing_rows_or_endpoints_are_rejected():
    with pytest.raises(ValueError, match="confirmation"):
        evaluate_gate_table(_table(), evaluation_seed=4242)
    table = _table()
    table.pop(("joint", "M1"))
    with pytest.raises(ValueError, match="exactly eight"):
        evaluate_gate_table(table, evaluation_seed=777001)
    table = _table()
    table["joint", "M1"].pop("specificity")
    with pytest.raises(ValueError, match="nine named"):
        evaluate_gate_table(table, evaluation_seed=777001)


def test_a_failed_row_does_not_change_other_writer_reader_decisions():
    table = _table()
    table["joint", "M1-lin"]["specificity"]["gate"]["ci_low"] = 0
    rows = evaluate_gate_table(table, evaluation_seed=777001)
    assert {(row["writer"], row["reader"]) for row in rows if not row["passed"]} == {("joint", "M1-lin")}
