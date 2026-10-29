"""LSR test statistics on synthetic traces (docs/prereg/LSR.md; the registered endpoints and the per-seed conditions)."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from looplora.llm import lsr_rules as rules
from looplora.llm import lsr_stats as stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_lsr  # noqa: E402

E = 200


def make_trace(hop1, long, episodes=E):
    """Answers are 0, so p^s = 0 is correct. ``hop1`` and ``long`` map an episode index to its p^0..p^8 for hop 1 / hops 4-8."""
    pred = np.ones((episodes, 8, 9), dtype=np.int32)
    for e in range(episodes):
        pred[e, 0] = hop1(e)
        for h in range(3, 8):
            pred[e, h] = long(e)
        pred[e, 1] = pred[e, 2] = 0
    return dict(pred=pred, answer=np.zeros((episodes, 8), dtype=np.int32),
                change=np.full((episodes, 8, 8), 9., dtype=np.float32),
                read_ratio=np.full((episodes, 8, 8), 9., dtype=np.float32))


GOOD_HOP1 = lambda e: [0] * 9                                           # noqa: E731  right at every depth: R2 stops at 2, correct
# p^s alternates between two wrong answers, so R2 (three equal predictions) reaches s = 8; in the first 30% of episodes p^6..p^8 are right
LATE = lambda e: [1, 2, 1, 2, 1, 2, 0, 0, 0] if e < 60 else [1, 2, 1, 2, 1, 2, 1, 2, 1]   # noqa: E731
NEVER = lambda e: [1, 2, 1, 2, 1, 2, 1, 2, 1]                                              # noqa: E731


def entry(r2_feasible=True):
    infeasible = dict(grid=[], selected=None, feasible=False)
    return dict(rules=dict(R1=infeasible, R3=infeasible, R2=dict(grid=[], selected=None, feasible=r2_feasible)),
                fixed=dict(grid=[], selected=dict(L=4), feasible=True),
                oracle_best_fixed_L_by_hop=[2, 4, 5, 6, 8, 8, 8, 8])


def run(traces, **kwargs):
    return stats.evaluate_backbone(traces, entry(**kwargs), "0.6b", draws=400)


def test_a_rule_that_gains_on_long_hops_without_hurting_hop_1_passes_every_condition():
    traces = {s: make_trace(GOOD_HOP1, LATE) for s in rules.SEEDS}
    result = run(traces)["rules"]["R2"]
    assert result["status"] == "evaluated" and result["passes"] and all(result["conditions"].values())
    assert result["gain48"][0] == pytest.approx(30.0) and result["gain48"][1] > 0       # L = 4 is wrong on hops 4-8, R2 stops at 8
    assert result["hop1"][0] == 0.0 and result["hop1"][1] == result["hop1"][2] == 0.0
    assert all(v["gain48"][0] == pytest.approx(30.0) for v in result["per_seed"].values())
    assert result["mean_depth"] == pytest.approx((3 * 2 + 5 * 8) / 8)      # hops 1-3 stop at 2, hops 4-8 at 8
    assert result["upper_layer_passes"] == pytest.approx(result["mean_reads"] + 1)
    assert sum(result["stop_depth_counts_by_hop"][0]) == 3 * E and result["stop_depth_counts_by_hop"][0][1] == 3 * E


def test_one_seed_without_a_gain_fails_the_per_seed_condition_even_when_the_pool_passes():
    traces = {17: make_trace(GOOD_HOP1, LATE), 29: make_trace(GOOD_HOP1, LATE), 43: make_trace(GOOD_HOP1, NEVER)}
    result = run(traces)["rules"]["R2"]
    assert result["conditions"] == dict(gain=True, hop1=True, per_seed=False, feasible=True) and not result["passes"]
    assert result["gain48"][0] == pytest.approx(20.0)
    assert result["per_seed"]["43"]["gain48"][0] == 0.0 and result["per_seed"]["17"]["gain48"][0] == pytest.approx(30.0)


def test_a_hop_1_loss_beyond_10_pp_fails_even_with_a_gain():
    # L = 4 is right on hop 1, R2 stops at s = 2 on p^2 = 1 and is wrong
    bad_hop1 = lambda e: [1, 1, 1, 0, 0, 0, 0, 0, 0]     # noqa: E731
    traces = {s: make_trace(bad_hop1, LATE) for s in rules.SEEDS}
    result = run(traces)["rules"]["R2"]
    assert result["hop1"][0] == pytest.approx(-100.0) and not result["conditions"]["hop1"] and not result["passes"]
    assert result["conditions"]["gain"]


def test_a_gain_below_3_pp_fails_and_so_does_a_gain_whose_interval_reaches_zero():
    small = lambda e: [1, 2, 1, 2, 1, 2, 0, 0, 0] if e < 4 else [1, 2, 1, 2, 1, 2, 1, 2, 1]     # noqa: E731  +2 pp
    result = run({s: make_trace(GOOD_HOP1, small) for s in rules.SEEDS})["rules"]["R2"]
    assert result["gain48"][0] == pytest.approx(2.0) and not result["conditions"]["gain"] and not result["passes"]
    thin = lambda e: [1, 2, 1, 2, 1, 2, 0, 0, 0] if e % 25 == 0 else [1, 2, 1, 2, 1, 2, 1, 2, 1]       # noqa: E731  +4 pp, 8 of 200 episodes
    result = stats.evaluate_backbone({s: make_trace(GOOD_HOP1, thin) for s in rules.SEEDS}, entry(), "0.6b", draws=2000)["rules"]["R2"]
    assert result["gain48"][0] == pytest.approx(4.0) and result["gain48"][1] > 0 and result["conditions"]["gain"]


def test_infeasible_rules_fail_without_any_test_statistic_and_r2_is_always_evaluated():
    result = run({s: make_trace(GOOD_HOP1, LATE) for s in rules.SEEDS})
    for rule in ("R1", "R3"):
        assert result["rules"][rule] == dict(status="infeasible", passes=False, test_statistics=None)
    assert result["rules"]["R2"]["status"] == "evaluated" and result["l_c"] == 4
    poisoned = {s: make_trace(GOOD_HOP1, LATE) for s in rules.SEEDS}
    for trace in poisoned.values():
        trace["change"][:] = np.nan
        trace["read_ratio"][:] = np.nan
    assert run(poisoned)["rules"]["R2"]["passes"]                      # R1/R3 inputs are never read for infeasible rules


def test_a_feasible_r1_or_r3_uses_its_locked_tau_and_pays_one_more_read_for_r3():
    traces = {s: make_trace(GOOD_HOP1, LATE) for s in rules.SEEDS}
    for trace in traces.values():
        trace["change"][:] = 9.
        trace["change"][..., 2] = .01                                    # fires at s = 3 for tau >= .01
        trace["read_ratio"][:] = trace["change"]
    locked = entry()
    for rule in ("R1", "R3"):
        locked["rules"][rule] = dict(grid=[], feasible=True, selected=dict(k=-6, tau=2.0 ** -3))
    result = stats.evaluate_backbone(traces, locked, "0.6b", draws=200)["rules"]
    assert result["R1"]["tau"] == result["R3"]["tau"] == 0.125
    assert result["R1"]["mean_depth"] == result["R3"]["mean_depth"] == 3.0
    assert result["R1"]["mean_reads"] == 3.0 and result["R3"]["mean_reads"] == 4.0 and result["R1"]["upper_layer_passes"] is None


def test_oracles_and_the_depth_matrix_use_labels_and_dominate_fixed_depths():
    traces = {s: make_trace(GOOD_HOP1, LATE) for s in rules.SEEDS}
    result = run(traces)
    oracle, matrix = result["oracles"], result["fixed"]
    assert oracle["O3_any_L_in_1_8_correct"]["hops48"] >= oracle["O2_best_test_cell_by_hop"]["hops48"] >= \
        oracle["O1_best_fixed_L_by_hop_from_validation"]["hops48"] - 1e-12
    assert oracle["O2_best_test_cell_by_hop"]["hops48"] == pytest.approx(30.0 / 100)
    assert oracle["O1_best_fixed_L_by_hop_from_validation"]["em_by_hop"][3] == matrix["L6"]["em_by_hop"][3] == pytest.approx(.3)
    assert matrix["L4"]["em_by_hop"][3] == 0.0 and matrix["L8"]["em_by_hop"][0] == 1.0
    assert set(matrix["L4_em_ci95"]) == {f"hop{h}" for h in range(1, 9)}


def test_the_bootstrap_is_shared_deterministic_and_calibrated():
    counts = stats.resample_counts(2000, 1000, 7)
    assert counts.shape == (1000, 2000) and (counts.sum(1) == 2000).all()
    assert np.array_equal(counts, stats.resample_counts(2000, 1000, 7)) and not np.array_equal(counts, stats.resample_counts(2000, 1000, 8))
    generator = np.random.default_rng(1)
    a, b = generator.random(2000) < .5, generator.random(2000) < .4
    out = stats.bootstrap(counts, dict(a=a.astype(float), b=b.astype(float), d=a.astype(float) - b.astype(float),
                                       flat=np.ones(2000)), stats.ALPHA)
    assert out["flat"] == (100.0, 100.0, 100.0)
    assert out["d"][0] == pytest.approx(out["a"][0] - out["b"][0])               # one shared resample: the difference is linear
    half = (out["a"][2] - out["a"][1]) / 2                                       # 99.17% of a Bernoulli(.5) mean over 2000: ~ 2.9 pp
    assert 2.3 < half < 3.6 and out["a"][1] < out["a"][0] < out["a"][2]
    assert stats.ALPHA == pytest.approx(0.05 / 6) and stats.COMPARISONS == 6 and stats.DRAWS == 10_000
    assert (stats.GAIN_PP, stats.HOP1_PP) == (3.0, -10.0)


def test_results_serialise_without_nan_and_the_tables_list_every_rule(tmp_path):
    result = dict(backbones={"0.6b": run({s: make_trace(GOOD_HOP1, LATE) for s in rules.SEEDS})})
    text = json.dumps(result, allow_nan=False)
    assert json.loads(text)["backbones"]["0.6b"]["rules"]["R1"]["status"] == "infeasible"
    analyze_lsr.write_tables(result, tmp_path)
    endpoints = list(csv.DictReader((tmp_path / "endpoints.csv").open()))
    assert [(row["rule"], row["status"], row["passes"]) for row in endpoints] == [
        ("R1", "infeasible", "False"), ("R2", "evaluated", "True"), ("R3", "infeasible", "False")]
    series = {row["series"] for row in csv.DictReader((tmp_path / "accuracy.csv").open())}
    assert {"R2", "L1", "L8", "O1", "O2", "O3"} <= series and "R1" not in series and "R3" not in series
    depth_rows = list(csv.DictReader((tmp_path / "stop-depth.csv").open()))
    assert {row["rule"] for row in depth_rows} == {"R2"} and len(depth_rows) == 8 * rules.L_MAX
