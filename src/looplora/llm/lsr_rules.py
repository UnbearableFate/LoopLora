"""LSR rules, grid and validation selection (pure numpy; docs/prereg/LSR.md). No torch: the analysis runs anywhere.

A trace is a dict of arrays over (..., hops): ``pred`` [..., 9] holds p^0..p^8 (token ids), ``change`` and ``read_ratio``
[..., 8] hold step s = 1..8 at index s - 1 (read_ratio[..., 7] is NaN: the rule is forced at s = 8), ``answer`` the label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EPS = 1e-12
L_MAX = 8
DEPTHS = tuple(range(L_MAX + 1))                  # p^0 .. p^8
HOPS = tuple(range(1, 9))
RULES = ("R1", "R2", "R3")
REPO = Path("/work/xg24i002/x10041/LoopLora")
SEEDS = (17, 29, 43)
SEED = 4242
# docs/prereg/LSR.md: validation [1000, 3000) (P1-L / P5L used [0, 1000)); test window `lsr` = CONFIRMATION_WINDOWS["lsr"].
VAL_RANGE = dict(split="val", start=1000, count=2000, seed=SEED)
SMOKE_RANGE = dict(split="val", start=0, count=64, seed=SEED)
LATENCY_RANGE = dict(split="val", start=1000, count=128, seed=SEED)
WINDOW = "lsr"
GRID = tuple(2.0 ** (k / 2) for k in range(-20, 5))
GRID_K = tuple(range(-20, 5))
MARGIN_POOLED = 8     # pp, on validation (the decision bound is 10 pp)
MARGIN_SEED = 10      # pp, on validation, every seed
BACKBONES = {
    "0.6b": dict(label="Qwen3-0.6B-Base", manifest="runs/p1l/main/r128/training-manifest.json", root=REPO, backbone=None),
    "1.7b": dict(label="Qwen3-1.7B-Base", manifest="reports/p5l/r128/training-manifest.json", root=None,
                 backbone={"id": "Qwen/Qwen3-1.7B-Base", "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"}),
}


# ------------------------------------------------------------------------------------------ rules

def stop_depths(rule, trace, tau=None):
    """First qualifying s in 1..8 per question, array shaped like trace['answer'] ([..., hops]). s = 8 always fires.

    trace: dict with pred [..., 9], change [..., 8], read_ratio [..., 8] (index s - 1 holds step s).
    """
    shape = trace["pred"].shape[:-1]
    fires = np.zeros((*shape, L_MAX), dtype=bool)
    if rule == "R1":
        fires = trace["change"].astype(np.float64) <= tau
    elif rule == "R3":
        fires = np.nan_to_num(trace["read_ratio"].astype(np.float64), nan=np.inf) <= tau
    elif rule == "R2":
        p = trace["pred"]
        for s in range(2, L_MAX + 1):
            fires[..., s - 1] = (p[..., s] == p[..., s - 1]) & (p[..., s - 1] == p[..., s - 2])
    else:
        raise ValueError(f"unknown rule: {rule}")
    fires = fires.copy()
    fires[..., L_MAX - 1] = True
    return fires.argmax(-1) + 1


def fixed_depths(trace, depth):
    return np.full(trace["pred"].shape[:-1], depth, dtype=np.int64)


def reads_of(rule, depths):
    """Memory reads at the answer position per question: s* (R1, R2, fixed), s* + 1 (R3, the read at z^{s*} is paid)."""
    return depths + (1 if rule == "R3" else 0)


def correct_at(trace, depths):
    """Boolean correctness of p^{s*} per question."""
    predicted = np.take_along_axis(trace["pred"], depths[..., None], axis=-1)[..., 0]
    return predicted == trace["answer"]


def grid_k(tau):
    return int(round(2 * np.log2(tau)))


# ------------------------------------------------------------------------------------- selection

def _candidate(rule_name, depth_by_seed, traces, baseline):
    """Integer counts of one candidate over the pooled seeds (traces: dict seed -> trace; hop column j is hop j + 1)."""
    seeds = sorted(traces)
    correct = {s: correct_at(traces[s], depth_by_seed[s]) for s in seeds}
    hop1 = {s: int(correct[s][:, 0].sum()) - int(baseline[s][:, 0].sum()) for s in seeds}
    questions = {s: int(correct[s].shape[0]) for s in seeds}
    objective = sum(int(correct[s][:, 3:].sum()) for s in seeds)
    reads = sum(int(reads_of(rule_name, depth_by_seed[s]).sum()) for s in seeds)
    em = np.mean([correct[s].mean(0) for s in seeds], axis=0)       # equal seed weights, equal questions per seed
    n_pooled = sum(questions.values())
    feasible = (sum(hop1.values()) * 100 >= -MARGIN_POOLED * n_pooled
                and all(hop1[s] * 100 >= -MARGIN_SEED * questions[s] for s in seeds))
    return dict(feasible=bool(feasible), objective=objective, reads=reads, hop1_delta=hop1,
                em_by_hop=[float(v) for v in em],
                mean_reads=float(reads / (n_pooled * correct[seeds[0]].shape[1])),
                mean_depth=float(np.mean([depth_by_seed[s].mean() for s in seeds])))


def select_thresholds(traces):
    """Validation selection (docs/prereg/LSR.md, "Threshold selection"). traces: dict seed -> trace of ONE backbone.

    Per rule R1/R3: the grid value that maximises the summed correct count on hops 4-8 over the feasible grid values;
    ties go to fewer reads, then to the smaller k. R2 has no parameter and L_c is the same selection over fixed depths 1..8.
    Returns a JSON-able dict.
    """
    seeds = sorted(traces)
    baseline = {s: correct_at(traces[s], fixed_depths(traces[s], 4)) for s in seeds}
    result = dict(rules={}, fixed={}, baseline_em_by_hop=[float(v) for v in np.mean(
        [baseline[s].mean(0) for s in seeds], axis=0)])

    def choose(rows, key_index):
        feasible = [row for row in rows if row["feasible"]]
        if not feasible:
            return None
        return min(feasible, key=lambda row: (-row["objective"], row["reads"], row[key_index]))

    for rule in ("R1", "R3"):
        rows = []
        for k, tau in zip(GRID_K, GRID):
            depths = {s: stop_depths(rule, traces[s], tau) for s in seeds}
            rows.append(dict(k=k, tau=tau, **_candidate(rule, depths, traces, baseline)))
        best = choose(rows, "k")
        result["rules"][rule] = dict(grid=rows, selected=None if best is None else dict(k=best["k"], tau=best["tau"]),
                                     feasible=best is not None)
    depths = {s: stop_depths("R2", traces[s]) for s in seeds}
    result["rules"]["R2"] = dict(grid=[dict(k=None, tau=None, **_candidate("R2", depths, traces, baseline))],
                                 selected=None, feasible=True)
    rows = []
    for depth in range(1, L_MAX + 1):
        depths = {s: fixed_depths(traces[s], depth) for s in seeds}
        rows.append(dict(k=depth, tau=None, **_candidate("fixed", depths, traces, baseline)))
    best = choose(rows, "k")
    result["fixed"] = dict(grid=rows, selected=None if best is None else dict(L=best["k"]), feasible=best is not None)
    # O1: the best fixed depth per hop on validation (ties: the smaller L)
    per_hop = np.mean([[correct_at(traces[s], fixed_depths(traces[s], d)).mean(0) for d in range(1, L_MAX + 1)]
                       for s in seeds], axis=0)                                  # [8 depths, 8 hops]
    result["oracle_best_fixed_L_by_hop"] = [int(per_hop[:, h].argmax() + 1) for h in range(per_hop.shape[1])]
    result["fixed_em_by_depth_hop"] = per_hop.tolist()
    return result
