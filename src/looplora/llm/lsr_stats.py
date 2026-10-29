"""LSR test statistics (pure numpy; docs/prereg/LSR.md, "Test evaluation and endpoints").

Per backbone, the rules that had a feasible validation threshold (R2 always) are compared with fixed L = 4 on the test traces of the
three seeds. The unit of resampling is the episode: one resample is shared by every statistic, seed, hop and rule (whole-episode joint
bootstrap). Registered intervals are two-sided percentile intervals at 1 - 0.05/6; descriptive intervals are 95%.
"""
from __future__ import annotations

import numpy as np

from looplora.infra.seeds import derive_seed
from looplora.llm import lsr_rules as rules

COMPARISONS = 6                       # 3 rules x 2 backbones
ALPHA = 0.05 / COMPARISONS
DRAWS = 10_000
GAIN_PP = 3.0                         # hops 4-8 gain against fixed L = 4, point estimate
HOP1_PP = -10.0                       # hop-1 change against fixed L = 4, point estimate and every seed
HOPS_LONG = (3, 4, 5, 6, 7)           # columns of hops 4-8
HOPS_SHORT = (0, 1, 2)                # columns of hops 1-3


def resample_counts(episodes: int, draws: int, seed: int) -> np.ndarray:
    """[draws, episodes] multiplicities of one shared whole-episode bootstrap (int16)."""
    generator = np.random.default_rng(seed)
    counts = np.empty((draws, episodes), dtype=np.int16)
    for row in range(draws):
        counts[row] = np.bincount(generator.integers(0, episodes, episodes), minlength=episodes)
    return counts


def bootstrap(counts: np.ndarray, vectors: dict[str, np.ndarray], alpha: float) -> dict[str, tuple[float, float, float]]:
    """Point, lower and upper bound (percent points) of the episode mean of every vector, one shared resample."""
    names = list(vectors)
    matrix = np.stack([vectors[name] for name in names], axis=1).astype(np.float64)            # [E, K]
    boot = (counts.astype(np.float64) @ matrix) / counts.shape[1] * 100.0                       # [draws, K]
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    point = matrix.mean(0) * 100.0
    return {name: (float(point[i]), float(low[i]), float(high[i])) for i, name in enumerate(names)}


def correctness(traces: dict, depths: dict) -> np.ndarray:
    """[seeds, episodes, hops] correctness of p^{s*}; ``depths`` maps seed -> [episodes, hops] stop depths."""
    return np.stack([rules.correct_at(traces[s], depths[s]) for s in sorted(traces)])


def fixed(traces: dict, depth: int) -> np.ndarray:
    return correctness(traces, {s: rules.fixed_depths(traces[s], depth) for s in traces})


def difference(left: np.ndarray, right: np.ndarray, hops, seed=None) -> np.ndarray:
    """Per-episode mean over seeds (or one seed index) and the given hop columns of correct(left) - correct(right)."""
    delta = left.astype(np.float64) - right.astype(np.float64)
    delta = delta[..., list(hops)]
    return delta.mean(axis=(0, 2)) if seed is None else delta[seed].mean(axis=1)


def evaluate_backbone(traces: dict, entry: dict, backbone: str, *, draws: int = DRAWS, base_seed: int = 2718) -> dict:
    """Endpoints and descriptive tables of one backbone from its test traces and its locked threshold entry."""
    seeds = sorted(traces)
    episodes = traces[seeds[0]]["pred"].shape[0]
    counts = resample_counts(episodes, draws, derive_seed(base_seed, "lsr", backbone))
    base = fixed(traces, 4)
    chosen_fixed = entry["fixed"]["selected"]
    l_c = None if chosen_fixed is None else int(chosen_fixed["L"])
    matrix = {depth: fixed(traces, depth) for depth in range(1, rules.L_MAX + 1)}
    result = dict(backbone=backbone, episodes=episodes, seeds=seeds, draws=draws, alpha=ALPHA, rules={}, l_c=l_c)
    vectors, registered_vectors, reports = {}, {}, {}
    for rule in rules.RULES:
        chosen = entry["rules"][rule]
        if not chosen["feasible"]:       # no feasible validation threshold: the rule fails; no test statistic is computed
            result["rules"][rule] = dict(status="infeasible", passes=False, test_statistics=None)
            continue
        tau = None if rule == "R2" else chosen["selected"]["tau"]
        depths = {s: rules.stop_depths(rule, traces[s], tau) for s in seeds}
        corr = correctness(traces, depths)
        reports[rule] = dict(depths=depths, corr=corr, tau=tau)
        registered_vectors[f"{rule}|gain48"] = difference(corr, base, HOPS_LONG)
        registered_vectors[f"{rule}|hop1"] = difference(corr, base, (0,))
        for index, seed in enumerate(seeds):
            registered_vectors[f"{rule}|gain48|{seed}"] = difference(corr, base, HOPS_LONG, index)
            registered_vectors[f"{rule}|hop1|{seed}"] = difference(corr, base, (0,), index)
        for label, hops in (("hops13", HOPS_SHORT), ("hop2", (1,)), ("hop3", (2,))):
            vectors[f"{rule}|{label}"] = difference(corr, base, hops)
        for hop in range(8):
            vectors[f"{rule}|vsL4|hop{hop + 1}"] = difference(corr, base, (hop,))
        for name, other in (("L8", matrix[8]), ("Lc", matrix[l_c] if l_c else base)):
            vectors[f"{rule}|vs{name}|gain48"] = difference(corr, other, HOPS_LONG)
            vectors[f"{rule}|vs{name}|hop1"] = difference(corr, other, (0,))
        for hop in range(8):
            vectors[f"{rule}|em|hop{hop + 1}"] = corr[..., hop].astype(np.float64).mean(0)
    for label, array in (("L4", base), ("L8", matrix[8])):
        for hop in range(8):
            vectors[f"{label}|em|hop{hop + 1}"] = array[..., hop].astype(np.float64).mean(0)
    registered = bootstrap(counts, registered_vectors, ALPHA) if registered_vectors else {}
    descriptive = bootstrap(counts, {**registered_vectors, **vectors}, 0.05)
    for rule, report in reports.items():
        gain, hop1 = registered[f"{rule}|gain48"], registered[f"{rule}|hop1"]
        per_seed = {str(s): dict(gain48=registered[f"{rule}|gain48|{s}"], hop1=registered[f"{rule}|hop1|{s}"]) for s in seeds}
        conditions = dict(
            gain=bool(gain[0] >= GAIN_PP and gain[1] > 0),
            hop1=bool(hop1[0] >= HOP1_PP),
            per_seed=all(v["gain48"][0] > 0 and v["hop1"][0] >= HOP1_PP for v in per_seed.values()),
            feasible=True)
        depths = np.stack([report["depths"][s] for s in seeds])                                   # [S, E, 8]
        cost = np.zeros(8)
        counts_by_hop = np.zeros((8, rules.L_MAX), dtype=int)
        for hop in range(8):
            counts_by_hop[hop] = np.bincount(depths[..., hop].ravel(), minlength=rules.L_MAX + 1)[1:]
            cost[hop] = rules.reads_of(rule, depths[..., hop]).mean()
        result["rules"][rule] = dict(
            status="evaluated", tau=report["tau"], passes=all(conditions.values()), conditions=conditions,
            gain48=gain, hop1=hop1, per_seed=per_seed,
            descriptive={k.split("|", 1)[1]: v for k, v in descriptive.items() if k.startswith(rule + "|")},
            mean_reads=float(rules.reads_of(rule, depths).mean()), mean_reads_by_hop=[float(x) for x in cost],
            mean_depth=float(depths.mean()),
            upper_layer_passes=float((depths + 1).mean()) if rule == "R2" else None,
            stop_depth_counts_by_hop=counts_by_hop.tolist(),
            em_by_hop=[float(x) for x in report["corr"].mean(axis=(0, 1))],
            em_by_hop_per_seed={str(s): [float(x) for x in report["corr"][i].mean(0)] for i, s in enumerate(seeds)})
    result["fixed"] = {f"L{d}": dict(em_by_hop=[float(x) for x in matrix[d].mean(axis=(0, 1))],
                                    em_by_hop_per_seed={str(s): [float(x) for x in matrix[d][i].mean(0)]
                                                        for i, s in enumerate(seeds)}) for d in matrix}
    result["fixed"]["L4_em_ci95"] = {f"hop{h + 1}": descriptive[f"L4|em|hop{h + 1}"] for h in range(8)}
    result["fixed"]["L8_em_ci95"] = {f"hop{h + 1}": descriptive[f"L8|em|hop{h + 1}"] for h in range(8)}
    pooled = np.array([matrix[d].mean(axis=(0, 1)) for d in range(1, rules.L_MAX + 1)])         # [8 depths, 8 hops]
    by_depth = entry["oracle_best_fixed_L_by_hop"]
    any_correct = np.stack([matrix[d] for d in range(1, rules.L_MAX + 1)]).any(0)                # [S, E, 8]
    long, short = list(HOPS_LONG), list(HOPS_SHORT)
    em_o1 = [float(pooled[by_depth[h] - 1, h]) for h in range(8)]
    em_o2 = [float(pooled[:, h].max()) for h in range(8)]
    em_o3 = [float(x) for x in any_correct.mean(axis=(0, 1))]
    result["oracles"] = dict(
        note="labels used: descriptive only",
        O1_best_fixed_L_by_hop_from_validation=dict(L=by_depth, em_by_hop=em_o1, hops13=float(np.mean([em_o1[h] for h in short])),
                                                      hops48=float(np.mean([em_o1[h] for h in long]))),
        O2_best_test_cell_by_hop=dict(L=[int(pooled[:, h].argmax() + 1) for h in range(8)], em_by_hop=em_o2,
                                      hops13=float(np.mean([em_o2[h] for h in short])), hops48=float(np.mean([em_o2[h] for h in long]))),
        O3_any_L_in_1_8_correct=dict(em_by_hop=em_o3, hops13=float(np.mean([em_o3[h] for h in short])),
                                     hops48=float(np.mean([em_o3[h] for h in long]))))
    return result

