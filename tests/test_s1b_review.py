"""Behavior regressions for the S1B cross-review; synthetic data, compute only."""
import numpy as np

from looplora.llm import s1b_analysis as stats, s1b_data as data


def test_review_zero_boundary_keeps_storage_null_and_selects_matrix():
    """RVC-01: the registered 1/120 percentile has two exactly-zero bracketing sums."""
    n = 2000
    vector = np.roll(np.r_[np.ones(1000, dtype=np.int64), np.zeros(103, dtype=np.int64),
                           -np.ones(897, dtype=np.int64)], 3)
    main = {(arm, seed): dict(correct=np.zeros((n, 8), dtype=np.int64),
                             nll=np.ones((n, 8)), episode_id=np.arange(n, dtype=np.int64), hops=np.arange(1, 9))
            for arm in ("M1", "M4", "S1", "S4") for seed in data.RUN_SEEDS}
    for sign, arm in ((1, "S4"), (-1, "M4")):
        for j, i in enumerate(np.flatnonzero(vector == sign)):
            main[arm, data.RUN_SEEDS[j % 3]]["correct"][i, 1] = 1
    sweeps = {(arm, seed, depth): dict(main[arm, seed], correct=(main[arm, seed]["correct"].copy()
              if depth >= 4 else np.zeros((n, 8), dtype=np.int64)))
              for arm in ("M4", "S4") for seed in data.RUN_SEEDS for depth in (1, 2, 4, 8)}
    result = stats.paired(main, sweeps, "advance")
    storage = result["contrasts"]["S4-M4"]
    assert storage["interval_pp"][0] == 0
    assert storage["tests"]["lower_positive"] is False
    assert storage["passed"] is False
    assert result["continuation"]["M"]["qualified"] and result["continuation"]["S"]["qualified"]
    assert result["s2_primary"] == "matrix"
