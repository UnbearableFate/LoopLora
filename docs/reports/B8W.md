# WP-L10 / B8W - B8 warm start from M1-L4 (G3), 2026-10-29

Author: sn-p1l (Claude Sonnet 5.5), branch `wp/p1l`. Decision record: `reports/decisions/B8W.md`. Tables: `reports/b8w/results/` (`results.md`, `endpoints.json`, CSVs),
lock `reports/b8w/checkpoint-lock.json`, ledger `reports/b8w/jobs.json`, state `docs/reports/B8W-state.md`, prereg `docs/prereg/B8W.md` (hash `7414b6e0…`).
Raw runs: `/work/xg24i002/x10041/LoopLora/runs/b8w/`.

## Result (test [108000, 110000), eval seed 4242, 2,000 episodes, L = 4, seeds 17/29/43, 95% CIs)
**Registered decision: untying keeps the solution.** Delta = untied@4 - shared@4, hops 2-3: **+0.90 pp [0.09, 1.71]** (32.40% vs 31.50%), per seed +1.05 / +0.90 / +0.75.
Every seed loses at most 2 pp (none loses), and the pooled lower bound is above -2 pp. B8 gets no equal-parameter status, G1-L stays failed, and nothing is said about why B8 failed
from a random start.

| Item | Result |
| --- | --- |
| Equality gate (M1-L4 vs its B8 copy; 64 val episodes, bf16 logits, tolerance 0) | passed in every seed, max abs difference 0.0, control equal |
| Protocol check (shared more than 2 pp below the start) | not triggered: shared - start +5.09 pp [4.18, 6.01] |
| untied - shared, hop 1 | +2.77 [1.55, 4.02] |
| untied / shared - start (hops 2-3) | +5.99 [5.07, 6.88] / +5.09 [4.18, 6.01] |
| untied - original B8@4 (P1-L) / - B4@1 | +26.07 [24.98, 27.13] / +26.98 [25.87, 28.07] |
| Drift (R relative change from start) | shared 0.44-0.49; untied copies 0.48-0.58; copies separated (pairwise cosine distance R 0.19-0.21) |

The copied solution is kept and both arms keep improving (the starting M1-L4 was not converged). Exploratory only: untied is nominally above shared; no superiority test was registered.

## Jobs and cost (`reports/b8w/jobs.json`)
10 jobs, **1.004 node·h** (budget 1.0-2.0): tests and smoke 0.20, three continuation jobs 0.704 (14 min each), pipeline tests 0.012, authorization check 0.006, evaluation (all 15 groups, one
job) 0.073, finalizer 0.007. One failing job (3723982: a defect in my new test, fixed and rerun clean).

## Limits
Three seeds, one N (2,000), one lr, one range, oracle condition, seen names. The shared reference was still improving, so the result is sensitive to N. No claim about sharing as a necessity,
about B8's random-start failure, or beyond the oracle condition.
