# P2-S-F1 — matched-depth sharing and shortcut-free budget contrasts (2026-10-27)

**Carry B8 into P1-L, trained at fixed L4. The shortcut-free learned-code
capability claim passes at every tested budget.** These are the registered
[P2S-F1](../prereg/P2S-F1.md) decisions; writer selection stays with P1-L.

Completed 24 validation pilots and **24 fresh main runs**, seeds 17/29/43,
**15000 steps**, batch 64. The eight LRs were locked in `44c1fd5` before main
training. New models are M1-L4 under joint/oracle_random/joint_fact, plus
joint_fact B6/r64,r128, B7/m64,m128 and B8/L4. Reused 18 unchanged reference
checkpoints: P2-S B8, WP-H mixed-depth M1, and F2 joint_fact M1/B4.

Every main run finished before confirmation evaluation. All 42 blind sample
sets preceded aggregation on test seed **888001**, 2000 episodes, one query
per hop 1–8. The [full tables](../../reports/p2s-f1/tables.md) retain all
97.5% pooled paired bootstrap CIs and per-seed effects; [CSV](../../reports/p2s-f1/endpoints.csv)
also includes 95% intervals and absolute per-seed accuracies. Reach means
hops {2,3}; differences below are percentage points:

- **F-E1, B8 − M1-L4:** joint **+13.217 [11.983,14.450]**;
  joint_fact **+5.458 [4.242,6.750]**; oracle_random
  **+3.200 [1.958,4.517]**. Sharing still costs short-reach accuracy at
  matched training depth. B8 is also better at hop 1, but worse on hops 4–8
  for all three writers; this choice does not establish extrapolation superiority.
- **F-E2, M1-L4 − M1-mixed:** joint **+20.517 [19.208,21.783]**;
  joint_fact **+3.483 [2.250,4.758]**; oracle_random
  **+8.242 [7.117,9.400]**. All three writer-specific rules select fixed L4.
  Joint's observed B8 gap shrinks from 33.733 to 13.217 pp. LRs are tuned
  separately, so this contrast includes the selected training protocol.
- **F-E3, joint_fact M1-mixed@4 − larger single read:** B6/r64
  **+24.017 [23.000,25.108]**, B6/r128 **+24.242 [23.242,25.333]**,
  B7/m64 **+23.875 [22.858,24.975]**, B7/m128
  **+23.550 [22.533,24.659]**. All per-seed reach effects are positive.
  M1 reaches **24.483%**, versus **0.242–0.933%** for these single reads.

Uncertainty uses shared `sym.stats`, 2000 episode-bootstrap draws, seed 2718,
conditional on three fixed training seeds. As preregistered, these episodes
were already used in P2-S; this is a registered follow-up on a reused set.

Validation: `pytest -q -o addopts= tests/test_p2s_f1.py tests/test_p2s.py` —
**26 passed**, final PBS **3715040**, `mg0003`; evaluation/statistics exited 0.
The [audit](../../reports/p2s-f1/audit.json) passed **3,312,000 rows**, sample
seals, paired labels, immutable depth snapshots and checkpoint checks.
All earlier runs stayed read-only. No shared model/evaluation code changed.

[Resource ledger](../../reports/p2s-f1/budget.json): **1.099/2.5 node-hours**,
pilot including overhead **0.293/0.5**; all ten jobs exited 0, one node each,
≤25-minute requests through `scripts/submit.py`. No latency benchmark was
required. [Compact artifacts](../../reports/p2s-f1/README.md) are below 1 MB;
raw checkpoints and samples remain under `runs/p2s-f1/`. No scientific
deviations or open checks.
