# P2-S — symbolic budget matching (2026-10-27)

**Gate 3's capability and practical claims do not pass.** Larger single reads
match or exceed M1/r32@L4 in three registered joint-writer comparisons. Depth
claims therefore remain restricted to the qualified matched-parameter and
matched-state comparisons of P1-S.

Completed **24 validation pilots and 24 main runs**, joint/oracle_random, seeds
17/29/43, batch 64, **15000 main steps**. LRs were locked in `be6dd16` before
main training. Evaluated all 24 new models and **60 unchanged references** on
test seed **888001**, 2000 episodes, hops 1–8, L0/1/2/4/8/16 where defined
(B8: L4 only). F1's 20000-step models enter descriptive cost curves only.

[E1–E3 tables](../../reports/p2s/tables.md) retain every 97.5% CI and per-seed
effect, including accompanying hop-1 and hops 4–8 differences;
[CSV](../../reports/p2s/endpoints.csv) also preserves 95% intervals and absolute
per-seed accuracies. All contrasts below are M1 minus the comparator, in pp:

- **E1 rank:** joint/r64 is positive, **+14.875 [13.658,16.184]**; joint/r128
  fails, **−3.233 [−4.675,−1.833]**. Both oracle_random sizes are positive,
  **+34.750/+34.775 pp**, with positive CIs.
- **E2 state:** joint/m64 and m128 fail: **−1.458 [−2.858,−0.125]** and
  **−2.283 [−3.592,−0.892]**.
- **E3 sharing:** B8 beats M1 for joint (**−33.733 [−34.942,−32.433]**) and
  oracle_random (**−11.442 [−12.792,−10.058]**). Sharing saves **9216 trainable
  parameters** and is not required for this tested L4 comparison. B8 trains
  only at L4; M1's reference training mixes L1/2/4.

[E4](../../reports/p2s/cost.csv) has **212 cost points** and
[636 seed-level rows](../../reports/p2s/cost_per_seed.csv), including parameters,
FP32 state, analytic read FLOPs and both timing clocks at batch 64/1. Joint
B6/r128 achieves **57.24% versus 48.02%** on hops 1–3 and **0.352 versus
0.848 ms/batch64** for M1@4, using **114690 versus 13314** trainable parameters
and **64 versus 4 KiB** state. Every joint M1 point is matched by an eligible
single-step point at both batch sizes. Oracle M1@4 remains unmatched, but the
overall capability requirement fails. The [cost plot](../../reports/p2s/cost_plane.png)
marks F1 descriptive points separately. Latency includes prepared reader and
candidate decoding, excludes preparation, and uses five warmups/20 repeats.

Uncertainty conditions on the three training seeds: joint/r128's per-seed
effects are **+7.175, −21.600, +4.725 pp**. These episode-bootstrap intervals
do not establish training-seed robustness or generalize beyond tested budgets.

Validation: `pytest -q -o addopts= tests/test_p2s.py` — **19 passed**, PBS
**3714837**, `mg0016`; initial accounting/invariant acceptance **153 passed**
(3714722, `mg0013`). Final evaluation, benchmark and analysis exited 0. The
[audit](../../reports/p2s/audit.json) passed **7,584,000 rows**, immutable depth
snapshots, paired labels, file seals, all checkpoints and **2862 timing cells**.
All 24 training stages and all 84 blind sets preceded the single confirmation
aggregation. [Artifacts](../../reports/p2s/README.md) remain below 1 MB in git;
raw checkpoints, samples and full metadata remain under `runs/p2s/`.

[Cost](../../reports/p2s/budget.json): **1.345/3 node-hours**, pilot **0.352/1**;
one node/job, ≤30-minute requests, all via `scripts/submit.py`. Eleven jobs:
ten succeeded; one smoke used the main checkout's HEAD and failed before
training, with its 8 seconds included. No scientific deviations or open checks.
