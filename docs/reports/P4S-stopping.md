# WP-C3 — P4-S stopping pilot (2026-10-26)

**Relative-state-change stopping improves oracle-random/r128 accuracy, but does not uniformly remove hop-1 overshoot or reduce latency.** Evaluated 24 fixed F1 checkpoints: oracle_random/joint × r64/r128 × M1/M1-lin × seeds 17/29/43. Each used 2000 validation and 2000 screening-test episodes, seed 4242, hops 1–8. No training, r32 checkpoints or confirmation episodes. Thresholds were selected per family/criterion on validation, shared across seeds and locked before test.

Predictions use the first qualifying post-update state z^L (1≤L≤16). Read norm uses ‖UMQz^L‖/‖UMQz^0‖ and costs L+1 reads; state change uses ‖z^L−z^(L−1)‖/‖z^(L−1)‖; stability counts consecutive unchanged argmax pairs. The [design](../../reports/p4s/design.json), [thresholds](../../reports/p4s/thresholds.csv), and [eight-family tables](../../reports/p4s/tables.md) retain every rule, fixed L=1/2/4/8/16, true-hop control, per-hop accuracy and mean steps.

Percentages below are screening means over three seeds; SC is relative-state-change stopping. Long means hops 4–8. GPU time is milliseconds per prepared batch of 64 queries.

| Oracle-random family | H1: L1 / L4 / SC | Long: L8 / SC | SC mean steps | GPU: L16 / SC |
| --- | --- | --- | --- | --- |
| r64 M1 | 97.65 / 63.35 / 67.20 | 33.46 / 40.95 | 9.49 | 2.90 / 10.62 |
| r64 M1-lin | 98.80 / 65.62 / 61.88 | 36.35 / 45.77 | 11.02 | 2.73 / 9.99 |
| r128 M1 | 99.98 / 79.10 / 87.88 | 45.75 / 70.69 | 8.30 | 2.91 / 11.06 |
| r128 M1-lin | 100.00 / 81.48 / 96.62 | 44.85 / 64.41 | 7.15 | 2.69 / 10.29 |

SC partly recovers short-hop loss at r128 while improving long-hop accuracy; r64/M1-lin loses hop-1 accuracy versus L4. No rule uniformly restores L1 hop-1 accuracy. Joint families mostly stop near 16 updates; r128/M1 stability uses 12.19 steps with 65.83% overall accuracy versus 65.68% at L16, a descriptive pilot difference.

**No adaptive rule is faster than fixed L16 at batch 64 or 1.** Timings use FP32, a GH200, five warmups and 20 repeats, CUDA events and synchronized wall time; batch 1 averages eight separate hop queries. They include candidate decoding and active-query bookkeeping, excluding context preparation. GPU/wall mean/median/p95 values are in [latency.csv](../../reports/p4s/latency.csv); preparation timing remains in the raw run. This reference implementation gains no speed from fewer updates.

**0/96000 episode certificates satisfy c<1**; bounds span 4.57–37.07 and are vacuous. This proves neither divergence nor correctness. True-hop stopping is a label-informed reference, **not an accuracy upper bound**: learned update count need not equal graph hop count (oracle_random/r128/M1: true-hop 23.56%, SC 75.30% overall).

Validation: **126 tests passed** (3713584); evaluation **3713666**, mg0835, exit 0, 528.6s; final [audit](../../reports/p4s/audit.json) **3714004**, mg0888, exit 0. All 24 checkpoints were unchanged, all 960 fixed screening cells exactly matched F1, threshold ties passed, and all 1728 latency cells passed decision/state checks. An initial FP32 componentwise check was investigated with FP64 and replaced by a recorded vector-error check; [reproduction details](../../reports/p4s/reproduce.md) document this. [Cost](../../reports/p4s/jobs.json): **0.1825/2 node·h**, including failed smoke and probe. Raw outputs: `/work/xg24i002/x10041/LoopLora/runs/p4s/main-v1/`. All runtime used compute via submit.py; no unresolved checks. Results are screening-only, without significance claims.
