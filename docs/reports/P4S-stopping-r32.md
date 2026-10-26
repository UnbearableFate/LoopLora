# WP-C4 — r32 learned-code stopping (2026-10-27)

**None of the four factorized families removes overshoot with WP-C3's selected rules.**
The best hop-1 preservation among the three rules still loses **24.90–41.73 pp**
versus L1, despite positive reach gain and walk specificity. The unfactorized
joint/M1 contrast satisfies the screening criteria with effectively fixed L16;
joint/M1-lin has negative walk specificity for every rule.

Evaluated all 18 released checkpoints: joint_fact/t1_frozen_fact/joint × M1/M1-lin
× seeds 17/29/43, r=32. Each used 2000 validation and 2000 screening-test episodes,
seed 4242, one query per hop 1–8. Reused WP-C3's reader, threshold grids, first
qualifying post-update prediction (1≤L≤16), and validation-accuracy selection over
eight hops and three seeds. Family thresholds were locked before screening.
No training, confirmation episodes, latency benchmarks or certificates.

Below, each factorized row shows its **least hop-1 loss among the three selected
rules**, a descriptive screening comparison, not a new selection procedure.
Unfactorized rows show their qualifying/economical contrast. Deltas are pp versus
L1; specificity = reach delta (hops 2/3) minus far delta (6/7/8).

| Family | Rule | Hop1 Δ | Reach Δ | Far Δ | Specificity | Steps | Qualifies? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint_fact / M1 | state change | −33.03 | +21.35 | +3.44 | +17.91 | 6.71 | No |
| joint_fact / M1-lin | stability | −41.73 | +21.46 | +6.36 | +15.10 | 10.46 | No |
| t1_frozen_fact / M1 | stability | −29.68 | +10.23 | +2.91 | +7.33 | 7.45 | No |
| t1_frozen_fact / M1-lin | read norm | −24.90 | +9.52 | +2.53 | +6.99 | 9.07 | No |
| joint / M1 | all ≈L16 | +5.15 | +8.67 | +6.03 | +2.64 | 16.00 | Yes |
| joint / M1-lin | stability | +2.78 | +2.74 | +2.93 | −0.19 | 4.35 | No |

Qualification means hop-1 degradation ≤1 pp with positive reach and specificity;
hop-1 improvement is allowed. All factorized families retain positive reach and
specificity in every seed, but fail preservation. joint/M1's minimum seed
specificity is −0.125 pp. [Full tables](../../reports/p4s-r32/tables.md) include
every rule, fixed L=1/4/8/16 and the label-informed true-hop reference (not an
upper bound); [per-seed endpoints](../../reports/p4s-r32/endpoints_per_seed.csv)
retain variation. These are point estimates: WP-C3 had no bootstrap implementation.

**Recommendation: no formal P4 test with a fresh confirmation seed for any of
these rule/checkpoint combinations.** The learned walk families fail by a wide
margin, and joint/M1 adds no stopping benefit over fixed L16. This conclusion is
specific to the unchanged rules and overall-validation selection, not a claim
that label-free stopping is impossible.

Validation: **77 tests passed**, then six-family smoke passed (PBS **3714219**,
`mg0013`); evaluation and [audit](../../reports/p4s-r32/audit.json) **3714230**,
`mg0005`, exit 0, 33.51 application seconds. All 36 traces/576000 queries passed;
all 576 fixed screening cells exactly matched release, all checkpoint seals were
unchanged, and threshold tie/lock audits passed. [Commands and artifacts](../../reports/p4s-r32/README.md)
record the only execution correction: an initial `HEAD` submission selected the
main checkout and stopped before tests; explicit commit 613376b fixed it.
[Total cost](../../reports/p4s-r32/jobs.json): **0.02056/1 node·h**, including that
attempt. All runtime used compute via submit.py. Compact outputs are <1 MB;
raw traces remain under `LoopLora/runs/p4s-r32/main-v1/`. No open checks or
scientific deviations.
