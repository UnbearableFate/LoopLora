# P1-S main results — WP-H (2026-10-26)

Completed **105 stages**: 72 primary, 18 replication, 12 secondary and three new shared
writers. Every stage used batch 64 and **15,000 steps**, selected by the preregistered
throughput fallback before main training. LRs, sets, seeds and thresholds remained
fixed. No training failure or LR retry occurred.

**M1 passes G1 and G2 with `oracle_orth` and `joint`; the intended `t1_frozen`
protocol does not pass.** Decisions use 2000 joint episode-bootstrap resamples and
97.5% confidence intervals; 95% intervals and every component are in the
[gate tables](../../reports/p1s/gates.md).

| Writer | M1: G1 / G2 | M1-lin: G1 / G2 |
| --- | --- | --- |
| oracle_random | Fail / Pass | Fail / Pass |
| oracle_orth | **Pass / Pass** | Fail / Pass |
| t1_frozen | Fail / Pass | Fail / Pass |
| joint | **Pass / Pass** | Fail / Pass |

For oracle_orth/M1, reachable-hop gain is **92.88 pp [92.23, 93.55]** and walk
specificity **92.87 pp [92.21, 93.56]**; for joint/M1 they are **8.29 pp [7.38, 9.29]**
and **2.49 pp [1.24, 3.73]** (97.5% CIs). Random writers and orthogonal/M1-lin fail
the hop-1 preservation criterion. Both frozen readers fail typed-replication
specificity; joint/M1-lin fails the 3 pp gain and primary/replication specificity.
The no-condition-passes-G1 stop rule is not triggered. The supported learned-writer
claim is limited to M1 with joint training and fixed-memory depth sweeps.

Training-seed heterogeneity remains material: joint/M1−B5 effects are
**+9.30, −17.70, +31.70 pp** for seeds 17/29/43. The specified episode bootstrap
conditions on these seeds. Shared-writer T1 validation accuracies remain
**87.4%, 88.2%, 87.9%**, below 95%; see [writer scores](../../reports/p1s/writer_t1_scores.md).

Confirmation seed 777001 was first aggregated after the
[global barrier](../../reports/p1s/barrier.json) verified all 105 stages and 102 sample
sets. The single-pass [audit](../../reports/p1s/artifact_audit.json) passed
**92,451,000 rows**, reproduced all tables, checked fixed snapshots across depths
and 34 controlled-writer groups, and matched all 24 frozen writers exactly. Outputs
include **46,227 accuracy cells, 444 nongating contrasts and 29 figures**. The
[artifact index](../../reports/p1s/README.md) links full tables, unseen/screening
results, training curves, ordinary counterfactuals and A5 `swap_successors` rates
by original hop and modified-hop reachability. A5 never enters gates.

Validation: **335 tests passed** on `mg0698`, PBS **3712766**, using
`python -m pytest -q -o addopts= tests/test_sym_*.py` through `scripts/submit.py`.
Reusable pipeline commit: `0f9de87`; training: `03b62f8`; finalization: `7e1c96f`.
The complete analysis/registry job **3713367** on `mg0841` exited 0. The
[registry](../../reports/experiment_registry.csv) preserves 24 previous rows and
adds 105 main stages plus 40 train-only throughput trials.

All **19 PBS jobs exited 0** and used **4.437222 / 8 node-hours** for WP-H;
P1-S including prior work totals **8.478056 / 13**. Every job used one node and
requested at most 39 minutes. The [ledger](../../reports/p1s/jobs.md) records each
job and source; the [budget decision](../../reports/p1s/training_budget.json)
documents the only preregistered adjustment. No requested work remains open.
