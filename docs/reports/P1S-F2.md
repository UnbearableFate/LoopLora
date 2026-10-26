# P1-S F2 — input-factorized writer (2026-10-26)

**Neither factorized condition passes for either reader.** F2 has no replication
setting and is weaker than P1-S main. All four rows pass G2 and the reach/specificity
checks, but fail G1 solely because extra reads degrade hop-1 accuracy far beyond
the allowed 1 pp. Under the registered F2 decision rule, **do not start P1-L**:
factorization alone does not satisfy the protocol at this scale, while
[WP-H's oracle_orth/M1](P1S-results.md) passes. This does not establish that learned codes cannot compose.

Reused [WP-C2](WP-C2.md) and WP-H; F2 adds drivers, four-row gates and paired
contrasts. Completed three fresh T1 writers and 36 readers (two conditions × six
variants × seeds 17/29/43), r=r_mem=32, E=12, chains (8,4), batch 64,
**15000 steps/stage**. Validation-only 5000-step pilots locked the registered
LR choices. Frozen readers share their seed's fresh writer; no failure or retry.

Confirmation seed 777001, 2000 episodes; differences below are percentage points.
Intervals use 2000 joint episode-bootstrap draws, conditional on the three seeds.

| Writer / reader | Reach [97.5% CI] | Specificity | Hop-1 change | G1 / G2 |
| --- | --- | --- | --- | --- |
| t1_frozen_fact / M1 | +11.68 [10.97, 12.43] | +8.94 | −29.82 | Fail / Pass |
| t1_frozen_fact / M1-lin | +11.58 [10.82, 12.35] | +9.12 | −27.93 | Fail / Pass |
| joint_fact / M1 | +22.98 [22.02, 23.98] | +20.16 | −35.62 | Fail / Pass |
| joint_fact / M1-lin | +27.61 [26.53, 28.71] | +24.81 | −37.47 | Fail / Pass |

[Full gates](../../reports/p1s-f2/gates.md) preserve both confidences, every
criterion and per-seed estimates. [Paired WP-H contrasts](../../reports/p1s-f2/cross_family_contrasts.csv)
show L4 accuracy differences of −19.26 to −11.03 pp and depth-gain differences
of +1.28 to +25.17 pp; every corresponding 97.5% CI excludes zero.
These endpoints never alter gates.
[Depth profiles](../../reports/p1s-f2/depth_profiles.png) show the low L1 baseline.

Main T1 validation accuracies are **91.8%, 91.2%, 92.4%**. The selected pilot
scored 86.5%; its single-fact/multiple-relation-head strata scored 91.7%/80.2%,
without identifying a unique cause. [Screening/unseen contrasts](../../reports/p1s-f2/reported_contrasts.csv)
retain positive specificity and large hop-1 losses. Diagnostic-only
[A5](../../reports/p1s-f2/swap_successors_new_path_length.csv): joint/M1@4 follows
48.0/33.7/16.7/5.5% of new terminals at path lengths 1/2/3/4, versus 4.1–4.6%
on paired unswapped queries. Original-hop/reachability strata are also retained.

Validation: `pytest -q -o addopts= tests/test_p1s_f2.py` — **14 passed**, PBS
**3713855**, `mg0008`; F2/factorized/compatibility checks **69 passed**
(3713528, `mg0023`). Finalization
**3714047**, `mg0017`, exited 0: [audit](../../reports/p1s-f2/artifact_audit_summary.json)
passed **32,835,000 rows**, all depth/controlled snapshots and all **18 frozen
writers**. All 39 training stages and 36 blind sets preceded aggregation;
released WP-H inputs were opened after F2's training barrier. Twelve matched
reference runs passed seals, snapshot and episode/query/hop/answer pairing checks.

All **18 PBS jobs exited 0**: pilot **0.384/2**, main **1.348/6**, total
**1.732/8 node-hours** ([ledger](../../reports/p1s-f2/job_ledger.csv)). Every job
used one node, requested ≤30 minutes, through `scripts/submit.py` on compute.
[Artifact index](../../reports/p1s-f2/README.md) links compact CSVs;
checkpoints, full tables and raw samples remain in `runs/p1s/f2/`. No scientific
deviations or open execution issues; A6 already registers this second follow-up.
Scope remains three seeds, no replication, separately tuned LRs and this budget.
