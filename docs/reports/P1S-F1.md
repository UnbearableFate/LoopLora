# WP-F1 — Symbolic rank follow-up (2026-10-26)

Completed all **60 preregistered runs**: oracle_random/joint × r=r_m=64/128 ×
five variants × seeds 17/29/43, **20000 steps**, batch 64, unchanged selected LRs.
Training source `7aed0f3`; evaluation/analysis `a76e46b`, reusing tested WP-H
pipeline `b900d92`. F1 additions are its drivers, gate wrapper and tests; the
shared pipeline code was reused unchanged.

**Both joint readers pass G1 and G2 at both ranks.** Values below are percentage
points on confirmation seed 777001, 2000 episodes; gates use the preregistered
97.5% episode bootstrap, 2000 joint resamples across the three training seeds.

| Joint rank | Reader | Δ_reach | Δ_reach−Δ_far | Hop-1 change | Pass |
| --- | --- | --- | --- | --- | --- |
| 64 | M1 | +30.53 | +20.93 | +9.02 | yes |
| 64 | M1-lin | +22.42 | +15.23 | +6.08 | yes |
| 128 | M1 | +33.99 | +24.57 | +8.42 | yes |
| 128 | M1-lin | +40.59 | +32.77 | +5.57 | yes |

For joint/r64/M1, the reach CI is **[29.22,31.90] pp**, specificity CI
**[19.05,22.93] pp**. All four oracle_random rows pass every predicate except
hop-1 preservation: hop-1 degrades **17.43–33.10 pp**, so none passes G1.
The [complete gates](../../reports/p1s-f1/gates.md) retain both confidences,
per-seed effects and every criterion; [rank trends](../../reports/p1s-f1/rank_trend.md)
and [figure](../../reports/p1s-f1/rank_trend.png) show reach, far and hop-1 changes.

A5 is diagnostic only: on screening at L=4, joint/r64/M1 follows the new terminal
on **59.15%** versus **35.78%** of paired unswapped predictions; M1-lin gives
**59.87% versus 38.25%** (all original hops, pooled seeds). B3's paired differences
are exactly zero. [A5 tables](../../reports/p1s-f1/swap_successors.csv) retain
original-hop and new-hop≤L strata for every variant/rank at L=1/4/8.
[Unseen contrasts](../../reports/p1s-f1/reported_contrasts.csv) preserve positive
joint reach/specificity effects; [other counterfactuals](../../reports/p1s-f1/counterfactual_consistency.csv)
and [all accuracy cells](../../reports/p1s-f1/accuracy_tables.csv) are retained.

**Limitations:** F1 omits replication and is weaker than WP-H. The entity backbone
remains 64-dimensional; rank changes reader capacity and memory together. Rank 64
is the smallest tested passing F1 rank. WP-H r32 and oracle_orth/r32 results
were absent on master at completion, so the explicitly permitted placeholders
remain; the r32 comparison and P1-L rank decision await those results. FP32 M
state is 16/64 KiB per episode at r64/r128, versus 4 KiB at r32 (4×/16× for P2 pricing).

Validation: `pytest -q -o addopts= tests/test_p1s_f1.py tests/test_sym_main_evaluate.py tests/test_sym_specificity.py tests/test_sym_gates.py`
— **83 passed**, PBS **3712884**, `mg0041`. The [artifact audit](../../reports/p1s-f1/artifact_audit.json)
passed **54,720,000 sample rows**, including depth snapshots, oracle controls,
labels, pairing and file seals. All 60 training runs and all 60 blind confirmation
sets preceded the [global release](../../reports/p1s-f1/release_barrier.json);
confirmation analysis ran once, PBS **3713306**, `mg0003`, exit 0. The A5 summary
check also passed (3713353). Every workload ran through `scripts/submit.py` on
Miyabi-G compute. **16 jobs, all exit 0; 4.184722/6 node·h**, each one node and
≤50 min requested ([ledger](../../reports/p1s-f1/jobs.md), [registry](../../reports/p1s-f1/experiment_registry.csv)).
Checkpoints/samples remain under `runs/p1s/f1/`. No protocol deviations,
non-finite retries or skipped required F1 checks; r32 placeholders are authorized
by the task.
