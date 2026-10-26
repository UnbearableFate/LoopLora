# WP-F1 — P1-S follow-up F1: rank sweep for random and learned codes

Read first: `docs/prereg/P1S-F1.md` (binding), `docs/prereg/P1S.md` (final v1: sets, steps, lrs, endpoints),
`docs/reports/P1S-pilot.md`, `docs/tasks/WP-H_p1s_main.md` (the same pipeline and statistics). Branch/worktree:
`wp/symf` from `master`. Report: `docs/reports/P1S-F1.md`; machine-readable outputs under `reports/p1s-f1/`;
runs under `runs/p1s/f1/` (not committed).

## Runs
`oracle_random` and `joint`, r = r_m ∈ {64, 128}, variants M1, M1-lin, B5, B5-lin, B3, seeds 17/29/43, 20000
steps, batch 64, lr from the P1-S table for the same (writer, variant). Evaluate exactly as WP-H §2 (screening,
`test_unseen`, counterfactuals on screening; confirmation eval seed 777001, 2000 episodes). Do not look at
confirmation metrics until every F1 run has finished training.

## Statistics and report
WP-H §3 machinery per (condition, r, reader): G1 with walk specificity (no replication), hop-1 criterion, G2;
gate table `reports/p1s-f1/gates.{json,md}`; trend table/figure of Δ_reach, Δ_far and hop-1 change versus r
(include the r = 32 values from WP-H once `reports/p1s/` exists on master; otherwise leave a placeholder).

## Compute
One node per job, several processes per node as measured; each job ≤ 1 h; F1 total ≤ 6 node·h. wk-sym runs WP-H
at the same time: do not touch `wp/sym`, `runs/p1s/main` or its jobs. Code changes only if the pipeline cannot
express a run; keep them minimal and tested.
