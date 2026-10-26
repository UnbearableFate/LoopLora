# WP-D2 — P1-S pipeline amendments and pilots (validation only)

Read first: `docs/prereg/P1S.md` (binding), `docs/reports/WP-D.md`. Branch/worktree: `wp/sym` (continue; merge
`master` first). Report: `docs/reports/P1S-pilot.md` (tables + decisions proposed by the rules; the orchestrator
confirms them in the prereg).

## 1. Code amendments (with tests)
a. `looplora.sym.evaluate` CLI: `--set key=value` overrides for `eval.seed`, `eval.episodes`, `eval.splits`,
   `eval.batch_size`, `eval.counterfactuals` (recorded in the output metrics/config). Needed for the
   confirmation set: split `test`, `eval.seed = 777001`, 2000 episodes (disjoint from the screening set).
b. `looplora.sym.stats`: pooled multi-seed paired bootstrap. Input: per-seed sample files that share episodes;
   resample episodes jointly (same indices for every seed), statistic = mean over seeds of the per-seed paired
   accuracy difference on a hop subset. Report pooled estimate, 95% CI, per-seed estimates. Both forms:
   same checkpoint different L, and different checkpoints at given L.
c. Config: B6/B7 must not be validated against a hard-coded 64; require an explicit larger `r` / `r_mem` than a
   declared `model.reference_r` (default: the main comparison r).
d. T1 writer pretraining: optional `pretrain.task` mapping merged into the T1 spec (`head_sharing`, `n_rel`);
   the prereg uses `n_rel = 4`, `head_sharing = 0.3`.
e. Evaluation of 2000 episodes × 8 hops × 9 depths must fit the job budget; measure and report.

## 2. Pilot-1 (T1 capacity; val split only; seed 17)
`task t1` (n_rel 4, head_sharing 0.3), writer `joint`, variant B4 (this is the writer-pretraining setup),
r ∈ {16, 32} (r_mem = r), E ∈ {8, 12, 16, 24}, lr ∈ {3e-4, 1e-3, 3e-3}, batch 64, 5000 steps, val 1000 episodes.
Also oracle_random and oracle_orth (where feasible) with B4 at the same E and r, lr chosen the same way.
Report the full val accuracy table, train-loss curves (converged or not), and the prereg rule's E* and lr per r.

## 3. Pilot-2 (T2 trainability and lr; val only; seed 17; r = 32; E = E* from Pilot-1)
`t2_chain_end` with chains (8,4,2,2) if E* = 16 or (8,8,4,2,2) if E* = 24 (else as prereg says). Conditions
`oracle_random`, `oracle_orth` (if #entities ≤ 32), `t1_frozen` (one shared writer pretrained with Pilot-1's
lr for r = 32 at E*, reused by every variant via `writer_from`), `joint`. Variants M1, M1-lin, B5, B3, B4.
lr ∈ {3e-4, 1e-3, 3e-3}; batch 64; steps 5000 (same for all). Select lr per (condition, variant) by val
accuracy on hops 1–3 at L = 4 (B4 at L = 1). Report: selected lr, val hop × L tables at the selected lr, and
whether training converged. Also run the replication-setting configs at the selected lr (val only):
`t2_typed` (E = E*, n_distract = E*/4, chains with one length-8 chain summing to E − n_distract) for
`t1_frozen` and `joint`; `t2_chain_end` at E = 1.5·E* (one more length-8 chain plus short chains) for
`oracle_random`.

## 4. Do not
Evaluate on `test` or the confirmation set; change prereg thresholds; run seeds other than 17.
Each PBS job ≤ 8 nodes × 1 h (use one node, several procs). Record node·h per job in the report.

## Addendum A1 (orchestrator, 2026-10-26; see `docs/prereg/P1S.md` amendment A1)
Add variant `B5-lin` (`fixedq` with identity activation, named like `M1-lin`) to the config/variant tables,
tests and Pilot-2 (same conditions, lr grid and selection rule as B5). `looplora.sym.stats` pooled CIs must
take a confidence level (gates use 0.975, reports also 0.95). Report Pilot-2 tables for B5-lin as well.
