# WP-H — P1-S main runs, evaluation and gate statistics

Read first: `docs/prereg/P1S.md` (final v1, binding: sets, lrs, steps, matrix, endpoints, amendments A1–A4),
`docs/reports/P1S-pilot.md`. Branch/worktree: `wp/sym` (merge `master` first). Report:
`docs/reports/P1S-results.md`; machine-readable outputs under `reports/p1s/`; raw run directories under
`runs/p1s/main/` (not committed). The orchestrator writes the decision record (`reports/decisions/P1S.md`).

## 1. Training (prereg "Main runs"; 20000 steps, batch 64, the prereg lr table)
- Primary: 4 writer conditions × M1, M1-lin, B5, B5-lin, B3, B4 × seeds 17, 29, 43 (r = 32).
- Replication: M1, M1-lin × seeds 17, 29, 43 in `t1_frozen`/typed, `joint`/typed, `oracle_random`/E = 18
  (layouts as in Pilot-2's replication).
- Secondary: r = 16, seed 17: `oracle_random`, `oracle_orth`, `joint` × M1, M1-lin, B5, B5-lin.
- `t1_frozen`: one T1 writer per seed (r = 32, E = 12, lr 3e-3, 20000 steps, pretraining seed = run seed), shared
  by all variants and the typed replication of that seed. No Pilot checkpoint is reused (steps changed).
- First job: measure throughput at 4 vs 8 (or more) processes per node on a few real configs for ≤ 10 min; if
  the projection exceeds 8 node·h, switch every stage to 15000 steps (prereg Budget) before starting the matrix.
- A non-finite loss → rerun once at the next lower grid lr; record both attempts.

## 2. Evaluation (every run; depths 0,1,2,3,4,6,8,12,16; hops 1–8, one query per hop per episode)
- screening test: split `test`, eval seed 4242, 2000 episodes; also `test_unseen` (same seed and count) and
  counterfactuals (`replace_key_fact`, `remove_required_fact`, `swap_memory`) on the screening set.
- confirmation test: split `test`, eval seed 777001, 2000 episodes, no counterfactuals.
- Keep sample-level files for every set. Do not compute or look at any confirmation-set metric until every run
  in §1 has finished training and its confirmation samples are written; then compute §3 in one pass.

## 3. Statistics (`looplora.sym.stats` pooled paired bootstrap, 2000 resamples, episodes resampled jointly across
seeds; gate decisions at 97.5%, 95% also reported)
For each writer condition and each co-primary reader P ∈ {M1, M1-lin} (F = B5 for M1, B5-lin for M1-lin):
- G1: Δ_reach (L=4 − L=1, hops {2,3}): pooled estimate, CI, per-seed estimates; Δ_far (hops {6,7,8}); the
  walk-specificity contrast Δ_reach − Δ_far (per-episode difference of the two hop-subset contrasts, bootstrapped
  as one statistic); hop-1 pooled L=4 − L=1; the same Δ_reach and walk-specificity contrast in the condition's
  replication setting (`oracle_orth` and `oracle_random` share `oracle_random`/E = 18).
- G2: P@4 − F@4 and P@4 − B3@4 on hops {2,3} (different checkpoints, paired by episode).
- Gate table `reports/p1s/gates.{json,md}`: one row per (condition, reader) with every criterion of the prereg,
  pass/fail by the prereg thresholds only.
- Reported, not gating (same machinery): L=8 vs L=4 on hops 4–8; M1 vs M1-lin at L=4; B4 vs M1@1; r = 16 rows;
  seed-17 screening-set values of every gate statistic; `test_unseen`; counterfactual consistency.

## 4. Figures (`reports/figures/p1s/`, plan §9.3 items 1, 3, 4, 5)
Accuracy–L curves by hop per condition (M1 and M1-lin, seed mean with per-seed lines); hop × L heatmaps
(M1, M1-lin, B5, B5-lin, B3); load × L (E = 12 vs 18, oracle_random); paired-difference plot (P − F, P − B3,
M1 − M1-lin, with CIs) per condition.

## 5. Records and compute
- Append every run to `reports/experiment_registry.csv` with the plan §9.4 fields.
- Each PBS job ≤ 8 nodes × 1 h, walltime ≈ 1.5 × measured; record node·h per job and the WP-H total (≤ 8).
- Do not change prereg thresholds, sets, seeds, lrs or steps beyond the prereg's own 15000-step rule. Anything
  unexpected goes into the report as observed; ask the orchestrator before deviating from this spec.

## Addendum 1 (2026-10-26 21:30, prereg A5; diagnostic only, never gating)
WP-B2 is merged (master `bbe2a24`, kind `swap_successors` in `looplora.data.relations.counterfactual`); merge
`master` into `wp/sym` before the screening evaluation. For every primary and replication run, add
`swap_successors` (counterfactual seed 4242) on its screening set (`t2_chain_end`, or `t2_typed` for typed
replication) at L = 1, 4, 8 and keep sample-level files. Report per (condition, variant): the follow rate (share
of rows predicting the new terminal) on the swapped rows, and the share predicting that same terminal on the
matching unswapped row, both by original hop and by whether the new `q_hop_true` ≤ L. Never enters a gate.
