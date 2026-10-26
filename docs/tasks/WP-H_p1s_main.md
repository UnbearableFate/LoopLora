# WP-H — P1-S main runs, evaluation and gate statistics

Read first: `docs/prereg/P1S.md` (final version, binding; values marked ⟨pilot⟩ are fixed there),
`docs/reports/P1S-pilot.md`. Branch/worktree: `wp/sym` (merge `master` first). Report:
`docs/reports/P1S-results.md`; machine-readable outputs under `reports/p1s/`; raw run directories under
`runs/p1s/main/` (not committed). The orchestrator writes the decision record (`reports/decisions/P1S.md`).

## 1. Training matrix (all at the prereg's per-pair lr, steps, batch; `t1_frozen` uses one writer per seed)
| set | task | writers | variants | r | seeds |
| --- | --- | --- | --- | --- | --- |
| primary | `t2_chain_end`, E = E* | oracle_random, oracle_orth, t1_frozen, joint | M1, M1-lin, B5, B5-lin, B3, B4 | 32 | 17, 29, 43 |
| replication | `t2_typed` (prereg) | t1_frozen, joint | same 6 | 32 | 17, 29, 43 |
| replication | `t2_chain_end`, E = 1.5·E* (prereg) | oracle_random | same 6 | 32 | 17, 29, 43 |
| secondary | `t2_chain_end`, E = E* | oracle_random, t1_frozen, joint | same 6 | 16 | 17 |

- Seed-17 checkpoints from Pilot-2 at the selected lr may be reused for the primary and replication sets
  if their resolved config and science-relevant code are identical (record the reuse); otherwise retrain.
- `t1_frozen` writer for seed s: T1 pretraining with seed s at (r, E*) and Pilot-1's lr; shared by all
  variants of that seed through `pretrain.writer_checkpoint`. r = 16 uses Pilot-1's r = 16 lr at E*.
- r = 16 uses the r = 32 selected lr of the same (writer, variant) pair; it is reported, never gating.
- If the projected P1-S total exceeds 12 node·h, drop B4 and B5-lin from the replication sets first and
  record it.

## 2. Evaluation (every run; depths 0,1,2,3,4,6,8,12,16; hops 1–8, one query per hop per episode)
- screening test: split `test`, eval seed 4242, 2000 episodes; also `test_unseen` (same seed and count)
  and counterfactuals (`replace_key_fact`, `remove_required_fact`, `swap_memory`) on the screening set.
- confirmation test: split `test`, eval seed 777001, 2000 episodes, no counterfactuals.
- Keep sample-level files (episode, query, hop, L, answer, prediction, correct) for every set.
- Do not compute or look at any confirmation-set metric until every run in §1 has finished training
  and its confirmation samples are written. Then compute §3 in one pass.

## 3. Statistics (`looplora.sym.stats` pooled multi-seed paired bootstrap, 2000 resamples, 95% CI, by episode)
For each writer condition and each co-primary reader P ∈ {M1, M1-lin} (comparator F = B5 for M1, B5-lin for
M1-lin), on the confirmation set, with pooled 97.5% CIs for the gate decision and 95% CIs reported:
- G1: P, Acc(L=4) − Acc(L=1) on hops {2,3}: pooled estimate, CI, per-seed estimates; hop-1 pooled
  Acc(L=4) − Acc(L=1); the same depth contrast on the condition's replication setting (`oracle_orth`
  and `oracle_random` share the `oracle_random` 1.5·E* setting, as in the prereg).
- G2: P@4 − F@4 and P@4 − B3@4 on hops {2,3} (different checkpoints, paired by episode).
- Gate table `reports/p1s/gates.{json,md}`: one row per (condition, reader), every prereg criterion, pass/fail by
  the prereg thresholds, no other rule.
- Reported, not gating (same machinery): L=8 vs L=4 on hops 4–8 (M1); M1 vs M1-lin at L=4 on hops {2,3};
  B4 vs M1@1; r = 16 rows (single seed, screening set); `test_unseen`; counterfactual consistency
  (`replace_key_fact`: prediction equals the new answer; `swap_memory`: accuracy vs chance); every primary
  and replication contrast also on the screening set for seed 17.

## 4. Figures (`reports/figures/p1s/`, plan §9.3 items 1, 3, 4, 5)
Accuracy–L curves by hop for one checkpoint per condition (M1, seed-averaged with per-seed lines);
hop × L heatmaps (M1, M1-lin, B5, B5-lin, B3); load × L (E* vs 1.5·E*, oracle_random); paired-difference plot
(M1 − B5, M1-lin − B5-lin, P − B3, M1 − M1-lin with CIs) per condition.

## 5. Records and compute
- Append every run to `reports/experiment_registry.csv` with the plan §9.4 fields (commit, config hash,
  seeds, data version, parameters, state bytes, train tokens/time, eval counts, peak memory, failures).
- Each PBS job ≤ 8 nodes × 1 h; walltime ≈ 1.5 × measured. Record node·h per job and the stage total.
- Do not change prereg thresholds, sets, seeds or the selected lrs. Anything unexpected (non-finite
  loss, failed run, a set that cannot be built) goes into the report as observed; ask the orchestrator
  before deviating from this spec.
