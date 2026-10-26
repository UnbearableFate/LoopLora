# Study IS — memory load × read depth (symbolic), pre-registered

Registered 2026-10-27 03:17 JST, before any run or evaluation below. Known then: P1-S, F1, F2, P2-S and
P2-S-F1 results (`docs/paper/evidence_symbolic.md`); no load other than the trained one (E = 12, chains (8, 4))
and the P1-S oracle replication (E = 18, chains (8, 8, 2)) has been evaluated.

## Questions
Q1 (Prop. 4, same checkpoint). Does the depth benefit of a fixed checkpoint shrink as the number of stored facts
grows at fixed rank (interference amplified by repeated reads)?
Q2 (capability boundary). Trained at a given load, up to which load do several reads still beat one read?
Q3 (secondary). With updated facts (`t3_update`), do multi-read paths follow the current edge or the stale one?

## Design
- Writers: `oracle_random` and `joint_fact` (shortcut-free codes; P2-S-F1). `joint` is evaluated descriptively
  only (its terminal shortcut changes with the number of chains). r = r_mem = 32 throughout.
- Loads (task `t2_chain_end`, eval hops 1–8 where the layout has them): E = 6 (4, 2); 12 (8, 4); 18 (8, 8, 2);
  24 (8, 8, 8); 36 (8, 8, 8, 8, 4). Any change needed to make a layout valid is fixed in the design-lock commit
  below, before any evaluation.
- Same-checkpoint evaluation (no training): the E = 12 checkpoints of P2-S-F1's reference and main sets,
  seeds 17/29/43 — M1 mixed-depth (WP-H / F2; depth scan L = 1…8), M1-L4@4, B8@4, B4@1, B5@4, B3@4.
- Trained at load (new runs): E ∈ {24, 36} × writers {oracle_random, joint_fact} × {M1 mixed-depth (L ∈ {1, 2, 4}),
  B8 (L = 4), B4 (L = 1)} × seeds 17/29/43 = 36 runs, 15000 steps, batch 64, the lr locked for the same
  writer/variant at E = 12 (no new lr search; flagged deviation from the pilot rule).
- Q3: `t3_update` with base `t2_chain_end`, E = 12 (8, 4) plus the generator's default `n_updates = 2`, same
  E = 12 checkpoints, eval only.
- Confirmation: split `test`, **eval seed 999001**, 2000 episodes per load (new; never evaluated before). The
  validation split (eval seed 4242) is used only to confirm that trained-at-load runs train (no selection).
- Statistics: 97.5% CIs (two writers); paired episode bootstrap within a load, independent bootstraps per load
  across loads; 2000 draws; per-seed values reported.

## Endpoints (hops {2, 3} unless stated)
- IS-E1 (Q1): D(E) = Acc(L = 4) − Acc(L = 1) of the mixed-depth M1 checkpoint at each E; registered contrast
  ΔD = D(36) − D(12) per writer. Prop. 4 predicts ΔD < 0. Reported: D(E), the best L per E, hops {6, 7, 8}.
- IS-E2 (Q2): C(E) = Acc(M1@4) − Acc(B4@1) and Acc(B8@4) − Acc(B4@1) for models trained at E ∈ {12, 24, 36}.
- IS-E3 (descriptive): trained at E minus trained at 12 and evaluated at E, for M1@4 and B8@4 (E ∈ {24, 36}).
- IS-E4 (Q3, descriptive): accuracy at L = 1…8 on queries whose path contains an updated edge vs not, and the
  stale rate (answer = terminal reached through the obsolete edge).

## Use of the results (stated before results)
- IS-E1: ΔD < 0 with CI < 0 for a writer is reported as consistent with Prop. 4 for that writer; CI containing 0
  as no detected load dependence; ΔD > 0 (CI) as contradicting the depth-amplification prediction for these
  checkpoints. The paper's depth claims state the tested load range.
- IS-E2: the capability claim "several reads beat one read" is stated for a writer up to the largest trained E
  with C(E) ≥ 3 pp and CI > 0; beyond it, as not holding at r = 32.
- No gate depends on IS and no follow-up study is registered from it. Fills plan §9.3 figure 4.

## Limits
Each job ≤ 8 nodes × 1 h; IS total ≤ 2 node·h; node·h recorded per job.
