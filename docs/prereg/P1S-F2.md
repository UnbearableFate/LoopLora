# P1-S follow-up F2 — input-factorized learned writer (pre-registered 2026-10-26 22:11 JST, commit 74ad11c)

Registered before any WP-H, F1 or F2 confirmation-set metric has been computed (WP-H and F1 report only
training progress at this time). Deviation from the P1-S stop rule's "one follow-up": recorded as P1S.md A6.

## Motivation
In the validation pilots (`docs/reports/P1S-pilot.md`) every learned-writer variant, B4 included, scored 41–50%
on every hop with flat depth profiles, the signature of guessing among chain terminals. `docs/theory.md`
Prop. 10 shows why a learned writer can do this: its value v = V A [e_h; e_r; e_t] sees the head, so a common-key
read can approximate Σ(e_terminal − e_start) (telescoping); oracle writers store pure entity codes and cannot.
Removing head and relation information from values (and tail information from keys) removes that direct
mechanism (Prop. 10.5), though not every possible shortcut. F2 tests whether learned codes then compose. It is
the symbolic precondition for the P1-L writer design (Prop. 10.6), since P1-L can only use learned writers.

## Design (everything not listed is as in `docs/prereg/P1S.md` final v1, including the WP-H stage length)
- Writer `fact` (input-factorized): keys and queries from A_k [e_h; e_r] (query input [e_s; e_r + e_hop]),
  values from A_v e_t; k = K A_k[e_h; e_r] normalised, v = V A_v e_t, scalar λ, β as before. At initialisation
  A_k and A_v equal the head/relation and tail column blocks of the original A (same seed), so parameter count,
  state size and arithmetic equal the unfactorized writer. Effective maps satisfy VA_h = VA_r = 0 and KA_t = 0.
- Conditions: `t1_frozen_fact` (factorized writer trained on `t1`, n_rel = 4, head sharing 0.3, E = 12, then
  frozen; one writer per seed) and `joint_fact`. r = r_m = 32. Task `t2_chain_end`, E = 12, chains (8,4).
- Variants M1, M1-lin, B5, B5-lin, B3, B4; seeds 17, 29, 43; **15000 steps** for every stage (the WP-H stage
  length fixed by the P1-S budget rule, `reports/p1s/training_budget.json`), batch 64.
- Pilot (validation only, seed 17, 5000 steps, lr ∈ {3e-4, 1e-3, 3e-3}): T1 writer lr by val single-hop accuracy;
  reader lr per (condition, variant) by val accuracy on hops 1–3 at L = 4 (B4: L = 1), ties → smaller lr.
- Evaluation as WP-H §2 and its Addendum 1 (screening seed 4242 with counterfactuals and `swap_successors`;
  `test_unseen`). Confirmation: split `test`, eval seed **777001**, 2000 episodes (the WP-H set, so cross-family
  contrasts are paired by episode), not computed until every F2 run has finished training.

## Endpoints (97.5% pooled paired bootstrap, 2000 resamples, as P1-S)
- Per (condition, reader P ∈ {M1, M1-lin}), F = B5 / B5-lin: G1 = Δ_reach ≥ 3 pp with CI > 0 and positive in each
  seed, walk specificity Δ_reach − Δ_far CI > 0, hop-1 L = 4 − L = 1 ≥ −1 pp; G2 = P@4 − F@4 > 0 and P@4 − B3@4 > 0
  on hops {2, 3}. No replication setting (a weaker design than P1-S; stated with every claim).
- Reported: factorized − unfactorized (WP-H, same condition and reader) at L = 4 on hops {2, 3} and in Δ_reach,
  paired by episode; `swap_successors` follow rates by new-path length; the T1 factorized writer's single-hop
  accuracy; screening and `test_unseen` values.

## Decision use
- A factorized condition passes for some reader → P1-L adopts the LLM analog of factorization (Prop. 10.6: key
  from the position before the tail, value from the raw tail-token embedding), subject to WP-L2.
- Both fail while an oracle condition passes in WP-H → factorization alone does not make learned codes compose
  at this scale; P1-L is not started and claims stay restricted to the reader with oracle codes.

## Budget
≤ 8 node·h (pilot ≤ 2, main ≤ 6); every job ≤ 1 node × 1 h.

## Post-result note (2026-10-26 23:59 JST)
Result: both conditions fail G1 on hop-1 preservation only. The second decision-use bullet is not applied; the
reason and the resulting decision are recorded in `reports/decisions/P1S.md` §7. The registered rule text above
is unchanged.
