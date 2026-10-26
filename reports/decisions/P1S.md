# Decision record — P1-S (symbolic mechanism stage), 2026-10-26

Sources: `docs/prereg/P1S.md` (final v1 + A1–A6), `docs/reports/P1S-results.md` and `reports/p1s/` (WP-H,
merge 758c06a), `docs/reports/P1S-F1.md` and `reports/p1s-f1/` (F1, merge 7155dc1), `docs/theory.md` Props. 3,
6, 7, 10. All numbers are confirmation-set values (eval seed 777001, 2000 episodes, seeds 17/29/43) unless stated.

## 1. Hypothesis
With one module, a fixed writer and the same memory snapshot, the evolving-query multi-read reader (M1/M1-lin)
beats the single read (B4), the fixed-query loop (B5/B5-lin) and the memory-free loop (B3) on multi-hop
composition: G1 (depth effect on hops {2,3}, walk-specific, hop 1 preserved, replicated) and G2 (P > F, P > B3).

## 2. Configuration and cost
Prereg final v1; r = 32, ρ = 0.25, task `t2_chain_end` E = 12 chains (8,4); **15,000 steps** for every stage
(the preregistered budget fallback, decided from measured throughput before any main run). WP-H: 105 stages,
19 one-node jobs, **4.44 node·h**; P1-S total with pilots **8.48 / 13 node·h**. F1 (r ∈ {64,128}, 20,000 steps,
no replication): 60 runs, **4.18 / 6 node·h**. Every job ≤ 1 node × 1 h.

## 3. Results (97.5% CIs; gate = prereg thresholds)
| Writer | Reader | Gate | Δ_reach pp | Walk specificity pp | Hop-1 L4−L1 pp | P − F pp | Failed criterion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_orth | M1 | **pass** | +92.9 [92.2, 93.6] | +92.9 [92.2, 93.6] | +17.8 | +92.9 | — |
| oracle_orth | M1-lin | fail | +99.8 [99.7, 99.9] | +99.8 | −3.0 | +99.8 | hop 1 |
| oracle_random | M1 | fail | +33.3 [32.0, 34.5] | +31.5 | −18.9 | +33.9 | hop 1 |
| oracle_random | M1-lin | fail | +47.3 | +45.3 | −29.0 | +47.3 | hop 1 |
| t1_frozen | M1 | fail | +8.1 [7.4, 9.0] | +1.7 [0.6, 2.7] | +9.0 | +4.6 | replication specificity +0.7 [−0.1, 1.5] |
| t1_frozen | M1-lin | fail | +10.3 | +2.9 | +13.5 | +7.7 | replication specificity +0.2 [−0.7, 1.1] |
| joint | M1 | **pass** | +8.3 [7.4, 9.3] | +2.5 [1.2, 3.7] | +4.7 | +7.8 [6.5, 9.1] | — |
| joint | M1-lin | fail | +2.4 | −0.4 [−1.5, 0.8] | +2.8 | +3.1 | ≥ 3 pp; specificity (both) |

F1 (no replication; weaker design): `joint` passes G1 and G2 for **both readers at r = 64 and 128** (r = 64, M1:
Δ_reach +30.5 [29.2, 31.9], specificity +20.9 [19.1, 22.9], hop 1 +9.0, P − F +24.0); `oracle_random` fails only
hop 1 (−17 to −33 pp) at both ranks.

Patterns behind the gates (accuracy, seed mean):
- **Oracle codes walk one hop per read.** oracle_orth M1-lin: hops 1–3 at 97–100% at L = 4 and hops 4–6 at
  97–100% at L = 8 (trained on hops ≤ 3, L ≤ 4); B5/B5-lin and B4 answer hop 1 only (100%) and 0% beyond; B3 is
  at chance (7–8%). With no stopping, reads continue past the chain end, so short hops degrade as L grows
  (oracle_orth M1-lin hop 1: 100 → 97 → 26% at L = 1/4/8). This "overshoot" is the only failed criterion of
  four oracle rows.
- **Learned codes compose weakly at r = 32 and strongly at r ≥ 64.** joint M1 at r = 32: hops 1–3 rise from
  54/36/35% (L = 1) to 58/46/41% (L = 4); far hops stay near 40%, the terminal-guess level of Prop. 10. At r = 64
  (F1) hops 1–3 reach 94/79/61% at L = 4 and hop 4 59% at L = 8. joint M1-lin at r = 32 is flat at 45–49% on every
  hop and L (pure terminal guessing); at r = 128 it walks to hop 8 at 64% (L = 8).
- **A5 (`swap_successors`, screening, diagnostic only).** Follow rate of the new terminal vs the unswapped rate
  at L = 4, rows whose new path fits: oracle_orth M1-lin 65.6 vs 0.2%; joint M1 44.2 vs 35.3% (r = 32) and
  59.2 vs 35.8% (r = 64, all rows); joint M1-lin 48.3 vs 47.9% (r = 32): the r = 32 learned rows mostly guess.
- **t1_frozen**: single-hop pretrained writers (T1 val 87–88%) give hop-1 accuracy of 32–54% in the T2 layout and
  small depth gains that fail typed-replication specificity.
- **Uncertainty**: the bootstrap conditions on three training seeds. joint M1 − B5 is +9.3, −17.7, +31.7 pp by
  seed, so the r = 32 learned-writer pass is fragile; the r = 64 effects are consistent across seeds.

## 4. Claims
- Supported: a single low-rank module reading one fixed memory snapshot several times with an evolving query
  composes stored associations (identical x/u/M across L; oracle codes ≈ exact; learned joint codes significant at
  r = 32 for M1, large at r ≥ 64 for both readers). Fixed-query loops and single reads cannot walk beyond hop 1
  (0% on hops ≥ 2 with oracle codes); the memory-free loop is at chance (Gate 2).
- Narrowed: composable codes need **joint** training; the plan's single-hop-pretrained frozen writer (§5.2) does
  not pass. At r = 32 the learned-writer effect is small and seed-dependent; at r ≥ 64 it is robust (F1).
- Not supported at r = 32: M1-lin as a co-primary reader (fails G1 in all four conditions). The Prop. 6
  concern about ReLU did not bite at depth: M1 loses hop 1 at L = 1 with orthogonal codes (59%) but reaches 99% at
  L = 2.
- Open: no reader stops at the chain end; extra reads damage short hops (motivates P4 stopping, WP-C3 running).

## 5. Decision
**Continue** to P1-L with reader **M1** (comparator B5, plus B4 and B3), writer **joint** only (t1_frozen not
carried), memory-only protocol primary — but not before the LLM writer can learn single-hop retrieval
(P1-L Pilot-1 failed; WP-L2 diagnosis running). Rank: the F1 rule leaves r = 32 because joint M1 passes at
r = 32; given the fragile r = 32 effect, the P1-L prereg will fix r from {32, 64} by its validation pilot before
any P1-L test result (to be recorded there, with the state cost: 4 vs 16 KiB per sequence in FP32).
F2 (factorized writer, registered before these results) continues; it informs whether t1_frozen-style
pretraining or r = 32 can be rescued.

## 6. Next stage's single most important question
Can a writer trained on a real LM's features produce memory codes that the evolving-query reader composes in the
memory-only protocol (beginning with: can it store and retrieve single-hop facts at all)?
