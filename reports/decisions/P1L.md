# Decision record — P1-L (LLM-module stage, oracle_random only), 2026-10-29

Sources: `docs/prereg/P1L.md` (v2 + Amendments B1–B7), `reports/p1l/` (r = 32) and `reports/p1l/r128/` (r = 128)
(`results.md`, `endpoints.json`, `memory-use.csv`, CSV tables), `reports/p1l/rank-comparison.json`,
`reports/p1l/jobs.json` (ledger), `docs/reports/P1L.md`. All numbers are confirmation-set values (test split
[100000, 102000), eval seed 4242, 2000 episodes, seeds 17/29/43, memory-only protocol, primary setting) with 97.5%
paired episode-bootstrap CIs (2000 draws) unless stated. Backbone Qwen3-0.6B-Base, adapter at layer 14 `o_proj`, writer
`oracle_random` (oracle seed 17; the memory is identical across readers and seeds). Pinned source `P1L_SOURCE` =
`419276578c16f8d191a7ccb4d2ca67ea07fdec00`; authorization `reports/p1l/confirmation-authorization.json`.

## 1. Hypothesis
With a clean (oracle) memory and no learned writer, a multi-read reader (B8: four unshared reads; M1-L4: one shared
module iterated four times) composes stored associations inside the LM better than a single read (B4@1), a
fixed-query loop (B5-L4) and a memory-free loop (B3-L4): G1-L (B8@4 − B4@1 on hops {2, 3} ≥ 3 pp, CI > 0, positive in
every seed, hop 1 ≥ −1 pp, typed replication CI > 0) and G2-L (B8@4 above B5-L4@4 and B3-L4@4). Amendment B7 adds a
memory-use check before any gate is interpreted.

## 2. Configuration and cost
r ∈ {32, 128}, S = 17,000 steps for every trained reader, three seeds; lr .003 for M1-L4 at r = 32 and for
B8/M1-L4/B4/B5-L4 at r = 128, .001 for all other trained readers (locked before main training). Only the `oracle_random` condition ran (WP-L4 selected no learned
writer). Ledger (`reports/p1l/jobs.json`, `--require-final`): **101 jobs, 41.39 node·h**, one node per job, longest
request 2 h; 39 main-training jobs 30.94 node·h, pilot-2 12 jobs 6.99, the 13 confirmation evaluation jobs 2.11,
finalizers 0.02, preflight/smoke/tests/selection the rest. The 13 evaluation groups wrote complete records (22 r = 32 and 15 r = 128 runs, each asserted complete
before the finalizers).

## 3. Results
### B7 memory-use check (rule 1; hop-1 full-vocabulary EM minus B3-L4@4)
| Rank | B4@1 | B8@4 | M1-L4@4 | B3-L4@4 | Flag |
| --- | --- | --- | --- | --- | --- |
| r = 32 | 0.07% (+0.03 [0.00, 0.10] pp) | 0.07% (+0.03 [−0.05, 0.13]) | 0.15% (+0.12 [0.02, 0.23]) | 0.03% | **fail** (B8 and B4 CI cross 0) |
| r = 128 | 19.63% (+19.63 [18.42, 20.93]) | 19.03% (+19.03 [17.87, 20.37]) | 47.85% (+47.85 [46.02, 49.67]) | 0.00% | **pass** (every seed) |

### r = 32: not interpreted
G1-L fail, G2-L fail (`results.md`). Every memory reader is at the no-memory floor (hop-1 EM 0.07–0.15%, B3-L4 0.03%;
every memory-only cell of every non-context reader at every hop and depth ≤ 0.23%). By Amendment B7 rule 1 this rank is reported as **"the readers did not learn to use the oracle
memory"**; its G1-L, G2-L and B6 outcomes are not interpreted, and by rule 2 **no claim about multiple reads in the LM
at r = 32**. It is a failure of the oracle design (three random maps learned through a rank-32 adapter), not a reader
comparison. Descriptive: the context-retained controls work (`context_retained` protocol, seed mean: B1 99.75 / 99.85 /
99.65% and B2 99.78 / 99.93 / 99.68% on hops 1–3, hop 4 82.45 and 83.43%, hops 6–8 14.47 / 12.62 / 11.68% (B1) and
15.07 / 11.50 / 11.35% (B2); B0, the untrained adapter, seed 17 only, 1.00 / 0.40 / 0.45% on hops 1–3 and 0.30–1.00% on
hops 1–8; in the `memory_only` protocol B0, B1 and B2 are 0.00% on every hop), so the pipeline itself is sound.

### r = 128: interpreted
| Gate / contrast (hops {2, 3}) | Difference pp | 97.5% CI | Per seed 17/29/43 | Outcome |
| --- | ---: | --- | --- | --- |
| G1-L B8@4 − B4@1 | +0.98 (6.34 vs 5.36%) | [0.27, 1.72] | 0.92 / 0.05 / 1.98 | **fail**: below 3 pp (CI > 0 and all seeds > 0) |
| G1-L hop 1 | −0.60 | [−1.98, 0.77] | −2.30 / 1.20 / −0.70 | ok (≥ −1 pp) |
| G1-L typed replication | +1.55 | [0.77, 2.37] | 1.57 / 0.60 / 2.48 | CI > 0 |
| G2-L B8@4 − B5-L4@4 | +0.83 | [0.06, 1.55] | 0.88 / 0.10 / 1.52 | pass |
| G2-L B8@4 − B3-L4@4 | +6.34 | [5.76, 6.93] | 6.53 / 5.67 / 6.83 | pass |
| B6 M1-L4@4 − B4@1 | **+21.55** (26.91 vs 5.36%) | [20.38, 22.76] | 24.45 / 15.10 / 25.10 | meets ≥ 3 pp, CI > 0, every seed |
| M1-L4@4 − B5-L4@4 | +21.40 | [20.22, 22.55] | 24.40 / 15.15 / 24.65 | secondary |
| M1-L4@4 − B3-L4@4 | +26.91 | [25.76, 27.98] | 30.05 / 20.72 / 29.95 | secondary |
| B8@4 − M1-L4@4 | −20.57 | [−21.78, −19.36] | −23.5 / −15.1 / −23.1 | secondary |
| M1 depth (L4 − L1), hops {2, 3} | +25.92 | [24.82, 27.05] | 29.55 / 18.52 / 29.70 | secondary |
| M1 depth, hops {6, 7, 8} | +3.68 | [3.21, 4.18] | 3.83 / 3.83 / 3.37 | secondary |
| M1 walk specificity | +22.25 | [21.16, 23.37] | 25.72 / 14.69 / 26.33 | secondary |

Accuracy by hop (primary, seed mean, trained L): M1-L4@4 hops 1–4 = 47.9 / 35.4 / 18.5 / 7.2%, hops 5–8 ≤ 5%; M1-L4
at L = 1 is 37.8 / 1.1 / 0.9%; B4@1 19.6 / 5.4 / 5.4%; B8@4 19.0 / 6.7 / 6.0%; B5-L4 17.3 / 5.4 / 5.6%; B3-L4 0.0%.
Typed replication, M1-L4@4 − B4@1 on hops {2, 3}: 28.5 vs 5.2% (point estimates; the finalizer computes a typed CI
only for the gate contrast).
Descriptive (screening, no gate): `swap_successors` follow rate of the swapped terminal at L = 4 (hops 1–3) is
20.45 / 17.10 / 14.08% for M1-L4 against 5.82 / 4.75 / 4.82% unswapped, and 8.12 / 7.60 / 6.83% for B8 against
5.82 / 5.78 / 6.72%.

Per-step readout (descriptive, J-lens-inspired endpoint of the prereg; `reports/p1l/readout-pooled.csv`, all cells;
numbers from `scripts/p1l_record_numbers.py`, output in `reports/p1l/record-numbers.txt`). Top-1 rate of the hop-e entity
when the state z_s (s = 1…4) is decoded at the answer position, seed-pooled over 2000 episodes per seed; chance is about
0.1% (mean rank about 512 of 1024 names). At r = 128:
- *Logit lens*, M1-L4: hop-1 entity (e = 1) 38.6 / 64.3 / 61.1 / 47.9% at s = 1…4 on hop-1 questions (per-seed maximum 73.3%
  at s = 2), and 37.6 / 54.7 / 37.4 / 22.5% on hop-2 questions; hop-2 entity 1.1 / 11.0 / 30.2 / 35.4% (hop-2
  questions); hop-3 entity 1.0 / 1.4 / 7.5 / 18.5% (hop-3 questions). The s = 4 rows match the deployed EM within 0.1 pp; the early states
  carry the earlier-hop entities, and the hop-e entity is decoded best at a step s ≥ e rather than at s = e. These rates
  track the exact-match accuracy of the same reader at depth s (M1-L4 at L = 2, 3 hop-1 EM 64.15 / 61.28%), so the lens
  shows what the reader would answer if it stopped at step s. The largest single-seed cell for e = 1, s = 1 is 51.8%
  (hop 6 questions).
- *Logit lens*, B8: only the last step decodes names: e = 1 at s = 4 is 19.1 / 19.3 / 18.4% (hops 1–3), e = 2 at s = 4 is
  6.6 / 6.9%, e = 3 is 6.1%; steps 1–3 are 0.05–0.22% (chance); the maximum cell is 19.3% (single seed 20.1%).
- *J-lens contrastive*, *raw logit* and the *random direction*: at or near chance for both readers at every cell (maximum
  pooled cell 0.27% / 0.28% contrastive, 0.27% / 0.08% raw logit, 0.10% / 0.22% random for M1-L4 / B8; mean ranks
  476–509 for contrastive and raw logit, 500–526 for random).
At r = 32 every cell of every method is at chance: maximum pooled cell 0.27% / 0.20% (logit lens M1-L4 / B8), 0.22% /
0.22% (contrastive), 0.17% / 0.25% (raw logit), 0.02% / 0.08% (random); the logit-lens mean ranks (382–431) are only
modestly below the 512 of chance. r = 128 vs r = 32 (secondary,
`rank-comparison.json`): B8@4 hops {2, 3} +6.24 pp [5.64, 6.84], per seed 6.4 / 5.6 / 6.7.

## 4. Claims (within Amendment B7 rules 2–3)
- **Supported.** At r = 128 with a clean oracle memory the readers learn to use the memory (B7 pass in every seed), and
  one shared module iterated four times composes stored associations inside the LM: M1-L4@4 beats the single read
  at equal parameters (B6, +21.6 pp [20.4, 22.8], positive in every seed; both have 475,138 adapter parameters), the
  fixed-query loop and the memory-free loop on hops {2, 3}, with a depth effect (+25.9 pp) and walk specificity
  (+22.3 pp). By rule 3 this shows that the reader learns the value→key lookup over the 768 seen names and composes
  reads inside the LM; it does not establish equivalence with the symbolic oracle condition, and no unseen-name claim
  is made.
- **Negative, registered.** The gated reader B8 (four unshared reads) fails G1-L at r = 128: +0.98 pp [0.27, 1.72] on
  hops {2, 3}, below the 3 pp bar (seed 29 +0.05). Per the decision-use text this is a negative LLM result for the
  condition. Its G2-L pass is small against B5-L4 (+0.83 pp [0.06, 1.55]) and large only against the memory-free B3-L4.
  B8 has +31.0% parameters over B4 (B6), so its contrast is not an equal-parameter one; B8's hops 2, 3 accuracy stays near 6%
  (B4@1 5.4%), versus M1-L4 35.4 / 18.5%.
- **Narrowed.** Unshared reads (B8) did not learn to compose in 17,000 steps while the shared module did; this is a
  difference between two readers of one fixed oracle memory, not a statement about the architecture in general.
- **Not supported.** Any multiple-read claim at r = 32 (rule 2); any learned-writer or end-to-end claim (no learned
  condition ran; WP-L4 selected no learned writer); any equivalence to the symbolic oracle or unseen-name claim
  (rule 3); P3 is not planned from this stage (registered rule: P3 only if the learned condition passes).

## 5. Limits
Three training seeds; the bootstrap conditions on them (M1 − B4 per-seed 24.5 / 15.1 / 25.1 pp, large but varying).
One oracle seed (17) and one task layout. The oracle's random key and value tables make hops ≥ 2 a learned lookup;
r = 32 cannot learn it (B7). Only hops up to about 3 compose in M1-L4 (hop 4 at 7.2%, hops 5–8 ≤ 5%; depth effect on
hops 6–8 is +3.7 pp). B7 is a floor check at hop 1, not a proof that composition uses the memory (that is P1L-C,
registered, not started). The causal and read-position study P1L-C runs only after the orchestrator's go.

## 6. P1L-C follow-up (2026-10-29; `docs/reports/P1L-C.md`, `reports/p1lc/`)
Run after the r = 128 B7 pass, test [102000, 104000), one source (`395e309`). Memory at the final question position only
collapses every reader (M1-L4 hops 1–3 −29.9 pp [−30.8, −29.1], B8 −9.5, B4@1 −9.1), a distribution shift that shows the
deployed readers use earlier question positions. The P1-L contrasts replicate (M1-L4 − B4 +20.81 pp [19.72, 21.85]; B8 − B4
+0.92 pp [0.28, 1.53]). Bridge patching gives **no causal bridge claim** for either reader (final-position shift M1-L4
+0.39 pp [−0.13, 0.78], B8 0.00; registered bar 10 pp): the M1-L4 answer follows world B when the memory changes (37.2%
t_B) but not when z_1 alone is replaced. The Claims in §4 are unchanged; "composition inside the LM" is supported
behaviourally, and the causal-bridge reading is not.
