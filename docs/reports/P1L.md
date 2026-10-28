# WP-P1L — P1-L confirmation (oracle_random, r = 32 and r = 128), 2026-10-29

Author: sn-p1l (Claude Sonnet 5.5), branch `wp/p1l`. Decision record: `reports/decisions/P1L.md`. Tables:
`reports/p1l/results.md` (r = 32), `reports/p1l/r128/results.md`, CSVs next to them, `reports/p1l/rank-comparison.json`,
ledger `reports/p1l/jobs.json`, reproduction `reports/p1l/README.md`. Raw runs: `/work/xg24i002/x10041/LoopLora/runs/p1l/main/`.

## Result (confirmation set, test [100000, 102000), eval seed 4242, seeds 17/29/43, 97.5% CIs)
Only `oracle_random` ran (WP-L4: no learned writer). Qwen3-0.6B-Base, adapter at layer 14, S = 17,000 steps.

| Rank | B7 memory-use | G1-L (B8@4 − B4@1, hops {2, 3}) | G2-L | Reading |
| --- | --- | --- | --- | --- |
| r = 32 | **fail**: B4 +0.03 [0.00, 0.10], B8 +0.03 [−0.05, 0.13], M1-L4 +0.12 [0.02, 0.23] pp hop-1 EM over B3-L4 | fail (−0.01 [−0.11, 0.10]) | fail | "the readers did not learn to use the oracle memory"; outcomes not interpreted; no multiple-read claim (rule 2) |
| r = 128 | **pass**: B4 +19.6 [18.4, 20.9], B8 +19.0 [17.9, 20.4], M1-L4 +47.9 [46.0, 49.7] pp, every seed | **fail**: +0.98 [0.27, 1.72] pp, below 3 pp (seed 29 +0.05) | pass: vs B5-L4 +0.83 [0.06, 1.55], vs B3-L4 +6.34 [5.76, 6.93] | interpreted (rule 3) |

At r = 128 the shared-module reader M1-L4 composes: M1-L4@4 − B4@1 on hops {2, 3} is **+21.55 pp [20.38, 22.76]**
(per seed 24.5 / 15.1 / 25.1; equal parameters, 475,138 each), so the Amendment-B6 equal-parameter criterion is met;
depth effect +25.9 pp, walk specificity +22.3 pp; accuracy by hop 47.9 / 35.4 / 18.5 / 7.2% (hops 1–4), hops 5–8 ≤ 5%.
The registered gated reader B8 (four unshared reads) does not: hops 2, 3 at 6.3% versus 5.4% for B4@1; B8 − M1-L4
−20.6 pp [−21.8, −19.4]. The claim is limited by Amendment B7 rule 3: the reader learns the value→key lookup over the
768 seen names and composes inside the LM; no equivalence with the symbolic oracle, no unseen-name claim, no learned
writer, no P3 claim. Context-retained B1/B2 at r = 32 reach 99.65–99.93% on hops 1–3 (B0, seed 17 only: 1.00 / 0.40 / 0.45%; all memory-only
cells of B0-B2 are 0.00%).

## Jobs and cost (`reports/p1l/jobs.json`, `--require-final`)
101 jobs, **41.39 node·h**, one node each, no job above 2 h requested (B2: per-job cap only). Main training: 39 jobs,
30.94 node·h (all 37 trainings complete; four cancelled pack jobs cost 0). Pilot-2: 12 jobs, 6.99. Confirmation
(Step B, this stretch): tests 3720632 (`t-p1l-c-tests`, exit 0, 0.008), pre-confirmation check 3720681 (0.006),
13 evaluation groups 2.11 node·h in total, all exit 0:

| r = 128 | Job | node·h | r = 32 | Job | node·h |
| --- | --- | ---: | --- | --- | ---: |
| B8 | 3720689 | 0.241 | B8 | 3720694 | 0.212 |
| M1-L4 | 3720690 | 0.606 | M1-L4 | 3721028 | 0.591 |
| B4 | 3720691 | 0.031 | B4 | 3721029 | 0.031 |
| B5-L4 | 3720692 | 0.033 | B5-L4 | 3721030 | 0.031 |
| B3-L4 | 3720693 | 0.029 | B3-L4 | 3721031 | 0.030 |
| | | | B1 / B2 / B0 | 3721032 / 3721033 / 3721290 | 0.117 / 0.118 / 0.040 |

Finalizers `fin-p1l-r128` 3721292, `fin-p1l-r32` 3721297, `fin-p1l-ranks` 3721299 (exit 0, 0.022 node·h together).
Earlier failures kept in the ledger: `t-p1l-swap` 3715928 (caught the swap-grouping bug, fixed and rerun 3715942) and
sizing/eval smoke jobs that exited non-zero before their fixes.

## Source and checks
Confirmation ran at `P1L_SOURCE` `419276578c16f8d191a7ccb4d2ca67ea07fdec00` (authorization
`reports/p1l/confirmation-authorization.json`: k = 2, 97.5%). The new tests (B7, P1L-C library, P1L-C statistics, gates)
passed on compute before the confirmation (3720632). Exit 0 was not taken as success: the evaluation manifests were
built with every record asserted complete and the finalizer status was read (`complete`).

## Limits and next
Three seeds, one oracle seed, one layout. B7 is a floor check at hop 1, not a causal proof that composition uses the
memory. P1L-C (causal and read-position study of the r = 128 readers; `docs/reports/P1L-C-state.md`) is not started and
waits for the orchestrator's go; it is gated on the r = 128 B7 pass, which holds.
