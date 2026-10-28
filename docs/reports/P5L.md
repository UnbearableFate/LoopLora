# WP-P5L - second-scale confirmation (Qwen3-1.7B-Base, oracle_random r = 128), 2026-10-29

Author: sn-p1l (Claude Sonnet 5.5), branch `wp/p1l`. Decision record: `reports/decisions/P5L.md`. Tables:
`reports/p5l/r128/results.md` and the CSVs and `endpoints.json` beside it; ledger `reports/p5l/jobs.json`; state and
walltime notes `docs/reports/P5L-state.md`; prereg `docs/prereg/P5L.md`. Raw runs: `/work/xg24i002/x10041/LoopLora/runs/p5l/`.

## Result (confirmation set, test [104000, 106000), eval seed 4242, 2000 episodes, seeds 17/29/43, 97.5% CIs)
**Registered decision: B7 passes and B6 passes, so the equal-parameter result holds at a second scale** (oracle condition,
seen names, trained hops; B7 rule 3 limits).

| Endpoint | 1.7B | 0.6B (P1-L, r = 128) |
| --- | --- | --- |
| B7 hop-1 EM minus B3-L4@4: B4@1 / B8@4 / M1-L4@4 | +30.6 [29.2, 32.2] / +28.9 [27.5, 30.4] / +58.7 [57.0, 60.4] pp: **pass** | +19.6 / +19.0 / +47.9: pass |
| B6 M1-L4@4 - B4@1, hops {2, 3} | **+25.78 pp [24.53, 27.03]** (31.96 vs 6.17%; per seed 24.9 / 26.5 / 25.9): **met** | +21.55 [20.38, 22.76] |
| G1-L B8@4 - B4@1 (reported only) | +1.03 [0.29, 1.80]; hop 1 -1.72 [-3.32, -0.02]: fail | +0.98 [0.27, 1.72]: fail |
| G2-L B8@4 above B5-L4 / B3-L4 (reported only) | +1.67 [0.91, 2.38] / +7.17 [6.58, 7.76]: pass | +0.83 / +6.34: pass |

Descriptive: M1-L4@4 accuracy by hop 58.7 / 41.5 / 22.5 / 6.2% (hops 1-4), hops 5-8 at most 3.9%; depth effect +31.5 pp,
walk specificity +28.2 pp; M1-L4 - B5-L4 +26.4 pp; B8 - M1-L4 -24.8 pp. The 0.6B pattern repeats at the larger scale:
M1-L4 composes, B8 does not beyond a small margin, and the memory-free loop stays at 0.0%.

## Cost
`reports/p5l/jobs.json`: 38 jobs, **16.12 node·h**: step 1 (tests, smokes) 0.36; 15 trainings plus B0 14.74 (projection
14.8); pre-evaluation tests 0.08; 10 evaluation jobs 0.93 (projection 1.2-3.2); test job 0.01; finalizer 0.01.

## Limits and what is not claimed
Three seeds, one oracle seed, one layout, carried lr (no pilot at 1.7B). No equivalence with the symbolic oracle, no
unseen-name claim, no learned writer, no P3, no causal claim (P1L-C was not run at 1.7B), no scale beyond 0.6B and 1.7B.
Exploratory readings are marked as such in the decision record; none enters the decision.
