# WP-P1LC / P1L-C — causal and read-position analysis of the r = 128 oracle readers, 2026-10-29

Author: sn-p1l (Claude Sonnet 5.5), branch `wp/p1l`. Registered in `docs/prereg/P1L-C.md` (with Clarification C1); run under
the P1-L confirmation authorization after the r = 128 B7 pass (`reports/p1l/r128/endpoints.json`, `memory_use_passed`
true, commit `395e309`). Tables: `reports/p1lc/results.md` (+ `read-position.csv`, `replication.csv`, `bridge-rates.csv`,
`bridge-shift.csv`, `results.json`, `run-manifest.json`). Raw runs: `/work/xg24i002/x10041/LoopLora/runs/p1l/main/r128/causal/`.
Nothing was retrained: M1-L4, B8 and B4 at seeds 17/29/43, test [102000, 104000), eval seed 4242, 2000 bootstrap draws.
Source `P1LC_SOURCE` = `395e30914149898006d63cd289b20dac15d732fe` for all nine runs (the audit checked one `git_commit`
and `source_hash`, and equal episode, world and oracle-memory hashes across readers and seeds). All checks passed.

## Analysis A — read position (descriptive, gates nothing; 95% CIs)
Memory kept only at the final question position (M = 0 at the other question tokens), against the deployed reads:

| Reader | Hops 1 / 2 / 3 deployed EM | Final-position-only EM | Difference hops 1–3 pooled |
| --- | --- | --- | --- |
| M1-L4@4 | 48.4 / 34.2 / 18.3% | 5.0 / 4.3 / 1.9% | −29.9 pp [−30.8, −29.1] |
| B8@4 | 18.7 / 6.7 / 6.1% | 2.3 / 0.5 / 0.2% | −9.5 pp [−10.1, −8.9] |
| B4@1 | 20.3 / 5.4 / 5.5% | 2.8 / 0.4 / 0.5% | −9.1 pp [−9.7, −8.6] |

The drop is negative in every seed and every hop, including hop 1 of the single read. These readers, trained with
reads at every question position, do not work when only the last position reads: a distribution shift, not evidence of
leakage, and it says the deployed readers use the memory at earlier question positions too. On this fresh range the
P1-L descriptive contrasts replicate: M1-L4@4 − B4@1 (hops 2, 3) +20.81 pp [19.72, 21.85] (97.5% [19.57, 21.94]),
per seed 24.4 / 15.3 / 22.8; B8@4 − B4@1 +0.92 pp [0.28, 1.53] (97.5% [0.19, 1.61]), per seed 0.8 / 1.6 / 0.4.

## Analysis B — bridge patching (M1-L4 and B8, L = 4, 256 episodes, hop-2 questions on the long chain)
World A is the original, world B changes only the first edge's tail to b_B (the other chain's final edge b_B → t_B),
world C to an unrelated seen entity. Patch z_1 on world A's memory at the final question position:

| Reader | Deployed answer on world B follows t_B | Final-position B-patch shift | CI (95%) | Per seed | Causal bridge claim |
| --- | --- | --- | --- | --- | --- |
| M1-L4 | 37.2% (world A 7.0%) | +0.39 pp | [−0.13, 0.78] | 0.39 / 0.78 / 0.00 | **no** (needs ≥ 10 pp, CI > 0, every seed) |
| B8 | 4.7% (world A 9.4%) | 0.00 pp | [−0.52, 0.00] | 0 / 0 / 0 | **no** |

Secondary, all-position patch: M1-L4 +0.91 pp [−0.65, 2.34]; B8 −0.26 pp [−0.78, 0.00]. Rates of answering t_B under
the B-, C- and random patch are 7.6 / 7.0 / 7.0% (M1-L4, final) and 9.5 / 9.5 / 9.4% (B8, final), and t_A stays at
33% (M1-L4). So M1-L4 answers follow the memory when the memory itself carries the other bridge (world B), but
substituting z_1 alone, at the final position or at every question position, does not move the answer: there is no
evidence that the first-step state z_1 at the answer position is what carries the bridge. B8 does not follow world B
at all, in line with its not composing (P1-L G1-L fail). The result is negative for the registered causal claim; it
does not show where the bridge is carried (the A-result suggests the earlier question positions, untested).

## Jobs (all exit 0; Step C total 0.14 node·h of 41.53 in the ledger, `reports/p1l/jobs.json`, 106 jobs)
| Job | Name | node·h | Note |
| --- | --- | ---: | --- |
| 3721352 | `sm-p1l-c-smoke` | 0.008 | val [0, 64), timing only; B4/B8/M1-L4 seed 17 end to end |
| 3721357 / 3721358 / 3721359 | `ev-p1l-c-s17` / `s29` / `s43` | 0.041 / 0.041 / 0.043 | three runs each, 15 min requested, ~150 s used |
| 3721377 | `fin-p1l-c` | 0.007 | audit and statistics, status `complete` |

Walltime came from the smoke (31.25 × read-position + 16 × bridge-patch seconds ≈ 3.2 min, ×1.5 plus load → 15 min).
No interrupted runs; no deviation from the registration; `docs/reports/P1L-C-state.md` has the protocol.

## Limits
Descriptive A has no gate. B uses 256 episodes and the three seeds (shift CIs condition on them); one oracle seed and
layout. The C-patch is not norm-matched (the random patch is; norms 6.9 vs 6.7 for M1-L4). Rule 3 of Amendment B7 still
applies to every r = 128 statement; no learned writer, no unseen-name and no P3 claim.
