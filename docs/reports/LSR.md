# LSR report: label-free stop rules on the r = 128 M1-L4 checkpoints, evaluation only (WP-L9, G2; sn-core, 2026-10-29)

Registered in `docs/prereg/LSR.md` (blake2b-128 `63da9c626d699252c239cc86e149f1ad`, with Clarification C1). Checkpoints: the six registered
r = 128 M1-L4 runs, Qwen3-0.6B-Base (P1-L) and Qwen3-1.7B-Base (P5L), seeds 17/29/43, nothing retrained. Thresholds selected on validation
[1000, 3000) (`reports/lsr/thresholds.json`, locked at `47d9110`, checked by the orchestrator); test [106000, 108000), eval seed 4242, 2,000
episodes x 8 hops per checkpoint, read once under `reports/lsr/test-authorization.json`. Statistics: `scripts/analyze_lsr.py test` at the
analysis code committed before any test scan (`4a52120`), run at source `270ac8e` (job 3724253). Paired whole-episode bootstrap, 10,000 draws,
shared across seeds, hops and rules; registered intervals are 99.17% (1 - 0.05/6, six comparisons); descriptive intervals 95%.
Tables: `reports/lsr/results.json`, `endpoints.csv`, `accuracy.csv`, `stop-depth.csv`.

## Registered decision
| Rule | 0.6B | 1.7B |
| --- | --- | --- |
| R1 relative state change | **fails: no feasible validation threshold** (no test statistic computed) | **fails: infeasible** (none computed) |
| R2 logit-lens stability (no parameter) | **passes** | **passes** |
| R3 read norm | **fails: infeasible** (none computed) | **fails: infeasible** (none computed) |

R1 and R3 lost more than the 8 pp pooled (10 pp per seed) hop-1 margin against fixed L = 4 at every one of the 25 grid values, at both backbones
(`thresholds.json`). Reasons visible in the traces: at 0.6B the read ratio ||r^s||/||r^0|| never falls below 1 (median 1.0-1.1), so a threshold
below 1 never fires and one above 1 stops at s = 1; at 1.7B it grows to about 10 and the relative change stays at 0.5-0.7, so R1 and R3 stop at
s = 1 or run to 8. fable's recorded prediction (R3: hops 4-6 >= 12%, hop 1 >= 55% at 0.6B) is therefore not met by R3; it was not a gate.

R2, against fixed L = 4 (L_c, the best fixed depth under the same validation constraint, is also L = 4, so "rule minus L_c" equals "rule minus L = 4"):

| | 0.6B | 1.7B |
| --- | --- | --- |
| Hops 4-8 gain, pp [99.17% interval] | **+3.90** [3.35, 4.43] | **+5.68** [5.09, 6.25] |
| Hop 1 change, pp [99.17% interval] | +6.28 [5.09, 7.51] | +4.70 [3.48, 5.92] |
| Per-seed hops 4-8 gain (17 / 29 / 43) | +4.58 / +3.25 / +3.87 | +5.46 / +5.71 / +5.86 |
| Per-seed hop-1 change (17 / 29 / 43) | +8.90 / +3.95 / +6.00 | +7.35 / +4.40 / +2.35 |
| Conditions (gain >= 3 pp and interval > 0; hop 1 >= -10 pp; every seed) | all met | all met |
| Mean reads per question | 5.60 | 5.66 |
| Upper-layer passes (layers 15-28) per question | 6.60 (= s* + 1) | 6.66 |

By the registered use of results: a label-free per-question stop (R2) raised hops 4-8 over fixed L = 4 at each backbone, in the oracle-memory
condition (r = 128, seen names, the checkpoints' trained task), evaluation only, and so at both scales. The report must state the cost with it:
R2 reads 5.6-5.7 times per question against 4, and needs s* + 1 passes through layers 15-28 (one per lens evaluation), so there is **no
efficiency or latency claim**; R2 was not timed. The gain is not a better fixed depth: L_c = 4.

## What R2 does not do (descriptive; read before quoting the pass)
- **Hops 2 and 3 get worse.** Against L = 4: hop 2 -10.83 pp [-11.85, -9.87] (0.6B), -12.27 [-13.32, -11.23] (1.7B); hop 3 -3.28 / -3.65; hops 1-3
  pooled -2.61 [-3.18, -2.06] / -3.74 [-4.32, -3.16] (95%). The hop-1 gain (+6.3 / +4.7) does not offset it. R2 is not a Pareto improvement.
- **It is far from the oracle and from L = 8 on long hops.** Hops 4-8 pooled EM (%), 0.6B / 1.7B: L = 4 5.2 / 4.5; R2 9.1 / 10.2; L = 8 12.6 / 15.3;
  O1 (best fixed L per hop chosen on validation) 13.0 / 15.8; O2 (best test cell per hop) 13.0 / 15.8; O3 (any L in 1..8 correct, inflated by chance)
  22.6 / 25.5. R2 recovers about half of the O1 headroom on hops 4-8. R2 minus L = 8 on hops 4-8 is -3.54 pp [-3.93, -3.15] / -5.05 [-5.47, -4.62], while
  hop 1 is +34.1 / +39.4 pp above L = 8.
- **The stop is only weakly hop-dependent.** Mean reads by hop 1..8: 4.97, 5.48, 5.67, 5.76, 5.67, 5.70, 5.73, 5.86 (0.6B); 4.87, 5.46, 5.72, 5.87,
  5.80, 5.77, 5.84, 5.93 (1.7B). Stop depth is spread from 3 to 8 within each hop (`stop-depth.csv`), the earliest stop being s = 2 by construction
  (two stops at 0.6B, seven at 1.7B on hop 1). Hop 1 with R2 (54.2% / 63.9%) stays below its L = 2 peak (64.5% / 77.2%).
- A fixed L = 5 reaches hops 4-8 8.5% (0.6B) with hop 1 36.6%, close to R2's 9.1% with hop 1 54.2%. That unconstrained fixed depth is outside L_c
  (it lost more than the validation hop-1 margin), and no interval is attached to this comparison; it is in `accuracy.csv`.

## EM by hop on test (%), pooled over the three seeds
| Hop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6B L = 2 | 64.5 | 9.6 | 1.5 | 1.9 | 1.8 | 2.3 | 2.1 | 1.8 |
| 0.6B L = 4 | 48.0 | 33.8 | 18.1 | 6.1 | 4.4 | 4.9 | 5.1 | 5.5 |
| 0.6B L = 6 | 28.9 | 23.0 | 20.8 | 15.8 | 12.1 | 9.6 | 8.9 | 9.3 |
| 0.6B L = 8 | 20.2 | 16.7 | 15.5 | 13.8 | 14.4 | 13.2 | 11.5 | 10.3 |
| **0.6B R2** | 54.2 | 22.9 | 14.8 | 10.5 | 9.4 | 9.1 | 8.2 | 8.2 |
| 1.7B L = 2 | 77.2 | 10.9 | 0.8 | 1.3 | 1.2 | 1.2 | 1.3 | 1.5 |
| 1.7B L = 4 | 59.2 | 42.4 | 22.9 | 6.5 | 3.4 | 4.7 | 4.1 | 4.0 |
| 1.7B L = 6 | 36.8 | 29.9 | 28.2 | 22.4 | 13.0 | 8.5 | 7.6 | 8.2 |
| 1.7B L = 8 | 24.6 | 21.1 | 21.1 | 19.6 | 18.8 | 15.5 | 12.2 | 10.2 |
| **1.7B R2** | 63.9 | 30.1 | 19.3 | 13.9 | 11.2 | 10.0 | 8.4 | 7.5 |

Full L = 1..8 matrices per backbone and per seed, with the O1-O3 rows, are in `accuracy.csv`. R2 minus L = 4 by hop (95%): 0.6B +6.28, -10.83, -3.28,
+4.48, +4.98, +4.27, +3.08, +2.68; 1.7B +4.70, -12.27, -3.65, +7.35, +7.85, +5.32, +4.33, +3.53 (the hops 4-8 intervals lie above 0).

## Latency (descriptive, within one job, validation [1000, 1128), 1,024 questions per checkpoint; no claim)
ms per question, 0.6B / 1.7B: fixed L = 4 0.56 / 0.62; fixed L = 8 0.74 / 0.81; the L = 8 trace pass 0.81 / 0.87. R1 and R3 had no feasible
threshold, so their grouped stopped pass was not timed; R2's early exit is not timed (Clarification C1; it needs s* + 1 upper-layer passes).

## Audits and provenance
- All six test records complete, the registered window, checkpoint hashes equal to `reports/lsr/design.json`, one source commit `270ac8e`, not
  dirty, prereg hash `63da9c62`. Audits inside each scan (first two batches): traced L = 8 logits equal the plain L = 8 logits; inputs and memory
  unchanged across depths. The unit and synthetic-data tests passed on compute (jobs 3724000, 3724166, 3724188).
- No deviation from the registration beyond Clarification C1 (latency-audit tolerance and placement). The smoke observations on val [0, 64) changed no rule.

## Limits
Oracle memory, the 768 seen names, the checkpoints' trained task; evaluation only (no training claim); the intervals condition on the three trained
seeds per backbone and one oracle seed and layout, and do not cover new training seeds. One stop rule (R2) passes; R1 and R3, as defined and on the
registered grid, do not, and this study does not show that no other label-free signal works. The hops 2-3 loss and the cost in reads and upper-layer
passes must accompany any statement of the R2 result.

## Cost
0.50 node·h in total (cap 0.8): tests 0.05, smokes and diagnosis 0.08, validation 0.16, select 0.002, test scans 0.20, analysis 0.003. Per-job
table in `docs/reports/LSR-state.md`.
