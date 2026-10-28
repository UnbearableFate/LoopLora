# Decision record — P5L (second-scale confirmation on Qwen3-1.7B-Base, oracle_random r = 128), 2026-10-29

Sources: `docs/prereg/P5L.md` (v1, registered on master 99f0dae), `reports/p5l/r128/` (`results.md`, `endpoints.json`,
`memory-use.csv`, `accuracy.csv`, `swap-successors.csv`, `inference.csv`, `run-manifest.json`, `record-numbers.txt`),
`reports/p5l/jobs.json` (ledger), `docs/reports/P5L.md`, `docs/reports/P5L-state.md`. Every number is a confirmation-set
value (test split [104000, 106000), eval seed 4242, 2000 episodes, seeds 17/29/43, memory-only protocol, primary setting)
with 97.5% paired episode-bootstrap CIs (2000 draws) unless stated. Backbone `Qwen/Qwen3-1.7B-Base` @
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, adapter at layer 14 `o_proj` (2048 to 2048), writer `oracle_random` (oracle
seed 17, one memory for all readers and seeds), r = r_mem = 128, S = 17,000. Trainings pinned to `e8ae1f9` (B0 `24fcc07`),
evaluation and finalizer to `ad8efb5eab5821bae898ebfcb2838869c4de16e5`; authorization
`reports/p5l/confirmation-authorization.json`. The 0.6B values are P1-L's r = 128 (`reports/decisions/P1L.md`).

## 1. Registered decision
**B7 passes and B6 passes: the equal-parameter result holds at a second scale** (Qwen3-1.7B-Base; oracle condition, seen
names, trained hops; Amendment B7 rule 3 limits apply). `endpoints.json`: `decision = holds_at_second_scale`.

| Registered endpoint | Result | Outcome |
| --- | --- | --- |
| B7, hop-1 EM minus B3-L4@4 (0.03%), 97.5% CI lower bound > 0 for all three | B4@1 +30.62 [29.17, 32.23] pp (30.65%); B8@4 +28.90 [27.52, 30.43] (28.93%); M1-L4@4 +58.67 [56.98, 60.40] (58.70%); every seed positive | **pass** |
| B6, M1-L4@4 minus B4@1 on hops {2, 3}: at least 3 pp, CI > 0, positive in every seed | **+25.78 pp [24.53, 27.03]** (31.96% vs 6.17%); per seed 24.93 / 26.50 / 25.92; equal adapter parameters (606,210 each) | **met** |

At 0.6B the same contrast was +21.55 pp [20.38, 22.76] (26.91% vs 5.36%; per seed 24.45 / 15.10 / 25.10). The 1.7B value
is larger and its seeds are tighter; this is a descriptive comparison, no test was registered across scales.

## 2. Reported only (not part of the decision)
- **G1-L: fail.** B8@4 minus B4@1 on hops {2, 3}: +1.03 pp [0.29, 1.80] (7.21% vs 6.17%), per seed 0.20 / 1.73 / 1.18, below
  the 3 pp bar. The hop-1 condition (at least -1 pp) also fails here: -1.72 pp [-3.32, -0.02] (B8 28.93% vs B4 30.65%);
  at 0.6B it held (-0.60 [-1.98, 0.77]). Typed replication +1.13 [0.27, 1.94] (CI > 0). Per the registered text this is a
  negative result for the gated reader B8, as at 0.6B (+0.98 pp).
- **G2-L: pass.** B8@4 minus B5-L4@4: +1.67 pp [0.91, 2.38] (per seed 0.55 / 3.60 / 0.88); minus B3-L4@4: +7.17 pp
  [6.58, 7.76]. Small against the fixed-query loop, large only against the memory-free loop (as at 0.6B: +0.83, +6.34).
- No claim about B8 is made beyond G1-L and G2-L as reported.

## 3. Secondary, descriptive (registered as descriptive; same definitions as P1-L)
| Contrast | 1.7B pp [97.5% CI] | Per seed 17/29/43 | 0.6B |
| --- | --- | --- | --- |
| M1-L4@4 - B5-L4@4 (hops 2, 3) | +26.42 [25.14, 27.82] | 25.27 / 28.38 / 25.62 | +21.40 |
| M1-L4@4 - B3-L4@4 | +31.92 [30.73, 33.21] | 30.95 / 33.60 / 31.20 | +26.91 |
| B8@4 - M1-L4@4 | -24.75 [-26.07, -23.45] | -24.73 / -24.77 / -24.75 | -20.57 |
| M1-L4 depth (L = 4 minus L = 1), hops {2, 3} | +31.46 [30.27, 32.63] | 30.20 / 33.32 / 30.85 | +25.92 |
| M1-L4 depth, hops {6, 7, 8} | +3.28 [2.86, 3.71] | 3.62 / 3.22 / 3.00 | +3.68 |
| M1-L4 walk specificity | +28.18 [26.84, 29.43] | 26.58 / 30.11 / 27.85 | +22.25 |

Accuracy by hop (primary, seed mean, trained L; `record-numbers.txt`): M1-L4@4 hops 1-4 = 58.7 / 41.5 / 22.5 / 6.2% and
hops 5-8 3.1-3.9% (0.6B: 47.9 / 35.4 / 18.5 / 7.2%, hops 5-8 at most 5%); M1-L4 at L = 1 is 34.1 / 0.5 / 0.5% (hop-1 per
seed 46.1 / 30.1 / 26.1%, a wide seed spread); B4@1 30.7 / 6.2 / 6.2%; B8@4 28.9 / 8.2 / 6.3%; B5-L4 28.9 / 5.2 / 5.9%;
B3-L4 0.0%. Hops 2 and 3 pooled: M1-L4 31.96%, B8 7.21%, B4 6.17%, B5-L4 5.53%, B3-L4 0.04% (typed: 34.27 / 7.06 / 5.93 /
5.73 / 0.00%). `swap_successors` follow rate of the swapped terminal at L = 4 (hops 1-3): M1-L4 25.6 / 20.1 / 16.8%
against 4.8 / 4.0 / 3.5% unswapped; B8 10.2 / 8.7 / 8.1% against 6.2 / 6.4 / 6.7%. Adapter parameters: B3-L4 540,672;
B4, B5-L4 and M1-L4 606,210; B8 753,666 (+24.3% over B4; 0.6B +31.0%). Mean inference time per memory-only question:
B4 0.49, M1-L4 0.63, B8 0.63 ms (B3-L4 0.45, B5-L4 0.51, B0 0.30).
B0 (frozen base, seed 17): memory-only 0.00% on every hop; context-retained 26.35 / 11.00 / 7.05% on hops 1-3 (all
hops 7.05-26.35%). B1 and B2 were not run (registered).

## 4. Claims (within Amendment B7 rules 2-3)
- **Supported.** At 1.7B with a clean oracle memory the readers learn to use the memory (B7 pass in every seed), and one
  shared module iterated four times composes stored associations inside the LM better than a single read at equal
  parameters (B6, +25.8 pp [24.5, 27.0], positive in every seed). The result of P1-L at 0.6B (oracle condition,
  r = 128) holds at the second scale in the registered sense. By rule 3 this shows that the reader learns the
  value-to-key lookup over the 768 seen names and composes reads inside the LM.
- **Negative, registered (reported only).** B8 (four unshared reads) fails G1-L at 1.7B as at 0.6B (+1.03 pp on hops
  {2, 3}; hop 1 -1.72 pp); G2-L passes, with a small margin over B5-L4.
- **Not supported.** Equivalence with the symbolic oracle condition; any unseen-name claim; any statement about trained
  hops beyond the ones trained (hops 1-3 trained; M1-L4 hop 4 at 6.2%, hops 5-8 at most 3.9%); any learned-writer,
  end-to-end or P3 claim (no learned condition ran; the learned-writer line has stopped); any causal claim about the
  bridge (P1L-C found none at 0.6B; it was not run at 1.7B); any claim for a scale other than 0.6B and 1.7B.

## 5. Limits
Three training seeds (the bootstrap conditions on them); one oracle seed and one task layout; one carried lr per reader
(no lr pilot at 1.7B, registered), so B8's G1-L fail is conditional on lr 0.003 and S = 17,000 and is not a statement
about the architecture in general. M1-L4 hop-1 accuracy at L = 1 varies a lot across seeds (46.1 / 30.1 / 26.1%) while the
registered contrast does not (24.9 / 26.5 / 25.9 pp). The B7 rule 3 limits above apply to every claim. Cross-scale
comparisons are descriptive. The per-step readout (J-lens) and P1L-C analyses were not part of P5L.

## 6. Cost and provenance
Ledger (`reports/p5l/jobs.json`, `--study p5l`, `--require-final`): **38 jobs, 16.12 node·h**, all exit 0 except the
three recorded test-and-fix failures (`t-p5l-backbone`, `t-p5l-backbone2`, `t-p5l-pipeline`; test bugs, fixed on the
branch): step 1 (tests and smokes) 0.36; main training 16 jobs (15 trainings and B0) 14.74; pre-evaluation tests and
checks 0.08; confirmation evaluation 10 jobs 0.93 (B8 0.19, M1-L4 0.60, B4 0.03, B5-L4 0.03, B3-L4 0.03, B0 0.04); test
job 3722023 0.01 (58 passed, no F or E); finalizer 3722052 0.01. The evaluation manifest
(`reports/p5l/r128/evaluation-manifest.json`) holds 16 complete records; each was checked against the window, the seed, the
episode count, the hops and registered loop counts, the checkpoint hash of the training manifest and source `ad8efb5`.
