# Study P1L-C — causal and read-position analysis of the P1-L r = 128 readers, pre-registered

Registered 2026-10-27 09:40 JST, before any P1-L confirmation score is read (P1-L Amendment B7). Proposed by the
external review `docs/review/rv-astra_2026-10-27.md` §5. Evaluation only: no training, no change to P1-L.

## Question
At r = 128, does the answer of a multi-read reader depend on the bridge entity carried by its state after the
first read (a causal test of composition), and how much of its accuracy comes from the memory read at the final
question position versus reads at the other question positions?

## Condition to run
Runs only if r = 128 passes P1-L's memory-use check (Amendment B7, rule 1). Otherwise not run, and the report says
so.

## Fixed design
- Checkpoints: P1-L r = 128 `oracle_random` main runs of M1-L4, B8 and B4, seeds 17/29/43, as locked in
  `runs/p1l/main/r128/training-manifest.json` (nothing retrained).
- Episodes: task, protocol (memory-only), templates and oracle exactly as P1-L; split `test`, indices
  [102000, 104000), eval base seed 4242 — a range no earlier evaluation used and P1-L does not read.
- Precision, batch and backbone as the P1-L confirmation evaluation.

## Analyses
**A. Read position** (descriptive). For each reader at its trained L, full EM by hop (1–3) on all 2,000 episodes
under (a) deployed reads (every question position reads M) and (b) final-position memory: the module at every
other question position runs with M = 0. Report (b) − (a) per seed and pooled with a 95% paired episode bootstrap
CI (2,000 draws). A drop is a distribution shift, not evidence of leakage.

**B. Bridge patching** (M1-L4 and B8, L = 4, hop-2 questions). From the first 256 episodes per seed that have a
hop-2 question on the long chain, build three memories that differ only in the question's first edge
(s → b): world A (the original, s → b_A → t_A), world B (s → b_B, where b_B is the head of an edge b_B → t_B on the
other chain; all other facts, including b_A → t_A, unchanged), and world C (s → c, where c has no outgoing fact).
b_B and c are drawn with a seed derived from the episode index. Run A, but at the final question position replace
the reader state after the first read, z_1, with z_1 from the same position in world B (intended patch), in world C
(unrelated-entity control), or with z_1^A plus a random direction of norm ‖z_1^B − z_1^A‖ (random control); steps
2–4 continue with world A's memory.
- Primary measure: shift = rate(answer = t_B | B-patch) − max(rate(answer = t_B | C-patch),
  rate(answer = t_B | random patch)).
- Secondary: rate(answer = t_A) under each patch; the same with z_1 patched at every question position.

## Decision rule (stated before results)
- A **causal bridge claim** for a reader requires the pooled shift ≥ 10 pp with a 95% paired bootstrap CI > 0
  and a positive shift in every seed. Otherwise the paper makes no causal claim and reports per-step readability
  (P1-L's readout) as decoding only.
- Analysis A gates nothing; it qualifies how the readout and patching results are stated.
- Descriptive replication: M1-L4@4 − B4@1 and B8@4 − B4@1 on hops {2, 3} on this range, per seed. P1-L's claim
  rules (G1-L, Amendment B6) are not changed by it.

## Limits
Evaluation only; each job ≤ 1 node × 1 h; expected ≤ 0.6 node·h in total; node·h recorded per job. Implemented
and run by a Claude Sonnet-5.5 subagent after the P1-L confirmation evaluation. Report: `docs/reports/P1L-C.md`.
