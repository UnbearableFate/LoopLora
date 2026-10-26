# WP-C3 — P4-S pilot: label-free stopping for the evolving-query reader (evaluation only)

Read first: plan §4.7, §6.3, §10.6; `docs/theory.md` Props. 3 and 7 (terminal leakage; contraction is not
correctness); `docs/reports/P1S-F1.md` and `reports/p1s-f1/accuracy_tables.csv` (oracle_random walks one hop per
read but overshoots: hop-1 accuracy falls from 97–100% at L = 1 to 64–83% at L = 4). Branch/worktree: `wp/core`
(merge `master` first). Report: `docs/reports/P4S-stopping.md`; outputs `reports/p4s/`.

## Question
Can a stopping rule computed only from the reader's own state halt each query near its chain end, recovering
short-hop accuracy without losing long-hop accuracy, and what does it cost in steps and measured latency?

## Design (no training; fixed checkpoints)
- Checkpoints: F1 `oracle_random` and `joint`, r ∈ {64, 128}, M1 and M1-lin, seeds 17/29/43
  (`runs/p1s/f1/`). Add WP-H r = 32 checkpoints only after the orchestrator says they are released.
- Criteria at read step s (all label-free): relative read norm ‖U M Q z^s‖ / ‖U M Q z^0‖; relative state change
  ‖z^{s+1} − z^s‖ / ‖z^s‖; output stability (argmax unchanged between s and s+1). Halt at the first step that
  meets the threshold, max L = 16; the prediction is read at the halting step (state the convention).
- Thresholds: one per (checkpoint family, criterion), chosen on `val` (eval seed 4242) to maximise mean accuracy
  over hops 1–8; then report on the screening `test` set (eval seed 4242). Do not use confirmation episodes.
- References: fixed L ∈ {1, 2, 4, 8, 16}; oracle stop at the true hop (upper bound, labelled as such).
- Bounds: compute the §6.3 certified c (ρ L_φ (‖R‖₂ + ‖U‖₂‖M‖₂‖Q‖₂)) per episode and report how often c < 1;
  state plainly when the bound is vacuous (Prop. 7 predicts non-contractive chain solutions).
- Latency (plan §10.6): GPU time per query batch for fixed L vs adaptive halting with masking (batch 64 and 1,
  warmup and repeats stated), since fewer mean steps need not mean lower latency.

## Report
Per family: accuracy by hop and mean steps for each rule versus fixed L and the oracle stop; whether halting
removes the hop-1 overshoot; c statistics; latency table. Tests for the halting code on compute. Each job
≤ 1 node × 1 h; total ≤ 2 node·h.
