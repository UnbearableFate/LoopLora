# P4-S-L — learned halting on frozen symbolic readers, pre-registered

Registered 2026-10-27 03:45 JST, before any run below. Known then: WP-C3 and WP-C4 (`docs/reports/P4S-stopping.md`,
`P4S-stopping-r32.md`): no label-free rule (read norm, relative state change, output stability) keeps hop-1
accuracy within 1 pp of L1 for factorized or random-oracle codes, true-hop stopping is not an upper bound, and no
adaptive rule was faster than fixed L16. Plan §4.7 asks for a learned router compared with the residual rules.

## Question
Can a small halting head, trained on a frozen reader's own states with correctness labels, stop each query where
its answer is right — repairing the hop-1 overshoot while keeping the multi-hop gain — where label-free rules fail?

## Design
- Frozen readers (no retraining), r = 32, seeds 17/29/43: WP-C4's 18 checkpoints (`joint_fact`, `t1_frozen_fact`,
  `joint` × M1, M1-lin) and WP-H's `oracle_random` M1 and M1-lin (6). Primary families: the four factorized and the
  two `oracle_random` families (all overshoot); `joint` families descriptive.
- States: the reader run to L = 16 on each episode, recording per step s the post-update state z^s and the
  quantities the label-free rules use (read norm, relative change, output stability).
- Head per checkpoint: MLP, input [z^s; ‖z^s‖; ‖z^s − z^{s−1}‖ / ‖z^{s−1}‖; read-norm ratio; one-hot(s)], one
  hidden layer of 64 ReLU units, sigmoid output p_s. Target y_s = 1 if the candidate argmax at step s is correct.
  BCE, Adam lr 1e-3, batch 256 episodes × 16 steps, 2000 steps, fixed (no tuning).
- Training episodes: split `train`, generator seed 31337, 20000 episodes, one query per hop 1–8 (as WP-C4).
- Inference: stop at the first s ∈ [1, 16] with p_s ≥ θ, else at argmax_s p_s; θ ∈ {0.05, 0.10, …, 0.95}
  selected per family on validation (split `val`, eval seed 4242, 2000 episodes) by mean accuracy over hops 1–8
  and the three seeds, as WP-C3 selected its thresholds; locked in a commit before confirmation.
- Confirmation: split `test`, **eval seed 999002**, 2000 episodes, one query per hop 1–8 (never evaluated before).
- References on the same episodes: L = 1, 4, 16; each label-free rule at its WP-C3/C4 locked threshold; the
  label-informed **any-step oracle** (correct at some s ≤ 16; the upper bound for every stopping policy over these
  states) and true-hop stopping.
- Statistics: pooled paired episode bootstrap, 2000 draws, CI level 1 − 0.05/6 (six primary families); per-seed
  values reported.

## Endpoints (per primary family; Δ = halted − L1 unless stated)
- Hop-1 Δ (point estimate) ≥ −1 pp; reach Δ on hops {2, 3} ≥ 3 pp with CI > 0; specificity (reach Δ − far Δ on
  hops {6, 7, 8}) CI > 0. A family **qualifies** if all three hold.
- Descriptive: halted − best label-free rule on hops 1–8; gap to the any-step oracle; mean steps; CUDA ms per
  prepared batch of 64 (WP-C3 protocol) for halted vs L = 4 and L = 16.

## Use of the results (stated before results)
- A family that qualifies: "a learned halting head repairs overshoot for <family> on these frozen readers"
  enters the paper with its CI; the label-free failure stays reported.
- No family qualifies: the paper states that neither label-free nor post-hoc learned halting repairs overshoot
  in these readers, and that fixed-depth training (B8, M1-L4; P2-S-F1) is the remedy found so far.
- No gate depends on P4-S-L; no follow-up is registered from it.

## Limits
Evaluation and head training only; ≤ 1 node·h; each job ≤ 8 nodes × 1 h.
