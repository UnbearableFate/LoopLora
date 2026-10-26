# WP-F2 — P1-S follow-up F2: input-factorized learned writer

Read first: `docs/prereg/P1S-F2.md` (binding), `docs/prereg/P1S.md` (final v1 + A6), `docs/theory.md` Prop. 10
(10.1, 10.5), `docs/reports/P1S-pilot.md`, WP-H pipeline notes (`docs/reports/WP-H-pipeline.md` and the
WP-H-*-notes.md interfaces). Branch/worktree: `wp/symf` (merge `master` first). Report: `docs/reports/P1S-F2.md`;
outputs `reports/p1s-f2/`; runs `runs/p1s/f2/` (not committed).

## Implementation (minimal, tested)
Writer conditions `t1_frozen_fact` and `joint_fact` in `looplora.sym`: a `FactorizedWriter` with A_k (r × 2d_e)
on [e_h; e_r] and A_v (r × d_e) on e_t, initialised as the column blocks of the `LearnedWriter` A for the same
seed; keys K A_k[e_h; e_r] (normalised as now), values V A_v e_t, scalar λ, β; `u(x)` for queries uses A_k on the
query's first 2·d_e inputs. Tests: parameter count and initial A blocks equal the unfactorized writer; values are
invariant to e_h/e_r and keys to e_t (gradients and outputs); query u equals the unfactorized query u at
initialisation; save/load and freezing for `t1_frozen_fact`. Do not change the existing writers' behaviour.

## Runs
Pilot then main exactly as the prereg (15000 steps main, 5000 pilot, seeds 17/29/43, batch 64). Evaluation,
statistics and gates with the WP-H pipeline; F2 gate rows (no replication) in your F2 code, same thresholds.
Confirmation (eval seed 777001) only after every F2 run has finished training. The paired factorized −
unfactorized contrasts read WP-H's confirmation samples only after WP-H has released them (ask the orchestrator).

## Compute
One node per job, ≤ 1 h each, F2 total ≤ 8 node·h (pilot ≤ 2). Do not touch `wp/sym`, `runs/p1s/main` or F1
outputs.
