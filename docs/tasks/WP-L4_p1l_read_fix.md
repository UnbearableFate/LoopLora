# WP-L4 — P1-L writer: read supervision, injection scale, key decorrelation (validation only)

Written 2026-10-27 02:40 JST, before WP-L3's final results; known then: A1 rerun val EM at step 6580 was 33.7%
(OA) and 8.5% (W5). Starts only if WP-L3 (with Amendment A1) selects no writer. Read first: `docs/reports/P1L-writer-fix.md`,
`docs/theory.md` Prop. 11 (§11.5: ranked follow-ups and their falsifiers), `docs/tasks/WP-L3_p1l_writer_fix.md`.
Branch/worktree: `wp/llm` (merge `master` first). Report: `docs/reports/P1L-read-fix.md`.

## Design
T1 exactly as WP-L3 (E = 16, n_rel 4, head sharing 0.3, seed 17, r = 32, layer 14, L = 1, batch 32, lr 3e-3,
val 1000 episodes at eval seed 4242), fast path (`scan_final`, `train.prefetch_workers: 2`). Each arm gets one
≤ 1 h job (two arms per node if they fit); fix the step count from a 10% smoke before any new validation score
is read; val EM at five evenly spaced checkpoints.
0. Audit (no training): Prop. 11.5 step 0 measurements on the final WP-L3 A1 checkpoints (OA, W5 and the third
   rerun arm): read coefficients, exact Gram reconstruction vs the actual read, ‖q‖, ‖v_t‖, ‖Mq‖, ‖u‖, ‖Ru‖,
   ‖UMq‖, value-head decoding of clean v_t vs actual Mq, key Gram RMS / max |cos|, answer NLL/EM with zeroed and
   shuffled memory.
Arms (base W5 = standardization + InfoNCE addressing + clean-value CE):
- R: W5 + raw-read loss ‖Mq − sg(v_t)‖² / (sg(‖v_t‖²) + ε) at the final question token, λ = 1.
- Rρ: R with ρ = 0.5 (a plan candidate).
- RK: R + key decorrelation: mean squared off-diagonal cosine over distinct addresses within an episode, λ = 1.
- OAρ: OA with ρ = 0.5, compared with A1's OA at matched steps (diagnostic).

## Decision rule (stated before results)
Order of simplicity R < Rρ < RK. The P1-L writer is the simplest arm with val EM ≥ 0.80 at its final checkpoint.
If none reaches 0.80 but the best reaches ≥ 0.50, retrain that arm once at E = 8 (same schedule) and accept it if
val EM ≥ 0.80 (P1-L then uses E = 8, flagged, as the Pilot-1 rule allows). Otherwise P1-L stops; the report gives
the audit, every curve and OAρ's outcome. No further writer screens after WP-L4.

## Limits
Val only; seed 17; no T2, reader or joint runs. Each job ≤ 1 node × 1 h (≤ 2 h under Amendment A1); WP-L4 total ≤ 8 node·h (A1); record node·h per
job. Final line: `WP-L4 done: docs/reports/P1L-read-fix.md`.

## Amendment A1 (2026-10-27 03:39 JST, after WP-L3's final report, before any WP-L4 arm was trained or scored)
Information seen: WP-L3 final curves (`docs/reports/P1L-writer-fix.md`): OA 33.7 / 55.9 / 65.2 / 70.3% val EM at
6,580 / 13,160 / 19,740 / 32,900 steps, W5 8.5 → 14.4%, both still rising at the end; WP-L4 smoke timing only.
Reason: the 8,225-step schedule ends where OA was still below 40%, so an arm that fixes the read could fail the
0.80 rule for lack of steps. Change: every arm (and the E = 8 retrain, if triggered) trains the largest step count
whose smoke-measured time × 1.5 fits one ≤ 2 h job (`scripts/submit.py --scale-up
docs/tasks/WP-L4_p1l_read_fix.md`), fixed before training, with val EM at five evenly spaced checkpoints. The
decision rule is unchanged and applies at the final checkpoint. WP-L4 total ≤ 8 node·h.

## Amendment A2 (2026-10-27 04:31 JST, user request "并行验证 r = 128"; before any WP-L4 validation score is read)
Reason: the read noise in WP-L3 (key top-1 82% but EM 14%) is what interference among E = 16 facts at r_mem = 32
predicts (Prop. 11); a larger rank lowers it. Add, in parallel with the running arms, three arms at
**r = r_mem = 128**: W5 (WP-L3's base), R (W5 + raw-read loss) and OA (diagnostic), everything else as A1 (T1 E = 16,
seed 17, batch 32, lr 3e-3 without a new search, flagged; largest step count whose smoke-measured time × 1.5 fits
one ≤ 2 h job; val EM at five checkpoints). Decision rule: the threshold is unchanged; the candidate order is
r = 32 first (R < Rρ < RK), then r = 128 (W5 < R); the E = 8 retrain branch applies to the best arm over both ranks.
The P1-L learned condition, if any, uses the selected arm at its rank. The WP-L4 node·h total is no longer capped
(user, 2026-10-27: the cap is per job); record node·h per job. This is still WP-L4, the last writer screen.
