# WP-P2S — symbolic budget-matched comparisons (P2-S)

Read first: `docs/prereg/P2S.md` (binding), `docs/prereg/P1S.md`, `docs/reports/P1S-results.md`,
`docs/reports/P1S-F1.md`, `docs/reports/P4S-stopping.md` (latency protocol). Branch/worktree: `wp/symf`
(merge `master` first). Report: `docs/reports/P2S.md`; compact outputs `reports/p2s/` (≤ 1 MB in git); raw runs
under `runs/p2s/`.

## Do
1. Check that B6, B7 and B8 run end to end in the existing symbolic pipeline at the prereg sizes (short smoke on
   compute) and measure step time for the job plan.
2. Pilot (prereg), lock the lrs in a committed file, then the 24 training runs (15000 steps).
3. After the training barrier, evaluate every P2-S run and every listed reference checkpoint on the confirmation
   set (eval seed 888001) at L ∈ {0, 1, 2, 4, 8, 16} where defined; compute E1–E3 with the P1-S bootstrap code
   and E4 (parameters, state bytes, analytic FLOPs, measured latency with the WP-C3 benchmark protocol).
4. Report ≤ 1 page + tables: E1–E3 with CIs and per-seed values, the E4 table/plot, and the Gate 3 outcome under
   the prereg rule. No deviations without asking the orchestrator first.

## Limits
Do not touch `runs/p1s/main`, F1/F2 runs or released checkpoints (read only). One node per job, ≤ 1 h; total
≤ 3 node·h; record node·h per job. Final line: `WP-P2S done: docs/reports/P2S.md`.
