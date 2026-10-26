# WP-P2S-F1 — sharing at matched training depth; shortcut-free budget contrasts

Read first: `docs/prereg/P2S-F1.md` (binding), `reports/decisions/P2S.md`, `docs/reports/P2S.md`. Branch/worktree:
`wp/symf` (merge `master` first). Report: `docs/reports/P2S-F1.md`; compact outputs `reports/p2s-f1/` (≤ 1 MB in
git); raw runs under `runs/p2s-f1/`.

Reuse the P2-S drivers, statistics and audit. Smoke the new configurations (M1 at L = 4 only; `joint_fact` with
B6/B7/B8) on compute first, then pilot, lock lrs in a commit, train 24 runs, evaluate after the training barrier,
compute F-E1–F-E3 and report ≤ 1 page + tables with the prereg decision outcomes (reader and training depth for
P1-L; the shortcut-free capability claim). No latency benchmark needed. Read-only on every earlier run. One node
per job, ≤ 1 h; total ≤ 2.5 node·h. Final line: `WP-P2S-F1 done: docs/reports/P2S-F1.md`.
