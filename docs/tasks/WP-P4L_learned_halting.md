# WP-P4L — learned halting on frozen symbolic readers (study P4-S-L)

Read first: `docs/prereg/P4S-L.md` (registered design, endpoints, rules), `docs/reports/P4S-stopping.md`,
`docs/reports/P4S-stopping-r32.md`, `src/looplora/core/stopping.py`, `scripts/run_p4s_r32.py`,
`scripts/analyze_p4s_r32.py` (reader states, locked label-free thresholds, checkpoint list). Branch/worktree:
`wp/sym` (merge `master` first). Outputs: `reports/p4s-l/` (design lock, head checkpoints with blake2b, θ per
family, accuracy by family/hop/rule/seed with counts, endpoints with CIs, timing), report `docs/reports/P4S-L.md`.

## Steps
1. Design lock commit before training any head: checkpoint list and hashes, the label-free thresholds reused,
   feature definitions, generator seeds, θ grid.
2. 10% smoke (time and memory only), then head training, θ selection on validation, θ locked in a commit.
3. Confirmation (eval seed 999002) and timing; endpoints exactly as registered; report every family, per-seed
   values, and any departure from the prereg.

## Limits
≤ 1 node·h; each job ≤ 8 nodes × 1 h; record node·h per job. Final line: `WP-P4L done: docs/reports/P4S-L.md`.
