# WP-IS — memory load × read depth (symbolic study IS)

Read first: `docs/prereg/IS.md` (the registered design, endpoints and rules), `docs/prereg/P2S-F1.md`,
`docs/reports/P2S-F1.md` (checkpoints, locked lrs, evaluation code you wrote). Branch/worktree: `wp/symf`
(merge `master` first). Outputs: `reports/is/` (manifests, accuracy by writer/variant/E/hop/L/seed with sample
counts, endpoints CSV with CIs and per-seed values, `t3_update` table, figure: accuracy vs L by load), report
`docs/reports/IS.md`.

## Steps
1. Design lock (commit before any evaluation or training): the exact layouts per E (validated by the generator),
   the checkpoint list with paths and blake2b hashes, the locked lrs, the `t3_update` settings and the IS-E4
   definitions, as `reports/is/design.json`. Note any change to the prereg design there and in the report.
2. Smoke at 10% (training and evaluation path for E = 36, both writers, one variant each); read only time and
   memory, then size jobs (walltime 1.5× measured).
3. Trained-at-load runs (36), then the confirmation evaluation (eval seed 999001) after every run has finished.
4. Endpoints and figure exactly as registered; report what holds per writer, per-seed values, and anything
   that departs from the prereg.

## Limits
Each job ≤ 8 nodes × 1 h; total ≤ 2 node·h; record node·h per job. Validation is not used for selection.
Final line: `WP-IS done: docs/reports/IS.md`.
