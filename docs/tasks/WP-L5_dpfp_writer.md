# WP-L5 — DPFP writer screen (study DPFP)

Read first: `docs/prereg/DPFP.md` (design, arms, decision rule; binding), `docs/tasks/WP-L4_p1l_read_fix.md` and
`docs/reports/P1L-read-fix.md` (the R arm, the schedule rule, the audit), `src/looplora/core/memory.py`
(decay-first delta rule), `src/looplora/llm/writer_fix.py`, `read_fix.py`, `read_fix_run.py`, `scripts/submit.py`,
`AGENTS.md`. Worktree/branch: `LoopLora-wt/llm` on `wp/llm` (merge `master` first). Report: `docs/reports/DPFP.md`.

## Rules
- Login node: edit code and files only. Never run torch, pytest or training on the login node; every test, smoke
  and training run goes through `scripts/submit.py` on compute (available again after 2026-10-28 22:00 JST).
- No SHA-256 in generated code (blake2b for content hashes). uv for dependencies, git for state; commit often.
- No validation score is read before all three arms' step counts are fixed and committed.

## Steps
1. Now (no compute): implement DPFP as an option of the learned-writer path (keys at write, queries at read,
   φ-space similarities in the InfoNCE and diagnostics, raw-read loss on M q̂), configs for D32-3, D128-1, D128-3,
   and tests: φ_ν matches the reference definition (shape 2·d·ν, nonnegativity, homogeneity), exact delta-rule
   overwrite with normalized φ keys at β = 1, DPFP-off bitwise equality with the current code path, and a
   200-step DPFP-off reproduction of WP-L4 R's losses at r = 32 and r = 128. Commit; stop and report what is ready.
2. After 2026-10-28 22:00: run the tests on compute, then the 10% smokes (time and memory only, no scores), fix
   and commit step counts, submit the three arms (one 1-node job each), then the final audit and the report with
   every curve, diagnostics, per-job node·h and the decision under the registered rule.

Final line: `WP-L5 done: docs/reports/DPFP.md`.
