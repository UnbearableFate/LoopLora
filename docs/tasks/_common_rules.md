# Rules for every subagent task (2026-10-27)

- Until the orchestrator says step 2 may start (after the Miyabi maintenance, 2026-10-28 22:00 JST), work on the
  login node only: edit files and run static checks (`python3 -m py_compile`, `pyflakes`/`ruff` if present).
  Never import torch, run pytest, train or evaluate on a login node. Compute work goes through
  `scripts/submit.py` (see `docs/reports/P1L-state.md` for the pattern) and only in step 2.
- Each job ≤ 1 node × 1 h by default; ≤ 2 h only with `--scale-up <prereg>` where that prereg allows it. Walltime =
  measured × 1.5. Record node·h per job.
- Dependencies through uv; do not change `pyproject.toml`/`uv.lock` without saying why in your report.
- Hashes: blake2b only, never SHA-256. Seeds via `derive_seed(base, label…)`.
- Work in your own worktree and branch; commit there; do not merge into master, do not push, do not rewrite
  history. Preserve unrelated changes.
- Every new study gets a pre-registration draft in `docs/prereg/<NAME>.md` on your branch. The orchestrator
  reviews it and commits it to master before any run. Decision rules come from `plan/followup_2026-10-27.md`;
  flag any deviation instead of silently changing it. Never read test/confirmation data before its registered
  time.
- Keep the off path bitwise unchanged when you add an option (add a test that proves it).
- Final reply: short — commits, files, what was verified and how, open questions. Details go in the report file.
