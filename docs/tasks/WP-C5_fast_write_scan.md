# WP-C5 — exact fast write path for write-only support passes (core)

Read first: `src/looplora/core/memory.py` (`scan_prewrite`, `write`, `delta_update`),
`src/looplora/core/wrapper.py` (`write_only` mode), `src/looplora/llm/residual.py`, `docs/reports/P1L-pilot1.md`
(throughput table). Branch/worktree: `wp/core` (merge `master` first; it must contain WP-L2). Report:
`docs/reports/WP-C5.md`.

## Problem
P1-L memory-only training writes the support with `AdaptedLinear` in `write_only` mode, which calls
`scan_prewrite`: a Python loop over every support token with a host sync (`nonzero`) per token and one stored
snapshot per token, although only the final state is used and only write tokens change M (`delta_update` leaves
rows with write = 0 untouched). Training is overhead-bound: episodes/s grow almost in proportion to batch
(16 / 32 / 61 at batch 8 / 16 / 32, one process), so a step costs ≈ 0.5 s at any batch.

## Task
- Add `memory.scan_final(...)` with the `scan_prewrite` arguments, returning only the final state, numerically
  equivalent to `scan_prewrite(...)[1]`: gather each row's write positions (valid & write) into a padded
  [B, W_max] index, then iterate W_max steps with sync-free masked updates (no per-step `nonzero`/`index_select`;
  rows without a write at a step keep their state exactly). Gather [B, T] gates the same way. Resets: choose the
  simplest exact handling (e.g. fall back to the reference scan when any reset falls inside the sequence) and test it.
- Use it in the `write_only` path of `AdaptedLinear` and of the residual-input wrapper. Keep `scan_prewrite` as the
  reference for `online` mode and tests.
- Tests (on compute): random shapes with padding, rows with zero writes, all-token writes, resets, scalar and
  [B, T] gates: final state and gradients w.r.t. u, K, V and gate parameters match the reference (fp32, rtol 1e-5,
  atol 1e-6); a P1-L memory-only loss on a fixed batch matches before/after (relative 1e-5), for both input sites.
- Benchmark (on compute): P1-L T1 training step, E = 16, `fact_final` and `all_tokens`, batch 32 / 64 / 128, one
  and two processes per GPU, before/after: ms per step, episodes/s, peak memory. Also report where the remaining
  step time goes (support forward, write, question forward/backward, optimizer).

## Limits
No change to numerics (dtypes, per-row op order), no new dependencies, no `torch.compile` unless exactness is
tested and the gain measured. One node per job, ≤ 1 h; total ≤ 0.5 node·h. Final line:
`WP-C5 done: docs/reports/WP-C5.md`.
