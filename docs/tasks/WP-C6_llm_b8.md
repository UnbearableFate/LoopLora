# WP-C6 — LLM pipeline support for the unshared reader (B8) and fixed-depth training

Read first: `reports/decisions/P2S.md` §6 (P1-L carries B8 trained at L = 4, with M1 trained at L = 4 as the
shared variant), `src/looplora/core/adapters.py` (`UnsharedAdapter`), `src/looplora/llm/config.py`,
`src/looplora/llm/train.py` (`core_variant` mapping, writer handling), `src/looplora/sym/config.py` (B8 in the
symbolic pipeline: `unshared`, train and eval depths fixed at 4). Branch/worktree: `wp/core`; merge `master` and
then `wp/llm` (WP-L3 code lives there and will reach master later). Report: `docs/reports/WP-C6.md`.

## Task
- Add variant `B8` to the LLM pipeline: core `unshared` with `unshared_steps = 4`, train and eval depth fixed at
  4, the same writer handling as M1 (writer stage, joint and reader stages, checkpoint signatures), both
  protocols, both input sites, and every WP-L3 option that exists on `wp/llm` (they must work unchanged).
- Confirm that `train.depths: [4]` gives fixed-depth training for M1, B5 and B3 (add a test if missing).
- Tests on compute: B8 forward/backward and gradients reach every per-step R/U/Q; checkpoint save/load
  round-trip; a fixed-batch loss equals a direct `UnsharedAdapter` computation; a short memory-only training smoke
  for B8 and for M1 at fixed L = 4 with the fast path (`scan_final`, `train.prefetch_workers`).
- Measure ms/step for B8 vs M1 at L = 4 (batch 32 and 128, one process, prefetch 2).

## Limits
No changes to existing variants' numerics. Coordinate through files only: if a conflict with `wp/llm` appears,
keep `wp/llm`'s behaviour and report it. One node per job, ≤ 1 h; total ≤ 0.3 node·h. Final line:
`WP-C6 done: docs/reports/WP-C6.md`.
