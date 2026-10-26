# WP-C7 — oracle writer in the LLM pipeline

Why: WP-L2/WP-L3 locate the P1-L failure in the learned LLM writer (`docs/reports/P1L-writer-fix.md`). P1-S
separated reader from writer with oracle writers; P1-L needs the same condition so that the reader (question
encoding from the LM, repeated reads, output through the LM) is tested whatever WP-L4 decides.

Read first: `src/looplora/sym/writers.py` (`OracleWriter`: global random entity codes, key = code(head),
value = code(tail), decay-first delta rule with λ = β = 1), `docs/prereg/P1L.md` (draft v2: T2 task, readers,
fast path), `docs/reports/WP-C6.md`, `src/looplora/llm/config.py`, `src/looplora/llm/train.py`.
Branch/worktree: `wp/core`; merge `master`, then `wp/llm` (WP-L3/L4 code lives there). Report:
`docs/reports/WP-C7.md`.

## Task
- Add writer `oracle_random` to the LLM pipeline, memory-only protocol: the memory is written from the episode's
  fact list (no support text, no LM features), with `OracleWriter` codes (r_mem = 32, fixed seed; codes are
  buffers, never trained). If a head can carry several relations in the P1-L T2 task (n_rel 4, head sharing 0.3),
  the key must identify (head, relation): use a fixed random code per (entity, relation) pair, and say which.
- The reader side is unchanged: u from the LM at the question's read position, every variant (M1, M1-L4, B8, B4,
  B5, B3) and the fast path (`scan_final`, `train.prefetch_workers`) must work with this writer.
- Tests on compute: the written state equals `OracleWriter`'s for the same facts (exact); no gradient reaches the
  codes; checkpoint save/load round-trip; a memory-only training smoke (100 steps) for M1-L4, B8 and B4.
- Measure ms/step for B8 and B4 with this writer (batch 64, one process, prefetch 2).

## Limits
No scientific runs; no change to existing writers' numerics; one node per job, ≤ 1 h; total ≤ 0.3 node·h.
Final line: `WP-C7 done: docs/reports/WP-C7.md`.
