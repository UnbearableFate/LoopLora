# WP-T4 — claims–evidence table for the symbolic stages (T-10, part 1)

Read first: plan §14 (claims, narrower acceptable results, narratives not to use), `reports/decisions/P1S.md`
(incl. §7 F2), `reports/decisions/P2S.md` (incl. §6 P2-S-F1), `docs/reports/P4S-stopping.md`,
`docs/reports/P4S-stopping-r32.md`, `docs/theory.md` (Props. 1–11), `docs/reports/lit_check.md`. Branch/worktree:
`wp/theory` (merge `master` first). Output: `docs/paper/evidence_symbolic.md` (no new experiments).

## Deliver
1. A table: one row per claim the symbolic results can support, narrow or reject, phrased as a paper sentence;
   columns: status (supported / narrowed / not supported / open), the exact numbers with CIs and the set they come
   from (confirmation seed 777001 or 888001, screening, validation), source file and commit, and the conditions
   under which it holds (writer type, rank, training depth, sharing).
2. The theory rows: each proposition used, what it predicts, and which measurement confirms or contradicts it.
3. A short list of statements the paper must not make (plan §14.3 plus anything these results rule out, e.g.
   efficiency claims after P2-S Gate 3).
4. Every number checked against the committed CSV/JSON, not copied from summaries; list any discrepancy found.
No new runs; reading committed files on the login node is fine. Final line:
`WP-T4 done: docs/paper/evidence_symbolic.md`.
