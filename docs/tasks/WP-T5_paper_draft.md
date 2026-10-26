# WP-T5 — paper draft, symbolic part (T-10 part 2)

Read first: `docs/paper/evidence_symbolic.md` (the only source for numbers, claim wording and the binding
"must not make" list), plan §14 (structure §14.4, §14.2, §14.3) and §9.3 (required figures), `docs/theory.md`,
`docs/notation.md`, `docs/reports/lit_check.md`, `reports/decisions/P1S.md`, `reports/decisions/P2S.md`.
Branch/worktree: `wp/theory` (merge `master` first). Output: `docs/paper/draft.md`, `docs/paper/figures/`,
`scripts/paper_figures.py`, report `docs/reports/WP-T5.md`.

## Deliver
1. An English paper draft in Markdown following plan §14.4: problem, related work and boundaries, method, theory,
   mechanism experiments, budget experiments, end-to-end section, limitations and conclusion; appendix with proof
   sketches (full proofs by reference to `docs/theory.md`) and the experimental protocol (sets, seeds, CIs).
2. Every empirical sentence cites its claim ID from `evidence_symbolic.md` in an HTML comment (`<!-- S9 -->`) and
   keeps its status and conditions; "not supported" claims appear only as negative results; nothing on the
   "must not make" list appears. Negative results (Gate 3 failure, stopping, writer dependence) stay in the
   main text.
3. Theory section: the propositions the experiments use (at least 1, 2, 3, 5, 7, 10, 11), each with its
   assumptions and what it does not imply.
4. Figures from committed CSV/JSON only, plan §9.3 items 1, 2, 3, 5 and 7: quality–L by hop for one checkpoint
   (P1-S oracle and F2 factorized), hop × L heatmap, paired differences vs B5 and B3, accuracy–latency points
   with the larger-rank single reads marked (P2-S `cost.csv`), stopping accuracy vs mean latency (P4-S).
   `scripts/paper_figures.py` reads only committed files; run it in one compute job through `scripts/submit.py`
   (1 node, ≤ 10 min) and commit PDF and PNG. Item 4 (memory load × L) comes from study IS
   (`docs/prereg/IS.md`, running) and item 6 from P1-L: leave marked placeholders, as for the LLM section,
   which states only what is merged at the time (P1-L writer diagnosis and screens) and is marked TODO.
5. Main text about 5000 words.

No new experiments. Final line: `WP-T5 done: docs/paper/draft.md`.
