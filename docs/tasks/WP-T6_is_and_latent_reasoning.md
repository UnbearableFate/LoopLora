# WP-T6 — IS results into the evidence table and draft; latent-reasoning / Jacobian literature

Read first: `docs/reports/IS.md` and `reports/is/` (merged 641911d), `docs/paper/evidence_symbolic.md`,
`docs/paper/draft.md`, `docs/reports/lit_check.md`, plan E.2 row "与潜空间推理 / Jacobian 研究的联系",
`docs/prereg/JS.md`, `docs/prereg/P1L.md` (per-step readout). Branch/worktree: `wp/theory` (merge `master` first).
Report: `docs/reports/WP-T6.md`.

## Deliver
1. Evidence table: claim rows for IS-E1…E4 (numbers checked against `reports/is` CSV/JSON, set C999 = eval seed
   999001, conditions), and the Prop. 4 ledger row updated with what IS does and does not establish.
2. Draft: the memory-load result and plan §9.3 figure 4 from `reports/is` (extend `scripts/paper_figures.py`,
   render in one compute job), claim IDs as before.
3. Literature check (primary sources, same method as `lit_check.md`; record what could not be opened): Anthropic
   "Verbalizable Representations Form a Global Workspace in Language Models" (J-lens / J-space, 2026), STARS
   (arXiv 2605.26733), recurrent depth / Huginn (2502.05171), Ouro (looped LMs with learned depth), Relit
   (2608.08113), "A Survey on Latent Reasoning" (2507.06203), Relaxed Recursive Transformers, Mixture-of-Recursions,
   Coconut. For each: what it does, what overlaps with this project (reader as vertical latent recurrence,
   overshoot, depth sampling, sharing, stopping, verbalizable directions), and what it implies for our claims.
   Add them to `lit_check.md` and the draft's related-work section without novelty claims beyond the evidence.

No new experiments except the figure render. Final line: `WP-T6 done: docs/reports/WP-T6.md`.
