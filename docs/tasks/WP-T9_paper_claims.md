# WP-T9 — claim contract and paper changes from the external reviews (sn-paper; LoopLora-wt/paper, wp/paper)

Read first: `docs/tasks/_common_rules.md`, `plan/followup_2026-10-27.md` (§1, §2, §7), both reviews
(`docs/review/rv-fable_2026-10-27.md`, `docs/review/rv-astra_2026-10-27.md`, including their reference lists),
`docs/prereg/P1L.md` (Amendments B4–B7), `docs/paper/draft.md`, `docs/paper/evidence_symbolic.md`,
`docs/paper/evidence_llm.md`.

Text only; no compute. P1-L, DPFP and every follow-up study are pending: write no result for them.

1. Claim contract: a table (in `docs/paper/draft.md`, appendix) listing each principal claim with its comparator
   (same rank; "equal parameters" only M1 vs B4), writer, split, gate and evidence row.
2. Related work: add the lines the reviews identify (two-hop/multi-hop composition, test-time-regression
   memories, slot/lookup memories, retrieval in weights, looped-model overshoot). Cite only references the
   reviews list as opened/verified; mark anything else as unverified or leave it out.
3. Must-not-say additions in both ledgers and the draft: LLM oracle ≠ symbolic oracle; no LLM multi-read
   conclusion at r = 32; no unseen-entity claim; OA is a value/output ceiling, not an addressing control; B8 is not
   parameter-matched.
4. Tables and statistics: B8's parameter overhead (+8.9% r = 32, +31.0% r = 128, +69.2% symbolic) wherever B8
   appears; rename OA as the value/output ceiling; per-seed effects next to pooled ones where the ledgers have
   them; mark C777/C888 reuse; S8 rank/steps confound; leave a placeholder for the neighbour-confusion rate
   (computed by WP-S9).
5. LLM evidence ledger: add the B7 facts (training-log summary, capacity check) as a new row, labelled as
   training/validation evidence, not a result of the registered test.
6. Scope statements: closed 1,024-name vocabulary; within the trained depth; oracle and learned writers stated
   separately.
Report: `docs/reports/WP-T9.md` (what changed where, what was left out and why). Done line: `WP-T9 done`.
