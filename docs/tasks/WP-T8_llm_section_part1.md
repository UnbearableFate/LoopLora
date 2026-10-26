# WP-T8 — LLM section, part 1: writer screens and the P1-L protocol; manuscript wording

Read first: `docs/reports/P1L-pilot1.md` (WP-L1), `docs/reports/P1L-diagnosis.md` (WP-L2),
`docs/reports/P1L-writer-fix.md` (WP-L3), `docs/reports/P1L-read-fix.md` (WP-L4, merged 39f7f0b) with their
`reports/` directories, `docs/reports/WP-C6.md`, `docs/reports/WP-C7.md`, `docs/prereg/P1L.md` (v2 and
Amendments B1–B5), plan T-04/T-05 LLM row, `docs/paper/draft.md`, `docs/paper/evidence_symbolic.md`.
Branch/worktree: `wp/theory` (merge `master` first). Report: `docs/reports/WP-T8.md`.

## Deliver
1. `docs/paper/evidence_llm.md`, same format as the symbolic table: claim rows for the LLM writer results
   (information present at the input positions; output path usable with oracle answer codes; key–query matching
   not formed at r = 32; W5/R/Rρ/RK; OA/OAρ; the r = 128 arms; read-distortion diagnostics), numbers checked
   against the committed CSV/JSON, with scope (single seed, validation only, unequal step budgets) and a
   must-not-say list (for example: no claim that learned writing is impossible in LLMs).
2. Draft §8 (LLM): protocol (memory-only, layer-14 `o_proj`, Qwen3-0.6B-Base revision, full-vocabulary EM), the
   writer screens and their negative outcome with the read-distortion diagnosis, and the `oracle_random` P1-L
   design at r = 32 and r = 128 (same-rank gates, k = 2, 97.5% CIs, per-step readout). Result paragraphs are
   explicit placeholders until the P1-L confirmation; no invented numbers.
3. Wording: where the draft cites "the user's decision" (FR1 in §1 and §6), state it as the study's design
   rationale (same-rank comparison at equal parameters; latency reported as a cost); keep the provenance only in
   the evidence table.

No experiments or compute. Final line: `WP-T8 done: docs/reports/WP-T8.md`.
