# WP-T7 — P4-S-L and JS into the evidence table and draft; same-rank comparison framing

Read first: `docs/reports/P4S-L.md` with `reports/p4s-l/` (merged 9c84f4c), `docs/reports/JS.md` with `reports/js/`
(merged 1e40833), `docs/paper/evidence_symbolic.md`, `docs/paper/draft.md`, `docs/theory.md` Props. 3 and 7, plan
E.2 row "多次读取的比较基准（2026-10-27 用户决定）" and §11.6, `docs/prereg/P1L.md` Amendment B4.
Branch/worktree: `wp/theory` (merge `master` first). Report: `docs/reports/WP-T7.md`.

## Deliver
1. Evidence table: rows for the P4-S-L endpoints and JS-E1…E4, the RS reference contrasts and the ρ(J)
   summaries (numbers checked against the committed CSV/JSON; confirmation sets 999002 and 999003). Update the
   Prop. 3 and Prop. 7 ledger rows with what JS does and does not establish (pointwise radii at reached states,
   not fixed-point or contraction certificates).
2. Draft: the stopping/overshoot section with P4-S-L (post-hoc halting fails) and JS (training-time terminal
   stability reduces overshoot, does not remove it, costs longer walks); one JS figure from `reports/js`
   (extend `scripts/paper_figures.py`, render in one compute job); claim IDs as before.
3. Framing (user decision 2026-10-27): multi-read is compared with the single read at the same rank; M1-L4 has
   exactly B4's parameters, B8 +8.9% (its per-step R/U/Q); the extra read latency is a cost that is reported,
   not a claim. Check every sentence of the draft and the must-not-say list: no claim of efficiency over a
   larger-rank single read; the symbolic P2-S / P2-S-F1 larger-rank results stay as reported evidence.

No new experiments except the figure render. Final line: `WP-T7 done: docs/reports/WP-T7.md`.
