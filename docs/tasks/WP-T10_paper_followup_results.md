# WP-T10 — follow-up results into the evidence ledgers and the draft (sn-paper; LoopLora-wt/paper, wp/paper)

Read first: `docs/tasks/_common_rules.md`, `plan/followup_2026-10-27.md` (§3 F2, F5, F8–F11 with their result lines,
§7), `docs/paper/draft.md` (Appendix C claim contract), `docs/paper/evidence_symbolic.md`, `docs/paper/evidence_llm.md`.

Text only, login node, no compute. Merge master into wp/paper first (master f47165d or later).

## Part 1 (now): studies with registered results
Sources (each report quotes its prereg; use the prereg's "Use of the results" wording, nothing stronger):

| study | report | prereg |
| --- | --- | --- |
| IVD | `docs/reports/IVD.md` (master) | `docs/prereg/IVD.md` |
| CVH | `docs/reports/CVH.md` (master) | `docs/prereg/CVH.md` |
| TSM | `docs/reports/TSM.md` (master) | `docs/prereg/TSM.md` |
| UNS | `docs/reports/UNS.md` (master) | `docs/prereg/UNS.md` |
| NCR | `docs/reports/NCR.md` (master) | `docs/prereg/NCR.md` |
| RFX | `git show faacd37:docs/reports/RFX.md` (on wp/llm, not merged yet) | `docs/prereg/RFX.md` |
| WP-L6 | `git show 98d048c:docs/reports/WP-L6.md` (on wp/oracle, not merged yet) | `docs/prereg/WP-L6.md` |

1. **Evidence ledgers.**
   - Add one row per registered result, with the gate, the interval level and the source file.
   - Symbolic rows: IVD, CVH, TSM, UNS. NCR qualifies the reading of D (rv-fable C4); fill the placeholder WP-T9 left.
   - LLM rows: RFX verdicts, and the WP-L6 stop (P1L-S not run).
   - Label descriptive rows as descriptive.
2. **Draft.** Update the sections the results touch:
   - §6 memory load: IVD qualifies the IS shrinkage per writer.
   - §7 stopping and overshoot: CVH and TSM.
   - §5 or §6: NCR, where D is read.
   - §1 and §9 scope: UNS wording.
   - §8.3: RFX.
   - §8.4 and §8.5: WP-L6 stop.

   Keep the claim contract consistent: same-rank comparisons only; "equal parameters" only for M1 against B4; latency
   is a cost. Change the abstract only where a stated result changed.
3. **Must-not-say list.** Update it where a result removes or adds a restriction. Examples: interference is
   identified only for the writers IVD passed; no latency claim for CVH arms whose latency condition failed; the
   UNS wording is limited to what the rule allows.
4. **Leave pending:** P1-L confirmation and the B7 check, P1L-C, DPFP, P1L-D. Write no result for them; keep or add
   placeholders.

Report: `docs/reports/WP-T10.md` (what changed where; any wording you could not settle from the preregs). Commit on
wp/paper; no merge to master, no push. Done line: `WP-T10 part 1 done`.

## Part 2 (only when the orchestrator says so)
P1-L confirmation with B7, P1L-C (if run), DPFP and P1L-D (if run).
