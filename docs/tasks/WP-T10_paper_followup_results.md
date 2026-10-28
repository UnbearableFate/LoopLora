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

## Part 2 (now): P1-L confirmation and P1L-C
Merge master into wp/paper first (master 9ceb068 or later; wp/p1l, wp/oracle and wp/llm are merged).
Sources:
- `reports/decisions/P1L.md` (§1–§6), `docs/reports/P1L.md`, `reports/p1l/` (r = 32), `reports/p1l/r128/`,
  `reports/p1l/rank-comparison.json`;
- `docs/reports/P1L-C.md`, `reports/p1lc/`;
- preregs `docs/prereg/P1L.md` (Amendments B5–B7, with the decision-use text) and `docs/prereg/P1L-C.md`.

1. Fill LP4/LP6 and PC1 in `docs/paper/evidence_llm.md`.
   - B7 per rank.
   - r = 32 is reported as "the readers did not learn to use the oracle memory", with no interpretation.
   - r = 128: G1-L fail for B8, G2-L pass, and the Amendment-B6 equal-parameter result for M1-L4 against B4, with per-seed
     values and the parameter counts.
   - P1L-C: no causal-bridge claim; the read-position result is descriptive.
2. Update draft §8.4 and §8.5, the abstract, §1 and §9, and the claim contract.
   - Stay within B7 rules 2–3: no r = 32 multi-read claim; no symbolic-oracle equivalence; no unseen-name claim.
   - The equal-parameter claim is only M1-L4 against B4 at r = 128.
   - B8 is not parameter-matched (+31.0%).
   - Latency is a cost.
3. Repoint the RFX and WP-L6 citations to the master paths.
4. DPFP and P1L-D stay pending.

Report: append to `docs/reports/WP-T10.md`. Done line: `WP-T10 part 2 done`.

## Part 3 (only when the orchestrator says so)
### 3a (now): LP5 per-step readouts
Merge master into wp/paper first (master 5eca4d2 or later; it carries the corrected P1-L record, f4f9a98).
Source: `reports/decisions/P1L.md` §3 "Per-step readout" (numbers from `scripts/p1l_record_numbers.py`, output
`reports/p1l/record-numbers.txt`; cells in `reports/p1l/readout-pooled.csv`, `reports/p1l/per-step-readout.csv`).
1. Fill LP5 in `docs/paper/evidence_llm.md` as descriptive (no gate):
   - r = 128 logit lens: M1-L4 decodes earlier-hop entities from early states, and it does so best at a step s ≥ e
     rather than at s = e. The rates track the depth-s EM of the same reader, so the lens shows what the reader would
     answer if it stopped at step s. B8 decodes names only at s = 4.
   - J-lens contrastive, raw logit and random direction are at or near chance for both readers. r = 32 is at chance
     for every method.
2. Wording limits: decoding only, no ordered-trajectory or pointer-chasing claim (P1L-C gave no causal bridge, PC1).
   The readouts are not evidence for the LP4 claim. Remove the "readouts need their own reading" mismatch note,
   which the corrected record resolves, and cite the corrected record instead.
3. Update the draft where LP5 is cited (§8.4/§8.5 and the must-not-say list) only as far as these limits allow.
Report: append to `docs/reports/WP-T10.md`. Done line: `WP-T10 part 3a done`.

### 3b (when the orchestrator says so): DPFP negative result; P1L-D not run
Merge master into wp/paper first (the commit named in the orchestrator's message, which carries wp/llm with the DPFP
report). Sources: `docs/reports/DPFP.md`, `reports/p1l-dpfp/` (arms, arms-e8, audit files, e8-lock.json),
`docs/prereg/DPFP.md` (decision rule, Amendment A1), `plan/followup_2026-10-27.md` §F3.
1. DP1 in `docs/paper/evidence_llm.md`: not supported.
   - E16 val EM: D32-3 39.0%, D128-1 58.0%, D128-3 60.8%.
   - The registered E = 8 retrain of D128-3 (flagged) reached 73.0%, below 0.80, so no writer is selected.
   - Give the registered consequence: a negative result and no further writer screen. LSQ ran no arm (RX1), so the
     learned-writer line ends.
   - Descriptive only: read error and read-value head accuracy per arm, and the zero/shuffled-memory audit. The audit
     shows the answers depend on the episode's own memory; it does not show the read is accurate.
   - Amendment A1: DPFP changed both the feature geometry and the query normalization, so the E8 gain over E16 and the
     gain over R-r128 are not attributed to decorrelation alone.
2. PD1: not registered, because no writer was selected. State that it was not run.
3. Draft: replace the §8 "DPFP and P1L-D — TODO" paragraph, update the header pending list, §9, the abstract and §1
   only where they mention a pending writer result, and the must-not-say list. Add "no learned writer reaches the
   WP-L4 gate in this project", and remove any wording that leaves end-to-end learned use open as a result still to
   come.
   - Keep the claim contract: same-rank comparisons only; no end-to-end or learned-writer claim; latency is a cost.
4. Record the test and smoke overrun (0.63 against the registered ≤ 0.5 node·h) only if the ledger records deviations.
Report: append to `docs/reports/WP-T10.md`. Done line: `WP-T10 part 3b done`.
