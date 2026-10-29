# WP-T12: G2–G5 results into the paper (sn-paper; LoopLora-wt/paper, wp/paper)

Plan: `plan/followup_2026-10-29.md` §5. This task has two parts. Part 1 (G2–G4) starts now. Part 2 (G5: LSQT and SQ2) is
appended here when those studies report.

Read first:
- `docs/tasks/_common_rules.md`
- `docs/tasks/WP-T11_paper_revision.md`: the manuscript, tags, checks and build rules all still apply.
- The reports and decision records:
  - `docs/reports/LSR.md` (G2)
  - `docs/reports/B8W.md` and `reports/decisions/B8W.md` (G3)
  - `docs/reports/CTC.md` and `reports/ctc/decision.json` (G4)
  - their preregs in `docs/prereg/{LSR,B8W,CTC}.md`

This is text work on committed artifacts. Merge master into wp/paper first, since the reports are on master. Read no new data
and run no training. Compute every new number with a committed stdlib script from committed CSV/JSON files, for example by
extending `scripts/paper_tables_g1.py` or adding `scripts/paper_tables_g2.py`. Never import torch on the login node.

## Part 1: G2–G4
1. **Ledger.** Add rows to `docs/paper/evidence_llm.md` with new IDs. Each row needs the source file, the registered decision
   and the exact limits as the report states them.
2. **LSR (G2)**, in the (hop, L) matrix section and its appendix.
   - Add the R2 row beside the fixed-L rows and the oracle-stop row, at both backbones.
   - State the registered result: a label-free per-question stop raised hops 4–8 over fixed L = 4 at both scales, in the
     oracle-memory condition, evaluation only.
   - Give its costs in the same sentence or the next one:
     - hop 2 falls by 10.8 / 12.3 pp;
     - hops 4–8 stay below fixed L = 8;
     - 5.6–5.7 reads per question, plus s* + 1 upper-layer passes;
     - no efficiency or latency claim.
   - R1 and R3 had no feasible threshold. Say so with the reason from the report.
3. **B8W (G3)**, in the B8 discussion.
   - Replace "its warm start is untested until G3 reports" with the registered result: a copied B8 keeps the M1-L4 solution
     within the non-inferiority margin over 2,000 steps, with Delta +0.90 pp [0.09, 1.71].
   - Keep the registered wording:
     - B8 has no equal-parameter status;
     - G1-L stays failed;
     - nothing is said about why B8 fails from a random start.
   - "untied > shared" is exploratory and is not stated as a finding.
4. **CTC (G4)**, in the writer section.
   - Use the C1 wording: "a learned parser plus a fixed code table drives the frozen composing reader".
   - p = 1, and the EM equals the oracle reference within 0.2 pp. Say why: the entities are read at their own tokens and
     the relation is constant.
   - It is not a learned writer and is not comparable with DPFP or DPFP-X. No confirmation was run.
5. **Must-not-say additions:**
   - "adaptive stopping is efficient" or "R2 improves every hop";
   - "untying improves B8";
   - "CTC is a learned writer" or "the model learns its own codes";
   - any CTC number placed beside the DPFP numbers as if the two were comparable.
6. **Abstract, introduction and limits.** Update them only where these results change a sentence. Keep every registered stop
   statement.
7. **Build.** Rebuild `docs/paper/paper.pdf` with tectonic as in WP-T11 Part B. Use the `/write` skill for new prose.

## Checks and report
- The tag check (`scripts/check_paper_tex.py`), the ledger checks and `git diff --check` must pass. The new numbers must be
  reproduced by the committed script.
- Write the report as `docs/reports/WP-T12.md`, with the page count and the PDF size. Commit on wp/paper.
- STOP with `WP-T12 part 1 done` and the commit hash. No merge, no push.

## Part 2: G5 (SQ2 and LSQT), appended 2026-10-29 15:54 JST
Merge master (875108d or later) into wp/paper first. Sources:
- `docs/reports/SQ2.md`, `reports/p1l-sq2/decision.json` and `docs/prereg/SQ2.md` (with Clarification C1);
- `docs/reports/LSQT.md`, `reports/p1l-lsqt/decision.json` and `docs/prereg/LSQT.md`;
- `plan/followup_2026-10-29.md` §6.

The same rules as Part 1 apply: committed stdlib scripts only, no torch on the login node, and no new data.

1. **Ledger.** Add the SQ2 and LSQT rows with new IDs, each with its source file, its registered decision and its exact limits.
2. **SQ2**, in the writer section beside DPFP and DPFP-X.
   - State the registered result: the primary contrast S-RC − S-DE does not advance. The EM gain is +4.1 pp where +10 is
     needed, and the read-error ratio is 0.554 where at most 0.5 is needed.
   - No cell reaches 0.80, so the learned-writer line stays stopped.
   - Give the four cells (EM and read error) and the parent D128-3.
   - These are descriptive only, and each needs its own limit next to it:
     - Ridge storage on the DPFP features halves the read error and is +6.9 pp over the continued delta control. This is one
       seed, and validation was used to select D128-3.
     - The delta continuation alone gains 4.6 pp, so the parent had not converged.
     - The coefficient loss did not help: the interaction is −0.4 pp.
     - Swapping only the store at step 0 gives 68.7 %.
     - The gradient clip is active at every logged step.
3. **LSQT**, in the same section.
   - State the registered result: the best ridge arm, LT-RG001, reaches 46.1 %. That is inside the 40 % to < 48.5 % band, so
     it neither falsifies the line nor triggers W3, and the trained-LSQ line stays open.
   - Give all three arms against R-r128 (33.5 %) and D128-3 (60.8 %).
   - Report that the registered prediction (RG ≥ 50 % with read error ≤ 0.35) did not hold.
   - Make no store-versus-query attribution: the unit query is in every LSQT arm and not in R-r128. The U control is being
     prepared. Write the sentence so that it says the split is untested. Part 3 will replace it when U reports.
4. **Must-not-say additions:**
   - "ridge storage solves (or removes) the read distortion";
   - "the coefficient loss helps";
   - "trained least-squares storage works" or "is falsified";
   - any SQ2 cell or LSQT arm called a writer candidate or a selected method;
   - any attribution of LSQT's gain to the store before U reports.
5. **Abstract, introduction and limits.** Update them only where G5 changes a sentence. The learned-writer stop statement
   stays.
6. **Build.** Rebuild the PDF as in Part 1. Use the `/write` skill for new prose.

Checks as in Part 1. Append a "Part 2" section to `docs/reports/WP-T12.md` with the page count and the PDF size, and commit
on wp/paper. STOP with `WP-T12 part 2 done` and the commit hash. No merge, no push.

## Part 3: the LSQT U control, appended 2026-10-29 19:35 JST
Merge master (37abbfc or later; the U control merged at b60f144) into wp/paper first. Sources:
- `docs/reports/LSQT.md`, section "U control", and `docs/reports/LSQT-state.md`;
- `reports/p1l-lsqt-u/split.json`, `record` and `audit.json`;
- `docs/prereg/LSQT.md`, decision 2 at registration (U is descriptive and enters neither outcome);
- `plan/followup_2026-10-29.md` §6.

The same rules as Parts 1 and 2 apply: committed stdlib scripts only, no torch on the login node, and no new data.

1. **Ledger.** Add one row with a new ID for the U control. Give its source, its role (a descriptive control, one seed, not a registered test, outside both registered outcomes) and the exact limits from the report.
2. **Replace the LQ3 sentence** in `sec_writers.tex` ("…cannot be split between the store and the unit query; that split is untested"). The new text gives the descriptive split from `split.json`:
   - U reached 36.0 % with read error 0.589, below 0.80. Its final checkpoint is the registered selection and also its highest.
   - Point estimates of RG001's +12.6 pp over R-r128: +2.5 pp for the unit query (U − R-r128, standard error 2.1) and +10.1 pp for the store (RG001 − U, standard error 2.2).
   - Put these limits next to the numbers:
     - one seed;
     - the standard errors cover only the sampling of the 1,000 episodes, unpaired;
     - the split goes through U, and no ridge store with a learned query was trained, so an interaction cannot be separated;
     - the step counts differ. Add the matched-fraction fact: RG001 is above U at all five fractions, and already at 22,404 steps.
   - State that the registered LSQT decision is unchanged.
3. **Table.** Add U as a descriptive row and the split to the G5 tables. Extend `scripts/paper_tables_g5.py` and regenerate `docs/paper/g5_tables.txt` and `docs/paper/tex/g5_tables.tex`. The committed script must reproduce every new number.
4. **Must-not-say.**
   - Remove Part 2's "any attribution of LSQT's gain to the store before U reports". Allow the split only as the descriptive one through U, with its limits.
   - Add:
     - "the store causes (or explains) LSQT's gain" without the through-U and one-seed limits;
     - "the unit query helps" or "does not matter" as a finding. +2.5 pp with standard error 2.1 separates from neither zero nor a few pp.
     - U called a registered test or an outcome of LSQT.
5. **Abstract, introduction and limits.** Change them only where U changes a sentence.
6. **Build.** Rebuild the PDF as in Part 1. Use the `/write` skill for the new prose.

Checks as in Part 1. Append a "Part 3" section to `docs/reports/WP-T12.md` with the page count and the PDF size, and commit on wp/paper. STOP with `WP-T12 part 3 done` and the commit hash. No merge, no push.
