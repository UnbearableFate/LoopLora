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
