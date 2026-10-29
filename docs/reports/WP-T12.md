# WP-T12 part 1 - G2 (LSR), G3 (B8W) and G4 (CTC) results in the ledger, the LaTeX manuscript and the PDF

Worktree `wp/paper`, master `28cd028` merged first. Text only: no training, no compute, no torch, no data read beyond committed
CSV/JSON. Part 2 (G5: LSQT and SQ2) is not started and nothing in this commit anticipates it.

## What changed

- `scripts/paper_tables_g2.py` (stdlib): reads `reports/lsr/`, `reports/b8w/results/` and `reports/ctc/` and prints every number the paper
  uses (output `docs/paper/g2_tables.txt`; `--tex` writes the LSR table `docs/paper/tex/g2_tables.tex`). It asserts the registered
  outcomes (R1 and R3: 0 of 25 grid values feasible at both backbones; R2 passes; B8W `keeps`; CTC `candidate`, arm CE, p = 1, perfect
  lookup equals the oracle memory) and checks its key values against the figures the three reports state (a mismatch stops it).
  Rounding is half up on the stored value, so a cell ending in an exact 5 (for example 64.55) prints 0.1 above the LSR report (64.5);
  the same applies to a few B8W intervals (28.08 against 28.07). This is stated in the ledger, the script and Appendix I.
- `docs/paper/evidence_llm.md`: rows LS1-LS4 (LSR), BW1-BW3 (B8W), CT1-CT3 (CTC); populations V-L12-V-L14; source keys LSRE, B8WE, CTCE, G2T;
  a scope statement; three must-not-say bullets (adaptive stopping or R2 efficiency; untying improves B8; CTC as a learned writer or
  beside DPFP); the B8 bullet now states the G3 result instead of "warm start untested"; an audit-boundary paragraph. No status or value of
  an earlier row changed.
- `docs/paper/tex/`: new subsection 3.4 "A label-free stop at inference" with the R2 row beside the fixed-L and oracle-stop rows at both
  backbones (Table 8); a B8 warm-start paragraph in 3.2; new subsection 4.4 "A learned parser with a fixed code table (CTC)"; the negative
  matrix sentence about the warm start; abstract, introduction, claim contract and limits, stops and open items, conclusion; new
  Appendix I (full tables of the three studies); rows in Appendix B (protocol map) and C (claim contract); Appendix F and G no longer say the
  warm start is untested. `paper.tex` inputs `g2_tables.tex` and `app_I.tex`. `scripts/check_paper_tex.py` now also draws numbers from
  the G2 tables.
- `docs/paper/draft.md` is the superseded record (WP-T11 Part B) and was not extended; the new tags exist in the ledger and the .tex only.
- `docs/paper/paper.pdf` rebuilt with `timeout 600 tectonic -X compile docs/paper/tex/paper.tex --outdir docs/paper`: 60 pages,
  517,170 bytes (about 505 KiB), no undefined references or citations.

## Where each registered statement stands

- LSR: R2 (logit-lens stability) raised hops 4-8 over fixed L = 4 at both scales (+3.90 [3.35, 4.43] and +5.68 [5.09, 6.25] pp), in the
  oracle-memory condition, evaluation only. Costs in the same paragraph: hop 2 falls by 10.83 and 12.27 pp, hops 4-8 stay below fixed L = 8,
  5.60 and 5.66 reads per question plus s* + 1 upper-layer passes, R2 not timed, no efficiency or latency claim. R1 and R3: no feasible
  threshold (0 of 25), with the reason from the report.
- B8W: "its warm start is untested" replaced by the registered result (Delta +0.90 [0.09, 1.71] pp, every seed within 2 pp). Registered
  wording kept (no equal-parameter status, G1-L stays failed, nothing about the random-start failure). "Untied > shared" is stated as
  exploratory, not a finding.
- CTC: C1 wording; p = 1 and EM equal to the oracle reference within 0.2 pp, with the reason (entities read at their own tokens, relation
  constant); not a learned writer, not comparable with DPFP or DPFP-X, no confirmation run. No CTC number sits beside a DPFP number.
- Must-not-say additions are in the ledger and in the "does not claim" list of Section 6. Every registered stop statement is kept.

## Checks run

`scripts/check_paper_tex.py` (118 tex tags resolve to ledger rows; all 108 draft tags present; no number outside draft, ledgers and generated
tables; no en or em dash; refs.bib 76 = 76), the ledger md check (tags resolve, 5 columns in every new row; its duplicate-ID list contains only
the known reader and amendment names), `scripts/paper_tables_g2.py` (its internal assertions and report cross-checks), the Punctuation Gate on
the new prose (only LaTeX-option false positives), `git diff --check`, pages 9-10 and 55-60 rendered and inspected.

## Notes and limits

1. The LSR matrix of Table 8 uses the LSR range [106000, 108000), so its L = 4 cells differ slightly from the confirmation-range matrices
   of Tables 6-7 (hop 1 48.0 against 47.9); the text says so and the two are not pooled.
2. The LSR report rounds exact ties by float arithmetic (for example R2 hop 1 54.2); the paper rounds half up (54.3).
3. Remaining overfull boxes are small (largest about 21 pt in a landscape Appendix C table, 14 pt in one Appendix E table, 12 pt in an
   Appendix I table); the tables render legibly.
4. All three studies are one oracle seed, three training seeds (CTC: one seed) and the 768 seen names; the paper says so next to each.

# WP-T12 part 2 - G5 (SQ2 and LSQT) in the ledger, the LaTeX manuscript and the PDF

Master `f2c438d` merged first. Text only: no training, no compute, no torch, no data read beyond committed CSV/JSON.

## What changed

- `scripts/paper_tables_g5.py` (stdlib): reads `reports/p1l-sq2/`, `reports/p1l-lsqt/` and the two DPFP reference records (D128-3, R-r128), prints every number
  the paper uses (output `docs/paper/g5_tables.txt`; `--tex` writes `docs/paper/tex/g5_tables.tex`), asserts the registered outcomes (SQ2 does not advance, no candidate;
  LSQT outcome no_candidate, band open, trained-LSQ line open, prediction not held) and checks its key values against the figures the two reports state.
  Differences in pp come from the integer correct counts out of 1,000.
- `docs/paper/evidence_llm.md`: rows SQ1-SQ3 (SQ2) and LQ1-LQ3 (LSQT), population V-L15, source keys SQ2E, LSQTE, G5T, a scope statement, five must-not-say bullets, the
  exact-store bullet updated (LSQT has run), an audit-boundary paragraph. No status or value of an earlier row changed; the RX1 and DPFP statements that no trained LSQ arm
  ran were true when entered, and the ledger says so.
- `docs/paper/tex/`: SQ2 and LSQT paragraphs with Tables 11 and 12 in Section 4.2, next to DPFP and DPFP-X; two new rows in the negative-intervention table; the "three kinds of evidence"
  row, abstract, introduction, claims and "does not claim" lists, stops and open items and conclusion updated; Appendix B and C rows; Appendix I extended to G5
  (trajectories, diagnostics, limits); statements that "no trained least-squares arm ran" or that the LSQ screen ran no arm are now scoped to the RFX rule (Sections 4.2 and 4.3,
  Appendices C, F and G). `scripts/check_paper_tex.py` also draws numbers from the G5 tables.
- `docs/paper/paper.pdf` rebuilt with `timeout 600 tectonic -X compile docs/paper/tex/paper.tex --outdir docs/paper`: 64 pages, 536,171 bytes (about 524 KiB), no undefined references.

## Registered statements

- SQ2: the primary contrast S-RC against S-DE does not advance (+4.1 pp where +10 is needed, read error ratio 0.554 where at most 0.5 is needed); no cell reaches 0.80, so the learned-writer
  line stays stopped. The four cells (EM and read error) and the parent D128-3 are in Table 11. Each descriptive reading has its limit next to it: ridge storage on the DPFP features
  halves the read error and is +6.9 pp over the continued delta control (one seed, validation set that selected D128-3); the delta continuation alone gains 4.6 pp (parent not converged);
  the coefficient loss did not help (interaction -0.4 pp); the store swap alone at step 0 gives 68.7%; the gradient clip is active at every logged step.
- LSQT: best ridge arm LT-RG001 46.1%, inside the 40% to below 48.5% band, so it neither falsifies the line nor triggers W3 and the trained-LSQ line stays open. All three arms against R-r128 (33.5%)
  and D128-3 (60.8%) are in Table 12; the registered prediction (RG at least 50% with read error at most 0.35) did not hold. The store-versus-query split is stated as untested (the unit query
  is in every LSQT arm and in none of R-r128); Part 3 replaces that sentence when the U control reports.
- Must-not-say additions are in the ledger and in the "does not claim" list of Section 6. The learned-writer stop statement stays.

## Checks run

`scripts/check_paper_tex.py` (124 tex tags resolve; all 108 draft tags present; no number outside draft, ledgers and generated tables; no dash; refs.bib 76 = 76), the ledger md check
(tags resolve; known reader-name duplicates only), `scripts/paper_tables_g5.py` (internal assertions and report cross-checks), the Punctuation Gate on the new prose (LaTeX-option false positives only),
`git diff --check`, pages 12-13 rendered and inspected.

## Notes and limits

1. SQ2 and LSQT are one seed on the 1,000 validation episodes that selected D128-3; the standard error of a difference is about 2.2 pp, and the paper says so beside each reading.
2. Costs of the two studies (2.18 and 5.10 node-hours with the shared tests) are not printed in the paper; they are in the study reports.
3. Remaining overfull boxes are small (largest about 21 pt in a landscape Appendix C table, 14 pt in one Appendix E table, 12 pt in one Appendix I table).
4. The U control of LSQT has not reported and is not mentioned beyond the sentence that the split is untested.

## Part 2 fix (review wording)
- The early dip of the SQ2 ridge arms is now described as consistent with starting from AdamW moments of another objective (registered deviation 2), cause not tested
  (`sec_writers.tex`, ledger row SQ2). "Because" and "explains" are removed.
- W3 is no longer called a retraining: it is a separate ridge-plus-DPFP study that would need its own registration (`sec_writers.tex`, ledger row LQ1); no other place calls it a retraining.

# WP-T12 part 3 - the LSQT U control (descriptive split) in the ledger, the LaTeX manuscript and the PDF

Master `ff77119` merged first (the U control is on master at `b60f144`). Text only: no training, no compute, no torch, no data read beyond committed CSV and JSON.

## What changed

- `scripts/paper_tables_g5.py` (stdlib): now also reads `reports/p1l-lsqt-u/` (`split.json`, `audit.json`, `arms/LT-U/`) and the R-r128 record with its five checkpoint metrics. It recomputes the split of each arm's gain over R-r128 through LT-U
  from the integer correct counts and the unpaired binomial standard error (the formula of `scripts/analyze_lsqt_u.py`), and asserts every value against `split.json` and the figures the report states: LT-U 360 correct (36.0%, read error 0.589,
  clean head 93.7%, read head 21.6%, NLL 3.382), its five EM and read-error checkpoints (the final one is the highest), the parts add up exactly, the matched-fraction differences, the sizes of the parts in standard errors (1.2, 1.1, 4.6, 3.0),
  RG001 above U at all five fractions and at 22,404 steps, RG1 at 22,596 steps, the audit (2 and 0 correct under zero and shuffled memory, original equal to the record), the gradient clip (493 of 493), the compute facts, and that the recorded
  configurations of LT-U and R-r128 differ only in `model.unit_query` and the step count. A mismatch stops it. `--tex` now writes `\lsqttable` with the LT-U row, a new `\lsqtsplittable` (main text) and a new `\lsqtmatchtable` (Appendix I).
  Outputs `docs/paper/g5_tables.txt` and `docs/paper/tex/g5_tables.tex` are regenerated.
- `docs/paper/evidence_llm.md`: new row LQ4 (the U control: its numbers, the split with standard errors, the matched fractions, and the limits); LQ3 status moved from `open` to `narrowed` (see note 1); population V-L15 and the G5 section
  introduction extended; source key LSQUE added and G5T updated; the scope statement for the G5 rows; must-not-say: the Part 2 bullet that left the split untested is replaced by "the store causes (or explains) LSQT's gain" without the
  through-U and one-seed limits, and two bullets are added ("the unit query helps" or "does not matter" as a finding; LT-U called a registered test, a registered outcome or a result that changes the decision); a Part 3 header paragraph and
  an audit paragraph. LQ1, LQ2 and every other row are unchanged.
- `docs/paper/tex/sec_writers.tex`: the LQ3 sentence "that split is untested" is replaced by a new paragraph "LSQT: the U control and the split of the gain" and Table 13 (`tab:lsqt-split`). The U row is added to Table 12. The limits sit next to the numbers:
  one seed, each part a difference of two single evaluations; unpaired standard errors that cover the sampling of the 1,000 episodes only; no ridge store with a learned query, so no interaction is separated (the store part is measured with the
  unit query in place, the unit-query part with the delta store); different step counts, with the matched-fraction fact (RG001 above U at all five fractions, +7.7 to +14.1 pp, and 46.9% at 22,404 steps against U's final 36.0% at 24,560; RG1 also above U
  at all five, but its final lead rests on its last interval). The registered LSQT decision is stated as unchanged. The unit-query part (+2.5 pp, standard error 2.1, about 1.2 standard errors) is stated as separated neither from zero nor from a few pp.
- `docs/paper/tex/sec_limits.tex`: the "does not claim" item on the LSQT gain now allows only the descriptive split through U with its limits, and a new item says the unit query is neither shown to help nor to not matter; the one-seed sentence names the U control.
  `app_B.tex` (LSQT design row), `app_C.tex` (claim row, comparator and evidence cells) and `app_I.tex` (U rows in the trajectory table, a "The U control" paragraph, Table 41 of matched-fraction differences, the limits) are updated.
- Abstract, introduction and conclusion are unchanged: none of them mentions the unit query or the split, so U changes no sentence there. The negative-intervention table row for LSQT is unchanged (LT-U is a control without a criterion of its own).
- `docs/paper/paper.pdf` rebuilt with `timeout 600 tectonic -X compile docs/paper/tex/paper.tex --outdir docs/paper`: 66 pages, 544,151 bytes (about 531 KiB), no undefined references or citations (64 pages and 536,238 bytes before).

## Where each registered statement stands

- LSQT decision (LQ1) unchanged: no arm reaches 0.80, the best ridge arm LT-RG001 (46.1%) lies in the open band 40% to below 48.5%, the trained-LSQ line stays open, W3 is not triggered, the learned-writer line stays stopped.
- U control (LQ4, descriptive): LT-U 36.0%, read error 0.589, below 0.80; its final checkpoint is the registered selection and also its highest. It is a conditional descriptive control (registration decision 2), not a registered test, and it enters neither outcome.
- Split (LQ3, narrowed): LT-RG001's +12.6 pp over R-r128 is +2.5 pp for the unit query (standard error 2.1) and +10.1 pp for the store (standard error 2.2); LT-K4 +4.8 = +2.5 + 2.3; LT-RG1 +9.1 = +2.5 + 6.6. Limits as listed above.
- Must-not-say: "the store causes (or explains) LSQT's gain" without the limits; "the unit query helps" or "does not matter" as a finding; U called a registered test or an outcome of LSQT.

## Checks run

`scripts/check_paper_tex.py` (125 tex tags resolve; all 108 draft tags present; no number outside draft, ledgers and generated tables; no dash; refs.bib 76 = 76), the ledger md check (tags resolve, columns match in every table, links and anchors resolve; its
duplicate-ID list holds only the known reader and amendment names), `scripts/paper_tables_g5.py` (all internal assertions and report cross-checks pass), the Punctuation Gate (`--lang en`) on every added line of the tex sources and the ledger (ok),
`git diff --check` (clean), pages 13, 14, 62 and 63 rendered and inspected.

## Notes and limits

1. LQ3 changes status (`open` to `narrowed`). Part 2 said the split was untested until U reports and that Part 3 would replace the sentence, so the placeholder row is updated rather than left `open` beside a sentence that now states a split. The header
   paragraph of the ledger declares it. If the coordinator prefers LQ3 to stay `open`, only that cell and the matching words in the Part 3 header paragraph change.
2. The split uses the final checkpoints as registered; LT-RG001's 80% checkpoint (46.9%) is used only in the matched-fraction fact. The sizes of the parts in standard errors (1.2, 1.1, 4.6, 3.0) are printed as size references and labelled so, not as tests.
3. The standard errors are the unpaired ones of the report; the paired standard error was not computed, and training-seed variation was not measured. The two parts of one arm share the sampling error of U.
4. Remaining overfull boxes are the same small ones as in Part 2 (largest about 21 pt in a landscape Appendix C table, 14 pt in one Appendix E table, 12 pt in one Appendix I table); none was added.
5. Rounding is half up on the stored value, as in Part 2; the U trajectory and split cells show no exact-5 case.
