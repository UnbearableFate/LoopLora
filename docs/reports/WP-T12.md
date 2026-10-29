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
