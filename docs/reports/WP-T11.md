# WP-T11 - paper revision after review round 2 (Part A) and readable LaTeX/PDF manuscript (Part B)

Worktree `wp/paper`. Master e43ccda (Part A) and a9acba2 (Part B) merged first. Text only: no training, no compute, no torch, no new
test data. Part A commit `a12ed8f`; Part B is the commit that adds this report.

## Part A (G1): changes by file

- `scripts/paper_tables_g1.py` (stdlib): reads `reports/p1l/r128/accuracy.csv` and `reports/p5l/r128/accuracy.csv` (and
  `endpoints.json`, asserted against) and prints every new number: the (hop, L) matrix for L = 1..8 and hops 1..8 with the oracle-stop
  row, the same-rank main LLM tables (B4, M1-L4, B8 at both backbones), and the seed-level statistics. Exact Fraction arithmetic,
  rounded half up. `--tex` writes the LaTeX macros used by Part B. Output committed as `docs/paper/g1_tables.txt`.
- `docs/paper/evidence_llm.md`: new rows LP21 (matrix and oracle stop), LP22 (seed-level statistics), LP23 (main table, state bytes);
  source key G1T; five new must-not-say bullets (reader cannot go beyond hop 3; B8 cannot compose; an exact store would not help;
  recurrence composes across question positions and its converse; intervals cover future seeds); one oracle seed in the limits.
- `docs/paper/draft.md`: abstract, section 1, 8.5 pointers; new 8.6 (same-rank tables, three kinds of evidence, negative-intervention
  matrix with the gate each failed, depth matrices, seed-level tables, mechanism wording); round-2 related work with the reviewer-listed
  works; section 9 limitations and conclusion; Appendix B and C rows. Claim contract and every registered stop kept.

Numbers where the CSV differs from the reviewers: the CSV wins. Examples: oracle-stop hops 4-8 are 9.9-16.6 % at 0.6B (reviewer
"10-17 %"); the reviewers' 7.5, 1.8, 7.3, 12.4 are reproduced by exact half-up rounding.

## Part B: readable PDF

- Build: `timeout 600 tectonic -X compile docs/paper/tex/paper.tex --outdir docs/paper` (login node, tectonic 0.17.0, XeTeX).
  Result `docs/paper/paper.pdf`: 54 pages, 487,075 bytes (about 476 KiB). No undefined references or citations.
- Sources in `docs/paper/tex/`: `paper.tex`, `abstract.tex`, `sec_intro.tex`, `sec_setting.tex`, `sec_oracle.tex`, `sec_writers.tex`,
  `sec_mechanism.tex`, `sec_other.tex`, `sec_limits.tex`, `sec_related.tex`, `sec_conclusion.tex`, `g1_tables.tex` (generated),
  appendices `app_A.tex` to `app_H.tex`, `refs.bib` (76 entries, `\nocite{*}`). Figures from `docs/paper/figures/`.
- Structure: short abstract and introduction (question, result, limits), one notation (Section 2), main results in sections 3-5 (oracle
  composition with the depth-by-hop matrix and seed-level tables; learned writers with the negative-intervention table; mechanism),
  other registered results, limits, related work, conclusion. Appendices: A proofs, B protocol map, C claim contract (landscape),
  D method and theory, E symbolic studies, F language-model studies, G limitations, H related work. Letters A-C keep the old
  cross-references valid; D-H keep the draft section numbers inside.
- Prose: main sections were condensed and rewritten with the /write skill (wording and structure only; every number lifted from the
  draft or ledgers). Appendices D-H are the draft sections converted mechanically by `scripts/paper_md2tex.py`, not rewritten.
- Evidence tags are kept as `% TAG` comments. `scripts/check_paper_tex.py` checks that every tex tag resolves to a ledger row, that
  every draft tag is present in the tex (108 = 108), that no en or em dash or `--` appears, that main-text numbers occur in the
  draft, ledgers or g1 tables (rounding tolerated; no hits), and that refs.bib has 76 entries as the draft has 76 reference lines.
- `docs/paper/draft.md` has a first line marking it superseded by `docs/paper/tex/paper.tex`. The ledgers are unchanged by Part B.

## Checks run

`scripts/check_paper_tex.py` (pass), the draft/ledger md checks (tags resolve, no duplicate IDs in the LLM ledger, table column counts,
relative links, no SHA-256), `python3 scripts/paper_tables_g1.py` output identical to `docs/paper/g1_tables.txt`, rerunning
`scripts/paper_md2tex.py` is idempotent, `git diff --check`, pages rendered to PNG and inspected (main text, depth matrices,
negative-intervention table, landscape Appendix C).

## Notes and limits

1. Citations are used as the reviewers recorded them. I did not re-open or independently confirm them, and added no sources. The draft
   says so. LOTUS in rv-astra is the same arXiv 2606.31779 as Fan, Svete and Lee in rv-fable; it is cited by title with the alias noted.
2. State bytes are derived (4 r^2 = 65,536 B at r = 128, FP32, formula of `src/looplora/llm/run.py`), not a recorded measurement.
3. Seed-level interval: mean of three seed effects +/- t(0.9875; df = 2) = 6.205 times the sample standard error (two-sided 97.5 %).
   B6 contrast: [+1.50, +41.60] (0.6B) and [+22.93, +28.64] (1.7B). G2-L against the memory-free reader excludes 0 at both backbones;
   all G1-L contrasts and G2-L against the fixed-query reader include 0 at both. G2-L over B5-L4@4 includes 0.
4. The B8 warm start from M1-L4 is untested (G3 is registered but not run, not anticipated here). Nothing of G2-G4 is anticipated.
5. Remaining overfull boxes are small (largest about 21 pt in a landscape Appendix C table, about 14 pt in one Appendix E table, about
   11 pt in the main seed tables); the tables render legibly. Not eliminated.
6. Appendix prose D-H is a mechanical conversion of the draft, so its tone is the draft's; only main text and lead-ins got the /write pass.
7. `scripts/analyze_dpfp_x.py` needs no `-I`.
