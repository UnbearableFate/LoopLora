# WP-T10 part 1 — follow-up results in the ledgers and the draft

Worktree `wp/paper`, master `abc2fa1` merged first (already an ancestor of the branch tip). Text only: no code,
preregistration, figure or result file changed. Static checks only (no compute, no pytest): comment tags resolve to ledger
IDs, no duplicate ledger IDs, table column counts, relative links, no SHA-256, `git diff --check`; every number printed in the
draft was compared with the committed CSV/JSON of IVD, CVH, TSM and UNS (no mismatch; two UNS unseen-minus-seen oracle_random
differences are not printed on purpose).

Commits on wp/paper: `cd712d7` (symbolic ledger), `f4b7b57` (LLM ledger), `9dc0d8a` (LLM ledger: pointer row no longer repeats
the LP4 ID), `073f9b0` (draft), plus this report.

## What changed where

**[evidence_symbolic.md](../paper/evidence_symbolic.md).** New section "Follow-up studies of WP-S9": IV1 (supported, oracle_random),
IV2 (not supported, joint_fact), IV3 (descriptive); CV1 (supported, A1 oracle_random), CV2 and CV3 (not supported: A2 and the
factorized arms; latency), CV4 (descriptive); TS1 (supported: no arm passes), TS2 (descriptive); UN1 (supported: Part A audit),
UN2–UN4 (descriptive); NC2 (descriptive); PM1 moved from Pending to a filled descriptive row. Sets C999004–C999007, source keys
(IVE, IVD, CVE, CVD, TSE, TSD, UNA, UNB, UND, NCE with commits), a new "descriptive" status label defined in "Reading the tables",
discrepancy item 14, vocabulary-scope exception for UNS, IS1 cross-reference to IV1/IV2, audit-boundary paragraph. Must-not-say:
the unseen bullet is rewritten; new bullets for IVD, CVH, latency, TSM, NCR.

**[evidence_llm.md](../paper/evidence_llm.md).** Section "RFX and WP-L6": RX1 (supported: verdicts store+query at R/r128 and R/r32,
none at W5/r32, no LSQ arm), RX2, RX3 (descriptive), WL1 (supported: stop), WL2 (descriptive cost). Populations V-L6/V-L7, source
keys RFXS and WL6D (cited by commit because the records are not on master), LP6 cross-note. Section "Pending follow-up
placeholders": pointer to LP4/LP6 (P1-L confirmation, B7 check), DP1 (DPFP), PC1 (P1L-C), PD1 (P1L-D), all open with no value.
Scope and must-not-say bullets, audit boundary.

**[draft.md](../paper/draft.md).**
- Header, abstract: changed only for the halting statement (one registered CVH pass, random-oracle writer, about two executed
  updates, no latency gain) and the unseen-scope sentence (descriptive scope row on the unseen pool).
- §1 and §2 scope sentences, §9 exclusions: UNS wording ("the depth gain is also present on unseen-pool entities"; nothing beyond the
  closed vocabulary or about new codes); §9 UNS paragraph and table (Part A and Part B kept apart, never pooled).
- §5/§6: NCR replaces the PM1 TODO; reading of D qualified only; no fraction of D estimated. §6: IVD per writer (oracle_random
  identified; joint_fact not offered as identified, IS load/layout/candidate-distribution statement).
- §7: CVH subsection, TSM subsection (no arm passes; fixed-depth training stays the remedy found).
- §8.3: RFX; §8.4: WP-L6 stop and that P1L-S does not run; §8.5: DPFP, P1L-C, P1L-D TODO placeholders; Appendix B/C rows and reading rules
  extended. Claim contract kept: same-rank comparisons only, "equal parameters" only for M1 versus B4, latency stated as a cost.
- §9 must-not-say: unseen, IVD (joint_fact), CVH latency, TSM, NCR, RFX/WL6 impossibility.

## Left pending (placeholders, no result written)

P1-L confirmation and the B7 memory-use check (LP4, LP6), P1L-C (PC1), DPFP (DP1), P1L-D (PD1).

## Wording not settleable from the preregistrations (orchestrator decision wanted)

1. **CVH latency.** The prereg gives no latency gate; I state that no latency claim is made (passing arm 7.829 ms vs 3.108 ms fixed L16;
   the four arms below L16 stop at step 1 and do not pass).
2. **NCR.** The prereg fixes no threshold; "58.667–100.000% against 7.692% uniform, every lower bound above uniform" is stated
   descriptively and D is only "qualified", not corrected.
3. **UNS Part A scope.** Only the 18 audited rows are called reproduced; other committed unseen rows stay unaudited.
4. **RFX citation.** The records are at wp/llm `379fccd` (JSON) and `faacd37` (report); both are cited as printed in the branch history.
   Please confirm the intended pinned commit before merge.
5. **New status label "descriptive"** in both ledgers (rows without a gate); the existing labels were not changed.
6. **PM1 relocation.** PM1 left "Pending" and became a descriptive row; the draft tag is unchanged.
7. **Abstract.** Changed only where a stated result changed (CVH pass, unseen scope); IVD, TSM, RFX, WP-L6 are not in the abstract.
8. **RX3 reading.** The preregistered reading guide is quoted as descriptive ("write interference ... dominates" for one seed); I added
   "not an identification of the cause".

WP-T10 part 1 done

# WP-T10 part 2 — P1-L confirmation with B7, and P1L-C

Master `5d9c121` merged first (the branch already contained part 1). Text only, static checks only (tags resolve, no new duplicate
ledger IDs, table columns, links, no SHA-256, `git diff --check`; every number in the new ledger rows and in Section 8.5 was
compared with `reports/p1l/**` and `reports/p1lc/**`; derived values are the seed means and the latency means and ratios).

Commits: `4bf8f44` (LLM ledger), `ac49905` (draft), `d914079` (symbolic-ledger pointer, figure TODO), plus this report.

## What changed where

**[evidence_llm.md](../paper/evidence_llm.md).**
- New section "P1-L confirmation and P1L-C: registered results": LP7 (B7 check, supported: rank 32 fails, rank 128 passes, per-seed
  values), LP8 (rank-32 gate outcomes, descriptive and not interpreted), LP9 (B8 G1-L at rank 128, not supported: +0.983333 pp
  [0.266563, 1.716667], below 3 pp), LP10 (G2-L passes, supported, small against B5-L4), LP11 (M1-L4@4 against B4@1 at rank 128,
  supported: +21.550000 pp [20.383229, 22.758333], per seed 24.450/15.100/25.100, 475,138 parameters each), LP12 and LP13
  (descriptive: depth effect, walk specificity, accuracy by hop; parameters, latency, cost), PC1 (causal bridge, not supported: M1-L4
  +0.390625 pp [-0.130208, 0.781250], B8 0.000000, bar 10 pp), PC2 (read position, descriptive), PC3 (patching rates, descriptive),
  PC4 (replication on [102000, 104000), descriptive).
- LP4 rewritten and narrowed; LP3, LP5 and LP6 cross-referenced; PC1 and the LP4/LP6 pointer row removed from the pending section
  (DP1 and PD1 stay open); populations V-L8 and V-L9; source keys P1LE and P1LC; scope bullet; must-not-say list (B8, equal
  parameters and symbolic-oracle equivalence, causal bridge, latency, rank 32); audit boundary.
- RFXS and WL6D now cite the master paths (merged `d4de948` and `ddfcbc2`; `git diff` shows the files unchanged from `379fccd`,
  `faacd37` and `98d048c`).

**[draft.md](../paper/draft.md).** Header and abstract (the oracle-reader sentence replaces "confirmation remains pending"; only the
confirmation outcomes changed); Section 1 scope sentence and the "separate questions" sentence; Section 8 title; Section 8.3 closing
sentence; Section 8.4 (the LP6 prediction is now "reported in Section 8.5"); Section 8.5 rewritten (memory-use table per rank,
rank-128 results, P1L-C; DPFP/P1L-D and readout TODOs kept); Section 9 (new bounding paragraph, intervals, conclusion); Appendix B
rows (P1-L updated, P1L-C added); Appendix C (P1-L row rewritten, P1L-C row added, reading rules). Claim contract kept: same-rank
comparisons only, "equal parameters" only for M1-L4 against B4 at rank 128, B8 +31.0% not parameter-matched, latency a cost
(M1-L4 about 1.42 and B8 about 1.57 times B4 per question), no rank-32 multi-read claim, no symbolic-oracle equivalence, no
unseen-name claim.

**[evidence_symbolic.md](../paper/evidence_symbolic.md).** The pending-measurements pointer now names the reported P1-L and P1L-C rows.

## Left pending (placeholders, no result)

DPFP (DP1), P1L-D (PD1), and the per-step readouts (LP5; see below).

## Discrepancies and wording not settleable from the sources (orchestrator decision wanted)

1. **Decision record versus readout CSV.** `reports/decisions/P1L.md` says the per-step readouts put the hop-s entity at top-1 with
   rate at most 0.3% at every step. The committed `reports/p1l/r128/per-step-readout.csv` has logit-lens top-1 rates for M1-L4 up to
   0.643 (seed mean, hop 1, entity step 1, reader step 2), and 0.110 on the hop-2 diagonal. I did not enter or cite any readout
   result and left LP5 open with a note; someone who owns the readout should reconcile the two before it is reported.
2. **B0 in the decision record.** The record gives B0 (context-retained) as 0.1–0.3%; `reports/p1l/accuracy.csv` (primary) has
   0.30–1.00% for hops 1–8. B0 is not cited.
3. **Rank-32 wording.** The task asks for "the readers did not learn to use the oracle memory" with no interpretation, so the paper
   gives only the B7 rule 1 sentence; the rule 2 attribution to the oracle design is kept out of the paper (ledger notes it as an
   untested inference). The B4 lower bound at rank 32 is exactly 0.000000, which fails "interval above 0" together with B8.
4. **B8 versus M1-L4 sentence.** "One fixed oracle memory, 17,000 steps, not a statement about unshared reads in general" paraphrases
   the decision record's "Narrowed" bullet; the prereg has no wording for it.
5. **P1L-C read-position inference.** The report's reading that the deployed readers also use earlier question positions is an
   inference from the drop; the ledger marks it so and the paper does not state it.
6. **Latency.** Timings are per-question means from one run per seed; the rank-32 and rank-128 timings come from different jobs and
   are not compared. B4 at rank 32 (0.438 ms) is slower than at rank 128 (0.392 ms), which shows job-to-job variation.
7. **Abstract.** Changed only for the confirmation outcomes (shared reader at rank 128, B8 fails, no causal bridge claim, rank 32
   check). A reviewer may prefer to leave the bridge-patching clause out.

WP-T10 part 2 done

# WP-T10 part 3a — LP5 per-step readouts

Master `88e8652` merged first (corrected P1-L record `f4f9a98`, merged `5eca4d2`). Text only, static checks only (tags resolve, table
columns, links, `git diff --check`; every LP5 number was compared with `reports/p1l/readout-pooled.csv` and
`reports/p1l/record-numbers.txt`; the three accuracy values quoted for comparison were checked against `accuracy.csv`).

Commit: the draft/ledger commit titled "WP-T10: LP5 per-step readouts entered as descriptive ..." plus this report.

## What changed where

**[evidence_llm.md](../paper/evidence_llm.md).** LP5 is rewritten as a descriptive row (no gate): at rank 128 the logit lens recovers
earlier-hop entities from M1-L4's early states, best at s >= e rather than at s = e (hop-1 questions, e = 1: 38.62/64.28/61.13/47.93%
at s = 1..4); the s = 4 cells match the deployed EM within 0.1 pp and the s = 2, 3 cells are close to the same checkpoint's EM at
L = 2, 3 (64.15, 61.28%); B8 decodes names only at s = 4 (19.08/19.33/18.38% for e = 1, hops 1-3; at most 0.23% at s = 1..3);
contrastive, raw-logit and random directions are at or near chance for both readers (maximum cell 0.28%); every method is at chance
at rank 32 (maximum cell 0.27%). The "readouts need their own reading" mismatch note is removed and the corrected record is cited.
The limits are stated in the row: decoding only, no ordered trajectory or path following, not evidence for LP4, no causal bridge
(PC1); "what the reader would answer at step s" is marked as the record's interpretation of the EM matches. P1LE gains the readout
files and the correction commits. LP4, the scope bullet, the must-not-say list (new bullet on trajectory/path-following/composition
readings; readout item removed from the DPFP/P1L-D bullet) and the audit boundary are updated.

**[draft.md](../paper/draft.md).** Header (readouts no longer pending); Section 8.5 "Per-step readouts (descriptive)" replaces the TODO
paragraph (the J-lens-proxy and cost caveats kept); Section 9 conclusion; Appendix B row; figure-table row; reading rules (LP5 added to
the descriptive rows; pending list now DPFP and P1L-D only). No abstract, Section 1 or claim-contract row changed, because the
readouts support no claim.

## Left pending

DPFP (DP1) and P1L-D (PD1), part 3b.

## Notes

1. The corrected record says the B8 logit-lens cells at s = 1..3 are 0.05-0.22%; over all hops 1-8 in `readout-pooled.csv` the maximum
   is 0.233%. LP5 and the draft use "at most 0.23%".
2. The record's "largest single-seed cell for e = 1, s = 1 is 51.8% (hop 6 questions)" is not cited.
3. No wording was unsettleable; the only interpretive sentence ("what the reader would answer if it stopped at step s") is the
   record's and is labelled as such.

WP-T10 part 3a done

# WP-T10 part 3b — DPFP negative result; P1L-D not run

Master `711d1ac` merged first (wp/llm with the DPFP report). Text only, static checks only (tags resolve, no duplicate DP/PD/LP/PC IDs,
table columns, links and anchors, `git diff --check`). All numbers of the DPFP report were reproduced with
`python3 scripts/analyze_dpfp.py reports/p1l-dpfp` (login node, stdlib only) and the audit values were read from
`reports/p1l-dpfp/audit-e16.json` and `audit-e8.json`; the percentage-point differences are arithmetic on those values.

Commits: the draft/ledger commit titled "WP-T10: DPFP negative result (DP1-DP3) and P1L-D not run (PD1) ..." plus this report.

## What changed where

**[evidence_llm.md](../paper/evidence_llm.md).** The pending-placeholder section is replaced by "DPFP and P1L-D: registered result":
DP1 (not supported: E=16 val EM 39.0/58.0/60.8% against the 0.80 gate, the registered E=8 retrain of D128-3 reached 73.0% and is not
accepted; registered decision stop, no writer, no further writer screen; LSQ ran no arm, so no learned writer reaches the WP-L4 gate in
this project; Amendment A1: DPFP changes feature geometry and query normalization together, so no gain is credited to decorrelation
alone, and the E=8 gain also mixes load), DP2 (descriptive: read error, head accuracies, value SIR and key overlap per arm; the
original-space key overlap is not lower than R's), DP3 (descriptive: zero- and shuffled-memory audit; answers depend on the episode's own
memory, not evidence of an accurate read), PD1 (new status "not run": P1L-D was not registered because no writer was selected). Population
V-L10, source key DPFE, scope bullet, three must-not-say bullets (replacing the DPFP/P1L-D placeholder bullet), audit boundary,
header notes. Nothing is pending in this ledger now.

**[draft.md](../paper/draft.md).** Section 8.5 "DPFP and P1L-D" paragraph replaces the TODO; header and abstract no longer leave a writer
result pending (the abstract sentence now says the screens, including the last registered DPFP screen, select no learned writer);
Section 8.3 (LSQ/DPFP sentence and the E8 note), Section 9 conclusion, Appendix B (DPFP row), Appendix C (the LLM writer-screens claim row
extended), reading rules (descriptive rows, pending list: nothing pending). Claim contract kept: same-rank comparisons only, no end-to-end or
learned-writer claim, latency a cost. The sentence "no learned writer reaches the WP-L4 gate in this project" is in Section 8.5.

**[evidence_symbolic.md](../paper/evidence_symbolic.md).** Pointer sentence updated to the new section.

## Test and smoke overrun

The ledger records no cost deviations, so the overrun is not entered there. For the record: `docs/reports/DPFP.md` gives about **0.78** node·h for
tests plus smokes against the registered 0.5 (the task text's 0.63 is superseded); 0.17 tests, reproduction, failed run, rerun and debug,
0.45 E=16 smokes, 0.13 E=8 smoke, 0.03 reload checks, 0.01 audit test; the E=16 arms (4.02) and the E=8 retrain (1.31) are science jobs outside
that cap; total DPFP about 6.14 node·h.

## Notes

1. DP1's statement that the E=8 gain "mixes load with the DPFP features" is my reading (E=8 halves the facts per episode); the report says the
   halving raises EM by 12.2 pp but does not say it is non-comparable. The prereg does allow the E=8 retrain as the registered fallback.
2. "Not run" is a new status label (PD1 only), defined in the ledger's status paragraph.
3. The report's reading that DPFP changes "the addressing geometry and the read, not a decorrelation of the learned projections K" is quoted
   in DP2 as a description of the original-space overlap numbers; phi-space numbers are not compared with R.

WP-T10 part 3b done
