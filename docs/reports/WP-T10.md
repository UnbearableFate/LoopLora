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
