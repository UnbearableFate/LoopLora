# WP-T11: paper revision after review round 2 (G1; sn-paper; LoopLora-wt/paper, wp/paper)

Plan: `plan/followup_2026-10-29.md` G1. The user approved G1–G4 on 2026-10-29 13:00 JST.

Read first:
- `docs/tasks/_common_rules.md`
- `docs/review/rv-fable_2026-10-29.md`, §1.3 (the (hop, L) matrix), §1.6 and §4 (the paper)
- `docs/review/rv-astra_2026-10-29.md`, §1 (what the results establish, statistical and framing threats) and §3.1
- `docs/paper/draft.md`, `docs/paper/evidence_llm.md`, `docs/paper/evidence_symbolic.md`

This is text work on existing artifacts. Read no new test data and run no training. The new numbers come from committed
CSV and JSON files, computed with a stdlib script that you commit, e.g. `scripts/paper_tables_g1.py`. Never import torch
on the login node.

## Changes
1. **(hop, L) matrix.** Replace any "composition reaches (only) hop 3" wording with the M1-L4 matrix of seed-mean EM by
   evaluation depth L = 1…8 and hop 1…8.
   - Use the primary setting at both backbones: `reports/p1l/r128/accuracy.csv` and `reports/p5l/r128/accuracy.csv`.
   - Add an "oracle stop" row: the best L per hop. It uses the labels, so it is an upper bound and descriptive only.
   - Model sentence: "at the trained depth the reader reaches hop 3; the same checkpoint reaches hops 4–8 at x–y % when run
     deeper, while losing hop 1." Fill in x–y from the CSV, for each backbone.
   - Check the reviewer's numbers against the CSV; the CSV wins.
2. **Main LLM table** (astra §3.1). One table at the same rank for B4, M1-L4 and B8 at both backbones, with:
   hop-wise EM; adapter parameters; state bytes; measured latency, reported as a cost and compared only within one
   backbone and job.
   - Separate three kinds of evidence: oracle composition, learned-writer access, and mechanism evidence.
   - Add a compact matrix of the negative interventions, each with the exact gate it failed. It must not read as a claim
     that composition is impossible.
3. **Seed-level statistics.** Beside each episode-bootstrap interval of a headline contrast, print:
   - the three seed effects;
   - a seed-level interval, with its method stated.

   The headline contrasts are the B6 contrast, G1-L and G2-L, at r = 128 on both backbones.
   - Say that the intervals are conditional on three fitted seeds and one fixed oracle codebook.
   - Add "one oracle seed" to the limitations.
4. **Must-not-say additions** (fable §4.2):
   - "the reader cannot go beyond hop 3";
   - "B8 cannot compose": B8 contains M1-L4, and its warm start is untested until G3 reports;
   - "an exact store would not help": that rests on evaluation-only evidence (RFX), and trained LSQ never ran;
   - "the recurrence composes across question positions", and its converse.
5. **Mechanism wording.** Each question position performs its own vertical walk, and the frozen upper layers aggregate
   across positions. Which position's walk carries the answer is untested.
6. **Related work.** Add the works cited by the two round-2 reviews that bear on the paper (fable §4.5; astra §2):
   - Cite abstracts only. Claim no priority, and say which of them this project did not study.
   - Use each citation as the reviewer verified it. Mark any citation you could not confirm, and do not add sources of
     your own.
7. **Claim contract** (fable §4.3).
   - Keep: the equal-parameter oracle contrast at two backbones; a negative B8 under one optimizer; a negative learned
     writer (DPFP, DPFP-X); readouts as decoding only; no causal bridge at the final position; no scale trend.
   - Keep every registered stop statement.

Do not anticipate G2–G4. Their results enter the paper in a later part.

## Checks and report
- The ledger and draft checks from WP-T10 must pass: tags resolve, no duplicate IDs, tables, links and anchors are
  intact, and `git diff --check` is clean.
- Every new number must be reproduced by the committed script from committed files.
- Write the report in `docs/reports/WP-T11.md`, commit on wp/paper, and STOP with the line `WP-T11 done` and the commit
  hash. No merge, no push.
