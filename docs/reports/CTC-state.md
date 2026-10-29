# CTC state (WP-L11, G4): finished, candidate flagged

As of 2026-10-29 14:30 JST. Author: sn-oracle (Claude Sonnet-5.5). Branch `wp/oracle`, worktree
`/work/xg24i002/x10041/LoopLora-wt/oracle`. Registration: `docs/prereg/CTC.md` (C1, blake2b-128 546bbe39...). Report:
`docs/reports/CTC.md`. Decision record: `reports/ctc/decision.json`.

## Outcome
Candidate flagged (arm CE): M1-L4@4 hop-1 EM 50.7 %, gain over B4@1 on hops 2-3 +24.05 pp (CI 21.8 to 26.3), zero and
shuffled memory below 0.3 %, p = 1.00000 on val [0, 1,000). Equal to the oracle reference within 0.2 pp. Wording: a
learned parser plus a fixed code table (entities read at their own tokens; the relation is constant in t2_chain_end).
Any confirmation needs its own registration and the user's decision.

## Artifacts
- Code: `src/looplora/llm/ctc.py`, `scripts/run_ctc.py`, `tests/test_ctc.py` (17 tests pass; new files only).
- `reports/ctc/`: `stage-a-lock.json`, `stage-a-probes.csv`, `smoke-timings.json`, `stage-b-{ce,ceans}-summary.json`,
  `stage-b-{ce,ceans}-train-log.jsonl`, `eval-metrics.json`, `decision.json`.
- Not committed (hashes in the lock and `decision.json`): `runs/ctc/stage-a/{feature-stats.pt,predictions.npz}`,
  `runs/ctc/stage-b-{ce,ceans}/writer.pt`, `runs/ctc/eval/samples.npz`.
- Jobs and node-hours: 10 jobs, 1.028 node-h (table in the report). Job 3724010 failed once: the tiny test reader ignored
  its memory, so a "zero differs" assertion could not hold; a test fault, fixed in the test only.

## Open
Nothing is pending on this worktree. Not done: a multi-seed run, test-range evaluation, any other task or rank. Do not
compare these EM values with DPFP or DPFP-X as like for like.
