# LSR state (WP-L9 step 2; sn-core, wp/core). Registered `docs/prereg/LSR.md`, blake2b-128 `63da9c626d699252c239cc86e149f1ad` (with Clarification C1, master ad1a9b5; the validation runs recorded the pre-C1 hash `91235d80a38f200e3a7b3bec907c1f1e`, which `design.json` lists in `earlier_prereg_hashes`)

Status 2026-10-29: thresholds locked (`reports/lsr/thresholds.json`, awaiting the orchestrator's check). No test episode has been generated.
node·h so far: 0.288 (cap 0.8; projected test jobs about 0.15-0.2 including the latency audit).

## Code (commit `5348211`, plus the commit that carries this file)
- `src/looplora/llm/lsr_rules.py` (numpy only): grid, rules R1/R2/R3, reads, selection (`select_thresholds`), constants and the six checkpoints.
- `src/looplora/llm/lsr.py`: `instrument` (trace of z^s and r^s at each question's final position in the L = 8 pass; grouped per-row depth pass),
  `scan_batch` (p^0..p^8 and the trace), `stopped_logits`.
- `scripts/evaluate_lsr.py` (`--split smoke|val|test`, `--latency`; the test range needs a committed authorization checked by
  `check_test_authorization` before any episode, and an output directory that does not exist), `scripts/analyze_lsr.py` (`design`, `select`).
- `reports/lsr/design.json` (design lock: prereg hash, grid, margins, ranges, the six checkpoint hashes), written by `analyze_lsr.py design`.
- `CONFIRMATION_WINDOWS["lsr"] = 106000`; `tests/test_p1ls_pipeline.py` now asserts the registered entries individually (prereg flag 9).

## Jobs (node·h = job-script runtime; PBS start/end lines in the logs)
| Job | Name | node·h | Note |
| --- | --- | ---: | --- |
| 3724000 | `lsr-tests` | 0.045 | `tests/test_lsr.py`, `test_p1ls_pipeline.py`, `test_p5l.py`, `test_p5l_pipeline.py`, `test_p1lc.py`: all passed (1 skip) |
| 3724030 / 3724031 | `lsr-smoke-06` / `-17` | 0.018 / 0.019 | smoke val [0, 64), seed 17; stopped in the latency audit (below) |
| 3724051 | `lsr-debug` | 0.002 | wrong path, exit 2 |
| 3724055 | `lsr-debug` | 0.006 | diagnosis of the audit failure (development worktree, nothing kept) |
| 3724062 / 3724063 | `lsr-smoke-06` / `-17` | 0.018 / 0.018 | smoke after the audit change; exit 0 |
| 3724070 / 3724071 | `lsr-val-0.6b` / `-1.7b` | 0.076 / 0.083 | validation [1000, 3000), 3 seeds each, exit 0 (4 min 35 s / 4 min 59 s) |
| 3724125 | `lsr-select` | 0.002 | `analyze_lsr.py select`; wrote `reports/lsr/thresholds.json` |

## Smoke (val [0, 64), seed 17; timing and audits only, not used for any decision)
- 64 episodes scan in 5.3 s (both backbones) including warm-up; the per-checkpoint job took 46 s with the latency audit.
- Audits passed on the first two batches: traced L = 8 logits equal the plain L = 8 logits, inputs and memory unchanged across depths.
- Latency audit (128-episode design shortened to the 64 smoke episodes; placeholder taus .25 / .5): ms per question, 0.6B / 1.7B: fixed L = 4
  0.59 / 0.59, fixed L = 8 0.78 / 0.76, trace pass 0.84 / 0.82, grouped phase 2 (R1) 1.47 / 1.42, (R3) 0.89 / 0.87. The grouped pass is slower
  than a full fixed-depth pass in this implementation (rows regrouped per depth). No latency claim.
- **Audit deviation (needs the orchestrator's OK before the test job).** Prereg: more than 0.2% disagreement between the grouped stopped
  pass and p^{s*} stops the job. The smoke saw 1.4-2.2% for R1 (0-0.1% for R3, whose rows mostly share one depth). Cause: bf16 answer logits are
  tied or within a few ulps; every disagreeing question has a top-2 gap <= 0.125, and a plain fixed-depth L = 4 pass on half of the batch flips
  2 of 128 questions against the same pass inside the batch of 16 (max logit difference 0.27). So it is batch-composition noise, not a depth bug
  (the same grouping with a constant depth is bitwise equal to the plain pass: `test_the_stopped_pass_equals_...`). The code now records the
  count, fraction and maximum top-2 gap, and raises only if a disagreeing question has a top-2 gap above 1.0 (`TIE_GAP`). The statistics use
  p^{s*} from the fixed-depth scan either way.
- Observation, not a result: on val [0, 64) hop-1 EM peaks at L = 2 and hop 4 at L = 6-7, the P1-L shape; the median read ratio ||r^s||/||r^0|| is
  1.0-1.1 at 0.6B and grows to about 10 at s = 7 at 1.7B, and the median relative change falls from 0.13 to 0.01 at 0.6B and stays 0.5-0.7 at
  1.7B. Whether the grid has a feasible value is for the validation selection to say.

## Clarification C1 (orchestrator, ad1a9b5)
The audit change above, the placement of the latency audit in the test job on val [1000, 1128), the val [0, 64) smoke observations as
information seen before the lock, and the unchanged handling of infeasible rules are registered. `design.json` is re-locked with the new
prereg hash; `analyze_lsr.py select` accepts validation records carrying either hash (no rule and no code changed with C1).

## Validation selection (val [1000, 3000), 2,000 episodes x 3 seeds per backbone; `reports/lsr/thresholds.json`, status locked)
Validation EM (%) of fixed L = 4 by hop 1..8: 0.6B 48.1 / 34.4 / 19.6 / 7.2 / 4.4 / 4.6 / 5.4 / 5.2; 1.7B 58.1 / 43.1 / 23.1 / 5.9 / 3.5 / 4.1 / 4.4 / 4.5.
- **R1 and R3: no feasible grid value at either backbone** (hop-1 loss against L = 4 beyond the 8 pp pooled / 10 pp per-seed margins at every
  tau). They fail as registered and their test statistics are never computed. Best pooled hop-1 change counts (of 6,000 questions): R1 0.6B
  -441 (k = -8, mean depth 4.8, hops 4-8 6.8%), R1 1.7B -1,229; R3 -596 / -1,525 (its read ratio never falls below 1 at 0.6B, grows to about 10 at 1.7B;
  tau below 1 never fires, above 1 stops at s = 1).
- **R2 (no parameter): feasible at both**: hop 1 54.4% / 63.2% against 48.1% / 58.1% at L = 4, hops 4-8 mean 9.0% (0.6B) and 10.3% (1.7B)
  against 5.3% and 4.5% at L = 4, mean 5.6 / 5.7 reads. Validation numbers only; the test decides.
- **L_c = 4 at both backbones** (the best fixed depth under the same constraint is the deployed one: L = 5 and deeper lose more than the margin on hop 1).
- O1 (best fixed L per hop on validation): [2, 4, 5, 6, 8, 8, 8, 8] at both backbones.
