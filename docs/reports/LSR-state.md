# LSR state (WP-L9 step 2; sn-core, wp/core). Registered `docs/prereg/LSR.md`, blake2b-128 `63da9c626d699252c239cc86e149f1ad` (with Clarification C1, master ad1a9b5; the validation runs recorded the pre-C1 hash `91235d80a38f200e3a7b3bec907c1f1e`, which `design.json` lists in `earlier_prereg_hashes`)

Status 2026-10-29: code, tests and smoke done; validation jobs running. No test episode has been generated.

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
| 3724070 / 3724071 | `lsr-val-0.6b` / `-1.7b` | pending | validation [1000, 3000), 3 seeds each |

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
