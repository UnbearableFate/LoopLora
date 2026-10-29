# WP-L10: B8 warm start, sharing as an optimization scaffold (G3, study B8W; sn-p1l; LoopLora-wt/p1l, wp/p1l)

Plan: `plan/followup_2026-10-29.md` G3. The user approved G1–G4 on 2026-10-29 13:00 JST.

Read first:
- `docs/tasks/_common_rules.md`
- `docs/review/rv-astra_2026-10-29.md`: §1, "Oracle readers and the B8 discrepancy", and proposal 3
- `docs/review/rv-fable_2026-10-29.md`: §1.2 and proposal R2
- `src/looplora/core/adapters.py`: `UnsharedAdapter` and `IterativeQueryAdapter`
- `docs/prereg/P1L.md`, `reports/decisions/P1L.md` and `reports/p1l/r128/`

## Question
B8 contains M1-L4: copying the shared R/U/Q into every step reproduces M1-L4 exactly. Does B8 keep the M1-L4 solution
when it starts there and its steps are untied? If it does, B8's failure is an optimization problem, not a capacity
problem.

## Fixed design (change only with a flagged reason)
- **Backbone and checkpoints.** Qwen3-0.6B-Base, r = 128, oracle_random, primary setting. Use the registered M1-L4
  checkpoints, seeds 17, 29 and 43.
- **Equality gate.** Convert each M1-L4 checkpoint into a B8 (`UnsharedAdapter`) with every step's R/U/Q set to the shared
  ones. Before any continuation, show identical logits on a validation batch. State the tolerance and the dtype in the
  prereg. If the gate fails, STOP.
- **Arms.** Both arms run the same number of continuation steps, with the same data stream, the same lr (the registered
  M1-L4 lr at r = 128, from `lr-S-lock.json`) and freshly reset optimizer state:
  - **shared:** continue training M1-L4;
  - **untied:** continue training the converted B8.
- **Continuation length.** Propose it within the budget, with a reason. astra suggests 2,000 steps; the original budget
  was 17,000.
- **Descriptive references:**
  - B4@1 and the original B8@4 from P1-L;
  - the starting M1-L4@4, evaluated on the new range.
- **Metric.** Hops 2–3 EM at L = 4 on a fresh test range, untied minus shared, paired by seed. Also report hop 1.
- **Fresh test range.** [108000, 110000), 2,000 episodes at eval seed 4242. Add a `CONFIRMATION_WINDOWS` entry `b8w` and
  keep the existing entries unchanged; a test must show this. Read the range once, after registration and after the
  checkpoints are locked.
- **Decision rule** (astra #3; flag any change). Untying keeps the solution if both hold:
  - its hops 2–3 loss against the shared arm is at most 2 pp in every seed;
  - the pooled interval excludes a loss worse than 2 pp (non-inferiority).

  Otherwise, report that untying loses the solution. That is fragility of the solution, not lack of expressivity. Either
  way:
  - B8 does not gain equal-parameter status;
  - G1-L stays failed;
  - nothing is said about why B8 failed from a random start.
- **Budget.** 1.0–2.0 node·h in total, including tests and smokes. Each job is 1 node, ≤ 1 h.

## Step 1 (now)
1. Merge master (at or after the commit that adds this file) into wp/p1l.
2. Draft `docs/prereg/B8W.md`, based on the code:
   - how to convert, and how to check equality;
   - the continuation length;
   - the intervals;
   - the cost estimate from P1-L step times.
3. STOP with the prereg path. Submit no compute job. The orchestrator registers the prereg on master.

Step 2 (code, tests on compute, smokes, runs, test evaluation, report `docs/reports/B8W.md` and the state file) follows the
orchestrator's OK.
