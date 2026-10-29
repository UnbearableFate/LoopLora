# WP-L9: label-free stop rules on the r = 128 M1-L4 checkpoints, evaluation only (G2, study LSR; sn-core; LoopLora-wt/core, wp/core)

Plan: `plan/followup_2026-10-29.md` G2. The user approved G1–G4 on 2026-10-29 13:00 JST.

Read first:
- `docs/tasks/_common_rules.md`
- `docs/review/rv-fable_2026-10-29.md`, §1.3 and proposal R1
- `docs/prereg/P1L.md` and `docs/prereg/P5L.md`
- `reports/decisions/P1L.md` and `reports/decisions/P5L.md`
- `docs/reports/P1L-C.md`: the read-state hooks (`p1lc.read_states`)
- the symbolic stopping evidence: `docs/paper/evidence_symbolic.md` H1–H6, `docs/reports/CVH.md`, and the P4-S stopping
  reports

## Question
The (hop, L) matrix shows the deployed L = 4 under-reads deep hops and over-reads hop 1 (fable §1.3). Can a per-question
stop rule that uses no labels raise hops 4–8 over the fixed L = 4, without a large loss on hops 1–3?

## Fixed design (change only with a flagged reason)
- **Checkpoints.** The registered r = 128 M1-L4 checkpoints at both backbones: P1-L on Qwen3-0.6B-Base and P5L on
  Qwen3-1.7B-Base, seeds 17, 29 and 43. Evaluation only: no training and no gradient step.
- **Rules.** Each runs every question at L = 1…L_max = 8 and stops at the first s where the rule fires:
  1. the relative state change ‖z^{s+1} − z^s‖ / ‖z^s‖ is below τ1;
  2. the logit-lens top-1 is unchanged for two consecutive steps;
  3. the read norm ‖M Q z^s‖ is below τ2.

  Choose every threshold on validation data only. State the selection objective in the prereg before you compute
  anything.
- **References:**
  - fixed L = 4, the deployed depth and the primary comparator;
  - fixed L = 8;
  - the oracle stop (best L per hop). It uses labels, so it is descriptive only.
- **Metrics:**
  - EM by hop at the stopped depth;
  - mean reads per question, as a cost;
  - measured latency, as a cost, compared within one job only.
- **Fresh test range.** [106000, 108000), 2,000 episodes at eval seed 4242. Add a `CONFIRMATION_WINDOWS` entry `lsr` and
  keep the existing entries unchanged; a test must show this. Read this range once, after registration and after the
  thresholds are locked.
- **Decision rule** (draft it from fable R1 and flag any change). Per backbone, a rule passes if both hold:
  - its hops 4–8 EM exceeds the fixed L = 4 by at least 3 pp, with a paired interval above 0;
  - its hop-1 EM loses at most 10 pp against the fixed L = 4.

  Give the intervals, the multiplicity correction over 3 rules × 2 backbones, and the per-seed condition. Also report
  hops 1–3 together.
- **Budget.** At most 0.8 node·h in total, including tests. Each job is 1 node, ≤ 1 h.

## Step 1 (now)
1. Merge master (at or after the commit that adds this file) into wp/core.
2. Draft `docs/prereg/LSR.md`. Base it on the code: which hooks give z^s, M Q z^s and the logit lens, and the validation
   set and size. Also give the cost estimate.
3. STOP with the prereg path. Write no code beyond what the draft needs to be exact, and submit no compute job. The
   orchestrator registers the prereg on master.

Step 2 (code, tests on compute, threshold lock on validation, test evaluation, report `docs/reports/LSR.md` and the state
file) follows the orchestrator's OK.
