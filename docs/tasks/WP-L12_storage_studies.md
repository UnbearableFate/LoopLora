# WP-L12: trained least-squares stores and the storage × query 2×2 (G5, studies LSQT and SQ2; sn-llm; LoopLora-wt/llm, wp/llm)

Plan: `plan/followup_2026-10-29.md` G5. On 2026-10-29 the user approved G5 with both designs ("两个方案都试一下").

Read first:
- `docs/tasks/_common_rules.md`
- `docs/review/rv-fable_2026-10-29.md`: §1.1 and proposal W2 (and W3, which is conditional and not part of this task)
- `docs/review/rv-astra_2026-10-29.md`: §1 "Why learned writing remains weak" and proposal 4
- `docs/prereg/LSQ.md` and `src/looplora/llm/lsq.py`, both registered and written but never run
- `docs/reports/RFX.md`, `docs/reports/DPFP.md`, `docs/reports/DPFP-X.md`

## Why now
- DPFP-X raised EM at r = 256 but not the read error (0.397 against D128-3's 0.363).
- RFX closed LSQ on evaluation-only evidence: it swapped the store on checkpoints trained with the delta rule. It cannot
  show whether a writer trained with an exact store would co-adapt.
- Both reviewers rank a trained-store test high.

The two studies are independent registrations. Both are validation screens with the WP-L4 protocol: T1, E = 16,
val 1,000 episodes at eval seed 4242, full-vocabulary EM. Neither reads any test range.

## Study LSQT (fable W2): least-squares stores trained from scratch at r = 128
- **Design.** Reuse `docs/prereg/LSQ.md` and `lsq.py` at r = 128 with the R objective and unit queries, as registered,
  with these changes. Flag each one in the new prereg `docs/prereg/LSQT.md`:
  - The arms run unconditionally. RFX's trigger is replaced by the user's approval.
  - There are two ridge arms, η = 1 and η = 10⁻², which fable proposed. LSQ's η rule would take η from RFX's R-r128
    fit, clipped to [1e-4, 1e-1]. State both η values and why.
  - K4 is the third arm. Add U (unit query, delta store) as a control only if the budget allows.
  - The step budget follows LSQ (the DPFP budget rule at r = 128).
- **Decision.** From LSQ:
  - val EM ≥ 0.80 flags a writer candidate. A confirmation would need its own registration.
  - Record fable's falsifier: if the best RG arm is below 40 %, the trained-LSQ line closes.
  - Also report RG against R-r128 (33.5 %) and against D128-3 (60.8 %). These comparisons are descriptive.
- **Budget.** About 3.2 node·h for two arms and 4.6 for three, smokes included. Each job is 1 node, ≤ 2 h with
  `--scale-up docs/prereg/LSQT.md`.

## Study SQ2 (astra #4): storage × read-coefficient supervision, continued from D128-3
- **Design.** Continue the D128-3 checkpoint (DPFP, ν = 3, r = 128, E = 16) for the same number of steps in every arm.
  Use the same samples and optimizer schedule, and keep the feature width and query normalization fixed. The 2×2 is:
  - storage: delta (as trained) or ridge on the DPFP features;
  - loss: the existing losses, or the existing losses plus a directional coefficient loss
    ‖d/(‖d‖ + ε) − e_target‖² on the read coefficients d from `gram_read` (Mq = Σ dᵢ vᵢ).

  The old raw-read loss stays in every arm. Fix the ridge η and the loss weight before any validation result, from
  training-only numerics.
- **Continuation length.** astra suggests 6,000 steps per arm. Propose the length within the budget, with a reason.
- **Primary contrast** (registered before results): ridge + coefficient loss against the continued delta arm with the
  existing losses. It advances only if all three hold:
  - EM is at least 10 pp higher;
  - the normalized read error is at most half;
  - value decoding is stable, defined in the prereg.

  The other two cells are descriptive. They are not promoted by picking the best of four after the fact.
  ≥ 0.80 is only a candidate. Confirmation needs its own three-seed registration, which astra estimates at 9–12 node·h.
- **New code.** Ridge storage for rectangular DPFP keys and values. The current `ridge_final` assumes square projected
  keys and values. Also add the coefficient loss.
  - With every new option off, the existing paths must stay bitwise unchanged; a test must show this.
  - Log conditioning, query norms and coefficient cancellation.
- **Budget.** 1.8–2.5 node·h. Each job is 1 node, ≤ 1 h, or ≤ 2 h with `--scale-up docs/prereg/SQ2.md`.

## Step 1 (now)
1. Merge master (at or after the commit that adds this file) into wp/llm.
2. Draft both `docs/prereg/LSQT.md` and `docs/prereg/SQ2.md`, based on the code. Give:
   - the arms;
   - the steps, from the existing schedule rules;
   - η and the loss weight, and how they were fixed;
   - the decision rule;
   - the cost estimate from the measured DPFP and WP-L4 step times.
3. STOP with both paths. Submit no compute job. The orchestrator registers them on master.

Step 2 (code, tests on compute, smokes, release, STOP) and step 3 (runs, audit, reports `docs/reports/{LSQT,SQ2}.md` and
the state file, STOP) follow the orchestrator's OK. Run the two studies' jobs in parallel where the 6-job cap allows.
