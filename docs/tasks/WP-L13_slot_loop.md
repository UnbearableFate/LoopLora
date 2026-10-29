# WP-L13: SLOT-LOOP-v1, stages S0 and S1A (sn-slot; LoopLora-wt/slot, wp/slot)

The specification is `plan/looplora_diagnosis_and_execution_plan_2026-10-29.md` (4e6778a, "the plan").
- It was written against a255e81. Since then master has gained only docs, and the LSQT U control is still running on wp/llm.
- The plan is a reference, not an authority. Where it disagrees with the code, follow the code and flag the difference.
- The user asked on 2026-10-29 to continue according to the plan.

## Read first
- **Rules:** `docs/tasks/_common_rules.md`, all of it, including "Step 2 on compute" and the WAIT/STOP protocol.
- **The plan, all of it.** It is the design.
- **Reports:**
  - `docs/reports/SQ2.md` and `docs/reports/SQ2-state.md`: S-RE, the parent.
  - `docs/reports/RFX.md`: frozen store/query interventions.
  - `docs/reports/P1L-C.md`: read-position ablation.
  - `docs/reports/DPFP.md`: D128-3.
- **Code:** the plan's [R16]–[R30]:
  - `src/looplora/core/adapters.py`, `src/looplora/core/wrapper.py`
  - `src/looplora/llm/sq2.py`, `src/looplora/llm/writer_fix.py`, `src/looplora/llm/read_fix_run.py`, `src/looplora/llm/rfx.py`
  - `src/looplora/llm/batching.py`, `src/looplora/llm/protocols.py`, `src/looplora/llm/inject.py`, `src/looplora/llm/config.py`
  - `src/looplora/data/relations.py`
  - `scripts/submit.py`, `scripts/pbs/job.pbs`
- **Patterns to reuse:**
  - releases and reload checks: `scripts/prepare_dpfp_release.py`, `scripts/check_smoke_reload.py`;
  - prereg structure: `docs/prereg/SQ2.md`, `docs/prereg/LSQT.md`.

## Scope and budget (authorized)
- **Authorized:** S0 (plan §6) and S1A (plan §7) only.
  - S1B (§8) needs its own registration after S1A.
  - S2 and S3 are not authorized.
- **Budget:** S0 with its tests and preflight ≤ 1.0 node·h. S1A with its pilot, release, runs, evaluation and analysis ≤ 4.5 node·h.
- **Jobs:** one node each. Each job ≤ 1 h; ≤ 2 h only with `--scale-up docs/prereg/SLOT-LOOP-v1.md` where that prereg allows it. Keep at most 6 of your jobs queued or running.
- **Old studies are untouched:**
  - The old verdicts (P1L, SQ2, LSQT, DPFP and the 0.80 writer gate) stay as they are.
  - Read no old test or confirmation window.
  - The old `val[0,1000)` at seed 4242 is used only for the Native/Matrix identity replay (723/1000, NLL 1.642127…).

## Step 1 (login node only): preregistration draft, then STOP
Write `docs/prereg/SLOT-LOOP-v1.md` on wp/slot. It must contain the following.

1. **Hypotheses and claim boundaries.** H1 (plan §3.1) for this registration. State that S1B, S2 and S3 are out of scope.
2. **Data.** Fix every value below before any result.
   - The manifest scheme of plan §5.1:
     - `derive_seed(20261029, "slot-loop-v1", purpose, stage[, run_seed])`, with the integers resolved;
     - generator version and layout digest;
     - the sizes: smoke 64, calibration 2,048, dev 1,000, confirmation 2,000, plus the unseen-name diagnostic if you keep it.
   - How confirmation is protected in code. Register the new study the way `CONFIRMATION_WINDOWS` protects P1-L windows, or with a purpose-seed guard. Keep the existing protections, add a test, and check that episode IDs do not overlap.
   - S0 and S1A use separate stage namespaces.
3. **S0** (plan §6).
   - Conditions: Native, Matrix, SoftSlot, HardSlot, Clean-final (oracle diagnostic), and the recency-matched SoftSlot (§6.2). Register the last one now as descriptive, so it cannot be picked after the fact.
   - Scope: `all_valid_question`, with `final_only` only as a separate localisation table.
   - Calibration grid, selection rule and tie-break as in §6.3, on the calibration split only.
   - Metrics as in §5.3. State which read-error definition you use and how it relates to the SQ2 report's.
   - The S0 decision table (§6.5) mapped to the S1A choices it makes: backend, g/τ, and whether S1A runs as the "limited adaptation".
   - D128-3 once as an auxiliary recheck.
4. **S1A** (plan §7). Lock all of these before any S1A data is seen:
   - the two arms Matrix-L1 and SoftSlot-L1, from the same S-RE state;
   - the frozen set, including the shared A, the buffers and the DPFP/ridge settings, with how the freeze is verified;
   - the trainable set Q/R/U/B;
   - fresh AdamW with its settings;
   - the LR pilot: candidates, steps, shard, the selection metric and its tie-break;
   - how the step count is frozen from the timing pilot (same steps for both arms);
   - checkpoint frequency and the selection rule (dev NLL, earlier on ties);
   - seeds 17/29/43.
5. **S1A decision rule** (plan §5.4): SoftSlot-L1 − Matrix-L1 ≥ +3 pp EM on confirmation, with the paired 95 % whole-episode bootstrap lower bound > 0 (10,000 resamples, seed fixed now), and all three seed point estimates positive. Also state:
   - what the outcomes mean;
   - the one pre-budgeted interface ablation of §7.4;
   - that the common-fixed-LR comparison is descriptive and optional within budget.
6. **Verification list.** Plan §12, each item mapped to a test or a preflight check.
7. **Budget table.** Per job, using measured step times where they exist: SQ2 runs took about 0.2 s per step, and a 1,000-episode evaluation took 6 to 9 s.
8. **Deviations from the plan.** Number each one with its reason. Also list the plan's §4.3 claims about the code, each checked against file:line and marked true or false.

STOP with `STOP: SLOT-LOOP-v1 prereg draft <hash>`. Write no science code before the orchestrator registers the prereg on master. Reading code is fine.

## Step 2 (after registration): implement, test, preflight, S0, then STOP
- **Implement** what plan §11 needs for S0 and S1A. Keep the old matrix path bitwise unchanged when the new option is off, and prove it with a test.
- **Run the tests on compute.** Then ask the orchestrator for a static review of the new code before the preflight, by ending with `STOP: review <hash>`.
- **Preflight.** It must show:
  - Native and Matrix identity with S-RE on the old val set (723/1000);
  - the owner/mask sentinels;
  - read-only isolation;
  - the frozen set;
  - bytes;
  - capacity errors when E > S.
- **Lock** the manifest and the checkpoint inventory (blake2b-128 of the real files).
- **Run S0.** Write `reports/slot-loop-v1/s0/` (metrics, samples, calibration candidates) and `decision_s0.md`.
- **Report.** Update `docs/reports/SLOT-LOOP-v1.md` and `-state.md`, including the job ledger and node·h.
- STOP with `STOP: S0 done <hash>`, the decision, and the S1A choices it implies.

## Step 3 (after the orchestrator's OK): S1A
- **Pilot and release.** Run the LR and timing pilot, then write the release: steps, LR per arm, g/τ, config hashes, prereg digest and a reload check. STOP for `approval.json`.
- **Runs.** Run the six arm-seed jobs, select checkpoints on dev, then run the confirmation evaluation.
- **Analysis.** Run the bootstrap analysis on compute. Before the confirmation results are opened, the orchestrator has an independent reviewer re-run the analysis script.
- **Report and stop.** Write the decision and the report, and STOP with `STOP: S1A done <hash>`.

## Rules repeated here because they bit us before
- Pass `--commit` the full 40-character hash. `HEAD` resolves in the main checkout.
- debug-g allows at most 30 minutes. Use regular-g for anything longer.
- Outputs go to absolute paths under `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/`, because the job's snapshot directory is deleted.
- Exit 0 is not success. Check the steps, the evaluations, NaN/inf, and the config and commit hashes.
- Use blake2b and never SHA-256. Put no dates in code.
- Commit on wp/slot only. No merge, no push.
