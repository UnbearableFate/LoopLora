# WP-L13: SLOT-LOOP-v1, stages S0 to S3 (sn-slot; LoopLora-wt/slot, wp/slot)

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
- **Authorized:** all stages of the plan: S0, S1A, S1B, S2 and S3. The user said on 2026-10-29: "S2和S3以及新计划的所有任务都一起授权，按计划推进".
- **Order:** the stages run in the plan's order. Each stage starts only when its entry condition is met:
  - plan §0.3;
  - §6.5 (S0 to S1A);
  - §7.4 and §14.3 (S1A to S1B);
  - §8.3 (S1B to S2; S2 runs on the matrix reader if slots add nothing);
  - §10.5 (S3, and the real task within it).
  If a condition fails, the plan's own consequence applies (§14.3). The line is not pushed past a failed gate.
- **Registration:** `docs/prereg/SLOT-LOOP-v1.md` registers S0 and S1A now. S1B, S2 and S3 each get an addendum: a section of the same file or `SLOT-LOOP-v1-S1B.md` and so on. Each addendum has its own data namespace, decision rule and budget table, and the orchestrator registers it before any data of that stage is seen. An S2 that changes after an opened S1 test uses new confirmation data (plan §5.1).
- **Budget:**
  - S0 with its tests and preflight ≤ 1.0 node·h;
  - S1A with its pilot, release, runs, evaluation and analysis ≤ 4.5 node·h;
  - S1B ≤ 11.5 node·h, the plan's figure;
  - S2 and S3: set in their addenda from measured step times, and approved by the orchestrator at registration.
  Record cumulative node·h in the state file.
- **Jobs:** one node each. Each job ≤ 1 h; ≤ 2 h only with `--scale-up docs/prereg/SLOT-LOOP-v1.md` where that prereg allows it. Keep at most 6 of your jobs queued or running.
- **Parallel help:** once the S0 code is merged, the orchestrator may give S2 or S3 preparation to a second worker. Examples are the S2 labels and reader code on top of your `MemoryBackend`, or the S3 measurement harness. Keep the backend interface stable and documented in the state file.
- **Old studies are untouched:**
  - The old verdicts (P1L, SQ2, LSQT, DPFP and the 0.80 writer gate) stay as they are.
  - Read no old test or confirmation window.
  - The old `val[0,1000)` at seed 4242 is used only for the Native/Matrix identity replay (723/1000, NLL 1.642127…).

## Step 1 (login node only): preregistration draft, then STOP
Write `docs/prereg/SLOT-LOOP-v1.md` on wp/slot. It must contain the following.

1. **Hypotheses and claim boundaries.** H1 (plan §3.1) is locked here. H2 and H3 (§3.2–3.3) are named, with their tests left to the S1B and S2 addenda. Include an overview of all stages and their entry conditions.
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

## Later steps: S1B, S2 and S3
Each later stage follows the same cycle:
1. Write the addendum and STOP for registration.
2. Implement and test, then STOP for review.
3. Pilot and release, then STOP for `approval.json`.
4. Do the runs, the selection, the confirmation and the analysis, with the independent re-run before opening.
5. Write the decision and the report, then STOP.

Sequence of the stages:
- **S1B:** §8, the 2×2 on registered T2. The two mechanism controls of §8.2 run only after a loop gain appears.
- **S2:** §9, step by step through the factor ladder.
  - `t2_typed` with `n_distract` already exists in `src/looplora/data/relations.py`. Check it against §9.4 before writing new generators.
  - The bridge, NULL and role labels go only to losses and metrics, never to `forward`, `read` or `stop`.
- **S3:** §10. Same-content runs, fixed-byte-budget runs, load E = 16/32/64/96 (out of distribution first), and deployment timing (§10.4). The real-task extension (§10.5) needs its own addendum after the others are reliable.

## Rules repeated here because they bit us before
- Pass `--commit` the full 40-character hash. `HEAD` resolves in the main checkout.
- debug-g allows at most 30 minutes. Use regular-g for anything longer.
- Outputs go to absolute paths under `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/`, because the job's snapshot directory is deleted.
- Exit 0 is not success. Check the steps, the evaluations, NaN/inf, and the config and commit hashes.
- Use blake2b and never SHA-256. Put no dates in code.
- Commit on wp/slot only. No merge, no push.

## Review round 1: fixes before the preflight (2026-10-29 20:40 JST)
**Handover.** From now on the worker for this task is a codex worker (wk-slot, GPT-6.1-sol high, herdr tab, same worktree and branch). It takes over from the subagent sn-slot (user request, 2026-10-29). Its starting point is wp/slot `ccdcc96` and `docs/reports/SLOT-LOOP-v1-state.md`.

**Reviews.** The static review of `ccdcc96` is on master `6c21e19`:
- `docs/review/static_2026-10-29/slot-data-analysis.md` (D-F1 to D-F5);
- `docs/review/static_2026-10-29/slot-model-training.md` (M-F1 to M-F4).
There is no high finding. Fix all of the items below in one round, because they are small and one cross-review then covers them. Merge master into wp/slot first.

**Before the preflight and S0:**
1. **D-F5 (c).** In `stage_s0`, R0 holds when `matrix_uncal/all_valid_question` differs from `native/all_valid_question` on S0 dev, in any per-sample correctness or NLL.
2. **D-F5 (a).**
   - Add `--manifest` (the locked `manifest.lock.json`, whose digest goes into the analysis output) to `scripts/analyze_slot_study.py`.
   - Refuse any set whose n or ids digest differs from the manifest. This covers S0 calibration 2,048, S0 dev 1,000, S1A dev 1,000, S1A confirmation 2,000 and the unseen set.
   - The test that says "a different n is reported, not refused" changes to "refused".
3. **M-F2.** Science modes refuse to run when the source they run differs from the source that passed the preflight.
   - Bind the lock to the `src/` content (blake2b) rather than to the commit, so edits to scripts or docs do not force a new preflight.
   - After any `src/` change, the preflight runs again. That job is small.
4. **D-F3.** Write the runbook commands with an interpreter: `.venv/bin/python` or `uv run python` for `prepare_slot_manifest.py`, and `python3` for the analysis.
5. **D-F4.**
   - `raw_episodes`/`episodes` refuse a sealed set that was not opened through `eval_set` with the token. Add a test that `episodes(SETS["S1A", "confirmation"])` raises.
   - Drop the sealed sets' `base_seed`/`query_seed` from `manifest()`. Their id digest and count stay.

**Before the S1A release (do them now as well):**
6. **D-F1.** Add the analysis stage `S1A-ABL`, with the reviewer's rule, which is registered here as Interpretation 9(f).
   - Inputs: `ablation/matrix_L1/s17/record.json` and `ablation/soft_slot_L1/s17/record.json`. Both must be `complete`, carry `ablation: true`, and share n and dev ids with each other and with the six main records.
   - Comparison: each run's dev-NLL-selected checkpoint, with the same selection rule as the main arms. Flag iff `(c_soft - c_matrix) * 100 >= 3 n` in exact integers. Print counts and the flag only.
   - `parse_run` requires `ablation is False` for the six main records.
7. **M-F3 / D-F2.**
   - Document the open-record schema in the state file.
   - `load_open_token` takes the S-RE reference digest from the locked inventory, so the record needs no seventh field. Add a test with a reference entry.
   - `--open-root` has no default and must be absolute.
   - Add a release builder script with self-checks. It carries:
     - the six main and two ablation config hashes;
     - the steps, the LR per arm and g/tau;
     - the walltime;
     - the prereg digest and the analysis-script digest.
   - `reload_check` also requires that the config equals the release's.
8. **D-F5 (b).** A missing reference file (`s_re_native`, `matrix_step0`, `soft_step0`) is refused.
   - The unseen set stays optional, as the prereg says ("descriptive, optional"). If its files are absent, the report says "not run", and the analysis does not fail.
9. **M-F1.** In the LR pilot, a non-finite candidate is excluded and recorded rather than aborting the pilot (prereg line 108). Add a test.
10. **M-F4.** No code change. Note in the state file that a first load long enough to make the N rule and the walltime cap contradict each other ends in a STOP, as it does now.

Then run the changed slot test files, and any existing test file whose imports changed, on compute. Update the state file (interpretations, schemas, runbook, ledger) and end with `STOP: review <hash>`. The review of this round is a cross-review by two independent reviewers: a codex reviewer (GPT-6-Astra xhigh) and an Opus subagent. After it, the steps continue as in Step 2: preflight, lock, S0, `decision_s0.md`, then `STOP: S0 done`.

## Review round 2: fixes before the preflight (2026-10-29 21:25 JST)
**Reviews.** Two reviewers cross-reviewed `2438599` independently:
- rv-code (codex GPT-6-Astra xhigh): `docs/review/static_2026-10-29/round2-rv-code.md`;
- rv-opus (Opus-5.5 xhigh subagent): `docs/review/static_2026-10-29/round2-rv-opus.md`.

Neither found code that opens a sealed set early or changes a registered number when used as registered. Round-1 items are done except item 7 (and item 4 for rv-code), which the findings below cover.
- Both reviewers found defects 1, 2 and 4 independently.
- rv-opus also found defect 3.

Fix all four now, before the preflight. Defects 1-3 change `src/`, and M-F2 binds the lock to `src/`.

1. **The reload check can never pass** (rv-code R2-C2, rv-opus R2-1; both reproduced it with the stdlib).
   - Cause: `save_export` stores `slot_data.TASK` with tuples, the JSON release holds lists, and `reload_check` compares them exactly.
   - Fix: store the slot metadata in one JSON-shaped form and keep the comparison strict.
   - Test: an integration test along the real path: smoke, then `smoke.json`, then `scripts/prepare_slot_release.py`, then the JSON release, then `reload_check` as the CLI calls it. It must fail before the fix.
2. **Confirmation rows are not bound to checkpoints** (rv-code R2-C1, rated High; rv-opus R2-2, rated Medium).
   - I rate it Medium. An entry error is needed to change a verdict, but the verdict record cannot show which checkpoint made which row, and the prereg needs that.
   - Keep the `arm/seed → digest` map in the open token.
   - In `evaluate_entries`, require each of the six main labels exactly once. Its digest must equal `selected_checkpoints[arm/seed]`. The reloaded export's `slot.arm`, `slot.run.seed` and `slot.run.ablation is False` must match the label.
   - Admit the S-RE digest only under the three reference labels.
   - Save the checkpoint blake2b-128 and the read in each evaluation file. Stage `S1A` checks them against the open record and refuses a mismatch.
   - Tests: two valid entries swapped; one checkpoint under two labels; the reference under a main label. Each must be refused.
3. **The analysis-script digest is not bound to the release** (rv-opus R2-3, Low).
   - `load_open_token` requires the open record's digest = the ready release's digest = the file's digest.
   - Stage `S1A` refuses when its own digest differs from the release's.
   - Test: a changed script with a matching open record is refused.
4. **Runbook step 4 cannot write the S0 decision** (rv-code R2-C3, rv-opus R2-4).
   - The job must run on the lock commit: the reviewed commit has no lock file.
   - Its output must be absolute and outside the job snapshot, which is deleted at exit (for example `R/s0/decision`). Copy `decision_s0.md` to the report location afterwards.
   - Write the `submit.py` argv without a second interpreter: `-- scripts/analyze_slot_study.py ...`.
   - Check steps 1-3 the same way, so each command runs exactly as written.

**Noted, no change:**
- The D-F4 guard is a workflow guard, not a security boundary (rv-code).
- The optional common-fixed-LR comparison has no config hash in the release, so `run_train` would refuse it (rv-opus). It is optional and descriptive. If it is ever run, it first needs a release entry.

Then run the changed test files on compute, update the state file, and end with `STOP: review <hash>`. Both reviewers then check the fix diff `2438599..<hash>` before the preflight.
