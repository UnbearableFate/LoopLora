# S1B static review — rv-code

- Reviewer: **rv-code**; date: **2026-10-29**.
- Reviewed commit: `67deeee5db78146d5eb9a9b55c93b081399f1c3a` (`wp/loop`), verified in the clean detached checkout `/work/xg24i002/x10041/LoopLora-wt/review-loop`.
- Scope: `a9178d5...67deeee5db78146d5eb9a9b55c93b081399f1c3a`, following `s1b_brief.md` from main `1d7fa24aea7f53057b5dffbd280cd24d2906ffad`. Read both complete preregistrations, WP-L14's task and ledger, plan §8, all changed implementation/tests/scripts, and relevant inherited reader, trainer, data, preflight and analysis code. The other review was not read.
- Checks run: Git identity/status/diff inspection; `git diff --check` (passed); stdlib AST parsing of all 14 changed Python files (passed); blake2b-128 checks of both preregistrations (matched `6f2e776f2c6cfde131380fb343854784` and `56e8c611c01e0ef9b7ae32db97b0e334`); bounded NumPy-only synthetic bootstrap checks, including the production `paired()` function extracted through AST without importing application modules. No torch was imported.
- Execution plane: confirmed `miyabi-g2`, aarch64, login/control, no allocation. Only this review file was written. Codegraph was unavailable in the tool catalog; inspection used Git, rg and bounded file reads.

## Verdict

**Changes required before locking this implementation. Findings: 1 High, 4 Medium, 0 Low.**

The bootstrap can falsely declare a storage advantage and consequently choose the wrong S2 primary backend at an exact-zero confidence boundary. This was reproduced with 2,000 synthetic episodes, the registered seed, 10,000 draws and the registered three-contrast quantile. The remaining findings concern evidence that the normal pipeline discards or fails to establish. They are not claims that a real model reload has already failed.

The isolated modules leave the inherited source files unchanged. Static inspection supports the intended separate data namespace, mechanical branch choice, common-parent loading, trainable set and matched training schedule; it does not replace the required compute preflight. Fix validation should follow the brief's compute-test process, without another review round.

## Findings

Paths and line numbers below refer to the reviewed detached checkout.

| ID | Severity | File:line | Concrete failure | Evidence | Suggested fix | Required by |
| --- | --- | --- | --- | --- | --- | --- |
| RVC-01 | **High** | `src/looplora/llm/s1b_analysis.py:70`, `:80`, `:84` | Dividing episode differences by six before resampling introduces cancellation error. A mathematically zero lower confidence bound becomes positive, so `C_storage` passes and `s2_primary` becomes `soft` when the registered rule would choose `matrix`. | Reproduction below: exact lower bound **0**; returned lower bound **2.2204460492503132e-17 pp**, `lower_positive=True`, storage `passed=True`. Both backend loop/sweep screens pass. §5 requires exact comparisons from counts. The inherited analysis already uses integer bootstrap sums and exact percentile interpolation (`scripts/analyze_slot_study.py:186`). Existing S1B bootstrap tests use an integer-valued 0/1 episode vector and do not exercise cancellation at zero. | Resample the integer sum of the six seed/hop correctness differences. Apply the registered linear percentile interpolation and positivity comparison in exact arithmetic; divide by `6*n` only for reported values. Keep the shared draws and registered quantiles. Add this zero-boundary regression, including the resulting continuation choice. | **Before preflight**, so the corrected analysis is included in the source lock. |
| RVC-02 | **Medium** | `src/looplora/llm/s1b_run.py:102`, `:104`, `:169` | The calibrated smoke→release→CLI reload gate compares a reloaded model with another reload. A deterministic export/load defect that changes the smoke model identically on every reload can pass this gate; it does not establish reproduction of the original smoke predictions/NLL. | `smoke()` closes the trained experiment at line 101, then creates `reload_samples.npz` from `model.load_export()`. `reload_check()` loads again and compares against that file. No samples from the original calibrated smoke instance are compared. `test_s1b_integration.py:21` follows this same path; `test_s1b_model.py:160` likewise compares two loads. The separate preflight has an in-memory comparison, but uses its own uncalibrated runs, not the calibrated smoke exports bound into the ready release. §4 requires the actual smoke/export/reload chain. | Capture the selected smoke checkpoint's IDs, predictions, correctness and NLL from the original in-memory instance before closing it. Compare both the initial load and CLI reload to those samples. Test that a deterministic altered reload fails, even when repeated reloads agree with each other. | **Before preflight** and the subsequent smoke/reload gate. |
| RVC-03 | **Medium** | `src/looplora/llm/s1b_train.py:97`, `:101`; `src/looplora/llm/s1b_analysis.py:35` | An improving checkpoint overwrites the preceding candidate before the independent selection audit. For example, an improvement at step 1200 destroys the step-600 export; non-improving scheduled checkpoints are never exported. The retained artifacts cannot reproduce the candidate comparison, and dev selection copies the final digest without binding each dev row to its evaluated export. | The only save target in `fit()` is `selected.pt`. Dev rows contain summaries, not checkpoint digests or retained per-checkpoint sample files. `select()` re-derives a step from those summaries but accepts `selected_checkpoint_blake2b_128` separately. §7 explicitly retains checkpoints needed to reproduce selection until its audit; §3 requires checkpoint provenance. The trainer tests assert only that the final selected export reloads. | Retain the scheduled candidate exports until the independent audit, bind each dev row to its export digest/step and verify that the selected file matches the winning row. Keep selection as a reference/copy to the retained winner; defer cleanup until the registered audit permits it. Exercise multiple checkpoint improvements in a retention/provenance regression. | **Before formal runs**; land before the first preflight to avoid invalidating its lock later. |
| RVC-04 | **Medium** | `src/looplora/llm/s1b_model.py:243`, `:247`; `src/looplora/llm/s1b_run.py:242` | Every diagnostic evaluation discards the actual captured `q_s`, `m_s`, `z_s` trajectories. Saved samples contain scalar diagnostics and `trace_agrees=True`, so the requested final-token trajectories cannot be inspected or independently checked from the retained artifacts. | `trace_diagnostics()` returns errors, norms, weights and drift summaries only. `evaluate()` appends these to samples and then clears `adapter.trace` at every chunk. `evaluate_selected()` persists only those samples. §5 explicitly says to save actual query/read/core trajectories at every loop of the final valid token. Existing tests inspect the in-memory trace or scalar metrics, not the saved trajectory artifact. | Persist the compact final-valid-token q/m/z tensors for each requested depth, with episode/hop/step ordering and provenance. Keep transient full-token verification state out of the artifact. Add a saved-artifact check for complete ordered trajectories and agreement with the captured forward. | **Before formal runs**, with the persistence implementation included in the preflight/source lock. |
| RVC-05 | **Medium** | `src/looplora/llm/s1b_run.py:297`; `src/looplora/llm/s1b_analysis.py:110` | Successful confirmation analysis produces a verdict without the registered confirmation count, per-arm/per-seed/per-hop EM and NLL tables, or the descriptive S1−M1 and Memory×Depth interaction. Thus the normal frozen analysis stage does not produce the required metric report, including after a failed contrast. | `evaluate_selected()` drops `result['summary']`; `paired()` returns contrasts and continuation screens only. `analyze_confirmation()` adds identities and optional **unseen** per-hop summaries but no confirmation summaries. The raw NPZ arrays retain enough information to recover most of these quantities, but no supplied frozen analysis path performs that reporting. §5 requires full counts/NLL/per-hop EM irrespective of pass/fail and specifies the descriptive contrasts. Tests check verdicts/intervals, not completeness of this output. | Have the frozen confirmation analysis emit the registered confirmation tables and descriptive contrasts directly from the validated paired samples, separately labeled from inferential decisions. Preserve metric-free stdout. Verify report completeness for both passing and failing outcomes. | **Before formal runs**, with the reporting code frozen before confirmation opens. |

### RVC-01 reproducible evidence

The NumPy-only check executed the reviewed `paired()` body with its actual `branch_arms()` and `contrasts()` helpers, extracted through AST. It used no generated graphs or scientific observations.

Construct the integer episode contrast vector:

```python
n = 2000
v = np.roll(np.r_[np.ones(1000, dtype=np.int64),
                  np.zeros(103, dtype=np.int64),
                  -np.ones(897, dtype=np.int64)], 3)
```

Start all four arms' correctness arrays at zero, shape `[3, 2000, 8]`. For the positive positions of `v`, assign one S4 hop-2 correct answer, cycling over seeds 17/29/43 in position order. For the negative positions, assign one M4 hop-2 correct answer with the same seed cycling. Keep independent M1/S1 at zero. For each trained-L4 sweep, use zero correctness at L1/L2 and its primary correctness at L4/L8. All rows have identical IDs and hops 1–8. Then the storage episode difference is exactly `v/6`, and every seed's storage difference is positive.

With `default_rng(5799536610111230359)`, 10,000 draws in the production blocks of 500, the two sorted integer bootstrap sums bracketing quantile `1/120` are **both zero**. Exact linear interpolation therefore gives zero. Production returned:

```text
C_storage delta_pp:        0.8583333333333333
C_storage lower_pp:        2.2204460492503132e-17
C_storage per_seed_pp:     [0.875, 0.85, 0.85]
C_storage passed:          True
Matrix / SoftSlot qualify: True / True
s2_primary:               soft
```

Under the exact registered comparison, storage fails (`0 > 0` is false), while both loop/sweep qualifications remain true; the primary-backend rule then selects Matrix. Changing only the numerical representation changes a registered decision.

## Not checked

- No torch import, pytest, model execution, training, evaluation of real episodes, PBS query or submission. Real-parent GPU identity, gradient behavior, freeze checks, memory usage and timing remain compute checks. The ledger's 160-test result was read, not independently rerun or treated as evidence that the uncovered cases pass.
- No sealed S1A/S1B data or other review report was opened. Full registered-prefix disjointness was inspected in code, not re-executed on the login node. Parent tensor files, the complete preflight, formal releases and actual selected checkpoints were not loaded or validated.
- No scientific conclusions are drawn from the synthetic correctness fixture. No code fixes or new test files were written, and no branch or commit was changed.
