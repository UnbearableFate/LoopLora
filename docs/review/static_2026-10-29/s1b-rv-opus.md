# S1B core-code cross-review: rv-opus

## 1. Header

| item | value |
| --- | --- |
| reviewer | rv-opus (Opus-5.5 subagent, effort xhigh), independent of rv-code; `s1b-rv-code.md` not read |
| date | 2026-10-29 |
| commit | `wp/loop` `67deeee5db78146d5eb9a9b55c93b081399f1c3a`, read-only detached checkout `/work/xg24i002/x10041/LoopLora-wt/review-loop` |
| scope | `git diff a9178d5...67deeee`: `src/looplora/llm/s1b_{data,rules,model,train,run,artifacts,analysis,preflight}.py`, `scripts/{prepare,analyze}_s1b.py`, `tests/test_s1b_*.py`. Also the inherited code these call: `slot_read`, `slot_experiment`, `slot_data`, `slot_rules`, `slot_preflight`, `writer_fix`, `core/adapters`, `core/memory_backend`, `data/relations`, `sym/evaluate.stratify_queries`, `llm/prefetch`, `llm/train.score`, and the `submit.py`/`job.pbs` source export |
| authority | `docs/prereg/SLOT-LOOP-v1-S1B.md` (blake2b-128 `6f2e776f…`), decisions 1-6; parent `SLOT-LOOP-v1.md` |

Checks run (login node, no torch, no pytest):
- **stdlib:** all 11 S1B base seeds and the §2 layout digest `f556c1b9…` re-derived; all match `s1b_data.py`. Measured `derive_seed` at 2.8 µs, so `data.manifest()` (about 2.1 M IDs, called by every `require_lock`) costs about 6 s per job.
- **numpy** (`.venv`, `-B`):
  - Chunked `default_rng(5799536610111230359).integers` draws (500 rows at a time) equal one 10,000-row draw (checked at n = 2000).
  - `(.05/3)/2 == 1/120` and `1-(.05/3)/2 == 119/120` hold exactly.
  - Float means of resamples whose exact integer sum is 0 were tested: 281 of 338 are nonzero, 142 of them positive.
  - Simulation of borderline datasets for F1 (below).
- **git:** outside the new `s1b_*` files, `src/` is identical at `67deeee`, master `1d7fa24` and `wp/slot`. On `wp/slot`, `scripts/analyze_slot_study.py` differs only by a release-digest field. The `stage`/`verdict`/`outcome`/`reasons`/`flag` fields that `resolve_branch` reads are unchanged there.

## 2. Verdict

The S1B science path is sound and I found nothing that invalidates a run. These items match the addendum:
- **Data isolation and seals:** the registered seeds, full-prefix disjointness, refusal of sealed generation, and the separate S1B open token.
- **Branch:** comes mechanically from the S1A records. `A_query` exists and trains only in `query_copy`. `A` stays frozen.
- **Identity:** the L4 recurrence is operation-for-operation the native `IterativeQueryAdapter._read`. Capture only appends.
- **Training:**
  - Frozen-set and support checks run at every checkpoint.
  - N = 6000 with identical `BatchStream` batches per seed.
  - The LR pilot and its stop rule, and the dev schedule with its 1e-6 tie rule.
- **Calibration:** grid and tie rule, the shared L1/L4 read, and the hop-1 < 10 % stop, which `require_calibration` re-checks.
- **Release gates:** lock bound to the `src/` digest, `pending_reload` cannot train, approval bound to the exact release digest, and the ≤ 16 h walltime check.
- **Statistics:**
  - d_i averages seeds before resampling and uses hops 2-3.
  - One whole-episode index draw is shared by all contrasts, with the Bonferroni quantile levels.
  - Δ ≥ 3 pp, hop-1 and per-seed tests use integer counts.
  - The continuation and S2-primary logic matches §5.

One defect can flip a registered decision, though only in an exact-tie case: the lower-bound sign test (F1). One Medium (F2) concerns registered outputs that the frozen analysis does not produce. Both live in `src/`, which the lock freezes, so fix them **before the preflight/lock**.

## 3. Findings

| id | sev | file:line | concrete failure | evidence | suggested fix | fix by |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | **High** (rare trigger) | `src/looplora/llm/s1b_analysis.py:70,76,80,84` | **Mechanism.** The test "adjusted lower bound > 0" (loop gain M4−M1 / S4−S1, and C_storage) is a float comparison, not an exact one from counts as §5 requires. `d_i` is a float mean of six ±1/0 values (multiples of 1/6, line 70), and each resample mean is a float mean of 2000 of them (line 76). When a resample's exact sum is 0, its float mean comes out as ±1e-17 in most cases. **Effect.** If the order statistics at the lower quantile position (index 83.325 for 1/120, 249.975 for .025) are exact zeros, the exact lower bound is 0 and the registered test fails. Float noise can make `interval[0] > 0` True (line 84). That turns a failed contrast into a pass: a false "confirmed loop gain", a false C_storage pass, and possibly a changed `s2_primary`. | **Exact-zero resamples** (numpy, the code's own formula): 281 of 338 have a nonzero float mean, 142 of them positive. **Borderline datasets** (lower bound within about ±1 SE of 0, registered seed, 10,000 draws, three-contrast level): 5 of 1188 had an exact lower bound of 0, and the float and exact decisions differed in 3 of 1188. A difference can only be a false pass: when the exact bound is > 0 the interpolation weight is positive, so the float bound is positive too. **Tests** use 0/1 fixtures whose float means are exact, so they cannot catch it. | Bootstrap the integer numerators: `num = diff[:, :, 1:3].sum(axis=(0, 2))` (int64), `S = num[indices].sum(axis=1)`. Take `np.quantile(S.astype(float64), [a/2, 1-a/2], method="linear")`, which is exact on integer-valued floats ≤ 12,000, and test `q[0] > 0` on that. Report `100*q/(6n)` as pp; this equals the registered percentile of resample means, since scaling commutes with linear interpolation. Add a tie fixture whose exact lower quantile is 0 and assert `lower_positive` is False. | before preflight/lock (analysis code is in `src/`; its digest goes into the lock, the release and the open record) |
| F2 | Medium | `src/looplora/llm/s1b_analysis.py:90-92,110-112`; `src/looplora/llm/s1b_run.py:297-326` | §5 says "Report full intervals, counts, NLL and per-hop EM irrespective of pass/fail; S1−M1 and the Memory×Depth interaction are descriptive". The frozen confirmation analysis emits per contrast only Δ, interval, per-seed Δ_s, hop-1 difference and tests, plus unseen per-hop. It never reads `nll`. It gives no per-arm or per-hop EM or correct counts, no per-depth sweep EM, no S1−M1 and no interaction. The frozen `verdict.json` therefore cannot support the registered report. After confirmation opens, these numbers would need unfrozen post-hoc code on sealed outputs. Adding them to `src/` changes the source digest, so `require_lock` refuses the frozen analysis under the existing lock. | `paired()` consumes only `sample["correct"]` (lines 63-69). Its result keys (lines 90-92, 110-112) contain none of the listed outputs. | Add a descriptive block to `paired()`/`analyze_confirmation`: per arm, seed and hop, `correct` counts, EM and float64 NLL means, the sweep's per-depth per-hop EM, S1−M1 point Δ and (S4−S1)−(M4−M1). Make it descriptive only, with no effect on `passed`. | before preflight/lock |
| F3 | Low | `src/looplora/llm/s1b_run.py:102-106,169-172`; `tests/test_s1b_model.py:160-171`; `tests/test_s1b_integration.py` | The smoke saves `reload_samples.npz` from the **reloaded** export, not from the trained in-memory model. `reload_check` then compares a second reload against it. The release's reload reports therefore show only that reload is deterministic, not that "reload reproduces ordinary-smoke … predictions … and NLL" (§4). An export/reload path that consistently diverged from the trained model would still yield a `ready` release. | `smoke()` evaluates `restored` (line 104) and never compares it with the in-memory step-20 dev row. The S1B tests also compare reload with reload. The preflight compares in-memory with reload (`s1b_preflight.py:97-110`, uncalibrated reads, 8 episodes), so the code path itself is covered once. | In `smoke()`, evaluate the in-memory experiment after `fit` (step 20 is the only selectable step, so it is the selected weights) and save those samples. Compare the reload against them, then hand them to `reload_check`. | before the ready release |
| F4 | Low | `src/looplora/llm/s1b_rules.py:33-39` | `resolve_branch` accepts an ablation record even when the S1A dev outcome is `no_trigger`. With `verdict="NOT ADVANCED"`, a stray flagged `S1A-ABL` file (for example from a test or another run) selects `query_copy` although no ablation was triggered. That breaks the mechanical branch rule (§1, decision 4). Nothing binds the three records to one S1A run, such as a shared release or manifest digest. | Line 35 checks only `stage`/`flag`. Line 39 uses `ablation["flag"]` whatever `dev["outcome"]` is. | Refuse `ablation is not None and dev["outcome"] != "ablation"`. Also require that decision, dev and ablation share the S1A release digest; the `wp/slot` S1A-DEV summary now carries `release_blake2b_128`. | before calibration |
| F5 | Low | `src/looplora/llm/s1b_run.py:66-72`; `src/looplora/llm/s1b_model.py:140-141` | The inherited S0 calibration rule excludes nonfinite candidates (`slot_rules.choose_candidate`, `slot_rules.py:47`). In S1B, `per_hop` raises `FloatingPointError` on any nonfinite NLL, and `calibration()` does not catch it. One nonfinite candidate (for example g=2 or τ=0.03) aborts the whole calibration instead of being excluded. | `calibration()` calls `model.evaluate` with no try (line 70), and `per_hop` raises (lines 140-141). | Catch the error per candidate and record `nll=None` with the error; `json` `allow_nan=False` forbids NaN. Make `choose_candidate`/`require_calibration` treat `None` as excluded. | before calibration |
| F6 | Low | `src/looplora/llm/s1b_preflight.py:68,74,76,79` | The lock's checks record less evidence than the §8 table requires: <br>- `legacy` is only the S1A L1 native replay. "Existing T1 replay/golden cases at L1/2/4" through the S1B adapter are not run, although `slot_preflight.matrix_identity(..., depths=(1, 2, 4))` exists for this. <br>- `owner_mask` runs at L1 only, so the depth forks are not checked. <br>- The `[B,12,768]/[B,12,128]` capture shapes are not asserted (`bytes_check` with capacity ≠ 16 checks only the formula of the actual dims). <br>A passed lock therefore claims §8 compatibility without the listed evidence. | See the lines. | Add `checks.matrix_identity(exp, checks.old_range(exp), depths=(1, 2, 4))` per branch, the owner check at L = 4, and asserts that `batch.keys.shape[1:] == (12, 768)` and `batch.values.shape[1:] == (12, 128)`. | before preflight |

Note on F1's probability: it fires only when the exact lower bound is exactly 0, which in the borderline simulation was about 0.25 % of datasets. When it fires, it always errs toward a pass. All other registered threshold tests (Δ ≥ 3 pp, hop-1 ≤ 2 pp drop, per-seed > 0, storage Δ > 0, the sweep screen) are exact integer comparisons and are correct.

## 4. Not checked

- **Runtime.** Nothing was executed with torch: login-node rule. I did not run GPU bitwise identity, `verify_trace` determinism, the real-parent `s1b_preflight.run`, or memory and time of diagnostics at 2000 × 8 × L8. `s1b_preflight.run` is not executed by any test.
- **Prefetch path on T2.** The training `BatchStream` path (`workers=2`, CPU tokens, so `writer_fix.token_fact_positions`) is exercised by no S1B test, because all S1B tests pass `source=`. It is first used in the LR pilot. By reading, it handles E12/(8,4) like T1, but this is not executed evidence.
- **Old windows' layouts.** I did not verify whether the inherited old seed-4242 windows were originally generated under T1. `old_ids()` uses the T1 digest. S1B base seeds differ from all old ones, so ID disjointness holds either way.
- **Real records.** I did not inspect real S1A output records (none exist yet); `resolve_branch` was checked against the S1A analysis script's schema only. The plan document §8 was not read; the addendum governs.
- **Out of scope.** Cost and walltime arithmetic beyond the formula (`walltime`, `control_release`), and queue choice: these are orchestrator operations.
