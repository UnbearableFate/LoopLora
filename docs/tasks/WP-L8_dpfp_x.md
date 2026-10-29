# WP-L8: DPFP-X, the best DPFP writer at r = 256 and at Qwen3-1.7B-Base (sn-llm; LoopLora-wt/llm, wp/llm)

Read first:
- `docs/tasks/_common_rules.md`
- `docs/prereg/DPFP-X.md` (registered on master; the design and decision rule are fixed)
- `docs/prereg/DPFP.md`
- `docs/reports/DPFP.md`
- `docs/tasks/WP-P5L_second_scale.md` and `docs/reports/P5L-state.md`. The `backbone` config option is there, kept
  outside DEFAULTS. The tiny-model backbone test is there too.

## Step 1 (now)
1. **Merge.** Merge master into wp/llm. Use master at or after the commit that registers DPFP-X.
2. **Code.**
   - Add the two arms X256 and X17 to the DPFP runner and release scripts, with the same schedule rule, checkpoints,
     records and inference export. Add `backbone` to the DPFP configuration, using P5L's option.
   - The three DPFP arms keep their config hashes, and the default backbone is unchanged. A test must prove both.
   - Lift any rank guard only for X256.
   - Size every shape from the loaded model config: d_in, d_out, r and key dimension 2·r·ν.
   - `choose()` stays as it is for DPFP. Write the DPFP-X decision as its own function, following the prereg's rule.
3. **Tests on compute.**
   - The existing DPFP tests and `tests/test_llm_dpfp_repro.py`.
   - Tiny-model tests: the r = 256 path, and the backbone option through the DPFP runner.
   - `tests/test_audit_dpfp.py` with an X-arm export.
4. **Smokes.** Run a 10% smoke of each arm for time and memory only. Read no score. Then do reload checks as in DPFP.
5. **Release.** Write `reports/p1l-dpfp-x/release.json`. It holds the step counts from the rule, the checkpoints, the
   source commit, the prereg's blake2b hash and the config hashes. Commit it.
6. **STOP** with the release path, the smoke timings, peak memory and projected node·h.

Budget: steps 1–5 ≤ 0.6 node·h together with the audit later. Each job is 1 node; science jobs ≤ 2 h with
`--scale-up docs/prereg/DPFP-X.md`.

## Step 2 (after the orchestrator's OK on the release)
Run X256 and X17 in parallel at the released source, then the zero/shuffled-memory audit. Apply the decision rule
exactly as registered. Write the report and the state file:
- `docs/reports/DPFP-X.md`: the decision, curves, the matched-fraction table, diagnostics and audit, jobs and node·h.
- `docs/reports/DPFP-X-state.md`.

STOP when committed. No merge, no push.
