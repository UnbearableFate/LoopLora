# WP-L6 pre-run static review (branch wp/oracle, head 3a788b1)

Reviewer: read-only subagent. Date: 2026-10-27. Nothing was executed on a GPU and no torch/pytest was run (login node).
Checks that were run: `git diff master...wp/oracle`, full reads of every changed/new file and of the code they call,
py_compile (bytecode redirected away from the repo) of all 9 changed/new Python files, pyflakes (clean), an `ast` scan with
the login node's Python 3.9.25 of `scripts/run_wp_l6.py` and `scripts/run_p1l_main.py` (no 3.10+ runtime constructs), and
reads of the installed torch 2.14.0+cu132 Python sources where a message text or a determinism rule mattered.
No repo file, worktree file, branch or commit was touched.

## Verdict

No defect of consequence in the WP-L6 science code (oracle.py, train.py, config.py, run.py), the runner logic, the
configs, or the three test files. I found no input or state that makes the new writers, the optimizer wiring, the
decision rule or the sizing rule produce a wrong value, crash a registered run, or violate the registered protocol.
One low-severity runbook defect (F1, trivially fixed now). F2 lists concrete traps in the P1-L pipeline for the P1L-S
step 3 (all are inside the already-planned step-3 scope; none can affect the WP-L6 pilot).

## Findings, ranked

### F1 (low, runbook; certain) `scripts/run_wp_l6.py` is not executable but the runbook executes it directly
- Evidence: file mode is 100644 (`git ls-tree wp/oracle scripts/run_wp_l6.py`; `ls -l` shows `-rw-r-----`). Sibling
  login-node scripts `launch_p1l_main.py`, `p1l_job_ledger.py`, `submit.py` are 100755.
- `docs/reports/WP-L6.md:46` (`$W/scripts/run_wp_l6.py size ...`) and `:53` (`$W/scripts/run_wp_l6.py decide ...`) run it as a
  command on the login node. Failure: `bash: ...run_wp_l6.py: Permission denied`. The tests do not catch it
  (`tests/test_wp_l6_runner.py:176,183` call it through `sys.executable`). The `submit.py ... -- scripts/run_wp_l6.py ...`
  lines (44, 48, 50, 51) are fine: the job runs `$PYTHON_BIN` on the path.
- Fix: `git update-index --chmod=+x scripts/run_wp_l6.py` (plus `chmod +x`) and commit, or write the two commands as
  `python3 $W/scripts/run_wp_l6.py size|decide ...` (login-node python3 is 3.9.25; the script has
  `from __future__ import annotations` and passes the 3.9 scan).

### F2 (info; step-3 scope) P1-L pipeline traps for P1L-S, beyond the per-group `writer_hash` already noted
Line numbers are in the wp/oracle worktree; sn-p1l's step-A changes may move them. Failure scenario for each is "P1L-S
run through the P1-L script unchanged":
1. `scripts/finalize_p1l_main.py:131,155-156` one `writer_hash` across the 5 primary readers; `:132-138` identical memory
   snapshots/write counts across readers and seeds. P1L-S mixes writers: reused B3-L4 (trained and evaluated under
   oracle_random) vs new runs under W, so even for O-rot the memory hashes differ and the check raises
   ("oracle support snapshots/write counts differ ..."); under O-shared every run has its own C, so both checks fail.
2. `finalize_p1l_main.py:107` requires `episode_ranges == {"confirm": {split: "test", start: 100000, count: 2000}}`;
   `src/looplora/llm/evaluate.py:144,259` and `scripts/evaluate_p1l_main.py:73,131` hard-code `("test", 100000)`. If only one
   side is parameterized, P1L-S either fails finalize or, worse, silently evaluates on P1-L's range [100000, 102000)
   instead of the registered [110000, 112000). Suggest one constant, taken from the lock/authorization and recorded in
   `metrics["episode_ranges"]`; finalize compares against the locked value.
3. `finalize_p1l_main.py:54` (`endpoints()` result `writer="oracle_random"`) and `:121` (accuracy rows `writer="oracle_random"`)
   would label W runs as oracle_random without error (silent mislabel in endpoints.json / accuracy.csv).
4. `scripts/evaluate_p1l_main.py:28-48` `check_training` requires all 21 runs plus B0-s17, `gated_conditions >= 2`, and both
   `oracle-r32` and `oracle-r128` manifests; P1L-S has 12 new runs plus 3 reused B3-L4, no B0/B1/B2, k = 1.
   `finalize_p1l_main.py:157` already derives the CI level from `gated_conditions`, so k = 1 gives 95% once check_training allows it.
5. `scripts/launch_p1l_main.py:20` rejects any job that would end on or after 2026-10-28 00:00 UTC ("beyond the maintenance
   deadline"); maintenance ends 2026-10-28 22:00 JST, so every P1L-S launch through this script is refused until the guard
   is removed. `:31` passes `--scale-up docs/prereg/P1L.md` (P1L-S registers `docs/prereg/P1L-S.md`); `:53` and `:99` hard-code
   `reports/p1l/lr-S-lock.json` and condition `oracle-r{rank}`; `scripts/run_p1l_main.py:43,131` hard-code `configs/p1l` and
   the lock path.
6. `finalize_p1l_main.py:29-55` `endpoints()` has `G1_hop1` (line 33) (B8 - B4) but not the registered memory-use check of P1L-S
   (hop-1 full EM of B4@1, B8@4, M1-L4@4 each > B3-L4@4, CI > 0). It must be added before P1L-S interpretation.
7. `scripts/evaluate_p1l_main.py:200-202` `inference.json` counts adapter parameters only; for O-shared the 32,800-parameter C
   is not in it (the prereg asks for C to be counted separately), so add it from `_parameter_counts`' `external_writer`.
8. `evaluate_p1l_main.py:235,242` read `configs/p1l/main/eval-confirmation.json` / `eval-typed-replication.json` templates.

### Observations (not defects)
- `t2_chain_end` writes relation id 0 for every base edge (`data/relations.py:_layout`, facts zeros), so the pilot exercises
  only Pi_0; Pi_1..Pi_3 are first used by P1L-S's `t2_typed` distractors (relations 1..3). The unit tests check all four
  rotations at the writer level; a pilot pass says nothing about Pi_1..Pi_3 on the LM path.
- The timing smoke evaluates the full 1,000-episode set (derived eval seed), so score tables exist under `<smoke>/<arm>/validation`
  on disk; the runner never reads them (`fit(..., evaluation="timing")` keeps only the time). As registered.
- With P1-L's measured step time (0.197 s) one arm is about 1,190 s, so the pack rule (1.5 x (sum + 90 s) <= 3600 s) will
  probably recommend separate jobs (about 33 min each, regular-g). Both outcomes are handled by `size`.

## Verified OK (with evidence)

Codes and writers (`src/looplora/llm/oracle.py`)
- `relation_rotations` (41-52): local seeded generator `derive_seed(seed, "oracle", "rotation", rel)`, float32 Gaussian, float64 QR,
  columns scaled by `sign(diag R)` (broadcast over the last dim = column scaling), giving the unique positive-diagonal Q, so the
  result depends on the seed only; relation `rel` does not depend on `n_rel`; global RNG not consumed; stacked [n_rel, r, r] float32.
- `RotatedOracleWriter.fact_codes` (78-83): invalid slots are zeroed (all three columns) before any table is indexed, so -1
  sentinels never index; key = normalize(Pi_rel @ code(h)) (unit), value = code(t) (O-rot: the unit global codes; O-shared:
  normalize(C_t)); `forward` (85-90) writes in list order with `delta_update(..., lam=1.0, beta=1.0)` and the valid mask, from
  `zeros_state(batch, r_mem, float32, device)`. With unit keys this is the exact overwrite `M <- M + (v - Mk)k^T`.
- `OracleSharedWriter` (97-112): `C = nn.Parameter(global_codes.clone())` (1025 x 32), `entity_codes()` returns C, so keys and values
  are differentiable in C; `.to(device)` keeps the Parameter object, so the optimizer built afterwards holds the live tensor;
  state_dict keys {C, global_codes, rotations}; rank guard {32, 128}.
- `OracleRandomWriter` (13-38) is unchanged. All oracle_random paths produce the same objects as before: `ORACLE_WRITERS`
  membership gives the same booleans, error texts for oracle_random are character-identical, `groups = params` (same list)
  when `writer_params` is empty, `clip_grad_norm_` gets an equal list, the frozen-writer snapshot check still runs, summary
  keys and `writer_signature` unchanged. Off-path test (baseline dc659e0): the four changed files at dc659e0 equal the merge-base
  versions (no `src/` diff dc659e0..merge-base or merge-base..master), so the "baseline copy" really is the pre-change source.

Training wiring (`train.py`, `run.py`, `config.py`)
- Gradient reaches C: `prepare()` (train.py:215-221) calls `self.writer(facts, valid)` in grad mode in the training loop; the state
  goes through `repeat_snapshot` (batching.py:101, `repeat_interleave`, documented to retain the autograd graph); no `detach` /
  `no_grad` in the read path (core/adapters.py, wrapper.py, inject.py, protocols.py except the inference-only `greedy_answer`).
- Optimizer: `[{"params": reader}, {"params": [C], "weight_decay": 0.}]` with default lr and weight decay for the reader group
  (train.py:294-299); the per-step warmup loops over all `optimizer.param_groups` (so C gets the same lr and warmup); clipping is
  `clip_grad_norm_(reader + [C], clip, error_if_nonfinite=True)` (324). Rows of C that never appear in a fact (names >= 768,
  row 1024, the sentinel row 0) receive exactly zero gradient and, with weight decay 0, an exactly zero Adam update, so they stay
  bitwise at init (as the prereg and `test_train_export_reload...` assert). B3 never reads the memory: `C.grad` stays None and
  both clip and AdamW skip it.
- Frozen-writer assumptions: the "frozen writer changed" check is skipped only when `writer_params` exist; `writer_frozen` is False
  for O-shared; nothing else in `src/looplora/llm` consumes `writer_frozen`; `writer_state()` is the full state_dict, so C is in
  `writer.pt` and `checkpoint.pt`; `_load_writer` checks signature, key set, shapes, fp32, then `load_state_dict(strict=True)`.
  Prefetch workers produce only numpy episodes/tokens (`include_support=False`); the memory is written in the main process from the
  current C every step and every evaluation batch (`prepare()`); no cached snapshot or stale-C path exists in train/evaluate.
- `run.py`: `_parameter_counts` counts C once as an external writer parameter (reader keys unchanged); record `writer`,
  `writer_signature`, `writer_hash` for every ORACLE_WRITERS member; `_comparison_groups` oracle_seed key; `compare_suite` skips
  O-shared in the frozen-hash check and drops the `x,u` fields for oracle writers.
- Determinism mode: the new backward ops (index/index_put with accumulate on CUDA, `repeat_interleave`, bmm) are not in torch 2.14's
  list of ops that raise under `use_deterministic_algorithms(True)` (read from `torch/__init__.py` docstring).

Runner and configs (`scripts/run_wp_l6.py`, `configs/wp_l6/*.json`)
- Constants: STEPS 5666 = 17000 // 3, SMOKE_STEPS 567, THRESHOLD 0.30 inclusive, ARMS order (rot first). `arm_outcome` reads only the
  registered cell (`memory_only/original`, hop 1, L 4, `full_accuracy`, n == 1000) after checking the full registered
  configuration; every key it reads exists in the raw JSON and in `resolve_config` output (`task.seen` is added by TaskSpec in
  `resolve_config`, so `code_drift` finds it). `decide` writes `status` selected/stop and `selected` = the arm name (the P1L-S key).
- Failure classes: non-finite loss (`FloatingPointError`, train.py:322), non-finite evaluation (`FloatingPointError`, evaluate.py:39),
  non-finite grad norm (`RuntimeError: ... is non-finite, so it cannot be clipped`, text confirmed in the installed
  `torch/nn/utils/clip_grad.py:111-114`) are all classified as divergence; `train()` closes the experiment on any exception
  (train.py:384-386), so the other arm reuses the backbone; other errors raise and `arm_outcome` demands an unchanged rerun.
- `size`: pack rule and `hhmmss` rounding match the prereg and the report; arithmetic re-derived for the test cases.
- Data: configs differ from `configs/p1l/main/M1-L4-s17.json` only in name, writer, train.steps, eval.depths; `eval.splits == ["val"]`,
  seed 4242, 1,000 episodes, `allow_confirmation` false; the smoke forces derived training and eval seeds
  (`derive_seed(20261027, ...)`, `resolve_config:171-172`) and `resolve_config` rejects any non-val split in smoke; nothing in the
  runner reads test or confirmation data.
- Login-node safety: module-level imports are stdlib plus `run_p1l_main` (stdlib only); torch is imported lazily
  (`test_import_is_login_safe` asserts it); Python 3.9 compatible.

Tests (desk-checked only)
- Tolerances against float32 error budgets (Gram 2e-6, Gram-Schmidt comparison 1e-5, state vs float64 reference 1e-5, lstsq 1e-4);
  `patch` targets exist (`looplora.core.memory.scan_final`, `looplora.llm.train.encode_support`); the AdamW/clip patches snapshot
  groups before AdamW fills defaults; error-message `match=` strings agree with the code ("hashes differ", "signature|fp32|missing");
  RECORDED_CODE_HASH is the value in `reports/llm-oracle/benchmark.json` (3 cells, r_mem 32, V 1024) and is computed by the same
  formula as `scripts/benchmark_llm_oracle.py:86`; every test file named in the runbook exists. The mutation sensitivity of the main
  tests (sign fix, mask, group, clip list, checkpoint C) is adequate: each realistic bug flips at least one assertion.

## Could not verify (needs the GPU)
1. That the tests actually pass (nothing ran). In particular the tiny-model tests' bitwise equalities (`atol=0` after reload) rely on
   same-process kernel determinism.
2. Cross-process bitwise equality of the off-path probe (baseline vs current): `configure_numerics()` runs in `LLMExperiment.__init__`,
   i.e. before the probe's first cuBLAS call, so it should hold; in the pytest process the writer-level CUDA tests run before any
   `LLMExperiment`, so cuBLAS may initialize before `CUBLAS_WORKSPACE_CONFIG` is set (affects only cross-process reproducibility, not
   the in-process comparisons).
3. CPU RNG/QR bit patterns are platform-dependent; the recorded code hash and the pilot both come from Miyabi-G (aarch64), the same
   platform as the tests.
4. O-shared step time and the packed/separate recommendation (the smoke measures it); peak memory.
5. The real `index_put`-accumulate path on CUDA under deterministic mode (the tiny-model O-shared tests exercise it first).
