# Review of WP-L7 step 1c (`git diff 17b27c2 0a3048a`, branch `wp/llm`, worktree `/work/xg24i002/x10041/LoopLora-wt/llm`)

Read-only static review plus stdlib-only executions on the login node (system `python3` 3.9.25, `-I`, no torch, no pytest, no
writes outside the scratchpad). Scope: the release path that turns 10% smoke records into the scientific step schedule
(`src/looplora/llm/release.py`, `lsq_release.py`, `dpfp_release.py`, `scripts/prepare_lsq_release.py`,
`scripts/prepare_dpfp_release.py`, `tests/test_release_scripts.py`) against WP-L4's reference path (`read_fix_run.py`,
`scripts/prepare_p1l_read_fix_a2.py`, `docs/reports/P1L-read-fix.md`), and the RFX entity-EM guard (`rfx.py`, `rfx_run.py`).

## Result

**No defect of consequence in the release path.** The new stdlib path reproduces WP-L4's frozen A2 release exactly when fed
WP-L4's real rank-128 smoke records (see Evidence 1), so the registered step counts are the ones WP-L4's rule gives. I found no
case where the release is inconsistent with the RFX summary, no stale importer of the moved registries, and no way for a
refused input to write a release. Two low-severity observations follow; neither blocks step 2.

## Findings (ranked)

### F1 (LOW, probability a few percent, unverified): the 64-row RFX smoke can trip the entity-EM guard on correct code, through exact bf16 ties

- Where: `src/looplora/llm/rfx.py:160-179` (`entity_em_failures`: allowance `max(ENTITY_EM_SLACK, 1/rows) + TOLERANCE`),
  `src/looplora/llm/rfx_run.py:95-98`.
- Arithmetic of the allowance is exact (Evidence 6): 64 rows allow a deficit of 1 row, 100/128/200/256 rows 1 row, 1,000 rows 5,
  4,000 rows 20; there is no floating-point misclassification at any hit count.
- The guard tests the **net** deficit `lm_em - lm_em_entity` = (A - B)/rows with A = rows where the full-vocabulary argmax hits and
  the restricted argmax misses, B = the reverse. For a correct glue A comes only from exact bf16 ties between the answer and
  another *entity* token (the answer token is in the restricted set). Full argmax resolves a tie to the lower **token id**
  (`torch.argmax`, first maximum), the restricted argmax to the lower **entity index**. The name pool is a seeded hash shuffle
  (`src/looplora/data/assets/build_name_pool.py:75`; 511 of 1,023 adjacent pairs are inversions in `qwen3_names.v1.json`), so the
  two orders are independent: a tie with the answer is an A row with probability 1/4 and a B row with probability 1/4. The net
  is zero-mean; non-entity winners (B rows that are legitimate rescues) push it negative.
- Failure scenario: at 64 rows the net must stay within 1 row. A 2-row net tie deficit in any of the 42 cells (3 checkpoints x 14
  variants) raises `RuntimeError` before `summary.json` is written. The smoke is deterministic on rerun, so the operator cannot
  just resubmit: either a code change (new commit) or a judgement that the failure is ties (`full_hit_entity_miss` equals the
  deficit) is needed.
- Size of the risk (my estimate, not measured): the pool of tie-prone rows is large. In WP-L4's saved validation samples
  (`runs/p1l/read-fix-20261027/arms-r128/R/val-24645/samples.npz`, `arms/R/val-23590/samples.npz`) 16.7% (R-r128) and 10.7% (R-r32) of
  rows are full-vocabulary hits with `p(answer) < 0.5`, 5.3% and 7.5% with `p < 0.2`. With a top-2 tie probability of 5-10% in such rows
  (bf16 spacing 0.125 at logit magnitude 16-32) the tie-induced A rate is about 0.3 rows per cell per 64 rows; the net must still beat
  the legitimate rescues (B), which are several rows per 64 in the deployed cells and few only in the near-clean cells. A Skellam
  calculation (`skellam.py`) gives per-cell P(net >= 2) of 0.4% at 0.1 rows and 2% at 0.25 rows, so an overall smoke false-alarm
  probability of roughly 2-10%. At 4,000 rows the margin is 21 net rows against an A/B rate of a few rows: negligible (<1e-9 for
  mu <= 5), so the science run is safe.
- Suggested fix (optional, cheap): allow 3 rows at the smoke (`max(ENTITY_EM_SLACK, 3/rows)`; identical at 4,000 rows), or make the
  restricted argmax tie-consistent with the full one (break ties by lower token id, e.g. index_select over the entity ids sorted
  ascending and map the answer entity through the inverse permutation), after which A is exactly 0 and the guard can be strict.
  Without a code change: if the smoke trips with a deficit of 2-3 rows and the same count in `full_hit_entity_miss`, compare the
  answer logit with the restricted maximum before treating it as a glue bug; a wrong slice/axis/pairing costs EM x rows (6-20 rows
  per 64 in a deployed cell), not 2.
- The guard does catch the glue bugs it is meant for: any wrong slice, axis or pairing leaves chance-level restricted hits, a
  deficit of about EM x rows in every cell with hits (>= 6 rows per 64 in the deployed R-r32/R-r128/W5-r32 cells, hundreds at 4,000).
  It cannot catch a bug that *raises* the restricted EM (not a plausible glue error); the scripted-logit tests cover that direction.

### F2 (INFO, design gap, not a defect): the release gate is timing plus identity only

`smoke_timing` (`release.py:37-64`) checks the smoke record's identity and timing fields. WP-L4's A2 gate
(`scripts/prepare_p1l_read_fix_a2.py:24-75`) additionally reloaded `writer.pt`, re-ran `validate` on 100 episodes and compared the
predictions, checked the samples, logs and the Gram error, and compared `rec['config']` with `configuration(...)`. None of that can
run on the login node and the step-1c task did not ask for it (the release status string says so:
`released_after_smoke_timing_gate`). Failure scenario: a smoke whose export cannot be reloaded passes the release, and three 2 h
DPFP jobs (or up to six LSQ jobs) produce checkpoints the audit cannot load. The tiny-model tests (`tests/test_llm_dpfp.py`,
`tests/test_llm_lsq.py`: 5-step `fit`, `validate`, export, exact reload per arm kind) run before the smokes in the step-2 order, so
the residual risk is small. Optional: add one compute-side reload check per smoke arm to step 2 before the release is committed.

### Notes (no action needed)

- Smoke and science packing must stay identical. WP-L4's smokes and science jobs ran two arms concurrently on one GPU
  (`P1L-read-fix.md` cost table); `dpfp_run`/`lsq_run` run one arm per job, smokes included, so the measured step time applies. If
  anyone packs two arms into one science job while the smokes were solo, the per-step time is higher than measured and the 1.5 margin
  is consumed (the in-run guard `read_fix_run.py:175` raises at 7,000 s).
- `EVAL_SCALE = 10` is a literal; it is only right while `SMOKE_EPISODES = 100` and the science evaluation is 1,000 episodes
  (`read_fix.configuration` sets `eval.episodes` to 100 or 1000). `validate` loops in chunks of 16 (7 chunks for 100 episodes, 63 for
  1,000, a ratio of 9), and the first evaluation carries warm-up, so x10 on the maximum is conservative (WP-L4's r128 smokes: first
  evaluation 1.07 s against 0.89-0.91 s for the others).
- `validate` times its loop after the validation data are generated (`read_fix_run.py:64`), so `metrics[step]['seconds']` excludes
  that CPU time; the same holds for WP-L4's extraction, and the 1.5 margin absorbed it there (R-r128 science: 4,920 s for a pair,
  estimate with margin 7,199 s).
- `write_new` uses exclusive creation, so a crash mid-write leaves a truncated file that the script then refuses to overwrite
  (fail-closed; the operator removes it by hand).

## Verified OK (with evidence)

1. **Same numbers as WP-L4 on real data.** `real_l4.py` runs `smoke_timing` and `schedule_fields` on the real rank-128 smoke records
   `runs/p1l/read-fix-20261027/smoke-r128/{W5,R,OA}/record.json` (timing and identity fields only): steps `{W5: 24635, R: 24645,
   OA: 31175}`, the five evaluation marks and `sizing_by_arm` (including `smoke_fraction_of_fixed_steps`) equal the frozen
   `reports/p1l-read-fix/release-r128.json` field by field. This also shows that a real `fit`/`run_one` record passes every identity
   check, including `config_hash == stable_hash(config)` after the JSON round trip.
2. **Field existence, units, meaning (item 1).** `dpfp_run.run_one`/`lsq_run.run_one` store `record['training'] = read_fix_run.fit(...)`
   unchanged. `fit` writes `actual_steps`, `training_episodes = 32 x steps`, `eval_steps`/`primary_eval_steps = sorted({round(steps k/5)})`,
   `matched_eval_steps = []` (only `read_arm == 'OArho'` adds 6580, and the DPFP/LSQ `read_arm` is the arm label), `metrics[str(step)]`
   from `validate` with `n = config['eval']['episodes']` (100 for smokes, 1000 for science) and `seconds` (the loop only),
   `steady_step_mean_seconds = mean(durations[10:])` where `durations` excludes validation, logging and checkpointing, `step_p95_seconds`
   and `seconds` (total, including evaluations; unused for sizing). WP-L4's A2 extraction was exactly `training['steady_step_mean_seconds']`
   and `[metric['seconds'] for metric in training['metrics'].values()]` with `max(...) x 10`; `sized` does the same with `EVAL_SCALE = 10`.
   The run-level fields `rank`, `store`, `store_eta`, `dpfp_nu`, `config['lsq']['arm']`, `config['dpfp']['arm']`, `source_hash`,
   `job_id`, `host` are written by the two runners (`lsq_run.py:58-62`, `dpfp_run.py:57-61`).
3. **Rule parity.** The AST-extracted `read_fix_run.largest_schedule` and `release.largest_schedule` agree on 220,000 random
   (step, evaluation, evaluations, wall, margin, reserve) draws and on the error cases (`parity.py`); the constants 7200, 90, 1.5, 5, 10,
   100 equal the A2 literals. `sized`'s consistency `require` never fired in 400,000 random rows (`sized_fuzz.py`).
4. **Per-arm schedules (item 2).** `schedule_fields` sizes every arm from its own row (`steps_by_arm`, `primary_eval_steps_by_arm`
   `steps*k//5`, `sizing_by_arm`), as DPFP.md and LSQ.md register ("each in one job"). E16 DPFP needs all three smokes (a missing one is a
   `FileNotFoundError`); E8 sizes only `lock['choice']['arm']` from a smoke with `config['task']['E'] == 8` and
   `eligible_e8` (action `retry_e8`, arm in the registry). The E16 release is checked with the runner's own `scheduled_steps`
   (all arms at once for E16); the E8 release holds the retrained arm only. `cli_e2e.py` ran both scripts end to end on python 3.9
   with records derived from a real smoke record: DPFP E16 and E8, LSQ E16 (K4+RG at both ranks) and E8 (L128-RG); the runner-side
   `scheduled_steps`/`release_eta` accept the outputs; a second run refuses with `FileExistsError`.
5. **LSQ hand-off (item 3).** `check_summary` refuses a smoke summary, dirty source, any instrumentation failure, a missing or failed
   `entity_em_guard`, an arm set `check_arms` rejects, and eta maps whose keys are not exactly the RG ranks. `arms` is
   `check_arms(summary['lsq_arms'])` (a copy); both eta maps are copied; `check_release` re-asserts `arms`, both maps, `release_arms`,
   `release_eta` (which also enforces `eta_by_rank == lsq_eta(rfx_eta_by_rank)`) and `scheduled_steps` before `write_new`; E8 requires
   the retrained arm among RFX's arms and keeps the full `arms` and eta maps with `steps_by_arm` for that arm only. `rfx_run.decisions`
   produces exactly the keys the script expects (`rfx_eta_by_rank` only for storage ranks, string keys). No path writes before
   all checks pass; an existing release is refused before any input is read.
6. **Guard arithmetic (item 5).** `rows = episodes x queries_per_episode` = 64 (smoke, `SMOKE = dict(fit_episodes=64, eval_episodes=64)`)
   and 4,000 (`EVAL_EPISODES`), `queries_per_episode = 1` (`stratify_queries(..., [1], ...)`). `guard_fp.py` sweeps every hit count at
   64/100/128/200/256/1000/4000 rows: the largest passing deficit is 1/1/1/1/1/5/20 rows with no floating-point anomaly; NaN fails closed.
   `rfx_run.run` raises after the per-label JSON are written and before `summary.json`; the summary is written before the instrumentation
   `AssertionError` (the LSQ script refuses such a summary). V = 1024 (`TaskSpec.V`), entity e renders as `token_ids[e]`
   (`render.py:207`, `batching.py:87`), val answers lie in the seen pool [0, 768), so the restricted set contains every answer.
7. **Moved registries and importers (item 4).** `verbatim.py` compares AST dumps old (17b27c2) against new: `ARMS`, `SPECS`,
   `ETA_BOUNDS`, `lsq_eta`, `check_arms`, `release_arms`, `release_eta`, `scheduled_steps`, `eligible_e8` (LSQ) and `ARMS`, `SPECS`,
   `scheduled_steps`, `eligible_e8` (DPFP) are identical, and nothing else changed in the four old modules. Every importer in `src/`,
   `tests/`, `scripts/` is re-pointed (`lsq.py`, `lsq_run.py`, `dpfp.py`, `dpfp_run.py`, `rfx_run.py`, `tests/test_llm_lsq.py`,
   `tests/test_llm_dpfp.py`, `tests/test_rfx.py`); no attribute access on the old locations remains; `pyflakes` is clean on all 16
   touched Python files; `py_compile` passes on python 3.9; the release modules import only `json`, `math`, `re`, `pathlib`,
   `hashlib` (blake2b), `argparse` and `looplora.infra.seeds`, and `looplora/__init__.py`, `llm/__init__.py`, `infra/__init__.py` import nothing
   (the only digest in the diff is blake2b).
8. **Tests (item 6).** With a stand-in `pytest` stub (`run_tests.py`, `stubs/pytest.py`) all 80 stdlib cases of
   `tests/test_release_scripts.py` pass on python 3.9 and the one torch case (`test_the_sizing_rule_is_wp_l4s`) is skipped, as the state doc
   says; I ran its assertion (copy vs WP-L4's function) by AST extraction instead (Evidence 3). Sensitivity: the sizing is checked
   against an independent transcription with distinct timings per arm, an outlier evaluation (so `max` vs `last`/`min`/`mean` change the
   result), `step_p95`/`total` decoys, a sentinel score that must not reach the release and that the release must not need, every
   identity field, the five marks, `n`, `E`, hashes, commits, eta maps, candidate order, E8 and refusal paths, and the no-torch/no-numpy
   subprocess run. I found no assertion that mirrors an implementation detail in a way that hides a realistic bug, and no spurious-failure
   risk on the compute node (deterministic, `tmp_path` only, subprocess given `PYTHONPATH=src`; the module-level `load_script` import
   needs only the repo layout).

## Could not verify

- Real DPFP/LSQ smoke records: nothing has run. The field list is traced through `dpfp_run`/`lsq_run` -> `fit` and confirmed on a real
  WP-L4 record written by the same `fit`; the DPFP/LSQ-specific identity fields come from the runners' `run_one` code.
- bf16 tie frequency of the real checkpoints (F1 is an estimate from saved WP-L4 NLL/correct columns and a bf16 spacing argument).
- The audited lock file that `--e8-decision` expects (`lock['choice']['action'/'arm']`): its writer (step-2 audit script) does not
  exist; the tests and `choose()` agree on the shape.
- `scripts/submit.py` wiring of `--commit <science_commit>` together with `--schedule RELEASE.json` (the release must be reachable
  from a snapshot that does not contain it); WP-L4 did the same, not exercised here.
- GPU test collection and run of `tests/test_llm_lsq.py`, `tests/test_llm_dpfp.py`, `tests/test_llm_rfx.py`, `tests/test_rfx.py`
  (static checks only: no undefined or unused names, imports resolve by name against the new modules).
- Whether a K4/RG smoke finishes inside the 25 min walltime of the state doc (2,400 steps; K4 repeats the 16 sequential writes four
  times); a killed smoke simply leaves no `record.json` and is rerun.

## Scratch material

All under `/tmp/claude-30041/-work-xg24i002-x10041-LoopLora/7ae01788-4591-4489-abeb-1056ba03ea57/scratchpad/review-release/`:
`real_l4.py`, `cli_e2e.py` (outputs under `e2e/`), `parity.py`, `sized_fuzz.py`, `verbatim.py`, `guard_fp.py`, `skellam.py`,
`contested.py`, `run_tests.py`, `stubs/pytest.py`. Nothing in the repo or the worktrees was modified.
