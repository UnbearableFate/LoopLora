# WP-P5L state — step 1 done; step 2: 15 trainings and B0 done and verified, authorization committed, no confirmation episode generated

As of 2026-10-29 05:20 JST. Author: sn-p1l (Claude Sonnet 5.5), branch `wp/p1l`. Task: `docs/tasks/WP-P5L_second_scale.md`
(master `1fcf56a`, merged into wp/p1l first). Prereg draft: `docs/prereg/P5L.md`. Nothing was trained beyond the smokes;
no score was computed or read.

## What was built (commits on wp/p1l)
- **Backbone option.** Optional top-level config field `backbone: {id, revision}` (`llm/config.py`, outside `DEFAULTS`,
  so resolved configs without it keep their hashes); `inject.load_model/load_tokenizer(backbone=None)` and
  `inject.backbone_pin`; `LLMExperiment` passes `config["backbone"]` (and the learned-writer signature uses the pin).
  Default = the pinned 0.6B, same `from_pretrained` arguments as before. Adapter dimensions come from the loaded model
  (`attach_adapters`: `o_proj` in/out features), so nothing needed deriving: 0.6B 2048 → 1024, 1.7B 2048 → 2048.
- `scripts/make_p5l_configs.py` and `configs/p5l/main-r128/*.json` (15 files = P1-L's r = 128 configs with `name` and
  `backbone` only changed; lr 0.003/0.001 and S 17,000 carried), `scripts/run_p5l.py` (`smoke` and a lock-guarded `main`
  stage that reuses P1-L's `fit`), a `p5l` ledger in `scripts/p1l_job_ledger.py` (`--study p5l`, `reports/p5l/jobs.json`).
  P1-L's `run_p1l_main.py`, evaluator, finalizers and the registered window table are untouched
  (`CONFIRMATION_WINDOWS` gets the `p5l` window in step 2, with `tests/test_p1ls_pipeline.py`'s equality test).
- Not built (step 2, after registration): the confirmation evaluator and finalizer for the `p5l` study (named window
  test [102000, 104000), 97.5%, backbone read from each checkpoint config), the B0 evaluation, the lr/S lock file and the
  authorization.
- `Qwen3-1.7B-Base` (cached): 28 layers, hidden 2048, 16 heads × 128, vocab 151,936; tokenizer files identical
  (blake2b) to the 0.6B's, so the name pool is unchanged.

## Fit (2 h job)
Largest reader (B8) 17,000 steps ≈ 1.11 h measured, 1.67 h at 1.5×: it fits one job, so no resume is needed. For the
record: the pipeline has **no resume path** (checkpoint written once at the end; `train` refuses an existing output
directory; no test of an interrupted run exists). A killed job is rerun from scratch.

## Verification (all on compute through `submit.py --commit`)
| Job | Name | node·h | Outcome |
| --- | --- | ---: | --- |
| 3721808 | `t-p5l-backbone` | 0.006 | exit 1: my test expected the P1-L run-manifest `config_hash`, which is another hash (the training record's is the one to compare with); test fixed |
| 3721815 | `t-p5l-backbone2` | 0.033 | exit 1: one test called a helper wrongly (test bug); all else passed, including `test_p1l_main_gates.py` and `test_p1ls_pipeline.py` |
| 3721817 | `t-p5l-backbone3` | 0.006 | exit 0: `tests/test_p5l.py`, 8 passed |
| 3721819 | `t-p5l-llm` | 0.024 | exit 0: `test_llm_train/oracle/evaluate/determinism/protocols`, `test_p1l_memory_use`, `test_p1lc`, `test_p1lc_stats` pass (1 skipped) |
| 3721818 | `sm-p5l-quick` | 0.053 | exit 0: 100-step smokes, five readers, seed 17, r = 128 |
| 3721827 | `sm-p5l-10pct` | 0.235 | exit 0: 1,700-step (10%) smokes, B8 and M1-L4 |
Total step 1: **0.358 node·h** of the 0.5 budget (`reports/p5l/jobs.json`).

`tests/test_p5l.py`: the 15 P1-L r = 128 training configs still resolve to the `config_hash` their finished runs recorded
(literals taken from the raw training records, which the new code reproduces); `backbone` is optional, validated and
outside `DEFAULTS`; default loading makes exactly the original `from_pretrained` calls and an explicit backbone changes
only id and revision; a tiny Qwen3 (`o_proj` 128 → 64) written to a temporary cache is loaded through the option, its
adapter has d_in 128, d_out 64, an `LLMExperiment` built from a config with `backbone` loads it, and without the option
the pinned 0.6B is looked up instead; the cached 1.7B has the registered shape and the 0.6B's tokenizer; the 15 committed
configs equal the generator's output; the runner's lock and backbone checks reject disagreements.

## Smoke (timing and memory only; `runs/p5l/smoke-100`, `runs/p5l/smoke-1700`, `smoke=True`, no score read)
| Reader | steps | steady s/step | train s | peak GPU memory |
| --- | ---: | ---: | ---: | ---: |
| B8 | 100 / 1,700 | 0.2388 / 0.2359 | 38.9 / 416.9 | 12.77 GB |
| M1-L4 | 100 / 1,700 | 0.2382 / 0.2321 | 36.7 / 408.9 | 12.77 GB |
| B4 | 100 | 0.1791 | 30.8 | 12.73 GB |
| B5-L4 | 100 | 0.2025 | 33.1 | 12.76 GB |
| B3-L4 | 100 | 0.1813 | 30.9 | 12.27 GB |
Deviation from the task: a full 10% smoke of all five readers would cost about 0.47 node·h alone and exceed the 0.5
budget with the tests, so the 1,700-step smoke covers the two slowest readers (B8, M1-L4) and the other three were
measured over 100 steps; the 100-step and 1,700-step step times of B8 and M1-L4 agree within 2%, and peak memory did
not change.

## Projection (see `docs/prereg/P5L.md`)
Main training 15 runs: 14.6 node·h of training + about 0.2 start-up = **14.8 node·h** (per-run walltime 1.27–1.67 h at
1.5×); confirmation 1.2–3.2 node·h (not measured at 1.7B); total 16.1–18.1 node·h.

## Open questions for the orchestrator
1. Register `docs/prereg/P5L.md` (see its "Deviations flagged"): the B6 reading, k = 2, the shared test window with P1L-C.
2. After registration, step 2 needs the evaluator/finalizer extension named above; say if it should be me.
3. Confirmation-evaluation cost at 1.7B is a projection; a short validation-only evaluation timing on the smoke
   checkpoints (`runs/p5l/smoke-100`) can replace it before the main runs if wanted.

## Step 2 (2026-10-29; prereg registered on master 99f0dae, window changed to test [104000, 106000))
- Lock `reports/p5l/r128/lr-S-lock.json` committed at e8ae1f9 (audited by the orchestrator). The 15 trainings ran pinned to
  e8ae1f9 (`git_commit` and `source_hash` identical in all 15 records, `source_dirty` false); B0 ran at 24fcc07 (it needs
  `run_p5l.py` B0 support, optimizes nothing).
- Extension built and tested: `CONFIRMATION_WINDOWS["p5l"] = 104000`, `scripts/evaluate_p5l.py` (with `--check-only`),
  `scripts/finalize_p5l.py`, `scripts/make_p5l_manifest.py`, `configs/p5l/main-r128/B0-s17.json`, `tests/test_p5l_pipeline.py`.
  Not done: the optional validation-only evaluation timing on the smoke checkpoints (confirmation cost stays a projection).

### Verification of the trainings (before the manifest; no confirmation episode, no score)
All 16 jobs exit 0, and each record was read: `status` complete; `config.train.steps` 17,000 and the train log's last step
17,000 (170 logged steps plus the final line) for all 15; lr 0.003 (B8, M1-L4, B4, B5-L4) or 0.001 (B3-L4) as locked; backbone
revision ea980cb in every config; the 15 committed config files still hash to the lock's `config_hashes`; each record's config
equals its committed config; `checkpoint.pt` present and equal to the recorded hash; exit file 0 and receipt source e8ae1f9
for every job. B0 (21 s wall, 0.0007 s "training"): `trainable_parameters` 0, empty `train_log.jsonl`, the same `writer_hash`
as P1-L's B0, so the same record shape as P1-L's B0 job (0.006 node·h); its wall time is the 1.7B model load, peak memory 3.44 GB
is the bf16 weights. Then `make_p5l_manifest.py` re-verified every run (records, checkpoints, committed configs, receipts,
source) and wrote `reports/p5l/r128/training-manifest.json` (16 runs). `evaluate_p5l.py --check-only` (job 3722004) accepted
the manifest and `reports/p5l/confirmation-authorization.json` (hashes of prereg, lock and manifest, k = 2, 97.5%, window
`p5l`) with the full `check_training`, without loading a model or generating an episode.

### Jobs (`reports/p5l/jobs.json`, `--study p5l`, all 26 jobs, 15.17 node·h)
| Jobs | Name | node·h | Walltime used |
| --- | --- | ---: | --- |
| 3721865 / 3721867 / 3721869 | `tr-p5l-B8-s17/29/43` | 1.149 / 1.136 / 1.108 | 1:08:56 / 1:08:08 / 1:06:28 |
| 3721866 / 3721868 / 3721870 | `tr-p5l-M1-L4-s17/29/43` | 1.136 / 1.121 / 1.120 | 1:08:11 / 1:07:16 / 1:07:12 |
| 3721879 / 3721882 / 3721885 | `tr-p5l-B5-L4-s17/29/43` | 0.961 / 0.946 / 0.951 | 0:57:38 / 0:56:45 / 0:57:02 |
| 3721880 / 3721883 / 3721886 | `tr-p5l-B4-s17/29/43` | 0.839 / 0.843 / 0.842 | 0:50:20 / 0:50:36 / 0:50:32 |
| 3721881 / 3721884 / 3721887 | `tr-p5l-B3-L4-s17/29/43` | 0.856 / 0.862 / 0.861 | 0:51:21 / 0:51:43 / 0:51:41 |
| 3721888 | `tr-p5l-B0-s17` | 0.006 | 0:00:21 |
| 3721889 / 3721890 | `t-p5l-pipeline` / `t-p5l-pipeline2` | 0.043 / 0.014 | first run failed one test of mine (B0 evaluator test built a synthetic config); fixed, rerun passes (57 tests) |
| 3722003 / 3722004 | `t-p5l-check`, `t-p5l-authcheck` | 0.014 / 0.006 | tests after adding `--check-only` (50 passed); real manifest/authorization check |
Main training: 16 jobs, 14.74 node·h (projection 14.8). Step 1 (tests and smokes): 0.36 node·h. Total so far 15.17 node·h.
Next, after the orchestrator's reply on the authorization: 16 evaluation jobs (one per reader group, `ev-p5l-*`), the manifest of
evaluations, `finalize_p5l.py`, the decision record `reports/decisions/P5L.md` and the report `docs/reports/P5L.md`.
