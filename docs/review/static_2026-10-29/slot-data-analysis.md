# SLOT-LOOP-v1 pre-run static review: data, rules, configs, manifest, analysis (branch wp/slot, head ccdcc96)

Reviewer: read-only subagent (Sonnet 5.5). Date: 2026-10-29. Nothing was executed on a GPU. torch was never imported, no pytest,
no training or evaluation, no PBS job (login node).
Authority: `docs/prereg/SLOT-LOOP-v1.md` as registered at master 5f04a5c (blake2b-128 `56e8c611c01e0ef9b7ae32db97b0e334`; the file on
master and on wp/slot is byte-identical to it). The plan was used as a reference only. Prereg line numbers below are those of that file.
Scope: `git diff c6af756..ccdcc96` (23 files), restricted to `src/looplora/llm/slot_data.py`, `slot_rules.py`,
`configs/slot_loop_v1/s0.json` and `s1a.json`, `scripts/prepare_slot_manifest.py`, `scripts/analyze_slot_study.py`,
`tests/test_slot_data.py`, `tests/test_analyze_slot_study.py`, and the code they call (`infra/seeds.py`, `data/relations.py`;
`slot_run.py`, `slot_experiment.py`, `slot_train.py` only where my files meet them).
Checks that were run: full reads of every in-scope file and of the registered prereg; stdlib recomputation of every registered seed and
of the layout digest; an independent stdlib re-implementation of the disjointness check; the registry's id formula against
`relations.generate` (six non-sealed sets for real, all four splits with a dummy seed, no sealed data generated); the old parents'
saved episode ids against the registry's formula; the script's own `resample_sums` and `interval` against a direct numpy draw and
`np.percentile`; the bootstrap index stream under numpy 1.20.1 / py3.9 and numpy 2.5.4 / py3.13; the committed stage configs against
`slot_rules.stage_config()`; py_compile under the login Python 3.9.25; `--help` of the analysis script under the login Python;
greps for hashing primitives, sealed-seed locations, protected-window calls and print/log calls on the sealed path.
The checks used the project venv interpreter (`-I -B`, `sys.path` pointed at the worktree `src`, no torch) or `/usr/bin/python3 -I -B`.
No tracked file, branch or commit was touched: `git status` is clean in the worktree and shows only `docs/review/static_2026-10-29/` in the
repo. One slip, disclosed: the py_compile run under the login Python used `-I`, which makes Python ignore `PYTHONPYCACHEPREFIX`, so four
git-ignored bytecode files were written into the worktree (`scripts/__pycache__/analyze_slot_study.cpython-39.pyc`,
`scripts/__pycache__/prepare_slot_manifest.cpython-39.pyc`, `src/looplora/llm/__pycache__/slot_data.cpython-39.pyc`,
`src/looplora/llm/__pycache__/slot_rules.cpython-39.pyc`). They are valid for the current sources, harmless, and left in place because the
brief forbids deleting; they can be removed. This file is the only intended write; it is not committed. The scratch scripts are in the
session scratchpad.

## Verdict

No defect that changes a registered number, opens a sealed set early, or makes a registered decision come out wrong. Every
registered seed derives as the data table says, the layout digest and the id formula equal the generator's, the new sets and
train streams are disjoint from the old protected ones, the S0 rows R0 to R5, the section 7.4 trigger and the S1A rule are implemented as
registered in exact integer arithmetic, and the bootstrap equals one draw of 10,000 resamples. The registered Interpretations 1 to 9 do not
contradict the registration.

Five findings, none of which blocks the preflight (step 1 of the runbook uses `scripts/submit.py`, which is executable):
- F1 (medium) the section 7.4 ablation dev comparison is outside the frozen script, and the state file's plan to add it "before the
  release if the ablation is triggered" is circular; fix before the release freezes the script.
- F2 (low) the open record's schema is only in code and the S-RE reference digest is a trap for the post-open evaluation job.
- F3 (low) the runbook's two login-node commands (lock and S0 decision) cannot run as written; fix before step 2.
- F4 (low) the seal is a lookup guard; the sealed sets are reachable as plain objects.
- F5 (low) the frozen analysis does not tie its inputs to the registry (sizes, ids, references, Matrix against Native on dev).
F1 and F5 (a, b) are decisions about the script that must be taken before its digest goes into the release, F5 (c) before the S0 outputs are
analyzed; F2 and F4 before S1A.

## Findings, ranked

### F1 (medium; certain) The section 7.4 ablation dev comparison has no frozen implementation, and the plan to add it cannot work as written
- Evidence: the prereg runs the ablation after the trigger and before `STOP: S1A selected` (prereg:114, :122-123); its outcome ("a dev
  gain of Soft over Matrix of at least +3.0 pp is flagged for a new registration") decides whether the S1B addendum enters slot arms with
  the `A_query` copy (prereg:215). The analysis script is frozen by digest in the release and used unchanged through the confirmation
  analysis (prereg:125).
- `docs/reports/SLOT-LOOP-v1-state.md:89` (Interpretation 9(e)): "Not scripted: the dev comparison of the section 7.4 ablation ... to be
  added before the release if the ablation is triggered, as the script is frozen only at the release."
- `scripts/analyze_slot_study.py:690` offers the stages `S0`, `S1A-DEV`, `S1A` only; `stage_dev` (:475-503) ends at the trigger outcome
  (:495); `parse_run` (:426-472) never reads the record's `ablation` field (written at `slot_run.py:409`).
- Failure: whether the ablation is triggered is known only after the six runs, which come after the release (prereg:112, :114), so "add it
  before the release if triggered" cannot be satisfied. Either the script changes after the freeze (its digest in the release and in
  `s1a-open.json` then no longer matches) or the ablation comparison is made by hand or by an ad hoc script. The unregistered details
  then become free choices at the moment the flag is decided: which checkpoint of each ablation run is compared (the dev-NLL-selected
  one, as for the main arms, or the last), `>=` against `>`, what happens when an ablation run is unstable or failed, and whether
  n and the dev ids are checked. Separately, `parse_run` would accept an ablation record placed under `matrix_L1/s17/` as a main-arm record.
- Also on the ablation path: `slot_run.py:402-404` looks up `release["config_hashes"]["<arm>/<seed>/ablation"]`, while the prereg's
  release lists "the six science config hashes" (prereg:112). A release with six hashes makes `run_train(..., ablation=True)` end in
  `StopRun("the science config of ... differs from the release")`, after the release is committed. (No release builder exists in this
  change, so this is a requirement for it.)
- Fix, before the freeze: add a fourth stage (for example `S1A-ABL`) with inputs `ablation/matrix_L1/s17/record.json` and
  `ablation/soft_slot_L1/s17/record.json`. Refuse unless both are `complete`, carry `ablation: true`, and share n and dev ids with each
  other and with the six main records. Compare the dev correct counts at each run's dev-NLL-selected checkpoint and flag iff
  `(c_soft - c_matrix) * 100 >= 3 n` (exact integers). Print counts and the flag only. Make `parse_run` require `ablation is False`
  for the six main records. Record the checkpoint choice as Interpretation 9(f). Have the release builder carry the two ablation
  config hashes. Nothing else of the registered section 7.4 flow is missing: the trigger itself (all six complete, all finite, selected
  dev NLL at most step-0 dev NLL, mean dev gap below +3.0 pp, dev only) is implemented exactly (:428-431, :483-488, :491-495).

### F2 (low; certain) The open record's schema exists only in code, and the S-RE reference digest is a trap
- Evidence: `slot_data.py:222-234` reads `prereg_blake2b_128`, `analysis_script_blake2b_128`, `selected_checkpoints` (exactly six keys with
  32-character string values) and an optional `reference_checkpoint`. The prereg says the record "names the six selected-checkpoint
  digests and the blake2b-128 of the frozen analysis script" (prereg:56-57) and names no reference digest. No document lists the key
  names (`docs/reports/SLOT-LOOP-v1-state.md:80` lists the contents without keys); the only specification is
  `tests/test_slot_data.py:115-119`, which never sets `reference_checkpoint`.
- The registered post-open job also evaluates the unmodified S-RE on the sealed sets (prereg:115; labels `s_re_native`, `matrix_step0`,
  `soft_step0`, `analyze_slot_study.py:38`), and each entry passes `slot_data.eval_set(..., checkpoint_digest=...)`
  (`slot_run.py:450-451`); for these entries the digest is the S-RE `writer.pt` digest.
- Failure: an open record written as the prereg describes (six digests and the script digest) has no `reference_checkpoint`. The
  evaluation job evaluates the six selected checkpoints and then raises `SealedSetError` at the first S-RE entry; the out directory
  was created with `exist_ok=False` (`slot_run.py:447`), so a rerun needs a new one. The reference files are then absent, and the
  analysis accepts the set without them (F5 b), so the report lacks "the parent S-RE row and the step-0 rows" (prereg:121). No data
  leaks; it costs a job and a re-issued record.
- Also: `--open-root` defaults to the working directory (`slot_run.py:489`, `:512`), and the token needs
  `docs/prereg/SLOT-LOOP-v1.md`, `scripts/analyze_slot_study.py` and `reports/slot-loop-v1/s1a-open.json` under that root. `submit.py
  --commit` runs a snapshot of one commit (`scripts/submit.py:91-92`), so a job pinned to the science commit (as `run_train` is,
  `slot_run.py:394`) does not contain a record written after the release. Any edit of the prereg file, even a status line, makes the token
  fail (by design, safe direction).
- Fix: write the schema into the state file or runbook now (the four key names, six digests, `reference_checkpoint` = the S-RE
  `writer.pt` digest of the locked inventory, and that `--open-root` names the directory that holds the record). Better, let
  `load_open_token` take the S-RE digest from the lock (it is registered, and the reference evaluation is registered), so the record
  needs no seventh field; add a test with a reference entry.

### F3 (low; certain) The runbook's login-node commands cannot run as written
- Evidence: `docs/reports/SLOT-LOOP-v1-state.md:96` (`scripts/prepare_slot_manifest.py --configs ...`) and `:98`
  (`scripts/analyze_slot_study.py --stage S0 ...`) are bare paths. Both files are mode 100644 (`git ls-tree HEAD scripts/...`; `ls -l`
  shows `-rw-r-----`); of the scripts the runbook calls, only `scripts/submit.py` (runbook :95, :97) is 100755. The docstring of `prepare_slot_manifest.py:7-12` also shows
  direct execution. This is the same class as F1 of `docs/review/static_2026-10-27/oracle.md`.
- Even through an interpreter, `prepare_slot_manifest.py` imports `looplora.llm.slot_data` (:26), which imports
  `looplora/data/relations.py:14` (`from numpy.typing import NDArray`). Under the login node's default `/usr/bin/python3` (3.9.25, numpy
  1.20.1) this raises `ImportError: cannot import name 'NDArray' from 'numpy.typing'` (reproduced with `-I -B` and the worktree `src` on
  `sys.path`). It needs the project interpreter (numpy 2.5.4, Python 3.13). The analysis script is not affected: it compiles and
  runs `--help` under the login Python, and its random stream is identical under both numpy versions (see Verified OK).
- Failure: runbook step 2 ends in `Permission denied` (or, via a bare `python3`, in the ImportError); step 4 ends in `Permission denied`.
- Fix: write the commands with an interpreter: `uv run python scripts/prepare_slot_manifest.py ...` or
  `/work/xg24i002/x10041/LoopLora/.venv/bin/python scripts/prepare_slot_manifest.py ...` for step 2, and
  `python3 scripts/analyze_slot_study.py ...` for step 4. Alternatively `git update-index --chmod=+x` on both and state the interpreter.

### F4 (low; certain) The seal is a lookup guard; the sealed sets are reachable as plain objects
- Evidence: `SETS` is a public dict that holds the sealed `EvalSet`s with their `base_seed` (`slot_data.py:71-72`, `:74`); the guard lives
  only in `eval_set()` (`:237-245`). `slot_experiment.raw_episodes` / `episodes` (`slot_experiment.py:76-83`) generate the episodes of
  any `EvalSet` they are handed and never look at `sealed`. `manifest()` writes `base_seed` and `query_seed` of the sealed sets
  (`slot_data.py:185-186`), so `reports/slot-loop-v1/manifest.lock.json` will carry them outside the scan of
  `tests/test_slot_data.py:85` (`src`, `scripts`, `tests`, `configs`), against the file's own statement that they occur in
  `slot_data.py` only (`slot_data.py:5-6`, `slot_rules.py:125`). The seeds are public in the prereg (data table, prereg:47-48), so this
  concerns accidents, not secrecy.
- Failure: a one-line mistake such as `for s in slot_data.SETS.values(): episodes(s)` or `episodes(slot_data.SETS["S1A", "confirmation"])`
  generates the sealed confirmation episodes with no token, and no test would notice. No in-tree caller does this: every call site goes
  through `eval_set(...)` (`slot_run.py:167, 204, 291, 338, 406-407, 451, 515`), and `_set_rows` (`slot_rules.py:124-131`),
  `manifest()` and `check_disjoint()` read ids and metadata only (they never call `relations.generate`).
- Fix: refuse a sealed `EvalSet` in `raw_episodes` (for example `eval_set()` returns a copy flagged `opened=True` for a sealed set and
  `raw_episodes` raises `SealedSetError` when `sealed and not opened`), or keep the sealed entries out of the public `SETS`; add a test that
  `episodes(SETS["S1A", "confirmation"])` raises. Optionally drop `base_seed` / `query_seed` of the sealed sets from `manifest()["sets"]`
  (the id digest and the count are what the lock needs) or state that the lock carries them.

### F5 (low; certain for the code, the failure needs an upstream defect) The frozen analysis does not tie its inputs to the registry
Decide (c) before the S0 outputs are analyzed, (a) and (b) before the script's digest goes into the release.
- (a) Sizes and ids are compared only between files. S1A: `EXPECTED_N = 2000` (`analyze_slot_study.py:40`) is printed (:557, :574) and never
  enforced; `tests/test_analyze_slot_study.py:846` pins "a different n is reported, not refused"; the set's id digest is emitted (:557) but
  not compared with the manifest's `ids["eval/S1A/confirmation"]`. S1A-DEV (:477-479): the six records need equal
  `dev_ids_blake2b_128`, n and steps, not the registered 1,000 ids. S0 (:385-395): calibration n 2,048 and dev n 1,000 are not pinned and
  the ids digest is compared only with the script's own `samples.npz`. The runner does enforce the registered ids at generation time
  (`slot_run.py:153-158`, called at :169, :408, :454), which is why this is low. Failure: confirmation files that come from a wrong
  or truncated set (a wrong entries file, a stale directory, a debug limit) give a verdict at n = 1,000 or on other episodes; the only
  traces are a stdout line (`confirmation n 1000 expected 2000`) and a digest inside `summary.json`. The registered "the analysis refuses a
  mismatch" (prereg:60) holds between files only, not against the manifest.
- (b) References are optional in `read_set` (:511-516): `s_re_native`, `matrix_step0`, `soft_step0` that are absent are skipped silently,
  although prereg:121 puts "the parent S-RE row and the step-0 rows" in the report (F2 shows how they get lost).
- (c) R0 is decided from the preflight alone (`read_preflight` :176-185, `stage_s0` :387-389). The registered R0 also reads "Matrix
  differs from Native" (prereg:82, :88). The S0 job evaluates `native/all_valid_question` and `matrix_uncal/all_valid_question` (g = 1) on the
  same 1,000 dev episodes, but `decide_s0` (:315-336) never compares them. The preflight identity covers the old range, the S0 smoke at
  depths 1, 2, 4 and one train-shaped batch (`slot_run.py:214-218`). A Matrix path that differs from Native on the dev data would still
  receive a row R1 to R5 with Matrix_cal as the baseline.
- Fix: add `--manifest reports/slot-loop-v1/manifest.lock.json` (its digest goes into `source_index.json`) and refuse when a set's n or ids
  digest differs from `manifest["ids"]["eval/<stage>/<purpose>"]` (the lock holds counts and digests, not the sealed seeds); refuse a
  missing reference file once the evaluation entries are fixed; in `stage_s0` give R0 when `matrix_uncal/all_valid_question` differs
  from `native/all_valid_question` in `correct` or NLL (the registered gate says bitwise-equal logits, which makes both equal).

## Observations (not defects)

- Calibration tie (Interpretation 1): "the `g` nearest 1" is the smallest absolute `|g - 1|`, so 0.5 beats 2.0; on a log scale they are
  equidistant. The registration does not say. The choice is documented and tested (`tests/test_slot_data.py:167-174`); it matters only for
  a tie that contains both 0.5 and 2.0 and not 1.0. Accept as is, or decide before S0 data are read.
- The runner's ties are float arithmetic (`slot_rules.py:51, 62, 90`), the script's are exact (`analyze_slot_study.py:221-223, 421`).
  They can differ only within about one ulp of 1e-4 or 1e-6, and the script then refuses (Interpretation 9(d)). Fine.
- The S1A analysis requires the unseen files (`stage_s1a` :563 reads both sets), while prereg:124 calls the unseen EM "Descriptive, optional"
  (prereg:115 and :158 put it in the same post-open job). If the unseen evaluation fails after the confirmation succeeded, the frozen script
  gives no verdict and cannot be edited without breaking the digest in `s1a-open.json`. The orchestrator may prefer that the primary
  verdict not depend on the descriptive set; decide before the freeze (same moment as F5).
- `prepare_slot_manifest.lock()` takes the required checks from the report itself (:64-68) and does not pin the nine registered names; the
  analysis script's `CHECKS` (:27) is the second guard, and the runner always writes the nine (`slot_run.py:241-243`).
- No test pins `LAYOUT_DIGEST` to the generator (grep of `tests/test_slot_data.py` finds none). I verified equality by hand (True). A
  generator change would be caught at run time by `ids_record` (`slot_run.py:153-158`), which stops when generated ids differ from the
  registry's.
- The sealed-evaluation job prints only `{label: {n, file digest}}` (`slot_run.py:471`) and the analysis prints counts, the verdict word and
  file digests; `tests/test_analyze_slot_study.py:843` asserts no decimal number reaches stdout. The `verdict` word is a status, as registered.

## Verified OK (with evidence)

Seeds and registry (`slot_data.py`)
- Every registered integer re-derived with an independent stdlib `derive_seed` (blake2b-8 of `json.dumps([base, *labels], default=str)`,
  little-endian, 63-bit mask): the eight evaluation bases (`:65-72`), the four train bases (`:77-80`), the two bootstrap seeds (`:82`), and
  the old bases 7003109974603132079, 292578375307266510 and `derive_seed(20261027, "p1l-smoke-eval")` = 557549155479573416. All equal the
  prereg table (prereg:39-51, :58). The run labels are the ints 17, 29, 43 and the string `"pilot"`; the string form of 17, 29, 43 would give
  different values, and the registered ones match the int form (`train_base("S1A", "pilot")` = 2442741374644229280).
- Layout digest `b7f93c1ca56e5dd1f79a9dccfe2f6054` = `stable_hash(asdict(TaskSpec(**TASK)))` = `relations._layout(spec).digest`. The 1,000 saved
  episode ids of S-RE (`val-6000/samples.npz`) and of D128-3 (`val-22370/samples.npz`) equal `episode_seed(4242, "val", i)` under this
  digest, so the old-id derivation uses the layout the old runs actually used.
- Registry ids equal the generator's `EpisodeArrays.episode_seed` for the six non-sealed sets; the formula is equal for `train`, `val`, `test`,
  `test_unseen` (dummy seed, no sealed data generated); a train step i covers indices [32 i, 32 i + 32).
- Splits, starts and sizes: S0 smoke train [0,64), calibration val [0,2048), dev val [0,1000); S1A smoke train [0,64), pilot val [0,1000), dev
  val [0,1000), confirmation test [0,2000), unseen test_unseen [0,2000); train streams [0, 32 N) with N <= 6,000 (`MAX_STEPS`) and the pilot
  stream 200 x 32 = 6,400 (`slot_data.py:28-30, 65-72, 157-177`). `verify_registry` rejects repeated and old base seeds (:117-119).

Disjointness
- Independent stdlib check: 926,840 old ids (3,000 + 4,000 + 12,000 + 22,370 x 32 + 6,000 x 32, equal to the expected count) and 591,576 new ids
  (9,176 evaluation ids, three 192,000-id S1A streams and the 6,400-id pilot stream): no repeats, pairwise disjoint, disjoint from the old ones.
  The old windows and streams are those of prereg:59-60 (`OLD_WINDOWS`, `OLD_TRAIN`, `slot_data.py:85-91`); `CONFIRMATION_WINDOWS` (all five
  windows lie inside test [100000, 112000)) is pinned by `tests/test_slot_data.py:66`.

Sealing and blinding
- The guard (`slot_data.py:211-245`) has no default that opens a sealed set (`token=None`, `checkpoint_digest=None` raise); `slot_run.main` builds
  a token only for `--set confirmation|unseen` (:512). Tests cover no token, wrong checkpoint, none, five of six checkpoints, a wrong prereg digest,
  a wrong analysis digest, a short digest, and a script edited after the record (`tests/test_slot_data.py:102-139`).
- The sealed seeds occur in `slot_data.py` only under `src`, `scripts`, `tests`, `configs` (scan test :83-91); the committed configs name the
  sealed sets without seeds (`slot_rules.py:124-131`; test :94-99).
- `config.py` (`CONFIRMATION_WINDOWS` :20, the 2,000 cap :179-180), `relations.py`, `seeds.py`, `rfx.py`, `lsr_rules.py` are untouched by
  `c6af756..ccdcc96`; the new code never calls `split_window` or `resolve_config`; the torch-free files (`slot_data.py`, `slot_rules.py`,
  `prepare_slot_manifest.py`, `analyze_slot_study.py`) contain no `generate(` call.
- The sealed evaluation and the analysis print counts, digests and the verdict word only (`slot_run.py:471`, `analyze_slot_study.py:574-575`,
  `tests/test_analyze_slot_study.py:843`). No other `print` exists on the sealed path (`slot_run.py:471, 499, 519`).

S0 rules (`analyze_slot_study.py`, `slot_rules.py`)
- First-match order R1, R2, R3, R4, else R5 (`decide_s0` :325-335); "improves" is `(c_a - c_b) * 20 >= n` (exactly +5.0 pp) and a strictly lower
  mean read error, at the scope of the comparison against `matrix_cal` (:320-324); Clean-final needs the count only and is a `final_only`
  condition (:332); R0 needs all nine required preflight checks (:27, :176-185).
- Calibration grid of 18 in registered order (`:190`, `slot_rules.py:37-41`); tie rule: lowest NLL, within 1e-4 tied, then smallest `|g - 1|`, then the
  larger tau (`pick` :215-223, `choose_candidate` `slot_rules.py:44-52`); the script re-derives `selected` and refuses a different one (:226-238).
  A hand-made surface with three candidates within 1e-4 of the minimum gave g = 1 (tau 0.03); the repo's test covers 0.5 over 2.0
  (`tests/test_slot_data.py:172`).
- Rows are pinned to registered gains and taus (`parse_conditions` :263-266); `soft_rec` is descriptive; the `tau = 0.03` note (:398-399); the
  two scopes are never mixed (`CONDITIONS` :32).
- D128-3 recheck, once, on S0 dev at S-RE's selected values, no recalibration, direction only (`d128_recheck` :363-373).
- Step count and walltime: `schedule_steps(0.5, 20, 60)` = 4240 and `schedule_steps(0.2, 7, 60)` = 6000 (cap), the 5-minute rounding and the 3,600 s cap
  (`slot_rules.py:65-82`); LR choice with the 0.01-nat tie to the lower LR and the worse-than-step-0 stop (:55-62); checkpoint selection with the
  1e-6 rule and an unselectable step 0 (:85-94, script :416-423, `slot_train.py:110, 126-130`).

S1A rule
- `Delta = sum(c_soft - c_matrix) / (3 n)`; ADVANCE iff `sum * 100 >= 9 n` (exact, :568), `L > 0` as an exact Fraction test (:546, :549) and every
  per-seed `Delta_s > 0` (:570). Bootstrap seed 9056755664197379988 (:24) = `derive_seed(..., "bootstrap", "S1A")`.
- 10,000 resamples = 20 sequential blocks of `rng.integers(0, n, size=(500, n))` from `default_rng(seed)`, one index matrix shared by the three
  seed vectors and the pooled vector (:152-161). The script's own `resample_sums` equals one `(10000, n)` draw (10,000 resamples at n = 120 with the
  registered S1A seed); 1,500 rows equal for the S1A seed at n = 2,000 and for the S0 seed at n = 1,000; also equal to one `rng.choice` per resample. The first
  200 rows of both seeds hash identically (111cd9427f0ecb7f) under numpy 1.20.1 / Python 3.9.25 (login) and numpy 2.5.4 / Python 3.13.
- The percentile interval is numpy's linear interpolation at 2.5 and 97.5 evaluated in exact Fractions on sorted integer sums (:164-173); the
  script's `interval` agrees with `np.percentile` to 4e-13 (float conversion only); `L > 0` is exact.

Section 7.4 trigger (`stage_dev`)
- All six records `complete` (refusal otherwise, :428-431), all finite (:483-484), selected dev NLL at most step-0 dev NLL (equality is stable, :487),
  mean over seeds of the dev EM gap (Soft minus Matrix) below +3.0 pp iff `sum * 100 < 9 n` at the selected checkpoints (:491-495); `stop_unstable` is
  the STOP; dev inputs only. The ablation comparison itself is F1.

Configs, manifest, hashing, Interpretations
- `configs/slot_loop_v1/s0.json` and `s1a.json` equal `slot_rules.stage_config("S0")` and `("S1A")` exactly (stable hashes
  `acb7adfe0416aa8ceadb6cf8b4a3becd` and `ebc66636a015c59d0eedbf57c01ffe83`); their values (grid, LR grid, warmup 20, clip 1.0, AdamW betas/eps/wd, ten
  checkpoints, the three ties, 6,000 / 3,600 s / 1.5 / 90 s / 11 / 3,000, trainable Q R U B and `A_query` for the ablation, run seeds, train bases, pilot
  200 steps, both bootstrap seeds) match the prereg. `prepare_slot_manifest.py` writes the pending manifest only if the prereg digest and the configs
  match (:41-56) and the lock only from a passed preflight on the parents' files and config now on disk (:59-85); `require_lock` re-derives the registry,
  the disjointness and the inventory (`slot_run.py:62-74`). The D128-3 file digests on disk equal the registered ones (`inventory()`).
- Hashing is blake2b only in all changed files (grep for sha, md5, crc: none).
- Interpretations 1 to 9 (state file :76-89) do not contradict the registration: 1 is an unspecified-tie choice (see Observations); 2 reproduces the
  registered 0.19612866640090942; 3 to 8 are implementation choices consistent with the prereg or its decisions; 9(a) to (d) are stricter than, not
  contrary to, the registration; 9(e) is F1.
- The recorded parent numbers in `slot_preflight.RECORDED` equal the parents' `metrics.json` (S-RE 723, loss 1.642127275466919, read error
  0.19612866640090942; D128-3 608, 2.1291537284851074, 0.36268818378448486).

## Could not verify

- No test was run (rule); the worker's results in the state file (`docs/reports/SLOT-LOOP-v1-state.md:35-44`: the four slot test files and the
  existing suite, all passed) were taken as reported. They ran at `fd8c914`; the reviewed head `ccdcc96` differs from it only in the state file
  (`git diff --stat fd8c914..ccdcc96`), so the in-scope code is the code that was tested.
- No sealed episode was generated (rule). Disjointness from the old sets is at the id level, as registered; whether `test` and `test_unseen` facts
  are disjoint from the fact set S-RE trained on is the generator's partition and was not re-derived.
- `prepare_slot_manifest.py` was not executed end to end (the lock path needs a real preflight report, and `--write-configs` and `--out` would write into
  the repo). Its functions were checked by reading and its inputs by recomputation (configs, registry, prereg digest, inventory).
- The release builder, the sealed-evaluation entries file, the S1A runbook and the ablation stage do not exist in this change; F1, F2 and F5 are
  requirements for them.
- The model-path files (`memory_backend.py`, `slot_read.py`, `slot_experiment.py`, `slot_train.py`, `slot_preflight.py`, `slot_run.py`) belong to the second
  reviewer; I read them only where my files call them.
