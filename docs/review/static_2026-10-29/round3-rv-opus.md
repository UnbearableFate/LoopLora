# SLOT-LOOP-v1 review round 3: rv-opus

## Header
- **Reviewer:** rv-opus, an Opus 5.5 subagent, read-only. I did not read `round3-rv-code.md`. I read both round-2 files, as the brief allows.
- **Date:** 2026-10-29.
- **Commit:** `wp/slot` at `8c283340ad67a7a8eead70c30ebff7b0e9b6a8ed`, detached in `/work/xg24i002/x10041/LoopLora-wt/review-slot`.
- **Diff:** `f1ec035..8c28334`, 9 files. I read all of it.
  - Commits: 29b171a (tests), 11ff92a (fixes), f39b212, 9d22fe9 and 8c28334 (state file only).
  - I also re-read the code the fixes reach: `save_export`/`load_export`, `run_smoke`, `reload_check`, `run_train`'s export, `evaluate_entries`, the `main` CLIs, `prepare_slot_release.prepare`, `prepare_slot_manifest.lock`, `scripts/submit.py` and `scripts/pbs/job.pbs`.
- **Fix list:** `docs/tasks/WP-L13_slot_loop.md`, "Review round 2: fixes before the preflight", fixes 1 to 4.
- **Authority:** `docs/prereg/SLOT-LOOP-v1.md`, blake2b-128 `56e8c611c01e0ef9b7ae32db97b0e334`.

### Checks run
All on the login node: no torch, no pytest, no jobs. Bytecode went to the scratchpad.
- **Export-metadata round trip, stdlib.** Simulated the path: slot dict with `TASK` tuples, then pickle protocol 2, then `smoke.json`, then the JSON release, then a strict `==`.
  - Pre-fix `save_export`: `False`.
  - With the new `json.loads(json.dumps(...))` normalisation (`slot_experiment.py:265`): `True`.
- **Red and green logs.** Read in full:
  - red: `logs/pbs/slot-r2-red-20261029-212940-3263592.log` (commit 29b171a, 22 failed, exit 1);
  - green: `slot-r2-meta-…3287227.log` (209 passed) and `slot-r2-exp-…3287235.log` (59 passed), both at f39b212 and both exit 0.
- **Tests were not changed after the red run.** `git diff 29b171a 8c28334 -- tests/` shows that `tests/test_slot_round2.py` and the eleven `round2` analysis tests are unchanged. Later test edits only add tests and adjust one fixture slice for the new scalar npz fields (`test_analyze_slot_study.py:1261`). `f39b212..8c28334` touches only the state file.
- **Runbook commands checked against the code they call:**
  - `scripts/submit.py` is mode 100755 with a `/usr/bin/python3` shebang;
  - `--commit` resolves through the main repository's shared object store;
  - `job.pbs` runs `"$PYTHON_BIN" "${ARGV[@]}"` inside the snapshot and deletes it on exit (`job.pbs:31-33,52-61,69`);
  - every flag exists in `submit.py`, `slot_run.main`, `prepare_slot_manifest.main` and `analyze_slot_study.main`;
  - every walltime is within the debug-g limit of 1,800 s;
  - `RUNS` is `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1` (`slot_run.py:34`);
  - the `LoopLora-wt/slot` worktree is clean at 8c28334, and `src/` there holds only ignored `__pycache__`, which `source_digest` skips (`slot_data.py:207`).
- **Sealed seeds:** grep over the checkout finds them only in `slot_data.py` and the prereg.

## Verdict
**The preflight may start.** No high or medium finding remains. All four fixes are done on every path the brief names, and each new regression test failed on the pre-fix code for its stated reason. I found no change to a registered number or rule, or to the sealed-set guard.

One low finding remains: R3-1. The confirmation run accepts any ready release, not specifically the one the six arm-seed runs trained under, so an analysis script re-frozen in a rebuilt release would still pass. It can be fixed entirely in `scripts/analyze_slot_study.py` plus the open-record schema, with no `src/` change. If the orchestrator also wants the evaluation side to check it (`slot_data.load_open_token`), that is a `src/` change and belongs before the preflight.

## Per fix

### Fix 1: the reload check can pass on a real release. Done.
- **Code.** `save_export` normalises the whole slot dict through JSON before `torch.save` (`slot_experiment.py:265-266`). The comparison in `reload_check` stays strict (`slot_run.py:324-327`).
- **Path, step by step:**
  1. `run_smoke` writes `export_config` from `torch.load(..., weights_only=True)["slot"]` (`slot_run.py:311`), which is now JSON-shaped.
  2. The builder copies it into `smoke_configs` (`prepare_slot_release.py:172,198`) and, on the ready pass, compares `report.slot == configs[arm]`, with both sides JSON.
  3. The CLI passes `json.loads(release)` to `reload_check` (`slot_run.py:571-572`).
- **No other consumer depends on the old tuple form.** The only readers of `experiment.slot` are `reload_check` (:323-331) and the sealed check in `evaluate_entries` (:502-507). Both compare against JSON values. `run_train` exports go through the same `save_export`.
- **Test.** `test_slot_round2.py:23-56` takes the real path for both arms:
  - `run_smoke` for 20 steps on a synthetic one-layer backbone with the registered adapter dimensions;
  - `smoke.json`, then `builder.prepare`, then the JSON candidate;
  - `slot_run.main(['--mode','reload-check',...])`.

  In the red log, both smokes train and the test then fails on `STOP: the reloaded export config differs from the release`, which is the defect itself and not a setup error. It passes in green, so it would catch a regression.

### Fix 2: confirmation rows are bound to checkpoints. Done.
- **Token.** `OpenToken` keeps `selected_checkpoints` (arm/seed → digest), the locked reference digest and the reference reads from the release (`slot_data.py:233-242,266`). `load_open_token` requires six distinct digests, none equal to the reference (`:256,262`).
- **Evaluation, before anything is opened or written** (`slot_run.py:470-485`, ahead of `mkdir` and of any `eval_set`):
  - exactly nine labels;
  - each main label has an `export`, no parent key, and its digest equals `selected_checkpoints[arm/seed]`;
  - each reference has no export, the locked parent digest and the read from the release.
- **Evaluation, after load** (`:502-508`): the export's `slot.arm`, `run.seed`, `run.ablation is False` and read must match the label. For the first entry this runs before the episodes are generated (`:509-511`).
- **The npz carries a verified digest and read** (`:516-517`). The digest was checked against the file (`:493-494`) or against the parent loader's `expected`.
- **Label kinds.** Main and reference labels are covered as above. Unseen goes through the same code (`sealed=True`, `slot_data.py:73`).
- **Analysis.**
  - `s1a_provenance` (`analyze_slot_study.py:578-623`) builds the expected digest and read per label from the open record, the locked inventory and the released reads, and checks the registered read kind, scope and recency.
  - `read_set` (`:626-660`) refuses any npz whose scalar Unicode `checkpoint_blake2b_128` or JSON `read` differs.
  - `analyze_set` copies both fields into `summary.json` (`:668`), for confirmation and unseen alike (`:690-694`).
- **Tests:**
  - evaluation: five cases through the CLI (`test_slot_round2.py:112-131`: swapped, duplicate checkpoint, reference under a main label, export ablation flag, duplicate label). In red, all exit 0, meaning the old code accepted them, which is the stated defect;
  - analysis: `test_analyze_slot_study.py:910-945` (swapped, duplicated, reference under main, reference digest, read). In red, all report DID NOT RAISE, again the stated defect;
  - a real `save_export`/`load_export` evaluation that checks the npz fields: `test_slot_experiment.py:765-809`.
- **Small gaps, not findings:**
  - No test has the open record itself swap arms, which would reach the post-load arm/seed check. Only the ablation branch of that condition (`:506`) is exercised.
  - The post-load check runs per entry. If the entries list the references first, an open record with wrong arm/seed mapping is detected only after the sealed set was generated and the three reference rows were written. No contrast is computed, the job stops, and a rerun needs a new `--out`, so no registered result is affected.

### Fix 3: the analysis-script digest is bound to the release. Done as specified.
- **Evaluation side.**
  - `load_open_token` requires `status == "ready"`, the registered prereg, and open record = release = file under `--open-root` (`slot_data.py:257-259`). It also requires the release's parent writer to equal the locked inventory (`:262`).
  - Sealed evaluation requires `--release` (`slot_run.py:546-548,567`).
- **Analysis side.**
  - Stage `S1A` requires `--open-record` and `--release` (`:842-846`), a version-1 ready S1A release (`:584-586`), and its own file digest equal to both (`:587-589`). The prereg must match in the manifest, release and open record (`:590-591`).
  - Both file digests go into `source_index.json` (`:116-119`).
- **Tests.**
  - `test_slot_round2.py:134-140` (evaluation) and `test_analyze_slot_study.py:947-961` (analysis): a changed script with a matching open record. Both fail in red with DID NOT RAISE, the stated defect.
  - Six further refusal cases: `:965-993`.
- **Residual:** R3-1. Which release counts as "the ready release" is not fixed.

### Fix 4: the runbook commands run as written. Done.
- **Steps 1 and 3** (`docs/reports/SLOT-LOOP-v1-state.md:96,98`):
  - full resolved commits in `$REVIEW_COMMIT` and `$LOCK_COMMIT`, set at :95 and :97;
  - absolute `--config`, `--manifest` and `--out`, and `$R` equals `RUNS`;
  - S0 runs on the lock commit and reads the committed lock by absolute worktree path.
- **Step 2** (:97) runs in the worktree. `prepare_slot_manifest.py:22-23` puts `$W/src` first, so the lock's `source_digest()` hashes the reviewed source (`prepare_slot_manifest.py:71-72`).
- **Step 4** (:99):
  - `-- scripts/analyze_slot_study.py ...` with no second interpreter, since `job.pbs` supplies it;
  - output to `$R/s0/decision`, which `emit` creates (`analyze_slot_study.py:818-820`) and which survives the snapshot's deletion;
  - the `cp` block (:102-103) names exactly the three files `emit` writes for S0.
- **Test.** `test_slot_round2.py:143-169` checks each command's shape: commit variable, module or script, absolute `--out`/`--manifest`, and the step-4 output directory.
  - Red failures with the stated reason: step 2 on a relative path, step 4 on `python3`. Step 3 failed on its literal `...`, which was itself the unrunnable text.
  - Step 1 failed earlier, on document format: the first backticked span was `debug-g`. So the red log does not show its path or commit check firing.
  - The test would catch a relative path, a wrong commit variable or a second interpreter coming back. It does not check the other flags; I checked those by reading.
- **Outside fix 4's scope, noted.** The S1A reload-check command (:134) still uses the placeholders `C`, `LOCK` and `R/...` and is marked "future". It must be rewritten in the same way before S1A.

## Findings

| id | severity | file:line | concrete failure | evidence | suggested fix |
| --- | --- | --- | --- | --- | --- |
| R3-1 | low | `src/looplora/llm/slot_run.py:567`; `src/looplora/llm/slot_data.py:257-259`; `scripts/analyze_slot_study.py:583-589` | The confirmation evaluation and the S1A analysis accept any file with `status: ready` and a matching analysis digest as "the ready release". They do not require it to be the release the six arm-seed runs trained under. Suppose `analyze_slot_study.py` is edited after training, for example after the orchestrator's S1A-DEV re-run. Rebuilding a release at a new path from the unchanged pilot, S0, smoke and reload reports (one builder call) yields a ready release with the new digest. The open record, the token and stage `S1A` then all agree, and the verdict comes from a script other than the one frozen when training started (prereg :112, :125). Nothing refuses. The mismatch appears only when someone compares `source_index.json`'s `release_blake2b_128` by hand with the `release_blake2b_128` in each training `record.json`. | 1. Each training record stores `release_blake2b_128=file_digest(release_path)` (`slot_run.py:433`). Neither `parse_run` (`analyze_slot_study.py:469-517`) nor `s1a_provenance` (`:578-623`) nor `load_open_token` (`slot_data.py:245-266`) reads it.<br>2. The evaluation CLI loads whatever path `--release` names (`slot_run.py:567`), and so does the analysis (`analyze_slot_study.py:583`).<br>3. The builder refuses only to overwrite the same path (`prepare_slot_release.py:159`), and `analysis_script_blake2b_128` is whatever file is on disk at build time (`:192`). | Bind the release by digest:<br>• the open record carries `release_blake2b_128`, copied from the six training records, which S1A-DEV should require to be equal;<br>• `s1a_provenance` refuses unless the `--release` file's digest equals it.<br>Optionally `load_open_token` checks the same, but that is a `src/` change, so before the preflight. Or read the release only from the committed `reports/slot-loop-v1/s1a-release.json` under the absolute open root, as prereg :112 names it. |

Severity count: high 0, medium 0, low 1.

Noted, not findings:
- The state file's summary of the token still omits the release (`docs/reports/SLOT-LOOP-v1-state.md:80`). The schema section (:129) is current.
- The two small Fix 2 test and ordering gaps above.

## Not checked
- **Nothing executed on compute.** No torch, pytest or GPU run. The red and green results come from the job logs.
  - The integration test's GPU determinism and the real-parent preflight are unverified.
  - The real-size smoke and reload predictions are unverified.
- **No real artefacts exist yet:** no `smoke.json`, release, training record or export. All S1A paths were checked by reading and with a stdlib simulation, not on real files.
- **Unchanged code.** Round-1 and round-2 items outside the fix diff were not re-reviewed beyond the code the fixes touch. That includes `check_disjoint`, the golden assets and the S0 runner numerics.
- **S1A commands.** The S1A release, reload and evaluation runbook commands (state file :131-134, :161) were not checked as runnable. They are placeholders marked future.
