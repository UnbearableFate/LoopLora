# SLOT-LOOP-v1 review round 2: rv-opus

## Header
- **Reviewer:** rv-opus, an Opus 5.5 subagent of the main session, read-only. I did not read `round2-rv-code.md`.
- **Date:** 2026-10-29.
- **Commit:** `wp/slot` at `24385991af6755b687f46d46ba5351a196902b8c`, detached in `/work/xg24i002/x10041/LoopLora-wt/review-slot`.
- **Scope:** `git diff master...2438599`, 25 files. I read every new source file, script, config and test in full. I also read the four changed existing files and the old code they call:
  - `core/adapters.py`, `core/memory.py`, `core/wrapper.py`;
  - `llm/writer_fix.py`, `read_fix.py`, `read_fix_run.py`, `sq2.py`, `dpfp_run.py`, `train.py` (`score`), `protocols.py` (`encode_support`, `answer_scores`), `inject.py`, `prefetch.py` (`BatchStream`);
  - `sym/run.py` (`source_metadata`), `scripts/submit.py`, `scripts/pbs/job.pbs`.
- **Authority:** `docs/prereg/SLOT-LOOP-v1.md`, including decisions 1 to 9. Its blake2b-128 is `56e8c611c01e0ef9b7ae32db97b0e334` at 5f04a5c, at master HEAD and at 2438599 (stdlib check). The task's "Review round 1" fix list and the state file (`docs/reports/SLOT-LOOP-v1-state.md`) are in the review checkout.

### Checks run (login node only)
None of these imported torch, ran pytest, trained, evaluated or submitted a job. Bytecode went to the scratchpad.
- **Seeds and layout.** I re-derived all 14 registered integers with stdlib blake2b (`derive_seed` formula): eight evaluation sets, four train streams and two bootstrap seeds. All equal `slot_data.py:65-83` and the prereg table. The layout digest, recomputed as `stable_hash(asdict(TaskSpec))` from S-RE's `record.json` task block, is `b7f93c1ca56e5dd1f79a9dccfe2f6054`, as registered.
- **Torch-free imports** (venv python): `slot_data.verify_registry()` passes, both committed stage configs equal `slot_rules.stage_config`, and torch was not imported. `source_digest` is `4eef72d9ddf2a8d5db21354d97aff0da` in both the review checkout and the `wp/slot` worktree. `src/` has only ignored `__pycache__`, no `.gitattributes` and no symlinks, so the `git archive` snapshot hashes the same.
- **Bootstrap** (numpy 2.5.4): 20 blocks of `(500, n)` equal one `(10000, n)` draw for n = 2000, 1000 and 7. The script's `interval()` equals `np.percentile([2.5, 97.5])` on 10,000 resampled sums.
- **Recorded parent samples:** `val-6000/samples.npz` and `val-22370/samples.npz` give
  - dtypes `int64`/`bool`/`float32`;
  - 723 and 1.642127275466919 with read error 0.19612866640090942 (S-RE), and 608 and 2.1291537284851074 with 0.36268818378448486 (D128-3), equal to `slot_preflight.RECORDED`;
  - an S-RE `gram_relative_error` maximum of 1.33e-7, below the preflight's 1e-6 tolerance.
- **Finding R2-1, reproduced** with stdlib `pickle` (protocol 2, which is torch's `DEFAULT_PROTOCOL`, `torch/serialization.py:59`) and `json`. I read the weights_only unpickler, which rebuilds tuples (`torch/_weights_only_unpickler.py:476-484`).
- **Sealed seeds:** grepped the whole checkout. They occur only in `slot_data.py:72-73` and the prereg.

### Consistent with the prereg (verified by reading)
- **Matrix (g = 1) against the old read.** It is the old read operation for operation (`memory_backend.py:193-197` against `adapters.py:228-231`), and the token loop mirrors `_masked_branch`. A tensor state takes the inherited path in all three overrides.
- **Support write** runs in `write_only` mode (`protocols.py:109-110`), so it does not depend on Q, R, U or B. Encoding it under `no_grad` (`slot_train.py:83`) is exact.
- **S0:**
  - the 18-candidate grid and its tie rule (runner and script agree);
  - all 13 dev conditions, the D128-3 recheck at S-RE's calibrated g/τ, and recency `log λ^(count-1-i)`;
  - R0 to R5 in order, with exact integer thresholds;
  - D-F5(c): any per-sample difference between Native and Matrix in correctness or NLL gives R0.
- **S1A:**
  - fresh AdamW with the registered settings, a 20-step linear warm-up and clipping at 1.0;
  - the pre-clip norm logged at every log row, and the clipped fraction and every step's loss and norm saved;
  - a 1e-6 tie that keeps the earlier checkpoint, and the 11 dev rows;
  - the N and walltime formulas;
  - the trigger `100·Σd < 9n`;
  - the S1A rule (`100·Σd ≥ 9n`, `L > 0` exactly, each `Δ_s > 0`), with one index matrix shared by every vector.

## Verdict
Nothing I found opens a sealed set early or changes a registered number or decision as the code runs. The preflight, the lock and S0 can proceed as far as the source is concerned. Two medium findings need fixing first:
- **R2-1 stops S1A at the release.** The reload check compares a JSON release with the export's tuples, so it can never pass. Because of M-F2, a fix after the preflight forces a new preflight, so it is cheapest to fix it now.
- **R2-2 leaves the confirmation verdict unbound to the checkpoints.** Nothing ties an evaluation label to the checkpoint the open record names for that arm and seed. A swapped or duplicated entry passes every check, and the verdict record cannot show which checkpoint each row came from.

Two low findings concern the analysis-digest chain and runbook step 4. Nine of the ten fix-list items are done and one is partial (item 7, through R2-1). Runbook step 4 has a separate path defect (R2-4).

## Findings

| id | severity | file:line | concrete failure | evidence | suggested fix |
| --- | --- | --- | --- | --- | --- |
| R2-1 | medium | `src/looplora/llm/slot_run.py:324-327` | Every production reload check ends in `STOP: the reloaded export config differs from the release` (exit 3). The release then never reaches `ready`, and S1A cannot start. Cost: two wasted reload jobs, plus a `src/` fix that forces a preflight rerun under M-F2. | 1. The export's slot metadata stores `task=dict(slot_data.TASK)` (`slot_experiment.py:262`). `TASK` holds the tuples `seen=(0, 768)`, `unseen=(768, 1024)` and `hops=(1,)` (`slot_data.py:38`), and `torch.save` keeps them (`:264`).<br>2. `load_export` reloads with `weights_only=True` (`:275`), which rebuilds tuples, and sets `experiment.slot = slot` (`:289`).<br>3. The release's `smoke_configs` comes from `smoke.json` (`slot_run.py:311,315`, then `prepare_slot_release.py:139,153,198`), so the tuples are now lists.<br>4. The CLI passes `json.loads(release)` (`slot_run.py:542-543`).<br>5. `[0, 768] == (0, 768)` is False in Python. The stdlib reproduction prints `release_cfg == experiment.slot: False`, and `True` after JSON-normalising the slot.<br>6. The tests miss it: `tests/test_slot_experiment.py:534-536,740-745` pass the in-memory `torch.load` dict as the "release", and `tests/test_slot_release.py:64,76-84` use synthetic JSON-shaped configs and reports. | Compare `json.loads(json.dumps(experiment.slot))`, or store `task` JSON-shaped in `save_export`. Add a test that runs the smoke, writes `smoke.json` and a builder release, and calls `main(['--mode','reload-check',...])` or `reload_check(release=json.loads(...))`. Fix it before the preflight, because a `src/` change invalidates the lock. |
| R2-2 | medium | `src/looplora/llm/slot_run.py:471-493`; `src/looplora/llm/slot_data.py:252-254`; `scripts/analyze_slot_study.py:572-621` | The confirmation and unseen rows are bound to arms and seeds only by the label in the operator's entries file. Each of these passes every check and yields a verdict:<br>• a swap of `soft_slot_L1_s17` and `matrix_L1_s17` flips the sign of `Δ_17` and of the pooled `Δ`;<br>• one export listed under two labels;<br>• the S-RE reference listed under a main label.<br>Afterwards, `summary.json` and `source_index.json` hold npz digests but no checkpoint digest. The verdict record cannot show that the six selected checkpoints produced the six rows (prereg :115, :118). | 1. `load_open_token` drops the `arm/seed` keys of `selected_checkpoints` and keeps only a set of digests (`slot_data.py:252-254`).<br>2. `evaluate_entries` (`slot_run.py:472-493`):<br>&nbsp;&nbsp;– checks only that the entry's digest is in that set (`:473`) and that the file matches it (`:478`);<br>&nbsp;&nbsp;– never compares the label with `experiment.slot["arm"]`, `slot["run"]["seed"]` or `slot["run"]["ablation"]`, or with `selected_checkpoints[f"{arm}/{seed}"]`;<br>&nbsp;&nbsp;– does not require six distinct digests;<br>&nbsp;&nbsp;– writes `<label>.npz` without the checkpoint digest (`:490-491`).<br>3. `read_set` and `analyze_set` check only file presence and episode ids (`analyze_slot_study.py:575-595`), then form the contrast from the labels (`:604`).<br>4. The state file promises "all six main labels" (`docs/reports/SLOT-LOOP-v1-state.md:124`), but only the analysis checks presence, and only by name. | 1. Keep the `arm/seed → digest` map in `OpenToken`.<br>2. In `evaluate_entries`, require each main label exactly once and each entry's digest to equal `selected_checkpoints[f"{arm}/{seed}"]`. Require the reloaded export's `slot.arm`, `slot.run.seed` and `slot.run.ablation is False` to match the label. Admit the S-RE digest only for the three reference labels.<br>3. Save `checkpoint_blake2b_128` and `read` in each npz.<br>4. Have stage `S1A` read `s1a-open.json` (or `s1a-selection.json`) and refuse a mismatch. |
| R2-3 | low | `src/looplora/llm/slot_data.py:247-251`; `scripts/analyze_slot_study.py:624-644` | Suppose the analysis script is edited after the release, for example a fix after the S1A-DEV re-run, and the open record names the edited script's digest. The token is then valid, the sealed sets open, and the verdict comes from a script other than the one frozen in the release. Nothing refuses, and the mismatch can be found only by comparing `source_index.json` with the release by hand. | 1. The token compares the record's `analysis_script_blake2b_128` only with the file under `--open-root` (`slot_data.py:250`), never with the release's `analysis_script_blake2b_128` (`prepare_slot_release.py:192`). `evaluate` mode does not read the release (`slot_run.py:536-539`).<br>2. Stage `S1A` records its own digest (`analyze_slot_study.py:109-114`) but compares it with nothing.<br>3. The prereg (:125) requires that the confirmation analysis use the frozen digest unchanged. | Pass the ready release to `load_open_token` and require record = release = file digest. Give stage `S1A` a `--release` (or the open record) and refuse when its own digest differs. |
| R2-4 | low | `docs/reports/SLOT-LOOP-v1-state.md:95,99` | Runbook step 4 cannot write the S0 decision as written. With `--commit C`, the reviewed tree, the snapshot has no lock: `refused: cannot read reports/slot-loop-v1/manifest.lock.json`. On the lock commit, the job exits 0, but `decision_s0.json`, `decision_s0.md` and `source_index.json` land inside the job snapshot, which is deleted at exit. Only the stdout row and digests survive, and the job must be rerun. | 1. `--out reports/slot-loop-v1/s0` and `--manifest reports/...` are relative (`:99`).<br>2. `C` is "the reviewed tree" (`:95`).<br>3. `job.pbs` extracts the commit into `$SCRATCH/src`, `cd`s there and removes `$SCRATCH` on exit (`scripts/pbs/job.pbs:31,33,52,61`).<br>4. The script resolves both paths against the working directory (`analyze_slot_study.py:81-82,752-760`). | Use absolute paths for `--manifest` (the committed lock in a checkout) and for `--out`, under `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/` or a checkout's `reports/`. |

Severity count: high 0, medium 2, low 2.

## Fix list (Review round 1)
1. **D-F5(c): done.**
   - In code: `analyze_slot_study.py:427-432` compares per-sample `correct` and `nll` of `native/AV` and `matrix_uncal/AV` with no tolerance, and returns R0 on any difference.
   - Tests: `tests/test_analyze_slot_study.py:226-252`.
2. **D-F5(a): done.**
   - `--manifest` is required (`:767`), and its digest goes into `source_index.json` (`:112-113`).
   - Count and id digest are refused on mismatch for S0 calibration and dev (`:425-426`), S1A dev (`:519-520`) and confirmation/unseen (`:631-632`).
   - All five sets are validated up front (`:777-778`), and the test now expects a refusal (`tests/test_analyze_slot_study.py:1011-1034`).
3. **M-F2: done.**
   - `source_digest` covers `src/` without bytecode (`slot_data.py:203-216`). The preflight writes it (`slot_run.py:247`), and the lock requires it equal to the current tree (`prepare_slot_manifest.py:71-74`).
   - `require_lock` checks it in every science mode, including reload-check (`slot_run.py:71-72`). Tests: `tests/test_slot_experiment.py:452-467` and `tests/test_slot_data.py:364`.
   - The state file says to rerun the preflight after any `src/` change.
4. **D-F3: done.** Interpreters are given for the lock (state file :97, `prepare_slot_manifest.py:8-11`) and for the analysis (:99). Step 4 has a separate path defect, R2-4.
5. **D-F4: done.**
   - `raw_episodes` refuses a sealed, unopened set (`slot_experiment.py:76-80`), and `episodes` goes through it. `eval_set` returns an opened copy (`slot_data.py:265`).
   - `manifest()` omits the sealed `base_seed` and `query_seed` (`slot_data.py:187`).
   - Tests: `tests/test_slot_experiment.py:676-681` and `tests/test_slot_data.py:355`.
6. **D-F1: done.**
   - Stage `S1A-ABL` (`analyze_slot_study.py:552-567`) needs complete records with `ablation: true`, the same n, dev ids and steps as the six main runs, and the dev-NLL-selected checkpoint. It flags iff `(c_soft - c_matrix)·100 ≥ 3n` in integers and prints counts and the flag.
   - `parse_run` requires `ablation is False` for the main records (`:465-466`).
   - The release carries both ablation hashes (`prepare_slot_release.py:175-177`). Tests: `tests/test_analyze_slot_study.py:758-853`.
7. **M-F3/D-F2: partial.**
   - Done:
     - the schema is in the state file (:111-124);
     - `load_open_token` admits the S-RE digest from the locked inventory (`slot_data.py:252-254`), with a reference-entry test (`tests/test_slot_experiment.py:765-788`);
     - `--open-root` has no default and must be absolute (`slot_run.py:516-519`, `slot_data.py:243-244`);
     - the release builder self-checks the eight config hashes, N, LR, g/τ, walltime and the prereg and analysis digests (`scripts/prepare_slot_release.py`).
   - Not working: `reload_check` compares with the release (`slot_run.py:323-327`), but a JSON release can never pass that comparison (R2-1).
8. **D-F5(b): done.**
   - Missing reference files are refused (`analyze_slot_study.py:575-578`). An absent unseen set is reported as "not run", and a partial one is refused (`:626-628`, `:641-643`).
   - Tests: `tests/test_analyze_slot_study.py:986-1009`.
9. **M-F1: done.**
   - The pilot catches only `FloatingPointError`, records `nll: null, all_finite: false, error`, and continues (`slot_run.py:363-370`). Excluded candidates enter neither the LR choice nor the timing (`:376-385`, `slot_rules.py:58`). Other errors still propagate.
   - `fit` turns non-finite loss, gradient, norm, logged values and dev NLL into `FloatingPointError` (`slot_train.py:37-38,87-99,110-111`).
   - Tests: `tests/test_slot_experiment.py:693-731,752-763`.
10. **M-F4: done.** The state file :108 notes that the conflict ends in a STOP. No rule changed.

## Not checked
- **No compute.** Nothing ran on compute: no torch import, no pytest, no real parents, no GPU. As a result:
  - bitwise determinism on GH200 (preflight Native/Matrix, the pilot's equal step-0 NLLs required by `prepare_slot_release.py:90`, the reload predictions) is unverified;
  - the real-model preflight tolerances are unverified, apart from the recorded `gram_relative_error` margin;
  - the compute results of the tests (182 + 48 passed) are taken from the state file.
- **Disjointness and real-file hashing.** I did not rerun `check_disjoint()` (about 1.5 M hashes) or `inventory()` on the login node. The tests ran them on compute, and round 1 reimplemented the former.
- **Golden test.** I did not review `tests/slot_golden.py` and `tests/assets/slot_golden_digests.json` beyond their role. Round 1 covered them, and no existing file changed since then except the four typed dispatches, which I re-read.
- **Optional common-fixed-LR comparison** (prereg :124). It has no config hash in the release, so `run_train` would refuse it. This is noted, not a finding, because the comparison is optional and descriptive.
- **Process gates.** I did not check the S1B addendum or the orchestrator's process gates (`approval.json`, the decision to run S1A after S0, the trigger before the ablation runs). The code has no hook for them and the prereg assigns them to the orchestrator.
