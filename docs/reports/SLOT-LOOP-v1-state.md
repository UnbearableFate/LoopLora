# WP-L13 state: SLOT-LOOP-v1, S0 complete (R2)

Branch `wp/slot` (worktree `LoopLora-wt/slot`), master merged through `8632497` after handover at `ccdcc96`; not merged into master, not pushed. Task: `docs/tasks/WP-L13_slot_loop.md`; registration `docs/prereg/SLOT-LOOP-v1.md` (blake2b-128 `56e8c611c01e0ef9b7ae32db97b0e334`, registered on master at `5f04a5c`, unchanged); S0 results are in `docs/reports/SLOT-LOOP-v1.md`.
**Status: S0 accepted (R2); S1A follow-ups, smoke, pilot and release preparation authorized.** Stop after the ready release is committed for approval.json. No training or sealed evaluation is authorized before the later gates. No src changes; the preflight source binding remains valid.

## What the code does
New files only, except four typed-dispatch edits (below). Every new option is off in the old paths.
- `core/memory_backend.py` (torch only, no LLM imports): `MatrixState`, `SlotState`, `build_slots`, `MemoryBackend` / `MatrixBackend` / `SoftSlotBackend` / `HardSlotBackend`, `MemoryRead`, `CapacityError`, `persistent_bytes`, `state_digest`, `select_rows`, `reset_rows`.
- `llm/slot_read.py`: `SlotFixAdapter` (the S-RE / D128-3 `FixAdapter` that reads through a `MemoryRead`; the inherited path is untouched), `ReadSpec`, `bind_batch`, `final_read`, `ScopedRead` (S0's two-state final-token diagnostic).
- `llm/slot_data.py` (torch-free): the data registry, purpose-seed guard (sealed S1A confirmation and unseen-name sets open only with an `OpenToken` built from `reports/slot-loop-v1/s1a-open.json`), episode-id disjointness checks, `manifest()`, `inventory()`.
- `llm/slot_rules.py` (torch-free): calibration grid and tie rule, LR choice, step count and walltime, checkpoint selection, `science_config`, `stage_config`.
- `llm/slot_experiment.py`: parent loaders (`load_parent`: file digests checked before the load), evaluation (`evaluate`, chunks of 16 as `read_fix_run.validate`), `summarize`, freeze verification (i) to (iv), exports (`save_export`, `load_export`: adapter, buffers, backend settings, scope).
- `llm/slot_train.py`: `fit`, the S1A trainer (fresh AdamW, 20-step warm-up then constant LR, clip 1.0 with `error_if_nonfinite`, answer-only cross-entropy at L = 1; pre-clip norm and clipped flag logged; `train_steps.npz` keeps every step's loss and norm).
- `llm/slot_preflight.py`: the preflight checks (Native replay against the recorded 723/1000 samples, Matrix against Native bitwise with the maximum difference reported, owner and mask sentinels, read-only isolation, feature consistency, bytes, capacity errors, the frozen set under a short training run).
- `llm/slot_run.py`: the runner (`--mode preflight | s0 | smoke | pilot | train | evaluate | reload-check`), the manifest lock gate, the S0 pipeline, the guarded evaluation of sealed sets.
- `scripts/prepare_slot_manifest.py` (login node, numpy and stdlib): writes the stage configs, the pending manifest, and (after a passed preflight) the locked manifest and the checkpoint inventory.
- `configs/slot_loop_v1/s0.json`, `s1a.json`: the frozen stage configurations (the runner refuses a file that differs from `slot_rules.stage_config`).
- `scripts/analyze_slot_study.py`, `tests/test_analyze_slot_study.py`: the frozen analysis (S0 decision rows R0 to R5 and the S1A verdict).

### The four edits to existing files (12 insertions, 4 deletions)
`llm/batching.py` (`repeat_snapshot` expands a `MemoryRead` by owner, a tensor state still goes through `repeat_interleave`), `llm/inject.py` (`AdapterBank.use`: a `MemoryRead` checks its tensors for fp32 and is cloned as a handle; a tensor state keeps the fp32 check), `llm/protocols.py` and `core/wrapper.py` (type hints and imports only).

## The old matrix path is unchanged
1. **Source diff.** `git diff c6af756..HEAD` (c6af756 is the last commit before any slot code) touches four existing files, 12 insertions and 4 deletions, each a typed dispatch on `isinstance(state, MemoryRead)`. All other changed paths are new files.
2. **Golden digests.** `tests/slot_golden.py` hashes (blake2b-128) the matrix state, the keys and values, the logits at L = 1, 2 and 4 for a four-question training-shaped batch and a one-question evaluation-shaped batch, the question-site inputs, and the adapter gradients of one training loss, for a tiny S-RE-style experiment (2-layer backbone, registered rank-128 DPFP/ridge adapter). Job 3727254 computed them on the source of c6af756 twice in fresh processes (identical), and on the capture commit (identical to c6af756's). They are pinned in `tests/assets/slot_golden_digests.json`; `test_the_old_matrix_path_is_bitwise_unchanged` recomputes them with the current tree in a fresh process and requires equality.
3. **Matrix (g = 1) against Native** through the new backend: logits bitwise equal at L = 1, 2, 4 for both batch shapes and the training gradients bitwise equal (`test_matrix_backend_logits_and_gradients_equal_native_bitwise`); the preflight repeats the logits check on the real S-RE over the old range, with the maximum difference reported (decision 5: a difference stops before any calibration or dev data are read).
4. **The existing suite** (every test file except the slot ones) ran on compute at `d6a5f9e` in two jobs: 3727337 (files `test_[a-l]*`) 1,054 passed, 3727338 (`test_[m-z]*` without `test_slot_*`) 1,864 passed and 1 skipped; no failure or error, both runs reached 100 %. Since `d6a5f9e` no existing file and no file they import has changed (`git diff d6a5f9e..fd8c914` lists `slot_experiment.py`, `slot_preflight.py`, `slot_run.py`, the new analysis script and tests, and this file), so the result holds for the reviewed tree.

## MemoryBackend interface (stable for S2 and S3 workers)
`MemoryState = MatrixState | SlotState`. `MatrixState.matrix` is `[B, dv, dk]`; `SlotState` is `keys [B, S, dk]`, `values [B, S, dv]`, `valid [B, S]` (a prefix of True), `count [B]`. `build_slots(keys, values, capacity, valid=None)` pads with zeros to S slots and raises `CapacityError` when a row has more facts than slots (nothing is truncated).
`MemoryBackend.read(state, query [N, dk], owner [N]) -> [N, dv]` reads only the memory of episode row `owner[n]`; backends hold settings (`gain`, `tau`, `decay`), never memory. `MatrixBackend`: `g M q`, the original operation (matmul). `SoftSlotBackend(tau, gain, decay=None)`: `g sum softmax(k.q/tau [+ log lam^age]) v` over valid slots. `HardSlotBackend(gain)`: `g v_j`, `j = argmax k.q`, ties to the lowest index. An empty row reads zero without NaN. `online_write` refuses (there is no silent skip). Everything is functional: no function writes into a state it was given.
`MemoryRead(backend, state, owner)` is the handle the adapter receives instead of a tensor state: `bind`, `expand(Q)` (repeats owners only), `select(rows)`, `read(query)`, `clone`, `check_fp32`. Bytes of one episode: `persistent_bytes(state)` = `numel * element_size` over all state tensors (S = 16 slots: 57,368; matrix: 393,216).
In the adapter, `SlotFixAdapter.process_sequence` and `_masked_view_branch` do the reading; a tensor state falls through to the inherited code. `ReadSpec(kind, gain, tau, scope, recency, capacity)` plus `bind_batch(experiment, batch, spec)` is the entry point for experiments; `slot_read.final_read` returns the read `m` and its statistics at each question's final token.

## Tests (compute)
Pre-review slot tests ran at `fd8c914` on `debug-g` (`ccdcc96` differs from it only in this file); dots are `pytest -q` progress characters, none of the runs had a failure, an error or a skip. The tiny model has two layers (hidden 16) with the registered rank-128 DPFP/ridge adapter at the last layer, T1 with E = 16, so the real tokenizer, task generator and adapter code run.
| file | job | result | what it checks |
| --- | --- | --- | --- |
| `tests/test_slot_backend.py` | 3727670 | 23 passed | the three backends against hand-computed cases, Matrix equal to the original read bitwise, small tau approaches Hard, empty rows read zero without NaN, padding slots inert, Hard ties to the lowest slot, the recency weight, owner-indexed reads under reordering, expansion and padding, bad owner and dtype refusals, the exact capacity boundary, byte accounting (masks and counts included), read-only state and `online_write` refusal, the scoped final-token read and the oracle, `ReadSpec` validation |
| `tests/test_slot_data.py` | 3727670 | 39 passed | registered integers re-derived, a changed one refused, all sets and train streams disjoint from the old protected ids and from each other (overlap found for an old train seed, an old window, two sets), the old `CONFIRMATION_WINDOWS` pinned, sealed seeds only in `slot_data.py`, the open-token guard (valid record, wrong checkpoint, five of six checkpoints, wrong prereg or script digest, script edited after the record), the registered rules at their boundaries (calibration choice and ties, LR choice, step count, walltime, checkpoint choice and ties), `science_config` and `science_hash`, the committed stage files equal the registered rendering, the manifest script (pending manifest, lock only from a passed preflight on the files now on disk) |
| `tests/test_slot_experiment.py` | 3727662 | 36 passed | the layout fixture; Matrix equal to Native bitwise (logits at L = 1, 2, 4 for four questions and for one, gradients); **the old matrix path against the golden digests**; tensor states keep the original paths; reads at the registered tokens for each owner (matrix, soft, hard, scoped, oracle); adapter checks (keys, values, matrix consistency, bytes, capacity, read-only); the freeze verification passes and catches each defect; `standardize` refused; export reload with buffers, settings and scope and reproduced predictions; the ablation query copy; the trainer (log, pre-clip norms, clipped fraction, selection by the rule, export, NaN stop); the S0 pipeline structure; the runner gates (lock, stage config, sealed sets); the S1A smoke with reload check, the pilot, an arm-seed run and its export evaluated to the same numbers, five ways a run can differ from its release (it does not start); the runner's S0 output and arm-seed record read by the frozen analysis script |
| `tests/test_analyze_slot_study.py` | 3727670 | 70 passed | the frozen analysis: R0 to R5 at their exact thresholds, calibration and checkpoint ties, the bootstrap draw and interval, the S1A verdict at +3.0 pp, one seed at zero and `L <= 0`, refusals of malformed inputs, byte-deterministic outputs |
| existing suite, files a to l | 3727337 (commit `d6a5f9e`) | 1,054 passed | |
| existing suite, files m to z without the slot tests | 3727338 (commit `d6a5f9e`) | 1,864 passed, 1 skipped | |
Defects the tests found in my own code (fixed, no assertion weakened): the fix container refused a state dict that already holds the buffers (`SlotFixAdapter`); `tensor_digests` and the golden helper could not byte-view 0-dim tensors (`a_lambda`, `a_beta`); a fixture left the backbone locked after a failed setup; `evaluate_entries` could not take a shared backbone; and a fixture that patched the registered pilot length, which the runner correctly refused because it is part of the stage config.

## Job ledger and node·h (S0 cap 1.0 node·h; S1A cap 4.5)
Elapsed time is the `start` to `end` of the `[job]` lines in each log (1 node each). All jobs `debug-g`.

| job | name | commit | result | node·h |
| --- | --- | --- | --- | --- |
| 3727185 | slot-golden | `0758987` | cancelled while queued (20 min walltime, resubmitted shorter) | 0 |
| 3727186 | slot-t-backend | `0758987` | failed: one test compared a digest with a tuple (test defect) | 0.006 |
| 3727199 | slot-t-backend | `05371c9` | 23 passed | 0.006 |
| 3727224 | slot-golden | `05371c9` | failed: `git archive` ran inside a snapshot without `.git` | 0.006 |
| 3727233 | slot-golden | `bfeedc9` | failed: child stderr hidden (then diagnosed) | 0.009 |
| 3727234 | slot-t-exp | `bfeedc9` | failed: the fix container refused a state dict that already holds the buffers; missing golden asset | 0.006 |
| 3727243 | slot-golden | `3f81746` | failed: 0-dim tensors cannot be byte-viewed (test helper) | 0.009 |
| 3727244 | slot-t-exp | `3f81746` | 19 passed, 7 failed: the same 0-dim defect in `tensor_digests`, missing golden asset | 0.008 |
| 3727254 | slot-golden | `38a923d` | passed: old tree twice and current tree identical | 0.014 |
| 3727255 | slot-t-exp | `38a923d` | cancelled while queued (superseded) | 0 |
| 3727256 | slot-t-data | `38a923d` | 39 passed | 0.020 |
| 3727336 | slot-t-exp | `d6a5f9e` | cancelled while queued (superseded) | 0 |
| 3727337 | slot-suite-a | `d6a5f9e` | existing suite, files a to l: 1,054 passed | 0.206 |
| 3727338 | slot-suite-b | `d6a5f9e` | existing suite, files m to z without the slot tests: 1,864 passed, 1 skipped | 0.133 |
| 3727342 | slot-t-exp | `ab4878c` | cancelled while queued (superseded) | 0 |
| 3727362 | slot-t-exp | `9c9414b` | cancelled while queued (superseded: the analysis script was not in that commit) | 0 |
| 3727479 | slot-t-exp | `21c7e81` | 26 experiment tests and 70 analysis tests passed, S0 cross-check passed, 9 runner tests failed: the fixture patched `PILOT_STEPS` (test defect) | 0.013 |
| 3727662 | slot-t-exp | `fd8c914` | 36 passed | 0.046 |
| 3727670 | slot-t-rest | `fd8c914` | 132 passed (23 + 39 + 70) | 0.022 |
| | | | **total to the review** | **0.503** |

Cumulative against the S0 cap of 1.0 node·h: 0.503 used, 0.497 left for the preflight and S0 (estimated 0.05 and 0.15 from the SQ2 evaluation times of 6 to 9 s per 1,000 episodes; the two largest items so far were the existing-suite shards, 0.34). S1A starts its own 4.5 node·h account.

## Interpretations and implementation choices (for the review)
1. **Calibration tie, "the `g` nearest 1".** Implemented as the smallest `|g - 1|` (0.5 is nearer to 1 than 2.0 is), then the larger `tau`. Test: `test_the_calibration_choice_is_the_lowest_nll_and_ties_go_to_the_gain_nearest_one_then_the_larger_tau`.
2. **Matrix read error in S0 uses the SQ2 report's einsum form** (`read_fix.question_terms`, `bij,bqj->bqi`), so the old-range replay reproduces 0.19612866640090942 exactly. The deployed per-token read is a matmul; the preflight checks that the Matrix backend's logits equal Native's bitwise, which is the claim that matters for training and decoding. A bitwise difference between the two forms in the diagnostic alone would not change any logit.
3. **Stage configs are JSON** (`configs/slot_loop_v1/s0.json`, `s1a.json`), written by `scripts/prepare_slot_manifest.py --write-configs` from `slot_rules.stage_config`, and refused by the runner and by the manifest script when they differ from it. The repository's other studies use YAML for run configs; these are frozen renderings of registered values, not editable inputs.
4. **Support under `torch.no_grad()` in S1A training.** The writer is frozen, so the support keys, values and matrix are constants of a step; no gradient reaches the trainable set Q, R, U, B through the write path (A is frozen and shared). Check (iv) confirms that Q, R, U, B have finite non-zero gradients and the frozen tensors have none, after the first backward.
5. **Sealed sets.** The confirmation and unseen-name sets are defined by their own base seeds in `slot_data.py` alone (a source scan test requires that no other file under `src/`, `scripts/`, `tests/`, `configs/` contains them); `slot_data.eval_set` refuses them without the `OpenToken` that `load_open_token` builds from the orchestrator's `reports/slot-loop-v1/s1a-open.json` (registered prereg digest, analysis script digest equal to the ready release and file, six selected checkpoints, plus locked reference digest/read configs). `slot_run.evaluate_entries` evaluates a sealed set only for a checkpoint digest the token names and prints counts and digests, never metrics. The old `CONFIRMATION_WINDOWS` are pinned by a test and not extended.
6. **Locked manifest.** Every science mode refuses without `manifest.lock.json`: registry and episode-id disjointness re-derived and equal to the lock's, the real parents' file digests equal to the lock's inventory, and a passed preflight. `prepare_slot_manifest.py` writes the lock only from a preflight report whose required checks all passed, on the files and configuration now on disk.
7. **Matrix against Native is a stop, not a warning.** `matrix_equals_native` is a required preflight check with the maximum absolute logit difference in the report; `prepare_slot_manifest.py` refuses a lock without it (decision 5).
8. **Fix container and buffers.** The WP-L3 fix container (`_WriterFix`) loads its initial adapter's state dict before it registers `input_mean` and `input_std`; `SlotFixAdapter` therefore builds from a view of the loaded adapter without the two buffers and then loads the complete state dict strictly, so weights and buffers equal the loaded S-RE's exactly.
9. **Frozen analysis script** (`scripts/analyze_slot_study.py`; stdlib and numpy; stages `S0`, `S1A-DEV`, `S1A-ABL`, `S1A`; written by a Sonnet 5.5 subagent from the registered rules, reviewed here; blake2b-128 not frozen until the release). Choices the prereg leaves open:
   (a) R0 requires all nine preflight checks (`native_replay`, `matrix_equals_native`, `owner_mask`, `isolation`, `frozen_set`, `bytes`, `capacity`, `feature_consistency`, `d128_3_replay`), as the runner's report lists them; a missing or false one gives R0 and no S0 data are opened.
   (b) The 10,000 bootstrap index rows are drawn as 20 sequential blocks of `rng.integers(0, n, size=(500, n))` from `default_rng(seed)`, one index matrix shared by every vector; this equals a single `(10000, n)` draw (pinned by a test). The percentile interval is numpy's linear interpolation at 2.5 and 97.5, evaluated exactly (`L > 0` is an exact test); thresholds (+5.0 pp, +3.0 pp) are integer comparisons.
   (c) A non-`complete` arm-seed record is refused (nonzero exit); `stop_unstable` is the outcome for a complete record with non-finite training, no finite selectable checkpoint, or a selected dev NLL above its step-0 NLL (equality is stable).
   (d) The calibration and checkpoint ties are compared exactly on the recorded doubles; the runner compares floats. They can differ only within about one ulp of 1e-4 or 1e-6, and then the script refuses ("differs from the registered rule") instead of repairing.
   (e) Superseded by Review round 1 item 6: the ablation comparison is now in the frozen script before release.
   (f) **Registered by Review round 1 item 6.** `S1A-ABL` reads the six main records (`ablation: false`) and `ablation/{matrix_L1,soft_slot_L1}/s17/record.json` (`complete`, `ablation: true`). Counts and dev id digests must agree with each other and the locked manifest. Each arm uses its dev-NLL-selected checkpoint under the main selection rule. Flag iff `(c_soft - c_matrix) * 100 >= 3 n`, using integers. Stdout carries counts and the flag; outputs contain no confirmation information.
   Tests: `tests/test_analyze_slot_study.py` (70 cases, numpy and pytest only) and two cross-checks in `tests/test_slot_experiment.py` that feed the runner's own S0 output and arm-seed record to the script.

## Not done / next
- Reviews closed: steps 1–4 completed; retain the commands below as the execution runbook. Commit the lock on wp/slot only after every required preflight check passes, including bitwise Matrix–Native identity; otherwise STOP before calibration/dev. End S0 with STOP: S0 done.
- After cross-review approval, use these commands from this worktree. These steps are now authorized by the user. Set `R=/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1`, `W=/work/xg24i002/x10041/LoopLora-wt/slot`, then `cd "$W"` and `REVIEW_COMMIT="$(git rev-parse HEAD)"` on the reviewed head. Replace no hashes with `HEAD` in `submit.py`; the variables below hold full resolved hashes.
  1. Preflight (one node, debug-g, estimated 0.05 node·h): `scripts/submit.py --name slot-preflight --queue debug-g --walltime 00:20:00 --commit "$REVIEW_COMMIT" -- -m looplora.llm.slot_run --mode preflight --config "$W/configs/slot_loop_v1/s0.json" --out "$R/preflight"`. It reads no calibration/dev data. Wait through the orchestrator, then inspect the complete log and required checks; failure or any Matrix–Native difference ends in STOP before the lock.
  2. Lock (login metadata only, after passed preflight): `/work/xg24i002/x10041/LoopLora/.venv/bin/python scripts/prepare_slot_manifest.py --configs "$W/configs/slot_loop_v1" --preflight-report "$R/preflight/preflight.json" --out "$W/reports/slot-loop-v1/manifest.lock.json" --inventory-out "$W/reports/slot-loop-v1/checkpoint_inventory.json"`. Copy the preflight report into the worktree, commit it with the lock and inventory on wp/slot, and only then set `LOCK_COMMIT="$(git rev-parse HEAD)"`. Any src change requires a new preflight.
  3. S0 (one node, debug-g, estimated 0.15 node·h): `scripts/submit.py --name slot-s0 --queue debug-g --walltime 00:25:00 --commit "$LOCK_COMMIT" -- -m looplora.llm.slot_run --mode s0 --config "$W/configs/slot_loop_v1/s0.json" --manifest "$W/reports/slot-loop-v1/manifest.lock.json" --out "$R/s0"`. This runs on the post-lock commit and writes all metrics/samples outside its deleted snapshot. Wait through the orchestrator and inspect outputs before analysis.
  4. Decision (compute, numpy only): `scripts/submit.py --name slot-s0-analysis --queue debug-g --walltime 00:03:00 --commit "$LOCK_COMMIT" -- scripts/analyze_slot_study.py --stage S0 --records "$R/s0" --out "$R/s0/decision" --preflight "$R/preflight/preflight.json" --manifest "$W/reports/slot-loop-v1/manifest.lock.json"`. The job wrapper supplies Python. After completion, copy the three persistent files as shown below, update the report and ledger, then STOP: S0 done with the full hash, decision row and S1A choices.

```bash
mkdir -p "$W/reports/slot-loop-v1/s0"
cp "$R/s0/decision/decision_s0.md" "$R/s0/decision/decision_s0.json" "$R/s0/decision/source_index.json" "$W/reports/slot-loop-v1/s0/"
```
- S1A (smoke, pilot, release, six arm-seed runs, analysis) only after the orchestrator's OK.


## Review round 1 changes and contracts (wk-slot)
- D-F5: S0 gives R0 if Native and uncalibrated Matrix differ in any per-episode correctness or NLL. Analysis requires `--manifest` for every stage, records its file digest in `source_index.json`, and refuses any count or ordered-id digest mismatch against the lock (S0 calibration/dev, S1A dev/confirmation/unseen). References `s_re_native`, `matrix_step0`, `soft_step0` are mandatory in each evaluated sealed set. An entirely absent unseen set is reported as `not run`; a partial set is refused.
- M-F2: preflight writes `source_blake2b_128`; lock creation requires that digest to equal the current `src/` contents and stores it in `preflight.source_blake2b_128`. Every science mode, including reload-check, checks it. Sorted relative paths and file bytes are hashed with blake2b-128; generated bytecode is excluded. Scripts/docs edits do not invalidate this check. **After any `src/` change, rerun preflight before science work.**
- D-F4: sealed registry objects carry `opened: false`; `eval_set` returns an opened copy only with a valid token and allowed checkpoint. `raw_episodes` and `episodes` reject the unopened object. Manifest set metadata omits sealed `base_seed` and `query_seed`, retaining counts and id digests.
- M-F1: non-finite loss, gradient, gradient norm, or dev NLL is a `FloatingPointError`. The pilot alone catches it and writes `{arm, lr, nll: null, all_finite: false, error}`. It excludes these candidates from LR selection and from S/Ev timing maxima; the first load still supplies Ld. Other runtime errors propagate. If no finite candidate remains for an arm, `pilot.json` is written and the runner stops.
- M-F4: no numerical rule changed. A first load long enough to make the registered N rule conflict with the 3,600 s walltime cap produces a STOP; the worker does not reduce N or extend walltime on its own.
- Only slot-specific modules changed imports; no existing non-slot test file imports the modified modules. The previous full existing-suite results remain the evidence for unchanged legacy paths; changed slot tests and the new release-builder tests run again in this round.

### Open-record schema and evaluation entries
The orchestrator writes the following JSON at `ABS_ROOT/reports/slot-loop-v1/s1a-open.json` after independent dev analysis and selection. `ABS_ROOT` also holds the registered prereg and the frozen analysis script. Use an explicit **absolute** `--open-root ABS_ROOT`; there is no default (a commit snapshot will usually predate this record).

```json
{
  "prereg_blake2b_128": "56e8c611c01e0ef9b7ae32db97b0e334",
  "analysis_script_blake2b_128": "<32 lowercase hex digits frozen in the release>",
  "release_blake2b_128": "<blake2b-128 of the exact ready-release file>",
  "selected_checkpoints": {
    "matrix_L1/17": "<digest>", "matrix_L1/29": "<digest>", "matrix_L1/43": "<digest>",
    "soft_slot_L1/17": "<digest>", "soft_slot_L1/29": "<digest>", "soft_slot_L1/43": "<digest>"
  }
}
```
`load_open_token(ABS_ROOT, lock["inventory"], ready_release)` also admits the S-RE `writer.pt` digest from the already checked locked inventory; no seventh digest is needed in the open record. Saved-run evaluation entries are `{label, export: ABS_PATH, checkpoint_blake2b_128}`. Reference entries are `{label, parent_writer_blake2b_128, read}`: `s_re_native` uses `read: null`; `matrix_step0` and `soft_step0` use the calibrated arm read configs from the release, all with the locked S-RE writer digest. Each evaluated set requires all six main labels (`<arm>_s<seed>`) and these three references.

### S1A execution runbook (current preparation; later commands remain gated)
Resolve the full tested head once before the smoke. Freeze the analysis script through the release. All artifacts use persistent absolute paths; fresh outputs must not already exist. The shared parent/inventory and source-bound lock are reused unchanged.

```bash
R=/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1
W=/work/xg24i002/x10041/LoopLora-wt/slot
cd "$W"
SCIENCE_COMMIT="$(git rev-parse HEAD)"
OPEN_ROOT="$R/open-root"
cp "$W/reports/slot-loop-v1/manifest.lock.json" "$R/manifest.lock.json"
cp "$W/configs/slot_loop_v1/s1a.json" "$R/s1a.json"
mkdir -p "$OPEN_ROOT/docs/prereg" "$OPEN_ROOT/scripts" "$OPEN_ROOT/reports/slot-loop-v1"
cp "$W/docs/prereg/SLOT-LOOP-v1.md" "$OPEN_ROOT/docs/prereg/SLOT-LOOP-v1.md"
cp "$W/scripts/analyze_slot_study.py" "$OPEN_ROOT/scripts/analyze_slot_study.py"
scripts/submit.py --name slot-s1a-smoke --queue debug-g --walltime 00:02:00 --commit "$SCIENCE_COMMIT" -- -m looplora.llm.slot_run --mode smoke --config "$R/s1a.json" --manifest "$R/manifest.lock.json" --s0-metrics "$R/s0/metrics.json" --out "$R/s1a-smoke"
```

Wait for the orchestrator watcher, then verify both 20-step arms, all frozen/support/gradient checks, finite losses/norms, selected exports and smoke predictions. Derive pilot walltime from the larger measured per-step time and 1,000-episode evaluation timing (the S0/preflight measurements supply the initial evaluation estimate); include all six 200-step candidates and load/setup with a 1.5 margin. The initial pilot request below is 10 minutes; update it to the measured estimate if needed, keeping debug-g at most 30 minutes.

```bash
scripts/submit.py --name slot-s1a-pilot --queue debug-g --walltime 00:10:00 --commit "$SCIENCE_COMMIT" -- -m looplora.llm.slot_run --mode pilot --config "$R/s1a.json" --manifest "$R/manifest.lock.json" --s0-metrics "$R/s0/metrics.json" --out "$R/s1a-pilot"
```

After completion, inspect all candidates, exclusion records, LR selection, S/Ev/Ld, N and walltime. A registered STOP remains a STOP; do not silently alter N/LR. Build a candidate, wait for its self-checks, then reload each arm against that persisted JSON.

```bash
scripts/submit.py --name slot-s1a-cand --queue debug-g --walltime 00:01:00 --commit "$SCIENCE_COMMIT" -- scripts/prepare_slot_release.py --manifest "$R/manifest.lock.json" --pilot "$R/s1a-pilot/pilot.json" --s0-metrics "$R/s0/metrics.json" --smoke "$R/s1a-smoke/smoke.json" --science-commit "$SCIENCE_COMMIT" --out "$R/s1a-candidate.json"
for arm in matrix_L1 soft_slot_L1; do
  scripts/submit.py --name "slot-reload-$arm" --queue debug-g --walltime 00:01:00 --commit "$SCIENCE_COMMIT" -- -m looplora.llm.slot_run --mode reload-check --manifest "$R/manifest.lock.json" --release "$R/s1a-candidate.json" --export "$R/s1a-smoke/$arm/selected.pt" --predictions "$R/s1a-smoke/$arm/smoke_predictions.npz" --out "$R/reload-$arm.json"
done
```

After both reports pass exact config and prediction checks, create the ready release. Copy the candidate/final release, pilot/smoke summaries and reload reports into reports/slot-loop-v1, commit on wp/slot and STOP: release with the full commit. Do not train before approval.json.

```bash
scripts/submit.py --name slot-s1a-release --queue debug-g --walltime 00:01:00 --commit "$SCIENCE_COMMIT" -- scripts/prepare_slot_release.py --manifest "$R/manifest.lock.json" --pilot "$R/s1a-pilot/pilot.json" --s0-metrics "$R/s0/metrics.json" --smoke "$R/s1a-smoke/smoke.json" --science-commit "$SCIENCE_COMMIT" --reload-check "matrix_L1=$R/reload-matrix_L1.json" --reload-check "soft_slot_L1=$R/reload-soft_slot_L1.json" --out "$R/s1a-release.json"
```

The following complete commands are for later, after approval and the registered gates. The ready release fixes science_commit; continue using that exact resolved commit, even if docs/release artifacts were committed afterward. TRAIN_WALLTIME and TRAIN_QUEUE come from measured walltime in the ready release. The orchestrator's approval.json is required before this loop; at most six jobs are active.

```bash
SCIENCE_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["science_commit"])' "$R/s1a-release.json")"
TRAIN_WALLTIME="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["walltime_seconds"]; print(f"{s//3600:02d}:{s//60%60:02d}:{s%60:02d}")' "$R/s1a-release.json")"
TRAIN_QUEUE="$(python3 -c 'import json,sys; print("debug-g" if json.load(open(sys.argv[1]))["walltime_seconds"]<=1800 else "regular-g")' "$R/s1a-release.json")"
for arm in matrix_L1 soft_slot_L1; do
  for seed in 17 29 43; do
    scripts/submit.py --name "slot-$arm-s$seed" --queue "$TRAIN_QUEUE" --walltime "$TRAIN_WALLTIME" --commit "$SCIENCE_COMMIT" -- -m looplora.llm.slot_run --mode train --config "$R/s1a.json" --manifest "$R/manifest.lock.json" --release "$R/s1a-release.json" --arm "$arm" --seed "$seed" --out "$R/s1a/$arm/s$seed"
  done
done
```

After the six jobs finish and records are checked, run dev selection. It must show one shared release_blake2b_128. If the registered stable-dev trigger requests the one interface ablation, use the two seed-17 ablation hashes already in this same release, wait for both runs, then analyze them. Otherwise skip the ablation commands. Do not open confirmation yet.

```bash
scripts/submit.py --name slot-s1a-dev --queue debug-g --walltime 00:01:00 --commit "$SCIENCE_COMMIT" -- scripts/analyze_slot_study.py --stage S1A-DEV --manifest "$R/manifest.lock.json" --records "$R/s1a" --out "$R/s1a-selection"
for arm in matrix_L1 soft_slot_L1; do
  scripts/submit.py --name "slot-abl-$arm" --queue "$TRAIN_QUEUE" --walltime "$TRAIN_WALLTIME" --commit "$SCIENCE_COMMIT" -- -m looplora.llm.slot_run --mode train --config "$R/s1a.json" --manifest "$R/manifest.lock.json" --release "$R/s1a-release.json" --arm "$arm" --seed 17 --ablation --out "$R/s1a/ablation/$arm/s17"
done
scripts/submit.py --name slot-s1a-abl --queue debug-g --walltime 00:01:00 --commit "$SCIENCE_COMMIT" -- scripts/analyze_slot_study.py --stage S1A-ABL --manifest "$R/manifest.lock.json" --records "$R/s1a" --out "$R/s1a-ablation"
```

STOP: S1A selected after registered dev/ablation selection. The independent reviewer re-runs the frozen dev analysis. The orchestrator verifies the ready release's file digest against all six training records before writing the opening record at $OPEN_ROOT/reports/slot-loop-v1/s1a-open.json. It includes release_blake2b_128, the frozen script digest, registered prereg digest and selected arm/seed checkpoint map. Prepare $R/s1a-entries.json with the six selected exports and three registered references, each label once, with absolute export paths. Only after that opening authorization run sealed evaluation; unseen is optional, confirmation required.

```bash
for set_name in confirmation unseen; do
  scripts/submit.py --name "slot-eval-$set_name" --queue debug-g --walltime 00:05:00 --commit "$SCIENCE_COMMIT" -- -m looplora.llm.slot_run --mode evaluate --manifest "$R/manifest.lock.json" --release "$R/s1a-release.json" --set "$set_name" --entries "$R/s1a-entries.json" --open-root "$OPEN_ROOT" --out "$R/s1a-evaluation/$set_name"
done
scripts/submit.py --name slot-s1a-analysis --queue debug-g --walltime 00:01:00 --commit "$SCIENCE_COMMIT" -- scripts/analyze_slot_study.py --stage S1A --manifest "$R/manifest.lock.json" --open-record "$OPEN_ROOT/reports/slot-loop-v1/s1a-open.json" --release "$R/s1a-release.json" --records "$R/s1a-evaluation" --out "$R/s1a-summary"
```

Inspect all outputs and account node·h before each gate. Copy persistent summary/selection/release artifacts into the repository before each required stop. The optional common-fixed-LR comparison has no release entry and is not authorized by these commands.



### Review-round-1 validation ledger
Initial test snapshot: `b78245003f2f75afc6f15055fb7d7d0acaf9f9b6`; experiment retry snapshot: `ca2855a2b26b08d85908b65925f81600a00e7533`. All jobs use `scripts/submit.py --commit` (one node, debug-g), no `--wait`; the orchestrator watcher supplies the next wake-up. No polling or science execution.

| job | files | requested walltime | result | actual node·h |
| --- | --- | --- | --- | --- |
| 3728078.opbs | test_slot_data.py, test_analyze_slot_study.py, test_slot_release.py | 00:03:00 | 182 passed; exit 0 on mg0006, 84 s | 0.023333 |
| 3728079.opbs | test_slot_experiment.py | 00:05:00 | 47 passed, 1 failed; exit 1 on mg0007, 169 s; fixture metadata incomplete | 0.046944 |
| 3728107.opbs | test_slot_experiment.py, snapshot `ca2855a2b26b08d85908b65925f81600a00e7533` | 00:05:00 | 48 passed; exit 0 on mg0018, 167 s | 0.046389 |

Logs: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-r1-meta-20261029-205152-3029969.log` and `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-r1-exp-20261029-205152-3029979.log`. Previous measured validation predicts about 0.07–0.09 node·h combined; requested ceiling is 0.133 node·h, inside S0's 0.497 node·h remaining at handover. Final total added, including the failed attempt and retry: (84 + 169 + 167) / 3,600 = 0.116667 node·h. Cumulative S0 account: inherited 0.503 + 0.116667 = 0.619667 node·h; 0.380333 remains under the 1.0 cap. All changed tests now pass. Next: `STOP: review <full hash>` for cross-review. Preflight/lock/S0 remain on hold.


Failure diagnosis for 3728079: `test_what_the_runner_writes_for_s0_is_what_the_frozen_analysis_reads` reaches `analyze_slot_study.main`, which validates all five manifest sets before stage dispatch (lines 777–778). The tiny-study fixture only supplied its five ordinary runtime sets, omitting `S1A/confirmation` and `S1A/unseen`, so analysis refused the incomplete metadata before reading S0 results. Production lock metadata already contains all registered sets. Corrected only the fixture: merge the complete registry with its tiny ordinary-set overrides when building manifest counts/id digests; no sealed episode generation and no assertion changed. Production code and the passing metadata/analysis/release tests are unchanged. Static compilation and `git diff --check` passed. Repeated the whole changed experiment file on compute in 3728107 (previous measurement 169 s; request 300 s gives >1.5×): 48 passed.

Retry log: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-r1-exp2-20261029-205734-3080878.log`. Verified complete 100% pytest progress with no failures or skips, matching source snapshot and expected command, CUDA available on mg0018, and both log exit and exit marker equal 0. Production source, analysis and builder did not change after the 182-test metadata job; only this state file and the tiny-study fixture changed, so those passing results still apply. No existing non-slot imports changed. All requested runtime checks are complete; real-parent preflight and all science runs are deliberately deferred to cross-review approval.


## Review round 2 (wk-slot; complete, awaiting cross-review)
Master 9035de8 merged before changes. Findings and requirements: task section "Review round 2: fixes before the preflight", reviews `round2-rv-code.md` and `round2-rv-opus.md`. No preflight, lock or S0 is authorized during this round.
RED regression snapshot `29b171a8229be723744e901dcaa09201fc56f345` contains tests only, before production/doc fixes. Job 3728243.opbs (debug-g, one node, requested 00:03:00) runs `-m pytest -q tests/test_slot_round2.py tests/test_analyze_slot_study.py -k round2`. All expected failures were checked against their intended causes, as recorded below. Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-r2-red-20261029-212940-3263592.log`; 22 expected failures, exit 1 on mg0037, 21:29:44–21:30:18 (34 s = 0.009444 node·h). No polling; orchestrator watcher supplies completion. S0 cumulative before round 2: 0.619667 node·h; 0.380333 remaining.

Round-2 changes validated on compute:
1. `save_export` serializes the complete slot metadata into a JSON-shaped value before torch persistence; reload equality remains strict. New integration regression uses a one-layer synthetic backbone with the registered adapter dimensions, runs the real 20-step smoke per arm, reads smoke.json, invokes the release builder, reads its JSON candidate and invokes the reload CLI. Parent weights and all data remain synthetic/ordinary smoke; no real preflight is run.
2. `OpenToken` preserves the exact six selected arm/seed digests and the locked reference digest, plus the reference reads from the ready release. Sealed `evaluate_entries` requires exactly six main labels and the three reference labels, each once. Main digest/export metadata/read must match its label and released arm; references use only the locked parent digest and their specified reads. NPZ files add scalar Unicode `checkpoint_blake2b_128` and scalar Unicode `read` (JSON, null for Native). Confirmation analysis refuses mismatched provenance and retains validated identity/read in summary rows.
3. Sealed evaluation now requires `--release READY_JSON`. The token requires open-record analysis digest = ready-release digest = script file under the absolute opening root. Stage S1A requires `--open-record ABS_JSON --release ABS_READY_JSON` and checks its own script digest against both, also recording those inputs in source_index. Earlier analysis stages do not accept these S1A-only flags. The complete parser-checked S1A command is in the execution runbook above.
4. Runbook steps 1–4 above now resolve full reviewed/post-lock commit variables separately, use persistent absolute artifact paths, submit analysis without a second interpreter, and copy decision files into the worktree after completion.
No registered numerical rule changed. The optional common-fixed-LR run remains unregistered in the release; if later requested, it first needs a release entry. The D-F4 seal remains a workflow guard. All src edits in this round occur before the first real preflight, as required by source binding.

RED outcome verified: the real smoke→smoke.json→builder→candidate JSON→reload CLI path reaches the strict-config STOP (not a setup failure); five evaluation-entry cases return success when they should refuse; changed-script/open-record case does not raise; four runbook cases identify obsolete commands or relative paths; all eleven direct analysis provenance/digest cases report DID NOT RAISE. Correction from round 3: the behavior regressions reproduced the defects, but runbook step 1 failed because the test extracted the first backticked token (debug-g), not the submission command. That red case did not establish the intended command defect; its command-extraction/parser test is a deferred S1A runbook follow-up. Cumulative S0 account after RED: 0.629111 node·h; 0.370889 remaining. Next validation uses the fixed source at `11ff92a9fbe42cdd07d6ea23bc6f90cae581a8a8` (and any following docs-only commit). No runtime assertion has been weakened.


Round-2 GREEN validation snapshot: `f39b21259b8c3758d79c8f466290f98d83a0a701`, includes all fixes and RED evidence. Jobs submitted with scripts/submit.py, no waiting/polling:

| job | test files | walltime | result | actual node·h |
| --- | --- | --- | --- | --- |
| 3728256.opbs | test_slot_data.py, test_analyze_slot_study.py, test_slot_release.py | 00:03:00 | 209 passed, exit 0 on mg0003; 83 s | 0.023056 |
| 3728257.opbs | test_slot_experiment.py, test_slot_round2.py | 00:05:00 | 59 passed, exit 0 on mg0005; 177 s | 0.049167 |

Logs: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-r2-meta-20261029-213408-3287227.log` and `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-r2-exp-20261029-213408-3287235.log`. Expected runtime basis: earlier metadata job 84 s, experiment file 167 s, full RED selection with integration 34 s; requested 180/300 s provide about 1.5× or more. Both logs reached 100% with no failures/errors/skips, the expected full source commit and argv, CUDA available, and exit marker/log exit both 0. Actual GREEN cost: (83 + 177) / 3,600 = 0.072222 node·h; total round-2 cost including RED: 294 / 3,600 = 0.081667 node·h. Cumulative S0 account: inherited 0.503 + round-1 420 / 3,600 + round-2 294 / 3,600 = 0.701333 node·h; 0.298667 remains under the 1.0 cap. No non-slot module imports changed; the earlier legacy-suite evidence remains applicable.


Final round-2 validation: both arms passed the integration through 20 optimizer steps on synthetic parent weights, persisted smoke JSON, release-builder self-checks, a JSON candidate and the actual reload CLI. The provenance and digest regressions now refuse the invalid inputs accepted on RED. Strict comparisons and registered numerical rules remain. All affected tests ran on compute; static compilation and git diff --check passed. Only documentation changed after the tested snapshot. Next: STOP for cross-review of the diff against 2438599. Real-parent preflight, manifest lock and S0 remain intentionally unrun pending the cross-review result.


## Preflight and S0 execution (reviews closed)
User authorization: merge master 8632497, execute runbook steps 1–4, commit the lock locally when all required checks pass, then S0 and its decision. No additional review stop and no src edits. The two round-3 S1A follow-ups (analysis-only release-digest binding, and complete S1A runbook/parser checks) are deferred until before their registered S1A deadlines. The old prereg file remains byte-identical; its stage budget caps no longer bind (task Budget section). Node·h accounting continues, each job ≤16 node·h.
Preflight outputs are new absolute paths under `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/preflight`; the S0 directory does not yet exist. Gate: all nine required checks true, Native replay valid, Matrix maximum absolute logit difference exactly zero, and parent/config/source identities current. No calibration or dev episodes are generated until this gate passes and the lock is committed.


| job | stage | snapshot | requested resources | outcome | actual node·h |
| --- | --- | --- | --- | --- | --- |
| 3728339.opbs | real-parent preflight | `7febf58c3440483fe081e99c263f68ad62512b10` | debug-g, 1 node × 00:20:00 | all nine checks passed on mg0031, 53 s, exit 0 | 0.014722 |

Preflight log: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-preflight-20261029-215406-3324539.log`. `REVIEW_COMMIT=7febf58c3440483fe081e99c263f68ad62512b10`. Runbook step 1 submitted exactly with resolved full commit and absolute config/output paths. Prior cumulative runtime: 0.701333 node·h. Next: inspect job log, preflight.json and identity.json; on complete success, run the metadata lock command, commit the lock/inventory/preflight on wp/slot, then submit S0 on that lock commit. Any failed required check stops before calibration/dev.


Preflight accepted: job 3728339 ran 21:54:33–21:55:26 JST on mg0031, 53 / 3,600 = 0.014722 node·h; cumulative measured 0.716056 node·h (stage caps superseded). Native S-RE replay is bitwise equal to archived per-sample results: 723/1,000, NLL 1.642127275466919. D128-3 replay is also bitwise: 608/1,000, NLL 2.1291537284851074. Matrix equals Native on all 1,000 old episodes, 64 S0 smoke episodes at L=1/2/4 and the 64 four-question train-shaped smoke episodes, with zero unequal logits and max absolute difference 0.0. The four reader gradients are bitwise equal with max differences 0.0. All owner/mask, isolation, feature, frozen-set, bytes and capacity checks passed. Slots: 57,368 B/episode; matrix: 393,216 B/episode. Source blake2b-128: `fe548da4c06d77b7f4a8ae65349650f9`.
Lock command completed successfully with the shared venv interpreter: re-derived registry/disjointness, verified stage configs, registered prereg digest, actual parent file inventory and source digest against the preflight. It generated `reports/slot-loop-v1/manifest.lock.json` and `checkpoint_inventory.json`; exact preflight and identity reports are copied to `reports/slot-loop-v1/preflight/`. These are committed locally on wp/slot before any S0 calibration or dev execution, per the user authorization. No src changes.


`LOCK_COMMIT=2f22aba2f6e93029198d1126ede2b9528cd3eef4` contains the lock, full parent inventory and exact preflight/identity reports. S0 job 3728354.opbs is pinned to that commit (debug-g, one node, requested 00:25:00; 0.416667 node·h requested, below the user cap). It reads the committed lock at the absolute worktree path and writes persistent outputs under `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s0`. Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-s0-20261029-215755-3330208.log`. Completed exit 0 on mg0046, 22:13:17–22:16:55 JST: 218 / 3,600 = 0.060556 node·h; cumulative measured 0.776611 node·h. Next: after the watcher, inspect all S0 metrics/candidates/samples and provenance, then submit the frozen analysis on the lock commit with persistent `$R/s0/decision`, copy decision artifacts into reports, update the report/state and STOP: S0 done. No S1A work is started.


S0 outputs inspected after job completion: correct lock commit and stage config hash, 2,048 calibration and 1,000 dev ids/counts matching the lock, all 18 candidates, 13 dev conditions and four D128-3 rows present, finite aggregate metrics, and D128-3 old replay passed. Calibration selects Matrix g=1, SoftSlot g=1/tau=0.1, HardSlot g=1. No manual selection or threshold change.
Frozen analysis job 3728547.opbs (debug-g, one node, requested 00:03:00) is pinned to lock commit `2f22aba2f6e93029198d1126ede2b9528cd3eef4`; Python argv starts `scripts/analyze_slot_study.py` with no extra interpreter. It validates per-sample Native/Matrix identity, sample ids and registered calibration choice, computes the registered descriptive bootstrap, and writes persistent decision files at `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s0/decision`. Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-s0-analysis-20261029-221922-3381749.log`. Outcome/node·h pending. Next: inspect decision and source index, archive raw S0/decision artifacts, write the one-page S0 report, update costs, commit and STOP: S0 done. S1A remains unstarted.



## S0 complete: R2
Frozen analysis job 3728547.opbs completed on mg0022, 22:19:28–22:19:34 JST (6 s = 0.001667 node·h), lock commit `2f22aba2f6e93029198d1126ede2b9528cd3eef4`, exit/log marker 0. Output: R2; all nine gates true; 1,000 dev and 2,048 calibration episodes match the lock. Native/Matrix per-sample correctness/NLL identity and registered calibration choices passed before the decision and 10,000 paired descriptive bootstrap resamples.
- Main table, all valid question tokens: Matrix 720/1,000 (NLL 1.577648, read error 0.183978); SoftSlot 745/1,000 (1.505075, 0.092405), +2.5 pp, descriptive CI [0.6, 4.4]; HardSlot 781/1,000 (2.434278, 0.137009), +6.1 pp [3.8, 8.4]. Soft is below the exploratory 5 pp threshold; Hard meets it with lower read error, so the first matching row is R2.
- S1A choices: full adaptation, Matrix-L1 g=1 and SoftSlot-L1 g=1/tau=0.1, all valid question tokens. HardSlot stays diagnostic. Calibration by NLL supplies these fixed settings. No S1A work has started.
- Final-only table stays separate: Matrix 72.0%, SoftSlot 75.5%, HardSlot 80.3%, Clean-final oracle 92.6%. D128-3 at S-RE's settings: Matrix 63.3%, SoftSlot 73.7%, HardSlot 74.8%, same direction. Recency SoftSlot 74.6% is descriptive.
- All source-index input digests, the script digest and three output digests printed by the job matched. Raw metrics/calibration and both sample files, decision JSON/Markdown and source_index were copied byte-for-byte into `reports/slot-loop-v1/s0/`. `artifacts.json` records all seven file sizes and blake2b-128 digests. Report: `docs/reports/SLOT-LOOP-v1.md`.

| job | phase | actual seconds | actual node·h | outcome |
| --- | --- | ---: | ---: | --- |
| 3728339.opbs | preflight | 53 | 0.014722 | all checks passed, Matrix bitwise Native |
| 3728354.opbs | S0 | 218 | 0.060556 | complete calibration/dev/recheck |
| 3728547.opbs | analysis | 6 | 0.001667 | R2, artifacts verified |

Execution total 277 / 3,600 = 0.076944 node·h. Cumulative including tests/retries: inherited 0.503 + (420 + 294 + 277) / 3,600 = 0.778278 node·h. Written caps no longer bind; every job is below 16 node·h. No src changes or sealed S1A data access. Next: STOP: S0 done and await the orchestrator. Before S1A, complete the two Review round 3 analysis/runbook follow-ups without src changes or another review round.


## S1A follow-ups and release preparation
S0 accepted by the orchestrator. Step 3 authorized through the release stop; no new review round. RED snapshot e6c98b69fbc6a813afdafac51536397a45e7905f, job 3728718.opbs (one node, debug-g, requested 00:01:00), runs the rebuilt-release, mismatched-record-release and complete-runbook-parser regressions before fixes. Log: /work/xg24i002/x10041/LoopLora/logs/pbs/slot-s1a-red-20261029-222707-3411194.log. Verified 3 intended RED failures: mismatched six-record release digests and rebuilt release both DID NOT RAISE; old reload submission rejected by actual submit parser for missing --name/--walltime. Exit 1 on mg0040, 22:27:14–22:27:32 (18 s = 0.005000 node·h). Production changes are analysis-only: shared training-record release digest and exact ready-release-file digest binding. The open-record token still checks script=release=file, while the orchestrator checks its release digest against all six records before opening, per task Review round 3. All S1A submission commands now include complete submit/target argv and persistent paths. The old round-2 step-1 RED was a command-extraction artifact; the improved extraction test selects the submission itself.

S1A preparation accounting starts at 0.005000 node·h for the RED tests; whole-study cumulative is 0.783278 node·h. GREEN validates the full analysis suite, complete runbook/target parser checks, updated extraction test and experiment record parser integration before any real S1A smoke. No src change; original preflight remains valid.


S1A follow-up GREEN snapshot: `18995b97dd685f3c306436ae11da40346324a947`.

| job | tests | requested | result | node·h |
| --- | --- | --- | --- | --- |
| 3728741.opbs | analysis, round2 integration/extraction, complete runbook parser | 1 node, debug-g, 00:02:00 | 158 passed on mg0019, 37 s, exit 0 | 0.010278 |
| 3728742.opbs | experiment suite including direct run-record parser | 1 node, debug-g, 00:05:00 | 48 passed on mg0020, 170 s, exit 0 | 0.047222 |

Logs: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-s1a-tests-20261029-223049-3437042.log`, `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-s1a-exp-20261029-223049-3437054.log`. Next: inspect both logs; if successful, stage unchanged manifest/config plus frozen analysis in persistent paths and submit the real 20-step-per-arm S1A smoke. Wait through the orchestrator watcher; no polling.

S1A follow-ups passed on compute: both GREEN jobs reached 100% without failures/skips; expected full commit/argv and exit markers verified. All 206 cases pass, including submit and target parser checks for every executable runbook submission and the direct parser consumer. No src change. Test cost GREEN 207 / 3,600 = 0.057500 node·h; S1A preparation including RED 0.062500; whole-study cumulative 0.840778. Staged exact lock/config copies at $R/manifest.lock.json and $R/s1a.json, and prereg/frozen analysis under $R/open-root. Analysis blake2b-128: `843d2ff4d1a03224ccd16e68720ba344`. This script is now frozen for the release and later confirmation analysis. No opening record is present or written. Real smoke is next.
