# WP-L13 state: SLOT-LOOP-v1 (S0, S1A), step 2: review round 1 fixes

Branch `wp/slot` (worktree `LoopLora-wt/slot`), master `c0bbbd5` merged after handover at `ccdcc96`; not merged into master, not pushed. Task: `docs/tasks/WP-L13_slot_loop.md`; registration `docs/prereg/SLOT-LOOP-v1.md` (blake2b-128 `56e8c611c01e0ef9b7ae32db97b0e334`, registered on master at `5f04a5c`, unchanged); results will go to `docs/reports/SLOT-LOOP-v1.md`.
**Status: wk-slot has implemented review-round-1 fixes 1–10; updated compute validation pending.** Earlier compute results below cover the pre-review implementation. No real preflight, manifest lock, S0 or S1A science run has started. After validation, STOP for the orchestrator’s cross-review before those steps.

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
All slot tests ran at `fd8c914` on `debug-g` (the reviewed commit differs from it only in this file); dots are `pytest -q` progress characters, none of the runs had a failure, an error or a skip. The tiny model has two layers (hidden 16) with the registered rank-128 DPFP/ridge adapter at the last layer, T1 with E = 16, so the real tokenizer, task generator and adapter code run.
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
5. **Sealed sets.** The confirmation and unseen-name sets are defined by their own base seeds in `slot_data.py` alone (a source scan test requires that no other file under `src/`, `scripts/`, `tests/`, `configs/` contains them); `slot_data.eval_set` refuses them without the `OpenToken` that `load_open_token` builds from the orchestrator's `reports/slot-loop-v1/s1a-open.json` (registered prereg digest, the frozen analysis script's digest, six selected checkpoints). `slot_run.evaluate_entries` evaluates a sealed set only for a checkpoint digest the token names and prints counts and digests, never metrics. The old `CONFIRMATION_WINDOWS` are pinned by a test and not extended.
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
- Complete changed-file compute tests, then STOP for two-reviewer cross-review. Do not begin preflight, lock or S0 before the orchestrator sends the result.
- After the review, in this order (R = `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1`, C = the full 40-hex commit of the reviewed tree):
  1. Preflight, one `debug-g` job (about 0.05 node·h): `scripts/submit.py --name slot-preflight --queue debug-g --walltime 00:20:00 --commit C -- -m looplora.llm.slot_run --mode preflight --config configs/slot_loop_v1/s0.json --out R/preflight`. It reads no calibration or dev data. A failed required check, or a Matrix-Native difference (the report carries the maximum absolute logit difference), ends in `STOP` before anything else runs.
  2. Lock, login node: `/work/xg24i002/x10041/LoopLora/.venv/bin/python scripts/prepare_slot_manifest.py --configs configs/slot_loop_v1 --preflight-report R/preflight/preflight.json --out reports/slot-loop-v1/manifest.lock.json --inventory-out reports/slot-loop-v1/checkpoint_inventory.json`; commit the lock, the inventory and the preflight report on `wp/slot`.
  3. S0, one job on the lock commit (estimated 0.15 node·h): `... -m looplora.llm.slot_run --mode s0 --config configs/slot_loop_v1/s0.json --manifest reports/slot-loop-v1/manifest.lock.json --out R/s0` (18 calibration candidates on 2,048 episodes, 13 dev conditions, the D128-3 recheck).
  4. Decision, compute job (numpy only): `python3 scripts/analyze_slot_study.py --stage S0 --records R/s0 --out reports/slot-loop-v1/s0 --preflight R/preflight/preflight.json --manifest reports/slot-loop-v1/manifest.lock.json`; submit this Python argv through `scripts/submit.py --commit C` (the wrapper supplies the interpreter); then `decision_s0.md`, `docs/reports/SLOT-LOOP-v1.md`, this file; `STOP: S0 done <hash>` with the decision row and the S1A choices it implies.
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
  "selected_checkpoints": {
    "matrix_L1/17": "<digest>", "matrix_L1/29": "<digest>", "matrix_L1/43": "<digest>",
    "soft_slot_L1/17": "<digest>", "soft_slot_L1/29": "<digest>", "soft_slot_L1/43": "<digest>"
  }
}
```
`load_open_token(ABS_ROOT, lock["inventory"])` also admits the S-RE `writer.pt` digest from the already checked locked inventory; no seventh digest is needed in the open record. Saved-run evaluation entries are `{label, export: ABS_PATH, checkpoint_blake2b_128}`. Reference entries are `{label, parent_writer_blake2b_128, read}`: `s_re_native` uses `read: null`; `matrix_step0` and `soft_step0` use the calibrated arm read configs from the release, all with the locked S-RE writer digest. Each evaluated set requires all six main labels (`<arm>_s<seed>`) and these three references.

### S1A release and reload runbook (after S0 and orchestrator approval)
Use `/work/xg24i002/x10041/LoopLora/.venv/bin/python scripts/prepare_slot_release.py --help` for the metadata builder. It writes a candidate release (`pending_reload`) from the locked manifest, S0 metrics, pilot, smoke and full science commit; repeat with both arm reload reports to obtain `ready`. Candidate preparation is reversible metadata work. Never train from `pending_reload`.
The release carries `science_commit`, `prereg_blake2b_128`, `analysis_script_blake2b_128`, manifest/inventory digests, `steps`, `walltime_seconds`, `parent.writer_blake2b_128`, per-arm `arms.<arm>.{read,lr}`, six main config hashes `<arm>/<seed>` plus two `<arm>/17/ablation` hashes, per-arm freeze and reload checks, and `smoke_configs` (the exact smoke-export slot metadata). Builder checks registered calibration/LR/timing choices and consistency before writing.
Smoke reports include each selected export's `export_config`. For each arm run reload-check through `scripts/submit.py --commit C -- -m looplora.llm.slot_run --mode reload-check --manifest LOCK --release CANDIDATE --export R/smoke/ARM/selected.pt --predictions R/smoke/ARM/smoke_predictions.npz --out R/reload-ARM.json`. Reload must reproduce episode ids/predictions/correctness/NLL and match the candidate's exact slot config and calibrated read; its report records the export digest. Final builder validation consumes both reports, then STOP for `approval.json` as before. These are future commands; this round executes none of them.


### Review-round-1 validation ledger (pending)
Test snapshot: `b78245003f2f75afc6f15055fb7d7d0acaf9f9b6`. Both jobs use `scripts/submit.py --commit` (one node, debug-g), no `--wait`; the orchestrator watcher supplies the next wake-up. No polling or science execution.

| job | files | requested walltime | result | actual node·h |
| --- | --- | --- | --- | --- |
| 3728078.opbs | test_slot_data.py, test_analyze_slot_study.py, test_slot_release.py | 00:03:00 | submitted; pending logs | pending |
| 3728079.opbs | test_slot_experiment.py | 00:05:00 | submitted; pending logs | pending |

Logs: `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-r1-meta-20261029-205152-3029969.log` and `/work/xg24i002/x10041/LoopLora/logs/pbs/slot-r1-exp-20261029-205152-3029979.log`. Previous measured validation predicts about 0.07–0.09 node·h combined; requested ceiling is 0.133 node·h, inside S0's 0.497 node·h remaining at handover. Update actual elapsed node·h from the job logs on completion; historical cumulative remains 0.503 until then. Next: read logs and outputs, resolve any failures, update this ledger, then `STOP: review <full hash>` for cross-review. Preflight/lock/S0 remain on hold.
