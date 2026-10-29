# WP-L13 state: SLOT-LOOP-v1 (S0, S1A), step 2: implementation, tests, awaiting review

Branch `wp/slot` (worktree `LoopLora-wt/slot`), based on master `177a4fe`; not merged into master, not pushed. Task: `docs/tasks/WP-L13_slot_loop.md`; registration `docs/prereg/SLOT-LOOP-v1.md` (blake2b-128 `56e8c611c01e0ef9b7ae32db97b0e334`, registered on master at `5f04a5c`, unchanged); results will go to `docs/reports/SLOT-LOOP-v1.md`.
**Status: code and tests done; no preflight, no S0 data, no S1A data touched.** The next action is the orchestrator's static review (`STOP: review <hash>`); then the preflight, the manifest lock and S0.

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
4. **The existing suite** (all test files except the three slot ones) was run on compute in two jobs, results below.

## MemoryBackend interface (stable for S2 and S3 workers)
`MemoryState = MatrixState | SlotState`. `MatrixState.matrix` is `[B, dv, dk]`; `SlotState` is `keys [B, S, dk]`, `values [B, S, dv]`, `valid [B, S]` (a prefix of True), `count [B]`. `build_slots(keys, values, capacity, valid=None)` pads with zeros to S slots and raises `CapacityError` when a row has more facts than slots (nothing is truncated).
`MemoryBackend.read(state, query [N, dk], owner [N]) -> [N, dv]` reads only the memory of episode row `owner[n]`; backends hold settings (`gain`, `tau`, `decay`), never memory. `MatrixBackend`: `g M q`, the original operation (matmul). `SoftSlotBackend(tau, gain, decay=None)`: `g sum softmax(k.q/tau [+ log lam^age]) v` over valid slots. `HardSlotBackend(gain)`: `g v_j`, `j = argmax k.q`, ties to the lowest index. An empty row reads zero without NaN. `online_write` refuses (there is no silent skip). Everything is functional: no function writes into a state it was given.
`MemoryRead(backend, state, owner)` is the handle the adapter receives instead of a tensor state: `bind`, `expand(Q)` (repeats owners only), `select(rows)`, `read(query)`, `clone`, `check_fp32`. Bytes of one episode: `persistent_bytes(state)` = `numel * element_size` over all state tensors (S = 16 slots: 57,368; matrix: 393,216).
In the adapter, `SlotFixAdapter.process_sequence` and `_masked_view_branch` do the reading; a tensor state falls through to the inherited code. `ReadSpec(kind, gain, tau, scope, recency, capacity)` plus `bind_batch(experiment, batch, spec)` is the entry point for experiments; `slot_read.final_read` returns the read `m` and its statistics at each question's final token.

## Tests (compute)
Pending: see "Job ledger".

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
