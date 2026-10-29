# SLOT-LOOP-v1 model, training, preflight and runner: pre-run static review (branch wp/slot, head ccdcc96)

Reviewer: read-only subagent. Date: 2026-10-29. Nothing was executed on a GPU and no torch/pytest was run (login node).
Scope: `git diff c6af756..ccdcc96` for `src/looplora/core/memory_backend.py`, `llm/slot_read.py`, `llm/slot_experiment.py`,
`llm/slot_train.py`, `llm/slot_preflight.py`, `llm/slot_run.py`, `llm/slot_rules.py` (the rules the runner applies online), the four
changed existing files (`core/wrapper.py`, `llm/batching.py`, `llm/inject.py`, `llm/protocols.py`), and `tests/slot_golden.py`,
`tests/assets/slot_golden_digests.json`, `tests/test_slot_backend.py`, `tests/test_slot_experiment.py`. Authority: `docs/prereg/SLOT-LOOP-v1.md`
including "Decisions at registration" 1-9; the plan (sections 6, 7, 11.2, 12) was used as reference only. Data sealing, seeds, configs,
`scripts/prepare_slot_manifest.py` and `scripts/analyze_slot_study.py` belong to the second reviewer; I read them only where the runner calls them.
Checks that were run: full reads of every file above and of the old code they call (`core/adapters.py`, `core/memory.py`,
`llm/writer_fix.py`, `sq2.py`, `dpfp_run.py`, `read_fix.py`, `read_fix_run.py` (`validate`, `fit`), `train.py`, `prefetch.py`, `release.py`,
`sym/run.py`, `scripts/pbs/job.pbs`, `scripts/submit.py`), the complete diff of the four existing files, `py_compile` of every file under
review (bytecode redirected to the scratchpad), a stdlib `ast` scan of the new files (no undefined name, no unused import), a grep of every
`hashlib` use in the diff, a read of the old `val-6000` / `val-22370` `samples.npz` with stdlib `zipfile` (dtypes and values the preflight
compares against), and reads of the installed torch 2.14.0+cu132 sources where a determinism rule mattered. The login node has Python
3.9.25 and the project requires >= 3.13, so nothing was imported. No repo or worktree file, branch or commit was touched; this file is the
only thing written, and it is not committed.

## Verdict

No defect of consequence in the model, the read operators, the training loop, the preflight or the runner. I found no input or state
that makes the new code change the old path, return a wrong read, let a frozen tensor move, apply a registered rule other than as
written (the one exception is F1, how the LR pilot treats a diverging candidate), or touch sealed data before the open token. The
old path is unchanged by construction (12 added and 4 replaced lines in four files, every one a new-type branch or a type hint) and by the golden test;
Matrix (g = 1) is the old read operation for operation; and the chain of decision 5 (preflight -> exit 2 -> no lock -> no science mode) is
closed. Four low-severity items, none of which can bias a result (each fails loudly or at the orchestrator's desk): F1 the pilot aborts on a
non-finite candidate instead of excluding it, F2 the lock is not bound to the code that passed the preflight, F3 two later-stage inputs whose
schemas exist only in code, F4 the registered N rule and the registered walltime cap contradict each other for a load time above about 60 s
(recorded loads are about 1.2 s). The golden test is strong for "the old path is unchanged" and deliberately tiny for everything else (see
Observations); the real-size evidence is the preflight, which is adequate.

## Findings, ranked

### F1 (low; certain on the code, unlikely to trigger) A diverging LR-pilot candidate aborts the whole pilot; the prereg says it is excluded
- Evidence: `docs/prereg/SLOT-LOOP-v1.md:108` ("Non-finite candidates are excluded; if every candidate of an arm is worse than its step 0,
  STOP"). The exclusion exists only after the fact: `slot_rules.py:58` filters `math.isfinite(r["nll"])`. A candidate that goes non-finite
  during training never produces a row: `slot_train.py:85-86` raises `FloatingPointError` on a non-finite loss and `slot_train.py:90`
  (`clip_grad_norm_(..., error_if_nonfinite=True)`) raises `RuntimeError` on a non-finite gradient, and `slot_run.py:350-354` is
  `try: fit(...) finally: experiment.close()` with no `except`, so the exception leaves `run_pilot` before `pilot.json` (`:378`) exists.
  `slot_run.py:44` and `slot_train.py:146` write with `allow_nan=False`, so a finished candidate whose dev NLL is NaN fails the same way,
  at the very end. The tests cover `fit` alone (`tests/test_slot_experiment.py:403-408`) and `choose_lr` on finite rows.
- Failure: if one of the six pilot runs diverges (the likeliest is 3e-3 on the SoftSlot arm; 3e-3 is S-RE's own training LR, and the
  warm-up and the clip make it unlikely), the pilot job ends with a traceback: no `pilot.json`, no S / Ev / Ld, no LR choice for either arm.
  The finished runs survive only as per-run `training.json`; a rerun needs a new `--out` (`mkdir(exist_ok=False)`, `:333`); the registered
  behaviour (drop the candidate, choose among the rest, still derive N) is never reached. Cost: one wasted pilot job and an orchestrator round
  trip. No bias: the failure is loud.
- Fix: catch `(FloatingPointError, RuntimeError)` around `fit` in `run_pilot` only, record `dict(arm, lr, nll=None, all_finite=False,
  error=str(error))`, make `choose_lr` skip `None`, take S and Ev from the finite runs and say so in the record, and add a test that forces
  one candidate to NaN. Or write into the prereg decisions that a diverging pilot candidate is a STOP.

### F2 (low; certain) The lock is not bound to the code that passed the preflight
- Evidence: `scripts/prepare_slot_manifest.py:72-73` stores the preflight's `git_commit` in the lock. `slot_run.py:62-74` (`require_lock`)
  checks status, prereg digest, `preflight.passed`, the registry and the parents' inventory, never the commit. Only `run_train` binds the
  code (`slot_run.py:394`: the release's `science_commit` must equal the running commit). S0, smoke, pilot and evaluate
  (`slot_run.py:165,288,335,511`) run under any commit.
- Failure: decision 5 and the whole section-12 gate are evidence about one commit. A later commit that touches the read path, the adapter, the
  batch code or the numerics (for example a fix made after this review) would run S0, the smoke and the pilot under a lock whose
  bitwise-identity evidence was produced by different code, and nothing in the lock, the runner or the S0 metrics (`provenance()` records
  whatever commit ran) would object. The identity property is exactly what a small edit breaks.
- Fix: in `require_lock` compare `lock["preflight"]["git_commit"]` with `provenance()["git_commit"]` (under `submit.py` this is the
  snapshot's `LL_SPEC` commit), or have the orchestrator re-run the preflight after any `src/` change and say so in the state file. If
  documentation-only commits must stay allowed, compare a blake2b over `src/looplora/**/*.py` instead of the commit.

### F3 (low; certain) Two inputs of later stages exist only in the code: the open record's `reference_checkpoint` and the release keys
- Evidence (a): the prereg says the open record "names the six selected-checkpoint digests and the blake2b-128 of the frozen analysis
  script" (`docs/prereg/SLOT-LOOP-v1.md:57`) and lists the unmodified S-RE as a sealed descriptive reference (`:115`).
  `slot_data.py:232-234` additionally reads an optional `reference_checkpoint` and admits it to the token; `slot_run.py:450-451,461`
  evaluates the S-RE entry under its `parent_writer_blake2b_128`. No document names the key (`grep reference_checkpoint` finds only
  `slot_data.py:232`).
- Evidence (b): `run_train` requires `science_commit`, `prereg_blake2b_128`, `steps`, `parent.writer_blake2b_128`, `arms.<arm>.{read,lr}`,
  `config_hashes["<arm>/<seed>[/ablation]"]` and optionally `train_time_limit` (`slot_run.py:386-404,420`); the prereg lists the release
  contents in prose only (`:112-113`). The schema is defined by `run_train` and the test helper `release_for`
  (`tests/test_slot_experiment.py:564`). No code builds the release, and `reload_check` (`slot_run.py:315-324`) implements two of the three
  registered reload checks (reload, reproduced predictions; not "its config equals the one the release implies", prereg `:113`).
- Failure: an open record written to the prereg's description makes the S-RE reference evaluation raise `SealedSetError` when that job
  starts, after all six runs are done (it fails closed); a release missing a key is a `KeyError` at the first arm-seed job's start. No result
  is affected; each costs a job slot and an orchestrator round trip.
- Fix: put both schemas, with an example, in the state file or a prereg addendum (state that the open record carries `reference_checkpoint`
  = the S-RE `writer.pt` digest), and have the release builder, which is not written yet, emit and self-check them, including the missing
  config-equality step.

### F4 (low; certain arithmetic, trigger about 50x above any recorded load time) The registered N rule and the registered walltime cap can contradict each other
- Evidence: prereg `:109-110`. `slot_rules.py:70-71` takes N from `(3600 - max(90, Ld)) / 1.5 - 11 Ev`, but `slot_rules.py:79-81` computes the
  walltime from `Ld` itself and raises `StopForOrchestrator` above 3600; `slot_run.py:371-376` turns that into the pilot's STOP. For the largest
  admissible N the raw walltime is at most `1.5 Ld + 3600 - max(90, Ld)`: `3510 + 1.5 Ld` for Ld <= 90 s (above 3600 once Ld > 60 s, less the
  up to 15 S seconds that rounding N down to a multiple of ten leaves), `3600 + 0.5 Ld` above 90 s.
- Failure: if the pilot's first load (backbone plus adapter, `slot_run.py:347-348`) takes more than about 60-75 s, the pilot ends in a STOP
  with `steps` recorded and no `walltime_seconds` although every registered formula was applied; the orchestrator must pick a smaller N or
  a 2 h request. Recorded backbone loads are 1.1-1.3 s (`reports/p1l-rfx/summary.json:2`, `reports/llm-prefetch/worker-1-0.json:2239`), so
  this is a latent trap, not an expected event.
- Fix: none needed before the run if the orchestrator accepts that STOP; otherwise add to the prereg decisions that a walltime above 3600 s
  lowers N to the largest multiple of ten that fits.

### Observations (not defects)
- Strength of the golden test (`tests/slot_golden.py`, `tests/assets/slot_golden_digests.json`, `tests/test_slot_experiment.py:101-111`).
  What it proves: the digests of the old classes (`sq2.SQ2Experiment`, unbound batches) on a 2-layer hidden-16 Qwen3 are the ones captured at
  c6af756 (job 3727254, torch 2.14.0+cu132, GH200; the note records two equal fresh processes at the old commit and equality with the current tree
  at the capture commit): written state, keys and values; logits at L = 1, 2, 4 for a training-shaped (four questions) and an evaluation-shaped
  (one question) batch; the captured inputs; and the gradients of Q, R, U, B, A, K, V, a_lambda. The test re-runs the generator in a fresh
  process and compares the whole key set (`set(now) == set(recorded)`), so a missing digest fails. That is a real regression gate for the four
  edited files and for every old module the 16 diff lines could affect (`repeat_snapshot` tensor branch, `AdapterBank.use` tensor branch). What it
  does not prove: the fixture puts the adapter at the last layer (`tests/test_slot_experiment.py:4-6`), so only the final question token's read
  reaches the logits, it uses ridge eta 1e-2 instead of 1e-4, and it never builds a `SlotFixAdapter`. The new Matrix path at real size
  (earlier-token reads propagating through layers 15-27, eta 1e-4, the real buffers) is established only at run time by the preflight
  (`native_replay` per-sample bitwise against the recorded `samples.npz`; `matrix_identity` `torch.equal` on bf16 logits over three batch
  families), and decision 5 makes any failure a STOP. That division of labour is sound; the golden test alone would not be enough and is not
  presented as such.
- `isolation_check` (`slot_preflight.py:156-184`) is satisfied by construction: `AdapterBank.use` clones the state (`inject.py:116`), so a read
  could not change the digests of the batch's originals even if it wrote in place. Read-only holds because every backend function is out of place
  (`memory_backend.py:155-272`) and `online_write` raises; the check that would catch a cross-episode leak is `owner_mask_check` (other episodes
  bitwise untouched, `slot_preflight.py:146`). Not misleading, only weaker than its name.
- Rule readings that matter only in a tie: "the `g` nearest 1" is the absolute distance (`slot_rules.py:52`; 0.5 beats 2.0, in log distance they
  would tie and the larger tau would decide, `slot_rules.py:149` states the reading); the checkpoint tie is incremental against the best so far
  (`slot_train.py:110`, `slot_rules.py:90`), the reading of "improves the best so far strictly" (prereg `:111`); the calibration and LR windows
  are inclusive (`<=`). Differences are at the 1e-6 to 1e-4 nat level, far below the spread between checkpoints.
- `slot_run.py:5` says every mode but `preflight` needs the lock; `reload-check` (`:514-517`) does not. Harmless: it evaluates the smoke set only.
- `slot_rules.schedule_steps(max_steps=6000)` (`slot_rules.py:65`) repeats `slot_data.MAX_STEPS` (`slot_data.py:29`); `check_disjoint` is
  parameterized by `MAX_STEPS`. Equal today; raising one alone would silently void the disjointness guarantee for the larger N.
- The `--out` guard (`slot_run.py:494`) is lexical, so a path with `..` that starts under `RUNS` passes; no realistic trigger.
- `fit` runs the support encoding (`prepare`) under `torch.no_grad()` (`slot_train.py:81-82`) while the old loop did not; with the writer frozen
  this changes nothing numerically and is the reason check (iii) can be a digest.

## Verified OK (with evidence)

Old path unchanged, Matrix (g = 1) equals Native, decision 5
- The diff of existing source is 4 files, 12 added and 4 replaced lines (`git diff --numstat`): `wrapper.py:11,23` (import, type hint), `batching.py:9,106-107` (`repeat_snapshot`: a `MemoryRead` is
  `expand`ed, a tensor takes the unchanged `repeat_interleave(queries, dim=0)`), `inject.py:13,111-114` (`MemoryRead` -> `check_fp32()`, a tensor
  keeps the same `dtype != float32` error; `state.clone()` at `:116` untouched), `protocols.py:16,21` (import, type hint). `core/adapters.py`,
  `core/memory.py`, `writer_fix.py`, `sq2.py`, `pyproject.toml`, `uv.lock`, `job.pbs`, `submit.py` are not in the diff (`git diff --stat`).
  `memory_backend.py` imports only torch and the stdlib, so the new imports add no cycle. In `read_only` mode the wrappers hand `context.state` to
  `process_sequence` unchanged (`wrapper.py:79-81`, `residual.py:55-56`); their only direct tensor operations on the state are in `write_only` mode
  (`wrapper.py:68-74`, `residual.py:45-50`), which the slot path never uses; and the `MemoryRead` takes the `isinstance` branch of `process_sequence`
  (`slot_read.py:107`).
- Operation-for-operation: new `MatrixBackend.read` (`memory_backend.py:193-197`: `index_select(0, owner)`, `matmul(rows, query.to(rows.dtype).unsqueeze(-1))
  .squeeze(-1)`, `_scale` returns the readout itself at g == 1.0, `:183-184`) then `F.linear(m.to(u.dtype), U)` (`slot_read.py:85`) against the old
  `core/adapters.py:228-231` (`matmul(M, q.to(M.dtype).unsqueeze(-1)).squeeze(-1)`, `F.linear(m.to(u.dtype), U)`). The token loop
  (`slot_read.py:95-125`) mirrors `_masked_branch` (`core/adapters.py:81-94`): same `valid.reshape(-1).nonzero`, `index_select` of the valid rows,
  `read`, `branch`, `index_copy`, zero rows at invalid tokens, `u` from `torch.where(valid_mask[..., None], x, 0)` as in `_prepare` (`:78`). The old
  state rows are `repeat_interleave`d then `index_select`ed; the new ones are the owner map; both give the same contiguous `[N, dv, dk]` in the same row
  order, so the same batched matmul.
- A tensor state falls through to the inherited method in all three overrides (`slot_read.py:72-73,82-83,107-108`), so the unbound Native read of a
  `SlotFixAdapter` is the parent's own code; `attach_slot_reader` accepts only a plain `FixAdapter` and copies weights and buffers strictly (`:155-164`,
  `:145`); the unbound path is what `native_replay` runs on the real parents.
- Decision 5 is closed end to end: `slot_preflight.matrix_identity` (`:68-84`, `torch.equal` on bf16 logits of `experiment.score(batch)` against
  `experiment.score(bound)` on the same prepared batch) is run on the old range, the smoke evaluation set at L = 1, 2, 4 and a training-shaped
  batch (`slot_run.py:214-218`) -> `passed` -> `main` returns 2 (`:500`) -> `prepare_slot_manifest.py:65-68` refuses the lock -> `require_lock`
  (`slot_run.py:66-69`) refuses S0, smoke, pilot, train and evaluate. `native_replay` (`slot_preflight.py:47-64`) compares per-sample
  `episode_id, prediction, correct, nll, rel_err` (and `key_top1`, `key_margin` when the old file has them) bitwise and the counts / NLLs exactly.
- Preflight identity numbers: `RECORDED` (`slot_preflight.py:31-35`) S-RE 723/1000, `loss_float32` 1.642127275466919; D128-3 608/1000,
  2.1291537284851074 (relative read errors 0.19612866640090942 and 0.36268818378448486); `old_range` (`:40-43`) is `validate`'s episodes (val [0, 1000),
  seed 4242, one question each); the NLL means are the float32 `np.mean` of the float32 per-sample array exactly as in `read_fix_run.validate`
  (`slot_experiment.summarize:152`), the `rel_err` replay uses the SQ2 einsum form (`slot_read.py:279-282`); the old files hold float32 arrays.

Read operators (`memory_backend.py`, `slot_read.py`)
- SoftSlot (`memory_backend.py:232-251`): `logits = scores / tau`, recency `age * log(lam)` added after the division with `age = (count - 1 - i).clamp_min(0)`
  (so `log(lam^(E-1-i))` on the valid slots), invalid slots -inf, an empty row softmaxed over zeros then zeroed (finite and differentiable, no NaN),
  g applied after the weighted sum. `decay` is validated to (0, 1] and `lam` is the adapter's (`slot_read.py:209-210`).
- HardSlot (`:259-272`): `g * v_j`, j the largest `k.q` over valid slots, ties to the lowest index (`amin` of `where(scores == best & valid, slot, S)`),
  empty row zero. Clean-final (`slot_read.py:226-246,55-61`): the written raw value of the question's own fact at the final valid question token only,
  g = 1 and `final_only` enforced (`:191-192`), one question per episode enforced. Scopes: `all_valid_question` reads the backend at every valid token;
  `final_only` is the two-state `ScopedRead` whose `final` flag is `last == t` per token (`:122-123`), other tokens read the native matrix.
- Owner indexing: `bind` validates dtype and range (`memory_backend.py:288-296`), `expand` repeats owners only (`:307-311`), `select` row-selects owners
  (`:313-314`, the base class takes and ignores `final`, so the plain-view call at `slot_read.py:101` is valid); a sentinel check with per-episode
  magnitudes, reorder, expansion, left/right/hole padding, a +500 perturbation of one memory and bitwise-untouched other episodes is
  `owner_mask_check` (`slot_preflight.py:108-152`).
- Capacity: `build_slots` (`memory_backend.py:71-98`) enforces a prefix, raises `CapacityError` when any row exceeds S and never truncates;
  `capacity_check` (`slot_preflight.py:234-248`) tests E = S (keeps all, tensors equal) and both E > S shapes. Zero rows: empty soft, hard and a zero
  matrix read zero. Bytes: `persistent_bytes` is `numel * element_size` over keys, values, valid and count (`:105-107`); `4*16*(768+128) + 16 + 8 = 57,368`
  and `4*768*128 = 393,216` are asserted against the registered numbers at the registered shape (`slot_preflight.py:220-230`).
- Features (`slot_preflight.py:188-216`): keys are the unit DPFP-nu-3 features of the recomputed `K u` (768-d), values the raw `V u`, `M q` equals the stored
  coefficient sum (tolerance 1e-6; the old `gram_relative_error` maximum is 1.3e-7), `base_keys` poisoned with NaN changes nothing.

Frozen set and freeze verification
- Trainable `("Q","R","U","B")` (`slot_rules.py:17`) and frozen `("A","K","V","a_lambda","a_beta")` plus buffers `input_mean, input_std` (`:18-19`);
  3 x 16,384 + 131,072 = 180,224 and 262,144 + 16,384 + 16,384 + 2 = 294,914, asserted (`slot_run.py:36,235-237`). `standardize()` is replaced by a refusal
  (`slot_experiment.py:36-46`); the parent's buffers load strictly (`slot_read.py:145`) and travel in exports (`slot_experiment.py:250-277`). The non-tensor
  frozen items (DPFP nu 3 and key dimension 768, ridge eta 1e-4, eps, rho 0.25, alpha 1, g, tau) are config values read from the parent's export or fields of
  the immutable `ReadSpec`; no slot file assigns to `adapter.cfg` or to any of them (grep), and an export reloads through the parent's own signature check
  (`slot_experiment.py:280`).
- (i) `freeze_for_s1a` + `trainable_report` (`slot_experiment.py:161-205`): the `requires_grad` set equals the trainable set, the backbone and value head hold
  none, the optimizer has one group with exactly those ids. (ii) `digest_report` (`:208-217`) blake2b-128 of every frozen tensor and buffer unchanged and every
  trainable one changed, raising otherwise; called at every dev checkpoint (`slot_train.py:108`) and at the end (`:118`). (iii) `support_digest` (`:221-228`)
  before and after, a hard error on a difference (`slot_train.py:121-125`). (iv) `gradient_report` (`:231-245`) after the first backward (`slot_train.py:88-89`):
  finite non-zero gradients for the four, `grad is None` for every frozen adapter tensor and the backbone; the read path is not under `no_grad`
  (`slot_train.py:83-87`), so the graph runs through the frozen upper layers. The `frozen_set_check` `passed` (`slot_preflight.py:271`) is the support-digest
  equality only because (i), (ii), (iv) raise on failure; the registered counts are ANDed in at `slot_run.py:237`.
- Could a frozen tensor change? Only the optimizer writes parameters and it holds exactly Q, R, U, B; the state, keys and values come from frozen A, K, V,
  a_lambda, a_beta under `no_grad`; buffers are not parameters; weight decay acts inside the optimizer on its own tensors. `test_the_freeze_verification_catches_each_defect`
  (`tests/test_slot_experiment.py:223-264`) injects each of (i)-(iv).

Training (`slot_train.py`, `slot_rules.py`)
- Loss is the full-vocabulary answer cross-entropy only (`slot_train.py:84`, D9). AdamW betas 0.9/0.999, eps 1e-8, wd 0.01 (`:59-60`, `slot_rules.py:21`), fresh
  (no optimizer state is loaded; `load_inference` rejects exports that carry one). Warm-up `lr * min(1, step / 20)` then constant (`:78-79`), the old loop's form
  (`read_fix_run.py:154`). Clip 1.0 with `error_if_nonfinite` returns the pre-clip global norm (`:90`), stored per step and logged with the clipped flag
  (`:95-100`), clipped fraction in the summary (`:131,140`; decision 8).
- Streams: pilot `slot_data.TRAIN_BASE["S1A","pilot"]` (`slot_run.py:341`), runs `TRAIN_BASE["S1A", seed]` (`:416`); `BatchStream` yields step-indexed
  episodes in submission order (`prefetch.py:75-99`), so both arms of one seed see the same batches (`tests/test_slot_experiment.py:347`).
  `check_disjoint(MAX_STEPS)` covers every N <= 6000.
- N: `slot_rules.schedule_steps` (`:65-74`) is the prereg formula `min(6000, 10*floor(((3600 - max(90, Ld))/1.5 - 11 Ev)/S/10))` (S over steps 11-200:
  `slot_train.py:132`; Ev the largest 1,000-episode evaluation; Ld the first load); `N < 3000` is a STOP (`slot_run.py:373-374`).
- Checkpoints: step 0 recorded and `selectable=False` (`slot_train.py:66-67`), then `k N / 10` (`slot_rules.py:97-101`); the incremental choice
  `nll < best - 1e-6` (`slot_train.py:110`) is cross-checked against `select_checkpoint` at the end (`:126-130`).
- LR pilot: grid 3e-4, 1e-3, 3e-3 x two arms, 200 steps, one shard evaluation after step 200 plus step 0, lower LR within 0.01 nats, STOP if every candidate is worse
  than step 0 (`slot_rules.py:55-62`, `slot_run.py:343-364`), the same 6,400 episodes for every candidate.
- Calibration: the 18 candidates (`slot_rules.py:37-41`), float64 NLL means (`slot_run.py:117`), ties 1e-4 nats then `|g - 1|` then larger tau (`:44-52`).

Runner and preflight
- Outputs: every mode requires `--out` absolute and under `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1` (`slot_run.py:33,494-495`); modes dispatch to
  distinct functions (`:497-517`); all but `preflight` and `reload-check` call `require_lock`, and the stage config must equal the registered rendering (`:54-59`).
- Sealed data: nothing in the reviewed files generates a sealed set except `evaluate_entries` (`slot_run.py:451`) through `slot_data.eval_set`, which needs the
  token and a named checkpoint (`slot_data.py:237-245`); `load_open_token` is called only for `confirmation` and `unseen` (`slot_run.py:512`); S0, smoke,
  pilot, train and preflight use only the S0 and S1A calibration/dev/smoke/pilot sets; `stage_config` names the sealed sets without seeds (`slot_rules.py:124-131`).
- Preflight: nine required checks (`slot_run.py:241-244`): native replay, Matrix identity, owner/mask, isolation, frozen set, bytes, capacity, feature
  consistency, D128-3 replay; the destructive frozen-set check runs on a fresh load (`:232`); the lock refuses an inconsistent report
  (`prepare_slot_manifest.py:62-70`).
- S0 pipeline (`slot_run.py:79-193`): the roles and scopes table, D128-3 at S-RE's calibrated values without recalibration, the identity of the
  generated episode ids with the registered ones (`ids_record`, a mismatch is a STOP).

Numerics and hashing
- Per-sample NLL float32; calibration and dev NLL float64 means; `loss_float32` float32 mean for the identity replays; adapter fp32 with the bf16 backbone as before;
  no autocast or TF32 setting in any new file. `LLMExperiment.__init__` calls `configure_numerics()` (`train.py:60`; `inject.py:26-36`) before the first model
  operation, so `use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` hold for every run; the new operations (index_select forward and backward, bmm,
  matmul, softmax, masked_fill, index_copy, where, amin/amax, cross_entropy) are not in torch 2.14.0's list of operations that raise in deterministic mode
  (`torch/__init__.py:1774-1841`; `index_select` backward, `index_copy` and `index_add` are on its "act deterministically" list; `NLLLoss` is on the
  raise list, but the loss here is the same `F.cross_entropy` on `[N, V]` logits that the old loop and the earlier runs used under the same mode).
- Hashing is blake2b only: `memory_backend.py:112`, `slot_data.py:136,195`, `slot_experiment.py:176,223`, `tests/slot_golden.py:59`, and `stable_hash`
  (`infra/seeds.py:9-12`) used for the science and stage hashes. No SHA-256, SHA-1 or MD5 in the diff.

Tests (desk-checked only)
- `tests/test_slot_backend.py` (13 tests): hand-computed soft case, small-tau limit, empty and padded slots, hard ties, recency, owner indexing under reorder and
  expansion, bad owner and dtype, capacity boundary, bytes, read-only and `online_write` refusal, scoped read and oracle, `ReadSpec` validation.
  `tests/test_slot_experiment.py` (31 functions, 36 cases): the layout fixture, bitwise Matrix = Native (logits at L = 1, 2, 4; gradients), the golden gate, tensor states
  keep the original paths, token placement per owner, adapter checks, each freeze defect, `standardize` refused, export reload with buffers and scope, the
  ablation copy of A, the trainer record and selection, a NaN stop, the S0 structure, the runner gates (lock, config, sealed sets), the smoke, pilot, arm-seed
  run and five release mismatches, and the analysis script reading the runner's records. The test counts are the state file's (3727662: 36 passed); I did not run them.

## Could not verify (needs the GPU)
1. That the tests pass (nothing ran): in particular the bitwise equalities (`torch.equal`, `atol=0` after reload) rely on same-process kernel determinism, and the
   golden digests on the current tree rely on the same torch/GPU as the capture.
2. Decision 5 at real size: whether `matrix_identity` and `native_replay` come out bitwise on the real S-RE and D128-3 (the static comparison above says the operation
   sequences are identical, with equal shapes, strides and dtypes into the same batched matmul).
3. S, Ev, Ld and therefore N and the walltime; peak memory of 128 sequences with the soft read and backward through layers 15-27.
4. That `index_select` / `index_copy` backward under deterministic mode on CUDA with torch 2.14 behave as the op list says (the old loop used the same operations).
5. The effect of the soft and hard reads at earlier question tokens through the real layers 15-27: the tiny fixture cannot observe it (adapter at the last layer);
   placement is verified at the adapter level (`owner_mask_check`, `tests/test_slot_experiment.py:131`), and the backend is the only difference from the Matrix path
   that the preflight does check at depth.
6. That `BatchStream` with `workers=2` yields the same batches for both arms (the test uses `workers=0`); by reading it, futures are consumed in submission order and
   episodes are indexed by step, so it should.
