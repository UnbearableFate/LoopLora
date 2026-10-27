# Review of the TSM failure protocol (WP-S9 steps 1c and 1d)

Reviewer: Sonnet 5.5 sub-agent, read-only. Subject: branch wp/sym at 87746bf (worktree
/work/xg24i002/x10041/LoopLora-wt/sym), diff be2a972..87746bf of scripts/run_tsm.py, scripts/analyze_tsm.py and
tests/test_tsm_scripts.py, read together with the full current files and the entry points they call
(src/looplora/sym/train.py, tsm.py, run.py, scripts/s9_common.py, submit.py, pbs/job.pbs).
Spec: docs/prereg/TSM.md ("Decisions at registration", C1, C2), docs/tasks/WP-S9_symbolic_followups.md (master copy,
steps 1c and 1d), command sheet in docs/reports/WP-S9-state.md.

Nothing was executed with torch or pytest (login node). The worktree is untouched (`git status` clean). Scratch files
are in this directory only.

## Method

- Static reading of the diff and the full files; hand trace of every new or changed test against the code (training
  attempts, rerun guards, classification table, evaluation restarts, audit); the one-line statements of the sheet
  compared with the argparse definitions.
- Stdlib-only harness (`harness.py`) that execs the real top-level definitions of run_tsm.py with stubs:
  - `simA_classify.py`: `classify_failure` on 25 exception cases, including the real torch 2.14 / cuDNN 9 texts.
  - `simB_eval_fuzz.py`: 4000 random evaluation sequences (killed / failed / cut-off manifest / complete, code
    changes, flags, aging); checks at most 3 accepted attempts, strictly increasing attempt numbers, restart rows equal
    to the moved directories, correct code in each row. 0 violations.
  - `simC_train_fuzz.py`: 6000 random single-run training sequences (completion, kill, infrastructure, numerical,
    code defect, death before the run directory; caps 3 to 5; code changes); checks attempts never exceed the cap in
    force, no mixed code across the attempts of a run, nothing started after complete, numerical or defect. 0 violations.
  - `simD_identity.py`: evidence for F1 and F2.
- Binary inspection without importing torch: `strings`, `nm`, `objdump` on torch 2.14.0+cu132 libc10.so,
  libc10_cuda.so, libtorch_cpu.so; reading torch/nn/utils/clip_grad.py and torch/cuda/__init__.py.
- py_compile (bytecode in this directory) and pyflakes (Python 3.9.25) on the three files: clean.

## Verdict

No defect found that breaks a registered quantity or decision rule, and no crash found on the paths the tests
exercise. The tests, traced by hand, agree with the code. Findings: one Medium (a silent protocol hole), one
Low-Medium (false study stop on a realistic I/O fault), six Low. Each has a concrete scenario and a small fix.

## Findings, by severity

### F1 - Medium - the evaluation code is never compared with the training code on an evaluation's first attempt

Where: scripts/run_tsm.py:804-830 (`evaluation_attempt`: compares only with an earlier evaluation manifest, lines
822-829), :855-888 (`stage_evaluate`: `training_source_hash` is recorded at 884 and `source` at 885 but the two are
never compared; the flag check at 865-866 only needs `--restart-interrupted`); scripts/analyze_tsm.py:81-90 (audit
checks that `training_source_hash` equals the run records and that `source_hash` is present, not that they are equal),
:277-287 (the code-change paragraph is printed only `if manifest["restarts"]`).

Spec: C1 "every TSM training and evaluation job runs at one CODE commit"; C2 "the evaluation job may use a later
commit only under this rule" (the `--new-evaluation-code` path).

Scenario: the sheet uses one `$CODE` for tsm-train, tsm-eval and tsm-analyze, and TSM is the last block, so a bump
of `$CODE` between the training jobs and `tsm-eval` (after a fix found while running IVD or CVH) is plausible.
`evaluate` at the new commit is attempt 1: no refusal, the audit passes, `summary.md` says nothing (no restarts), and
only `summary.json` holds two different values (`training_source_hash`, `evaluation_source_hash`). A second path:
`evaluate --restart-interrupted --new-evaluation-code` on a first attempt with nothing on disk is accepted at any
code, because the flag is checked only against `--restart-interrupted`, and there is no earlier manifest to compare.
Evidence: `simD_identity.py` runs the real `evaluation_attempt`: first attempt at code LATER with runs trained at
TRAIN returns attempt 1; a restart at LATER without the flag returns attempt 2; both flags on a first attempt return
attempt 1. The state report documents "compared with the training hash by no script" (section 9 item 7(4), line 421;
section 10.6, line 640), reading C2 as asking for no check; I report it because the C2 sentence reads as a
prohibition outside the C2 path, the bypass with both flags is not documented, and the fix is one `require`.
`git_commit` and `source_dirty` are recorded everywhere but compared nowhere (only `source_hash` is).

Fix: in `evaluation_attempt`, when `earlier_evaluation(root)` is None require
`manifest["source"]["source_hash"] == manifest["training_source_hash"]`, and refuse `--new-evaluation-code` when no
earlier attempt exists (neither in place nor moved). Print both hashes in `summary.md` always, and flag a difference
even when `restarts` is empty. Every existing test starts attempt 1 at the training code, so they stay valid; add one
test for the refusal.

### F3 - Low-Medium - torch.save / torch.load I/O faults are classified as a code defect and stop the study

Where: scripts/run_tsm.py:208-220 (`classify_failure`: only `OSError` counts as I/O), :421-424 (`train_one` records
`failure_kind`); the writes are src/looplora/sym/tsm.py:186 and src/looplora/sym/train.py:126, 141
(`torch.save(..., out / "checkpoint.pt")`), the read train.py:112 (`torch.load`).

Spec: C1 "CUDA out-of-memory and I/O errors are infrastructure failures".

Evidence: torch 2.14 libtorch_cpu.so carries "PytorchStreamWriter failed", "file write failed", "file close failed"
and "PytorchStreamReader failed": the C++ zip writer reports a failed write as a `RuntimeError`, not an `OSError`.
`simA_classify.py`: `RuntimeError("... PytorchStreamWriter failed writing file data/12: file write failed")` and
`RuntimeError("... file close failed")` give `code_defect`; `OSError(122, "Disk quota exceeded")` (the same fault in
`train_log.jsonl` or `np.savez`) gives `infrastructure`.

Scenario: a byte or file-count quota hit, ENOSPC or an OST fault during the final `torch.save` of a run (the largest
write of the run). The record becomes `failure_kind: code_defect`; `study_stop` then refuses every later `train` and
`evaluate`, and C1 says a code defect retrains every TSM run. The only exit is the orchestrator's manual move of the
run to `attempts/` (the sheet describes it for a diagnosed hardware fault, not for this case).

Fix: in `classify_failure` treat a `RuntimeError` whose lower-cased text contains "pytorchstreamwriter failed" or
"pytorchstreamreader failed" as I/O (an identical rerun of a genuinely corrupt checkpoint fails the same way three
times and then pauses, so the exposure stays bounded); add the two cases to the classification test. Alternatively
state in the sheet that the orchestrator reclassifies such records by hand.

### F2 - Low - `train` does not compare `source_hash` across runs; the sheet says it does

Where: scripts/run_tsm.py:645-661 (`stage_train` computes `source` and calls `check_rerun` per run), :304-325
(`check_rerun` compares only with the earlier attempts of the same run), :593-597 (`final_states`: the only cross-run
check, run by `evaluate` and the analysis); sheet docs/reports/WP-S9-state.md:253-254 ("All 18 runs must come from
the same $CODE ... `train`, `evaluate` and the analysis refuse otherwise").

Scenario: tsm-train-29 is submitted at another `--commit` than tsm-train-17. Both complete (up to about one node-hour
each); the mismatch is found only by `evaluate` ("not all trained with the same code"), before any confirmation data
is touched, and the odd runs must be retrained. Evidence: `simD_identity.py`: `check_rerun` accepts the first
attempt of run Y at code B while run X is complete at code A (a rerun of X at B is refused).

Fix: in `stage_train` (and in the `train-one` child, before `mkdir`) collect `source.source_hash` from the records in
`runs/*/record.json` and `attempts/*/record.json` and require that they equal the current one; or correct the sheet.

### F4 - Low - a host allocator failure inside torch is a code defect; the test pins a message torch 2.14 does not emit

Where: scripts/run_tsm.py:182-196 (`is_allocation_failure`); tests/test_tsm_scripts.py:510-511.

Evidence: in torch 2.14 `c10::alloc_cpu` (libc10.so) fails through `c10::ThrowEnforceNotMet`; no function of
libc10.so constructs `c10::OutOfMemoryError` (libc10_cuda.so does, as a control), so a CPU tensor allocation failure
is a plain `RuntimeError` with the text "[enforce fail at alloc_cpu.cpp:N] data. DefaultCPUAllocator: not enough
memory: you tried to allocate N bytes." The test uses "DefaultCPUAllocator: can't allocate memory: you tried to
allocate ...", which does not occur in the installed torch libraries; the expected value (`code_defect`) is the same
for both texts. `simA_classify.py`: the real text gives `code_defect`.

Scenario: host memory exhausted inside a torch CPU allocation (not a numpy or Python allocation, which is a
`MemoryError`) stops the study. C2 lists "host `MemoryError`" and the author reads it strictly (section 10.6, line
638). The probability is low (CPU tensors are small, and a cgroup kill leaves the record "running", which is
infrastructure). The same remark applies to `CUBLAS_STATUS_NOT_INITIALIZED`, the status commonly reported when
`cublasCreate` cannot get memory on a full GPU: it is a code defect by C2's wording.

Fix: either add "defaultcpuallocator: not enough memory" (and, if wanted, the NOT_INITIALIZED status) to the
`RuntimeError` matchers and fix the test text, or have the orchestrator confirm the strict reading and correct the
test text to the real message.

### F5 - Low - a stopping failure terminates the siblings of the same job; C1 says running jobs may finish

Where: scripts/run_tsm.py:475-476 and 486-488 (`schedule`: a fatal failure raises and the `finally` terminates every
running child), :676-684 (`stage_train`).

Scenario: in a 6-run job a T0 numerical failure at minute 5 sends SIGTERM to the 5 siblings. Their records stay
"running", they are incomplete (not outputs "kept" in the sense of C1), and each later counts one attempt when
rerun. For a code defect this is harmless (every run retrains). The state report documents the behaviour; I list it
so the orchestrator confirms it is intended.

Fix: on a stopping failure stop starting pending runs but let the running children finish, then raise.

### F7 - Low - the 300 s rule is not pinned by any test, and the silent gap of a live evaluation is unmeasured

Where: scripts/run_tsm.py:81 (`QUIET_SECONDS = 300`), :345-358 (`require_ended`), :817-821 (`evaluation_attempt`);
tests/test_tsm_scripts.py:141 (fixture patches 0), :654, :1187, :1259 (patch 3600).

Scenario 1: a change of `QUIET_SECONDS` to 30 passes every test. Scenario 2: a live evaluation writes a file only
when one run's evaluation ends (`evaluate_runs`, line 542; the first write of each writer follows the generation of
its 2000 episodes and one full run evaluation). If that gap exceeds 300 s, an operator who skips `qstat` can move a
live evaluation aside under its job. Training is safe (`train_log.jsonl` is flushed every `log_every` steps, 10 in
configs/p1s/default.yaml, a fraction of a second at the 0.033 node-h per run of the P2-S-F1 measurement).

Fix: add `assert run_tsm.QUIET_SECONDS == 300`; at the smoke review compare `evaluation_rows[].seconds` (200
episodes, scale by 10) with 300 s, and add a heartbeat write during evaluation if it is close.

### F6 - Low - the pause after three attempts and the "recorded decision" are enforced by the operator only

Where: scripts/run_tsm.py:304-325 (refuses only the fourth attempt of the same run), :630-632 (any N >= 3 accepted),
:673 (the manifest records N, not the decision).

Scenario: run R has used 3 attempts (all infrastructure); `train` for other runs is still accepted, so the C1
sentence "no further TSM job is submitted" depends on the orchestrator. The confirmation set stays protected
(`final_states` needs every run final). Same for `--max-attempts 9` on a first job without a prior decision.

Fix (optional): refuse a `train` start while a run that is not final has used `max_attempts` attempts, unless it is
the run being granted; or accept the operator-side enforcement.

### F8 - Low, spec level - C2's file-level check cannot be satisfied for realistic evaluation fixes

Where: docs/prereg/TSM.md:152 ("the fix changes no file that the training path uses"); sheet
docs/reports/WP-S9-state.md:261-264; code: evaluation lives in scripts/run_tsm.py (`evaluate_runs` 500,
`stage_evaluate` 855) next to `train_one` (384), and in src/looplora/sym/tsm.py (`evaluate_arm`, `gate_profile`,
`load_tsm`) next to `fit_tsm` and `train_arm`; tsm.py:33 imports `.evaluate._slice`.

Scenario: a defect in `evaluate_runs`, `evaluate_arm` or `stratify_queries` is fixed; any such fix changes a file the
training path imports, so a literal check fails and the choice is "retrain all 18 runs (about 1.2 node-h)" or a
judgment at function level.

Fix: before the lock, word the check at function level ("no function reachable from `train_one`"), or move the
evaluation code into its own module.

Minor (no failure mode beyond confusion): the `--rerun-incomplete` help (run_tsm.py:936-940) lists only I/O error,
CUDA out-of-memory, walltime and node fault and says "at most twice"; allocation failures and `--max-attempts` make it
stale.

## The seven check areas: what was verified

1. Training attempts. `MAX_ATTEMPTS = 3` counts moved attempts plus the run in place, in the parent
   (`stage_train` 657-661) and in the child (`train_one` 396), so an aborted rerun or a child that died before it
   started is counted correctly (test 619; fuzz C). `--max-attempts` below 3 is refused (630); N is passed on the
   child command line (`start_child` 448, test 317), recorded in the manifest (673); every granted attempt is
   compared with the latest earlier attempt, moved ones included (295-325). A run that never created its directory
   uses no attempt (log kept as `attempts/<run>-notstarted<k>.log`). Not enforced by the script: the decision itself
   and the cross-run pause (F6).
2. `failure_kind`. Matching agrees with torch 2.14 for the clip-grad text and type (clip_grad.py:109-115, a
   `RuntimeError` with "non-finite"), `torch.cuda.OutOfMemoryError` (= `torch._C.OutOfMemoryError`), "CUDA error: out
   of memory" as `RuntimeError` or `AcceleratorError`, `CUBLAS_STATUS_ALLOC_FAILED`, the cuDNN 9 host and device
   allocation statuses, `MemoryError`. The TSM path has no `raise ... from`, DataLoader, torch.compile or autocast, so
   wrapped or chained exceptions do not occur; only the top-level exception is read. A code bug can be rerun as
   infrastructure only through an `OSError` (for example a wrong path), `KeyboardInterrupt`/`SystemExit` or a
   misread allocation text; each rerun is identical, so it is bounded by three attempts and then goes to the
   orchestrator (C1 names I/O errors as infrastructure). Misclassified as a code defect: F3, F4. A `FloatingPointError`
   from a code bug would count as numerical; the registration cannot distinguish it, the smoke is the guard.
3. After a T0 failure. `study_stop` refuses every later `train` (642-644) and `evaluate` (`final_states` 579-581)
   before any confirmation data is touched; a code defect is reported before a T0 failure and an unfinished earlier
   run does not hide it (tests 942, 1054). Pending runs of the same job are not started; running siblings of the same
   job are terminated (F5). One `source_hash` across the 18 records is enforced by `final_states` (593-597) at
   `evaluate` and in the analysis, and per run by `check_rerun`; nothing else enforces it (F1, F2).
4. Evaluate restarts. At most 2 restarts: `used = moved + (state != none)` counts across invocations, including an
   aborted restart (rename done, manifest not yet written; fuzz B, test 1214). Partial output is moved by `rename`
   only; only `manifest.json` is read (`evaluation_state`, `earlier_evaluation`, `restart_rows`); a `chmod 000` test
   proves the samples are not opened (1107-1136). The 300 s rule uses the mtime of the directory and every file in it
   and is skipped for a manifest the job itself marked "failed". A cut-off or missing manifest counts as an interrupted
   evaluation (test 1255). `--new-evaluation-code` needs `--restart-interrupted`, exempts only the `code` entry,
   leaves design, lock commit and checkpoints compared, and both code hashes are recorded per attempt (restart rows,
   `source`, `training_source_hash`); the audit cross-checks the restart rows with `attempts/evaluation-attempt*`,
   the checkpoint seals and the training hash (analyze_tsm.py:79-90). Exception: F1.
5. Errors before a run directory exists. The child's traceback stays in `logs/<run>.log`; the run is "missing", not an
   attempt, the stage ends with "did not finish ... see logs", and the next `train` keeps the log as
   `notstarted<k>.log`. The manifest lacks the child's exit code (the return value of `schedule` is discarded at
   677), which only matters for a SIGKILL with an empty log.
6. Registered quantities. `git diff be2a972 87746bf -- src configs` is empty; no changed line of run_tsm.py or
   analyze_tsm.py touches seed 999007 or 4242, SEEDS, WRITERS, the arms, depths, hops, the thresholds or `DECISION`.
   The analysis decision logic is unchanged (it lives in src/looplora/sym/tsm.py).
7. Tests. Traced by hand against the code: they pass, and they would fail on the realistic regressions they target
   (identity check dropped, cap not passed to the child, classification table, restart accounting, unread output,
   flag semantics). Spurious-failure risks found: none on Miyabi; `_seal` (1112) fails its own precondition if the
   suite runs as root; no test depends on the 300 s value (patched to 0 or 3600), timestamps are shifted relative to
   the files' own mtimes; the children run in-process, so real process management is left to the smoke job (as the
   module docstring says); imports are complete (pyflakes clean). Not caught by the tests: F1, F2, F7, and any
   mismatch between the synthetic exception texts and the real library texts (F3, F4).

## Decisions the author documented (not new, listed so they are not read as missed)

- State report section 9 item 7(4) and section 10.6: evaluation hash not compared with the training hash (reported
  as F1 for the reasons given there); CPU allocator `RuntimeError` is a code defect as the registered text has it
  (F4); the orchestrator's C2 file check is not automated (F8); the evaluation keeps three attempts, `--max-attempts`
  applies to training only (matches C2).
- A T0 failure or code defect terminates same-job siblings (F5).

## Could not verify

- Nothing was run: the tests, `train`, `evaluate` and the analysis have not executed anywhere. The pass/fail
  statements above come from tracing and from the stub harness, not from pytest.
- Real process behaviour: `start_child`, `Child.stop` (terminate and wait without timeout), exit codes and signals,
  concurrent jobs on one root (tsm-train-17/29/43 run at once in the sheet; other jobs' partial records are read as
  incomplete by design, by reading only).
- The exact text of a real cuBLAS or cuDNN allocation failure under torch 2.14 and cuDNN 9.24 (only status names
  from the binaries); whether a full GPU shows ALLOC_FAILED or NOT_INITIALIZED.
- Whether T1 and T2 losses stay finite on real data (a deterministic NaN from a code bug would be recorded as a
  registered training failure; the smoke covers it).
- Lustre mtime behaviour across nodes for the 300 s rule, and the real write cadence of a live evaluation (F7).
- The worktree copy of docs/tasks/WP-S9_symbolic_followups.md ends at step 1b; steps 1c and 1d were taken from the
  brief and the master copy, and from C1 and C2 in docs/prereg/TSM.md.
