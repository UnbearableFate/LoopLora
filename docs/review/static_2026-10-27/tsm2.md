# Static review: WP-S9 step 1e (wp/sym 09be205), TSM scripts after fixes F1-F8 and Clarification C3

Reviewer: read-only subagent. Date 2026-10-27. No tracked file was edited and nothing was deleted; my scratch files are under
`/tmp/claude-30041/-work-xg24i002-x10041-LoopLora/7ae01788-4591-4489-abeb-1056ba03ea57/scratchpad/review-tsm2/`. One slip: eight git-ignored
bytecode caches were written into the sym worktree (see "Side effect of the review" at the end).

Scope: `git diff 3ca4d89 09be205` in `/work/xg24i002/x10041/LoopLora-wt/sym`
(fab4006: `scripts/run_tsm.py`, `scripts/analyze_tsm.py`, `tests/test_tsm_scripts.py`; 09be205: `docs/reports/WP-S9-state.md`).
Spec: `docs/prereg/TSM.md` (Decisions at registration, C1, C2, C3), master's `docs/tasks/WP-S9_symbolic_followups.md` step 1e, the previous
review `docs/review/static_2026-10-27/tsm.md`, state report sections 10.8 and 10.9.

Method: full read of `run_tsm.py` (1141 lines), `test_tsm_scripts.py` (1864 lines), the analyzer diff and the sheet; the AST of
`run_tsm.py` at 3ca4d89 against 09be205; and a stdlib-only harness that imports the REAL `run_tsm.py` (with stub numpy/torch/looplora modules) and
runs REAL child subprocesses through the real `schedule`/`stage_train`/`train_one`/`heartbeat`/`evaluation_attempt` (Python 3.9
`/usr/bin/python3 -I` and the venv's Python 3.13.13 with `-I -S`; no torch import, no pytest, bounded runs, no processes left).

## Result in one paragraph

No defect of consequence in the process management (F5), the pause (F6), the one-code rules (F1/F2) or the heartbeat (F7): every scenario I could
build behaved as specified, including the races. One concrete defect in the failure classification (F3): the open-failure text of `torch.save`
("File <path> cannot be opened.") is read as a code defect and stops the whole study; it is an I/O error in C1's own words, and C3 quotes only two of
the texts. One design dead end in F1 (low). Registered quantities are unchanged.

## Findings, ranked

### 1. Medium-Low (low likelihood, high cost): a file that `torch.save` cannot create is a `code_defect` and stops the study

Where: `scripts/run_tsm.py:99-100` (`SERIALIZATION_FAILURES`), `:217-221` (`is_serialization_failure`), `:233-247` (`classify_failure`).
Tests: `tests/test_tsm_scripts.py:38` (`SERIALIZATION_ERROR`), the classification table `:525-599` (cases `:572`, `:593`): no case for the text below.
Sheet: `docs/reports/WP-S9-state.md:244-245` states the rule with the two quoted texts only.

What happens. `is_serialization_failure` accepts a `RuntimeError` only if its text contains "pytorchstreamwriter failed" or
"pytorchstreamreader failed". For an ASCII path (all of ours), `torch/serialization.py:826-831` calls `torch._C.PyTorchFileWriter(path, ...)`, so the
C++ writer, not Python's `open`, creates the file. When it cannot, it raises `RuntimeError("File <path> cannot be opened.")` (and
`"Parent directory <dir> does not exist."`): the strings " cannot be opened." and "Parent directory " sit next to "PytorchStreamWriter failed " in
`libtorch_cpu.so` (read with `strings -a`, lines 350881-350911 of the output); in the upstream C++ `PyTorchStreamWriter::setup` the open stream is
checked with `TORCH_CHECK` before any `valid(...)` call that produces the "PytorchStreamWriter failed" text. I did not trigger a failing call (login
node), so the exact text is read from the binary and the upstream source logic, not observed.
None of the two marker substrings is in that text, it is not an `OSError`, not an allocation failure, not non-finite: `classify_failure` returns `code_defect`.

Failure scenario. `checkpoint.pt` (and `writer.pt`) is created at the end of every run, after the training time is spent. If the group's
file-count (inode) quota is exhausted (the Miyabi platform notes name file-count quotas; this repo's many studies write many files), or the Lustre MDS
returns EIO/ESTALE/EACCES at that moment, `torch.save` raises the text above. `train_one` records `failure_kind=code_defect`; `schedule` drains and
the stage ends `stopped`; every later `train` refuses (`study_stop`); the registered consequence of a code defect is "the fixed code retrains every TSM
run" (C1), i.e. all 18 runs in a new root. The same quota condition at `record.json`, `train_log.jsonl` or `config.json` (Python `open`) is an `OSError`
= infrastructure (rerun), so one condition gets two classifications depending on which file is created first. C3's "Bounded exposure" covers a defect
misread as infrastructure (three attempts); it does not cover the reverse misread, which stops everything at once.

Reproduced (real `classify_failure`, real `train` stage, stub child): `h/s7_cannot_be_opened.py` prints
`infrastructure` for `OSError(122, ...)` and for both "PytorchStreamWriter failed ..." texts, `code_defect` for "File ... cannot be opened." and for
"Parent directory ... does not exist."; the job ends `stopped` with `run_states` `code_defect`, and a later job for another seed is refused.

Fix (needs the orchestrator, before the lock, because C3 is registered text): add a C3 sentence ("a RuntimeError of torch's checkpoint writer that cannot
create the file, `File <path> cannot be opened.`, is an I/O error too") and in `is_serialization_failure` accept
`isinstance(error, RuntimeError) and ("cannot be opened." in text and text.startswith("file "))` besides the two markers. Add table cases: that text
-> `infrastructure`; the same text in a `ValueError` -> `code_defect`. Leave "Parent directory ... does not exist." a code defect (a wrong path is not
an infrastructure fault) unless the orchestrator decides otherwise. Update the sheet sentence at `WP-S9-state.md:244-245` and the `--rerun-incomplete`
help. If the orchestrator does not want to touch C3, the alternative is to record it in the sheet as a known misread whose remedy is a hand-written
decision; I would not leave it silent.

### 2. Low: an evaluation-code defect before the first evaluation manifest exists cannot be fixed under the rules

Where: `scripts/run_tsm.py:935-938` (`used > 0 or not new_code`), `:951-957` (first attempt must run at `training_source_hash`),
`:1025-1026` (`--new-evaluation-code` needs `--restart-interrupted`), `:1038-1051` (everything before the manifest write).

What happens. `stage_evaluate` runs `final_states`, `checkpoint_seals`, `training_source_hash`, `source_metadata`, `runtime_metadata` (CUDA init),
`evaluation_attempt`, `restart_rows` before it writes `evaluation/manifest.json`. A defect in that stretch (for example a `KeyError` on a real record
field that the stub tests do not have) leaves no evaluation directory and no moved attempt. The fix is a new commit, hence another `source_hash`
(the hash covers `scripts/`, `src/`, `tests/`, `configs/`); the first attempt must equal the training code (refused), and `--new-evaluation-code` is
refused without an earlier attempt. C2 says an evaluation-code defect is fixed by a restart at the fixed commit, without saying it must have a
manifest; the implementation is stricter than the text, and the way out is an unregistered manual step.

Fix: either document in the sheet that a defect before the first manifest is reported to the orchestrator and needs an explicit decision (the
analogue of C2's "Errors before a run directory exists"), or let `--new-evaluation-code --restart-interrupted` run as a first attempt at another code
when `evaluation/` and `attempts/evaluation-attempt*` are both absent and the decision is recorded (`--decision FILE`). The sheet-only fix is enough
for the risk (the code before the manifest is small and exercised by the stub tests).

### 3. Low / notes (each with a concrete but minor cost; none blocks the run)

a. Exit codes and logs on the stopped path: `stage_train` discards `schedule`'s return value (`run_tsm.py:791`) and replaces the error text by the
   `study_stop` text (`reason = stopped or str(error)`), so the stopped manifest does not carry the stopping child's exit code or its log path
   (`schedule`'s message has them). The run name and the failure text are there; the log is at `logs/<run>.log`. Cosmetic.
b. The stopped manifest is written only after the drain (`run_tsm.py:793-799`). A PBS walltime kill during a long drain leaves
   `train_manifest-<label>.json` at "running"; the stop is still enforced from the records by every later `train` (`study_stop`) and by `evaluate`
   (`final_states`). The sheet could say that a manifest left at "running" means the job was killed: read the run states.
c. `start_child`'s `OSError` (e.g. the log file of a run that another job just started; `run_tsm.py:561-562`) still terminates the running siblings
   (`finally` -> `Child.stop`), each of which stays "running" and burns an attempt after 300 s of silence. Pre-existing; only with overlapping run
   selections of two jobs, which the sheet forbids (one job per seed).
d. F2 is best-effort in time: the first `record.json` of a run exists only after CUDA initialisation (`runtime_metadata`), so two jobs at different
   codes that start inside that window both pass `require_one_code`; the last guard is `final_states` at the evaluation (after the runs are paid
   for). The orchestrator's submit step (one CODE per study) is the real control.
e. `require_one_code` vetoes on any readable `runs/*/record.json` or `attempts/*/record.json` at another code, including a stray copy (S5d).
   By design (the message names the codes), but a hand-made backup under `runs/` blocks training.
f. `heartbeat`'s `thread.join()` has no timeout (`run_tsm.py:1010`): a hung file system inside the 30 s beat would delay the end of `stage_evaluate`
   (the "complete" manifest is already written). Negligible.
g. Cross-job behaviour after a stop: pending runs of other seeds' jobs keep starting. The sheet says so (`WP-S9-state.md:509-510`); within C1 ("jobs
   already running may finish"), as an interpretation.

## Verified OK (with the evidence)

1. F5 process management (real child processes, `h/s1_drain.py`, runs under 3.9 and 3.13): T0 numerical failure at 1 s with siblings ending at 4 s
   (complete) and 6 s (code defect) and runs pending: no pending run started (no log, state `missing`), both siblings ran to their end and kept
   their records, the job ended at about 6 s with a `ValueError` naming the first stopping run, `train_manifest` `status=stopped`, `reason` = the
   `study_stop` text (a code defect before a T0 failure), `run_states` correct (`complete`, `training_failure`, `code_defect`, `missing`), no process
   left, a later job refused. A second stopping failure while draining does not overwrite the first (`stopped = stopped or ...`, `run_tsm.py:552`). A
   sibling SIGKILLed during the drain ends `incomplete`, no hang. Non-stopping failures only (T1 numerical, `OSError`, OOM, a child that exits
   before its directory exists): all selected runs are attempted, status `incomplete`, "did not finish" error, not stopped. The `finally` stops only
   children still running (none after a normal drain); the smoke stage (no `fatal`) is fail-fast as before. Child output goes to a log file (no pipe to
   fill).
2. F6 pause (`h/s5_pause.py`): `paused_runs` counts moved attempts plus the run in place; `evaluation-attempt*` directories and `-notstarted*.log`
   files are not counted; complete and registered-failure runs are never paused; a live third attempt (written within 300 s) does not block another
   seed's first `train` (a), it pauses the study after 400 s of silence (b). `--max-attempts` below 3 is refused; above 3 needs `--decision FILE`
   (existing non-empty file; manifest records `path` and the blake2b digest; the content is not read, by design); `train-one` is internal (`S9_CHILD=1`) and receives
   the stage's value. The smoke stage uses a fresh root (`exist_ok=False`), so the pause and the one-code rule cannot touch it.
3. F1/F2: the first evaluation attempt must equal `training_source_hash`; a restart compares design, lock commit, code and checkpoints with the latest
   readable earlier manifest (in place, else the last moved); `--new-evaluation-code` is refused without an earlier attempt and without
   `--restart-interrupted`; all checks run before the confirmation episodes are generated. `require_one_code` runs in `stage_train` and in `train_one`
   before `check_rerun` and `mkdir`. Three concurrent same-code jobs (one per seed, 6 runs each) x5 passed (`h/s4_concurrent.py`); a writer process
   rewriting a large `record.json` in place while the real `recorded_source_hashes`/`run_state`/`study_stop`/`paused_runs` read it for 8 s produced
   no exception and no wrong result (cut-off reads return `None`).
4. F3/F4 classification: `torch.cuda.OutOfMemoryError` and `MemoryError` by `isinstance`; "CUDA error: out of memory" (the `AcceleratorError` text)
   by text inside `RuntimeError`; the cuDNN 9 statuses, cuBLAS ALLOC_FAILED, "DefaultCPUAllocator: not enough memory" and
   `CUBLAS_STATUS_NOT_INITIALIZED` with "cublasCreate" are matched only inside `RuntimeError` (a `ValueError` with the same text stays a code defect);
   `CUBLAS_STATUS_NOT_INITIALIZED` of another call stays a code defect. In `libtorch_cuda.so` the only handle-creation string is `cublasCreate(`
   (the Lt handle is a cast of it), so the matcher covers torch's creation site. The 36-case table is consistent with the code; the C2 cases
   are unchanged.
5. F7 heartbeat (`h/s6_heartbeat.py`, real thread in a real process, silence rule shortened): a live evaluation that writes only the heartbeat is
   refused a restart; after it was killed and the silence elapsed, the restart is allowed (attempt number correct); the first beat is written before
   the thread starts; a failed beat (`OSError`) is skipped; the thread is a daemon, stopped and joined in `finally` (also on failure); the file is
   only ever touched, its content is never read (grep: no reader; `evaluation_state` reads only `manifest.json`; the silence rule uses mtimes of the
   directory and its entries). The directory-moved case: a restart needs 300 s of silence, so a live evaluation cannot be moved; a process that
   stalled for 300 s and then resumed would write its heartbeat (and samples, as before the change) into the new directory: pre-existing hazard
   of the silence rule, not made worse.
6. Registered quantities: `git diff 3ca4d89 09be205 -- src configs docs/prereg` is empty; the AST diff of `run_tsm.py` shows three added
   constants (`CPU_ALLOCATOR_FAILURE`, `SERIALIZATION_FAILURES`, `HEARTBEAT_SECONDS`) and no changed registered constant (seeds 999007/4242,
   arms, writers, thresholds, `TRAINING`, `EVALUATION`, `SMOKE`, `DECISION`, `QUIET_SECONDS=300`, `MAX_ATTEMPTS=3`). `analyze_tsm.py`: `render` now states both codes and flags a
   first attempt at another code; `audit_run` unchanged.
7. Tests: traced against the code; they pass on the correct code by reading and fail for the realistic mutations I tried to apply mentally
   (the state report's mutation check, 188 of 190, is consistent with that); the only wall-clock dependence is the heartbeat tests (a 30 s
   deadline); `_seal` needs a non-root user (pre-existing). `py_compile` and `pyflakes` clean. The diff and this report use blake2b only
   (the digest of the decision file, the seals).

## Could not verify

- Real torch/pytest execution of the tests (login node; the author ran a stub-torch emulation only).
- The exact text of a failing `torch.save` open or write on Miyabi, and real GPU allocation-failure texts (strings read statically, no failing
  call made).
- Lustre behaviour across nodes (mtime visibility of a file being appended, ESTALE), PBS walltime/SIGTERM behaviour during a drain, the real silent
  gap of a live evaluation and the `log_every` cadence on a GPU.
- My harness uses stub children (no torch, no GPU); it proves the process, record and file logic of the real module, not training.

## Side effect of the review (my mistake; disclosed, not undone)

`git status` of the worktree `/work/xg24i002/x10041/LoopLora-wt/sym` is clean and no tracked file differs from 09be205; nothing else in the repo or
the worktrees was touched by me. But my harness and `py_compile` runs used `python -I`, which ignores `PYTHONPYCACHEPREFIX`, so Python wrote bytecode
caches next to the sources. They are git-ignored and derived, and they are the only files of mine outside the scratchpad:

- `scripts/__pycache__/` (created 18:10 JST): `analyze_tsm`, `run_tsm`, `s9_common`, `audit_p4s_stopping` for cpython-39 and cpython-313 (7 files);
- `tests/__pycache__/test_tsm_scripts.cpython-39.pyc` (created 18:13 JST).

They are exact compilations of the committed sources (the pyc header carries the source mtime and size, so they invalidate themselves when a source
changes; `git archive` exports and PBS jobs never contain them), so they cannot change a result. I left them in place because the review rules forbid
deleting anything in the worktrees. To remove them:
`rm -r /work/xg24i002/x10041/LoopLora-wt/sym/scripts/__pycache__ /work/xg24i002/x10041/LoopLora-wt/sym/tests/__pycache__`
(both directories were created by these runs; the parent directories' mtimes equal the first file's creation second).
(`docs/tasks/WP-L7_rfx_lsq.md` and `docs/review/static_2026-10-27/release.md` in the main repo were modified after 18:00 by other agents, not by me.)

## Harness (scratch files, outside the repo)

`h/stubs.py` (stub numpy/torch/looplora, scripted child outcomes), `h/child_main.py` (runs the real `run_tsm.py` as `__main__` in the child),
`h/hlib.py` (imports the real module, patches the compute check, the lock and design verification, the poll interval), `h/s1_drain.py` (F5: S1-S3),
`h/s4_concurrent.py` (F1/F2 concurrency and the record race), `h/s5_pause.py` (F6, F2 leftovers), `h/s6_heartbeat.py` (F7), `h/s7_cannot_be_opened.py`
(finding 1). Run with `PYTHONDONTWRITEBYTECODE=1 /work/xg24i002/x10041/LoopLora/.venv/bin/python -I -B -S h/<script>.py` (stdlib only; `-B` matters: under `-I`
the environment variable alone does not stop the parent from writing bytecode next to the sources, which is how the side effect above arose).
