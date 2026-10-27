# WP-S9 — symbolic follow-ups IVD, CVH, TSM, UNS (sn-sym; LoopLora-wt/sym, wp/sym)

Read first: `docs/tasks/_common_rules.md`, `plan/followup_2026-10-27.md` (§1, §3 F8–F11, §7 item 4),
`docs/paper/evidence_symbolic.md`, the reviews' relevant parts (`docs/review/rv-astra_2026-10-27.md` proposals
3, 4, 6; `docs/review/rv-fable_2026-10-27.md` C4–C6, R1, R2, E2), the symbolic preregs and reports you need
(`docs/prereg/P1S.md`, `P2S.md`, `P2S-F1.md`, IS, JS and P4S-L halting under `docs/prereg/` and `docs/reports/`).
Code: `src/looplora/sym/`, `src/looplora/core/`, the matching `scripts/` (analyze_is.py, analyze_js.py,
analyze_p4s_*.py …).

## Step 1 (now, no compute)
1. Inventory: which existing checkpoints each study needs (paths, seeds, writers) and every evaluation seed /
   confirmation set used so far (C777, C888, 888001, IS, JS, halting …). New confirmation sets must use seeds never
   used before.
2. Pre-registration drafts, decision rules as in the follow-up plan:
   - `docs/prereg/IVD.md` (F8, frozen checkpoints, load × key-overlap, matched write age/order, decay-only control);
   - `docs/prereg/CVH.md` (F9, continuation-value halting heads vs existing heads and hop-1-constrained thresholds);
   - `docs/prereg/TSM.md` (F10: T0 M1-L4 retrained under the same code, T1 stay loss at L = 8, T2 stay loss + sticky
     gate; oracle_random and joint_fact; 3 seeds; B4 reference);
   - `docs/prereg/UNS.md` (F11, `test_unseen` vs seen pool, descriptive).
   Also specify the neighbour-confusion rate for single-read hop-2 errors (C4) as a descriptive analysis of
   existing evaluation samples.
3. Implement the code and tests for all four plus the confusion rate; off paths bitwise unchanged.
4. Node·h estimates from the measured costs of IS, JS, P4S-L and P2S-F1; step-2 commands.
Report: `docs/reports/WP-S9-state.md`. Done line in your final reply: `WP-S9 step 1 done`.

## Step 2 (only when the orchestrator says so)
Run in the plan's order (IVD, CVH, UNS and the confusion rate first; TSM after). Reports per study.

## Step 1b (now, no compute): align with the registrations
The orchestrator registered `docs/prereg/{IVD,CVH,TSM,UNS,NCR}.md` on master at 40ddf08; identical text is on wp/sym.
Read each file's "Decisions at registration" section.
1. Merge master into wp/sym.
2. TSM training failures, as registered:
   - an infrastructure failure gets an identical rerun, at most twice;
   - a non-finite loss in T1 or T2 fails that arm for that writer, and its completed runs are evaluated descriptively;
   - a non-finite loss in T0 stops the study.
   Adapt `run_tsm.py evaluate` and `analyze_tsm.py` to this, with tests. `evaluate` must no longer require all 18 runs.
3. Locks are committed by the orchestrator, on wp/sym. In the step-2 command sheet (`docs/reports/WP-S9-state.md`),
   replace each `git add … && git commit` after a lock or `theta.json` stage with "stop and report the file path".
   UNS Part B is registered (not optional): put it in the step-2 order.
4. `py_compile` and pyflakes; commit on wp/sym; no merge to master, no push.

Done line in your final reply: `WP-S9 step 1b done`.

## Step 1c (now, no compute): TSM Clarification C1
`docs/prereg/TSM.md` gained Clarification C1 (master 3477526; identical on wp/sym 68c19da). It answers your step-1b
questions a–e. Two answers differ from what the code does now:
1. **Other early ends.** Only CUDA out-of-memory and I/O errors (`OSError`) are infrastructure failures. Any other
   exception is a code defect: the run is recorded as such, and every later TSM `train` or `evaluate` refuses, as after
   a T0 failure.
2. **Interrupted evaluation.** An interrupted `evaluate` may be restarted from scratch with identical inputs, at most
   twice. The partial output is moved aside unread, and the restart is recorded in the evaluation manifest.

Add tests for both and update the command sheet. For IVD, CVH, UNS and NCR, the same operational rule applies to an
interrupted evaluation job (restart from scratch, partial output moved aside unread); note it in the sheet.
`py_compile` and pyflakes; commit on wp/sym; no merge, no push. Done line: `WP-S9 step 1c done`.

## Step 1d (now, no compute): TSM Clarification C2
`docs/prereg/TSM.md` gained Clarification C2 (master d488b78; identical on wp/sym a5c093a). It answers your step-1c
points 1–6. Code changes:
1. `train --max-attempts N`, default 3.
2. Allocation failures count as infrastructure failures: CUDA out-of-memory, cuBLAS/cuDNN allocation failures, host
   `MemoryError`. Other CUDA runtime errors stay code defects.
3. `evaluate --restart-interrupted --new-evaluation-code` permits a restart whose evaluation `source_hash` differs from
   the interrupted attempt's. The 18 training records must still share one `source_hash`, and the manifest records both
   hashes.
4. Add tests and sheet notes.

Points 2, 4 and 6 need no code beyond what C2 states. `py_compile` and pyflakes; commit on wp/sym; no merge, no push.
After this the TSM failure protocol is closed: any further edge case is decided by the orchestrator when it occurs and
recorded in the report. Done line: `WP-S9 step 1d done`.

## Step 1e (now, no compute): static review of the failure rules, TSM Clarification C3
Review: `docs/review/static_2026-10-27/tsm.md`. It found no break of a registered quantity or decision rule. Findings:
F1 Medium, F3 Low–Medium, F2 and F4–F8 Low. `docs/prereg/TSM.md` gained Clarification C3 (master d335fe7; identical
on wp/sym 3ca4d89). This step reopens the failure protocol once, for these findings only.
1. F1. The first evaluation attempt runs at the training code:
   - With no earlier attempt (in place or moved), require the evaluation `source_hash` to equal `training_source_hash`,
     and refuse `--new-evaluation-code`.
   - `summary.md` always prints both hashes and flags a difference.
   - Add a test of the refusal.
2. F2. `train` refuses a `source_hash` that differs from any record under `runs/` or `attempts/`. Check this in
   `stage_train`, and in `train-one` before `mkdir`. Add a test.
3. F3, F4 (C3). Classify these as infrastructure failures:
   - torch serialization I/O errors;
   - the CPU-allocator failure;
   - `CUBLAS_STATUS_NOT_INITIALIZED` from `cublasCreate`.

   Add test cases with the real torch 2.14 texts, and replace the test's "can't allocate memory" text.
4. F5. On a stopping failure (a T0 numerical failure or a code defect):
   - start no pending run;
   - let the running runs of the same job finish (C1: running jobs may finish, and their outputs are kept);
   - then stop.

   Add a test.
5. F6. Enforce the pause in code:
   - `train` refuses to start while a run that is not final has used its attempts.
   - `--max-attempts N` above 3 needs `--decision FILE`, the orchestrator's recorded decision. The manifest records the
     file's path and its blake2b digest.
   - Add tests.
6. F7. Add an evaluation heartbeat: while `evaluate` runs, touch a file in the evaluation directory at least every
   60 s, so the 300 s rule cannot mistake a live evaluation for an ended one. Add a test that pins `QUIET_SECONDS == 300`.
7. F8 (C3). In the sheet, the evaluation-fix check works at code level.
8. Correct the `--rerun-incomplete` help text (allocation failures, `--max-attempts`).
9. Update the state report and the sheet. Run `py_compile` and pyflakes. Commit on wp/sym; no merge, no push.
Done line: `WP-S9 step 1e done`.

## Step 1f (now, no compute): second review of step 1e, TSM Clarification C4
Review: `docs/review/static_2026-10-27/tsm2.md`. It found that the process management (F5), the pause (F6), the
one-code rules and the heartbeat behave as specified, and it found one classification defect.
`docs/prereg/TSM.md` gained Clarification C4 (master d6ceb6a; identical on wp/sym 76a00b3).
1. **Finding 1 (C4).** Classify the torch checkpoint writer's `RuntimeError` texts "File <path> cannot be opened." and
   "Parent directory <dir> does not exist." as I/O errors, hence infrastructure failures.
   - Add table cases: the `RuntimeError` gives infrastructure, and the same text in a `ValueError` gives `code_defect`.
   - Update the sheet sentence and the `--rerun-incomplete` help.
2. **Finding 2 (C4).** A first evaluation attempt at a code other than the training code is accepted only with
   `--new-evaluation-code --decision FILE`. A first attempt means that neither `evaluation/` nor
   `attempts/evaluation-attempt*` exists.
   - The manifest records the decision file's path and blake2b digest, and both code hashes.
   - Without `--decision` the current refusal stays. Add a test.
3. **Notes.**
   - (a) The stopped train manifest records the stopping child's exit code and log path.
   - (b) The sheet says that a train manifest left at "running" means that the job was killed, so read the run states.
   - (f) The heartbeat thread's `join` gets a timeout of 2 × `HEARTBEAT_SECONDS`.
4. Update the state report. Run `py_compile` and pyflakes, with bytecode kept out of the worktree. Commit on wp/sym; no
   merge, no push.
Done line: `WP-S9 step 1f done`.
