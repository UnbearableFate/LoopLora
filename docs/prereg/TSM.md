# TSM (F10) — terminal-semantics training of the iterative reader, pre-registered

Registered 2026-10-27 12:44 JST, when the orchestrator committed this file to master, before any lock, smoke or run of
this study. The draft text is unchanged except where "Decisions at registration" (last section) says otherwise.

Draft by sn-sym, 2026-10-27, on the login node: no run, no training, no evaluation, no result read. Registered at
the time the orchestrator commits this file to master, before any run below. Known then:
- JS (`docs/reports/JS.md`): the Jacobian spectral-radius penalty reduced but did not remove the overshoot of
  random-depth-trained M1 readers (S = L16 − L4 on hops 1–3: oracle_random −11.3 pp without and −2.8 pp with the
  penalty, joint_fact −3.0 and −1.8 pp) and cost accuracy on hops 4–8 at L16 (−5.5 and −1.4 pp); neither writer met
  the remedy rule.
- P2-S-F1 (`docs/reports/P2S-F1.md`): M1 trained at the fixed depth 4 beats the mixed-depth M1 at L4 on hops {2, 3}
  (oracle_random +8.2 pp, joint_fact +3.5 pp), and B8 beats M1-L4 by +3.2 and +5.5 pp.
- P4-S-L (`docs/reports/P4S-L.md`): no halting head repaired hop 1 on the frozen readers.
- rv-fable R1/R2 and rv-astra proposal 6 suggest stay training and a content-gated sticky update; neither has been run.

## Question
Can the reader be trained so that a query's answer stays decodable after its chain has ended (no overshoot at large
L), without losing the longer walks and while keeping the multi-read gain over a single read?

## Design
- Common setting: `t2_chain_end`, r = r_mem = 32, reader M1 (`iterq`, ReLU), writers `oracle_random` and
  `joint_fact` (trained jointly with the reader, as the released runs), three seeds 17/29/43, training hops 1–3,
  15000 steps, batch 64, the released M1 recipe of the same writer (lr 0.003, AdamW with weight decay 0.01, 20
  warm-up steps and gradient clipping 1.0, all taken from the released configuration; the lock checks steps, lr, batch
  size, depths {1, 2, 4}, hops and task of every released configuration). No learning-rate search and no retry
  (registered).
- Arms (18 new runs = 2 writers × 3 arms × 3 seeds; per (writer, seed) the three arms share initialisation seeds, the
  data stream and the depth stream):
  - **T0**: M1-L4 retrained under the same code version with the unchanged `looplora.sym.train.train`: the released
    configuration with `train.depths = [4]`. This is the P2-S-F1 M1-L4 recipe: the author compared the run
    configurations of all six P2-S-F1 M1-L4 runs of these writers with the released M1 configurations and found
    `train.depths` the only difference outside the eval block and the pretrain block (unused by joint writers).
  - **T1**: fixed training depth 8, and the answer loss at every step s ≥ hop ("stay"): for a query of hop h, the mean
    over steps s = h…8 of the candidate cross-entropy of the answer decoded from z^s (restricted to the episode's
    candidate entities, as in `train`); mean over queries. Everything else as T0; the T0 loss is the plain cross-entropy
    at the last step.
  - **T2**: T1 plus the content-gated sticky update z^{s+1} = z^s + g_s (f(z^s) − z^s), f the reader's usual update
    (u + ρ φ(R z + U M Q z)), g_s = σ(a · ‖U M Q z^s‖ / ‖u‖ + b); scalars a, b initialised to (0, 4); two extra
    trainable parameters.
- Reference: the released **B4** (single read, trained at L = 1) of the same writer, rank and seed, evaluated at L = 1.
  The B4 and M1 checkpoints used for the training recipes are sealed in the design lock.
- Evaluation: split `test`, **eval seed 999007** (never used before), 2000 episodes, one stratified query per hop
  1–8, batch 64; T0/T1/T2 at depths 1, 4, 8, 16; B4 at 1. All arms and B4 see exactly the same episodes.
- Order of operations: `lock` → the orchestrator commits the design → `smoke` (seed-17 runs of all six
  writer × arm cells, 200 steps, concurrently, then a validation evaluation, seed 4242, 200 episodes: timings, memory,
  gate profile; nothing is selected and no endpoint of it is used) → `train` (concurrent child processes; the jobs
  may split the 18 runs by seed or writer so that every job is at most 1 h; the runs are independent of the split)
  → `evaluate` (requires all 18 runs complete and trained under the registered configuration) →
  `scripts/analyze_tsm.py`, which audits the Git lock, the seals and the structure before any statistic.
- Statistics: pooled paired episode bootstrap over the three training seeds (whole-episode joint resampling), 2000
  draws, base seed 2718, confidence 1 − 0.05/4 (two writers × two arms); per-seed values reported.

## Endpoints (per writer and arm T1, T2; percentage points)
- **stay**: accuracy at L16 minus accuracy at L4 of the arm on hops {1, 2, 3}; the condition holds if |difference|
  ≤ 1 pp (point estimate).
- **far**: arm at L8 minus T0 at L8 on hops {4, …, 8}; holds if the difference ≥ −1 pp (point estimate).
- **gain**: arm at L4 minus B4 at L1 on hops {2, 3}; holds if the difference ≥ +3 pp with the interval above 0.
- An arm **passes** only if all three conditions hold for both writers.
- Reported, not gating: hop 1 at L8 and L16 against the arm's own L1; the arm against T0 at L4 on hops {2, 3} and at
  L16 on hops {4, …, 8}; accuracy by depth (1, 4, 8, 16) and hop group {1}, {2, 3}, {4, 5}, {6, 7, 8}, all, with 95%
  intervals; the mean gate per hop and step for T2; training loss, step time, parameter counts.

## Decision rule (as in plan F10)
For an arm: hops 1–3 at L16 and at L4 differ by ≤ 1 pp; the drop on hops 4–8 against T0 is at most 1 pp; the depth gain
over the same-rank B4 is ≥ 3 pp (CI > 0).

## Use of the results (stated before results)
- An arm that passes for both writers: the paper reports that terminal-semantics training <T1/T2> removes the overshoot
  of these r = 32 readers without the registered costs, with the three intervals and the scope (two writers, three
  seeds, trained hops 1–3). If both pass, the simpler arm is the one stated; the gate is reported as additional.
- No arm passes: the paper states that neither stay training nor the sticky gate removed the overshoot under this
  protocol, keeps fixed-depth training (M1-L4, B8) as the remedy found, and does not generalise from one learning rate.
  No follow-up is registered (plan §6).
- A run that fails to train (non-finite loss) is recorded as failed and reported; nothing is retried or re-tuned. The
  evaluation then needs the registered 18 runs and does not start: the failure is reported to the orchestrator, and
  any change of protocol needs an explicit amendment before the confirmation set is opened.

## Flagged interpretations and additions (the plan does not fix these)
1. The plan's T0 is "M1-L4 retrained under the same code version". It is implemented as the released recipe with the
   single fixed depth 4 (the P2-S-F1 recipe); the P2-S-F1 checkpoints are not reused as T0.
2. The plan states the three conditions without depths or writers. Registered here: stay uses L4 and L16 (the plan's
   depths); far uses L8 for the arm and for T0 (the depth at which T1/T2 are trained and all hops 4–8 are reachable;
   rv-fable R2 also evaluates hops 4–6 at L8), L16 is reported; gain uses L4 against B4 at L1 (the depth of the
   P2-S-F1 and IS multi-read contrasts); an arm passes only if it passes for both writers (rv-astra proposal 6: "both
   writers"), with the interval level corrected for the four arm × writer comparisons.
3. The stay loss weights every query equally and averages its steps s = hop…8; the plan does not give the weighting.
4. T2 follows the plan's formula without the PonderNet-style step cost and the fixed-prior depth supervision that
   rv-fable R1 mentions, and with the plan's initialisation made explicit: a = 0, b = 4 (g ≈ 0.98 at the start, so the
   initial dynamics are close to, but not equal to, the plain update).
5. T1 and T2 train at the fixed depth 8 (the plan: "L = 8 训练"), not at mixed depths.
6. The released lr (0.003) is used for every arm; the arms with a different loss or depth could prefer another lr, so a
   failure does not show that no recipe works (rv-fable R2, risk).

## Limits
- Two writers with shortcut-free codes and three training seeds; training hops 1–3 only; r = 32; intervals condition on
  the three seeds. T1 and T2 change loss, depth and (T2) dynamics together, so the contrast T2 versus T1 isolates the
  gate, and T1 versus T0 isolates stay loss and depth jointly.
- Cost: see `docs/reports/WP-S9-state.md` (estimate 1.2–1.8 node·h from the measured cost of P2-S-F1 and JS; the smoke
  fixes it). Each job is one node and at most 1 h; `--scale-up docs/prereg/TSM.md` (at most 2 h) only if the smoke
  shows that a seed's runs cannot be split further.
- Code: `src/looplora/sym/tsm.py`, `scripts/run_tsm.py`, `scripts/analyze_tsm.py`; tests `tests/test_tsm.py`,
  `tests/test_tsm_scripts.py` (written, to be run on a compute node before the lock). The T0 path is the unchanged
  `train()`; a test shows that `fit_tsm` without stay loss and gate equals the existing `_fit` loop bitwise.

## Decisions at registration (orchestrator, 2026-10-27)
- Flagged items 1–6 are accepted as written.
- **Both writers.** An arm passes only if it passes for both writers. If an arm passes for exactly one writer, the paper
  reports it as a writer-specific result, not as a remedy, and the "no arm passes" statement is worded per writer.
- **Training failures.** This replaces the draft's rule that the evaluation needs all 18 runs.
  - An infrastructure failure is a job ended by walltime or by a node or I/O fault before it completes, with no
    non-finite loss. The identical configuration (same code, seeds and data stream) is rerun, at most twice.
  - A non-finite loss in a T1 or T2 run is a registered training failure. That arm fails for that writer, with no retry
    and no re-tuning. Its completed runs are evaluated and reported descriptively, and the other arms are evaluated as
    registered. `evaluate` is adapted to this before the lock.
  - A non-finite loss in a T0 run stops the study before the confirmation set is opened, because "far" then has no
    reference. Any continuation needs an explicit amendment.
- The `--scale-up docs/prereg/TSM.md` allowance is accepted: at most 2 h, and only if the smoke shows that a seed's runs
  cannot be split further.
- **Locks.** The orchestrator commits each design lock on wp/sym, the branch whose worktree holds the lock files,
  after checking it against this text. sn-sym stops after each lock stage and reports the file path.

## Clarification C1 (2026-10-27 14:24 JST, before any TSM lock, smoke or run; from sn-sym's step-1b questions)
These rules fill gaps in the failure rules above. They change no endpoint, threshold or decision rule.
- **Three failed attempts.** A run whose three attempts all end in infrastructure failures pauses the study; no further
  TSM job is submitted. The orchestrator diagnoses the cause. A remedy that changes no registered quantity (walltime,
  job split, concurrency, node, I/O path) permits further identical attempts, and each is reported. Anything else needs
  an amendment before the confirmation set is opened.
- **Non-finite gradient norm.** A non-finite gradient norm with a finite loss is a numerical failure, treated as a
  non-finite loss. It is a property of the training dynamics, and an identical rerun reproduces it.
- **Other early ends.**
  - CUDA out-of-memory and I/O errors are infrastructure failures. The remedy may lower the number of concurrent runs
    per job.
  - Any other exception is a code defect. The study stops. Because every TSM run must use one code version, the fixed
    code retrains every TSM run, before the confirmation set is opened.
- **Interrupted evaluation.** An `evaluate` job that ends before it completes is restarted from scratch with identical
  inputs, at most twice. Its partial outputs are moved aside unread.
- **One code version.** "T0 retrained under the same code version" means one `source_hash` across all 18 runs, and
  every TSM training and evaluation job runs at one CODE commit. Smoke runs are exempt and never analysed.
- **After a T0 failure.** TSM jobs that are already running may finish, and their outputs are kept. No new TSM job is
  submitted until the orchestrator decides.

## Clarification C2 (2026-10-27 15:12 JST, before any TSM lock, smoke or run; from sn-sym's step-1c points)
These rules refine C1. They change no endpoint, threshold or decision rule.
- **More than three attempts.** After the orchestrator's recorded decision under C1 ("Three failed attempts"),
  `train --max-attempts N` grants further identical attempts. The default is 3.
- **Allocation failures are infrastructure failures.** These are CUDA out-of-memory, cuBLAS or cuDNN allocation
  failures, and host `MemoryError`. Other CUDA runtime errors count as code defects. If the orchestrator's diagnosis
  finds a hardware fault, that is recorded, and the run is retried as an infrastructure failure.
- **Defects in the evaluation code.** If the evaluation code (not the training code) has a defect:
  - The 18 trained runs are kept.
  - The orchestrator checks that the fix changes no file that the training path uses.
  - The evaluation restarts from scratch at the fixed commit (`evaluate --restart-interrupted --new-evaluation-code`).
  - The partial output is moved aside unread, and both code hashes are recorded.

  This refines C1's "one code version": the training jobs share one CODE commit, and the evaluation job may use a
  later commit only under this rule.
- **What a restart reads.** A restart reads only the interrupted attempt's `manifest.json` (status and inputs), never
  its samples or statistics.
- **Errors before a run directory exists.** Such an error leaves no record. The operator reports its traceback, and the
  orchestrator classifies it under C1 and C2.

## Clarification C3 (2026-10-27 16:58 JST, before any TSM lock, smoke or run; from the static review of the failure rules)
These rules refine C1 and C2. They change no endpoint, threshold or decision rule.
- **Allocation failures (C2).** These are infrastructure failures:
  - torch's CPU-allocator failure (a `RuntimeError` containing "DefaultCPUAllocator: not enough memory"), which
    counts as a host allocation failure;
  - `CUBLAS_STATUS_NOT_INITIALIZED` reported by `cublasCreate` (cuBLAS could not create its handle, usually on a full
    GPU), which counts as a cuBLAS allocation failure.
- **I/O errors (C1).** torch's checkpoint serialization raises errors when a file cannot be written or read
  ("PytorchStreamWriter failed", "PytorchStreamReader failed"). These are I/O errors, hence infrastructure failures.
- **Bounded exposure.** A code defect misread as one of the failures above repeats identically and reaches C1's
  three-attempt pause. If the diagnosis then finds a code defect, C1's code-defect rule applies.
- **Evaluation-code defects (C2).** The orchestrator checks the fix at code level, not file level, and records the
  check with its decision:
  - The fix must change no function, class or module-level statement that the training path executes. The training
    path is `train`, `train-one` and everything they call, including shared helpers.
  - The check cannot be file level because evaluation and training share files (`scripts/run_tsm.py`,
    `src/looplora/sym/tsm.py`); a file-level check could never pass.
