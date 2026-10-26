# wk-p1l continuation card — r32 main running; r128 Pilot-2 locked

Worktree /work/xg24i002/x10041/LoopLora-wt/p1l, branchwp/p1l. User assigned
WP-P1L; finish docs/reports/P1L.md and only then reply exactly
`WP-P1L done: docs/reports/P1L.md`. Do not call this task complete prematurely.
No Goal exists; Codex thread01a17999-7c19-7ac2-a6d2-cf6aee2583c0, wakeup message
mode authorized by the user. No tests/training/evaluation/analysis on login.
All application runtime goes through scripts/submit.py. Control scripts
launch_p1l_main.py/p1l_job_ledger.py are stdlib login-safe. Never import torch
on login. Read docs/README.md, task/prereg, this card before continuing.

Armed wakeups (message mode,300s polling, no Goal):
- r128 smoke3716021: .job-wakeup/27808c33-f6d6-4f4d-b9d7-ca4e673c498b,
  acknowledged; both smoke jobs verified complete, exit0.
- r32 pilot3715965: .job-wakeup/ee3ad75d-7773-4f90-92d8-b0bc5660b157,
  acknowledged; all seven jobs exit0 and all14 candidate records complete.
- r128 pilot3716169: .job-wakeup/d921e843-f2a6-44ff-82e7-83cb3267c07f,
  acknowledged; all five jobs exit0 and all ten candidates complete.
- r32 main3716305: .job-wakeup/a101bbc0-d502-4bf4-8978-4e613b043510,
  worker PID2276624; verify all19 jobs/22 run artifacts on wake.
Read the event state and acknowledge with the exact argv from the wakeup
message. Do not register duplicates or cancel the other pending watcher.

## Accepted decisions

- Design-lockea44bb6 precedes all training; J-probe A1b5d923c precedes pilots:
  name-contrastive gradients primary, raw secondary, unit-normalize after mean
  over512 fixed val questions. Final-step faithfulness vs actual name argmax.
- Master Amendment B2 (merged7bfaf16) supersedes B1 and the original total cap.
  S17000, Pilot5666, no study-total cap. Every job<=8nodes x2h, must finish
  before2026-10-28 09:00JST. B2 explicitly accepts the900-step timing smokes.
- Single GPU process, sequential backbone reuse, prefetch2. Oracle seed17 fixed.
  Main will pack B4/B3 sequentially; other trained variants separate jobs.
- Confirmation is HELD. Orchestrator has NOT decided WP-L4; it promises to
  message the decision. B3 supersedes the old writer-count CI rule: k counts
  oracle-r32,oracle-r128,plus any learned condition; confidence=1-.05/k fixed
  before the first confirmation. ALL conditions must finish training first.
  Proceed with oracle pilots/main while holding all test/confirmation data.
- B3 merged masterdf21654 at04:33JST: parallel r128,5readers x3seeds, its own
  Pilot-2; B0/B1/B2 shared fromr32. Configs configs/p1l/main-r128, outputs
  R/r128 and reports/p1l/r128. Design locked7503f54 before any r128 training.
- B4 merged master379115d: remove B3's B8r32-B4r128 and M1r32-B4r128 contrasts.
  B8r128-B8r32 stays descriptive; no gate/run/CI change. Add stored/trainable
  adapter counts and measured inference time per question for every reader/rank.
  Protocol locked in reports/p1l/design-lock.md; evaluate_p1l_main.py writes
  inference.json for each run and finalizer writes inference.csv. Fixed16val
  episodes/128questions,5warmups+20CUDA-synchronized score+argmax calls, trained
  depth, supports prepared outside timing; no audit capture/hashing. Includes
  score-path CE and reports amortized batched time, never serial latency.
- Orchestrator wording: call the diagnostic a **J-lens-inspired name-gradient
  probe**, per master docs/reports/lit_check.md WP-T6; it is not the published
  averaged-transport lens or sparse J-space. Numerics unchanged.

## Observed state

Raw R=/work/xg24i002/x10041/LoopLora/runs/p1l/main.
Smokes3715854(mg0037)/3715855(mg0045) exit0, all7 paths900steps; scores never
read. Sizing3715950(mg0014) exit0, committed reports/p1l/smoke-sizing.json in
1dc8a76: longest.26584s/step; S17000 fits1.5margin; pilot+main~23.48nodeh.
Old preflight/S sizing artifacts are historical; B1 cap is superseded.

Pilot jobs are all source1dc8a7606f3c2fa5a5358301fc4a383030d01af6:
3715959B8,3715960M1-L4,3715961B4,3715962B5-L4,3715963B3-L4,
3715964B1,3715965B2. Each runs lr.001/.003 sequentially, seed17,5666steps,
1000val episodes, hops1/2/3 at trained depth; selection by full-vocab EM,
ties lower lr (B1/B2 selected in their trained context-retained protocol).
R/pilot-jobs.json retains exact job IDs/log/spec/source. R/pilot/<LABEL>/
<LABEL>-lr0.001 or -lr0.003 contain records/checkpoints/validation samples.
All seven pilots completed, boundary verified. Compute selection3716286 onmg0019
passed, source716110a. reports/p1l/lr-S-lock.json fixes S17000, M1-L4 lr.003;
B8/B4/B5-L4/B3-L4/B1/B2 lr.001 (ties lower). All22 configs updated before
main submission, committed d644b9b. Pilot full-vocabulary pooled val EM is0.033–0.133% for oracle/
no-memory readers across candidates; context-retained B1/B2 is99.33–99.83%.
These are validation results only; no change to design, gates or main schedule.

R32 main launched source d644b9ba3b30f7fcc2f13020272f457ee60aae40,
19jobs/22runs, all1node, requests5min–1h55min. R/main-jobs.json retains
receipts; R/training-manifest.json maps all22 runs. Raw R/main/<GROUP>/<RUN>.
Seed17:3716288B8,3716289M1,3716290B5,3716291B4+B3,3716292B1,3716293B2.
Seed29:3716294B8,3716295M1,3716296B5,3716297B4+B3,3716298B1,3716299B2.
Seed43:3716300B8,3716301M1,3716302B5,3716303B4+B3,3716304B1,3716305B2.
B0 untrained checkpoint:3716306, passed onmg0019, zero trainable parameters,
hash6af601799a569397330e74196c99ebe8. Never rerun the launcher blindly or duplicate
submitted groups. All main run records must complete before confirmation;
main training performs no quality evaluation. Wakeup on3716305 (late long job)
is armed above; inspect all19 receipts/22 artifacts on wake, not only that job.

Probe full cache3715870, source d276308, R/jlens/directions.pt,512 fixed val
questions only, frozen model, completed onmg0865 in932.28s. Small metadata
copied to reports/p1l/name-gradient-probe.json; file hash9a76963e9587ae51db52b29188c004d8.
Shared across ranks: residual space/model/questions are identical. Retain raw
direction tensor. No confirmation used.

R128 support: OracleRandomWriter/config/train now accept32/128, default32
unchanged; signatures record actual rank and reject cross-rank writers.
Core/exact-checkpoint tests3716000 passed34; B3 gate/config tests3716014
passed7; r128 exact final-step replay3716027 passed3; global missing-rank/
wrong-rank/incomplete-other-condition gate3716028 passed1.
B4 inference timing/count wrapper3716084 passed1 onmg0025 (actual128-question
GPU batches,20timed calls; no confirmation data).
R128 preflight3716009(mg0022) exit0, all5paths20steps, peak<=7.70GB;
sizing3716015 gives
S17000/Pilot5666, no reduced-steps flag, slowest.20183s/step. Committed
reports/p1l/r128/preflight-sizing.json and design. Full10% smokes1700steps:
3716020(B8,M1-L4),3716021(B4,B5-L4,B3-L4), source16bef08. BOTH exit0.
Full-smoke sizing3716162 (mg0024), reports/p1l/r128/smoke-sizing.json,
committed e955765: S17000/Pilot5666, slowest .199397s, no reduced-steps flag,
projected Pilot-2+main14.408nodeh.
R128 Pilot-2 launched source e95576527c46cc1798d97c5af83384fc62bd0e06:
3716168B8,3716169M1-L4,3716170B4,3716171B5-L4,3716172B3-L4.
Receipts R/r128/pilot-jobs.json, two sequential lrs per label under
R/r128/pilot/<LABEL>/<LABEL>-lr0.001 or -lr0.003. Requested45–65min,
started04:59JST. All five completed, exit0; ten candidate records and stage
records complete, validation boundary verified. Compute selection3716392 on
mg0021 passed, source4bdb475. R128 lr-S lock fixes S17000, B3-L4 lr.001
(tie), B8/M1-L4/B4/B5-L4 lr.003. All15 configs updated before main submission.
Selected pooled val hops1/2/3 full-vocabulary EM: B8 5.667%, M1 8.133%,
B4 5.900%, B5 5.633%, B3 0.0667% (3000 queries each). Selection follows the
registered rule; no design or endpoint changes. No confirmation accessed.

Scheduler05:39JST: regular project limit16 active jobs; queued r32 seeds43
have placeholder/predicted starts after maintenance. Detail3716304 states
the group njobs-g limit as the blocker. Fourteen r32 main runs and two WP-L4
jobs already run, with expected slot releases around06:10–07:00JST. Recheck
queued jobs at the main wakeup and do not assume those estimates prove a run
will start. All work must still finish before2026-10-28 09:00JST.

Validation-only full-evaluation timing acceptance: script
scripts/benchmark_p1l_evaluation.py loads the immutable completed M1-L4
lr.001 r32 pilot checkpoint and fixed probe cache. Uses val[0,200),
allow_confirmationFalse, smokeTrue; primary+typed,swap,readout,timing only.
Never read its quality artifacts. Job3716191(mg0026) failed before readout
completion: faithfulness counter shadowed episode count and truncated later
batch slices. Fixed b88a742 with a multibatch/partial-tail regression.
Regression job3716212 passed onmg0030, exit0. Rerun3716213 passed onmg0031,
revealed max2.29e-5 FP32 error from recomputing only last-token states in another
matrix shape. Fixed2823982: capture actual alpha B z_s during prefix replay.
No direction, endpoint or selection rule changes. Exact capture+replay tests
3716225 passed5 onmg0023; full-model validation-only rerun3716226 passed onmg0022.
R/evaluation-preflight-v3/timing.json reports captured_bitwise_equalTrue;
its recomputed_bitwise_equalFalse documents why manual recomputation was removed.
Compact timing-only copies reports/p1l/evaluation-timing-M1.json and
evaluation-timing-B1.json retain source checkpoint hashes and compute evidence.
M1 total72.89s for200episodes: primary12.12s,typed10.03s,swap43.51s,
readout5.21s. For later confirmation sizing multiply episode-dependent phases
by10 for2000episodes, add load/reserve and1.5margin. Three M1 seeds fit one1h
evaluation job under this measurement. Baseline both-protocol total18.05s.
Baseline timing3716223 passed onmg0030, primary+typed, both protocols,200val
episodes, total18.05s. Wrong-source submission3716221 endedexit2 in8s (central
HEAD lacked script); replacement uses explicit3801bb2. Never use --commit HEAD
in submit.py: it resolves refs in the CENTRAL checkout. Use explicit worker
hash or wp/p1l. Main launcher already resolves its own ROOT hash explicitly.
Timing acceptance quality artifacts have not been read. R32 pilot scores were
first consumed by registered compute selection3716286 after all seven finished.

Passed: readout/oracle33checks3715857; gates/authorization4checks3715889;
swap-wrapper1check3715942. Failed development jobs, included in ledger:
3715842 sparse timing-log check (fixed3715847),3715928 counterfactual grouping
by rewired hop (fixed; source episode/query/hop now retained and new hop kept
separately; unchanged regression passed3715942). No further blanket retests.

## Immediate next actions

1. Validation-only evaluation acceptance is complete; never read its raw scores.
2. Both pilot wakeups acknowledged; r32 lock/configs committed and main running.
   Commit r128 lock/configs before launching its main training.
3. R32 main3716305 wakeup armed. At either wake preserve
   the other pending event; independently advance whichever rank is ready.

## Next actions after each rank's pilot wakeup

1. Read/ack wakeup state per skill; inspect ALL7 pilot jobs and records.
   A FINISH alone is insufficient. Require exit0 and record.statuscomplete
   for both rates per label. If one remains running, avoid duplicate jobs;
   arrange wakeup for that exact remaining job. Keep confirmation held.
2. On compute through submit.py, run:
   `scripts/analyze_p1l_main.py select --root R/pilot --sizing ABS/reports/p1l/smoke-sizing.json --out ABS/reports/p1l/lr-S-lock.json`.
   Commit generated lr-S lock; update all22 configs/p1l/main/*-s*.json with
   S17000 and each label's selected lr (B0 lr unchanged; no training), then
   commit BEFORE any main training. All routines use the same S.
   For r128 use --rank 128, R/r128/pilot, its smoke-sizing and
   reports/p1l/r128/lr-S-lock.json; update its15configs in main-r128. Main
   launcher also needs --rank 128 and R/r128 root. No B0/B1/B2 repetition.
3. On LOGIN, launch `scripts/launch_p1l_main.py main --root R --sizing reports/p1l/smoke-sizing.json`.
   This submits19 jobs (18 groups for21 trained runs plusB0), all through
   submit.py with1.5margin. Some queue because project limit16 running.
   R/training-manifest.json maps all22 runs. Do not rerun launcher blindly;
   its receipts guard submitted groups. Record actual node-hours per job.
4. Use wakeup message mode for main wait. No implicit Goal creation.
   Endpoint from existing daemon: unix:///home/x10041/.codex/app-server-control/app-server-control.sock.
   Watcher: /work/xg24i002/x10041/.codex/skills/miyabi-pbs-goal-wakeup/scripts/job_wakeup.py.
5. After ALL conditions' main jobs complete AND explicit WP-L4 decision arrives,
   commit authorization JSON {status:allowed,gated_conditions:k,
   condition_manifests:{oracle-r32:ABS_MANIFEST,oracle-r128:ABS_MANIFEST,...},
   orchestrator_decision:text}. Global gate checks matching condition/rank
   labels, complete checkpoints and fixed S/lrs for every listed condition.
   Evaluate each checkpoint once with scripts/evaluate_p1l_main.py; it requires
   --manifest R/training-manifest.json --authorization AUTH --out NEW_GROUP
   --runs RUN_NAMES --directions R/jlens/directions.pt. Batch short jobs,
   measure timing before large eval packs; M1 depth1..8 + swaps1/4/8 is slower.
   Creates primary/typed samples plus readout/faithfulness/swap artifacts.
   Fill evaluation manifest {runs:{RUN_NAME:EVAL_DIR}} (all22).
6. Finalize on compute: scripts/finalize_p1l_main.py --manifest R/training-manifest.json
   --evaluations EVAL_MANIFEST --authorization AUTH --out ABS/reports/p1l.
   Audits all sample metrics, source pairing, oracle codes and memories;
   computes registered G1/G2, secondary M1 contrasts and specificity, per-seed
   values, accuracy/counts/readout/faithfulness/swap tables and figures.
   Inspect end-to-end audit output; fix concrete issues without changing gates.
   Repeat per rank with separate report directory; then run compare_p1l_ranks.py
   for B4's retained descriptive B8 rank contrast. The two cross-rank single-read
   comparisons are removed. Report adapter counts/inference.csv; cite any unequal S.
7. Refresh costs on login: `/usr/bin/python3 scripts/p1l_job_ledger.py --out reports/p1l/jobs.json --require-final`.
   Finish <=1page docs/reports/P1L.md with actual results, validation jobs,
   deviations(A1/B1/B2, valid900-step smoke, probe wording), cost, limitations.
   Also write reports/decisions/P1L.md per prereg. Commit own paths only.
   Do not merge/push other branches. Reply final exact done line only at end.

If WP-L4 selects a writer and orchestrator adds the learned condition, run that
condition under its addendum with the same staged procedure. Await concrete
writer/checkpoint/interface instructions; do not invent a learned condition.
