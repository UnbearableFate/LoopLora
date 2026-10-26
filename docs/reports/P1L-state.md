# wk-p1l continuation card — r32 Pilot-2 and r128 smoke running

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
  worker PID2012956; also verify3716020 on wake.
- r32 pilot3715965: .job-wakeup/ee3ad75d-7773-4f90-92d8-b0bc5660b157,
  worker PID2041468; also verify the other six pilot jobs on wake.
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
3716020(B8,M1-L4),3716021(B4,B5-L4,B3-L4), source16bef08. BOTH running.
No r128 Pilot-2 has been launched; no pilot quality score of either rank read.

Passed: readout/oracle33checks3715857; gates/authorization4checks3715889;
swap-wrapper1check3715942. Failed development jobs, included in ledger:
3715842 sparse timing-log check (fixed3715847),3715928 counterfactual grouping
by rewired hop (fixed; source episode/query/hop now retained and new hop kept
separately; unchanged regression passed3715942). No further blanket retests.

## Immediate next actions after r128 smoke wakeup

1. Read/ack the exact wakeup event per skill. Check BOTH3716020/3716021 exit0
   and all5 complete records before reading timing. Never read smoke losses.
2. Through submit.py on compute: scripts/analyze_p1l_main.py size --rank 128
   --root R/r128 --out ABS/reports/p1l/r128/smoke-sizing.json. Collects smoke-a/b.
   S17000 if1.5margin fits; otherwise flags a lower multiple-of500. Commit.
3. On login: scripts/launch_p1l_main.py pilot --rank 128 --root R/r128
   --sizing reports/p1l/r128/smoke-sizing.json.
   Five jobs, two lrs each. Source and log receipts R/r128/pilot-jobs.json.
4. If r32 pilots still running, retain their armed3715965 watcher; if no
   watcher exists, register one exact job in message mode and return idle.
   For r128 pilots arrange an independent watcher on a late/slow job.

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
