# WP-P1L handover — Codex worker closed

Successor: Claude Sonnet-5.5 subagent under the orchestrator, resume after
2026-10-28 22:00 JST. This is a handover, NOT study completion. No compute was
started during handover. All Codex wake-ups were cancelled or already acknowledged;
do not resume/re-arm Codex thread01a17999-7c19-7ac2-a6d2-cf6aee2583c0.
Fresh status/artifact snapshots below are observations, not promises of completion.

## Binding decisions and locations

Read docs/README.md, docs/tasks/WP-P1L_main.md, docs/prereg/P1L.md through B5,
reports/p1l/design-lock.md, reports/p1l/README.md and this handover.
Worktree/branch: /work/xg24i002/x10041/LoopLora-wt/p1l, wp/p1l.
Set these shell variables in the successor's own session:

```bash
P1L_W=/work/xg24i002/x10041/LoopLora-wt/p1l
P1L_R=/work/xg24i002/x10041/LoopLora/runs/p1l/main
cd "$P1L_W"
```

R below means P1L_R. Authoritative mutable raw manifests are
R/training-manifest.json (22 runs, oracle-r32) and
R/r128/training-manifest.json (15 runs, oracle-r128). Committed copies and
main-jobs.json receipts are in reports/p1l/ and reports/p1l/r128/ for handover.
Use the RAW paths recorded in confirmation-lock.json for evaluation authorization.
Every run directory contains config.json, record.json, train_log.jsonl,
checkpoint.pt and, for oracle readers, writer.pt. The exact37 run/checkpoint paths
and observed states/hashes are in reports/p1l/handover-artifacts.json and the table below.

- B5/master6de902b merged by1940030. NO learned writer was selected; no learned
  condition and no P3. Exactly two oracle rank conditions; k=2, gate CIs97.5%,
  locked BEFORE confirmation in reports/p1l/confirmation-lock.json.
- Confirmation is already authorized, without another permission request, AFTER
  2026-10-28 22:00 JST AND every run of BOTH conditions has completed training.
  No test/confirmation data have yet been evaluated. Do not release early.
- /etc/motd: normal operation ends10/27 08:00; HPC Challenge until10/28 09:00,
  then maintenance until10/28 22:00. The old09:00 finish deadline is superseded
  by the orchestrator/B5 post-maintenance schedule. Each job remains<=8nodes x2h;
  actual implementation uses1node/job. B2 removed all study-total caps.
- Both ranks S17000, Pilot-2 floor(S/3)=5666, seeds17/29/43, oracle seed17.
  R32 selected M1-L4 lr.003, all other trained readers lr.001 (lockd644b9b).
  R128 selected B8/M1-L4/B4/B5-L4 lr.003, B3-L4 lr.001 (lock1d1bbfd).
  Configs configs/p1l/main/ and main-r128/ are locked; do not change them.
- E12 chains(8,4), train hops1/2/3, batch64, prefetch2, layer14/o_proj,
  Qwen3-0.6B-Base pinned da87bfb608c14b7cf20ba1ce41287e8de496c0cd, bf16 frozen
  backbone/fp32 adapters. B8 primary; M1-L4 shared secondary; B4 single read;
  B5-L4 fixed query; B3-L4 no memory. B0/B1/B2 shared from r32, not repeated.
- Only SAME-RANK B8-versus-B4 comparisons gate. B4 removed the cross-rank
  single-read contrasts. B8(r128)-B8(r32) remains descriptive only.

## Wake-ups: all disabled for this closing session

Both pending events were cancelled with job_wakeup.py cancel during handover:
187e742c-1c41-4da5-8700-cc998c31ce5e (job3716619) and
a101bbc0-d502-4bf4-8978-4e613b043510 (job3716305).
The other five events are acknowledged, not armed. Verified no waiting/ready/
sending/delivered/uncertain events remain; snapshot reports/p1l/handover-wakeups.json.
PBS jobs themselves were NOT cancelled during handover. A successor may use its
own Claude background wake-up workflow if necessary; never target this Codex session.

## What happened at the operation boundary

At06:51JST, with explicit orchestrator authorization, cancelled four QUEUED,
unstarted packs and submitted eight single runs using the ORIGINAL source hashes,
configs, seeds, S and lrs. Raw main-jobs.json files retain cancelled receipts under
superseded_jobs; training manifests point to replacement paths.

| Rank/seed | Cancelled pack (never ran) | B4 replacement | B3-L4 replacement |
| --- | --- | --- | --- |
| 32/43 | 3716303 | 3716612 | 3716613 |
| 128/17 | 3716399 | 3716614 | 3716615 |
| 128/29 | 3716403 | 3716616 | 3716617 |
| 128/43 | 3716407 | 3716618 | 3716619 |

3716612–3716617 started06:52;3716619 started06:58:28. All seven requested ends
are<=07:56:49, within07:58 at1.5x measured time including load reserve. Verified
their running records have unchanged configs. 3716618 did not get a slot before
its07:00:22 cutoff and remains queued. Evidence dedicated-boundary-splits.json /
dedicated-boundary-starts.json. Do NOT rerun split_p1l_packed_jobs.py.

The three intentionally deferred runs, already submitted, are:

| Run | PBS ID | Intended window | Raw run directory |
| --- | --- | --- | --- |
| r32 B1-s43 | 3716304 | after10/28 22:00 | R/main/B1-s43/B1-s43 |
| r32 B2-s43 | 3716305 | after10/28 22:00 | R/main/B2-s43/B2-s43 |
| r128 B4-s43 | 3716618 | after10/28 22:00 | R/r128/main/B4-s43/B4-s43 |

These should start automatically when normal operation resumes. Do not duplicate
them. At the07:21 snapshot25/37 records were complete; nine additional runs were
still running and the three above were unstarted. The fresher snapshot/table
below is authoritative for handover; requery after resumption.

## What is verified already

- Designea44bb6 and probe amendment A1b5d923c precede training/pilot scores.
- R32 timing smokes3715854/3715855 passed900steps; B2 explicitly accepts this
  ratio. Sizing3715950 passed, S17000 locked1dc8a76. All seven r32 Pilot-2 jobs
  3715959–3715965 exited0, both candidates complete. Selection3716286 onmg0019.
- R128 preflight3716009, full1700step smokes3716020/3716021, sizing3716162 passed;
  S17000 feasible. All five pilots3716168–3716172 exited0, ten candidates complete.
  Selection3716392 onmg0021. No smoke loss/quality was consumed.
- Full frozen-backbone512-question validation gradient cache3715870 onmg0865:
  R/jlens/directions.pt, blake2b16=9a76963e9587ae51db52b29188c004d8,
  metadata R/jlens/record.json and reports/p1l/name-gradient-probe.json. Shared
  across ranks because residual space, frozen model and questions are identical.
- Runtime checks:3715857 passed33;3715889 gate/auth4;3715942 swap fixture1;
  3716000 oracle/rank/reload34;3716014 B3 gates/config7;3716027 r128 readout3;
  3716028 global-condition gate1;3716084 inference wrapper1;3716212 batching1;
  3716225 exact residual/replay/batching5. No blanket rerun needed absent a new issue.
- Full pinned-model validation-only eval acceptance3716226 onmg0022 passed exact
  deployed-residual capture. M1 all-depth primary+typed+swap+readout200episodes
  took72.89s; B1 both-protocol acceptance3716223 onmg0030 took18.05s. Timing-only
  copies evaluation-timing-M1.json / evaluation-timing-B1.json; do not consume
  the acceptance directories' quality artifacts for scientific selection.
- B0 checkpoint3716306 onmg0019 has zero trainable parameters and recorded hash
  6af601799a569397330e74196c99ebe8. All pilot/checkpoint sources are retained.
- Handover checked only bounded scheduler/file metadata, statuses, expected config
  fields and recorded hashes. Actual checkpoint digests for all37 must be
  recomputed on compute before confirmation. No main failure.json was present.

## Remaining steps, in order

### 1. Resume after10/28 22:00 and let remaining training finish

All runtime/tests/evaluation/statistics MUST use scripts/submit.py on Miyabi-G.
Login is control only: no torch/numpy/project runtime imports, pytest or model work.
Shared venv /work/xg24i002/x10041/LoopLora/.venv and offline HF cache are unchanged.
Use the Miyabi PBS skills in the successor session.

```bash
/usr/bin/python3 -I /work/xg24i002/x10041/.codex/skills/miyabi-pbs-cmd/scripts/context.py --target-system Miyabi-G
date -Is
cat /etc/motd
/usr/bin/python3 -I /work/xg24i002/x10041/.codex/skills/miyabi-pbs-cmd/scripts/qstat_json.py --fields job_id,job_name,status,start_date,elapsed_seconds
/usr/bin/python3 scripts/p1l_job_ledger.py --out reports/p1l/jobs.json
```

Require each CURRENT job in R/main-jobs.json and R/r128/main-jobs.json to exit0
(receipt log+'.exit'), every37 record.status complete, checkpoint present, no
failure.json, correct seed/rank/S/lr and val-only training boundary. Ignore the
four superseded cancelled packs for success checks; PBS FINISH without a start
is cancellation, NOT successful training. Check every run, not just one late job.
If any running/queued job remains, wait in the successor's own session. If a job
failed, diagnose it and preserve artifacts; do not silently shorten/reseed/retune.

Do NOT rerun launch_p1l_main.py: its original packing would recreate cancelled
groups/unmap replacements, and its old absolute deadline rejects post-maintenance
launches. The three remaining jobs already exist; use their exact receipts.

### 2. Freeze authorization, then verify actual artifacts on compute

Only once the date and ALL37 metadata checks pass, create the allowed file from
the already committed B5 lock. Do not change the CI or condition count. The following
is a FUTURE command; no allowed authorization has been created during handover.

```bash
/usr/bin/python3 -I - <<'PY'
import json
from datetime import datetime
from pathlib import Path
w=Path('/work/xg24i002/x10041/LoopLora-wt/p1l')
a=json.loads((w/'reports/p1l/confirmation-lock.json').read_text())
assert datetime.now().astimezone() >= datetime.fromisoformat(a['not_before'])
assert a['gated_conditions']==2 and a['confidence']==.975
n=0
for path in a['condition_manifests'].values():
    m=json.loads(Path(path).read_text())
    jobs=json.loads((Path(path).parent/'main-jobs.json').read_text())
    for job in jobs['jobs'].values():
        assert int(Path(job['log']+'.exit').read_text())==0,job['job_id']
    for name,directory in m['runs'].items():
        d=Path(directory)
        r=json.loads((d/'record.json').read_text())
        assert r['status']=='complete' and (d/'checkpoint.pt').is_file(),name
        assert not (d/'failure.json').exists(),name
        n+=1
assert n==37
a['status']='allowed'
out=w/'reports/p1l/confirmation-authorization.json'
assert not out.exists(),'Inspect existing authorization; never rewrite after evaluation'
out.write_text(json.dumps(a,indent=2)+'\n')
PY
git add reports/p1l/confirmation-authorization.json
git commit -m 'WP-P1L: release B5 confirmation after all37 training completions'
P1L_SOURCE="$(git rev-parse HEAD)"
P1L_AUTH="$P1L_W/reports/p1l/confirmation-authorization.json"
/usr/bin/python3 scripts/submit.py --name t-p1l-preconfirm --walltime 00:05:00 --commit "$P1L_SOURCE" --wait -- -c '
import json,sys
from pathlib import Path
sys.path.insert(0,"scripts")
from evaluate_p1l_main import check_training
a=json.loads(Path("/work/xg24i002/x10041/LoopLora-wt/p1l/reports/p1l/confirmation-authorization.json").read_text())
for path in a["condition_manifests"].values():
    check_training(json.loads(Path(path).read_text()),a)
print("All37 immutable checkpoints and global training boundary verified")
'
```

Require that check's exit0 before evaluating. check_training hashes all actual
checkpoints, checks the complete two-rank matrix and fixed S/lrs and training
data boundary before any test episode generation. Commit authorization once;
do not mutate it or the training manifests after evaluation starts (hashes bind
the artifacts). Use full source hashes: submit.py resolves --commit HEAD in the
CENTRAL checkout, not this worktree. Passing P1L_SOURCE above avoids that pitfall.

### 3. Confirmation, typed replication, per-step readout and inference timing

No evaluation jobs/artifacts/manifests exist yet. Evaluate each main checkpoint
once, after the global check. This parameterized control-shell function submits
13 groups (8 at r32 including B0;5 at r128), each one node through submit.py.
The1h M1 request covers three runs at roughly10x the200-episode acceptance plus
load/reserve and1.5 margin; B8 has fewer depths/swaps. Baselines measured18s per
200episodes;20min covers three seeds. Keep every request<=2h; above1h requires
--scale-up docs/prereg/P1L.md. Do not change science to fit a job.

```bash
set -euo pipefail
submit_p1l_eval() {
  local rank="$1" label="$2" root="$P1L_R" wall=00:15:00 queue=debug-g
  local -a names=("$label-s17" "$label-s29" "$label-s43")
  if [ "$rank" = 128 ]; then root="$P1L_R/r128"; fi
  case "$label" in
    M1-L4|B8) wall=01:00:00; queue=regular-g ;;
    B1|B2) wall=00:20:00 ;;
    B0) wall=00:10:00; names=(B0-s17) ;;
  esac
  test ! -e "$root/confirmation/$label"
  /usr/bin/python3 scripts/submit.py --name "ev-p1l-r$rank-$label" \
    --queue "$queue" --walltime "$wall" --commit "$P1L_SOURCE" -- \
    scripts/evaluate_p1l_main.py --manifest "$root/training-manifest.json" \
    --authorization "$P1L_AUTH" --out "$root/confirmation/$label" \
    --runs "${names[@]}" --directions "$P1L_R/jlens/directions.pt" \
    | tee -a "$P1L_R/confirmation-submissions.jsonl"
}
for label in B8 M1-L4 B4 B5-L4 B3-L4 B1 B2 B0; do submit_p1l_eval 32 "$label"; done
for label in B8 M1-L4 B4 B5-L4 B3-L4; do submit_p1l_eval 128 "$label"; done
```

This block is a submission recipe, NOT an idempotent controller. If interrupted,
inspect submission receipts and existing output directories before submitting
only missing groups. Never blindly rerun it. Keep source and authorization fixed.
Wait in the successor's own session, verify every evaluation record complete.

evaluate_p1l_main.py performs primary + t2_typed on the SAME checkpoints,
confirm=test[100000,102000), seed4242,2000episodes, all hops1–8. M1 scans L1–8;
B8 stays L4. B0/B1/B2 report both protocols. For M1/B8 it also writes swap controls,
per-step readout and final-step faithfulness, using the cached fixed directions.
All readers get adapter stored/trainable counts and fixed-validation inference
timing. Do NOT separately recache directions or retrain/retune based on these scores.

Per-run evaluation files: record.json, primary/{samples.npz,metrics.json,config.json,
timing.json}, typed/same, inference.json; M1/B8 additionally readout.json,
swap-successors.json and swap-successors-samples.npz. readout.json contains ranks,
top1/tied-top1 for entity steps1–4 at each reader step and s4 faithfulness.

### 4. Build evaluation maps, audit and analyze on compute

After ALL37 evaluation records complete, construct the two maps on login from
metadata only. The exact path convention comes from the submission function.

```bash
/usr/bin/python3 -I - <<'PY'
import json
from pathlib import Path
r=Path('/work/xg24i002/x10041/LoopLora/runs/p1l/main')
for root in [r,r/'r128']:
    m=json.loads((root/'training-manifest.json').read_text())
    runs={name:str(root/'confirmation'/name.rsplit('-s',1)[0]/name) for name in m['runs']}
    for name,path in runs.items():
        assert json.loads((Path(path)/'record.json').read_text())['status']=='complete',name
    (root/'evaluation-manifest.json').write_text(json.dumps({'runs':runs},indent=2)+'\n')
PY
/usr/bin/python3 scripts/submit.py --name fin-p1l-r32 --walltime 00:10:00 --commit "$P1L_SOURCE" --wait -- \
  scripts/finalize_p1l_main.py --manifest "$P1L_R/training-manifest.json" \
  --evaluations "$P1L_R/evaluation-manifest.json" --authorization "$P1L_AUTH" --out "$P1L_W/reports/p1l"
/usr/bin/python3 scripts/submit.py --name fin-p1l-r128 --walltime 00:10:00 --commit "$P1L_SOURCE" --wait -- \
  scripts/finalize_p1l_main.py --manifest "$P1L_R/r128/training-manifest.json" \
  --evaluations "$P1L_R/r128/evaluation-manifest.json" --authorization "$P1L_AUTH" --out "$P1L_W/reports/p1l/r128"
/usr/bin/python3 scripts/submit.py --name fin-p1l-ranks --walltime 00:05:00 --commit "$P1L_SOURCE" --wait -- \
  scripts/compare_p1l_ranks.py --r32-manifest "$P1L_R/training-manifest.json" \
  --r128-manifest "$P1L_R/r128/training-manifest.json" \
  --r32-evaluations "$P1L_R/evaluation-manifest.json" \
  --r128-evaluations "$P1L_R/r128/evaluation-manifest.json" \
  --authorization "$P1L_AUTH" --out "$P1L_W/reports/p1l/rank-comparison.json"
```

Require all exit0. Finalizer verifies actual hashes, complete matrices, full-vocab
labels, reproduction of metrics from samples,2000paired queries/hop/depth/protocol,
confirmation range, Gate0, common oracle codes/M/write counts within rank, source
pairing and swap tables. It computes G1/G2 per rank, per-seed values and97.5% paired
joint-episode bootstrap CIs (2000draws); secondary M1 comparators/depth/far/walk
specificity and descriptive B8 rank contrast remain non-gating.

Outputs per rank: endpoints.json, accuracy.csv (counts), run-manifest.json (audited
hashes), per-step-readout.csv, readout-faithfulness.csv, swap-successors.csv,
inference.csv, results.md, accuracy.png/pdf and M1-depth.png/pdf.
Inspect audit results before drawing claims; diagnose genuine failures without
changing the pre-registered gates. Finalizer has gate/component regression coverage,
but a full real37-run finalization has NOT been executed yet.

### 5. Final report and commit

Refresh final accounting (login-safe):

```bash
/usr/bin/python3 scripts/p1l_job_ledger.py --out reports/p1l/jobs.json --require-final
```

It includes failed jobs and cancelled unstarted packs (zero actual allocation).
Its prefix filter captures t-p1l/ev-p1l/fin-p1l jobs, including the commands above.
Preserve checkpoints, samples and cached directions; no full-backbone copies.
Write reports/decisions/P1L.md using prereg decision rules, and replace the current
in-progress docs/reports/P1L.md with the final<=1page report: what ran/built, primary
and typed results per rank with CIs/per-seed effects, readout/faithfulness and
descriptive timing/parameters, actual node-hours/job evidence, deviations, limits.
Link detailed tables/figures rather than inflating that report. No learned/P3 claim.
Commit own task paths on wp/p1l; orchestrator merges. Task completion line, only
when every required step is actually finished: WP-P1L done: docs/reports/P1L.md.

## Known pitfalls and required wording

1. FINISH alone is not success. Four packed jobs were deliberately deleted before
   starting; their source IDs remain in the ledger, not the current run mapping.
2. Old Codex watchers are CANCELLED. Do not revive them or this session. PBS jobs
   remain submitted; do not qdel/resubmit the three deferred runs as routine work.
3. Never --commit HEAD directly in submit.py: it resolves in the central checkout.
   Pass an explicit full worktree commit. Job3716221 failed before runtime for this;
   replacement3716223 passed. No dependencies/pyproject/uv.lock edits are needed.
4. No scientific runtime on login. Even check_training/statistics/checkpoint hashing
   belong on compute through submit.py. Small JSON/status/file-metadata control is fine.
5. No smoke quality consumption; validation timing acceptance scores are not selection
   data. Pilot selection is already locked. No S/lr/seed/CI changes after seeing tests.
6. Call the diagnostic a J-lens-inspired name-gradient probe, NOT the published
   J-lens or sparse J-space (docs/reports/lit_check.md). Primary directions subtract
   each question's uniform1024-name gradient mean, average512questions, then unit
   normalize; raw-logit directions secondary; random unit directions fixed baseline.
7. Probe alpha*B*z_s is CAPTURED from actual per-prefix forwards, not separately
   recomputed from last-token batches.3716213 found2.29e-5 FP32 differences from
   changed matrix shapes;2823982 fixed capture;3716226 verified bitwise equality.
   B8 prefix replay retains its public L4 contract; never change unshared_steps.
8. s4 faithfulness compares probe/logit-lens top1 with the actual model top1 AMONG
   NAMES. Primary EM remains full-vocabulary. Keep strict argmax and tied-top1 separate.
9. Swap tables retain ORIGINAL source hop/episode/query; rewired hop is a separate
   counterfactual_hop.3715928 caught the earlier grouping bug;3715942 passed the fix.
   Readout batch slicing also had a counter-name shadowing bug, fixed b88a742 and
   regression3716212; current code has fixed episode_count bounds.
10. B4/B3 still prepare the same oracle support for hash comparisons, but B3 ignores
    memory. B0 config's unused writer marker does not define a learned condition.
11. Inference timing is amortized128-question batched score+full-vocab argmax,
   5warmups/20CUDA-sync repeats, prepared supports excluded, score-path CE included.
    Do not call it serial latency or include probe/audit capture time.
12. No published confirmation/analysis yet. Pilot oracle r32 scores were near chance;
    r128 selected val EM was B8 5.667%,M1 8.133%,B4 5.900%,B5 5.633%,B3 .0667%.
    These are validation only and do not establish final gates or motivate retuning.

## Handover snapshots (generated below; requery after resumption)


### Current37 run/checkpoint inventory

Captured 2026-10-27T07:25:34.909192+09:00; complete records 27/37. Actual digests still require compute verification.

R = /work/xg24i002/x10041/LoopLora/runs/p1l/main. Every directory below has its checkpoint at DIR/checkpoint.pt and record at DIR/record.json. Fully expanded paths, file sizes, hosts and recorded blake2b hashes are committed in handover-artifacts.json.

| Rank | Run | PBS ID | PBS state | Record state | Run directory |
| --- | --- | --- | --- | --- | --- |
| 32 | B0-s17 | 3716306 | FINISH | complete | R/main/B0-s17/B0-s17 |
| 32 | B1-s17 | 3716292 | FINISH | complete | R/main/B1-s17/B1-s17 |
| 32 | B1-s29 | 3716298 | FINISH | complete | R/main/B1-s29/B1-s29 |
| 32 | B1-s43 | 3716304 | QUEUED | not_started | R/main/B1-s43/B1-s43 |
| 32 | B2-s17 | 3716293 | FINISH | complete | R/main/B2-s17/B2-s17 |
| 32 | B2-s29 | 3716299 | FINISH | complete | R/main/B2-s29/B2-s29 |
| 32 | B2-s43 | 3716305 | QUEUED | not_started | R/main/B2-s43/B2-s43 |
| 32 | B3-L4-s17 | 3716291 | FINISH | complete | R/main/B4-B3-s17/B3-L4-s17 |
| 32 | B3-L4-s29 | 3716297 | FINISH | complete | R/main/B4-B3-s29/B3-L4-s29 |
| 32 | B3-L4-s43 | 3716613 | RUNNING | running | R/main/B3-L4-s43/B3-L4-s43 |
| 32 | B4-s17 | 3716291 | FINISH | complete | R/main/B4-B3-s17/B4-s17 |
| 32 | B4-s29 | 3716297 | FINISH | complete | R/main/B4-B3-s29/B4-s29 |
| 32 | B4-s43 | 3716612 | RUNNING | running | R/main/B4-s43/B4-s43 |
| 32 | B5-L4-s17 | 3716290 | FINISH | complete | R/main/B5-s17/B5-L4-s17 |
| 32 | B5-L4-s29 | 3716296 | FINISH | complete | R/main/B5-s29/B5-L4-s29 |
| 32 | B5-L4-s43 | 3716302 | FINISH | complete | R/main/B5-s43/B5-L4-s43 |
| 32 | B8-s17 | 3716288 | FINISH | complete | R/main/B8-s17/B8-s17 |
| 32 | B8-s29 | 3716294 | FINISH | complete | R/main/B8-s29/B8-s29 |
| 32 | B8-s43 | 3716300 | FINISH | complete | R/main/B8-s43/B8-s43 |
| 32 | M1-L4-s17 | 3716289 | FINISH | complete | R/main/M1-s17/M1-L4-s17 |
| 32 | M1-L4-s29 | 3716295 | FINISH | complete | R/main/M1-s29/M1-L4-s29 |
| 32 | M1-L4-s43 | 3716301 | FINISH | complete | R/main/M1-s43/M1-L4-s43 |
| 128 | B3-L4-s17 | 3716615 | RUNNING | running | R/r128/main/B3-L4-s17/B3-L4-s17 |
| 128 | B3-L4-s29 | 3716617 | RUNNING | running | R/r128/main/B3-L4-s29/B3-L4-s29 |
| 128 | B3-L4-s43 | 3716619 | RUNNING | running | R/r128/main/B3-L4-s43/B3-L4-s43 |
| 128 | B4-s17 | 3716614 | RUNNING | running | R/r128/main/B4-s17/B4-s17 |
| 128 | B4-s29 | 3716616 | RUNNING | running | R/r128/main/B4-s29/B4-s29 |
| 128 | B4-s43 | 3716618 | QUEUED | not_started | R/r128/main/B4-s43/B4-s43 |
| 128 | B5-L4-s17 | 3716398 | FINISH | complete | R/r128/main/B5-s17/B5-L4-s17 |
| 128 | B5-L4-s29 | 3716402 | FINISH | complete | R/r128/main/B5-s29/B5-L4-s29 |
| 128 | B5-L4-s43 | 3716406 | FINISH | complete | R/r128/main/B5-s43/B5-L4-s43 |
| 128 | B8-s17 | 3716396 | FINISH | complete | R/r128/main/B8-s17/B8-s17 |
| 128 | B8-s29 | 3716400 | FINISH | complete | R/r128/main/B8-s29/B8-s29 |
| 128 | B8-s43 | 3716404 | FINISH | complete | R/r128/main/B8-s43/B8-s43 |
| 128 | M1-L4-s17 | 3716397 | FINISH | complete | R/r128/main/M1-s17/M1-L4-s17 |
| 128 | M1-L4-s29 | 3716401 | FINISH | complete | R/r128/main/M1-s29/M1-L4-s29 |
| 128 | M1-L4-s43 | 3716405 | FINISH | complete | R/r128/main/M1-s43/M1-L4-s43 |

### Every WP-P1L PBS job (historical + current)

Complete inventory from reports/p1l/jobs.json;83 jobs, including failures and the four cancelled packs. FINISH/0 is observed application exit0; FINISH/— for the labelled cancelled packs means never started. Main artifact completeness is a separate check above. Node-hours below are provisional for running jobs.

| PBS ID | Job name | Observed state | Exit marker | Node-hours |
| --- | --- | --- | --- | ---: |
| 3715826 | pf-p1l | FINISH | 0 | 0.03917 |
| 3715842 | size-p1l-pf | FINISH | 1 | 0.00222 |
| 3715847 | size-p1l-pf2 | FINISH | 0 | 0.00222 |
| 3715850 | size-p1l-b1 | FINISH | 0 | 0.00222 |
| 3715854 | sm-p1l-oracle | FINISH | 0 | 0.22806 |
| 3715855 | sm-p1l-context | FINISH | 0 | 0.14639 |
| 3715857 | t-p1l-readout | FINISH | 0 | 0.01389 |
| 3715862 | pf-p1l-J | FINISH | 0 | 0.00639 |
| 3715870 | p1l-J-cache | FINISH | 0 | 0.26528 |
| 3715889 | t-p1l-gates | FINISH | 0 | 0.00528 |
| 3715928 | t-p1l-swap | FINISH | 1 | 0.00639 |
| 3715942 | t-p1l-swap2 | FINISH | 0 | 0.00583 |
| 3715950 | size-p1l-smoke | FINISH | 0 | 0.00194 |
| 3715959 | p1l-p-B8 | FINISH | 0 | 0.63222 |
| 3715960 | p1l-p-M1-L4 | FINISH | 0 | 0.60944 |
| 3715961 | p1l-p-B4 | FINISH | 0 | 0.43250 |
| 3715962 | p1l-p-B5-L4 | FINISH | 0 | 0.49111 |
| 3715963 | p1l-p-B3-L4 | FINISH | 0 | 0.43694 |
| 3715964 | p1l-p-B1 | FINISH | 0 | 0.85889 |
| 3715965 | p1l-p-B2 | FINISH | 0 | 0.88750 |
| 3716000 | t-p1l-r128 | FINISH | 0 | 0.01361 |
| 3716009 | pf-p1l-r128 | FINISH | 0 | 0.02972 |
| 3716014 | t-p1l-b3 | FINISH | 0 | 0.00528 |
| 3716015 | size-p1l-r128 | FINISH | 0 | 0.00222 |
| 3716020 | sm-p1l-r128-a | FINISH | 0 | 0.20222 |
| 3716021 | sm-p1l-r128-b | FINISH | 0 | 0.21750 |
| 3716027 | t-p1l-r128-read | FINISH | 0 | 0.00611 |
| 3716028 | t-p1l-global | FINISH | 0 | 0.00528 |
| 3716084 | t-p1l-inference | FINISH | 0 | 0.00639 |
| 3716162 | size-p1l-r128-sm | FINISH | 0 | 0.00222 |
| 3716168 | p1l-r128-p-B8 | FINISH | 0 | 0.60917 |
| 3716169 | p1l-r128-p-M1-L4 | FINISH | 0 | 0.62444 |
| 3716170 | p1l-r128-p-B4 | FINISH | 0 | 0.43500 |
| 3716171 | p1l-r128-p-B5-L4 | FINISH | 0 | 0.52000 |
| 3716172 | p1l-r128-p-B3-L4 | FINISH | 0 | 0.44778 |
| 3716191 | pf-p1l-eval | FINISH | 1 | 0.02583 |
| 3716212 | t-p1l-read-batch | FINISH | 0 | 0.00583 |
| 3716213 | sm-p1l-eval-v2 | FINISH | 0 | 0.02556 |
| 3716221 | sm-p1l-eval-B1 | FINISH | 2 | 0.00222 |
| 3716223 | sm-p1l-eval-B1v2 | FINISH | 0 | 0.01028 |
| 3716225 | t-p1l-exact-probe | FINISH | 0 | 0.00667 |
| 3716226 | sm-p1l-eval-v3 | FINISH | 0 | 0.02611 |
| 3716286 | p1l-select-r32 | FINISH | 0 | 0.00222 |
| 3716288 | p1l-m-B8-s17 | FINISH | 0 | 0.93556 |
| 3716289 | p1l-m-M1-s17 | FINISH | 0 | 0.94972 |
| 3716290 | p1l-m-B5-s17 | FINISH | 0 | 0.73000 |
| 3716291 | p1l-m-B4-B3-s17 | FINISH | 0 | 1.28056 |
| 3716292 | p1l-m-B1-s17 | FINISH | 0 | 1.27111 |
| 3716293 | p1l-m-B2-s17 | FINISH | 0 | 1.32167 |
| 3716294 | p1l-m-B8-s29 | FINISH | 0 | 0.92972 |
| 3716295 | p1l-m-M1-s29 | FINISH | 0 | 0.92306 |
| 3716296 | p1l-m-B5-s29 | FINISH | 0 | 0.72722 |
| 3716297 | p1l-m-B4-B3-s29 | FINISH | 0 | 1.29778 |
| 3716298 | p1l-m-B1-s29 | FINISH | 0 | 1.26500 |
| 3716299 | p1l-m-B2-s29 | FINISH | 0 | 1.27639 |
| 3716300 | p1l-m-B8-s43 | FINISH | 0 | 0.88806 |
| 3716301 | p1l-m-M1-s43 | FINISH | 0 | 0.89972 |
| 3716302 | p1l-m-B5-s43 | FINISH | 0 | 0.73750 |
| 3716303 | p1l-m-B4-B3-s43 | CANCELLED before start (PBS FINISH) | — | 0.00000 |
| 3716304 | p1l-m-B1-s43 | QUEUED | — | 0.00000 |
| 3716305 | p1l-m-B2-s43 | QUEUED | — | 0.00000 |
| 3716306 | p1l-m-B0-s17 | FINISH | 0 | 0.00556 |
| 3716392 | p1l-select-r128 | FINISH | 0 | 0.00222 |
| 3716396 | p1l-r128-m-B8-s17 | FINISH | 0 | 0.93833 |
| 3716397 | p1l-r128-m-M1-s17 | FINISH | 0 | 0.92889 |
| 3716398 | p1l-r128-m-B5-s17 | FINISH | 0 | 0.73750 |
| 3716399 | p1l-r128-m-B4-B3-s17 | CANCELLED before start (PBS FINISH) | — | 0.00000 |
| 3716400 | p1l-r128-m-B8-s29 | FINISH | 0 | 0.97167 |
| 3716401 | p1l-r128-m-M1-s29 | FINISH | 0 | 0.91806 |
| 3716402 | p1l-r128-m-B5-s29 | FINISH | 0 | 0.72917 |
| 3716403 | p1l-r128-m-B4-B3-s29 | CANCELLED before start (PBS FINISH) | — | 0.00000 |
| 3716404 | p1l-r128-m-B8-s43 | FINISH | 0 | 0.92528 |
| 3716405 | p1l-r128-m-M1-s43 | FINISH | 0 | 0.94944 |
| 3716406 | p1l-r128-m-B5-s43 | FINISH | 0 | 0.73278 |
| 3716407 | p1l-r128-m-B4-B3-s43 | CANCELLED before start (PBS FINISH) | — | 0.00000 |
| 3716612 | p1l-r32-one-B4-s43 | RUNNING | — | 0.54111 |
| 3716613 | p1l-r32-one-B3-L4-s43 | RUNNING | — | 0.54000 |
| 3716614 | p1l-r128-one-B4-s17 | RUNNING | — | 0.53972 |
| 3716615 | p1l-r128-one-B3-L4-s17 | RUNNING | — | 0.53972 |
| 3716616 | p1l-r128-one-B4-s29 | RUNNING | — | 0.54083 |
| 3716617 | p1l-r128-one-B3-L4-s29 | RUNNING | — | 0.54000 |
| 3716618 | p1l-r128-one-B4-s43 | QUEUED | — | 0.00000 |
| 3716619 | p1l-r128-one-B3-L4-s43 | RUNNING | — | 0.43972 |

Provisional allocation total: 35.25861 node-hours. All job sources, argv, log/spec paths and per-job walltime requests are retained in jobs.json and the central logs/pbs/submissions.jsonl.

### Wake-up terminal-state inventory

| Event | PBS job | State after handover |
| --- | --- | --- |
| ee3ad75d-7773-4f90-92d8-b0bc5660b157 | 3715965.opbs | acknowledged |
| 187e742c-1c41-4da5-8700-cc998c31ce5e | 3716619.opbs | cancelled |
| 404cd2e5-fa6f-482e-beb3-11f09a424a15 | 3716407.opbs | acknowledged |
| e098924a-8adc-43d9-b03c-063317e048ab | 3716290.opbs | acknowledged |
| 27808c33-f6d6-4f4d-b9d7-ca4e673c498b | 3716021.opbs | acknowledged |
| a101bbc0-d502-4bf4-8978-4e613b043510 | 3716305.opbs | cancelled |
| d921e843-f2a6-44ff-82e7-83cb3267c07f | 3716169.opbs | acknowledged |
