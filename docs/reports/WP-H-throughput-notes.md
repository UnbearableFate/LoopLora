# WP-H throughput benchmark preparation

`scripts/benchmark_p1s_main.py` is ready; no runtime measurement has been executed
by this worker. Run the first throughput job on one Miyabi-G node with walltime
at most 10 minutes, through `scripts/submit.py`, before starting the main matrix:

```bash
scripts/submit.py --name p1s-throughput --queue regular-g --walltime 00:10:00 \
  --commit <integrated-commit> --wait -- scripts/benchmark_p1s_main.py \
  --out /work/xg24i002/x10041/LoopLora/runs/p1s/main/throughput \
  --steps 250 --procs 2 4 8 --eval-seconds-per-reader 60
```

Each concurrency setting runs the same eight real train-only workloads, seed 17,
250 steps, batch 64, one CPU thread per child, all sharing the first visible GPU:

| Cases | Writer | Rank | Task / E / layout | Reader |
| --- | --- | --- | --- | --- |
| orth32-relu, orth32-linear | oracle_orth |32|chain_end /12/(8,4)|M1, M1-lin|
| orth16-linear | oracle_orth |16|chain_end /12/(8,4)|M1-lin|
| random32, random16 | oracle_random |32,16|chain_end /12/(8,4)|M1, M1-lin|
| joint-chain | joint |32|chain_end /12/(8,4)|M1|
| joint-typed | joint |32|typed /12/3distractors/(8,1)|M1-lin|
| joint-t1 | joint |32|T1 /12/head_sharing=.3/n_rel=4|B4|

These are timing trials, not main runs or scientific results. The script calls
`train.train` directly and never evaluates. No Pilot writer is loaded. The
final main writers must still be independently trained at the selected stage
length. Every trial retains config/provenance, train_log.jsonl, checkpoint.pt,
summary.json, timing.json and a collector-compatible record.json explicitly
marked `evaluation_performed:false`, zero evaluation samples and null scores.
Benchmark trials can therefore be added to the experiment registry normally.
Joint trials also retain writer.pt under the existing training contract.

`throughput.json` contains elapsed time, every trial's mean/p95 step timing,
per-class maximum mean step timing, aggregate node throughput and projection
rows for both 20000 and 15000 steps at each measured concurrency. No cutoff is
selected automatically; the parent applies the preregistered 8 node-hour rule.

Projection counts: three auxiliary T1 writers, 72 primary readers, 18 replication
readers and 12 secondary readers (105 training stages, 102 reader evaluations).
Frozen-reader cost uses joint training as an upper-bound proxy; secondary joint
r16 uses r32; random E18 multiplies measured E12 cost by 1.5. Per-class times
already reflect concurrent contention. The conservative projection serializes cost classes, dividing non-orthogonal
work by process count but capping orth32 at `min(procs,2)` and orth16 at one.
The eight-case mix measures at most two simultaneous orth32 jobs and one orth16
job, so homogeneous eight-orthogonal scaling must not be assumed. It retains
the measured packing/startup factor
`max(1, batch_elapsed*procs/sum(train_seconds))`, then applies a 1.5 margin.
The default evaluation allowance is 60 process-seconds per reader for all required
sets/conditions together; this is an explicit sizing assumption, not a new
measurement. Prior validation timing is in `reports/p1s-pilot/evaluation_benchmark.json`.
The benchmark's own elapsed node-hours are added separately to projected WP-H
cost. Retries are not separately scheduled in the estimate; the 1.5 margin is
retained. Each class also reports projected single-stage training time, 1.5× walltime
including evaluation allowance, and whether that stage fits one hour. Passing
the whole-matrix node-hour bound does not establish per-job feasibility. Use
`--procs 2 4 8` to measure lower contention when 4/8 exceed that limit. The parent
selects a compatible job mix and cutoff. Class assumptions and all arithmetic
remain in the JSON.

Static compilation passed. Actual throughput, cutoff and PBS job/node accounting
remain pending the first compute job; this note contains no measured timings.
