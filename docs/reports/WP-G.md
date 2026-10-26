# WP-G — completed 2026-10-26

Implemented writer/frozen-reader/joint training, B1 LoRA and B2 parallel context
baselines, B0, full-vocabulary EM, candidate accuracy, counterfactuals, depth
sweeps, adapter-only checkpoints, persistent-worker suites and artifact audits.
Reuses WP-D statistics and record format; flat per-seed samples remain compatible
with later pooled statistics. Master through `955ce44` is integrated. Final
implementation/smoke source: **`3cb908c`**, branch `wp/llm`; no core/data/sym or
dependency changes. Configuration/API details: [README](../../configs/p1l/README.md).

**Tests: 611 passed, none skipped**, exit 0, **3712076.opbs**, Miyabi-G **mg0904**,
20:20:27–20:21:53 JST. Includes exact truncated/full support states and logits
(gradients within FP32 roundoff), frozen writers, teacher-forced/greedy EM, B0
chance control, sample-derived metrics, B2, and independent BF16 workers with
different histories matching all 100 replacement batches.

```sh
scripts/submit.py --name t-wpg-determin --queue regular-g --walltime 00:05:00 \
  --worktree . --wait -- -m pytest -q tests
```

**Smoke: 14/14 runs and all audits passed**, exit 0, **3712085.opbs**, **mg0753**
(GH200 120GB), 20:23:05–20:44:00 JST. One seed 20261027; 100/1000 steps, batch 8;
200 validation episodes with a separate smoke evaluation seed, hops 1–8 (writer:
hop 1), both counterfactuals. B0 has no training. All five frozen readers have
bitwise-identical x/u/M/write-count hashes; saved writers remain bitwise unchanged.
All sample tables, paired comparisons, configurations and coverage reproduce.

```sh
scripts/submit.py --name s-wpg-final --queue regular-g --walltime 00:40:00 \
  --commit 3cb908c --wait -- scripts/smoke_llm.py --suite configs/p1l/smoke.yaml \
  --procs auto --runs /work/xg24i002/x10041/LoopLora/runs/p1l/wp-g-smoke-final-20261026 \
  --reports /work/xg24i002/x10041/LoopLora-wt/llm/reports
```

The bounded B0 scheduling comparison measured **14.87 s sequential / 11.88 s
with two processes**, including loads; selected two persistent workers on one
GPU. Suite time **1216.85 s**; entire accepted job **1255 s / 0.349 node·h**.
Measured seconds below include concurrent-worker contention. Each evaluation
set includes configured depths and counterfactuals; retained context applies to M1
and the baselines. BF16 backbone, FP32 adapters/state, SDPA, torch 2.14.0+cu132.

| Stage | seconds / 1k steps | 200-episode memory-only eval | retained-context eval |
| --- | ---: | ---: | ---: |
| Writer | 484 | 12.7 | — |
| Frozen readers | 279–322 | 33.3–98.8 | 360.5 |
| Joint variants | 310–554 | 32.4–102.9 | 358.8 |
| B1/B2 | 331–346 | 10.9–15.5 | 87.3–87.6 |
| B0 | — | 9.0 | 74.8 |

[Audit](../../reports/p1l_smoke_audit.json), [per-run timings](../../reports/p1l_smoke_timing.csv),
[compute log](/work/xg24i002/x10041/LoopLora/logs/pbs/s-wpg-final-20261026-202209-1815163.log).
Large artifacts remain at the persistent run path above. A repeat smoke fits a
one-node **35-minute** request with 1.5× measured margin. Linear extrapolation
puts full memory-only configurations within a 45-minute request; M1 with both
protocols projects about 82–86 minutes before margin, so later retained-context
evaluation must be split into smaller depth jobs before scaling.

**Decisions/limits:** initial smoke `3711870` was rejected for one process-dependent
memory hash (0.331 node·h); deterministic Torch/cuBLAS/cuDNN execution and the
new cross-process regression resolved it without relaxing hashes. `fact_final`,
E=24, chains=(8,8,8), and 1000 default steps are smoke implementation choices,
not selected scientific settings. Writer/B4/depth-independent baselines use L=1;
other readers sweep 0/1/2/3/4/6/8. No pilots, screening, confirmation access,
method selection, login-node Torch/pytest, or skipped required checks. WP-G has
no outstanding implementation or validation work.
