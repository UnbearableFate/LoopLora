# WP-D — completed 2026-10-26

Implemented the symbolic backbone, oracle/random and per-episode orthogonal
writers, shared T1-pretrained/frozen and joint writers, all eight reader controls
including **M1-lin**, training/checkpoint reload, fixed-snapshot depth sweeps,
counterfactuals, diagnostics, episode-cluster bootstrap, figures, parallel suite
runner and registry. Master `dbead6f` is integrated into `wp/sym`; smoke source
is **`dd0bad0`**. The later report commit only normalizes timing CSV newlines and
adds results. No dependencies changed.

**Validation (all through `scripts/submit.py`, regular-g, one Miyabi-G node):**

- `--name t-sym --walltime 00:10:00 --worktree . --wait -- -m pytest -q tests -rP`:
  **487 tests passed**, exit 0, **3711498.opbs**, `mg0853`. The 300-step GPU T1
  acceptance test achieved **81.05%** candidate accuracy versus **2.08%** chance
  in **5.056 s** training time.
- After fixing fresh-process CUDA initialization, the final archived source
  passed **25 runner tests**, including both `cuda` and `cuda:0`, then completed
  the smoke and artifact audit in **3711572.opbs**, `mg0292`, exit 0. This job ran
  `pytest -q tests/test_sym_run.py -rP` followed by
  `scripts/smoke_sym.py --suite configs/p1s/smoke.yaml --procs 4 --runs
  /work/xg24i002/x10041/LoopLora/runs/p1s/wp-d-smoke-final-20261026 --reports
  /work/xg24i002/x10041/LoopLora-wt/sym/reports`, submitted with
  `--queue regular-g --walltime 00:10:00 --commit dd0bad0 --wait`.

**Smoke:** 24/24 runs, seed 17, 300/3000 steps, batch 32: T1/T2 × eight variants,
two fixed-L=4 M1 runs (B4 supplies fixed L=1), one T1 writer and five T2 readers
sharing its exact checkpoint. Suite wall time **159.754 s**; whole final job
**193 s**, including runner tests and reporting. Per-run total time **12.93–28.41 s**;
reader training **6.02–21.78 s**; shared-writer pretraining **16.82 s**.
Mixed-depth mean L **2.3867**. Every run's measured components are in
[p1s_smoke_timing.csv](../../reports/p1s_smoke_timing.csv); per-L reader timings,
scope, synchronization and runtime metadata are also retained per run.
Hardware/software: GH200 120GB, FP32, Python 3.13.13, torch 2.14.0+cu132.
Times include concurrent-process contention (torch reports 72 CPU threads);
they are sizing measurements, not isolated method-latency comparisons. Repeating
this exact smoke fits a **one-node, five-minute** request using 1.5× measured job time.

All sample tables reproduce saved metrics; Gate 0 passed for every batch/depth.
The three controlled groups have identical x/u/M/write-count hashes across
B3/B4/B5/M1/M1-lin. The [audit](../../reports/p1s_smoke_audit.json),
[registry](../../reports/experiment_registry.csv),
[400 stratified paired contrasts](../../reports/p1s_smoke_paired.json), and
[PNG figures](../../reports/figures/p1s/) are committed. Large run artifacts remain
at the persistent `--runs` path above.

**Decisions/limits:** evaluation constructs exactly one query per hop from fixed
support; swaps preserve original candidates, inputs and labels. Orthogonal
counterfactuals share the original basis and require spare code dimensions.
B6/B7 are joint-trained capacity controls, outside equal-writer claims. T3
missing-data coverage and answerable-counterfactual restrictions are documented
in [configuration guidance](../../configs/p1s/README.md); T1/T2 are fully validated.
Initial job **3711517.opbs** failed before training on CUDA statistics startup;
the fix passed focused job **3711546.opbs** and final fresh-process tests above.
No login-node torch imports/pytest, pilots, screening, or method-selection claims.
No outstanding WP-D checks or required user action.
