# P2-S-F1 artifact index

Protocol: [P2S-F1](../../docs/prereg/P2S-F1.md); task [WP-P2S-F1](../../docs/tasks/WP-P2S-F1.md).
All earlier runs are read-only. Raw plans, checkpoints, validation samples, confirmation samples,
provenance and the complete statistics stay under
`/work/xg24i002/x10041/LoopLora-wt/symf/runs/p2s-f1/`.

- `design.json`, `smoke.csv`, `smoke_timing.json`, `cost_gate.json`: design and measured cost acceptance.
- `pilot_grid.csv`, `selected_lrs.csv`: all 24 validation pilots and the eight LR choices committed in
  `44c1fd5` before main training (seed 17, 5000 steps, trained depth, hops 1–3; smaller LR breaks ties).
- `training_manifest.csv`: the 24 fresh 15000-step main runs, seeds 17/29/43, batch 64.
- `checkpoint_manifest.csv`: 24 new and 18 unchanged reference checkpoint paths and BLAKE2b seals.
- `tables.md`, `endpoints.csv`: every F-E1/F-E2/F-E3 estimate and per-seed effect; CSV also retains
  absolute per-seed accuracies and both 95% and decision 97.5% CIs. F-E3 hop-1/long contrasts are accompanying diagnostics.
- `accuracy.csv`: per-hop, per-depth accuracy pooled over the three fixed training seeds.
- `decision.json`: registered reader, writer-specific training-depth choices and shortcut-free capability outcome.
- `audit.json`: complete training/sample barriers, paired labels, immutable depth snapshots and checkpoint checks.
- `jobs.csv`, `budget.json`: actual PBS resource use, with smoke and test/selection overhead charged to pilot.

Confirmation uses P2-S's test seed 888001, 2000 episodes, one query at each hop 1–8; this set was already
used by P2-S. All 24 new runs must finish before evaluation and all 42 blind sets must exist before aggregation.
Results: B8, fixed L4 for every writer; all four shortcut-free capability contrasts pass.
Bootstrap: shared `sym.stats`, 2000 pooled paired episode draws, seed 2718, conditional on the three seeds.
No latency benchmark is part of this follow-up. Compact committed outputs must remain below 1 MB.
