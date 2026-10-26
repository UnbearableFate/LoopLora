# IS — memory load × read depth

Registered design: [IS](../../docs/prereg/IS.md); immutable scientific lock `fb62cf1`.
All model training, inference, tests, audits and statistics run on compute through `scripts/submit.py`.
Raw plans, checkpoints, blind samples, complete metadata and analysis cache stay in
`/work/xg24i002/x10041/LoopLora-wt/symf/runs/is/`; earlier studies are read-only.

- `design.json`: exact validated layouts, 54 reference checkpoint paths/BLAKE2b hashes, six inherited LRs,
  confirmation seed 999001/2000 episodes, IS-E4 definitions and validity adjustments fixed before inference/training.
- `training_manifest.csv`: 36 new 15000-step, batch-64 runs (E24/36, two writers, M1 mixed/B8/B4, three seeds).
- `smoke_timing.json`: registered 10% E36 smoke, both writers, M1 mixed, 1500 steps/200 validation episodes.
- `packing_timing.json`, `runtime.json`, `cost_gate.json`: timing-only packing probe, tested all-valid execution,
  and job sizing. `scripts/is_runtime.py` removes per-fact GPU synchronizations with the same ordered FP32
  writes, using the reference for padding. Source and checkpoints remain reproducible through `run_is.py`.
- `accuracy.csv`: every writer/reader/variant/training-load/evaluation-load/update/hop/L/seed with counts;
  `best_L.csv`: descriptive best mixed-M1 L per load (hops2/3, pooled fixed seeds, ties choose smaller L).
- `endpoints.csv`, `tables.md`: IS-E1 D(E), independent-load DeltaD, IS-E2 capability, IS-E3 load adaptation;
  all 97.5% CIs and per-seed effects. No model or LR is selected using confirmation.
- `t3_update.csv`: IS-E4 correct/stale counts and rates by current-path update flag, hop (0 means pooled1–4),
  L, writer and reader, with per-seed values and pooled episode-bootstrap 97.5% CIs. Untouched stale rate
  is undefined, not correct-answer rate; empty strata are null. Stale terminal follows first occurrences
  of repeated keys, while current answers follow latest occurrences.
- `accuracy_vs_L.png`, `accuracy_vs_L.pdf`: registered fixed-checkpoint load/depth figure, reach and far panels.
- `decision.json`, `audit.json`: registered interpretation, reader-specific load boundary and artifact checks.
- `jobs.csv`, `budget.json`: actual node-hours, including design validation, smoke, packing and all tests.

E6 uses hops1–4; other T2 loads use hops1–8. Q3 keeps the (8,4) current graph and two obsolete facts, so
physical stored E=14; hops1–4 preserve generator-default mixed updated/untouched queries. B8 supports only L4.
These validity adjustments were committed in the design lock. Joint is descriptive throughout.
CIs condition on training seeds17/29/43. Within-load contrasts pair unchanged episode/query labels across
all seeds and readers; load-specific independent streams are derived from seed2718, 2000 resamples.
