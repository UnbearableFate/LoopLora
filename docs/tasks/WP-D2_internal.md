# WP-D2 worker coordination

Binding: WP-D2_p1s_pilot.md and prereg/P1S.md. Runtime work only in Miyabi-G
compute allocations via scripts/submit.py. Work on wp/sym; no agent commits.
Pilots use only seed17, split val, eval seed4242; no screening/confirmation runs.

Ownership:
- evaluation agent: evaluate.py CLI --set allowlist and focused tests in
  tests/test_sym_eval_overrides.py. Preserve existing outputs; record resolved
  config + overrides. Only eval.seed/episodes/splits/batch_size/counterfactuals.
- statistics agent: stats.py pooled multi-seed paired bootstrap and focused
  tests/test_sym_multiseed.py. Resample common episodes jointly for all seeds;
  mean per-seed paired accuracy difference for explicit hops, split, condition.
  Strict data matching across all seeds and both sides, per-seed estimates.
- config agent (runner): config.py model.reference_r and pretrain.task;
  train.py _pretrain_config merge; tests/test_sym_pilot_config.py. No edits run.py.
- root: pilot.py, configs, pilot runner/scheduling/selection/timing, integration,
  compute jobs and report. Keep total P1-S cost below12 node-hours, job<=1node-hour.

API additions:
- model.reference_r defaults to declared main r, not hard-coded64. B6 requires
  explicit model.r > model.reference_r; B7 explicit model.r_mem > reference_r
  and > model.r. Ordinary variants do not need reference_r declared. Existing
  smoke capacity configs must be updated with explicit sizes/reference64.
- pretrain.task default{}; merged over generated T1 TaskSpec mapping. T1 name
  remains t1; support n_rel/head_sharing etc. Reject incompatible n_rel with
  reader backbone (caller must configure same n_rel); no silent weight mismatch.
- stats.pooled_compare_runs(samples_a_by_seed: Mapping[int,SampleSource],
  samples_b_by_seed, depth_a, depth_b, *, hops, split='val',
  condition='original',resamples=2000,seed=2718,confidence=0.95)->dict. It handles same-checkpoint
  depth contrasts by passing the same mapping twice. Include CLI --pooled-a
  manifest.json --pooled-b manifest.json, depths/hops/split/condition; seed keys
  JSON strings parsed as ints. Report mean difference, CI, per-seed estimates,
  n_episodes,n_queries_per_seed, resampling/seed details. No averaging unmatched
  subsets or treating queries/seeds as independent clusters.

Subagents do not execute pilots or modify prereg thresholds. Root records and proposes
selection decisions; orchestrator confirms prereg changes later.

A1 is integrated: B5-lin is fixedq/identity; grids are60/72/18. Pooled confidence
0.975 uses the same bootstrap draws as0.95. Pilot-2 uses six shards after measured
orthogonal-writer step timing; stage-specific selection rules are recorded in plans.
