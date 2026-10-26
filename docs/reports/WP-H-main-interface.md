# WP-H execution/artifact contract

Binding design: docs/prereg/P1S.md final v1, docs/tasks/WP-H_p1s_main.md.

`looplora.sym.main` owns plan/run/barrier. Plan JSON at `<root>/plan.json` has
`steps`, `runs` (105 entries: 3 writer +72 primary +18 replication +12 secondary),
`root`, `prereg_commit`, `selection_source`, `throughput_decision`. Each run has
`name`, `group` (writer/primary/replication/secondary), `writer`, `variant`,
`seed`, `r`, `config` (resolved mapping), `run_dir` (absolute), `dependencies`
(names). Writer outputs `<root>/runs/<name>/writer.pt`; all consumers in its seed
use that exact artifact. No pilot checkpoint is consumed by a main run.

The run directory has config.json (envelope), record.json, train_log.jsonl,
checkpoint.pt, optional writer.pt, and evaluation directories:

- `screening`: test + test_unseen, seed4242, 2000episodes, screening test CFs.
- `confirmation`: test only, seed777001, 2000episodes, samples-only mode. No
  metrics.json or aggregate diagnostics until the global barrier.
- Auxiliary writer runs have `writer_validation` (val, seed4242, 1000episodes,
  single hop/L1); no T2 confirmation because these are not reader matrix runs.

record.json `status=complete` means training + required samples are written;
`training_complete=true`, `confirmation_samples_written=true` for reader runs.
`config` contains the effective trained config and `planned_config_hash` records
the initial specification. A non-finite loss permits one next-lower-grid retry;
failed attempt artifacts remain in `attempt-0`, successful artifacts at run root.
Records retain attempt metadata. For normal runs the config equals plan exactly.

Barrier `<root>/barrier.json` must say `status=passed`, list all105 runs and all102
confirmation sample paths, and record its creation timestamp. It is created only
after checking completion, checkpoints, samples, no pre-existing confirmation
metrics, and selected step budgets. Analysis must verify this barrier before
opening any confirmation samples or finalizing metrics. Evaluator metadata
`confirmation/evaluation.json` contains hashes/config but no metric table.

Analysis outputs under `reports/p1s/` and figures under `reports/figures/p1s/`.
Gate API contract: WP-H-stats-notes.md. Main training/retries never read confirmation
metrics. B4 is trained only at L1; WP-H explicitly evaluates its checkpoint across
all nine depths using eval.fixed_depth_sweep=true, and B4 comparisons use L1.

The runner never prints screening/confirmation scores. Per-job cost is tracked
separately by the root worker in reports/p1s/jobs.json. Final report is ≤1page plus
links to detailed tables and diagnostics; no orchestrator decision record edits.
