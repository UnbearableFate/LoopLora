# WP-H evaluation interfaces

- `evaluate(model, config, out, *, samples_only=False, provenance=None)` preserves
  the default `(metrics, timing)` result. Samples-only returns `(metadata, timing)`
  and writes `samples.npz`, `evaluation.json`, `config.json`, `timing.json`;
  it does not call aggregation or write `metrics.json`. Correctness and diagnostics
  remain per-sample arrays. The caller supplies checkpoint provenance (checkpoint
  path, checkpoint_config, eval_overrides). Metadata includes the effective config,
  both configuration hashes, sample-file blake2b seal, and all batch/depth snapshots.
  CLI `--samples-only` provides the same mode; the five approved `--set` keys remain.
- `finalize_evaluation(out, *, barrier_passed=False)` refuses unless the caller
  explicitly attests the global training-and-confirmation-samples barrier passed.
  It reads saved samples and metadata, computes metrics once, and writes metrics.json.
  It verifies the sample-file seal and refuses to overwrite existing metrics.
  The caller owns the global barrier; this function does not inspect checkpoints.
- `eval.fixed_depth_sweep` defaults false. True preserves configured eval depths
  for B4/B6/B7; their training depths remain `[1]`. B8 remains fixed-L4 because its
  unshared reader does not support other depths. Main configs must specify all nine
  depths and set this field true. B4's prespecified comparator remains its L1 row.
- Screening counterfactuals include `remove_required_fact`. Forked samples retain
  source episode/query ids and original hop. Insufficient answers enable the
  insufficient candidate for the whole removal condition, including answerable
  examples. Removal candidates retain original source entities plus modified-fact
  entities so a cut at the first edge can return the query start even when absent
  from modified facts. Orthonormal replace/remove controls use separate per-query
  fork extensions, preserving original entity slots and the full random basis;
  fresh entities across different forks never consume a shared union of slots.
- Minimal approved model helper change: `orthogonal_codebooks(...,
  include_insufficient=False)` reserves the insufficient orthogonal slot for
  fixed-hop removal only. No training behavior changes.
- A5: `eval.swap_successors` defaults false. With counterfactuals enabled on
  `test`, true adds `swap_successors` forks at L1/4/8 only (these three depths
  must be configured). The generator receives `eval.seed` directly, i.e.4242
  for screening. Each source episode/query has one fork, no fresh entities,
  and the original complete orthogonal codebook is reused. This is diagnostic
  only and never affects gates. Main enables it only for primary/replication.
- Every sample now has `counterfactual_hop_true = part.q_hop_true`; `hop` remains
  the original source-query hop. For swap_successors, the first field is the
  recomputed new path length. `original_prediction` is the matching unswapped
  prediction at the same split/episode_id/query_id/L, so the unswapped follow
  comparison is `original_prediction == answer` (the new terminal). Both the
  original and the changed answers remain separately available. Swap-memory
  retains the original queries, including their original hop annotations.

Implemented with focused tests in `tests/test_sym_main_evaluate.py`; updated the
existing orthogonal control regression to check preserved codes rather than a
global codebook call count. Static compilation and whitespace checks passed.
Runtime imports and pytest were not run on the login node; compute validation
remains with the parent.
All new aggregation/finalization unit fixtures use synthetic eval seed900001;
A5 fixtures use900002. They do not generate preregistered confirmation episodes.
