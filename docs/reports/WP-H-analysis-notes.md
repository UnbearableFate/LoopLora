# WP-H main analysis interface

`scripts/analyze_p1s_main.py` implements final-v1 confirmation analysis and the
A5/Addendum-1 diagnostic. Execute only in a compute allocation:

```sh
PYTHONPATH=src python scripts/analyze_p1s_main.py \
  --plan /absolute/main/root/plan.json \
  --out reports/p1s --figures reports/figures/p1s
```

`--cache-limit` defaults to 12 compact run/set arrays. No training, checkpoint
selection, cost accounting, registry append, or decision-record edit occurs.
Those remain with the main runner/orchestrator.

## Release and one-pass behavior

`verify_release(plan_path)` performs metadata/file-existence checks only. It
requires the final-v1 105-run matrix (3 auxiliary writers, 72 primary, 18
replication, 12 secondary), every current run record complete, all 102 reader
confirmation sample files written, consistent planned/effective config hashes,
and common selected steps. No numpy archive or checkpoint tensor is opened by
this function. The barrier must have `status="passed"`, `plan_hash` equal to
`stable_hash(plan)`, a valid ISO `created_at`, `runs` containing every run name,
and `confirmation_samples` containing exactly the 102 absolute sample paths.
Confirmation evaluation metadata must retain `aggregation_deferred=true`, no
aggregate table or diagnostics, eval seed 777001, test only, 2000 episodes,
hops 1–8 and all nine specified depths.

Only after the entire check passes does analysis access raw confirmation
samples or calculate any confirmation metric. Each raw archive is loaded
once, separately from other runs. The audit reproduces every public accuracy
cell, checks sample/query counts, correctness, depth-independent labels,
counterfactual source labels and donors, file digests, every depth snapshot,
controlled-variant snapshots, and all 24 frozen readers' exact writer tensors
against their seed's freshly trained main writer. Frozen-reader checkpoints
must also record no local pretraining and the frozen-writer state.

Confirmation metrics are finalized from those already loaded arrays with
`finalized_after_global_barrier=true`. Paired inference uses disk-cached fields
`episode_id/query_id/hop/L/answer/correct` for original rows at L=1,4,8 only;
constant split/condition labels are reconstructed. Full prediction/diagnostic
arrays are released after each run. In-memory compact caching is bounded.

Analysis refuses existing confirmation metrics or `analysis_started.json`.
This prevents accidental repeated confirmation passes. If a run fails partway,
its audit and artifacts remain for inspection; the script does not silently
resume/recompute or delete evidence. A new full analysis requires deliberately
handling that partial state first.

## Statistics and reports

`sym.stats.pooled_compare_runs` and `pooled_walk_specificity` supply every
contrast. 95% and 97.5% intervals use the same 2000 bootstrap draws/seed 2718;
episodes are resampled jointly across training seeds. Specificity is a single
per-episode difference of reachable and far depth effects. `sym.gates` alone
applies final-v1 thresholds to exactly eight confirmation rows. The two oracle
conditions both use oracle_random/E18 replication.

Outputs under `--out`:

- `artifact_audit.json`: counts, fingerprints, snapshot/frozen-writer checks.
- `gates.json` and `gates.md`: eight rows, all nine endpoint components, raw
  criterion values/thresholds, both intervals and per-seed estimates.
- `accuracy_tables.{json,csv,md}`: every run/set/hop/depth CSV/JSON cell,
  candidate/full-vocabulary/chance accuracy, complete confirmation Markdown.
- `reported_contrasts.{json,csv,md}`: H6 (L8−L4), M1−M1-lin, B4@1−M1@1,
  r16, seed-17 screening values of every gate statistic and unseen values;
  replication and secondary contrasts are reported without gate decisions.
- `counterfactual_consistency.{json,csv}`: follow-new-answer components,
  conditional replacement accuracy, original accuracy, change from original,
  and excess above candidate-set chance.
- `swap_successors_per_seed.{json,csv}` and `swap_successors.{json,csv}`:
  A5 counts/rates per seed and pooled, grouped by run group/writer/variant,
  L=1/4/8, original hop, and actual modified `counterfactual_hop_true <= L`
  (plus an all-rows stratum). Follow-new-terminal rate is compared with
  `original_prediction == answer` on the exact matching unswapped query.
  The original `hop` is preserved for stratification. A5 never enters gates.
- `writer_t1_scores.{json,csv,md}`: auxiliary writer T1 validation accuracy,
  candidate-set chance, and selected training steps for all three seeds.
- `analysis_summary.json`: completed analysis provenance/counts and figures.
- `compact_samples/`: compact original-query data reused by paired inference.

Figures under `--figures`: per-reader/per-writer accuracy–L panels by hop
(three seed lines and seed mean), per-writer five-reader hop×L heatmaps,
oracle_random E12/E18 load×L panels, per-writer paired differences with both
CIs, and training-loss curves for every group/writer/variant/seed.

## Validation status

`tests/test_sym_main_analysis.py` uses synthetic data only. It covers release
failure before archive access, all105/all102 membership, partial-depth A5
coverage, sample/table integrity, matching unswapped predictions, changed-hop
stratification, immutable snapshots, bounded caching without truncating large
episode IDs, shared-confidence bootstrap inputs, and the eight-row gate routing.
An aggregate-only synthetic smoke test renders all 29 required figure files.
No runtime, torch import, pytest, or real confirmation inspection was performed
by this worker on the login node. Allocation validation is owned by the root.
