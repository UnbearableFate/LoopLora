# P1-S F2 artifacts

Binding design: `docs/prereg/P1S-F2.md`. F2 has two conditions, six reader variants,
three training seeds and r=r_mem=32. Every main stage uses 15000 steps; pilots
use 5000. F2 has **no replication setting**, a weaker design than P1-S main.

- `pilot_writer_grid.csv`, `pilot_reader_grid.csv`, `selected_lrs.csv`: validation
  selection only; ties choose the smaller learning rate. The selected T1 pilot
  is diagnosed by head multiplicity in `pilot_writer_diagnosis.csv`.
- `main_manifest.csv`: three fresh main T1 writers and 36 readers. Frozen
  variants share their seed's main writer, never the pilot checkpoint.
- `job_ledger.csv`: exact submit specifications, code sources, compute hosts,
  scheduler elapsed seconds and exit statuses. Node-hours exclude queue time.
- `gate_endpoints.csv`, `gate_criteria.csv`, `gates.csv`, `gates.md`: every
  confirmation endpoint, per-seed effect and decision for four condition/reader
  rows. CSV differences/intervals are fractions; Markdown uses percentage points.
- `accuracy_summary.csv`: candidate/full-vocabulary/chance accuracy by
  condition, variant, split, original/counterfactual condition, hop and depth,
  pooled equally over the three training seeds. Auxiliary T1 validation scores
  are also retained per seed in `writer_t1_scores.csv`.
- `reported_contrasts.csv`: seed-17 screening, three-seed unseen, H6, nonlinear
  versus linear reader, and B4@1 versus M1@1 contrasts; never gating.
- `swap_successors.csv`: original-hop and new-hop<=L strata with the paired
  unswapped-query follow baseline. `swap_successors_new_path_length.csv` adds
  exact modified path lengths. Both are screening diagnostics, never gating.
- `cross_family_contrasts.csv`: factorized minus released WP-H unfactorized
  L4 accuracy and depth gains, paired by episode/query/hop/answer. The depth
  contrast is (F4-F1)-(U4-U1), bootstrapped jointly before computing its interval.
  These comparisons are reported endpoints and do not alter F2 gate decisions.
- `depth_profiles.{pdf,png}`, `training_losses.{pdf,png}`: three-seed mean
  confirmation profiles versus WP-H, and individual main training curves.
- `artifact_audit_summary.json`, `reference_audit_summary.json`: compact audit
  receipts; full audits and predictions remain under the ignored run root.

All inference uses 2000 joint episode bootstrap draws (seed 2718), with equal
training-seed weights. Both 95% and 97.5% intervals use the same draws. The CIs
condition on the three supplied training seeds. Decisions use unrounded 97.5%
intervals and the binding point thresholds, without replication.

Raw checkpoints, predictions, full tables, logs, plans, release barrier and
compact inference caches are retained under
`/work/xg24i002/x10041/LoopLora-wt/symf/runs/p1s/f2/`.
The runtime drivers are `scripts/run_p1s_f2.py` and `scripts/analyze_p1s_f2.py`;
all runtime commands must go through `scripts/submit.py`. Evaluation rejects
any unfinished main stage. Analysis requires all 36 blind confirmation sets
before aggregation; reference access additionally requires completed F2
analysis and the explicit WP-H release in `reference_release.json`.

Shared WP-H analysis and `sym.stats` supply the audits and paired statistics.
The F2 adapter aliases only audit role labels (`t1_frozen_fact` to `t1_frozen`,
`joint_fact` to `joint`), preserving actual configurations, model tensors and
predictions. This retains exact frozen-writer and controlled-variant checks.
