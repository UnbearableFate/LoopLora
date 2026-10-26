# WP-P4L analysis interface (agreed with root)

Root agreed the record contract after this proposal. No confirmation artifacts
have been read by this worker. Ownership: only `scripts/analyze_p4s_learned.py`
and `tests/test_p4s_learned_analysis.py`, plus this interface note.

## Manifest / lock

CLI: `python scripts/analyze_p4s_learned.py --root RUN_ROOT --out REPORT_DIR`.
`RUN_ROOT/manifest.json` should contain `status='complete'`, `smoke=false`,
`design`, `runs` (24 unique readers/heads), `design_hash`, `design_lock_commit`,
`theta_lock_commit`, absolute `theta_path`, `theta_blake2b`, and
`checkpoints_unchanged=true`.
Run identity: `name`, `writer`, `r=32`, `variant`, `seed`, `family` (convention
`writer/r32/variant`), reader/head checkpoint paths and blake2b seals.
Writers are `joint_fact`, `t1_frozen_fact`, `oracle_random`, `joint`; variants
`M1`, `M1-lin`; seeds 17/29/43. Six non-joint families are primary; joint is
descriptive. Traces use `traces/<name>/{val,test}.npz`.

Committed `REPORT/theta.json` contains `design_hash`, `locked_at`, `selected`
(8 family-wide theta rows, with validation score and mean chosen/executed
steps), and `grid` (all 19 theta rows per family). No self-referential commit
field appears in theta.json. Analysis
validates the selected theta against that frozen grid, then validation traces;
confirmation paths are opened only after lock checks. Agreed theta ties follow
WP-C3: accuracy, then fewer mean SELECTED steps, then smaller theta.

`select_theta(root,runs,design)` returns `{design_hash, selected, grid}` and
opens validation traces only. Root adds `locked_at` and commits the report copy.
`validate_theta(selection,runs,design)` returns `{family: theta}` without trace
I/O. Analysis independently recomputes validation selection before any test read.

Design supplies a frozen per-family table for all three label-free rules:
criterion, threshold, validation accuracy, threshold-source path/seal. Analysis picks the
best of these three rules from validation accuracy only (deterministic ties by
criterion order read_norm, relative_change, output_stability). The design's
`best_label_free` is a mapping family→selected full row. Per user clarification,
preparation selects the missing six oracle_random/r32 thresholds on validation
with the C3 grids before any head training; the other 18 C4 rows are reused.

## Trace arrays (no pickle)

For each of val/test: `episode_id`, `hop`, `answer`: [E,8]; `predictions`:
[E,8,17] for states 0..16; `probabilities`: [E,8,16] for states 1..16;
`selected_steps`, `executed_steps`, `read_calls`: [E,8]. E=2000, hops ordered
1..8 once per episode; episode ids repeat across queries and agree across all
seeds within each family. Validation and confirmation ids must be disjoint.

Proposed label-free fields: `reference_read_norm_steps`,
`reference_relative_change_steps`, `reference_output_stability_steps`: [E,8].
Alternatively save existing `read_norm` and `relative_change` [E,8,17] so the
released `halting_steps` definition can reconstruct references; predictions
provide output stability. Proposed corresponding execution/read-count fields
if needed: `reference_<criterion>_executed_steps` and
`reference_<criterion>_read_calls`. Otherwise use the existing WP-C3 contract:
reference selected/executed L; read_norm L+1 reads, other rules L reads.

Learned selection must equal first p_s>=theta, else earliest argmax p_s. Fallback
executes all 16 updates even when the chosen state is earlier. Head features use
post-state read-norm lookahead: learned read_calls=executed_steps+1. Test counters
are mandatory and checked against probabilities/theta; validation counters may
be absent because those traces are written before theta selection.

## Timing

Proposed `latency.json`: one row per run/method (`learned`, `fixed_4`,
`fixed_16`), batch_size=64, warmup/repeats, CUDA mean/median/p95 ms and wall
mean/median/p95 ms. Preserve existing WP-C3 field names `cuda_ms_mean`,
`cuda_ms_median`, `cuda_ms_p95`, `wall_ms_mean`, `wall_ms_median`, `wall_ms_p95`.
Pool equally over three seeds; report per-run rows and execution semantics.
Exactly 72 rows are required. Counters `mean_steps`, `mean_executed_steps`,
`mean_reads`, `audit_prediction_steps_exact`, `audit_state_rel_l2_max`, and
`audit_state_abs_max` are validated/preserved as in root's runner.

## Statistics / outputs

Synthetic L labels encode learned versus L1/locked reference correctness for
`sym.stats.pooled_compare_runs`; `pooled_walk_specificity` computes the one
per-episode reach-minus-far contrast. Use 2000 joint episode draws across seeds,
confidence `1 - .05/6` (plus descriptive 95% if useful), base seed 2718. Primary
qualification is hop1 point >=-.01, reach point >=.03 and adjusted CI lower>0,
specificity adjusted CI lower>0; joint rows are always labeled descriptive.

Outputs: accuracy/counts by family/seed/hop/rule; pooled/per-seed endpoint rows;
all-eight-family endpoint JSON/CSV/Markdown; learned-minus-best-label-free on
hops1..8; oracle-minus-learned gap (any correct step 1..16, excluding state0);
true-hop reference; chosen/executed/read means; locked theta/reference selections;
latency tables and audit metadata. No head/checkpoint/threshold tuning.

Implementation and synthetic tests are written. Existing sym.stats APIs are
unchanged. Ordinary descriptive comparisons bootstrap only their requested
overall endpoint, avoiding redundant per-hop-subset resampling. Any-step oracle
has no asserted inference cost; its execution/read fields are null. CSV headers
cover ordinary and specificity per-seed component fields. Compute validation
is delegated to root; no runtime imports/tests ran on login.
