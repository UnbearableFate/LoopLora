# WP-L1 Pilot-1 artifact analysis

Implemented `scripts/analyze_p1l_pilot1.py`; invoke on compute after all nine
writers finish:

```sh
python scripts/analyze_p1l_pilot1.py --policy-root POLICY_RUN_ROOT \
  --capacity-root CAPACITY_RUN_ROOT --throughput THROUGHPUT_JSON \
  --reports REPORT_DIRECTORY [--jobs JOB_METADATA_JSON]
```

Each run root may contain writer directories directly or a complete `suite.json`
with `run_root`. The analyzer checks all six E16 policy/lr cells before opening
capacity artifacts. It imports `validate_writer_config`, `choose_steps`,
`select_policy`, and `select_capacity` from `looplora.llm.pilot1_select`; no
duplicate selection implementation. Throughput input has top-level `selected`
with E24/all_tokens, batch_size/procs, episodes_per_second and
worst_worker_p95_step_seconds. The fixed shared rule supplies expected steps.

Audits include complete records and required artifacts; writer seed17/rank32/
layer14; selected batch/steps; validation-only memory-only L1, seed4242, no
counterfactuals; exactly 1000 distinct episodes with query0; full-vocabulary and
candidate correctness labels; sample-derived metric tables and matching config
hashes, evaluation ranges, record counts and timing coverage. Training logs must
span step1 through the requested final step and retain the writer data seed.
Capacity configs must use the selected policy/lr and cover exactly E8/12/24;
selected E16 supplies the fourth capacity row.

Outputs: `p1l_pilot1_runs.csv` (nine rows with scores/counts, configurations,
training/evaluation timings, job/host/GPU/commit and loss-trend metadata),
`p1l_pilot1_loss_curves.csv`, nine-panel loss PNG, and
`p1l_pilot1_summary.json` (shared policy/capacity selection and explicit failure
statement if selected-policy EM is below .95 at every E).

Loss trends use logged points strictly after 80% of requested optimization steps.
OLS slope, observed first-minus-last drop and fitted relative drop are reported.
A fitted drop >=1% of mean tail loss is `still_falling`; <=-1% is `rising`;
otherwise `converged_descriptively`. Fewer than two tail points is insufficient.
These labels are descriptive, do not establish optimization completion, and
never alter selection or request additional runs.

Optional jobs JSON maps exact PBS job ID to `{nodes, elapsed_seconds}`. CSV
per-job costs repeat for runs sharing a job; summary totals deduplicate job IDs
and cover writer jobs only. Parent owns PBS elapsed/cost registry and throughput
job costs. Without `--jobs`, per-job cost fields are null.

Validation: Python stdlib AST parse passed. No imports of NumPy/Torch/Matplotlib,
analysis execution, tests, submission or commits were performed by this worker.
Miyabi plane: login/control `miyabi-g2`, aarch64, no allocation. Runtime analysis
remains for the parent's compute job. No existing run results were read.
