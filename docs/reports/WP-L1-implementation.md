# WP-L1 configuration and decision helpers

`looplora.llm.pilot1_select` provides pure decision helpers and a resolved-config builder:

- `writer_config(name, E, policy, lr, batch, steps)` fixes writer/M1, seed17, rank32,
  layer14, memory-only L1, T1 training Q4/n_rel4/head_sharing.3; eval is 1000 validation
  episodes, seed4242, no counterfactuals. `validate_writer_config(config)` rejects scope
  changes. Writer T1 has no chain layout; the E16 chain layout belongs to later T2.
- `select_policy(rows)` requires exactly six E16 policy/lr cells. Row schema is
  `{E, write_policy, lr, correct, n:1000}` plus optional retained report fields.
  It returns a copy of the selected row: global-best minus correct ≤5, then fact_final,
  then the smallest lr. The primary count is full-vocabulary EM, not candidate accuracy.
- `select_capacity(rows)` requires exactly one row for E8/12/16/24 at one selected pair,
  returning `{E_star, flagged, diagnosis_required}`. It applies ≥950/1000 for largest
  E16/24, then E12; E8 needs ≥800 and is flagged; otherwise E_star is null and diagnosis
  is required. It never adds tuning or relaxes thresholds.
- `choose_steps(selected_throughput_row)` requires E24/all_tokens, batch_size8/16/32,
  procs1/2 and `worst_worker_p95_step_seconds`. It returns S in multiples of100:
  `floor((1800-90)/(1.5*p95)/100)*100`, rejecting a budget below100 steps. The90-second
  load/eval reserve is outside the1.5 training margin. Measured p95 and scheduling costs
  belong to the primary benchmark/report, not this helper.

Existing evaluator `stratify_queries` sets one hop1 query per writer evaluation episode,
so training Q4 is compatible with exactly1000 EM observations. No query filtering is used.
Tests cover scope guards, global-best tie boundaries, required grids, capacity thresholds
and diagnosis, and budget feasibility. Runtime tests/submissions/commits were not performed
in this implementation-only subtask. Only static syntax/whitespace checks are claimed.
