# WP-L1 analyzer read-only review

Reviewed source only; no analyzer execution or result-score access. Production files were not edited.

1. **P2: a persisted policy lock can return stale execution context.**
   `scripts/analyze_p1l_pilot1.py:254` checks only selected policy, lr, correct and n.
   `analyze_policy` returns the prior lock at line269 even after auditing a different
   throughput selection or six-run source set. If the chosen pair and its count coincide,
   the CLI at line363 emits the old batch/procs/steps, which may differ from newly audited
   runs and cause downstream capacity configs to use the wrong locked budget. Full analysis
   also labels this old lock verified. Reject mismatched budget and run provenance before
   accepting an existing lock; don't replace a prior scientific decision silently.

2. **P2: valid capacity shard packing is rejected.**
   `_run_paths` at line78 permits at most two capacity shards, although three additional
   capacity runs may legally be submitted as three single-run suites. Such a root fails
   before audit despite exactly three valid run directories. The task has no two-job cap.
   Bound shard count by expected run count, while retaining one-level discovery and exact
   total/name checks. Current three-shard/six-run policy layout is accepted.

No concrete error found in primary counts: each audited run requires 1000 unique episodes,
query0, memory-only original val/hop1/L1, exact full-vocabulary correctness aliases and
sample-derived metric reproduction. Policy selection is evaluated before capacity reads;
capacity selection uses the copied selected E16 row plus three required capacities. Cost
totals deduplicate exact job IDs at lines340–343, so repeated per-run job cost fields and
the selected-row copy do not double-charge the summary total. The persisted lock is read,
not mutated, during capacity analysis. Missing-lock status is explicitly reported false.

Resolved before policy analysis: locks now bind throughput, step budget and all
six runs' configuration/source/count provenance; final analysis requires the
lock and rejects capacity runs created before it. Shard bounds now equal the
expected run count. Regression tests cover stale budgets/sources, unchanged
locks, three capacity shards, duplicate names and missing-lock read ordering.
