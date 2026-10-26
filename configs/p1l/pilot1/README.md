# P1-L Pilot-1

Authority: `docs/tasks/WP-L1_p1l_pilot1.md` and the Pilots section of
`docs/prereg/P1L.md`. Only writer training and validation are authorized here.

The six-cell real writer throughput benchmark selected training batch **32**,
**two processes per GPU**, and **1500 steps** before any policy score was read.
Each scientific run trains on 48,000 episodes (192,000 single-token questions).
Training seed is 17; evaluation uses 1000 validation episodes with seed 4242,
one single-hop query per episode, memory-only, L=1, full-vocabulary EM primary.
The selected batch is the P1-L training batch for subsequent stages too; this
task executes only the writer stage.

The benchmark used E=24 and all-token writes, the largest authorized load.
For each batch/process cell it discarded five warmup steps and measured the
next 25 actual AdamW steps. It selected the highest summed simultaneous-worker
training episodes/s among fitting cells. The worst selected worker p95 was
0.717092 s/step. The fixed rule
`floor100((1800 - 90) / (1.5 * p95))` gives 1500 steps, reserving 90 seconds for
evaluation/loading and a 1.5 training margin. The benchmark's validation scores
are never used (it does not evaluate).

`policy-shard-1.yaml` through `policy-shard-3.yaml` cover the full six-cell
policy × lr grid at E=16. Each shard pairs both policies at one lr; shards can
run in separate one-node allocations with `--procs 2`. After all six complete,
`scripts/analyze_p1l_pilot1.py --policy-only` audits and saves the selection.
Candidates within five correct answers of the best out of 1000 tie; prefer
`fact_final`, then the smaller lr. That saved choice is verified before capacity
analysis. Capacity E=8/12/24 configs are created only after the choice is locked.

Writer T1 has four relations and 0.3 head sharing, without chains. The E=16
layout (8,4,2,2) belongs to downstream T2 and does not alter direct T1 facts.
Loss trends describe the final 20% of logged steps; they never extend the budget
or change selection. No reader/joint/baseline training, test/confirmation data,
other seeds, or further tuning are part of this task.

Submit through `scripts/submit.py` only. Source is a committed archive; run
artifacts stay under `/work/xg24i002/x10041/LoopLora/runs/p1l/pilot1-20261026/`.
`locked-plan.json` and the report retain timing, selected settings and provenance.
