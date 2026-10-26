# P1-S main study artifacts

[Worker report](../../docs/reports/P1S-results.md). The preregistered final-v1
matrix and A5 diagnostic completed without a training failure or LR retry.
All scientific sets, thresholds, seeds and selected LRs were fixed before the
main matrix. The original throughput projection invoked the permitted uniform
15,000-step fallback. The orchestrator owns the separate decision record.

| Artifact | Contents |
| --- | --- |
| [Gate decisions](gates.md) / [JSON](gates.json) | Eight writer/reader rows, every criterion, both confidence levels, per-seed effects |
| [Accuracy tables](accuracy_tables.md) / [CSV](accuracy_tables.csv) / [JSON](accuracy_tables.json) | 46,227 cells; candidate, full-vocabulary and chance accuracy; complete confirmation tables |
| [Reported contrasts](reported_contrasts.md) / [CSV](reported_contrasts.csv) / [JSON](reported_contrasts.json) | 444 nongating contrasts: H6, linear controls, B4, r16, seed-17 screening, unseen and replication |
| [Counterfactual consistency CSV](counterfactual_consistency.csv) / [JSON](counterfactual_consistency.json) | Replacement, removal and swapped-memory conditions; follow-new-answer, original accuracy and chance components |
| [A5 pooled CSV](swap_successors.csv) / [JSON](swap_successors.json) | All primary/replication variants, L=1/4/8, original hop and modified-hop reachability; swapped versus matching unswapped predictions; never gating |
| [A5 per-seed CSV](swap_successors_per_seed.csv) / [JSON](swap_successors_per_seed.json) | Counts and rates underlying the pooled A5 rows |
| [Writer validation](writer_t1_scores.md) / [CSV](writer_t1_scores.csv) | Three new shared writers, T1 accuracy/chance and training steps |
| [Artifact audit](artifact_audit.json) | 105 stages, 102 readers, 92,451,000 rows, 24 exact frozen-writer checks and 34 controlled-writer groups |
| [Barrier](barrier.json), [analysis start](analysis_started.json), [completion](analysis_summary.json) | Confirmation release chronology, plan hash, output counts and single-pass provenance |
| [Execution](analysis_execution.json), [registry audit](registry_audit.json) | Analysis runtime/source and 145 new records appended while preserving 24 previous rows |
| [Plan](plan.json), [schedule](scheduling.json), [budget decision](training_budget.json) | Every configuration, dependency, selected LR, shared writer and pre-matrix sizing decision |
| [Job ledger](jobs.md) / [JSON](jobs.json) | All 19 PBS jobs, source commits, logs, allocated duration and node-hour accounting |
| [Figures](../figures/p1s/README.md) | 29 figures: depth curves, heatmaps, load curves, paired differences and training losses |

Raw checkpoints, logs, screening samples and confirmation samples are retained at
`/work/xg24i002/x10041/LoopLora/runs/p1s/main/20261026`.
The local ignored `compact_samples/` directory contains derived L=1/4/8 arrays
used in this analysis; the immutable raw samples and committed tables support
reproduction. Neither raw samples nor checkpoints are committed.

Confirmation seed 777001 was aggregated only after all 105 stages and all 102 sample
sets passed the global barrier at 2026-10-26 13:32:07 UTC. One analysis process
audited each raw archive once, finalized confirmation metrics, calculated the
specified 2000-resample jointly episode-resampled statistics, and rendered every
figure. Both 95% and 97.5% intervals are reported; only 97.5% enters gate decisions.
The bootstrap conditions on the three selected training seeds and does not
estimate uncertainty from drawing new training seeds.

The reusable evaluation/statistics/gate pipeline passed 335 symbolic tests on
compute and was committed early as `0f9de87` for WP-F1. Main training used
`03b62f8`; final analysis/registry orchestration used `7e1c96f`.
