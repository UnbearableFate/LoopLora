# P1-S validation pilots — WP-D2 (2026-10-26)

Completed implementation, tests, Pilot-1, Pilot-2 and replication. Decisions below
follow the preregistered rules; the orchestrator confirms them in the preregistration.
No screening or confirmation
set was evaluated and no G1/G2 or scaling conclusion is made.

Pilot-1 completed all **60** configurations, Pilot-2 all **72**, and replication
all **18**, including A1's **B5-lin**. Every training run uses seed **17**,
**5000 steps**, batch **64**;
selection uses **1000 validation episodes**, generator seed **4242**. All accuracies
use per-episode candidates. Orthogonal T1 runs require 2E≤r so every permitted
entity draw fits. The
[complete tables](../../reports/p1s-pilot/tables.md) include all Pilot-1 LR cells,
all selected hop × depth tables, loss diagnostics and links to every loss curve.
Full LR grids: [Pilot-1 CSV](../../reports/p1s-pilot/pilot1/full_table.csv),
[Pilot-2 CSV](../../reports/p1s-pilot/pilot2/full_table.csv).

The proposed load is **E*=12**, the required fallback, with primary chains **(8,4)**.
Best validation LR per (r,E), followed by E* selection and the LR at E*, gives
writer-pretraining **LR=0.003 for both r=16 and r=32**. Joint/r32 capacity was:

| E | Best LR | Single-hop validation accuracy |
| --- | --- | --- |
| 8 | 0.003 | 89.4% |
| 12 | 0.003 | 84.7% |
| 16 | 0.003 | 74.7% |
| 24 | 0.003 | 60.0% |

The 95% capacity threshold was missed at every load, triggering the writer
diagnosis before T2 interpretation. At E=12, joint/r32 scores **91.47%** on the
551 single-fact-head queries and **76.39%** on the 449 shared-head queries.
The orthogonal control scores 100% and 49.44%, respectively: its head-only keys
overwrite relation-specific facts, with 100% accuracy on the last write for a
head and 0% on earlier writes (773/227 queries). The learned writer's errors also
include single-fact heads; these descriptive strata do not identify a unique
optimization, alignment or capacity cause. Its gates are lambda=0.999968 and
beta=0.565222; mean logged CE still falls **7.30%** between the last two 500-step
windows. See the [full diagnosis](../../reports/p1s-pilot/pilot1/writer_diagnosis.json).

Pilot-2 selects maximum mean validation accuracy over hops 1–3 at L=4 (B4: L=1),
with the smaller LR breaking ties. Proposed LRs are:

| Variant | oracle_random | oracle_orth | t1_frozen | joint |
| --- | --- | --- | --- | --- |
| M1 | 0.003 | 0.003 | 0.003 | 0.003 |
| M1-lin | 0.003 | 0.003 | 0.003 | 0.001 |
| B5 | 0.003 | 0.001 | 0.003 | 0.003 |
| B5-lin | 0.001 | 0.001 | 0.003 | 0.003 |
| B3 | 0.0003 | 0.001 | 0.0003 | 0.0003 |
| B4 | 0.003 | 0.001 | 0.003 | 0.003 |

At these LRs, the primary validation selection scores (%) are:

| Variant | oracle_random | oracle_orth | t1_frozen | joint |
| --- | --- | --- | --- | --- |
| M1 | 32.27 | 68.07 | 29.23 | 47.53 |
| M1-lin | 41.13 | 94.13 | 33.40 | 47.80 |
| B5 | 32.67 | 33.33 | 25.17 | 47.27 |
| B5-lin | 32.70 | 33.33 | 26.80 | 47.87 |
| B3 | 7.10 | 8.03 | 7.97 | 8.10 |
| B4 | 32.47 | 33.33 | 26.73 | 46.03 |

Convergence is not established for every selected run. **18/24** Pilot-2 selections
meet the diagnostic plateau criterion: absolute relative change in mean logged
training CE ≤2%, comparing steps 4001–4500 and 4501–5000 (50 logged points each).
The six outside it are random/M1-lin, orthogonal/M1 and M1-lin, and joint/M1, B5
and B5-lin. Non-B4 training mixes L=1,2,4, so minibatch loss also varies with
sampled depth (B4 stays at L=1). These flags and the [loss curves](../../reports/p1s-pilot/tables.md)
describe optimization at 5000 steps; a flat loss is not proof of convergence.

Replication uses all six variants at the selected LRs, without reselection:
learned writers use `t2_typed`, E=12, three distractors, chains (8,1);
`oracle_random` uses `t2_chain_end`, E=18, chains (8,8,2).
Replication validation scores (same hops/depth metric) are:

| Variant | oracle_random, E=18 | t1_frozen, typed | joint, typed |
| --- | --- | --- | --- |
| M1 | 24.90% | 30.87% | 56.33% |
| M1-lin | 28.63% | 31.50% | 52.20% |
| B5 | 31.77% | 28.27% | 45.73% |
| B5-lin | 31.43% | 27.83% | 47.70% |
| B3 | 5.33% | 7.17% | 7.73% |
| B4 | 31.13% | 29.53% | 46.37% |

**16/18** meet the plateau diagnostic; joint/M1-lin and joint/B5-lin do not.
The [replication CSV](../../reports/p1s-pilot/replication/full_table.csv) and
[complete tables/curves](../../reports/p1s-pilot/tables.md) retain all results.

The shared frozen writer is the actual Pilot-1 joint/r32/E12/LR0.003 export at
`/work/xg24i002/x10041/LoopLora/runs/p1s/pilot-20261026/pilot1/shard-02/p1-joint-r32-E12-lr0.003/writer.pt`.
The [pre-replication audit](../../reports/p1s-pilot/pre_replication_audit/p1s_pilot_audit.json)
passed for all 132 runs and 4,476,000 sample rows, reproducing metric tables and
checking fixed x/u/M/write-count hashes across depths and controlled conditions.
The [final audit](../../reports/p1s-pilot/p1s_pilot_audit.json), PBS **3712357** on
`mg0904`, passed for all **150** runs and **5,580,000** sample rows, including
**17** controlled writer groups. All **24** frozen readers (18 primary and six
typed replication) match the exact selected writer tensors, with no local
pretraining; writer cost belongs to its Pilot-1 source run. Replication LR reuse
was checked explicitly. The [registry](../../reports/p1s-pilot/experiment_registry.csv)
retains all 150 run records; the appendix contains all **42** selected hop × depth
tables. The two timing evaluations add 288,000 separately audited sample rows.

The **2000 episodes × 8 hops × 9 depths** benchmark passed all artifact checks
on `mg0919`, PBS **3712318**, using selected M1 checkpoints and validation only:

| Writer | Evaluator time | Whole evaluation process |
| --- | --- | --- |
| oracle_orth | 3.297 s | 6.444 s |
| t1_frozen | 2.288 s | 5.374 s |

Evaluator time includes generation, preparation, writer/read computation, hashing
and saving; whole-process time additionally includes startup/checkpoint loading.
Each retained output contains 144,000 rows and 72 cells of 2000 episodes, with
identical query labels and snapshots across depths. Hardware/software: GH200
120GB, FP32, Python 3.13.13, torch 2.14.0+cu132, one CPU thread per process.
Both timing evaluations were sequential and used already selected LRs. Their
[timing/provenance report](../../reports/p1s-pilot/evaluation_benchmark.json)
also records per-depth timing. A five-minute one-node request comfortably fits
this measured evaluation workload.

Implemented amendments: allowlisted evaluation CLI overrides with original and
effective configuration provenance; jointly episode-resampled, equal-seed-mean
paired bootstrap with per-seed estimates and selectable confidence (default 95%,
A1 gates 97.5%); explicit reference-rank validation for B6/B7; `pretrain.task`
overrides; B5-lin; pilot manifests, selection, reporting and exact cross-stage
writer reuse. No preregistered thresholds changed. Bootstrap tests exercise both
same-checkpoint depth and different-checkpoint contrasts, joint episode draws,
pairing failures, and confidence handling; no pilot gate bootstrap was performed.

All runtime checks used `scripts/submit.py`, `regular-g`, one Miyabi-G node:
**182 tests** in 3711784 (`mg0852`), **119 tests** including A1 in 3711957
(`mg0921`), and **8** plan regressions in 3712017 (`mg0268`), all exit 0.
These suites overlap. The 300-step GPU acceptance reached 81.05% versus 2.08%
chance. No torch imports or pytest execution ran on the login node.

Pilot-1 source: `e94ff2b`; Pilot-2: `0ac1813`; replication and timing: `3769148`
(training code unchanged from Pilot-2). Persistent configurations, checkpoints,
samples and logs are under
`/work/xg24i002/x10041/LoopLora/runs/p1s/pilot-20261026`.
All **18 WP-D2 PBS jobs** exited 0 and used **3.966111 node-hours**. Including
the earlier WP-D tests/smoke conservatively gives **4.040833 / 12 node-hours**
(7.959167 remaining). The [per-job ledger](../../reports/p1s-pilot/jobs.md)
records every requested/actual duration and links logs; [jobs.json](../../reports/p1s-pilot/jobs.json)
also records source commits. Every job used one node and requested at most
45 minutes. No requested WP-D2 implementation or validation work remains.
