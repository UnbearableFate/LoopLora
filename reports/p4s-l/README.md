# P4-S-L: learned halting on frozen r32 readers

[Worker report](../../docs/reports/P4S-L.md) ·
[Registered design](../../docs/prereg/P4S-L.md) ·
[Task](../../docs/tasks/WP-P4L_learned_halting.md)

The study fits one 51→64 ReLU→1 halting head for each of 24 unchanged readers.
It tests six primary families (factorized joint/frozen and random-oracle, each
with M1/M1-lin); the two unfactorized joint families are descriptive.

| Artifact | Contents |
| --- | --- |
| [Design lock](design.json) | All 24 reader paths and blake2b seals, features, seeds, grids, optimizer and inference rules; committed before any study head training |
| [Label-free references](label_free_thresholds.csv) | The 18 reused C4 rows and six newly selected random/r32 rows, with source seals |
| [Preparation audit](preparation_audit.json) | Checkpoint identities/seals, validation-only scope and confirmation exclusion |
| [Head checkpoints](heads/), [manifest JSON](head_manifest.json), [CSV](head_manifest.csv) | All 24 exported head tensors and seals, exact initialization/sampler seeds, 2000-step records and unchanged reader hashes |
| [Theta lock](theta.json), [lock audit](theta_lock_audit.json) | Eight validation-selected thresholds and all 152 grid rows; committed before confirmation |
| [Confirmation manifest](confirmation_manifest.json), [timing](confirmation_timing.json) | Post-lock confirmation scope, source commit and measured execution times |
| [Results](tables.md), [qualification CSV](qualification.csv) | Every family's registered endpoints and adjusted-CI qualification, with joint families marked descriptive |
| [Endpoint JSON](endpoints.json), [CSV](endpoints.csv), [per-seed CSV](endpoints_per_seed.csv) | Paired differences, 2000-resample joint episode-bootstrap intervals, and individual seed effects |
| [Accuracy CSV](accuracy.csv), [per-seed CSV](accuracy_per_seed.csv) | Counts and accuracy by family/hop/rule; chosen states, executed updates and memory reads |
| [Theta CSV](theta.csv), [grid CSV](theta_grid.csv) | Readable exports of the committed validation-only choice |
| [Latency CSV](latency.csv), [per-seed CSV](latency_per_seed.csv) | CUDA-event and synchronized wall mean/median/p95 per prepared batch of 64; five warmups and 20 repeats |
| [Analysis audit](analysis.json) | Complete matrix, lock commits, adjusted confidence, counts and execution conventions |
| [Validation](validation.json), [smoke](smoke.json), [compute plan](compute_plan.json) | 159 compute tests, uniform 10% smoke without scientific score selection, and measured walltime sizing |
| [Job ledger](jobs.md), [JSON](jobs.json) | PBS source commits, logs and allocated node-hours, including the initial fixture failure |

Clarification approved before any head training: C3 had oracle-random thresholds
only at r64/r128, and C4 had none for oracle_random/r32. The two missing r32
families therefore select their label-free references on validation using the
unchanged C3 grids and tie rule. All six C4 families reuse their existing
thresholds. This clarification is also recorded in the design lock and report.

Training uses split `train`, generator seed 31337, 20,000 episodes, and 2000 Adam
updates at lr 0.001. Each sampled minibatch contains 256 whole episodes with all
eight queries and all sixteen post-update states. Features contain no answer,
hop label, correctness label or decoder confidence; targets are the frozen
reader's own candidate-argmax correctness. No reader parameter is optimized.

Validation uses split `val`, seed 4242, 2000 episodes. One theta per family
maximizes mean accuracy over all hops and three seeds; ties prefer fewer mean
chosen steps, then smaller theta. Confirmation uses split `test`, fresh seed
999002, 2000 episodes. The adjusted CI level is `1 - 0.05/6` (99.166667%).
Episode draws are paired and shared across the three fixed reader/head seeds;
intervals do not estimate uncertainty from sampling new training seeds.

Inference chooses the first post-update state with p≥theta, or the earliest
maximum-probability state if none qualifies. Fallback executes all sixteen
updates even when its chosen state is earlier. The post-state read-norm feature
requires one lookahead read, so learned inference costs executed updates +1
memory reads. Fixed-depth and label-free references retain C3's read accounting.
The label-informed any-step oracle uses states 1..16 and is an accuracy upper
bound; true-hop stopping is a separate label-informed reference.

Raw head checkpoints, training records, validation/confirmation traces and
timing records are retained under
`/work/xg24i002/x10041/LoopLora/runs/p4s-l/20261027/main-v1`.
Preparation and discarded-head smoke records are sibling directories. The
committed manifests contain each head's exact path and blake2b seal. Small head
checkpoints are also exported under `heads/` with identical byte hashes.
Frozen-reader checkpoints and sample arrays remain outside Git.
