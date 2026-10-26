# P1-S figures

All figures use the locked main matrix. Accuracy panels and paired differences use confirmation seed 777001; faint lines show individual training seeds and bold lines show their mean. Paired differences show both 95% and 97.5% intervals. Training plots show logged training loss, not confirmation selection.

| Writer | M1 depth curves | M1-lin depth curves | Hop × depth heatmap | Paired differences |
| --- | --- | --- | --- | --- |
| oracle_random | [PNG](accuracy-depth-oracle_random-M1.png) | [PNG](accuracy-depth-oracle_random-M1-lin.png) | [PNG](hop-depth-oracle_random.png) | [PNG](paired-differences-oracle_random.png) |
| oracle_orth | [PNG](accuracy-depth-oracle_orth-M1.png) | [PNG](accuracy-depth-oracle_orth-M1-lin.png) | [PNG](hop-depth-oracle_orth.png) | [PNG](paired-differences-oracle_orth.png) |
| t1_frozen | [PNG](accuracy-depth-t1_frozen-M1.png) | [PNG](accuracy-depth-t1_frozen-M1-lin.png) | [PNG](hop-depth-t1_frozen.png) | [PNG](paired-differences-t1_frozen.png) |
| joint | [PNG](accuracy-depth-joint-M1.png) | [PNG](accuracy-depth-joint-M1-lin.png) | [PNG](hop-depth-joint.png) | [PNG](paired-differences-joint.png) |

Oracle-random E=12 versus E=18 load curves: [M1](load-depth-oracle_random-M1.png), [M1-lin](load-depth-oracle_random-M1-lin.png).

| Training group | Writer | Loss curves |
| --- | --- | --- |
| writer | joint | [PNG](training-loss-writer-joint.png) |
| primary | oracle_random | [PNG](training-loss-primary-oracle_random.png) |
| primary | oracle_orth | [PNG](training-loss-primary-oracle_orth.png) |
| primary | t1_frozen | [PNG](training-loss-primary-t1_frozen.png) |
| primary | joint | [PNG](training-loss-primary-joint.png) |
| replication | t1_frozen | [PNG](training-loss-replication-t1_frozen.png) |
| replication | joint | [PNG](training-loss-replication-joint.png) |
| replication | oracle_random | [PNG](training-loss-replication-oracle_random.png) |
| secondary | oracle_random | [PNG](training-loss-secondary-oracle_random.png) |
| secondary | oracle_orth | [PNG](training-loss-secondary-oracle_orth.png) |
| secondary | joint | [PNG](training-loss-secondary-joint.png) |

[Result report](../../../docs/reports/P1S-results.md) · [Machine-readable tables and audit](../../p1s/README.md)
