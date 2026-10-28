# CVH summary

Confirmation episodes: split test, seed 999005, 2000 episodes per reader; three training seeds pooled; percentage points; 99.17% intervals. Arm A0 is the existing P4-S-L head at its locked theta (reference); A1 is that head with the hop-1-constrained theta; A2 is the continuation-value head.

## Registered pattern

| family | arm | theta | hops 2-3 minus L1 [interval] | hop 1 minus L1 | mean executed updates | pattern holds | latency claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_random/r32/M1 | A0 | 0.25 | +32.88 [+31.41, +34.40] | -14.80 | 14.21 | no | no |
| oracle_random/r32/M1 | A1 | 0.10 | +9.95 [+9.15, +10.84] | +2.88 | 1.97 | yes | no |
| oracle_random/r32/M1 | A2 | 0.25 | +0.49 [+0.34, +0.67] | -0.28 | 1.05 | no | no |
| joint_fact/r32/M1 | A0 | 0.45 | +21.31 [+20.13, +22.58] | -34.83 | 16.00 | no | no |
| joint_fact/r32/M1 | A1 | 0.05 | +0.00 [+0.00, +0.00] | +0.00 | 1.00 | no | yes |
| joint_fact/r32/M1 | A2 | 0.05 | +0.00 [+0.00, +0.00] | +0.00 | 1.00 | no | yes |
| t1_frozen_fact/r32/M1 | A0 | 0.25 | +9.13 [+8.34, +9.94] | -22.60 | 15.99 | no | no |
| t1_frozen_fact/r32/M1 | A1 | 0.05 | +0.00 [+0.00, +0.00] | +0.00 | 1.00 | no | yes |
| t1_frozen_fact/r32/M1 | A2 | 0.05 | +0.00 [+0.00, +0.00] | +0.00 | 1.00 | no | yes |

## Latency (batch 64, ms; learned read against the fixed 16-update read)

| family | arm | wall | wall L16 | cuda | cuda L16 | faster than L16 |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_random/r32/M1 | A0 | 21.947 | 3.108 | 21.927 | 3.089 | no |
| oracle_random/r32/M1 | A1 | 7.829 | 3.108 | 7.809 | 3.089 | no |
| oracle_random/r32/M1 | A2 | 5.097 | 3.108 | 5.078 | 3.089 | no |
| joint_fact/r32/M1 | A0 | 20.828 | 3.081 | 20.808 | 3.063 | no |
| joint_fact/r32/M1 | A1 | 2.402 | 3.081 | 2.384 | 3.063 | yes |
| joint_fact/r32/M1 | A2 | 2.395 | 3.081 | 2.376 | 3.063 | yes |
| t1_frozen_fact/r32/M1 | A0 | 20.164 | 3.114 | 20.145 | 3.096 | no |
| t1_frozen_fact/r32/M1 | A1 | 2.420 | 3.114 | 2.402 | 3.096 | yes |
| t1_frozen_fact/r32/M1 | A2 | 2.415 | 3.114 | 2.396 | 3.096 | yes |

## Accuracy by hop group (percent)

| family | method | hop1 | reach | mid | far | all |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_random/r32/M1 | L1 | 69.1 | 0.9 | 1.2 | 1.3 | 9.7 |
| oracle_random/r32/M1 | L4 | 49.5 | 34.1 | 4.1 | 2.9 | 16.8 |
| oracle_random/r32/M1 | L16 | 14.1 | 13.8 | 13.3 | 12.7 | 13.3 |
| oracle_random/r32/M1 | A0 | 54.3 | 33.8 | 16.8 | 6.5 | 21.9 |
| oracle_random/r32/M1 | A1 | 72.0 | 10.8 | 2.0 | 1.6 | 12.8 |
| oracle_random/r32/M1 | A2 | 68.8 | 1.4 | 1.2 | 1.3 | 9.8 |
| joint_fact/r32/M1 | L1 | 84.4 | 1.6 | 1.7 | 1.2 | 11.8 |
| joint_fact/r32/M1 | L4 | 49.6 | 25.9 | 4.1 | 3.8 | 15.1 |
| joint_fact/r32/M1 | L16 | 38.4 | 19.7 | 12.0 | 7.0 | 15.4 |
| joint_fact/r32/M1 | A0 | 49.6 | 22.9 | 11.5 | 6.2 | 17.1 |
| joint_fact/r32/M1 | A1 | 84.4 | 1.6 | 1.7 | 1.2 | 11.8 |
| joint_fact/r32/M1 | A2 | 84.4 | 1.6 | 1.7 | 1.2 | 11.8 |
| t1_frozen_fact/r32/M1 | L1 | 85.1 | 1.5 | 1.1 | 1.2 | 11.7 |
| t1_frozen_fact/r32/M1 | L4 | 55.6 | 12.7 | 4.1 | 3.9 | 12.6 |
| t1_frozen_fact/r32/M1 | L16 | 49.1 | 13.0 | 6.3 | 5.1 | 12.9 |
| t1_frozen_fact/r32/M1 | A0 | 62.5 | 10.6 | 4.7 | 3.6 | 13.0 |
| t1_frozen_fact/r32/M1 | A1 | 85.1 | 1.5 | 1.1 | 1.2 | 11.7 |
| t1_frozen_fact/r32/M1 | A2 | 85.1 | 1.5 | 1.1 | 1.2 | 11.7 |

## Validation lock

Infeasible (family, arm) choices: none.
Run: 0.006 node-h of confirmation; design lock 8a13c45d2207, theta lock 19c56137c122.
