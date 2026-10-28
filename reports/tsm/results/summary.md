# TSM summary

Confirmation episodes: split test, seed 999007, 2000 episodes per run; three training seeds pooled; percentage points; 98.75% intervals. T0 is the retrained reference, T1 trains at depth 8 with the stay loss, T2 adds the sticky gate; B4 is the released single-read reference.

## Registered conditions

| writer | arm | hops 1-3: L16 - L4 | <= 1 pp | hops 4-8: arm@8 - T0@8 | >= -1 pp | hops 2-3: arm@4 - B4@1 | >= 3 pp, interval > 0 | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_random | T1 | -6.66 [-7.77, -5.65] | no | -11.67 [-12.64, -10.82] | no | +16.58 [+15.47, +17.58] | yes | no |
| oracle_random | T2 | -6.40 [-7.48, -5.26] | no | -11.78 [-12.76, -10.86] | no | +16.57 [+15.49, +17.64] | yes | no |
| joint_fact | T1 | -0.79 [-1.43, -0.14] | yes | -5.73 [-6.39, -5.05] | no | +16.95 [+15.91, +17.90] | yes | no |
| joint_fact | T2 | -4.36 [-5.36, -3.43] | no | -4.55 [-5.25, -3.86] | no | +21.53 [+20.55, +22.57] | yes | no |

An arm passes only if it passes for both writers: T1: no, T2: no

## Registered reading

| arm | oracle_random | joint_fact | outcome |
| --- | --- | --- | --- |
| T1 | fails | fails | passes for neither writer (a training failure counts as not passing) |
| T2 | fails | fails | passes for neither writer (a training failure counts as not passing) |

An arm that passes for exactly one writer is a writer-specific result, not a remedy. The statement that no arm passes is worded per writer; a training failure counts as not passing, and the table above says which cells failed in training:

- oracle_random: no arm passes
- joint_fact: no arm passes

## Accuracy by depth (percent, hops 2-3)

| writer | method | L | seeds pooled | hops 2-3 | hops 6-8 | hop 1 |
| --- | --- | --- | --- | --- | --- | --- |
| joint_fact | B4 | 1 | 17;29;43 | 0.8 | 0.6 | 92.8 |
| joint_fact | T0 | 1 | 17;29;43 | 3.1 | 3.2 | 52.0 |
| joint_fact | T0 | 4 | 17;29;43 | 28.6 | 3.2 | 52.3 |
| joint_fact | T0 | 8 | 17;29;43 | 21.9 | 5.8 | 39.3 |
| joint_fact | T0 | 16 | 17;29;43 | 20.8 | 8.9 | 37.4 |
| joint_fact | T1 | 1 | 17;29;43 | 0.9 | 0.8 | 88.9 |
| joint_fact | T1 | 4 | 17;29;43 | 17.8 | 2.6 | 62.4 |
| joint_fact | T1 | 8 | 17;29;43 | 15.9 | 3.5 | 63.5 |
| joint_fact | T1 | 16 | 17;29;43 | 16.2 | 3.8 | 63.1 |
| joint_fact | T2 | 1 | 17;29;43 | 0.8 | 0.7 | 88.8 |
| joint_fact | T2 | 4 | 17;29;43 | 22.4 | 2.5 | 58.9 |
| joint_fact | T2 | 8 | 17;29;43 | 25.4 | 3.4 | 48.8 |
| joint_fact | T2 | 16 | 17;29;43 | 22.9 | 6.7 | 44.8 |
| oracle_random | B4 | 1 | 17;29;43 | 0.6 | 0.3 | 96.2 |
| oracle_random | T0 | 1 | 17;29;43 | 0.1 | 0.1 | 13.4 |
| oracle_random | T0 | 4 | 17;29;43 | 41.3 | 2.1 | 51.7 |
| oracle_random | T0 | 8 | 17;29;43 | 20.6 | 12.9 | 17.1 |
| oracle_random | T0 | 16 | 17;29;43 | 11.5 | 11.2 | 11.5 |
| oracle_random | T1 | 1 | 17;29;43 | 1.5 | 1.6 | 58.2 |
| oracle_random | T1 | 4 | 17;29;43 | 17.1 | 2.1 | 64.1 |
| oracle_random | T1 | 8 | 17;29;43 | 24.0 | 5.7 | 51.8 |
| oracle_random | T1 | 16 | 17;29;43 | 21.4 | 11.7 | 35.5 |
| oracle_random | T2 | 1 | 17;29;43 | 1.6 | 1.5 | 58.4 |
| oracle_random | T2 | 4 | 17;29;43 | 17.1 | 2.3 | 64.1 |
| oracle_random | T2 | 8 | 17;29;43 | 23.4 | 5.6 | 51.8 |
| oracle_random | T2 | 16 | 17;29;43 | 21.6 | 11.3 | 35.9 |

Evaluation: 0.003 node-h (the completed attempt); design lock 1bf689be3589.

Code (source_hash): the 18 runs were trained at 9433cb32a72c; the evaluation ran at 9433cb32a72c (the same code).
