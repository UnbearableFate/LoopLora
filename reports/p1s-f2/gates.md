# F2 confirmation gates

Differences and intervals are percentage points. Decisions use unrounded 97.5% intervals. F2 omits replication and is a weaker design.

| Writer | r | Reader | G1 | G2 | Pass |
| --- | --- | --- | --- | --- | --- |
| joint_fact | 32 | M1 | False | True | False |
| joint_fact | 32 | M1-lin | False | True | False |
| t1_frozen_fact | 32 | M1 | False | True | False |
| t1_frozen_fact | 32 | M1-lin | False | True | False |


## joint_fact / r=32 / M1

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +22.98 [+22.10, +23.88] | +22.98 [+22.02, +23.98] | 17: +23.18, 29: +23.57, 43: +22.18 |
| far | +2.82 [+2.46, +3.18] | +2.82 [+2.40, +3.23] | 17: +2.95, 29: +2.57, 43: +2.93 |
| specificity | +20.16 [+19.22, +21.03] | +20.16 [+19.06, +21.19] | 17: +20.23, 29: +21.01, 43: +19.24 |
| hop1 | -35.62 [-37.03, -34.10] | -35.62 [-37.22, -33.93] | 17: -36.50, 29: -38.05, 43: -32.30 |
| vs_fixed | +23.23 [+22.34, +24.12] | +23.23 [+22.16, +24.25] | 17: +23.45, 29: +23.90, 43: +22.35 |
| vs_loop | +16.44 [+15.45, +17.47] | +16.44 [+15.31, +17.68] | 17: +17.05, 29: +17.20, 43: +15.07 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.22975 | >= 0.03 | True |
| reach_ci_positive | 0.2201666666666667 | > 0.0 | True |
| reach_each_seed_positive | 0.22175 | > 0.0 | True |
| specificity_ci_positive | 0.19058333333333352 | > 0.0 | True |
| hop1_degradation_at_most_1pp | -0.3561666666666667 | >= -0.01 | False |
| vs_fixed_ci_positive | 0.22158020833333333 | > 0.0 | True |
| vs_loop_ci_positive | 0.15308333333333335 | > 0.0 | True |


## joint_fact / r=32 / M1-lin

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +27.61 [+26.64, +28.53] | +27.61 [+26.53, +28.71] | 17: +27.90, 29: +26.35, 43: +28.57 |
| far | +2.79 [+2.44, +3.13] | +2.79 [+2.39, +3.18] | 17: +2.67, 29: +2.82, 43: +2.90 |
| specificity | +24.81 [+23.83, +25.81] | +24.81 [+23.69, +25.96] | 17: +25.23, 29: +23.53, 43: +25.67 |
| hop1 | -37.47 [-38.93, -36.02] | -37.47 [-39.10, -35.85] | 17: -37.95, 29: -36.65, 43: -37.80 |
| vs_fixed | +27.89 [+26.95, +28.88] | +27.89 [+26.83, +28.99] | 17: +28.50, 29: +26.52, 43: +28.65 |
| vs_loop | +20.82 [+19.79, +21.96] | +20.82 [+19.63, +22.11] | 17: +21.68, 29: +19.62, 43: +21.15 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.2760833333333333 | >= 0.03 | True |
| reach_ci_positive | 0.26533333333333337 | > 0.0 | True |
| reach_each_seed_positive | 0.2635 | > 0.0 | True |
| specificity_ci_positive | 0.2369437500000003 | > 0.0 | True |
| hop1_degradation_at_most_1pp | -0.3746666666666667 | >= -0.01 | False |
| vs_fixed_ci_positive | 0.26833229166666667 | > 0.0 | True |
| vs_loop_ci_positive | 0.19633229166666666 | > 0.0 | True |


## t1_frozen_fact / r=32 / M1

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +11.68 [+11.08, +12.32] | +11.68 [+10.97, +12.43] | 17: +12.45, 29: +10.78, 43: +11.82 |
| far | +2.74 [+2.38, +3.08] | +2.74 [+2.33, +3.13] | 17: +3.55, 29: +2.13, 43: +2.53 |
| specificity | +8.94 [+8.25, +9.63] | +8.94 [+8.16, +9.72] | 17: +8.90, 29: +8.64, 43: +9.29 |
| hop1 | -29.82 [-31.20, -28.45] | -29.82 [-31.37, -28.20] | 17: -30.35, 29: -28.70, 43: -30.40 |
| vs_fixed | +11.68 [+11.03, +12.34] | +11.68 [+10.93, +12.42] | 17: +12.68, 29: +10.50, 43: +11.88 |
| vs_loop | +5.12 [+4.35, +5.93] | +5.12 [+4.19, +6.07] | 17: +6.25, 29: +4.25, 43: +4.88 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.11683333333333334 | >= 0.03 | True |
| reach_ci_positive | 0.10975 | > 0.0 | True |
| reach_each_seed_positive | 0.10775 | > 0.0 | True |
| specificity_ci_positive | 0.08158194444444447 | > 0.0 | True |
| hop1_degradation_at_most_1pp | -0.2981666666666667 | >= -0.01 | False |
| vs_fixed_ci_positive | 0.10933229166666668 | > 0.0 | True |
| vs_loop_ci_positive | 0.041915625 | > 0.0 | True |


## t1_frozen_fact / r=32 / M1-lin

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +11.58 [+10.89, +12.24] | +11.58 [+10.82, +12.35] | 17: +12.30, 29: +10.12, 43: +12.30 |
| far | +2.45 [+2.13, +2.77] | +2.45 [+2.09, +2.82] | 17: +2.83, 29: +1.98, 43: +2.53 |
| specificity | +9.12 [+8.42, +9.81] | +9.12 [+8.34, +9.90] | 17: +9.47, 29: +8.14, 43: +9.77 |
| hop1 | -27.93 [-29.23, -26.58] | -27.93 [-29.48, -26.42] | 17: -26.80, 29: -28.15, 43: -28.85 |
| vs_fixed | +11.84 [+11.13, +12.53] | +11.84 [+11.04, +12.62] | 17: +12.55, 29: +10.40, 43: +12.57 |
| vs_loop | +5.12 [+4.29, +5.99] | +5.12 [+4.22, +6.12] | 17: +6.12, 29: +3.95, 43: +5.27 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.11575 | >= 0.03 | True |
| reach_ci_positive | 0.10825 | > 0.0 | True |
| reach_each_seed_positive | 0.10125 | > 0.0 | True |
| specificity_ci_positive | 0.08344305555555555 | > 0.0 | True |
| hop1_degradation_at_most_1pp | -0.2793333333333333 | >= -0.01 | False |
| vs_fixed_ci_positive | 0.11041666666666668 | > 0.0 | True |
| vs_loop_ci_positive | 0.042248958333333336 | > 0.0 | True |
