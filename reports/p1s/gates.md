# P1-S confirmation gates

Accuracy differences and intervals are percentage points. Decisions use unrounded 97.5% intervals and final-v1 thresholds; 95% intervals are descriptive.

| Writer | Reader | G1 | G2 | Pass |
| --- | --- | --- | --- | --- |
| oracle_random | M1 | False | True | False |
| oracle_random | M1-lin | False | True | False |
| oracle_orth | M1 | True | True | True |
| oracle_orth | M1-lin | False | True | False |
| t1_frozen | M1 | False | True | False |
| t1_frozen | M1-lin | False | True | False |
| joint | M1 | True | True | True |
| joint | M1-lin | False | True | False |


## oracle_random / M1

| Endpoint | Estimate [95% CI] (pp) | Estimate [97.5% CI] (pp) | Per-seed estimate (pp) |
| --- | --- | --- | --- |
| reach | +33.26 [+32.16, +34.33] | +33.26 [+32.02, +34.48] | 17: +34.12, 29: +31.72, 43: +33.92 |
| far | +1.72 [+1.36, +2.08] | +1.72 [+1.32, +2.13] | 17: +2.17, 29: +0.98, 43: +2.02 |
| specificity | +31.54 [+30.32, +32.72] | +31.54 [+30.17, +32.87] | 17: +31.96, 29: +30.74, 43: +31.91 |
| replication_reach | +20.50 [+19.55, +21.51] | +20.50 [+19.47, +21.67] | 17: +21.35, 29: +20.95, 43: +19.20 |
| replication_far | +1.76 [+1.43, +2.08] | +1.76 [+1.39, +2.14] | 17: +2.00, 29: +1.63, 43: +1.65 |
| replication_specificity | +18.74 [+17.74, +19.75] | +18.74 [+17.62, +19.87] | 17: +19.35, 29: +19.32, 43: +17.55 |
| hop1 | -18.93 [-20.78, -17.17] | -18.93 [-21.05, -16.80] | 17: -17.70, 29: -19.35, 43: -19.75 |
| vs_fixed | +33.87 [+32.70, +35.00] | +33.87 [+32.53, +35.16] | 17: +34.77, 29: +32.17, 43: +34.65 |
| vs_loop | +26.95 [+25.53, +28.33] | +26.95 [+25.38, +28.45] | 17: +27.90, 29: +25.40, 43: +27.55 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.3325833333333333 | >= 0.03 | True |
| reach_ci_positive | 0.3201625 | > 0 | True |
| reach_each_seed_positive | 0.31725 | > 0 | True |
| specificity_ci_positive | 0.30174930555555546 | > 0 | True |
| hop1_degradation_at_most_1pp | -0.18933333333333335 | >= -0.01 | False |
| replication_reach_ci_positive | 0.19466562499999998 | > 0 | True |
| replication_specificity_ci_positive | 0.1762215277777778 | > 0 | True |
| vs_fixed_ci_positive | 0.3253333333333333 | > 0 | True |
| vs_loop_ci_positive | 0.25383229166666665 | > 0 | True |


## oracle_random / M1-lin

| Endpoint | Estimate [95% CI] (pp) | Estimate [97.5% CI] (pp) | Per-seed estimate (pp) |
| --- | --- | --- | --- |
| reach | +47.33 [+45.98, +48.69] | +47.33 [+45.81, +48.83] | 17: +48.08, 29: +46.15, 43: +47.75 |
| far | +2.07 [+1.71, +2.47] | +2.07 [+1.63, +2.54] | 17: +2.22, 29: +1.87, 43: +2.12 |
| specificity | +45.26 [+43.83, +46.63] | +45.26 [+43.68, +46.82] | 17: +45.86, 29: +44.28, 43: +45.63 |
| replication_reach | +31.20 [+29.98, +32.48] | +31.20 [+29.87, +32.68] | 17: +31.52, 29: +31.52, 43: +30.55 |
| replication_far | +1.97 [+1.64, +2.32] | +1.97 [+1.59, +2.39] | 17: +1.95, 29: +2.22, 43: +1.75 |
| replication_specificity | +29.23 [+27.89, +30.51] | +29.23 [+27.70, +30.73] | 17: +29.58, 29: +29.31, 43: +28.80 |
| hop1 | -29.00 [-30.95, -26.93] | -29.00 [-31.32, -26.67] | 17: -28.80, 29: -28.10, 43: -30.10 |
| vs_fixed | +47.34 [+45.97, +48.74] | +47.34 [+45.80, +48.91] | 17: +48.18, 29: +45.92, 43: +47.93 |
| vs_loop | +40.40 [+38.75, +41.99] | +40.40 [+38.59, +42.20] | 17: +41.33, 29: +39.10, 43: +40.77 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.47325 | >= 0.03 | True |
| reach_ci_positive | 0.45808125 | > 0 | True |
| reach_each_seed_positive | 0.4615 | > 0 | True |
| specificity_ci_positive | 0.4368024305555553 | > 0 | True |
| hop1_degradation_at_most_1pp | -0.29 | >= -0.01 | False |
| replication_reach_ci_positive | 0.2986645833333333 | > 0 | True |
| replication_specificity_ci_positive | 0.2770270833333336 | > 0 | True |
| vs_fixed_ci_positive | 0.458 | > 0 | True |
| vs_loop_ci_positive | 0.38591562500000004 | > 0 | True |


## oracle_orth / M1

| Endpoint | Estimate [95% CI] (pp) | Estimate [97.5% CI] (pp) | Per-seed estimate (pp) |
| --- | --- | --- | --- |
| reach | +92.88 [+92.32, +93.50] | +92.88 [+92.23, +93.55] | 17: +93.40, 29: +93.05, 43: +92.20 |
| far | +0.01 [+0.00, +0.03] | +0.01 [+0.00, +0.03] | 17: +0.00, 29: +0.02, 43: +0.02 |
| specificity | +92.87 [+92.29, +93.48] | +92.87 [+92.21, +93.56] | 17: +93.40, 29: +93.03, 43: +92.18 |
| replication_reach | +20.50 [+19.55, +21.51] | +20.50 [+19.47, +21.67] | 17: +21.35, 29: +20.95, 43: +19.20 |
| replication_far | +1.76 [+1.43, +2.08] | +1.76 [+1.39, +2.14] | 17: +2.00, 29: +1.63, 43: +1.65 |
| replication_specificity | +18.74 [+17.74, +19.75] | +18.74 [+17.62, +19.87] | 17: +19.35, 29: +19.32, 43: +17.55 |
| hop1 | +17.78 [+15.67, +19.97] | +17.78 [+15.22, +20.15] | 17: +18.40, 29: +25.05, 43: +9.90 |
| vs_fixed | +92.89 [+92.26, +93.46] | +92.89 [+92.15, +93.54] | 17: +93.40, 29: +93.05, 43: +92.22 |
| vs_loop | +85.23 [+84.32, +86.08] | +85.23 [+84.23, +86.18] | 17: +85.67, 29: +85.17, 43: +84.85 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.9288333333333334 | >= 0.03 | True |
| reach_ci_positive | 0.9223333333333334 | > 0 | True |
| reach_each_seed_positive | 0.922 | > 0 | True |
| specificity_ci_positive | 0.9220829861111111 | > 0 | True |
| hop1_degradation_at_most_1pp | 0.17783333333333332 | >= -0.01 | True |
| replication_reach_ci_positive | 0.19466562499999998 | > 0 | True |
| replication_specificity_ci_positive | 0.1762215277777778 | > 0 | True |
| vs_fixed_ci_positive | 0.9214989583333333 | > 0 | True |
| vs_loop_ci_positive | 0.8423322916666666 | > 0 | True |


## oracle_orth / M1-lin

| Endpoint | Estimate [95% CI] (pp) | Estimate [97.5% CI] (pp) | Per-seed estimate (pp) |
| --- | --- | --- | --- |
| reach | +99.80 [+99.72, +99.87] | +99.80 [+99.70, +99.87] | 17: +99.78, 29: +99.62, 43: +100.00 |
| far | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 17: +0.00, 29: +0.00, 43: +0.00 |
| specificity | +99.80 [+99.72, +99.87] | +99.80 [+99.70, +99.88] | 17: +99.78, 29: +99.62, 43: +100.00 |
| replication_reach | +31.20 [+29.98, +32.48] | +31.20 [+29.87, +32.68] | 17: +31.52, 29: +31.52, 43: +30.55 |
| replication_far | +1.97 [+1.64, +2.32] | +1.97 [+1.59, +2.39] | 17: +1.95, 29: +2.22, 43: +1.75 |
| replication_specificity | +29.23 [+27.89, +30.51] | +29.23 [+27.70, +30.73] | 17: +29.58, 29: +29.31, 43: +28.80 |
| hop1 | -3.02 [-3.53, -2.53] | -3.02 [-3.60, -2.47] | 17: -3.90, 29: -5.15, 43: +0.00 |
| vs_fixed | +99.80 [+99.72, +99.87] | +99.80 [+99.71, +99.88] | 17: +99.78, 29: +99.62, 43: +100.00 |
| vs_loop | +92.14 [+91.51, +92.78] | +92.14 [+91.41, +92.89] | 17: +92.05, 29: +91.75, 43: +92.62 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.9979999999999999 | >= 0.03 | True |
| reach_ci_positive | 0.997 | > 0 | True |
| reach_each_seed_positive | 0.99625 | > 0 | True |
| specificity_ci_positive | 0.997 | > 0 | True |
| hop1_degradation_at_most_1pp | -0.030166666666666665 | >= -0.01 | False |
| replication_reach_ci_positive | 0.2986645833333333 | > 0 | True |
| replication_specificity_ci_positive | 0.2770270833333336 | > 0 | True |
| vs_fixed_ci_positive | 0.9970833333333333 | > 0 | True |
| vs_loop_ci_positive | 0.9140822916666665 | > 0 | True |


## t1_frozen / M1

| Endpoint | Estimate [95% CI] (pp) | Estimate [97.5% CI] (pp) | Per-seed estimate (pp) |
| --- | --- | --- | --- |
| reach | +8.12 [+7.43, +8.88] | +8.12 [+7.36, +9.00] | 17: +9.15, 29: +6.78, 43: +8.45 |
| far | +6.44 [+5.76, +7.16] | +6.44 [+5.64, +7.24] | 17: +7.43, 29: +6.10, 43: +5.78 |
| specificity | +1.69 [+0.75, +2.57] | +1.69 [+0.57, +2.73] | 17: +1.72, 29: +0.68, 43: +2.67 |
| replication_reach | +5.16 [+4.37, +5.90] | +5.16 [+4.27, +5.97] | 17: +5.33, 29: +4.95, 43: +5.20 |
| replication_far | +4.48 [+3.78, +5.16] | +4.48 [+3.67, +5.30] | 17: +4.85, 29: +5.05, 43: +3.55 |
| replication_specificity | +0.68 [+0.02, +1.38] | +0.68 [-0.07, +1.49] | 17: +0.48, 29: -0.10, 43: +1.65 |
| hop1 | +9.03 [+7.87, +10.18] | +9.03 [+7.67, +10.35] | 17: +9.30, 29: +8.90, 43: +8.90 |
| vs_fixed | +4.64 [+3.82, +5.42] | +4.64 [+3.68, +5.50] | 17: +6.68, 29: +2.83, 43: +4.42 |
| vs_loop | +16.41 [+15.45, +17.39] | +16.41 [+15.33, +17.55] | 17: +20.42, 29: +13.33, 43: +15.47 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.08125 | >= 0.03 | True |
| reach_ci_positive | 0.07358333333333333 | > 0 | True |
| reach_each_seed_positive | 0.06775 | > 0 | True |
| specificity_ci_positive | 0.005720833333333338 | > 0 | True |
| hop1_degradation_at_most_1pp | 0.09033333333333333 | >= -0.01 | True |
| replication_reach_ci_positive | 0.042666666666666665 | > 0 | True |
| replication_specificity_ci_positive | -0.0006961805555555509 | > 0 | False |
| vs_fixed_ci_positive | 0.036750000000000005 | > 0 | True |
| vs_loop_ci_positive | 0.15325000000000003 | > 0 | True |


## t1_frozen / M1-lin

| Endpoint | Estimate [95% CI] (pp) | Estimate [97.5% CI] (pp) | Per-seed estimate (pp) |
| --- | --- | --- | --- |
| reach | +10.29 [+9.53, +11.07] | +10.29 [+9.43, +11.19] | 17: +10.95, 29: +10.27, 43: +9.65 |
| far | +7.43 [+6.70, +8.14] | +7.43 [+6.61, +8.23] | 17: +8.65, 29: +6.53, 43: +7.12 |
| specificity | +2.86 [+1.92, +3.80] | +2.86 [+1.72, +3.99] | 17: +2.30, 29: +3.74, 43: +2.53 |
| replication_reach | +5.17 [+4.38, +5.97] | +5.17 [+4.26, +6.08] | 17: +6.40, 29: +5.40, 43: +3.70 |
| replication_far | +4.97 [+4.25, +5.71] | +4.97 [+4.13, +5.81] | 17: +5.77, 29: +5.13, 43: +4.00 |
| replication_specificity | +0.20 [-0.53, +0.91] | +0.20 [-0.65, +1.05] | 17: +0.63, 29: +0.27, 43: -0.30 |
| hop1 | +13.52 [+12.35, +14.72] | +13.52 [+12.22, +14.85] | 17: +13.15, 29: +14.30, 43: +13.10 |
| vs_fixed | +7.68 [+6.82, +8.49] | +7.68 [+6.69, +8.62] | 17: +9.85, 29: +6.53, 43: +6.68 |
| vs_loop | +20.29 [+19.24, +21.33] | +20.29 [+19.09, +21.48] | 17: +23.52, 29: +18.15, 43: +19.20 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.10291666666666666 | >= 0.03 | True |
| reach_ci_positive | 0.09425 | > 0 | True |
| reach_each_seed_positive | 0.0965 | > 0 | True |
| specificity_ci_positive | 0.017165972222222194 | > 0 | True |
| hop1_degradation_at_most_1pp | 0.13516666666666666 | >= -0.01 | True |
| replication_reach_ci_positive | 0.042583333333333334 | > 0 | True |
| replication_specificity_ci_positive | -0.00647326388888888 | > 0 | False |
| vs_fixed_ci_positive | 0.06691666666666667 | > 0 | True |
| vs_loop_ci_positive | 0.19091666666666665 | > 0 | True |


## joint / M1

| Endpoint | Estimate [95% CI] (pp) | Estimate [97.5% CI] (pp) | Per-seed estimate (pp) |
| --- | --- | --- | --- |
| reach | +8.29 [+7.48, +9.15] | +8.29 [+7.38, +9.29] | 17: +14.92, 29: +1.32, 43: +8.62 |
| far | +5.80 [+4.94, +6.59] | +5.80 [+4.83, +6.70] | 17: +9.40, 29: +1.33, 43: +6.67 |
| specificity | +2.49 [+1.39, +3.59] | +2.49 [+1.24, +3.73] | 17: +5.53, 29: -0.01, 43: +1.96 |
| replication_reach | +37.47 [+36.22, +38.78] | +37.47 [+36.07, +38.93] | 17: +37.72, 29: +37.05, 43: +37.62 |
| replication_far | +32.77 [+31.53, +33.97] | +32.77 [+31.37, +34.14] | 17: +32.50, 29: +33.03, 43: +32.77 |
| replication_specificity | +4.70 [+3.70, +5.67] | +4.70 [+3.61, +5.82] | 17: +5.23, 29: +4.02, 43: +4.86 |
| hop1 | +4.68 [+3.70, +5.67] | +4.68 [+3.62, +5.80] | 17: +6.80, 29: +2.25, 43: +5.00 |
| vs_fixed | +7.77 [+6.68, +8.86] | +7.77 [+6.52, +9.05] | 17: +9.30, 29: -17.70, 43: +31.70 |
| vs_loop | +35.26 [+34.06, +36.43] | +35.26 [+33.89, +36.55] | 17: +46.30, 29: +17.55, 43: +41.93 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.08291666666666667 | >= 0.03 | True |
| reach_ci_positive | 0.07383125 | > 0 | True |
| reach_each_seed_positive | 0.01325 | > 0 | True |
| specificity_ci_positive | 0.012444097222222204 | > 0 | True |
| hop1_degradation_at_most_1pp | 0.04683333333333334 | >= -0.01 | True |
| replication_reach_ci_positive | 0.36074999999999996 | > 0 | True |
| replication_specificity_ci_positive | 0.036055555555555584 | > 0 | True |
| vs_fixed_ci_positive | 0.06516458333333333 | > 0 | True |
| vs_loop_ci_positive | 0.3389135416666667 | > 0 | True |


## joint / M1-lin

| Endpoint | Estimate [95% CI] (pp) | Estimate [97.5% CI] (pp) | Per-seed estimate (pp) |
| --- | --- | --- | --- |
| reach | +2.44 [+1.78, +3.08] | +2.44 [+1.72, +3.19] | 17: +2.97, 29: +0.88, 43: +3.48 |
| far | +2.85 [+2.04, +3.62] | +2.85 [+1.97, +3.74] | 17: +2.93, 29: +2.63, 43: +2.98 |
| specificity | -0.41 [-1.37, +0.59] | -0.41 [-1.47, +0.75] | 17: +0.04, 29: -1.76, 43: +0.49 |
| replication_reach | +23.05 [+21.87, +24.38] | +23.05 [+21.67, +24.56] | 17: +24.80, 29: +21.22, 43: +23.12 |
| replication_far | +22.84 [+21.71, +24.04] | +22.84 [+21.51, +24.20] | 17: +24.62, 29: +21.32, 43: +22.58 |
| replication_specificity | +0.21 [-0.34, +0.75] | +0.21 [-0.42, +0.86] | 17: +0.18, 29: -0.09, 43: +0.54 |
| hop1 | +2.80 [+1.92, +3.68] | +2.80 [+1.80, +3.78] | 17: +2.05, 29: +2.90, 43: +3.45 |
| vs_fixed | +3.08 [+1.98, +4.27] | +3.08 [+1.78, +4.41] | 17: +1.88, 29: +2.57, 43: +4.78 |
| vs_loop | +39.78 [+38.53, +41.02] | +39.78 [+38.36, +41.27] | 17: +40.33, 29: +39.65, 43: +39.38 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.02441666666666667 | >= 0.03 | False |
| reach_ci_positive | 0.017166666666666667 | > 0 | True |
| reach_each_seed_positive | 0.00875 | > 0 | True |
| specificity_ci_positive | -0.014749999999999997 | > 0 | False |
| hop1_degradation_at_most_1pp | 0.028 | >= -0.01 | True |
| replication_reach_ci_positive | 0.21666666666666667 | > 0 | True |
| replication_specificity_ci_positive | -0.004194791666666665 | > 0 | False |
| vs_fixed_ci_positive | 0.017831250000000003 | > 0 | True |
| vs_loop_ci_positive | 0.38358333333333333 | > 0 | True |
