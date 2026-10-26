# F1 confirmation gates

Differences and intervals are percentage points. Decisions use unrounded 97.5% intervals. F1 omits replication and is a weaker design.

| Writer | r | Reader | G1 | G2 | Pass |
| --- | --- | --- | --- | --- | --- |
| joint | 64 | M1 | True | True | True |
| joint | 64 | M1-lin | True | True | True |
| joint | 128 | M1 | True | True | True |
| joint | 128 | M1-lin | True | True | True |
| oracle_random | 64 | M1 | False | True | False |
| oracle_random | 64 | M1-lin | False | True | False |
| oracle_random | 128 | M1 | False | True | False |
| oracle_random | 128 | M1-lin | False | True | False |


## joint / r=64 / M1

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +30.53 [+29.37, +31.71] | +30.53 [+29.22, +31.90] | 17: +29.83, 29: +33.00, 43: +28.78 |
| far | +9.61 [+8.35, +10.86] | +9.61 [+8.14, +11.05] | 17: +8.63, 29: +11.78, 43: +8.40 |
| specificity | +20.93 [+19.26, +22.69] | +20.93 [+19.05, +22.93] | 17: +21.19, 29: +21.22, 43: +20.38 |
| hop1 | +9.02 [+7.98, +10.02] | +9.02 [+7.87, +10.17] | 17: +9.80, 29: +6.60, 43: +10.65 |
| vs_fixed | +24.02 [+22.73, +25.37] | +24.02 [+22.57, +25.58] | 17: +22.57, 29: +25.07, 43: +24.40 |
| vs_loop | +62.10 [+60.82, +63.39] | +62.10 [+60.65, +63.58] | 17: +61.58, 29: +62.38, 43: +62.35 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.3053333333333333 | >= 0.03 | True |
| reach_ci_positive | 0.2922489583333333 | > 0.0 | True |
| reach_each_seed_positive | 0.28775 | > 0.0 | True |
| specificity_ci_positive | 0.19052361111111127 | > 0.0 | True |
| hop1_degradation_at_most_1pp | 0.09016666666666667 | >= -0.01 | True |
| vs_fixed_ci_positive | 0.22575 | > 0.0 | True |
| vs_loop_ci_positive | 0.6064989583333333 | > 0.0 | True |


## joint / r=64 / M1-lin

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +22.42 [+21.38, +23.42] | +22.42 [+21.23, +23.52] | 17: +21.88, 29: +23.60, 43: +21.80 |
| far | +7.20 [+6.11, +8.28] | +7.20 [+5.93, +8.43] | 17: +7.45, 29: +7.48, 43: +6.67 |
| specificity | +15.23 [+13.76, +16.72] | +15.23 [+13.50, +16.90] | 17: +14.42, 29: +16.12, 43: +15.13 |
| hop1 | +6.08 [+5.08, +7.08] | +6.08 [+4.93, +7.25] | 17: +5.95, 29: +5.60, 43: +6.70 |
| vs_fixed | +21.17 [+19.96, +22.30] | +21.17 [+19.80, +22.46] | 17: +21.40, 29: +22.18, 43: +19.93 |
| vs_loop | +58.82 [+57.55, +60.13] | +58.82 [+57.41, +60.29] | 17: +59.20, 29: +58.65, 43: +58.60 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.22424999999999998 | >= 0.03 | True |
| reach_ci_positive | 0.21233125000000003 | > 0.0 | True |
| reach_each_seed_positive | 0.218 | > 0.0 | True |
| specificity_ci_positive | 0.1350493055555555 | > 0.0 | True |
| hop1_degradation_at_most_1pp | 0.06083333333333333 | >= -0.01 | True |
| vs_fixed_ci_positive | 0.1979989583333333 | > 0.0 | True |
| vs_loop_ci_positive | 0.5740812500000001 | > 0.0 | True |


## joint / r=128 / M1

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +33.99 [+32.82, +35.17] | +33.99 [+32.65, +35.28] | 17: +30.95, 29: +37.97, 43: +33.05 |
| far | +9.42 [+8.18, +10.67] | +9.42 [+8.04, +10.94] | 17: +6.58, 29: +13.00, 43: +8.67 |
| specificity | +24.57 [+22.91, +26.29] | +24.57 [+22.68, +26.56] | 17: +24.37, 29: +24.98, 43: +24.38 |
| hop1 | +8.42 [+7.48, +9.38] | +8.42 [+7.37, +9.50] | 17: +9.65, 29: +6.15, 43: +9.45 |
| vs_fixed | +27.15 [+25.88, +28.44] | +27.15 [+25.69, +28.55] | 17: +26.67, 29: +27.18, 43: +27.60 |
| vs_loop | +66.14 [+64.97, +67.41] | +66.14 [+64.80, +67.59] | 17: +65.58, 29: +66.70, 43: +66.15 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.33991666666666664 | >= 0.03 | True |
| reach_ci_positive | 0.32649791666666667 | > 0.0 | True |
| reach_each_seed_positive | 0.3095 | > 0.0 | True |
| specificity_ci_positive | 0.22677673611111107 | > 0.0 | True |
| hop1_degradation_at_most_1pp | 0.08416666666666667 | >= -0.01 | True |
| vs_fixed_ci_positive | 0.2569166666666667 | > 0.0 | True |
| vs_loop_ci_positive | 0.648 | > 0.0 | True |


## joint / r=128 / M1-lin

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +40.59 [+39.31, +41.83] | +40.59 [+39.13, +41.98] | 17: +39.38, 29: +42.73, 43: +39.67 |
| far | +7.82 [+6.43, +9.20] | +7.82 [+6.32, +9.39] | 17: +7.63, 29: +10.00, 43: +5.82 |
| specificity | +32.77 [+31.01, +34.69] | +32.77 [+30.73, +35.01] | 17: +31.74, 29: +32.73, 43: +33.86 |
| hop1 | +5.57 [+4.73, +6.40] | +5.57 [+4.62, +6.47] | 17: +6.40, 29: +5.10, 43: +5.20 |
| vs_fixed | +36.28 [+35.00, +37.58] | +36.28 [+34.86, +37.82] | 17: +35.58, 29: +37.33, 43: +35.93 |
| vs_loop | +75.27 [+74.14, +76.41] | +75.27 [+73.96, +76.61] | 17: +74.58, 29: +76.10, 43: +75.12 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.40591666666666665 | >= 0.03 | True |
| reach_ci_positive | 0.3913333333333333 | > 0.0 | True |
| reach_each_seed_positive | 0.39375 | > 0.0 | True |
| specificity_ci_positive | 0.3072770833333333 | > 0.0 | True |
| hop1_degradation_at_most_1pp | 0.05566666666666666 | >= -0.01 | True |
| vs_fixed_ci_positive | 0.3485833333333333 | > 0.0 | True |
| vs_loop_ci_positive | 0.7395812500000001 | > 0.0 | True |


## oracle_random / r=64 / M1

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +64.98 [+63.91, +66.07] | +64.98 [+63.75, +66.18] | 17: +65.77, 29: +64.45, 43: +64.72 |
| far | +0.93 [+0.75, +1.13] | +0.93 [+0.72, +1.16] | 17: +0.95, 29: +0.97, 43: +0.88 |
| specificity | +64.05 [+63.01, +65.16] | +64.05 [+62.87, +65.33] | 17: +64.83, 29: +63.48, 43: +63.84 |
| hop1 | -33.10 [-34.60, -31.45] | -33.10 [-34.78, -31.28] | 17: -32.10, 29: -32.95, 43: -34.25 |
| vs_fixed | +64.98 [+63.85, +66.06] | +64.98 [+63.72, +66.16] | 17: +65.72, 29: +64.50, 43: +64.72 |
| vs_loop | +56.37 [+54.96, +57.67] | +56.37 [+54.73, +57.78] | 17: +57.23, 29: +55.85, 43: +56.03 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.6498333333333334 | >= 0.03 | True |
| reach_ci_positive | 0.6374979166666667 | > 0.0 | True |
| reach_each_seed_positive | 0.6445 | > 0.0 | True |
| specificity_ci_positive | 0.6286649305555555 | > 0.0 | True |
| hop1_degradation_at_most_1pp | -0.331 | >= -0.01 | False |
| vs_fixed_ci_positive | 0.637165625 | > 0.0 | True |
| vs_loop_ci_positive | 0.54733125 | > 0.0 | True |


## oracle_random / r=64 / M1-lin

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +74.11 [+72.97, +75.22] | +74.11 [+72.79, +75.36] | 17: +73.38, 29: +74.08, 43: +74.88 |
| far | +0.51 [+0.37, +0.66] | +0.51 [+0.36, +0.69] | 17: +0.65, 29: +0.45, 43: +0.42 |
| specificity | +73.60 [+72.50, +74.74] | +73.60 [+72.39, +74.91] | 17: +72.72, 29: +73.62, 43: +74.46 |
| hop1 | -30.83 [-32.43, -29.25] | -30.83 [-32.65, -29.03] | 17: -29.95, 29: -30.10, 43: -32.45 |
| vs_fixed | +74.09 [+72.98, +75.18] | +74.09 [+72.77, +75.33] | 17: +73.35, 29: +74.10, 43: +74.83 |
| vs_loop | +65.42 [+64.03, +66.73] | +65.42 [+63.79, +66.85] | 17: +64.80, 29: +65.35, 43: +66.12 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.7410833333333334 | >= 0.03 | True |
| reach_ci_positive | 0.7279166666666667 | > 0.0 | True |
| reach_each_seed_positive | 0.73375 | > 0.0 | True |
| specificity_ci_positive | 0.7238881944444447 | > 0.0 | True |
| hop1_degradation_at_most_1pp | -0.30833333333333335 | >= -0.01 | False |
| vs_fixed_ci_positive | 0.7277479166666667 | > 0.0 | True |
| vs_loop_ci_positive | 0.6379135416666666 | > 0.0 | True |


## oracle_random / r=128 / M1

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +84.23 [+83.42, +85.03] | +84.23 [+83.25, +85.14] | 17: +83.20, 29: +84.97, 43: +84.52 |
| far | +0.22 [+0.14, +0.29] | +0.22 [+0.13, +0.31] | 17: +0.25, 29: +0.28, 43: +0.12 |
| specificity | +84.02 [+83.24, +84.75] | +84.02 [+83.08, +84.84] | 17: +82.95, 29: +84.69, 43: +84.41 |
| hop1 | -21.55 [-22.65, -20.35] | -21.55 [-22.87, -20.17] | 17: -20.10, 29: -22.30, 43: -22.25 |
| vs_fixed | +84.22 [+83.45, +85.01] | +84.22 [+83.35, +85.11] | 17: +83.20, 29: +84.95, 43: +84.52 |
| vs_loop | +76.17 [+75.17, +77.18] | +76.17 [+75.02, +77.32] | 17: +75.15, 29: +76.83, 43: +76.55 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.8423333333333334 | >= 0.03 | True |
| reach_ci_positive | 0.8325 | > 0.0 | True |
| reach_each_seed_positive | 0.832 | > 0.0 | True |
| specificity_ci_positive | 0.8308052083333335 | > 0.0 | True |
| hop1_degradation_at_most_1pp | -0.21550000000000002 | >= -0.01 | False |
| vs_fixed_ci_positive | 0.8334979166666666 | > 0.0 | True |
| vs_loop_ci_positive | 0.7501666666666665 | > 0.0 | True |


## oracle_random / r=128 / M1-lin

| Endpoint | Estimate [95% CI] | Estimate [97.5% CI] | Per-seed effect (pp) |
| --- | --- | --- | --- |
| reach | +84.51 [+83.68, +85.30] | +84.51 [+83.54, +85.44] | 17: +84.65, 29: +84.45, 43: +84.42 |
| far | +0.07 [+0.03, +0.11] | +0.07 [+0.03, +0.12] | 17: +0.05, 29: +0.08, 43: +0.07 |
| specificity | +84.44 [+83.63, +85.22] | +84.44 [+83.53, +85.38] | 17: +84.60, 29: +84.37, 43: +84.36 |
| hop1 | -17.43 [-18.62, -16.27] | -17.43 [-18.80, -16.13] | 17: -16.10, 29: -18.40, 43: -17.80 |
| vs_fixed | +84.51 [+83.70, +85.35] | +84.51 [+83.58, +85.48] | 17: +84.65, 29: +84.45, 43: +84.42 |
| vs_loop | +76.45 [+75.39, +77.55] | +76.45 [+75.23, +77.68] | 17: +76.60, 29: +76.30, 43: +76.45 |


| Criterion | Raw value | Threshold | Pass |
| --- | --- | --- | --- |
| reach_at_least_3pp | 0.8450833333333333 | >= 0.03 | True |
| reach_ci_positive | 0.8354145833333334 | > 0.0 | True |
| reach_each_seed_positive | 0.84425 | > 0.0 | True |
| specificity_ci_positive | 0.8353055555555557 | > 0.0 | True |
| hop1_degradation_at_most_1pp | -0.1743333333333333 | >= -0.01 | False |
| vs_fixed_ci_positive | 0.8358322916666666 | > 0.0 | True |
| vs_loop_ci_positive | 0.7523322916666667 | > 0.0 | True |
