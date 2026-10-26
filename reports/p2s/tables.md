# P2-S tables

All differences are M1/r32@L4 minus the named comparator, in percentage points. B6/B7 use L1; B8 uses L4. Intervals are pooled paired 97.5% episode-bootstrap intervals (2000 draws, 2000 test episodes, seed 888001). Per-seed columns are 17, 29, 43; CIs condition on these seeds. Decisions use unrounded values. Both confidence levels and absolute per-seed accuracies remain in endpoints.csv.

## E1–E3 primary: hops 2–3

| Endpoint | Writer | Comparator budget | Estimate [97.5% CI] | Seed 17 | Seed 29 | Seed 43 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | joint | B6/r64 | +14.875 [+13.658, +16.184] | +33.125 | -20.675 | +32.175 |
| E1 | joint | B6/r128 | -3.233 [-4.675, -1.833] | +7.175 | -21.600 | +4.725 |
| E1 | oracle_random | B6/r64 | +34.750 [+33.467, +36.075] | +35.775 | +34.200 | +34.275 |
| E1 | oracle_random | B6/r128 | +34.775 [+33.492, +36.083] | +35.775 | +34.225 | +34.325 |
| E2 | joint | B7/r32/m64 | -1.458 [-2.858, -0.125] | +9.100 | -20.350 | +6.875 |
| E2 | joint | B7/r32/m128 | -2.283 [-3.592, -0.892] | +7.850 | -20.425 | +5.725 |
| E3 | joint | B8/r32 | -33.733 [-34.942, -32.433] | -26.400 | -52.200 | -22.600 |
| E3 | oracle_random | B8/r32 | -11.442 [-12.792, -10.058] | -10.125 | -12.725 | -11.475 |

## Accompanying hop-1 differences

| Endpoint | Writer | Comparator budget | Estimate [97.5% CI] | Seed 17 | Seed 29 | Seed 43 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | joint | B6/r64 | +19.417 [+17.717, +21.000] | +56.200 | -47.550 | +49.600 |
| E1 | joint | B6/r128 | -21.217 [-22.900, -19.583] | -2.100 | -53.000 | -8.550 |
| E1 | oracle_random | B6/r64 | -49.650 [-51.417, -47.850] | -50.600 | -48.600 | -49.750 |
| E1 | oracle_random | B6/r128 | -49.933 [-51.717, -48.116] | -50.900 | -48.900 | -50.000 |
| E2 | joint | B7/r32/m64 | -7.667 [-9.450, -5.950] | +11.250 | -39.450 | +5.200 |
| E2 | joint | B7/r32/m128 | -8.167 [-9.883, -6.467] | +10.050 | -40.400 | +5.850 |
| E3 | joint | B8/r32 | -32.067 [-33.517, -30.600] | -12.550 | -65.450 | -18.200 |
| E3 | oracle_random | B8/r32 | -12.367 [-14.250, -10.633] | -13.500 | -11.100 | -12.500 |

## Accompanying hops 4–8 mean differences

| Endpoint | Writer | Comparator budget | Estimate [97.5% CI] | Seed 17 | Seed 29 | Seed 43 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | joint | B6/r64 | +10.773 [+9.540, +12.027] | +25.640 | -21.000 | +27.680 |
| E1 | joint | B6/r128 | -7.583 [-8.963, -6.213] | -2.610 | -20.740 | +0.600 |
| E1 | oracle_random | B6/r64 | +3.590 [+3.283, +3.920] | +3.510 | +3.430 | +3.830 |
| E1 | oracle_random | B6/r128 | +3.593 [+3.283, +3.920] | +3.510 | +3.430 | +3.840 |
| E2 | joint | B7/r32/m64 | -4.703 [-5.987, -3.367] | +2.240 | -19.310 | +2.960 |
| E2 | joint | B7/r32/m128 | -5.533 [-6.887, -4.210] | +0.890 | -19.650 | +2.160 |
| E3 | joint | B8/r32 | +3.273 [+2.060, +4.560] | +10.410 | -10.800 | +10.210 |
| E3 | oracle_random | B8/r32 | +1.740 [+1.430, +2.073] | +1.670 | +1.590 | +1.960 |
## E4 selected primary points

Three-seed means. Accuracy columns are percentages; GPU columns are CUDA-event milliseconds per prepared query batch. State is FP32 KiB per episode. Read FLOPs omit input/output projection, decoding and elementwise work; the timed operation includes output projection and candidate decoding. The complete 212-point table (including every defined L1/2/4/8 point and F1 descriptive-only 20000-step references) is cost.csv; 636 seed-level rows are in cost_per_seed.csv.

| Writer | Variant (r, m, L) | Trainable params | State KiB | Read FLOPs | Acc 1–3 | Acc 1–8 | GPU batch64 ms | GPU batch1 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| joint | B6 (64, 64, 1) | 36,866 | 16 | 32,768 | 31.63 | 29.82 | 0.358 | 0.313 |
| joint | B6 (128, 128, 1) | 114,690 | 64 | 131,072 | 57.24 | 50.90 | 0.352 | 0.313 |
| joint | B7 (32, 64, 1) | 17,410 | 16 | 18,432 | 51.54 | 46.96 | 0.351 | 0.312 |
| joint | B7 (32, 128, 1) | 25,602 | 64 | 51,200 | 52.26 | 47.75 | 0.356 | 0.314 |
| joint | B8 (32, 32, 4) | 22,530 | 4 | 32,768 | 81.19 | 53.10 | 0.908 | 0.799 |
| oracle_random | B6 (64, 64, 1) | 16,384 | 16 | 32,768 | 33.26 | 12.47 | 0.352 | 0.312 |
| oracle_random | B6 (128, 128, 1) | 65,536 | 64 | 131,072 | 33.33 | 12.50 | 0.354 | 0.311 |
| oracle_random | B8 (32, 32, 4) | 13,312 | 4 | 32,768 | 51.62 | 20.52 | 0.907 | 0.799 |
| joint | M1 (32, 32, 4) | 13,314 | 4 | 32,768 | 48.02 | 42.70 | 0.848 | 0.758 |
| oracle_random | M1 (32, 32, 4) | 4,096 | 4 | 32,768 | 39.87 | 17.20 | 0.843 | 0.761 |

## Practical eligibility

Only 15000-step L1 competitors enter this check, within writer and batch size. Equal accuracy at equal or lower mean GPU latency matches M1. F1 is excluded. The overall practical claim also requires all E1/E2 capability contrasts to pass.

| Writer | Batch size | Unmatched M1/r32 depths |
| --- | --- | --- |
| joint | 64 | none |
| joint | 1 | none |
| oracle_random | 64 | 4 |
| oracle_random | 1 | 4 |
