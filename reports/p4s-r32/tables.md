# WP-C4 screening tables

Accuracy is percent; deltas are percentage points. Equal means over seeds 17/29/43 and the specified hops. Thresholds maximize validation accuracy over hops 1–8, as WP-C3. No bootstrap was implemented in WP-C3; all quantities here are screening point estimates.

First qualifying post-update state z^L, 1≤L≤16; read norm costs L+1 reads, other rules L. True-hop stopping is label-informed, not an upper bound. The candidate flag requires hop-1 delta ≥−1 pp, positive reach and positive walk specificity; it is not formal G1 acceptance.

## joint/r32/M1

Validation selection:

| Rule | Threshold | Val accuracy (%) | Val steps |
| --- | --- | --- | --- |
| read_norm | 0.3 | 43.375 | 15.999 |
| relative_change | 0 | 43.373 | 16.000 |
| output_stability | 16 | 43.373 | 16.000 |

Screening test:

| Method | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | Overall | Steps | Reads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_1 | 53.38 | 35.62 | 35.43 | 33.67 | 34.50 | 35.00 | 35.60 | 33.13 | 37.04 | 1.000 | 1.000 |
| fixed_4 | 58.55 | 46.48 | 40.10 | 40.30 | 40.40 | 40.38 | 40.23 | 39.95 | 43.30 | 4.000 | 4.000 |
| fixed_8 | 58.48 | 47.48 | 40.97 | 40.70 | 41.45 | 40.50 | 40.20 | 41.00 | 43.85 | 8.000 | 8.000 |
| fixed_16 | 58.53 | 47.37 | 41.02 | 40.63 | 41.40 | 40.43 | 40.42 | 40.97 | 43.85 | 16.000 | 16.000 |
| read_norm | 58.53 | 47.37 | 41.02 | 40.63 | 41.40 | 40.43 | 40.42 | 40.97 | 43.85 | 16.000 | 17.000 |
| relative_change | 58.53 | 47.37 | 41.02 | 40.63 | 41.40 | 40.43 | 40.42 | 40.97 | 43.85 | 16.000 | 16.000 |
| output_stability | 58.53 | 47.37 | 41.02 | 40.63 | 41.40 | 40.43 | 40.42 | 40.97 | 43.85 | 16.000 | 16.000 |
| oracle_true_hop | 53.38 | 44.98 | 39.03 | 40.30 | 41.07 | 40.03 | 40.02 | 41.00 | 42.48 | 4.500 | 4.500 |

Stopped-minus-L1 screening endpoints:

| Method | Hop1 Δ | Reach Δ (2,3) | Far Δ (6,7,8) | Walk specificity | Candidate |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | +0.000 | +0.000 | +0.000 | +0.000 | False |
| fixed_16 | +5.150 | +8.667 | +6.028 | +2.639 | True |
| fixed_4 | +5.167 | +7.767 | +5.611 | +2.156 | True |
| fixed_8 | +5.100 | +8.700 | +5.989 | +2.711 | True |
| oracle_true_hop | +0.000 | +6.483 | +5.772 | +0.711 | True |
| output_stability | +5.150 | +8.667 | +6.028 | +2.639 | True |
| read_norm | +5.150 | +8.667 | +6.028 | +2.639 | True |
| relative_change | +5.150 | +8.667 | +6.028 | +2.639 | True |

## joint/r32/M1-lin

Validation selection:

| Rule | Threshold | Val accuracy (%) | Val steps |
| --- | --- | --- | --- |
| read_norm | 0 | 48.490 | 16.000 |
| relative_change | 0.03 | 48.527 | 6.184 |
| output_stability | 3 | 48.492 | 4.330 |

Screening test:

| Method | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | Overall | Steps | Reads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_1 | 46.40 | 46.72 | 45.40 | 45.03 | 45.48 | 45.35 | 45.62 | 44.50 | 45.56 | 1.000 | 1.000 |
| fixed_4 | 49.28 | 49.53 | 48.48 | 47.92 | 48.48 | 48.62 | 48.20 | 48.02 | 48.57 | 4.000 | 4.000 |
| fixed_8 | 49.25 | 49.38 | 48.20 | 47.75 | 48.40 | 48.33 | 48.12 | 47.92 | 48.42 | 8.000 | 8.000 |
| fixed_16 | 49.18 | 49.42 | 48.13 | 47.75 | 48.30 | 48.40 | 48.13 | 47.77 | 48.39 | 16.000 | 16.000 |
| read_norm | 49.18 | 49.42 | 48.13 | 47.75 | 48.30 | 48.40 | 48.13 | 47.77 | 48.39 | 16.000 | 17.000 |
| relative_change | 49.12 | 49.33 | 48.18 | 47.68 | 48.38 | 48.33 | 48.17 | 47.72 | 48.36 | 6.137 | 6.137 |
| output_stability | 49.18 | 49.45 | 48.15 | 47.72 | 48.25 | 48.37 | 48.15 | 47.75 | 48.38 | 4.350 | 4.350 |
| oracle_true_hop | 46.40 | 49.53 | 48.07 | 47.92 | 48.18 | 48.37 | 48.15 | 47.92 | 48.07 | 4.500 | 4.500 |

Stopped-minus-L1 screening endpoints:

| Method | Hop1 Δ | Reach Δ (2,3) | Far Δ (6,7,8) | Walk specificity | Candidate |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | +0.000 | +0.000 | +0.000 | +0.000 | False |
| fixed_16 | +2.783 | +2.717 | +2.944 | -0.228 | False |
| fixed_4 | +2.883 | +2.950 | +3.122 | -0.172 | False |
| fixed_8 | +2.850 | +2.733 | +2.967 | -0.233 | False |
| oracle_true_hop | +0.000 | +2.742 | +2.989 | -0.247 | False |
| output_stability | +2.783 | +2.742 | +2.933 | -0.192 | False |
| read_norm | +2.783 | +2.717 | +2.944 | -0.228 | False |
| relative_change | +2.717 | +2.700 | +2.917 | -0.217 | False |

## joint_fact/r32/M1

Validation selection:

| Rule | Threshold | Val accuracy (%) | Val steps |
| --- | --- | --- | --- |
| read_norm | 1 | 15.854 | 13.275 |
| relative_change | 0.3 | 15.956 | 6.687 |
| output_stability | 4 | 16.206 | 9.908 |

Screening test:

| Method | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | Overall | Steps | Reads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_1 | 84.17 | 1.37 | 1.37 | 1.45 | 1.47 | 1.50 | 1.40 | 1.15 | 11.73 | 1.000 | 1.000 |
| fixed_4 | 50.37 | 33.80 | 16.73 | 5.08 | 3.05 | 3.58 | 3.43 | 3.45 | 14.94 | 4.000 | 4.000 |
| fixed_8 | 41.10 | 23.65 | 16.67 | 13.47 | 12.97 | 8.48 | 5.17 | 3.63 | 15.64 | 8.000 | 8.000 |
| fixed_16 | 37.57 | 22.80 | 16.15 | 13.18 | 10.68 | 8.40 | 7.12 | 6.08 | 15.25 | 16.000 | 16.000 |
| read_norm | 43.77 | 22.65 | 15.68 | 12.67 | 10.38 | 8.05 | 6.65 | 5.62 | 15.68 | 13.253 | 14.253 |
| relative_change | 51.13 | 28.43 | 17.00 | 10.43 | 7.43 | 5.20 | 4.87 | 4.32 | 16.10 | 6.712 | 6.712 |
| output_stability | 49.98 | 25.12 | 16.50 | 12.07 | 9.83 | 6.55 | 5.82 | 4.87 | 16.34 | 9.895 | 9.895 |
| oracle_true_hop | 84.17 | 17.67 | 6.90 | 5.08 | 4.70 | 3.78 | 3.80 | 3.63 | 16.22 | 4.500 | 4.500 |

Stopped-minus-L1 screening endpoints:

| Method | Hop1 Δ | Reach Δ (2,3) | Far Δ (6,7,8) | Walk specificity | Candidate |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | +0.000 | +0.000 | +0.000 | +0.000 | False |
| fixed_16 | -46.600 | +18.108 | +5.850 | +12.258 | False |
| fixed_4 | -33.800 | +23.900 | +2.139 | +21.761 | False |
| fixed_8 | -43.067 | +18.792 | +4.411 | +14.381 | False |
| oracle_true_hop | +0.000 | +10.917 | +2.389 | +8.528 | True |
| output_stability | -34.183 | +19.442 | +4.394 | +15.047 | False |
| read_norm | -40.400 | +17.800 | +5.422 | +12.378 | False |
| relative_change | -33.033 | +21.350 | +3.444 | +17.906 | False |

## joint_fact/r32/M1-lin

Validation selection:

| Rule | Threshold | Val accuracy (%) | Val steps |
| --- | --- | --- | --- |
| read_norm | 1 | 16.248 | 13.222 |
| relative_change | 0.3 | 17.283 | 10.489 |
| output_stability | 4 | 17.933 | 10.579 |

Screening test:

| Method | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | Overall | Steps | Reads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_1 | 89.25 | 1.18 | 0.92 | 0.97 | 0.90 | 1.03 | 1.08 | 0.80 | 12.02 | 1.000 | 1.000 |
| fixed_4 | 49.88 | 37.45 | 19.95 | 4.27 | 2.33 | 3.03 | 3.32 | 3.42 | 15.46 | 4.000 | 4.000 |
| fixed_8 | 35.38 | 24.27 | 20.20 | 19.93 | 17.60 | 10.77 | 5.58 | 3.93 | 17.21 | 8.000 | 8.000 |
| fixed_16 | 29.02 | 20.85 | 17.82 | 15.25 | 13.02 | 11.70 | 10.20 | 9.75 | 15.95 | 16.000 | 16.000 |
| read_norm | 39.12 | 20.22 | 16.23 | 14.00 | 11.67 | 10.53 | 8.95 | 9.17 | 16.24 | 13.246 | 14.246 |
| relative_change | 41.87 | 26.80 | 20.48 | 16.00 | 11.68 | 8.33 | 6.70 | 6.95 | 17.35 | 10.400 | 10.400 |
| output_stability | 47.52 | 25.77 | 19.25 | 14.85 | 11.95 | 8.68 | 7.13 | 6.17 | 17.66 | 10.464 | 10.464 |
| oracle_true_hop | 89.25 | 19.47 | 7.23 | 4.27 | 3.78 | 3.57 | 3.72 | 3.93 | 16.90 | 4.500 | 4.500 |

Stopped-minus-L1 screening endpoints:

| Method | Hop1 Δ | Reach Δ (2,3) | Far Δ (6,7,8) | Walk specificity | Candidate |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | +0.000 | +0.000 | +0.000 | +0.000 | False |
| fixed_16 | -60.233 | +18.283 | +9.578 | +8.706 | False |
| fixed_4 | -39.367 | +27.650 | +2.283 | +25.367 | False |
| fixed_8 | -53.867 | +21.183 | +5.789 | +15.394 | False |
| oracle_true_hop | +0.000 | +12.300 | +2.767 | +9.533 | True |
| output_stability | -41.733 | +21.458 | +6.356 | +15.103 | False |
| read_norm | -50.133 | +17.175 | +8.578 | +8.597 | False |
| relative_change | -47.383 | +22.592 | +6.356 | +16.236 | False |

## t1_frozen_fact/r32/M1

Validation selection:

| Rule | Threshold | Val accuracy (%) | Val steps |
| --- | --- | --- | --- |
| read_norm | 0.7 | 13.044 | 14.704 |
| relative_change | 0.003 | 12.990 | 15.138 |
| output_stability | 4 | 13.060 | 7.443 |

Screening test:

| Method | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | Overall | Steps | Reads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_1 | 84.60 | 1.73 | 1.37 | 1.38 | 1.22 | 1.07 | 1.42 | 1.15 | 11.74 | 1.000 | 1.000 |
| fixed_4 | 54.73 | 17.42 | 7.57 | 4.55 | 3.23 | 3.58 | 3.52 | 3.97 | 12.32 | 4.000 | 4.000 |
| fixed_8 | 50.00 | 16.35 | 8.83 | 6.62 | 5.05 | 4.88 | 4.17 | 4.23 | 12.52 | 8.000 | 8.000 |
| fixed_16 | 48.13 | 16.18 | 8.67 | 6.92 | 5.48 | 5.32 | 4.63 | 4.95 | 12.54 | 16.000 | 16.000 |
| read_norm | 48.73 | 16.17 | 8.73 | 6.60 | 5.40 | 5.17 | 4.63 | 4.80 | 12.53 | 14.705 | 15.705 |
| relative_change | 48.15 | 16.20 | 8.67 | 6.90 | 5.48 | 5.32 | 4.63 | 4.95 | 12.54 | 15.145 | 15.145 |
| output_stability | 54.92 | 15.58 | 7.98 | 5.60 | 4.20 | 4.33 | 3.85 | 4.17 | 12.58 | 7.454 | 7.454 |
| oracle_true_hop | 84.60 | 12.58 | 5.42 | 4.55 | 4.20 | 4.08 | 3.85 | 4.23 | 15.44 | 4.500 | 4.500 |

Stopped-minus-L1 screening endpoints:

| Method | Hop1 Δ | Reach Δ (2,3) | Far Δ (6,7,8) | Walk specificity | Candidate |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | +0.000 | +0.000 | +0.000 | +0.000 | False |
| fixed_16 | -36.467 | +10.875 | +3.756 | +7.119 | False |
| fixed_4 | -29.867 | +10.942 | +2.478 | +8.464 | False |
| fixed_8 | -34.600 | +11.042 | +3.217 | +7.825 | False |
| oracle_true_hop | +0.000 | +7.450 | +2.844 | +4.606 | True |
| output_stability | -29.683 | +10.233 | +2.906 | +7.328 | False |
| read_norm | -35.867 | +10.900 | +3.656 | +7.244 | False |
| relative_change | -36.450 | +10.883 | +3.756 | +7.128 | False |

## t1_frozen_fact/r32/M1-lin

Validation selection:

| Rule | Threshold | Val accuracy (%) | Val steps |
| --- | --- | --- | --- |
| read_norm | 1 | 13.479 | 8.987 |
| relative_change | 0.03 | 13.252 | 12.937 |
| output_stability | 8 | 13.294 | 11.824 |

Screening test:

| Method | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | Overall | Steps | Reads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_1 | 86.40 | 1.57 | 1.13 | 1.28 | 0.93 | 1.18 | 1.25 | 1.18 | 11.87 | 1.000 | 1.000 |
| fixed_4 | 57.92 | 18.23 | 6.95 | 3.93 | 3.10 | 3.32 | 3.13 | 3.42 | 12.50 | 4.000 | 4.000 |
| fixed_8 | 51.62 | 16.72 | 8.92 | 6.63 | 5.00 | 4.20 | 3.77 | 4.02 | 12.61 | 8.000 | 8.000 |
| fixed_16 | 48.42 | 16.90 | 9.07 | 6.95 | 5.30 | 4.57 | 4.58 | 4.77 | 12.57 | 16.000 | 16.000 |
| read_norm | 61.50 | 14.67 | 7.07 | 5.80 | 4.07 | 3.77 | 3.77 | 3.67 | 13.04 | 9.068 | 10.068 |
| relative_change | 48.57 | 16.83 | 8.95 | 6.95 | 5.32 | 4.53 | 4.55 | 4.82 | 12.56 | 12.998 | 12.998 |
| output_stability | 51.25 | 16.73 | 8.75 | 6.58 | 4.98 | 4.30 | 4.27 | 4.52 | 12.67 | 11.850 | 11.850 |
| oracle_true_hop | 86.40 | 11.87 | 4.92 | 3.93 | 3.87 | 3.50 | 3.48 | 4.02 | 15.25 | 4.500 | 4.500 |

Stopped-minus-L1 screening endpoints:

| Method | Hop1 Δ | Reach Δ (2,3) | Far Δ (6,7,8) | Walk specificity | Candidate |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | +0.000 | +0.000 | +0.000 | +0.000 | False |
| fixed_16 | -37.983 | +11.633 | +3.433 | +8.200 | False |
| fixed_4 | -28.483 | +11.242 | +2.083 | +9.158 | False |
| fixed_8 | -34.783 | +11.467 | +2.789 | +8.678 | False |
| oracle_true_hop | +0.000 | +7.042 | +2.461 | +4.581 | True |
| output_stability | -35.150 | +11.392 | +3.156 | +8.236 | False |
| read_norm | -24.900 | +9.517 | +2.528 | +6.989 | False |
| relative_change | -37.833 | +11.542 | +3.428 | +8.114 | False |
