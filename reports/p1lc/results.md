# P1L-C: causal and read-position analysis of the r = 128 oracle readers

Test [102000, 104000), eval seed 4242; checkpoints M1-L4, B8, B4 x seeds 17/29/43 (nothing retrained); 2000 bootstrap draws.

## A. Read position (descriptive; gates nothing)

Full-vocabulary EM with memory only at the final question position, minus deployed reads; 95% paired episode-bootstrap CIs. A drop is a distribution shift, not evidence of leakage.

| Reader | Hops | Deployed EM(%) | Final-position-only EM(%) | Difference(pp) | CI(pp) | Per-seed(pp) |
| --- | --- | ---: | ---: | ---: | --- | --- |
| M1-L4@4 | hop1 | 48.43 | 5.03 | -43.40 | [-44.95,-41.98] | 17:-47.25, 29:-33.70, 43:-49.25 |
| M1-L4@4 | hop2 | 34.23 | 4.28 | -29.95 | [-31.40,-28.55] | 17:-36.75, 29:-18.75, 43:-34.35 |
| M1-L4@4 | hop3 | 18.27 | 1.90 | -16.37 | [-17.55,-15.22] | 17:-18.60, 29:-11.75, 43:-18.75 |
| M1-L4@4 | hops1-3 | 33.64 | 3.74 | -29.91 | [-30.76,-29.09] | 17:-34.20, 29:-21.40, 43:-34.12 |
| B8@4 | hop1 | 18.70 | 2.33 | -16.37 | [-17.52,-15.25] | 17:-17.45, 29:-16.75, 43:-14.90 |
| B8@4 | hop2 | 6.68 | 0.47 | -6.22 | [-6.92,-5.53] | 17:-6.55, 29:-6.80, 43:-5.30 |
| B8@4 | hop3 | 6.05 | 0.17 | -5.88 | [-6.53,-5.27] | 17:-6.05, 29:-5.80, 43:-5.80 |
| B8@4 | hops1-3 | 10.48 | 0.99 | -9.49 | [-10.08,-8.91] | 17:-10.02, 29:-9.78, 43:-8.67 |
| B4@1 | hop1 | 20.30 | 2.83 | -17.47 | [-18.68,-16.28] | 17:-17.50, 29:-17.40, 43:-17.50 |
| B4@1 | hop2 | 5.43 | 0.43 | -5.00 | [-5.67,-4.38] | 17:-5.20, 29:-4.50, 43:-5.30 |
| B4@1 | hop3 | 5.45 | 0.52 | -4.93 | [-5.53,-4.32] | 17:-5.10, 29:-5.00, 43:-4.70 |
| B4@1 | hops1-3 | 10.39 | 1.26 | -9.13 | [-9.71,-8.59] | 17:-9.27, 29:-8.97, 43:-9.17 |

## Descriptive replication on this range (hops 2 and 3, deployed reads)

| Contrast | Difference(pp) | 95% CI(pp) | 97.5% CI(pp) | Per-seed(pp) |
| --- | ---: | --- | --- | --- |
| M1-L4@4 - B4@1 | 20.81 | [19.72,21.85] | [19.57,21.94] | 17:24.43, 29:15.25, 43:22.75 |
| B8@4 - B4@1 | 0.92 | [0.28,1.53] | [0.19,1.61] | 17:0.80, 29:1.60, 43:0.38 |

## B. Bridge patching (hop-2 questions on the long chain, L = 4)

shift = rate(t_B | B-patch) - max(rate(t_B | C-patch), rate(t_B | random patch)); pooled over seeds, 95% paired episode-bootstrap CI. Causal bridge claim: pooled shift >= 10 pp, CI > 0 and a positive shift in every seed.

| Reader | Patch | Shift(pp) | CI(pp) | Per-seed shift(pp) | Claim |
| --- | --- | ---: | --- | --- | --- |
| M1-L4 | final position | 0.39 | [-0.13,0.78] | 17:0.39, 29:0.78, 43:0.00 | no |
| M1-L4 | all position | 0.91 | [-0.65,2.34] | 17:0.78, 29:0.78, 43:1.17 | secondary |
| B8 | final position | 0.00 | [-0.52,0.00] | 17:0.00, 29:0.00, 43:0.00 | no |
| B8 | all position | -0.26 | [-0.78,0.00] | 17:0.00, 29:-0.39, 43:-0.39 | secondary |

Rates of answering t_B and t_A (pooled over seeds; world_X = deployed answer on memory X):

| Reader | Condition | t_B (%) | t_A (%) |
| --- | --- | ---: | ---: |
| M1-L4 | world_A | 7.03 | 33.59 |
| M1-L4 | world_B | 37.24 | 2.60 |
| M1-L4 | world_C | 6.90 | 2.47 |
| M1-L4 | B_final | 7.55 | 33.07 |
| M1-L4 | C_final | 7.03 | 33.20 |
| M1-L4 | R_final | 7.03 | 32.94 |
| M1-L4 | B_all | 9.51 | 28.65 |
| M1-L4 | C_all | 8.20 | 30.47 |
| M1-L4 | R_all | 7.03 | 30.99 |
| B8 | world_A | 9.38 | 6.64 |
| B8 | world_B | 4.69 | 2.73 |
| B8 | world_C | 7.29 | 4.69 |
| B8 | B_final | 9.51 | 6.64 |
| B8 | C_final | 9.51 | 6.77 |
| B8 | R_final | 9.38 | 6.77 |
| B8 | B_all | 9.24 | 6.64 |
| B8 | C_all | 9.38 | 6.64 |
| B8 | R_all | 9.38 | 6.77 |

Final-position state norms (mean over episodes and seeds). The C-patch is the registered unrelated-entity control and is not norm-matched to the B-patch; the random patch is.

| Reader | norm z1^A | norm (z1^B - z1^A) | norm (z1^C - z1^A) |
| --- | ---: | ---: | ---: |
| M1-L4 | 40.396 | 6.921 | 6.706 |
| B8 | 39.358 | 4.367 | 4.377 |
