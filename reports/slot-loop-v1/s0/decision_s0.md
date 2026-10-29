# S0 decision: R2

full adaptation; soft mixture or temperature suspected, HardSlot stays diagnostic

## Engineering gate (required preflight checks)

| check | passed |
|---|---|
| native_replay | yes |
| matrix_equals_native | yes |
| owner_mask | yes |
| isolation | yes |
| frozen_set | yes |
| bytes | yes |
| capacity | yes |
| feature_consistency | yes |
| d128_3_replay | yes |

S1A: arms Matrix-L1 and SoftSlot-L1; Matrix g = 1; SoftSlot g = 1, tau = 0.1.

n = 1000 dev episodes; calibration n = 2048.

## Decision comparisons (against matrix_cal at the same scope; +5.0 pp is (c_a - c_b) * 20 >= n)

| condition | vs | d correct | +5.0 pp met | read error lower | improves |
|---|---|---|---|---|---|
| soft_cal/all_valid_question | matrix_cal/all_valid_question | 25 | False | True | False |
| hard_cal/all_valid_question | matrix_cal/all_valid_question | 61 | True | True | True |
| soft_cal/final_only | matrix_cal/final_only | 35 | False | True | False |
| hard_cal/final_only | matrix_cal/final_only | 83 | True | True | True |
| clean_final/final_only | matrix_cal/final_only | 206 | True |  |  |

## Conditions, scope all_valid_question

| role | g | tau | EM % | correct | NLL | read error | cosine | norm ratio | w_true | w_gap | w_entropy | top slot true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 1.00 |  | 72.0 | 720 | 1.577648 | 0.1840 | 0.9288 | 0.7348 |  |  |  |  |
| matrix_uncal | 1.00 |  | 72.0 | 720 | 1.577648 | 0.1840 | 0.9288 | 0.7348 |  |  |  |  |
| matrix_cal | 1.00 |  | 72.0 | 720 | 1.577648 | 0.1840 | 0.9288 | 0.7348 |  |  |  |  |
| soft_uncal | 1.00 | 1.00 | 2.1 | 21 | 6.721904 | 0.4362 | 0.7648 | 0.7754 | 0.1075 | 0.0246 | 2.7475 |  |
| soft_cal | 1.00 | 0.10 | 74.5 | 745 | 1.505075 | 0.0924 | 0.9535 | 0.9299 | 0.7654 | 0.6657 | 0.6066 |  |
| hard_uncal | 1.00 |  | 78.1 | 781 | 2.434278 | 0.1370 | 0.9334 | 0.9964 |  |  |  | 0.8650 |
| hard_cal | 1.00 |  | 78.1 | 781 | 2.434278 | 0.1370 | 0.9334 | 0.9964 |  |  |  | 0.8650 |
| soft_rec_uncal | 1.00 | 1.00 | 2.1 | 21 | 6.721386 | 0.4362 | 0.7648 | 0.7754 | 0.1075 | 0.0246 | 2.7475 |  |
| soft_rec_cal | 1.00 | 0.10 | 74.6 | 746 | 1.504875 | 0.0924 | 0.9535 | 0.9299 | 0.7654 | 0.6657 | 0.6066 |  |

## Conditions, scope final_only

| role | g | tau | EM % | correct | NLL | read error | cosine | norm ratio | w_true | w_gap | w_entropy | top slot true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matrix_cal | 1.00 |  | 72.0 | 720 | 1.577648 | 0.1840 | 0.9288 | 0.7348 |  |  |  |  |
| soft_cal | 1.00 | 0.10 | 75.5 | 755 | 1.461426 | 0.0924 | 0.9535 | 0.9299 | 0.7654 | 0.6657 | 0.6066 |  |
| hard_cal | 1.00 |  | 80.3 | 803 | 2.454239 | 0.1370 | 0.9334 | 0.9964 |  |  |  | 0.8650 |
| clean_final | 1.00 |  | 92.6 | 926 | 0.360599 | 0.0000 | 1.0000 | 1.0000 |  |  |  |  |

Key top-1 0.8650, key margin 0.2475 (shared by every condition).

## Calibration (2048 episodes, mean NLL; g = 1, tau = 1 is the uncalibrated row)

| kind | g | tau | NLL | correct | seconds | selected |
|---|---|---|---|---|---|---|
| matrix | 0.5 |  | 1.875734 | 1374 | 6.25 |  |
| matrix | 1 |  | 1.690976 | 1448 | 5.97 | yes |
| matrix | 2 |  | 1.766732 | 1418 | 5.97 |  |
| soft | 0.5 | 0.03 | 1.833733 | 1615 | 6.54 |  |
| soft | 1 | 0.03 | 1.823099 | 1624 | 6.52 |  |
| soft | 2 | 0.03 | 1.915640 | 1579 | 6.50 |  |
| soft | 0.5 | 0.10 | 1.555333 | 1539 | 6.49 |  |
| soft | 1 | 0.10 | 1.515452 | 1564 | 6.46 | yes |
| soft | 2 | 0.10 | 1.649865 | 1491 | 6.52 |  |
| soft | 0.5 | 0.30 | 3.774785 | 679 | 6.51 |  |
| soft | 1 | 0.30 | 3.439175 | 732 | 6.45 |  |
| soft | 2 | 0.30 | 3.686310 | 610 | 6.50 |  |
| soft | 0.5 | 1.00 | 6.798727 | 28 | 6.47 |  |
| soft | 1 | 1.00 | 6.637938 | 48 | 6.44 |  |
| soft | 2 | 1.00 | 7.025215 | 50 | 6.47 |  |
| hard | 0.5 |  | 2.338064 | 1595 | 6.64 |  |
| hard | 1 |  | 2.310140 | 1623 | 6.62 | yes |
| hard | 2 |  | 2.411094 | 1601 | 6.64 |  |

## Differences against matrix_cal (descriptive: paired 95 % percentile bootstrap, never used in the decision)

| condition | vs | d correct | difference pp | 95 % interval pp | d read error |
|---|---|---|---|---|---|
| native/all_valid_question | matrix_cal/all_valid_question | 0 | +0.00 | [+0.00, +0.00] | +0.0000 |
| matrix_uncal/all_valid_question | matrix_cal/all_valid_question | 0 | +0.00 | [+0.00, +0.00] | +0.0000 |
| soft_uncal/all_valid_question | matrix_cal/all_valid_question | -699 | -69.90 | [-72.80, -67.10] | +0.2522 |
| soft_cal/all_valid_question | matrix_cal/all_valid_question | 25 | +2.50 | [+0.60, +4.40] | -0.0916 |
| hard_uncal/all_valid_question | matrix_cal/all_valid_question | 61 | +6.10 | [+3.80, +8.40] | -0.0470 |
| hard_cal/all_valid_question | matrix_cal/all_valid_question | 61 | +6.10 | [+3.80, +8.40] | -0.0470 |
| soft_rec_uncal/all_valid_question | matrix_cal/all_valid_question | -699 | -69.90 | [-72.70, -67.10] | +0.2522 |
| soft_rec_cal/all_valid_question | matrix_cal/all_valid_question | 26 | +2.60 | [+0.70, +4.50] | -0.0916 |
| soft_cal/final_only | matrix_cal/final_only | 35 | +3.50 | [+1.60, +5.50] | -0.0916 |
| hard_cal/final_only | matrix_cal/final_only | 83 | +8.30 | [+6.00, +10.60] | -0.0470 |
| clean_final/final_only | matrix_cal/final_only | 206 | +20.60 | [+18.00, +23.30] | -0.1840 |

## D128-3 recheck (descriptive; S-RE selected g and tau, no recalibration)

| role | EM % | correct | NLL | read error |
|---|---|---|---|---|
| native | 63.3 | 633 | 2.004491 | 0.3549 |
| matrix | 63.3 | 633 | 2.004491 | 0.3549 |
| soft | 73.7 | 737 | 1.904800 | 0.1477 |
| hard | 74.8 | 748 | 2.707143 | 0.2001 |

Soft minus matrix on D128-3: +104 (+10.40 pp, higher); hard minus matrix: +115 (+11.50 pp, higher); on S-RE soft minus matrix is higher; same direction: yes.
Old-range replay: 608/1000, loss 2.1291537284851074 (expected 608/1000 and 2.1291537284851074; reported, not judged).

## Notes

- soft_rec against soft: largest EM-count difference 1 of 1000; S-RE's decay is 0.999999, so near equality is expected
