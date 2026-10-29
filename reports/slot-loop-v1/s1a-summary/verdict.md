# S1A verdict: ADVANCE

Rule: Delta >= +3.0 pp (sum over seeds of C_soft - C_matrix, times 100, >= 9 n), lower interval end L > 0, and C_soft > C_matrix for each of seeds 17, 29, 43.

| test | met |
|---|---|
| delta_at_least_3pp | True |
| lower_end_positive | True |
| seeds_all_positive | True |

## confirmation (primary)

n = 2000 (expected 2000); episode ids blake2b-128 3cd75bc09be400623e523da95d11f10b (little-endian int64).

| contrast | d correct | difference pp | 95 % lower pp | 95 % upper pp |
|---|---|---|---|---|
| Delta (mean over seeds) | 224 | +3.733 | +2.700 | +4.817 |
| Delta_s seed 17 | 70 | +3.500 | +2.200 | +4.800 |
| Delta_s seed 29 | 84 | +4.200 | +2.900 | +5.500 |
| Delta_s seed 43 | 70 | +3.500 | +2.100 | +4.950 |

| arm or reference | seed | correct | EM % | mean NLL | mean read error |
|---|---|---|---|---|---|
| matrix_L1 | 17 | 1650 | 82.50 | 0.907350 | 0.1508 |
| matrix_L1 | 29 | 1639 | 81.95 | 0.918376 | 0.1507 |
| matrix_L1 | 43 | 1647 | 82.35 | 0.899218 | 0.1493 |
| matrix_L1 | mean |  | 82.27 | 0.908315 | 0.1503 |
| soft_slot_L1 | 17 | 1720 | 86.00 | 0.760876 | 0.0776 |
| soft_slot_L1 | 29 | 1723 | 86.15 | 0.819596 | 0.0758 |
| soft_slot_L1 | 43 | 1717 | 85.85 | 0.797308 | 0.0776 |
| soft_slot_L1 | mean |  | 86.00 | 0.792593 | 0.0770 |
| s_re_native |  | 1447 | 72.35 | 1.473178 | 0.1767 |
| matrix_step0 |  | 1447 | 72.35 | 1.473178 | 0.1767 |
| soft_step0 |  | 1527 | 76.35 | 1.313163 | 0.0808 |

## unseen (descriptive, no verdict)

n = 2000 (expected 2000); episode ids blake2b-128 10df09f596dbce1e23567f67ed6be852 (little-endian int64).

| contrast | d correct | difference pp | 95 % lower pp | 95 % upper pp |
|---|---|---|---|---|
| Delta (mean over seeds) | 179 | +2.983 | +2.083 | +3.883 |
| Delta_s seed 17 | 41 | +2.050 | +0.900 | +3.150 |
| Delta_s seed 29 | 77 | +3.850 | +2.550 | +5.150 |
| Delta_s seed 43 | 61 | +3.050 | +1.750 | +4.350 |

| arm or reference | seed | correct | EM % | mean NLL | mean read error |
|---|---|---|---|---|---|
| matrix_L1 | 17 | 212 | 10.60 | 5.126225 | 0.1919 |
| matrix_L1 | 29 | 207 | 10.35 | 5.101240 | 0.1925 |
| matrix_L1 | 43 | 208 | 10.40 | 5.087438 | 0.1920 |
| matrix_L1 | mean |  | 10.45 | 5.104967 | 0.1921 |
| soft_slot_L1 | 17 | 253 | 12.65 | 4.762352 | 0.1143 |
| soft_slot_L1 | 29 | 284 | 14.20 | 4.781072 | 0.1140 |
| soft_slot_L1 | 43 | 269 | 13.45 | 4.715673 | 0.1141 |
| soft_slot_L1 | mean |  | 13.43 | 4.753032 | 0.1142 |
| s_re_native |  | 97 | 4.85 | 5.825520 | 0.2199 |
| matrix_step0 |  | 97 | 4.85 | 5.825520 | 0.2199 |
| soft_step0 |  | 119 | 5.95 | 5.615395 | 0.1219 |
