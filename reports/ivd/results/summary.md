# IVD summary

Confirmation episodes: split test, seed 999004, 2000 episodes per checkpoint; three training seeds pooled; differences in percentage points.

## Registered decision (M1, load L24, hops 2,3, 97.5% interval)

Interference contrast = (L4 - L1) under low overlap minus (L4 - L1) under high overlap.

| writer | contrast [interval] | >= 5 pp and interval > 0 | overlap gap (high - low) | decision |
| --- | --- | --- | --- | --- |
| oracle_random | +13.98 [+12.64, +15.24] | yes | 0.500 | supported |
| joint_fact | +3.21 [+2.18, +4.21] | no | 0.285 | not_supported |

## Interference contrast by load and hop group (reported)

| writer | load | hops | contrast [interval] | pattern |
| --- | --- | --- | --- | --- |
| oracle_random | L24 | hop1 | -13.68 [-16.07, -11.47] | no |
| oracle_random | L6 | hop1 | -1.58 [-3.37, +0.27] | no |
| oracle_random | L24 | reach | +13.98 [+12.64, +15.24] | yes |
| oracle_random | L6 | reach | +1.57 [+0.52, +2.64] | no |
| oracle_random | L24 | mid | +0.58 [-0.25, +1.41] | no |
| oracle_random | L6 | mid | -0.29 [-0.87, +0.22] | no |
| oracle_random | L24 | far | -0.18 [-0.83, +0.47] | no |
| oracle_random | L6 | far | +0.06 [-0.34, +0.43] | no |
| joint_fact | L24 | hop1 | -7.33 [-9.48, -5.25] | no |
| joint_fact | L6 | hop1 | -1.70 [-3.55, +0.05] | no |
| joint_fact | L24 | reach | +3.21 [+2.18, +4.21] | no |
| joint_fact | L6 | reach | +0.99 [+0.03, +1.94] | no |
| joint_fact | L24 | mid | +0.64 [-0.14, +1.40] | no |
| joint_fact | L6 | mid | -0.12 [-0.68, +0.44] | no |
| joint_fact | L24 | far | +0.63 [-0.07, +1.33] | no |
| joint_fact | L6 | far | +0.09 [-0.38, +0.57] | no |

## oracle_random: M1 depth gain (L4 - L1) by cell, hops 2,3

| cell | gain | 95% interval | L1 acc | L4 acc | mean max |cos| | stored residual |
| --- | --- | --- | --- | --- | --- | --- |
| base | +33.09 | [+31.99, +34.19] | 1.1 | 34.2 |  | 0.395 |
| L6-low | +29.91 | [+28.83, +30.93] | 1.2 | 31.2 | 0.000 | 0.395 |
| L6-nat | +27.89 | [+26.84, +29.01] | 1.4 | 29.3 | 0.341 | 0.432 |
| L6-high | +28.34 | [+27.27, +29.42] | 1.4 | 29.7 | 0.500 | 0.415 |
| L24-low | +25.54 | [+24.57, +26.52] | 1.4 | 26.9 | 0.000 | 0.395 |
| L24-nat | +10.99 | [+10.18, +11.79] | 2.8 | 13.8 | 0.342 | 0.744 |
| L24-high | +11.57 | [+10.72, +12.43] | 3.0 | 14.6 | 0.500 | 0.700 |

B4 (single read) accuracy by cell, hops 2,3:

| cell | accuracy | 95% interval |
| --- | --- | --- |
| base | 0.4 | [0.3, 0.6] |
| L6-low | 0.5 | [0.3, 0.6] |
| L6-nat | 0.5 | [0.4, 0.7] |
| L6-high | 0.7 | [0.5, 0.9] |
| L24-low | 0.5 | [0.3, 0.6] |
| L24-nat | 2.9 | [2.4, 3.3] |
| L24-high | 2.8 | [2.4, 3.2] |

## joint_fact: M1 depth gain (L4 - L1) by cell, hops 2,3

| cell | gain | 95% interval | L1 acc | L4 acc | mean max |cos| | stored residual |
| --- | --- | --- | --- | --- | --- | --- |
| base | +23.38 | [+22.50, +24.26] | 1.4 | 24.8 |  | 0.667 |
| L6-low | +20.75 | [+19.91, +21.60] | 1.7 | 22.5 | 0.315 | 0.676 |
| L6-nat | +20.18 | [+19.32, +21.07] | 2.1 | 22.2 | 0.452 | 0.680 |
| L6-high | +19.76 | [+18.92, +20.53] | 1.9 | 21.6 | 0.601 | 0.683 |
| L24-low | +10.83 | [+10.11, +11.54] | 3.4 | 14.3 | 0.316 | 0.748 |
| L24-nat | +8.28 | [+7.58, +8.96] | 4.2 | 12.4 | 0.453 | 0.780 |
| L24-high | +7.62 | [+6.97, +8.28] | 4.2 | 11.8 | 0.601 | 0.802 |
| L6-decay | +23.38 | [+22.54, +24.29] | 1.4 | 24.8 |  | 0.667 |
| L24-decay | +23.38 | [+22.52, +24.22] | 1.4 | 24.8 |  | 0.667 |

B4 (single read) accuracy by cell, hops 2,3:

| cell | accuracy | 95% interval |
| --- | --- | --- |
| base | 0.9 | [0.8, 1.1] |
| L6-low | 1.4 | [1.2, 1.7] |
| L6-nat | 1.8 | [1.5, 2.0] |
| L6-high | 1.8 | [1.5, 2.0] |
| L24-low | 3.7 | [3.3, 4.1] |
| L24-nat | 4.4 | [4.0, 4.8] |
| L24-high | 4.6 | [4.2, 5.1] |
| L6-decay | 0.9 | [0.8, 1.1] |
| L24-decay | 0.9 | [0.8, 1.1] |

Run: 0.015 node-h; design lock 2a8d063dbd29.
