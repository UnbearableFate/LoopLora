# P2-S-F1 paired contrasts

Effects and intervals are percentage points; 97.5% pooled paired episode-bootstrap CIs.
Per-seed effects are ordered 17 / 29 / 43. The confirmation episodes are reused from P2-S.
Intervals condition on these three trained seeds. F-E3 hop-1/long rows accompany the registered reach endpoint.

| Endpoint | Writer | Contrast | Hops | Estimate [97.5% CI], pp | Per seed, pp |
| --- | --- | --- | --- | --- | --- |
| F-E1 | joint | B8@4 - M1-L4@4 | reach | +13.217 [+11.983, +14.450] | +12.450 / +21.600 / +5.600 |
| F-E1 | joint | B8@4 - M1-L4@4 | hop1 | +4.450 [+3.300, +5.750] | +3.950 / +10.250 / -0.850 |
| F-E1 | joint | B8@4 - M1-L4@4 | long | -5.533 [-6.534, -4.467] | -7.220 / -7.860 / -1.520 |
| F-E2 | joint | M1-L4@4 - M1-mixed@4 | reach | +20.517 [+19.208, +21.783] | +13.950 / +30.600 / +17.000 |
| F-E2 | joint | M1-L4@4 - M1-mixed@4 | hop1 | +27.617 [+26.150, +29.067] | +8.600 / +55.200 / +19.050 |
| F-E1 | oracle_random | B8@4 - M1-L4@4 | reach | +3.200 [+1.958, +4.517] | +2.425 / +4.500 / +2.675 |
| F-E1 | oracle_random | B8@4 - M1-L4@4 | hop1 | +8.217 [+6.466, +10.017] | +8.150 / +7.500 / +9.000 |
| F-E1 | oracle_random | B8@4 - M1-L4@4 | long | -0.537 [-0.797, -0.280] | -0.590 / -0.570 / -0.450 |
| F-E2 | oracle_random | M1-L4@4 - M1-mixed@4 | reach | +8.242 [+7.117, +9.400] | +7.700 / +8.225 / +8.800 |
| F-E2 | oracle_random | M1-L4@4 - M1-mixed@4 | hop1 | +4.150 [+2.467, +5.800] | +5.350 / +3.600 / +3.500 |
| F-E1 | joint_fact | B8@4 - M1-L4@4 | reach | +5.458 [+4.242, +6.750] | +8.825 / +4.325 / +3.225 |
| F-E1 | joint_fact | B8@4 - M1-L4@4 | hop1 | +6.467 [+4.500, +8.317] | +5.950 / +5.400 / +8.050 |
| F-E1 | joint_fact | B8@4 - M1-L4@4 | long | -0.753 [-1.077, -0.433] | -0.900 / -0.990 / -0.370 |
| F-E2 | joint_fact | M1-L4@4 - M1-mixed@4 | reach | +3.483 [+2.250, +4.758] | +1.150 / +5.200 / +4.100 |
| F-E2 | joint_fact | M1-L4@4 - M1-mixed@4 | hop1 | +2.617 [+0.683, +4.534] | +3.500 / +2.150 / +2.200 |
| F-E3 | joint_fact | M1-mixed@4 - B6/r64/m64@1 | reach | +24.017 [+23.000, +25.108] | +24.125 / +24.250 / +23.675 |
| F-E3 | joint_fact | M1-mixed@4 - B6/r64/m64@1 | hop1 | -47.183 [-48.817, -45.450] | -48.700 / -47.700 / -45.150 |
| F-E3 | joint_fact | M1-mixed@4 - B6/r64/m64@1 | long | +3.513 [+3.207, +3.823] | +3.560 / +3.600 / +3.380 |
| F-E3 | joint_fact | M1-mixed@4 - B6/r128/m128@1 | reach | +24.242 [+23.242, +25.333] | +24.275 / +24.350 / +24.100 |
| F-E3 | joint_fact | M1-mixed@4 - B6/r128/m128@1 | hop1 | -49.833 [-51.450, -48.216] | -50.650 / -49.850 / -49.000 |
| F-E3 | joint_fact | M1-mixed@4 - B6/r128/m128@1 | long | +3.673 [+3.380, +3.983] | +3.610 / +3.700 / +3.710 |
| F-E3 | joint_fact | M1-mixed@4 - B7/r32/m64@1 | reach | +23.875 [+22.858, +24.975] | +24.125 / +24.050 / +23.450 |
| F-E3 | joint_fact | M1-mixed@4 - B7/r32/m64@1 | hop1 | -45.017 [-46.734, -43.333] | -46.700 / -45.000 / -43.350 |
| F-E3 | joint_fact | M1-mixed@4 - B7/r32/m64@1 | long | +3.300 [+2.983, +3.623] | +3.230 / +3.380 / +3.290 |
| F-E3 | joint_fact | M1-mixed@4 - B7/r32/m128@1 | reach | +23.550 [+22.533, +24.659] | +23.700 / +23.850 / +23.100 |
| F-E3 | joint_fact | M1-mixed@4 - B7/r32/m128@1 | hop1 | -44.217 [-45.934, -42.550] | -46.050 / -44.450 / -42.150 |
| F-E3 | joint_fact | M1-mixed@4 - B7/r32/m128@1 | long | +3.187 [+2.873, +3.513] | +3.260 / +3.270 / +3.030 |

Reader for P1-L: **B8**.

| Carried writer | Training depth under F-E2 |
| --- | --- |
| joint | fixed L=4 |
| oracle_random | fixed L=4 |
| joint_fact | fixed L=4 |

Writer selection remains with the P1-L protocol; apply the corresponding depth above.
Shortcut-free capability claim: **passes**.
