# JS paired contrasts

Effects and 97.5% CIs are percentage points. Per-seed effects are ordered17 /29 /43.
L=17 is the internal stopped-policy sentinel, selecting an actual state between1 and16; it is not a seventeenth read.

| Endpoint | Writer | A - B | L_A / L_B | Hops | Estimate [97.5% CI], pp | Per seed, pp |
| --- | --- | --- | --- | --- | --- | --- |
| JS-E1-S | oracle_random | RS - RS | 16 / 4 | 1-3 | -11.267 [-12.356, -10.050] | -11.900 / -11.017 / -10.883 |
| JS-E1-S | oracle_random | RS+J - RS+J | 16 / 4 | 1-3 | -2.833 [-3.800, -1.806] | -3.533 / -2.400 / -2.567 |
| JS-E1-DeltaS | oracle_random | RS+J - RS | 16-4 / 16-4 | 1-3 | +8.433 [+7.483, +9.418] | +8.367 / +8.617 / +8.317 |
| JS-E2 | oracle_random | RS+J - RS | 4 / 4 | 1-3 | +0.789 [+0.261, +1.295] | +1.200 / +0.883 / +0.283 |
| JS-E3 | oracle_random | RS+J - RS | 16 / 16 | 4-8 | -5.547 [-6.200, -4.890] | -5.670 / -4.540 / -6.430 |
| JS-E4 | oracle_random | RS+J - RS | 17 / 17 | 1-8 | -1.040 [-1.538, -0.544] | -0.694 / -0.513 / -1.912 |
| RS-reference | oracle_random | RS - M1-L4 | 4 / 4 | 1-3 | -14.694 [-15.839, -13.589] | -16.500 / -14.367 / -13.217 |
| RS-reference | oracle_random | RS - M1-L4 | 4 / 4 | 4-8 | -0.880 [-1.167, -0.593] | -0.870 / -0.850 / -0.920 |
| RS-reference | oracle_random | RS - M1-L4 | 4 / 4 | 1-8 | -6.060 [-6.519, -5.621] | -6.731 / -5.919 / -5.531 |
| RS-reference | oracle_random | RS - M1-L4 | 16 / 16 | 1-3 | +7.817 [+6.794, +8.806] | +6.333 / +8.400 / +8.717 |
| RS-reference | oracle_random | RS - M1-L4 | 16 / 16 | 4-8 | +7.493 [+6.563, +8.523] | +7.820 / +6.990 / +7.670 |
| RS-reference | oracle_random | RS - M1-L4 | 16 / 16 | 1-8 | +7.615 [+6.856, +8.450] | +7.262 / +7.519 / +8.062 |
| RS-reference | oracle_random | RS - M1-mixed | 4 / 4 | 1-3 | -8.739 [-9.733, -7.767] | -9.233 / -9.283 / -7.700 |
| RS-reference | oracle_random | RS - M1-mixed | 4 / 4 | 4-8 | -2.013 [-2.303, -1.720] | -1.900 / -2.020 / -2.120 |
| RS-reference | oracle_random | RS - M1-mixed | 4 / 4 | 1-8 | -4.535 [-4.967, -4.102] | -4.650 / -4.744 / -4.213 |
| RS-reference | oracle_random | RS - M1-mixed | 16 / 16 | 1-3 | +5.300 [+4.367, +6.239] | +4.767 / +5.483 / +5.650 |
| RS-reference | oracle_random | RS - M1-mixed | 16 / 16 | 4-8 | +5.080 [+4.210, +6.023] | +6.180 / +3.950 / +5.110 |
| RS-reference | oracle_random | RS - M1-mixed | 16 / 16 | 1-8 | +5.163 [+4.394, +5.881] | +5.650 / +4.525 / +5.312 |
| JS-E1-S | joint_fact | RS - RS | 16 / 4 | 1-3 | -3.017 [-3.789, -2.272] | -2.800 / -1.950 / -4.300 |
| JS-E1-S | joint_fact | RS+J - RS+J | 16 / 4 | 1-3 | -1.772 [-2.484, -1.061] | -1.983 / -0.900 / -2.433 |
| JS-E1-DeltaS | joint_fact | RS+J - RS | 16-4 / 16-4 | 1-3 | +1.244 [+0.222, +2.200] | +0.817 / +1.050 / +1.867 |
| JS-E2 | joint_fact | RS+J - RS | 4 / 4 | 1-3 | -0.728 [-1.567, +0.067] | -0.233 / -0.733 / -1.217 |
| JS-E3 | joint_fact | RS+J - RS | 16 / 16 | 4-8 | -1.443 [-1.990, -0.933] | -0.630 / -1.250 / -2.450 |
| JS-E4 | joint_fact | RS+J - RS | 17 / 17 | 1-8 | -0.677 [-1.227, -0.115] | -0.163 / -0.606 / -1.263 |
| RS-reference | joint_fact | RS - M1-L4 | 4 / 4 | 1-3 | -2.522 [-3.533, -1.500] | -2.500 / -3.650 / -1.417 |
| RS-reference | joint_fact | RS - M1-L4 | 4 / 4 | 4-8 | -1.153 [-1.467, -0.847] | -1.630 / -0.890 / -0.940 |
| RS-reference | joint_fact | RS - M1-L4 | 4 / 4 | 1-8 | -1.667 [-2.113, -1.231] | -1.956 / -1.925 / -1.119 |
| RS-reference | joint_fact | RS - M1-L4 | 16 / 16 | 1-3 | +5.067 [+3.967, +6.172] | +3.850 / +6.000 / +5.350 |
| RS-reference | joint_fact | RS - M1-L4 | 16 / 16 | 4-8 | -2.620 [-3.307, -1.933] | -2.680 / -4.340 / -0.840 |
| RS-reference | joint_fact | RS - M1-L4 | 16 / 16 | 1-8 | +0.263 [-0.394, +0.921] | -0.231 / -0.462 / +1.481 |
| RS-reference | joint_fact | RS - M1-mixed | 4 / 4 | 1-3 | +0.483 [-0.489, +1.467] | +0.200 / -0.567 / +1.817 |
| RS-reference | joint_fact | RS - M1-mixed | 4 / 4 | 4-8 | -1.773 [-2.107, -1.433] | -1.840 / -1.720 / -1.760 |
| RS-reference | joint_fact | RS - M1-mixed | 4 / 4 | 1-8 | -0.927 [-1.373, -0.485] | -1.075 / -1.287 / -0.419 |
| RS-reference | joint_fact | RS - M1-mixed | 16 / 16 | 1-3 | +4.911 [+3.855, +6.022] | +4.900 / +5.667 / +4.167 |
| RS-reference | joint_fact | RS - M1-mixed | 16 / 16 | 4-8 | -1.033 [-1.620, -0.383] | -2.400 / -1.220 / +0.520 |
| RS-reference | joint_fact | RS - M1-mixed | 16 / 16 | 1-8 | +1.196 [+0.579, +1.808] | +0.338 / +1.363 / +1.887 |

Registered overshoot decision:

- oracle_random: does not pass; positive DeltaS CI=True; S(RS+J)>=-1pp=False.
- joint_fact: does not pass; positive DeltaS CI=True; S(RS+J)>=-1pp=False.
