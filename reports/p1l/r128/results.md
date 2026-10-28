# P1-L oracle-random r128 confirmation

S=17000; seeds17/29/43; 97.5% paired episode-bootstrap CIs,2000 draws.
G1-L: **fail**. G2-L: **pass**.

| Contrast | Difference(pp) | CI(pp) | Per-seed differences(pp) |
| --- | ---: | --- | --- |
| G1_reach | 0.98 | [0.27,1.72] | 17:0.92, 29:0.05, 43:1.98 |
| G1_hop1 | -0.60 | [-1.98,0.77] | 17:-2.30, 29:1.20, 43:-0.70 |
| G1_typed | 1.55 | [0.77,2.37] | 17:1.57, 29:0.60, 43:2.48 |
| G2_fixed_query | 0.83 | [0.06,1.55] | 17:0.88, 29:0.10, 43:1.52 |
| G2_no_memory | 6.34 | [5.76,6.93] | 17:6.53, 29:5.67, 43:6.83 |
| M1_vs_B4 | 21.55 | [20.38,22.76] | 17:24.45, 29:15.10, 43:25.10 |
| M1_vs_B5 | 21.40 | [20.22,22.55] | 17:24.40, 29:15.15, 43:24.65 |
| M1_vs_B3 | 26.91 | [25.76,27.98] | 17:30.05, 29:20.72, 43:29.95 |
| B8_vs_M1 | -20.57 | [-21.78,-19.36] | 17:-23.52, 29:-15.05, 43:-23.12 |
| M1_depth_reach | 25.92 | [24.82,27.05] | 17:29.55, 29:18.52, 43:29.70 |
| M1_depth_far | 3.68 | [3.21,4.18] | 17:3.83, 29:3.83, 43:3.37 |
| M1_walk_specificity | 22.25 | [21.16,23.37] | 17:25.72, 29:14.69, 43:26.33 |

## Amendment B7: memory-use check

Hop-1 full-vocabulary EM on the confirmation set; each reader minus B3-L4@4 (no memory); 97.5% paired episode-bootstrap CIs, 2000 draws.
Memory-use check: **pass**.

| Contrast | Reader EM(%) | B3-L4@4 EM(%) | Difference(pp) | CI(pp) | Per-seed differences(pp) |
| --- | ---: | ---: | ---: | --- | --- |
| B4_vs_B3_hop1 | 19.63 | 0.00 | 19.63 | [18.42,20.93] | 17:20.30, 29:18.75, 43:19.85 |
| B8_vs_B3_hop1 | 19.03 | 0.00 | 19.03 | [17.87,20.37] | 17:18.00, 29:19.95, 43:19.15 |
| M1_vs_B3_hop1 | 47.85 | 0.00 | 47.85 | [46.02,49.67] | 17:51.40, 29:39.65, 43:52.50 |

Rule 3: a G1-L or Amendment-B6 pass shows that the reader learns the value-to-key transition and composes reads inside the LM; it does not establish equivalence with the symbolic oracle condition, and no unseen-name claim is made.
