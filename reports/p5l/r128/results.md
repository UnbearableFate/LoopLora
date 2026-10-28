# P5L oracle_random r128 confirmation on Qwen3-1.7B-Base

S=17000; seeds 17/29/43; 97.5% paired episode-bootstrap CIs, 2000 draws; test [104000, 106000), eval seed 4242, 2000 episodes; B0 evaluated (see accuracy.csv).

**Decision (docs/prereg/P5L.md): holds_at_second_scale: the equal-parameter M1-L4 result holds on Qwen3-1.7B-Base (oracle condition, seen names, trained hops; B7 rule 3 limits)**

B7: **pass**. Amendment B6 (M1-L4@4 - B4@1, hops 2 and 3, >= 3 pp, CI > 0, every seed): **met**. G1-L: **fail**. G2-L: **pass**.

| Contrast | Difference(pp) | CI(pp) | Per-seed differences(pp) |
| --- | ---: | --- | --- |
| G1_reach | 1.03 | [0.29,1.80] | 17:0.20, 29:1.73, 43:1.18 |
| G1_hop1 | -1.72 | [-3.32,-0.02] | 17:-1.80, 29:-0.30, 43:-3.05 |
| G1_typed | 1.13 | [0.27,1.94] | 17:0.38, 29:3.30, 43:-0.30 |
| G2_fixed_query | 1.67 | [0.91,2.38] | 17:0.55, 29:3.60, 43:0.88 |
| G2_no_memory | 7.17 | [6.58,7.76] | 17:6.22, 29:8.82, 43:6.45 |
| M1_vs_B4 | 25.78 | [24.53,27.03] | 17:24.93, 29:26.50, 43:25.92 |
| M1_vs_B5 | 26.42 | [25.14,27.82] | 17:25.27, 29:28.38, 43:25.62 |
| M1_vs_B3 | 31.92 | [30.73,33.21] | 17:30.95, 29:33.60, 43:31.20 |
| B8_vs_M1 | -24.75 | [-26.07,-23.45] | 17:-24.73, 29:-24.77, 43:-24.75 |
| M1_depth_reach | 31.46 | [30.27,32.63] | 17:30.20, 29:33.32, 43:30.85 |
| M1_depth_far | 3.28 | [2.86,3.71] | 17:3.62, 29:3.22, 43:3.00 |
| M1_walk_specificity | 28.18 | [26.84,29.43] | 17:26.58, 29:30.11, 43:27.85 |

## Amendment B7: memory-use check

Hop-1 full-vocabulary EM; each reader minus B3-L4@4 (no memory).

| Contrast | Reader EM(%) | B3-L4@4 EM(%) | Difference(pp) | CI(pp) | Per-seed differences(pp) |
| --- | ---: | ---: | ---: | --- | --- |
| B4_vs_B3_hop1 | 30.65 | 0.03 | 30.62 | [29.17,32.23] | 17:28.95, 29:31.60, 43:31.30 |
| B8_vs_B3_hop1 | 28.93 | 0.03 | 28.90 | [27.52,30.43] | 17:27.15, 29:31.30, 43:28.25 |
| M1_vs_B3_hop1 | 58.70 | 0.03 | 58.67 | [56.98,60.40] | 17:57.20, 29:59.10, 43:59.70 |

Rule 3: a pass shows that the reader learns the value-to-key transition and composes reads inside the LM; it does not establish equivalence with the symbolic oracle condition, and no unseen-name claim is made.

## Cross-scale, descriptive (no test): Qwen3-0.6B-Base (P1-L r = 128) and Qwen3-1.7B-Base

| Contrast | 0.6B pp [CI] | 1.7B pp [CI] |
| --- | --- | --- |
| M1_vs_B4 | 21.55 [20.38,22.76] | 25.78 [24.53,27.03] |
| G1_reach | 0.98 [0.27,1.72] | 1.03 [0.29,1.80] |
| G2_fixed_query | 0.83 [0.06,1.55] | 1.67 [0.91,2.38] |
| G2_no_memory | 6.34 [5.76,6.93] | 7.17 [6.58,7.76] |
| B8_vs_M1 | -20.57 [-21.78,-19.36] | -24.75 [-26.07,-23.45] |
| M1_depth_reach | 25.92 [24.82,27.05] | 31.46 [30.27,32.63] |
