# P1-L oracle-random r32 confirmation

S=17000; seeds17/29/43; 97.5% paired episode-bootstrap CIs,2000 draws.
G1-L: **fail**. G2-L: **fail**.

| Contrast | Difference(pp) | CI(pp) | Per-seed differences(pp) |
| --- | ---: | --- | --- |
| G1_reach | -0.01 | [-0.11,0.10] | 17:0.03, 29:-0.07, 43:0.03 |
| G1_hop1 | 0.00 | [-0.10,0.10] | 17:-0.10, 29:0.00, 43:0.10 |
| G1_typed | 0.06 | [-0.02,0.15] | 17:0.07, 29:0.05, 43:0.05 |
| G2_fixed_query | -0.05 | [-0.17,0.07] | 17:0.07, 29:-0.20, 43:-0.03 |
| G2_no_memory | 0.05 | [-0.03,0.15] | 17:0.03, 29:0.03, 43:0.10 |
| M1_vs_B4 | 0.06 | [-0.05,0.17] | 17:0.03, 29:-0.05, 43:0.20 |
| M1_vs_B5 | 0.02 | [-0.11,0.14] | 17:0.07, 29:-0.18, 43:0.15 |
| M1_vs_B3 | 0.12 | [0.03,0.22] | 17:0.03, 29:0.05, 43:0.27 |
| B8_vs_M1 | -0.07 | [-0.19,0.06] | 17:0.00, 29:-0.03, 43:-0.18 |
| M1_depth_reach | 0.03 | [-0.08,0.13] | 17:-0.05, 29:0.00, 43:0.12 |
| M1_depth_far | 0.00 | [-0.10,0.11] | 17:-0.07, 29:-0.03, 43:0.10 |
| M1_walk_specificity | 0.02 | [-0.07,0.12] | 17:0.02, 29:0.03, 43:0.03 |

## Amendment B7: memory-use check

Hop-1 full-vocabulary EM on the confirmation set; each reader minus B3-L4@4 (no memory); 97.5% paired episode-bootstrap CIs, 2000 draws.
Memory-use check: **fail**.

| Contrast | Reader EM(%) | B3-L4@4 EM(%) | Difference(pp) | CI(pp) | Per-seed differences(pp) |
| --- | ---: | ---: | ---: | --- | --- |
| B4_vs_B3_hop1 | 0.07 | 0.03 | 0.03 | [0.00,0.10] | 17:0.05, 29:0.05, 43:0.00 |
| B8_vs_B3_hop1 | 0.07 | 0.03 | 0.03 | [-0.05,0.13] | 17:-0.05, 29:0.05, 43:0.10 |
| M1_vs_B3_hop1 | 0.15 | 0.03 | 0.12 | [0.02,0.23] | 17:0.00, 29:0.10, 43:0.25 |

The readers did not learn to use the oracle memory (rule 1): the G1-L, G2-L and Amendment-B6 outcomes above are reported but not interpreted as evidence for or against multiple reads.
Rule 2: no claim about multiple reads in the LM at r = 32, whatever the gate outcomes.
