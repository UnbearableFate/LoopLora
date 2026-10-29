# B8W: B8 warm start from M1-L4 (oracle_random, r = 128, Qwen3-0.6B-Base)

Test [108000, 110000), eval seed 4242, 2000 episodes, L = 4, seeds 17/29/43; 95% paired episode-bootstrap CIs, 2000 draws (97.5% in endpoints.json); N = 2000 continuation steps.

**Decision (docs/prereg/B8W.md): untying keeps the solution: the copied B8 stays within 2 pp of continued M1-L4 on hops 2 and 3 in every seed and the pooled interval excludes a loss worse than 2 pp (no equal-parameter status; G1-L stays failed; nothing about why B8 failed from a random start)**

Delta = untied@4 - shared@4 on hops 2 and 3: 0.90 pp [0.09, 1.71]; every seed within 2 pp: yes; interval excludes a loss worse than 2 pp: yes.

| Contrast | Comparison | Difference(pp) | 95% CI(pp) | Per-seed differences(pp) |
| --- | --- | ---: | --- | --- |
| DELTA_untied_vs_shared | untied@4 - shared@4 | 0.90 | [0.09,1.71] | 17:1.05, 29:0.90, 43:0.75 |
| hop1_untied_vs_shared | untied@4 - shared@4 | 2.77 | [1.55,4.02] | 17:2.85, 29:1.70, 43:3.75 |
| untied_vs_start | untied@4 - start@4 | 5.99 | [5.07,6.88] | 17:5.08, 29:6.38, 43:6.53 |
| untied_vs_start_hop1 | untied@4 - start@4 | 7.45 | [6.00,8.95] | 17:6.40, 29:8.75, 43:7.20 |
| shared_vs_start | shared@4 - start@4 | 5.09 | [4.18,6.01] | 17:4.03, 29:5.47, 43:5.78 |
| shared_vs_start_hop1 | shared@4 - start@4 | 4.68 | [3.33,6.02] | 17:3.55, 29:7.05, 43:3.45 |
| untied_vs_B8_original | untied@4 - B8@4 | 26.07 | [24.98,27.13] | 17:29.38, 29:20.03, 43:28.80 |
| untied_vs_B4 | untied@4 - B4@1 | 26.98 | [25.87,28.07] | 17:29.88, 29:21.38, 43:29.70 |
| shared_vs_B4 | shared@4 - B4@1 | 26.08 | [24.98,27.19] | 17:28.82, 29:20.47, 43:28.95 |
| start_vs_B4 | start@4 - B4@1 | 20.99 | [19.97,22.06] | 17:24.80, 29:15.00, 43:23.18 |

Descriptive: accuracy by hop, `accuracy.csv`; train loss, `train-loss.csv`; drift of the continued tensors, `drift.csv`.
