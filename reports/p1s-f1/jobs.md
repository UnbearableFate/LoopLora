# WP-F1 PBS ledger

All jobs used scripts/submit.py and one Miyabi-G node. Costs use scheduler elapsed time, including setup; every exit marker is 0.

| Job / log | Stage | Host | Queue requested | Requested | Actual seconds | Node-hours |
| --- | --- | --- | --- | --- | --- | --- |
| [3712630.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-smoke-20261026-212621-2219359.log) | f1-smoke | mg0584 | regular-g | 00:20:00 | 359 | 0.099722 |
| [3712754.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-train-0-20261026-213401-2279505.log) | f1-train-0 | mg0678 | regular-g | 00:50:00 | 2124 | 0.590000 |
| [3712755.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-train-1-20261026-213401-2279528.log) | f1-train-1 | mg0679 | regular-g | 00:50:00 | 2152 | 0.597778 |
| [3712756.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-train-2-20261026-213401-2279593.log) | f1-train-2 | mg0680 | regular-g | 00:50:00 | 2147 | 0.596389 |
| [3712757.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-train-3-20261026-213401-2279620.log) | f1-train-3 | mg0681 | regular-g | 00:50:00 | 2162 | 0.600556 |
| [3712758.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-train-4-20261026-213402-2279670.log) | f1-train-4 | mg0682 | regular-g | 00:50:00 | 2157 | 0.599167 |
| [3712759.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-train-5-20261026-213402-2279733.log) | f1-train-5 | mg0683 | regular-g | 00:50:00 | 2133 | 0.592500 |
| [3712884.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/t-f1-pipeline-20261026-214554-2351842.log) | t-f1-pipeline | mg0041 | debug-g | 00:05:00 | 13 | 0.003611 |
| [3713198.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-eval-0-20261026-221230-2495757.log) | f1-eval-0 | mg0045 | debug-g | 00:10:00 | 271 | 0.075278 |
| [3713199.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-eval-1-20261026-221230-2495773.log) | f1-eval-1 | mg0046 | debug-g | 00:10:00 | 272 | 0.075556 |
| [3713200.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-eval-2-20261026-221230-2495796.log) | f1-eval-2 | mg0004 | debug-g | 00:10:00 | 264 | 0.073333 |
| [3713201.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-eval-3-20261026-221230-2495819.log) | f1-eval-3 | mg0044 | debug-g | 00:10:00 | 264 | 0.073333 |
| [3713202.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-eval-4-20261026-221230-2495838.log) | f1-eval-4 | mg0040 | debug-g | 00:10:00 | 264 | 0.073333 |
| [3713203.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-eval-5-20261026-221231-2495845.log) | f1-eval-5 | mg0028 | debug-g | 00:10:00 | 272 | 0.075556 |
| [3713306.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-analysis-20261026-222145-2539147.log) | f1-analysis | mg0003 | debug-g | 00:30:00 | 204 | 0.056667 |
| [3713353.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/f1-report-20261026-222710-2563810.log) | f1-report | mg0003 | debug-g | 00:03:00 | 7 | 0.001944 |

**Total: 15065 node-seconds = 4.184722 / 6 node-hours.**

Training: commit 7aed0f3, jobs 3712754–3712759. Shared evaluation and inference: commit a76e46b. The initial validation/timing job and the 83-test F1 check used the working tree; their exact submission arguments are preserved in jobs.json.

Confirmation barrier: all 60 completed training records, followed by all 60 samples-only confirmation files. Analysis job 3713306 sealed the metadata-only global release before reading any confirmation archive; one analysis pass finalized metrics and computed all intervals. Job 3713353 summarizes existing published A5/statistics tables without additional bootstrap resampling.

The test job command was:

```sh
scripts/submit.py --name t-f1-pipeline --queue debug-g --walltime 00:05:00 --worktree . -- -m pytest -q -o addopts= tests/test_p1s_f1.py tests/test_sym_main_evaluate.py tests/test_sym_specificity.py tests/test_sym_gates.py
```

Result: 83 passed in 5.76 s, job 3712884.opbs, host mg0041. The timing job also passed the symbolic model, statistics and evaluation-override suites. No runtime workload ran on login.
