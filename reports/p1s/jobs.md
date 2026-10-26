# WP-H compute ledger

All 19 jobs exited 0. WP-H used **4.437222 / 8 node-hours**; including 4.040833 earlier P1-S node-hours gives **8.478056 / 13**. Requested walltime totals 7.400 node-hours. Each job used one Miyabi-G node and requested at most 39 minutes. Runtime is PBS allocated elapsed time, including startup; queue wait is excluded.

All submissions used `scripts/submit.py`, `regular-g` (PBS routed to `small-g`). The eight-process reader jobs used the [locked schedule](scheduling.json); shared writers used three processes. At most eight WP-H jobs were admitted concurrently. Source hashes and full log paths are in [jobs.json](jobs.json). The initial step fallback and packing assumptions are in [training_budget.json](training_budget.json).

| Job | Stage | Host | Requested (s) | Actual (s) | Node-hours | Exit |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [3712623.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-through-20261026-212508-2213548.log) | Original train-only throughput, procs2/4/8 | mg0116 | 600 | 220 | 0.061111 | 0 |
| [3712766.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/t-p1s-h-pipe-20261026-213436-2285325.log) | Reusable pipeline tests:335passed | mg0698 | 600 | 48 | 0.013333 | 0 |
| [3712797.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-fast-20261026-213728-2310558.log) | Batched-codebook throughput, procs4/8 | mg0840 | 300 | 67 | 0.018611 | 0 |
| [3712869.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-writers-20261026-214331-2340820.log) | Main shared T1 writers,3 processes | mg0843 | 1020 | 597 | 0.165833 | 0 |
| [3712912.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r00-20261026-214828-2363758.log) | reader-00 | mg0846 | 2340 | 1395 | 0.387500 | 0 |
| [3712913.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r01-20261026-214828-2363780.log) | reader-01 | mg0852 | 1740 | 1132 | 0.314444 | 0 |
| [3712914.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r02-20261026-214828-2363787.log) | reader-02 | mg0853 | 1740 | 1125 | 0.312500 | 0 |
| [3712915.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r03-20261026-214828-2363794.log) | reader-03 | mg0856 | 1740 | 1093 | 0.303611 | 0 |
| [3712916.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r04-20261026-214828-2363809.log) | reader-04 | mg0860 | 1740 | 1166 | 0.323889 | 0 |
| [3712917.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r05-20261026-214829-2363821.log) | reader-05 | mg0862 | 1740 | 1281 | 0.355833 | 0 |
| [3712918.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r06-20261026-214829-2363847.log) | reader-06 | mg0865 | 1740 | 1192 | 0.331111 | 0 |
| [3713034.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-analysisb-20261026-215749-2407568.log) | Synthetic full-scale analysis timing | mg0943 | 180 | 15 | 0.004167 | 0 |
| [3713061.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r07-20261026-215936-2416036.log) | reader-07 | mg0880 | 1740 | 1233 | 0.342500 | 0 |
| [3713146.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r08-20261026-220813-2474753.log) | reader-08 | mg0943 | 1740 | 1234 | 0.342778 | 0 |
| [3713170.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-r09-20261026-221016-2484567.log) | reader-09 | mg0841 | 1740 | 949 | 0.263611 | 0 |
| [3713171.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-f00-20261026-221016-2484582.log) | frozen-00 | mg0852 | 1740 | 951 | 0.264167 | 0 |
| [3713172.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-f01-20261026-221016-2484606.log) | frozen-01 | mg0865 | 1740 | 922 | 0.256111 | 0 |
| [3713173.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-f02-20261026-221016-2484609.log) | frozen-02 | mg1049 | 1740 | 951 | 0.264167 | 0 |
| [3713367.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-h-final-20261026-223118-2599400.log) | Global barrier, single-pass analysis, registry append | mg0841 | 720 | 403 | 0.111944 | 0 |

Main training used `03b62f8`; the single confirmation analysis and registry append used `7e1c96f`. The pipeline tests ran against the worktree immediately before code commit `0f9de87`: 335 passed in 40.32 seconds. No training failure or LR retry occurred. The 40 throughput trials were train-only; the analysis timing job used synthetic data exclusively.

The final analysis took 394.395 seconds inside the application (403 allocated seconds); peak RSS was 1,259,456 KiB. It completed under the 12-minute request derived from the 442.32-second synthetic projection. See [analysis_execution.json](analysis_execution.json).
