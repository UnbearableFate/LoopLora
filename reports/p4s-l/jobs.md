# P4-S-L compute ledger

WP-P4L used **0.144444 / 1 node-hours** across 5 one-node jobs. Requested walltime totals 0.433333 node-hours; the longest request was ten minutes. PBS allocated elapsed time includes startup and excludes queue wait. All scientific stages completed with exit 0; the initial test-only failure is included in the cost.

Every submission used `scripts/submit.py` and `regular-g` (PBS routed to `small-g`). No torch import, pytest, training or evaluation ran on the login node. The formal training wait used the authorized `miyabi-pbs-goal-wakeup` message workflow; its event was acknowledged before auditing the outputs.

| Job | Stage | Host | Source | Requested s | Allocated s | Node-hours | Exit |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [3715724.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p4l-prepare-20261027-035547-1719237.log) | Preparation tests and pre-head reference selection | mg0873 | 643c8c2 | 180 | 18 | 0.005000 | 0 |
| [3715740.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p4l-smoke-20261027-035955-1739281.log) | Initial compute tests; smoke blocked by aliased test fixture | mg0836 | 7265cdb | 300 | 19 | 0.005278 | 1 |
| [3715743.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p4l-smoke2-20261027-040139-1764891.log) | Corrected compute tests and 10-percent smoke | mg0897 | afda7ab | 300 | 60 | 0.016667 | 0 |
| [3715784.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p4l-train-20261027-040610-1788777.log) | Formal24head training and validation-only theta selection | mg0855 | ccb4ec5 | 600 | 351 | 0.097500 | 0 |
| [3715868.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p4l-confirm-20261027-041814-1853789.log) | Fresh confirmation, prepared-batch timing and adjusted-CI analysis | mg0852 | 6e4545e | 180 | 72 | 0.020000 | 0 |

The first suite exposed aliasing in a negative-test fixture: modifying the copied best-rule dictionary also modified its source threshold row. The fixture was isolated; production validation was not weakened. The corrected suite passed all 159 tests. No halting-head smoke or formal training ran in the failed job.

The complete uniform 10% smoke took 37.072 application seconds. Its scaled projection plus export/selection allowance was 372.329 seconds; 1.5× margin rounded up gave a ten-minute formal request. Formal training completed in 337.311 application seconds (351 allocated). Confirmation and timing took 44.815 application seconds; the combined confirmation/analysis job used 72 allocated seconds. See [compute_plan.json](compute_plan.json), [smoke.json](smoke.json), and [jobs.json](jobs.json).
