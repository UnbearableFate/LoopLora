# P1-S pilot PBS cost ledger

All WP-D2 jobs: `scripts/submit.py`, `regular-g` (routed to `small-g`), one Miyabi-G node; exit 0. Elapsed seconds are scheduler history, not requested walltime; node-hours = nodes × elapsed seconds / 3600. Training jobs packed four processes per node. Sources and log paths are also in [jobs.json](jobs.json).

| Job / log | Stage | Host | Requested seconds | Elapsed seconds | Node-hours |
| --- | --- | --- | --- | --- | --- |
| [3711784.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/t-sym-d2-20261026-194101-1596542.log) | amendment tests and Pilot1 planning | mg0852 | 300 | 36 | 0.010000 |
| [3711859.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p1-00-20261026-194613-1624876.log) | pilot1/shard-00 | mg0191 | 2700 | 1986 | 0.551667 |
| [3711860.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p1-01-20261026-194614-1624976.log) | pilot1/shard-01 | mg0660 | 2700 | 1982 | 0.550556 |
| [3711861.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p1-02-20261026-194615-1625055.log) | pilot1/shard-02 | mg0661 | 2700 | 1989 | 0.552500 |
| [3711957.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/t-sym-d2-a1-20261026-195644-1676672.log) | A1 B5-lin/confidence tests | mg0921 | 300 | 36 | 0.010000 |
| [3712017.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/t-sym-pilot-plan-20261026-200800-1747135.log) | stage-specific plan regression tests | mg0268 | 180 | 8 | 0.002222 |
| [3712100.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p1-report-20261026-202234-1816981.log) | Pilot1 tables, writer diagnosis, Pilot2 planning | mg0755 | 300 | 15 | 0.004167 |
| [3712115.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p2-00-20261026-202459-1827904.log) | pilot2/shard-00 | mg0755 | 2700 | 1194 | 0.331667 |
| [3712116.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p2-01-20261026-202500-1828057.log) | pilot2/shard-01 | mg0757 | 2700 | 1191 | 0.330833 |
| [3712117.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p2-02-20261026-202501-1828105.log) | pilot2/shard-02 | mg0758 | 2700 | 1192 | 0.331111 |
| [3712118.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p2-03-20261026-202502-1828290.log) | pilot2/shard-03 | mg0759 | 2700 | 1224 | 0.340000 |
| [3712119.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p2-04-20261026-202504-1828394.log) | pilot2/shard-04 | mg0760 | 2700 | 1223 | 0.339722 |
| [3712120.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p2-05-20261026-202505-1828482.log) | pilot2/shard-05 | mg0763 | 2700 | 1226 | 0.340556 |
| [3712318.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-p2-audit-20261026-205249-1985185.log) | Pilot2 audit, LR selection, replication planning, 2000-episode timing | mg0919 | 300 | 34 | 0.009444 |
| [3712328.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-rep-00-20261026-205504-2049227.log) | replication/shard-00 | mg0712 | 900 | 317 | 0.088056 |
| [3712329.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-rep-01-20261026-205505-2049343.log) | replication/shard-01 | mg0714 | 900 | 292 | 0.081111 |
| [3712330.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-rep-02-20261026-205506-2049500.log) | replication/shard-02 | mg0191 | 900 | 310 | 0.086111 |
| [3712357.opbs](/work/xg24i002/x10041/LoopLora/logs/pbs/p1s-final-audit-20261026-210250-2105623.log) | Replication tables, all-run audit, registry | mg0904 | 300 | 23 | 0.006389 |

WP-D2 total: **3.966111 node-hours** (18 jobs).

For conservative accounting, prior WP-D tests/smoke are included below. Job 3711517 failed at CUDA startup; its consumed time is included. See [WP-D report](../../docs/reports/WP-D.md) for those jobs.

| Prior WP-D job | Elapsed seconds | Node-hours |
| --- | --- | --- |
| 3711498 | 30 | 0.008333 |
| 3711517 | 23 | 0.006389 |
| 3711546 | 22 | 0.006111 |
| 3711572 | 194 | 0.053889 |

Including prior WP-D: **4.040833 / 12 node-hours**; **7.959167 node-hours** remain.

Replication requested 15 minutes from the measured non-orthogonal Pilot-2 maximum (206.27 seconds per run), allowing two waves, larger E=18 episodes and a margin. Actual replication jobs took 292–317 seconds. The initial capacity/Pilot-2 requests were 45 minutes; all jobs stayed below the one-hour cap. The 2000-episode timing and final reporting requests were five minutes.
