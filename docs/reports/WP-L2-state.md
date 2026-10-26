# WP-L2 state card — complete

Report: [P1L-diagnosis.md](P1L-diagnosis.md).
Task: docs/tasks/WP-L2_p1l_writer_diagnosis.md; plan: WP-L2-plan.md.
Branch wp/llm; merged master d274869 before implementation.

All scientific work: seed17, E16 T1/n_rel4/headsharing.3, r32/layer14/L1,
batch32/Q4, memory-only; val1000 seed4242. No other seeds, test/confirm,
reader/joint/T2. Runtime only compute via submit.py.

- D1 source bcab714, job3713110:96 frozen full/rank32 probes completed.
- D2 source99daecb, job3713139:oracle3lrs/S1500; bestEM98.7%, NLL.101.
- D3 source99daecb, job3713165:2lrs/S6400; finalEM0/.1%.
- D4 source3c1cffb, job3713853:residual input,2lrs/S6400; finalEM.1/0%.
  Triggered by D2 success/D3 failure; option implemented/tested, output unchanged.
- Tests3713089/3713111/3713137/3713800:2/18/3/68 passed (overlap).
- Invalid test path3713783 exited4; corrected command passed3713800.
- Audits3714411 and3714453 passed. Final audit on mg0846 reproduced all
  probe/validation tables, paired labels and key-alignment statistics.

All11 jobs:7446 allocated seconds =2.06833 node-hours (cap4); each1node<=1h.
No tested writer feasible for T2. Keep original Pilot-1 reference only for
further diagnosis (o_proj, fact_final,1500steps,batch32,lr3e-4).
Leading diagnosis: learned matching/retrieval; not a unique causal isolation.

Results/commands/costs: reports/p1l-diagnosis/.
Raw checkpoints/samples:
/work/xg24i002/x10041/LoopLora/runs/p1l/diagnosis-20261026/.
No WP-L2 checks remain. Acknowledged wake events stay in .job-wakeup/ as local
tool state and are excluded from the report commit.
