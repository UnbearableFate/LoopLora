# WP-L2 state card

Task: docs/tasks/WP-L2_p1l_writer_diagnosis.md; final docs/reports/P1L-diagnosis.md.
Branch wp/llm only; master d274869 merged. User authorized complete execution.
Plan: WP-L2-plan.md. All runtime compute via scripts/submit.py; never login imports/tests.
Scientific seed17, E16 T1/n_rel4/headsharing.3, rank32/layer14/L1, batch32 Q4;
val1000 seed4242. No other seeds/splits, no reader/joint/T2. Total <=4 node-hours;
each job1node <=1h. Seven planned tests/diagnostic jobs cost far below4h.

Persistent root: /work/xg24i002/x10041/LoopLora/runs/p1l/diagnosis-20261026/
- 3713089 t-l2-probes: 2 tests passed (mg0086), source bcab714.
- 3713110 D1: COMPLETE96 probes (mg0943), source bcab714; all artifacts root/D1.
- 3713111 t-l2-oracle: 18 tests passed (mg0841), source9ffabc4.
- 3713137 t-l2-budget: 3 tests passed (mg0852), source99daecb.
- 3713139 D2: COMPLETE (root/D2), source99daecb;3 lr,1500steps; bestEM.987.
- 3713165 D3: COMPLETE (mg1048,root/D3), source99daecb;2 lr,6400steps;
  finalEM.0/.001, keytop1 .071/.075 vs chance.0625. PBS3395s.
- 3713783 first residual test submission canceled (bad nonexistent test path);
  scheduler start raced qdel:7 allocated seconds. Include this cost; no check passed.
- 3713800 residual + training/evaluation/run tests:68passed (mg0746), source3c1cffb.
- D4 submitted 23:13 JST; root/D4, source3c1cffb; same2lrs,6400steps.
Logs: /work/xg24i002/x10041/LoopLora/logs/pbs/d-l2-*-20261026-*.log.

D1: question-final o14 rank32 start99.4%, relation100%; fact-final tail84.15%,
head80.8625%, relation99.9875%. Identity available; limited support generalization.
Per-target probes do not establish simultaneous encoding by shared A. Full matrix
often overfits (train~100%, lower val); do not claim an information impossibility.

D4 was triggered by D2 success / D3 failure. Residual option is implemented/tested:
model.input_site=residual, pre-layer14 residual to A, output still o_proj14.
Original default and its writer checkpoint signature remain compatible; residual
writer signatures carry site and actual dimensions and reject cross-site load.
After D4: run scripts/analyze_p1l_diagnosis.py on compute, record all PBS costs, write
<=1-page+tables report, commit everything on wp/llm. Final exact line:
WP-L2 done: docs/reports/P1L-diagnosis.md
