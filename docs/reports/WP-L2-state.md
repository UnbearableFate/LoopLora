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
- 3713139 D2: running (root/D2), source99daecb;3 lr,1500steps.
- 3713165 D3: running (mg1048,root/D3), source99daecb;2 lr,6400steps;
  val1500/3200/6400;3500s elapsed guard. Started22:10:38 JST Oct26.
Logs: /work/xg24i002/x10041/LoopLora/logs/pbs/d-l2-*-20261026-*.log.

D1: question-final o14 rank32 start99.4%, relation100%; fact-final tail84.15%,
head80.8625%, relation99.9875%. Identity available; limited support generalization.
Per-target probes do not establish simultaneous encoding by shared A. Full matrix
often overfits (train~100%, lower val); do not claim an information impossibility.

After D3: inspect D2/3 summaries and samples. D4 only if task condition satisfied
(D2 valEM>=.95 while D3 fails, or strong residual-only D1 evidence). If triggered,
add minimal tested model.input_site=residual option (u from pre-layer14 residual,
output still o_proj14); preserve default path and checkpoint site compatibility;
rerun2 D3 configs in <=1h job. This option is NOT built yet.
Then run scripts/analyze_p1l_diagnosis.py on compute, record all PBS costs, write
<=1-page+tables report, commit everything on wp/llm. Final exact line:
WP-L2 done: docs/reports/P1L-diagnosis.md
