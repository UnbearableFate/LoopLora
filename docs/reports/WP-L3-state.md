# WP-L3 state card

Task docs/tasks/WP-L3_p1l_writer_fix.md; final docs/reports/P1L-writer-fix.md.
Branch wp/llm only. Merged master8001347 then WP-C5 at7f68a3c BEFORE science.
Frozen numeric source ALL scientific runs/extension:3fbc504 (core.scan_final).
No future numerical/backend merge into these comparisons.
Root: /work/xg24i002/x10041/LoopLora/runs/p1l/writer-fix-20261027/.

Scope: T1 E16/n_rel4/headsharing.3, seed17/r32/layer14/L1, batch32/Q4,
lr.003/S4800, val1000 seed4242 at1200/2400/4800. No other seeds/splits,
T2/reader/joint. Eachjob1node<=1h; total<=4nodeh. All runtime via submit.py;
never import torch or run pytest on login.

- Test3714689 (mg0390):70 passed; source3fbc504.
- Probe3714703 (mg0390):4 pre-tail probes complete; full/r32 head83.975/95.6125%,
  relation99.99375/99.9875%. Completed BEFORE screen submissions.
- Screen3714716:OA/W0;3714717:W1/W4;3714718:W2/W3;3714719:W5.
  All use source3fbc504. First3 jobs40min; last30min. Logs under
  /work/xg24i002/x10041/LoopLora/logs/pbs/s-l3-*-20261027-0050*.log.

All implementation/tests ready. See WP-L3-plan.md for exact intervention choices.
Analyzer scripts/analyze_p1l_writer_fix.py is committed in cb61b18. After ALL
seven complete, run it ON COMPUTE with --screen-only and --release
reports/p1l-writer-fix/release.json; root above, out absolute reports directory.
It audits every sample/probe and writes immutable selection.json.

Rule W0<W1<W4<W3<W2<W5: simplest>=800 correct wins. Else if any>=300,
exactly ONE highest-final-EM arm extends (exact ties use simplicity), fresh same
seed/stream, source3fbc504, --extension ARM --out root/extension/ARM,1h job.
Do not extend before lock. Otherwise stop P1-L. OA never eligible.
Then final compute audit --verify-export, all job costs, <=1page+tables report,
commit on wp/llm and exact final WP-L3 done: docs/reports/P1L-writer-fix.md.
