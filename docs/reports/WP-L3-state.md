# WP-L3 state card

Task docs/tasks/WP-L3_p1l_writer_fix.md; final docs/reports/P1L-writer-fix.md.
Branch wp/llm only. Merged master8001347 then WP-C5 at7f68a3c BEFORE science.
Frozen numeric source for probes and the original seven-arm screen:3fbc504
(core.scan_final). Keep running jobs and the current wake-up unchanged.
Amendment A1 below authorizes a new common source for the conditional rerun.
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

## Amendment A1 takes precedence over the old wake-up message

Orchestrator steering/master3548d6e, received while the screen was running:
do not follow the original stop-below.30 / one-arm-one-hour extension rule.
After all seven arms finish, merge master3548d6e or later into wp/llm and read
the amended task, including WP-B4 prefetch. Do not merge while jobs are running.
If no eligible writer reaches.80, lock the two highest-final-EM arms (ties:
lower final main loss), plus OA if absent, for a fresh common-source fast rerun.
Use scan_final and actual train.prefetch_workers:2 in the custom writer-fix
training loop, same seed/data/batch32/lr.003. Fit the longest measured schedule
into one <=2h PBS job using --scale-up docs/tasks/WP-L3_p1l_writer_fix.md;
evaluate at five evenly spaced checkpoints. Total task remains<=4nodeh.
Adapt the existing analyzer/selection and old --extension runner BEFORE locking
or rerunning: their original thresholds and single-arm extension are obsolete.
Test prefetch equality on compute. Preserve original screen provenance3fbc504.
Simplest rerun writer>=.80 wins (W0<W1<W4<W3<W2<W5); otherwise stop P1-L,
reporting>=.30 as partial learning. OA remains diagnostic, never a writer.
Then final compute audit --verify-export, all job costs, <=1page+tables report,
commit on wp/llm and exact final WP-L3 done: docs/reports/P1L-writer-fix.md.
