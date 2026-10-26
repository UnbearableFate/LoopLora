# WP-L3 state card

Active task docs/tasks/WP-L3_p1l_writer_fix.md, including Amendment A1.
Final docs/reports/P1L-writer-fix.md; branch wp/llm only. No Goal is used.
Root /work/xg24i002/x10041/LoopLora/runs/p1l/writer-fix-20261027/.
Never import torch/run pytest on login; runtime only via scripts/submit.py.

Completed original source3fbc504 (WP-C5 scan_final, merged before science):
- Tests3714689 mg0390:70 passed,32s.
- Pre-tail probes3714703 mg0390:4 fits,33s, before screen submissions.
  Full/r32 head83.975/95.6125%; relation99.99375/99.9875%.
- Screens3714716 OA/W0 (1498s),3714717 W1/W4 (1587s),
  3714718 W2/W3 (1644s),3714719 W5 (1030s); all complete/exit0.
- Final screen EM: OA.221,W0.000,W1.000,W2.005,W3.006,W4.003,W5.078.

A1 supersedes old stop-below.30 / one-arm-one-hour extension instructions.
After ALL screen jobs finished, merged master3548d6e (includes WP-B4 prefetch).
New common numerical source0713cfb66fc9d1ee5f075f3bbc886167722292de.
Custom fit consumes BatchStream workers2 and aligns prefetched CPU token spans
without rendering again. Test/audit/systems job3714886 mg0848:68 passed;
all original samples/metrics and paired labels reproduced; OA/W5 workers0/2
100-step final parameters and logged losses exactly equal; total290s.

Audited immutable reports/p1l-writer-fix/selection.json (committeda4afc80)
locks OA and W5 as the highest two of all seven arms, following A1's explicit
"plus OA ... if it is not among them" wording. OA is diagnostic, never eligible.
W5 mean/p95 prefetched step=.199155/.209157s; OA mean=.166809s.
reports/p1l-writer-fix/rerun-release.json freezes common32900 steps,
evaluations6580/13160/19740/26320/32900, same seed17/data/init, batch32/Q4,
lr.003,T1E16/r32/layer14/L1/o_proj, val1000seed4242, scan_final/prefetch2.
Schedule is largest multiple100 fitting (7000-100)/slow-step-p95;
fit guard7000s, PBS7200s via --scale-up docs/tasks/WP-L3_p1l_writer_fix.md.
Training reruns fresh, not resumed. Final fixed checkpoint applies.80 threshold.

Spent6114s=1.69833nodeh before rerun; full7200s leaves1086s for final audit.
Total<=4nodeh; no more scientific training after A1. No other seeds/splits,
T2, reader, joint, test/confirmation runs. All per-arm records/samples persistent.

After rerun finishes, audit on COMPUTE via submit.py using current analysis
commit (numerics stay0713cfb): scripts/analyze_p1l_writer_fix.py --root ROOT
--out /work/xg24i002/x10041/LoopLora-wt/llm/reports/p1l-writer-fix
--release ABS/release.json --rerun-release ABS/rerun-release.json --verify-export.
Do not pass --screen-only. Analyzer verifies all samples, exact fixed5 evals,
source/config, auxiliary-head-free exports and identical reload predictions for
both rerun arms; writes summary, CSVs, loss and EM curves. Simplest rerun writer
with FINAL EM>=.80 wins; otherwise P1-L stops, reporting>=.30 as partial learning.
Collect all job costs including final audit; write <=1page+tables report with
all screen/rerun EM, diagnostics, probes, supervision, OA link limitations,
selection, curves and costs. Commit on wp/llm only. Final exact line:
WP-L3 done: docs/reports/P1L-writer-fix.md
