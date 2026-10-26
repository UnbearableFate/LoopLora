# WP-L4 state card — preparation authorized, science still gated

User: prepare WP-L4 while WP-L3 A1 job3714914 runs. Keep the existing A1 wake-up
.job-wakeup/bba58d0f-b223-4338-b04c-7399fdf9145b armed. On wake, finish WP-L3
FIRST: final compute audit/report/commit, then emit exact line
`WP-L3 done: docs/reports/P1L-writer-fix.md`. THEN continue WP-L4 if no writer
selected. WP-L4 scientific CLI checks final L3 decision and report. No Goal.

Merged master4a27457 (WP-C6 includes our previous WP-L3 code). New modules
read_fix.py/read_fix_run.py; audit_p1l_read_fix.py (no training), tests and
plan docs/reports/WP-L4-plan.md. ARMS R/Rrho/RK/OArho; Rrho/OArho are Rρ/OAρ.
Exact raw deployed read regression (stopped true-v target/norm), weight1;
RK distinct-address squared-cosine penalty1. Other W5 losses unchanged.
ρ.5 for Rrho/OArho, otherwise.25. Export strips auxiliary head/optimizer.

Initial tests3715417 mg0917 failed2/55: audit answer-logit BF16 NumPy conversion;
untrained shuffle test incorrectly assumed a quantized logit effect. Fixed float
export and strong exact deployed-memory/oracle-value assertions (production
control unchanged). Corrected tests3715437 mg0379 passed ALL55 in24.34s;
sourceb1c64c17fa34e57489f15e6ca2993ac55883a05d; includes existing WP-L3/B8 regressions.

Scientific provisional schedule8225 steps, eval1645/3290/4935/6580/8225; fourth
checkpoint exactly matches A1 OA6580 for OAρ comparison. Initial13160 proposal
reduced BEFORE smoke because1.5x A1 paired timing would exceed1h.
10% smoke823 steps / eval100 (five checkpoints), ALL4 arms/sourceb1c64c1,
submitted in two paired jobs s-l4-R-Rrho and s-l4-RK-OArho (see PBSspecs).
Root /work/xg24i002/x10041/LoopLora/runs/p1l/read-fix-20261027/smoke.
Do NOT read smoke quality scores before schedule lock. After both jobs pass,
run scripts/audit_p1l_read_fix_smoke.py ON COMPUTE via submit.py to verify all
samples/checkpoints and emit ONLY timings to reports/p1l-read-fix/smoke-timing.json.
It gates8225steps with1.5x mean step +5 extrapolated full evaluations +90s reserve.
Freeze release and schedule before any scientific val; no science before L3 gate.

Remaining: smoke audit, freeze source/schedule, wait existing A1 wake-up if needed.
After L3 report and negative decision, first no-training audit final A1 OA/W5
(no third rerun arm exists) with scripts/audit_p1l_read_fix.py --root L3ROOT
--out L4ROOT/audit --prior-decision ABS/L3reports/summary.json --prior-report
ABS/docs/reports/P1L-writer-fix.md. Then4scientificarms in2paired<=1h jobs,
seed17/T1E16/r32/layer14/L1/B32Q4/lr.003/scan_final/prefetch2/val1000seed4242.
Auditperquery b/a/d, exact Gram reconstruction, norms, gM/BgM, value-head decoding,
age-stratified SIR/parallel interference, zero/shuffle actual deployed memory,
identical repeated logits, train-minibatch per-loss grads without optimizer steps.

scripts/analyze_p1l_read_fix.py audits four arm records/samples and locks decision;
--e16-only first. R<Rrho<RK, finalEM>=.80 selects. Else best>=.50 gets ONE E8
retrain at same schedule via --e8-decision selection.json, then final audit
without --e16-only with --verify-export. No other writer screen after WP-L4.
Max4nodeh for ALL WP-L4 jobs, each1node<=1h. Track failed tests too. Never import
torch/runpytest or validators on login. Commit on wp/llm only. Final report
<=1page+tables docs/reports/P1L-read-fix.md and exact final line
WP-L4 done: docs/reports/P1L-read-fix.md
