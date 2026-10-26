# WP-L4 preparation and before-results choices

Task docs/tasks/WP-L4_p1l_read_fix.md; branch wp/llm. Merged master4a27457.
WP-L3 A1 source0713cfb job3714914 remains unchanged, with existing wake-up armed.
On that wake, FINISH WP-L3 FIRST: final audit, committed P1L-writer-fix.md, then
emit `WP-L3 done: docs/reports/P1L-writer-fix.md`. Continue WP-L4 only if its
final audited decision selects no writer. Tests/smoke preparation are authorized
now. Scientific CLI requires final WP-L3 summary and completed report.

Read task, WP-L3 task/state/plan, theory11.1/11.5, WP-C6. WP-L3 final report
is pending and must be read when it exists. Codegraph is unavailable.

Arms: R, Rrho (Rρ), RK, OArho (OAρ). R/Rrho/RK reuse W5's exact initialization,
pooled256-training-episode normalization, InfoNCE weight1/temperature.1 and
clean-value CE weight1. Add mean over questions of squared raw deployed-read
error divided by stopped target squared norm+1e-6, stopping ALL target/denominator
gradients; lambda1. Rrho/OArho use rho.5; other arms.25. RK additionally averages
squared off-diagonal KEY COSINES over distinct (head,relation) addresses, lambda1;
repeated addresses are excluded. No softmax/read/query normalization at runtime.
OAρ retains OA's unstandardized raw values, no extra auxiliaries. Auxiliary head
is discarded at inference, retained only in training checkpoint and audit.

All scope same T1E16/n_rel4/headsharing.3/seed17/r32/layer14/L1/o_proj/B32/Q4,
lr.003, AdamW(.01 weight decay),20-step warmup, gradclip1, val1000seed4242.
Use scan_final and two prefetch workers per process. No test/T2/reader/joint.
Before smoke scores, revised provisional target8225 steps, with a10% smoke823 steps
for ALL4 arms in two paired jobs, eval100episodes at five roughly equal steps.
The fourth evaluation is6580, an existing A1 OA checkpoint, giving OAρ an exact
matched-step reference. The initial13160 proposal would exceed1h with the
theory's1.5x margin at A1's measured.209s paired step, so it was reduced BEFORE
smoke submission or scores. Smoke ONLY timings/finite-output/artifact gates are read until
final schedule is fixed. Retain target if measured1.5x training time plus setup/
5 full evaluations fits1h; otherwise reduce paired packing or target by a declared
common schedule before any scientific score. Never choose from smoke accuracy.

No-training A1 audit uses its ACTUAL two rerun arms OA/W5: A1 selected OA and W5,
so the task's mentioned third rerun arm does not exist. Save all b/a/d vectors,
keys/values/Gram/raw q/stored and deployed reads, verify triangular reconstruction
relative error<1e-4; report norms, gM/BgM, SIR/parallel interference and target-age
strata. OA's deployed read is its raw true value; coefficient reconstruction is
of the bypassed stored matrix, labeled accordingly. W5 clean-v/read decoding
uses the saved training-only head; OA has no such head, report unavailable.
Evaluate each checkpoint twice (exact logit/noise floor) and zero/shuffle ACTUAL
memory (OA: supplied oracle_values), preserving x/u. Donors are a deterministic
1000-episode derangement, seed derive_seed(17,'read-fix','audit-memory-shuffle').
Four training minibatches measure per-loss parameter gradients/cosines and
would-clip fraction, no optimizer steps, no validation gradients.

Decision order R<Rrho<RK, final EM>=.80. Else best final EM>=.50 gets exactly
one E8 retrain at same fixed schedule; exact score ties use simplicity. No other
screen/extension. At most4 allocated nodeh across ALL WP-L4 jobs, each<=1node1h.
Scientific execution waits for WP-L3 report/decision even if smoke finishes first.

## Smoke timing gate, before any WP-L4 scientific score

All4 smoke arms/sourceb1c64c1 passed; artifact audit3715504 verifies all labels,
finite diagnostics, exports and exact shared normalization. Only timing metadata
was exposed. Slow RK mean.213188s; conservative full eval12.102s;1.5x training
plus5eval+90s=2780.715s. Freeze8225steps/five1645step intervals, two paired50min
jobs, with OAρ6580 exactly matched to A1 OA. Scientific release remains gated
on final negative WP-L3 decision/report; original A1 wake-up stays armed.

## Amendment A1 (03:39 JST), BEFORE ANY scientific submission

User/master70d29d4 (merged through ad966aa) replaces the8225-step/1h release:
each arm gets its largest smoke-sized2h schedule,1.5x margin; total8nodeh.
No scientific arms had been submitted, so no jobs were canceled. ML tensor
math/data/initialization remain unchanged from smoke b1c64c1; only budget/CLI
schedule and guard change. For each arm, choose largest multiple5 satisfying
1.5*(S*its measured mean step + N_eval*its conservative full-eval time)+90<=7200.
Full-eval time is its maximum100episode smoke eval x10. Primary evaluations are
five equally spaced checkpoints. OAρ additionally gets ONE predeclared6580-step
matched diagnostic to fulfill the original comparison to A1 OA; it does not
select checkpoints/arms and its cost is reserved (6evals for OAρ,5others).
If E8 is triggered, use its own smoke-sized A1 release before its retrain; the
.50/.80 gates, seed17, LR/objective and one-retrain limit are unchanged.

Stage0 audit3715621 mg0837 passed, optimizer steps0: exact Gram reconstruction,
repeat-logit drift0. Reconstruct dense coefficients independently and compare
all baseline predictions with saved A1 samples before release. All scientific
submissions use --scale-up docs/tasks/WP-L4_p1l_read_fix.md and the frozen
per-arm schedule JSON; no adaptive length/accuracy-driven extension.

## Amendment A2, before any WP-L4 validation score is read

Merged master2844d5e through df21654. Add r=r_mem128 W5/R/OA, seed17,
T1E16/B32/Q4/L1/layer14/o_proj, rho.25, lr.003 without search (flagged).
Rank32 science remains on frozen0b03d60; no running job is modified.
For rank128, run2400step systems smokes (10% of provisional24000),100episode
validation at480/960/1440/1920/2400, paired W5/R and soloOA matching science
packing. Read only artifact/finite-output checks and timings. Each arm then
gets its own largest multiple5 whose measured training+5validation time,
multiplied by1.5 plus90s reserve, fits7200s. Freeze a SEPARATE release-r128.json
before training; do not overwrite rank32release.json. No WP-L4 validation score
is read until all three rank128 scientific arms are submitted.

Selection waits for ALL7 E16runs. Candidate order r32R<Rrho<RK, then
r128W5<R, finalEM>=.80. Both oracle arms remain diagnostic only. If none
passes but best learned arm over both ranks>=.50, exactlyONE E8retrain at
that arm's rank, own smoke-sized<=2h release, same seed/LR/objective. Ties
use the candidate order. Totalnodeh cap removed by user; eachjob<=1node2h,
alljobs still recorded. No later writer screen/extension.
