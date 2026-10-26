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
