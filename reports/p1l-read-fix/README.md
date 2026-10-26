# WP-L4 read-fix preparation

Task `docs/tasks/WP-L4_p1l_read_fix.md`; choices `docs/reports/WP-L4-plan.md`;
continuation `docs/reports/WP-L4-state.md`. Persistent root:
`/work/xg24i002/x10041/LoopLora/runs/p1l/read-fix-20261027/`.

Tests55passed on3715437 mg0379, sourceb1c64c1. Initial3715417 had two failures
(fixed bf16 audit export and a faulty quantized-shuffle-effect assertion);
both XML reports remain. Four-arm 10% smoke (823/8225 steps,100/1000 eval episodes)
passed in3715461/3715462. Audit3715504 reproduced every smoke sample label,
checked all finite diagnostics, exact shared normalization buffers and inference
exports. `smoke-timing.json` deliberately exposes no quality scores. The 1.5×
training margin plus five extrapolated full evaluations and90s reserve predicts
2781s; use one50min job for each pair R/Rρ and RK/OAρ. Costs in `jobs.json`.

`release.json` freezes source/schedule before any new scientific score.
Science waits for final audited WP-L3 selecting no writer and its committed
report. Existing A1 job3714914 and its armed wake-up remain unchanged.

All commands below must run ON COMPUTE through `scripts/submit.py`, with an
explicit commit hash (submit.py resolves HEAD in the main checkout):

```sh
scripts/audit_p1l_read_fix.py --root <WP-L3-root> --out <WP-L4-root>/audit \
  --prior-decision <WP-L3-summary.json> --prior-report <P1L-writer-fix.md>

-m looplora.llm.read_fix_run --arms R Rrho --steps 8225 --out <WP-L4-root>/arms \
  --prior-decision <WP-L3-summary.json> --prior-report <P1L-writer-fix.md>
-m looplora.llm.read_fix_run --arms RK OArho --steps 8225 --out <WP-L4-root>/arms \
  --prior-decision <WP-L3-summary.json> --prior-report <P1L-writer-fix.md>

scripts/analyze_p1l_read_fix.py --root <WP-L4-root> --out <reports-directory> \
  --release <release.json> --e16-only
```

The immutable selection.json permits at most one E8 retrain (same8225 steps),
using `--arms <selected-arm> --e8-decision <selection.json> --out <root>/e8`
and the same prior-decision/report arguments. Only a best final EM>=.50 with no
E16writer>=.80 authorizes this. Final audit omits `--e16-only`, adds
`--verify-export`, verifies all inference predictions, and writes curves/CSVs.
The fourth validation checkpoint6580 matches A1 OA's first checkpoint exactly.

No-training audit saves per-query scalar samples and dense b/a/d/Gram/keys/
values/raw query/matrix/read reconstructions; headless OA is explicitly labeled.
It measures repeated-logit noise, zero/shuffled ACTUAL deployed memory (OA:
oracle value), age-stratified SIR, memory-output contrast, and per-loss gradients
on128 train episodes with zero optimizer steps. The task mentions a third A1
arm; none exists because OA is already among A1's two highest-EM arms.
