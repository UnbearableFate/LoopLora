# WP-L4 read-fix, Amendments A1/A2

A2 overrides the four-run decision and total8nodeh limit below. Master2844d5e
was merged through df21654 before any WP-L4 validation score was read. Add
r=r_mem128 W5/R/OA, same lr.003 without search (flagged), alongside unchanged
rank32 jobs. All seven E16runs must finish before selection: r32R<Rrho<RK,
then r128W5<R; both oracle arms diagnostic only. E8 is at most one retrain
of the best learned arm across both ranks if its finalEM>=.50 and none>=.80.
Totalnodeh is uncapped by the user; eachjob<=1node2h and all costs are recorded.
The rank32 `release.json` is historical and immutable. Rank128 has its own
`release-r128.json`, `smoke-r128-timing.json` and persistent `arms-r128/` runs.
Before the three rank128 scientific arms are submitted, do not inspect any
WP-L4 validation quality score. Rank128 smoke is2400steps/100evalepisodes,
five equal checkpoints, pairedW5/R and soloOA; it exposes only timing/artifact
checks and freezes each largest2h schedule before science. Scientific source
is the exact tested/smoked commit, recorded by that release.
These OA arms supply the true learned fact value to the reader (WP-L3 OA);
they use `writer: learned`, and are distinct from WP-C7's external random oracle.
Rank128 W5/R have132096 training-only head parameters (128-to1024 plus bias),
discarded in their inference exports; OA has no auxiliary head.

The seven-run analyzer now additionally requires `--rank128-release` and
preserves rank in all tables, selection locks and export checks. E8release
also records rank. This paragraph supersedes older rank32-only commands below.

Task `docs/tasks/WP-L4_p1l_read_fix.md`; choices `docs/reports/WP-L4-plan.md`;
continuation `docs/reports/WP-L4-state.md`. Persistent root:
`/work/xg24i002/x10041/LoopLora/runs/p1l/read-fix-20261027/`.

WP-L3 finished and reported first (`cd5a45c`): no writer selected. Its final
A1 checkpoints are audited here, with identical original predictions, zero
repeat-logit drift, and independently verified Gram read reconstruction.
`audit.csv`, `audit-age.csv` and `audit-summary.json` preserve scalar measurements;
per-query vectors/samples remain under the persistent root's `audit/` directory.
The task mentions a third A1 arm; none exists because OA was among its top two.

Initial test3715417 had two failures, fixed before science; corrected3715437
passed55. Initial10% smoke823/8225steps,100/1000evalepisodes passed all4 arms
in3715461/3715462; timing/artifact audit3715504 verified all samples, exports and
shared normalization. No smoke quality score selected settings. A1 was received
BEFORE ANY scientific submission and superseded8225steps/1h/4nodeh. No scientific
jobs were canceled. Reuse those measured timings for the new budgets; the
initial smoke is a smaller fraction of A1's expanded schedule, explicitly recorded.
A1 tests3715659 passed57, including maximal budget and nested E8-lock handling.

`release.json` freezes per-arm schedules, source0b03d60 and the A1 budget rule:
largest multiple5 satisfying1.5*(S*measured mean step + N_eval*conservative full
validation time)+90s setup<=7200s. Validation time is the maximum per-arm100episode
smoke evaluation x10. Primary checkpoints are five equal intervals. OAρ also
has one predeclared6580-step diagnostic, reserved in its6evaluation budget, to
fulfill its matched-step OA comparison; only the final primary checkpoint selects.
ML tensor math/data/initialization are unchanged from smoke b1c64c1.

| Arm | Steps | Primary checkpoints |
| --- | ---: | --- |
| R | 23590 | 4718 / 9436 / 14154 / 18872 / 23590 |
| Rρ | 24220 | 4844 / 9688 / 14532 / 19376 / 24220 |
| RK | 21980 | 4396 / 8792 / 13188 / 17584 / 21980 |
| OAρ | 27040 | 5408 / 10816 / 16224 / 21632 / 27040 |

Every command below runs ON COMPUTE through scripts/submit.py. Scientific pairs
use source0b03d60,1node/02:00:00 and
`--scale-up docs/tasks/WP-L4_p1l_read_fix.md`; total WP-L4<=8nodeh.

```sh
-m looplora.llm.read_fix_run --arms R Rrho --schedule <absolute-release.json> \
  --out <root>/arms --prior-decision <WP-L3-summary.json> \
  --prior-report <P1L-writer-fix.md>
-m looplora.llm.read_fix_run --arms RK OArho --schedule <absolute-release.json> \
  --out <root>/arms --prior-decision <WP-L3-summary.json> \
  --prior-report <P1L-writer-fix.md>

scripts/analyze_p1l_read_fix.py --root <root> --out <reports-directory> \
  --release <release.json> --e16-only
```

The immutable selection.json permits at most ONE E8 retrain, only if no E16writer
reaches.80 and the best reaches.50 (ties simplicity). Size its own A1 schedule
from E8 smoke without reading quality, save separate e8-release.json (do not
overwrite the E16 release), then use `--arms <eligible-arm> --e8-decision
<selection.json> --schedule <e8-release.json> --out <root>/e8` and the same prior
arguments. Same seed17/LR/objective, five fixed primary checkpoints,2h job cap.
Final audit omits `--e16-only`, adds `--verify-export` and, if triggered,
`--e8-release <e8-release.json>`. It verifies all inference predictions and
writes curves/CSVs. No further writer screen/extension follows WP-L4.

No-training audit measures b/a/d, exact Gram reconstruction, raw norms, gM/BgM,
clean/read value-head decoding (unavailable for headless OA), age-stratified
SIR/interference, zero/shuffled ACTUAL deployed memory (OA: supplied value), and
per-loss gradients on128train episodes with zero optimizer steps. Failed tests
and all audits/smokes count in `jobs.json`, separately from completed WP-L3 costs.
