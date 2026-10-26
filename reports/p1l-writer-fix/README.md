# WP-L3 writer screen and Amendment A1

Task: `docs/tasks/WP-L3_p1l_writer_fix.md`. Before-results intervention choices
and amendment receipt: `docs/reports/WP-L3-plan.md`. Persistent artifacts:
`/work/xg24i002/x10041/LoopLora/runs/p1l/writer-fix-20261027/`.

`release.json` preserves the original screen source3fbc504 (WP-C5 scan_final),
including its superseded extension rule. The pre-tail probe job completed before
any screen submission. All seven arms finished before master3548d6e/WP-B4 was
merged for A1. No running job was changed.

A1 replaces the original stop-below.30 and single-arm extension rules. The two
highest-final-EM arms are ranked over all seven arms, including diagnostic OA,
with lower final main loss breaking ties; OA is added only if absent. This follows
A1's explicit "plus OA ... if it is not among them" wording. OA is never eligible
as the P1-L writer. The audited immutable `selection.json` locks the chosen arms.

The custom writer-fix loop consumes WP-B4 `BatchStream` with two spawn workers
per training process, reusing CPU token tensors and deriving the same fact
positions without rerendering. `check_p1l_writer_fix_prefetch.py` compares two
concurrent real-model OA/W5 processes with workers0/2, 100 steps per setting:
exact final adapter/auxiliary-head parameters and logged losses, plus timings.
It reads no validation scores. This check sizes a common fixed rerun schedule.

Run every command below ON COMPUTE, using `scripts/submit.py --commit <hash>`,
never directly on login. Audit the completed screen before any rerun:

```sh
scripts/analyze_p1l_writer_fix.py --root <persistent-root> \
  --out <absolute-reports-directory> --release <absolute-release.json> --screen-only
```

A1's locked rerun uses the common source/schedule in `rerun-release.json`:

```sh
-m looplora.llm.writer_fix_run --rerun-selection <absolute-selection.json> \
  --steps <registered-steps> --out <persistent-root>/rerun
```

One node, two concurrent training processes, <=2h via
`--scale-up docs/tasks/WP-L3_p1l_writer_fix.md`; total task<=4nodeh. Fresh same
seed/initialization/episode stream, batch32, lr.003, scan_final, prefetch_workers2,
five equally spaced validation checkpoints. The final checkpoint applies the
original.80 writer threshold; OA is diagnostic. No other tuning or training.

Final audit omits `--screen-only`, adds `--rerun-release <absolute-rerun-release>`
and `--verify-export`, and reproduces all saved sample metrics and paired labels.
`writer.pt` contains neither optimizer nor auxiliary head; `checkpoint.pt` retains
training state. Auxiliary address/value labels are supervision beyond the answer.
