# WP-L3 writer screen

Binding task: `docs/tasks/WP-L3_p1l_writer_fix.md`; before-results choices:
`docs/reports/WP-L3-plan.md`; frozen numeric release: `release.json`.
All scientific jobs use source `3fbc504`, including WP-C5 from master7f68a3c.

Persistent root:
`/work/xg24i002/x10041/LoopLora/runs/p1l/writer-fix-20261027/`.
The pre-tail probe job completed before any screen job was submitted.
Screen jobs pair OA/W0, W1/W4, W2/W3; W5 is the final single arm.
No scores choose system settings. Exact PBS specs and costs are recorded at close.

After all seven arms complete, audit on compute through submit.py:

```sh
scripts/analyze_p1l_writer_fix.py --root <persistent-root> \
  --out /work/xg24i002/x10041/LoopLora-wt/llm/reports/p1l-writer-fix \
  --release /work/xg24i002/x10041/LoopLora-wt/llm/reports/p1l-writer-fix/release.json \
  --screen-only
```

This writes an immutable `selection.json`. Only action `extend` authorizes
one further training run of its named non-oracle arm, with the same frozen
source, initialization and episode stream:

```sh
-m looplora.llm.writer_fix_run --extension <selected-arm> \
  --out <persistent-root>/extension/<selected-arm>
```

Request <=1node/1h; the extension's3450s elapsed guard leaves evaluation/export
reserve. It starts from scratch; upper bound100000steps is a sentinel, not a
promised step count. Final selection still requires EM>=.80. No other extension
or tuning is allowed. Final audit omits `--screen-only` and adds `--verify-export`;
it reloads the chosen inference export (OA if no writer passes) and checks all
1000 predictions. `writer.pt` has no training head or optimizer; `checkpoint.pt`
retains training state. Every auxiliary term is supervision beyond the answer.
# Amendment A1

The original extension instructions below are superseded by Amendment A1 in
`docs/tasks/WP-L3_p1l_writer_fix.md` on master3548d6e. The original screen remains
on3fbc504. After it finishes, merge master and use a common rerun source with
scan_final and actual prefetch_workers:2; lock two highest-EM arms (ties lower
final main loss) plus OA if absent, if no writer reaches.80. One<=2h packed job,
five evenly spaced evaluations, total task<=4nodeh. See the updated state card
and plan before using the original analyzer's selection or extension commands.

