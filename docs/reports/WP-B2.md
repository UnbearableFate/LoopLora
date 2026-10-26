# WP-B2 — Terminal-preserving successor swaps

Completed on `wp/data`, implementation commit `a876ba3`. First fast-forwarded
to current master `f6e5cb9`, which includes requested commit `4ca9d4f`.

Added `counterfactual(arrays, "swap_successors", seed)` for `t2_chain_end`
and `t2_typed`. Each query forks into its own Q=1 context with the existing
source metadata and derived episode seed. The operation samples another chain
uniformly, samples one edge uniformly on each required path, then swaps tails.
It recomputes the new terminal answer and true hop length. Heads, tail counts,
chain starts, terminals, presentation order and typed distractors are preserved.
Unsupported tasks and episodes without a second chain raise clear errors.
No fresh entities are needed. Generator code and existing counterfactual paths
retain their behavior.

Validation ran through the submission wrapper on Miyabi-G compute:

```bash
scripts/submit.py --name t-wpb2 --walltime 00:03:00 --worktree . --wait -- -m pytest -q -o addopts= tests/test_relations.py
```

**120 passed in 3.35s**, exit 0; PBS `3712605.opbs`, host `mg0016`, queue
`debug-g`, 2026-10-26. This includes the existing 105 cases and 15 added cases
covering both supported tasks, seen/unseen entities, independent dictionary-walk
answers, acyclicity, functional keys, preserved global counts, changed answers,
new hop lengths, immutable inputs, deterministic output and metadata, untouched
distractors, uniform chain/edge sampling, fully occupied entity pools, and error
cases. The existing global RNG isolation check also exercises the new kind.

Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/t-wpb2-20261026-212340-2206376.log`.
All runtime checks ran on compute; `git diff --check` passed.

Deviations: none. Open issues: none. No required checks were skipped.
