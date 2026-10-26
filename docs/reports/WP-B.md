# WP-B — Symbolic relation episodes

Completed 2026-10-26 on `wp/data`; implementation commit `df55f9b`.

Implemented NumPy-only `TaskSpec`, `EpisodeArrays`, deterministic sliceable
`generate`, all six tasks (both T3 bases), and all three counterfactuals in
`src/looplora/data/relations.py`. Added independent dictionary-oracle tests and
`scripts/benchmark_relations.py`. No dependencies or infrastructure were changed.

**Validation:** 105 tests passed, exit 0, Miyabi-G `mg0852`, PBS
`3711448.opbs`. Oracle answers, topology, history order, hop/rate distributions,
slice/RNG independence, split separation, unseen entities and counterfactuals pass.

```bash
scripts/submit.py --name t-wpb-data --queue regular-g --walltime 00:03:00 --worktree . --wait -- -m pytest -q tests/test_relations.py
scripts/submit.py --name b-wpb-data --queue regular-g --walltime 00:03:00 --worktree . --wait -- scripts/benchmark_relations.py --count 20000 --repeats 3
```

**Throughput:** benchmark job `3711449.opbs`, `mg0853`, exit 0; one process,
E=24, Q=4, median of three 20,000-episode runs; Python 3.13.13, NumPy 2.5.4.

| Task / base | Episodes/s |
| --- | ---: |
| t1 | 26,067 |
| t2_chain_end | 27,514 |
| t2_walk | 26,283 |
| t2_typed | 21,843 |
| t3_update / t1 | 14,858 |
| t3_update / t2_chain_end | 8,628 |
| t3_missing / t1 | 22,763 |
| t3_missing / t2_walk | 21,706 |

All T2 variants exceed the 20k/s target. Logs are under
`/work/xg24i002/x10041/LoopLora/logs/pbs/`:
`t-wpb-data-20261026-190726-1279178.log` and
`b-wpb-data-20261026-190728-1279437.log`.

**Decisions and deviations:** [contract details](../tasks/WP-B_contract.md).
Path counterfactuals fork N×Q independent contexts with Q=1 to prevent conflicting
edits; source mappings are metadata. Swaps return recomputed oracle labels and
`meta['pairing']`; score the control against the original answers. Missingness is
per episode, anchored on query zero; chain-update rate counts paths using updated
keys. E includes obsolete facts. Seen splits partition the minimum fact entity
modulo three, preventing even tiny-pool collisions; this imposes a small explicit
label restriction. `regular-g` replaced the full debug queue; cancelled queued
job `3711344.opbs` never ran. All runtime checks ran on compute via the wrapper.

**Limits:** path counterfactuals require answerable nonempty paths and spare pool
entities; infeasible pools/specs fail explicitly. Mixed chain-update queries
require two eligible chains. No unresolved implementation issues or required
checks remain. Independent review notes: [WP-B_test_notes.md](WP-B_test_notes.md).
