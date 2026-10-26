# WP-C6 — B8 in the LLM pipeline (2026-10-27)

**B8 is supported with four step-specific R/U/Q readers and fixed L=4.**
Configuration, support passes, training and evaluation handle writer, joint and
reader stages; both input sites and both evaluation protocols work. Ordinary
writer signatures remain independent of reader sharing, and frozen writers can
be consumed by M1 or B8. Existing variants' numerics and defaults are unchanged.
Merged master **45f6d14** then wp/llm **c5879d3**, without conflicts.

All seven WP-L3 options work with B8 through
`configuration(arm, variant="B8")` and `WriterFixExperiment`: standardization,
address/value auxiliaries, paired-position writes, oracle addressing and
inference export. Address supervision uses the first read's Q[0] z⁰; OA uses
each step's R/U. WP-L3 retains its original truncated memory-only scope and its
default M1/L1 scientific configurations. M1/B5/B3 reader/joint training with
`train.depths: [4]` is explicitly tested as fixed-depth training.

**101 tests passed in 28.02 s**, PBS **3715193**, `mg0013`, source **214518c**.
Tests cover every per-step R/U/Q gradient, both sites/protocols, fixed-batch
loss against a direct `UnsharedAdapter` computation, exact checkpoint prediction
round-trips, frozen-writer compatibility, every WP-L3 arm and normalization
reload. They use a small Qwen3 architecture with the pinned tokenizer; existing
M1/WP-L3 regressions run alongside the new tests.

Pinned Qwen3-0.6B training smoke **3715157**, `mg0040`, passed both variants at
batches 32/128 with `scan_final` and two prefetch workers. The final comparison
in **3715193** used one process, r=32, layer14/o_proj, memory-only joint training,
E16/Q4 T2 train-only batches, fact_final writes and AdamW. Five warmups were
discarded before 20 measured production steps per cell:

| Batch | M1 mean ms/step | B8 mean ms/step |
| --- | --- | --- |
| 32 | 273.55 | 263.52 |
| 128 | 357.16 | 358.73 |

These are descriptive systems measurements with similar step cost, not a
quality comparison. Every cell logged only L=4, called `scan_final` 25 times
and `scan_prewrite` zero times, and both B8 cells had finite nonzero gradients
on all 12 R/U/Q matrices. [Full timing and gradient evidence](../../reports/llm-b8/benchmark.json),
[tables](../../reports/llm-b8/tables.md) and [exact commands](../../reports/llm-b8/README.md)
are committed; raw checkpoints/logs remain under `LoopLora/runs/llm-b8/`.

All three jobs exited 0 through submit.py on compute. [Total cost](../../reports/llm-b8/jobs.json):
**270 s = 0.075/0.3 node·h**. No new dependencies, scientific runs, confirmation
data, merge conflicts or unresolved checks.
