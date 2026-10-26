# WP-C7 — fact-only oracle writer in the LLM pipeline (2026-10-27)

**`writer: oracle_random` is implemented and validated.** Merged master
**2384eca**, then wp/llm **161e694**, into `wp/core` without conflicts.
Implementation and tests: **b54ec3a**. Usage is documented in
`configs/p1l/README.md`: memory-only, r = r_mem = 32, `stage: reader`, no
writer checkpoint required; `oracle_seed: 17` stays fixed across reader seeds.

Keys are fixed global random unit codes for **(entity, relation)** pairs,
indexed `entity * n_rel + relation`, so same-head relations have distinct
addresses. Values are exactly symbolic `OracleWriter.global_codes` at the
same seed. Both tables are fp32 buffers, never optimized, saved in the full
checkpoint and writer export. Facts write in list order using the unchanged
`OracleWriter` scan, λ = β = 1, with invalid slots skipped. No support text is
rendered or forwarded through the LM. The learned `scan_final` path is
bypassed because oracle writes have no LM features; its numerics are unchanged.
Two-worker prefetch prepares question tokens only. The existing readers and LM
output path are unchanged; question-side A trains, unused K/V/gates are frozen.
Cross-reader comparisons check M/write counts; each reader's depth comparisons
still check x/u/M/counts. Records include a blake2b code hash and writer signature.

**Compute tests:** initial job **3715661**, `mg0029`, passed 187 checks and
found one parameter-accounting bug (`OracleWriter.V` is a vocabulary integer).
After fixing it, **all 64 oracle/run-interface tests passed in 37.94 s**, job
**3715693**, `mg0036`, on b54ec3a. Coverage includes exact symbolic-state
equivalence with pair-address lifting (CPU/CUDA, padding, repeated updates),
no code gradients, all M1/M1-lin/B8/B4/B5/B3 readers at both input sites,
exact checkpoint predictions/buffer reload, malformed-checkpoint rejection,
depth invariance, and exact synchronous/prefetch training equivalence.
The initial run also passed existing training, rendering, prefetch, B8 and
WP-L4 regressions. [Commands and evidence](../../reports/llm-oracle/README.md).

**Pinned Qwen3-0.6B acceptance smoke:** after a 10-step timing preflight
(**3715702**, `mg0037`), job **3715715**, `mg0036`, completed **100 steps each**
for M1-L4, B8@4 and B4@1. Batch 64, one GPU process, two CPU prefetch workers,
layer14/o_proj, r32, T2 E12/Q4, chains (8,4), four relations, head sharing 0.3;
train-only development seed, full-vocabulary CE. CUDA-synchronized production
steps include batch wait/write/forward/backward/optimizer; five warmups excluded:

| Reader | L | Mean ms/step | Median ms/step |
| --- | --- | --- | --- |
| M1-L4 | 4 | 189.29 | 189.23 |
| B8 | 4 | 188.48 | 188.75 |
| B4 | 1 | 131.98 | 131.98 |

All cells retained identical code hashes, made zero support LM calls, and had
finite nonzero gradients on every active reader parameter (all 12 B8 R/U/Q
matrices). [Timings and gradients](../../reports/llm-oracle/benchmark.json)
are descriptive systems results, with no quality claim. Raw checkpoints/logs
remain under `LoopLora/runs/llm-oracle/`; compact summaries are committed.
[Total allocation](../../reports/llm-oracle/jobs.json), including the failed
check: **466 s = 0.1294/0.3 node·h**, one node/job, longest request five minutes.
No scientific runs, confirmation data, dependency changes or unresolved checks.
