# WP-B4 prefetch validation and benchmark

Implementation contract (2026-10-27): `looplora.llm.prefetch.BatchStream(spec,
split, batch_size, seed, steps, *, tokenizer, workers=0, start_step=0)` is a
context-managed iterator yielding `(EpisodeArrays, TokenBatch | None)` for
absolute steps `[start_step, steps)`. With workers=0 it generates synchronously
and returns None tokens, retaining the original GPU tokenization path. With
workers>0, spawn processes produce CPU token tensors; at most `2 * workers`
tasks are outstanding. Workers receive the caller's tokenizer and cache the
pinned name pool. Errors propagate and context exit joins workers.

`LLMExperiment.prepare(arrays, ..., tokens=None)` accepts the optional CPU
TokenBatch and transfers it to the experiment device before the unchanged GPU
support encoding. Evaluation is unchanged. `train.prefetch_workers` defaults to
0 and accepts nonnegative integers.

Benchmark follows WP-C5: pinned Qwen3-0.6B-Base, T1 E16/Q4, rank32,
layer14/o_proj, BF16 backbone / FP32 adapter, both write policies, AdamW,
seed17, five warmups and twenty measured steps. Batch32/64/128 is per training
process; one/two processes share one GPU. Sweep workers0/2/4/8. Record startup
separately and use barriers for achieved aggregate episodes/s. No evaluation
data, model checkpoints, or scientific hyperparameter selection.

## Validation and reproduction

Merged master `9721987` before implementation. Production/tests are commit
`e55f134`; the benchmark is `65c2ccb`. All application imports and execution
ran on Miyabi-G compute through `scripts/submit.py`; no dependencies changed.

```sh
scripts/submit.py --name t-wpb4 --queue debug-g --walltime 00:04:00 \
  --commit e55f134 --wait -- -m pytest -q -o addopts= \
  tests/test_llm_prefetch.py tests/test_llm_train.py tests/test_llm_residual.py tests/test_llm_run.py

scripts/submit.py --name smoke-wpb4 --queue debug-g --walltime 00:05:00 \
  --commit 65c2ccb --wait -- scripts/benchmark_llm_prefetch.py \
  --smoke --max-seconds 240 --out /work/xg24i002/x10041/LoopLora/runs/llm-prefetch/smoke-v1

scripts/submit.py --name bench-wpb4 --queue debug-g --walltime 00:24:00 \
  --commit 65c2ccb --wait -- scripts/benchmark_llm_prefetch.py \
  --max-seconds 1400 --out /work/xg24i002/x10041/LoopLora/runs/llm-prefetch/main-v1
```

Use new output directories for reruns. Tests: **95 passed in 181.57 s**, PBS
**3714692**, `mg0028`. This includes exact arrays and every token tensor at 50
steps for T1/T2 and both write policies, absolute resumed offsets, zero-worker
and empty streams, clean early shutdown, consumer/serialization/worker errors,
abrupt worker death, both GPU support protocols, and complete two-step optimizer
results compared exactly. Evaluation batches are not routed through prefetch.

Smoke: PBS **3714701**, `mg0021`, exit 0, eight cells. Five warmups and three
measurements ensure eight workers actually start. Conservative full-sweep
estimate: 1199.6 s, assuming all nonzero pools cost as much as eight workers
and all batches as much as batch128. The 25% margin exceeds the remaining
cumulative allowance, so the full sweep has a 1400 s controller cap and 1440 s
PBS cap. Tests plus smoke consumed 332 node-seconds; even the full PBS request
keeps the task at 1772/1800 node-seconds. No additional runtime is planned.

The benchmark validates exact cell membership, process completion, finite
timings and all loss trajectories against workers0 (rtol1e-5, atol0, as WP-C5).
It writes raw samples, loss errors, startup/close costs, GPU peak allocations,
CSV and Markdown tables.

## Results

Full job **3714713**, `mg0005`, exit 0: all 48 cells complete at source
`65c2ccb` / blake2b-128 `3c492b0ea04cd8cdfcea19fe37ae58c7`. All 1440 measured
worker steps and 360 warmup steps completed. Every 25-step loss trajectory is
exactly equal to its workers0 reference (maximum absolute and relative error 0).
The benchmark validates the requested matrix and all worker completion records
before writing status=complete. The smoke also passed; its samples are retained
under [smoke/](smoke/benchmark.json).

- [Full timing table](tables.md), [CSV](timings.csv), [metadata](benchmark.json).
- Per-worker samples: [one process](worker-1-0.json),
  [two processes / worker0](worker-2-0.json),
  [two processes / worker1](worker-2-1.json).
- [Job ledger](jobs.json): tests 191 s + smoke 141 s + full 1237 s = **1569 s,
  0.43583 node·h**, below the 0.5 node·h cap. Queue wait is excluded; PBS
  initialization and cleanup are included. No failed jobs or extra probes.

**Recommend two prefetch workers per training process**, particularly at
batch128 for long runs. This improves throughput 1.17–2.29x across all 12
batch/process/policy settings and comes within 3% of each setting's best observed
throughput. At batch128, fact_final with one process rises from 379.48 to 673.09
episodes/s; two processes rise from 305.31 to 698.08. all_tokens rises from
280.63 to 419.87 and 267.70 to 545.47, respectively. The highest observed
fact_final rate is 703.15 episodes/s with two processes/four workers; the small
gain over two workers costs more startup. GPU peak allocation is unchanged.

Startup matters: two workers take 12–14 s to the first rendered batch in the
full sweep, versus about 25 s for four and 31 s for eight. Close costs 0.4–0.6 s.
For batch128, the two-worker startup/close overhead amortizes after roughly
85 steps with one process, 30 with two (using measured per-step savings).
Short smokes should keep the default0. These are throughput benchmarks, not
retrieval-quality experiments, and small timing differences are not statistical
evidence of a better scientific configuration.
