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

Validation and measured results will be appended after compute jobs.
