# WP-B4 — ordered CPU batch prefetch (2026-10-27)

Merged master `9721987` into `wp/data`. Added `BatchStream` in
`src/looplora/llm/prefetch.py`: spawn workers generate and render CPU token
tensors in advance, with ordered consumption and at most `2 * workers` pending
batches. The main process transfers tensors and uses the existing GPU support
encoding and optimizer. `train.prefetch_workers=0` preserves the default path;
nonnegative integers enable the pool. A context manager joins workers on normal
completion, early exit and errors. Evaluation is unchanged. No dependencies,
generated data, tokenization or numerics changed.

**95 tests passed** in 181.57 s, PBS **3714692**, `mg0028`: exact arrays and all
CPU token tensors over 50 steps for T1/T2 and both policies; resumed offsets;
startup, shutdown and error handling (including abrupt worker death); exact GPU
preparation for both support protocols; exact two-step training logs and adapter
parameters. Command: `python -m pytest -q -o addopts= tests/test_llm_prefetch.py
tests/test_llm_train.py tests/test_llm_residual.py tests/test_llm_run.py`, submitted
through `scripts/submit.py` at commit `e55f134`.

Full benchmark **3714713**, `mg0005`, exit 0: all **48 cells** (batch32/64/128,
one/two training processes per GPU, workers0/2/4/8, both policies), pinned
Qwen3-0.6B-Base, T1 E16/Q4, rank32/layer14/o_proj, BF16 backbone/FP32 adapter,
AdamW, five warmups and twenty measurements. All **1440 measured worker steps**
completed; every 25-step loss trajectory matched workers0 **exactly** (maximum
error 0). Batch128 results below compare workers **0 → 2**; throughput includes
barriers and excludes startup.

| Policy | Processes | ms/step | episodes/s |
| --- | --- | --- | --- |
| fact_final | 1 | 337.31 → 190.17 | 379.48 → 673.09 |
| fact_final | 2 | 838.48 → 366.72 | 305.31 → 698.08 |
| all_tokens | 1 | 456.11 → 304.85 | 280.63 → 419.87 |
| all_tokens | 2 | 956.30 → 469.32 | 267.70 → 545.47 |

**Recommend `train.prefetch_workers: 2` for sustained training.** Across all
settings it improves throughput 1.17–2.29× and stays within 3% of the best
observed setting. Startup is 12–14 s, versus approximately 25/31 s with 4/8
workers; close costs 0.4–0.6 s. Keep default0 for short smoke runs. At batch128,
two workers amortize startup in roughly 85 steps for one process, 30 for two.

The eight-cell smoke **3714701**, `mg0021`, also passed. [Full tables and raw
samples](../../reports/llm-prefetch/README.md) preserve reproduction commands,
startup costs and provenance (production `e55f134`, benchmark `65c2ccb`). Total
PBS allocation: **1569 s = 0.43583/0.5 node·h**, including tests and smoke.
No deviations or open checks; no application imports or tests ran on login.
