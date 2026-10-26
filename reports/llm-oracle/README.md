# WP-C7 oracle writer acceptance evidence

Source: `b54ec3a21cc4037910396663f3c9b717bcaefbc9` on `wp/core`, after merging
master `2384eca` and wp/llm `161e694`. No scientific training/evaluation or
quality selection is authorized by these measurements.

The writer uses fp32 global random codes: keys index `(entity, relation)` pairs,
values are `OracleWriter.global_codes`; fixed oracle seed 17. Every valid fact
writes once in list order, λ = β = 1. There is no support LM forward and no
support text tokenization. The learned `scan_final` path is bypassed; its
existing behavior is covered by the regression suite. Reader inputs, updates
and LM output path are unchanged. Both code tables are checkpointed buffers;
question-side A stays trainable and unused K/V/gates are frozen.

All commands ran through `scripts/submit.py` on Miyabi-G compute. One node per
job; aggregate allocated time and the initial failed check are in `jobs.json`.

```bash
# Initial development run: one accounting failure, 187 passes.
scripts/submit.py --name t-c7 --queue debug-g --walltime 00:05:00 --worktree . --wait -- \
  -m pytest -q -o addopts= tests/test_llm_oracle.py tests/test_llm_train.py \
  tests/test_llm_run.py tests/test_llm_prefetch.py tests/test_llm_render.py \
  tests/test_llm_b8.py tests/test_llm_read_fix.py

# Corrected accounting plus all final oracle and run-interface checks.
scripts/submit.py --name t-c7-final --queue debug-g --walltime 00:02:00 --commit b54ec3a --wait -- \
  -m pytest -q -o addopts= tests/test_llm_oracle.py tests/test_llm_run.py

# Ten-step preflight, then the required 100-step acceptance smokes.
scripts/submit.py --name smoke-c7-trial --queue debug-g --walltime 00:02:00 --commit b54ec3a --wait -- \
  scripts/benchmark_llm_oracle.py --trial \
  --out /work/xg24i002/x10041/LoopLora/runs/llm-oracle/trial-v1 \
  --summary-out /work/xg24i002/x10041/LoopLora-wt/core/reports/llm-oracle/trial

scripts/submit.py --name smoke-c7-100 --queue debug-g --walltime 00:03:30 --commit b54ec3a --wait -- \
  scripts/benchmark_llm_oracle.py \
  --out /work/xg24i002/x10041/LoopLora/runs/llm-oracle/acceptance-v1 \
  --summary-out /work/xg24i002/x10041/LoopLora-wt/core/reports/llm-oracle
```

`benchmark.json` and `timings.csv` retain individual measured step durations,
gradient norms, code hashes, source/runtime metadata and cell startup costs.
`trial/` holds the preflight at the same batch shape. The 100-step smoke reuses
the pinned Qwen3-0.6B-Base backbone sequentially for M1-L4, B8@4 and B4@1 with
batch 64, one GPU process and two spawned CPU preparation workers. All batches
come from the train split at a derived acceptance seed; no EM is evaluated.
Timing includes batch wait, oracle writes, question forward/backward, clipping
and AdamW, with five warmup steps discarded from steady-state statistics.

Raw checkpoints and logs remain under
`/work/xg24i002/x10041/LoopLora/runs/llm-oracle/`; compact summaries alone are
committed here. Checkpoints are not needed to reproduce the reported timings
and are not scientific reader checkpoints for downstream evaluation.
