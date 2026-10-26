# WP-C6 — LLM B8 and fixed-depth training

Merged master 45f6d14 and wp/llm c5879d3 in merge commit 03087ea, without conflicts.
Implementation and final validation use **214518c**. No core reader numerics,
dependencies, model revision or existing variant defaults changed.

`variant: B8` installs `UnsharedAdapter` with four per-step R/U/Q triplets.
Configuration, support passes, training and evaluation use L=4 in writer,
joint and reader stages. Memory-only and retained-context evaluation, o_proj
and residual inputs, ordinary writer exports and self-contained checkpoints
are supported. Writer signatures intentionally exclude reader sharing.
`train.depths: [4]` also fixes M1/B5/B3 reader/joint training.

WP-L3 construction is `configuration(arm, variant="B8")` followed by
`WriterFixExperiment(config, arm)`. All OA/W0/W1/W2/W3/W4/W5 interventions keep
their prior behavior and truncated memory-only scope. A shared mixin applies
them to either the original iterative reader or the unshared reader. The
addressing auxiliary supervises the first read, normalize(Q[0] z⁰); OA uses
each step's R/U with the fact's raw stored value. Standardization buffers,
paired-position A_k/A_v writes, training-only value heads, prefetch and inference
exports retain their existing behavior. The original default configuration and
registered WP-L3 scientific runs remain M1/L1.

- [Full benchmark](tables.md), [timings CSV](timings.csv), [metadata and gradient evidence](benchmark.json).
- [Smoke](smoke/tables.md) and its [metadata](smoke/benchmark.json).
- [Job costs](jobs.json). All runtime executed on compute via submit.py.

Tests use a small Qwen3 architecture with the pinned tokenizer, both protocols
and input sites. They verify every per-step reader gradient, a fixed-batch loss
against a direct UnsharedAdapter computation, B8-writer consumption by both M1
and B8 readers, exact checkpoint prediction round-trips, explicit M1/B5/B3 L4
training, every WP-L3 arm at both sites, standardization/inference reload, and
L4-only evaluation. Existing M1 and WP-L3 regressions run alongside them.

The pinned 0.6B model smoke and benchmark use ordinary joint memory-only
training, E16/Q4 T2 train-only data, rank32, layer14/o_proj, fact_final,
AdamW 3e-4, one process and two prefetch workers. All logged depths must equal
4; counters require one scan_final per support pass and zero scan_prewrite
fallbacks. B8 must have finite nonzero final gradients on all 12 R/U/Q matrices.
These short systems runs make no retrieval-quality or statistical claims.

Exact final validation command (new output paths are required on rerun):

```sh
scripts/submit.py --name check-bench-c6 --queue debug-g --walltime 00:04:00 \
  --commit 214518c --wait --tail 100 -- -c 'import subprocess,sys; subprocess.run([sys.executable,"-m","pytest","-q","-o","addopts=","tests/test_llm_b8.py","tests/test_llm_train.py","tests/test_llm_run.py","tests/test_llm_writer_fix.py","tests/test_llm_residual.py"],check=True); subprocess.run([sys.executable,"scripts/benchmark_llm_b8.py","--out","/work/xg24i002/x10041/LoopLora/runs/llm-b8/main-v1","--summary-out","/work/xg24i002/x10041/LoopLora-wt/core/reports/llm-b8"],check=True)'
```

The preceding smoke used 46746c3, the same reader implementation before the
additional configuration helper, with `--smoke` and output `runs/llm-b8/smoke-v1`.
It ran three training steps per cell (one discarded, two measured), at both
batches and for both variants. The full comparison discards five warmups and
measures 20 steps per cell. Steady ms/step includes prefetch waits and the
complete synchronized training step; model/worker startup and checkpoint export
are recorded in cell time. Peak allocation includes warmup.

Raw checkpoints, per-step logs and provenance remain under
`/work/xg24i002/x10041/LoopLora/runs/llm-b8/`; compact summaries alone are committed.
No screening or confirmation data are used, and no formal P1-L trial is started.
