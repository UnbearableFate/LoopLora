# WP-C5 final-only support writes

Merged master 4bc1fcc and WP-L2 branch snapshot 382422f into wp/core (ca56c3b).
The final core implementation and full benchmark use **e60169f**; the compact
artifact audit uses **a8371ac**. No dependencies, dtypes, reader semantics or
training budgets were changed. No torch.compile.

`memory.scan_final` packs valid write positions per row, preserves their order,
gathers input-dependent gates, and applies masked dense updates without per-step
nonzero/index_select or host synchronization. It stores no explicit pre-write
snapshot stack. Token-zero resets run on the fast path; any interior reset uses
the unchanged reference scan. Autograd still retains the intermediates needed
for writer gradients. Both o_proj-input and residual-input write-only wrappers
use the new path; online mode keeps scan_prewrite.

- [Full tables](tables.md): all 24 method/settings cells, ms/step, achieved
  episodes/s, summed worker peak memory, separate phase profiles and real-model
  fixed-batch loss comparisons.
- [Timings CSV](timings.csv), [run metadata](benchmark.json), [audit](audit.json).
- Per-worker samples: [one process](worker-1-0.json), [two processes, worker 0](worker-2-0.json),
  [worker 1](worker-2-1.json). Each method starts from identical adapter weights,
  optimizer and episode order. Alternate method order across settings.
- [Job ledger](jobs.json) includes all tests, probes and failed smoke/diagnostic.

The first version projected all packed writes in one matrix multiply. Core
state/gradient tests passed, but a nonzero-B fixed-batch residual/all_tokens
Qwen3-0.6B loss failed the required relative 1e-5 tolerance. Probes reproduced
reference and fast results independently: FP32 bulk-versus-per-token projection
error reached 4.96e-5, with state relative drift 3.44e-7, enough to affect the
BF16 backbone ([initial-version probe](initial-projection-probe.json)). The final code retains per-token contiguous projections and
passes the original tolerances. No assertions were relaxed. A diagnostic-only
gather initially omitted clamping padded positions and failed; it was corrected
without changing production code. Raw failed runs remain preserved.

Final tests and smoke, PBS 3714480 (`mg0002`): **150 passed in 22.55 s**, followed
by successful pinned-model loss equivalence for both input sites and write
policies, plus batch-32 timing smoke. All commands ran on compute via submit.py:

```sh
scripts/submit.py --name t-smoke-c5 --queue debug-g --walltime 00:05:00 \
  --commit e60169f --wait --tail 60 -- -c 'import subprocess,sys; subprocess.run([sys.executable,"-m","pytest","-q","-o","addopts=","tests/test_scan_final.py","tests/test_state_lifecycle.py","tests/test_loop_invariants.py","tests/test_gradient_flow.py","tests/test_llm_residual.py","tests/test_llm_support.py"],check=True); subprocess.run([sys.executable,"scripts/benchmark_fast_write.py","--smoke","--out","/work/xg24i002/x10041/LoopLora/runs/fast-write/smoke-v2"],check=True)'

scripts/submit.py --name bench-c5 --queue debug-g --walltime 00:10:00 \
  --commit e60169f --wait --tail 80 -- scripts/benchmark_fast_write.py \
  --out /work/xg24i002/x10041/LoopLora/runs/fast-write/main-v1

scripts/submit.py --name report-c5 --queue debug-g --walltime 00:03:00 \
  --commit a8371ac --wait -- scripts/report_fast_write.py \
  --root /work/xg24i002/x10041/LoopLora/runs/fast-write/main-v1 \
  --out /work/xg24i002/x10041/LoopLora-wt/core/reports/fast-write
```

Use new output directories for reruns. Raw artifacts remain under
`/work/xg24i002/x10041/LoopLora/runs/fast-write/`; logs and source specs are under
`/work/xg24i002/x10041/LoopLora/logs/pbs/`. These are short throughput benchmarks,
not retrieval-quality experiments. Profile steps synchronize phase boundaries
separately and are not used for headline speedups. Question backward includes
writer backward. Summed per-worker peaks are not simultaneous GPU peak readings.
