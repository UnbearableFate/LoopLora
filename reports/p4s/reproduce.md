# WP-C3 reproduction and artifact locations

All runtime commands run through scripts/submit.py on Miyabi-G. The shared Python environment is unchanged. The evaluation source is fc32c7a; the audit source is f743a59. Code and results are on wp/core. Existing F1 checkpoints were only read and their blake2b seals were rechecked.

Tests (126 passed at source 110c7d5 in job 3713584; the first job 3713562 had 125 before the additional latency-hop regression):

```sh
scripts/submit.py --name t-p4s --queue regular-g --walltime 00:05:00 \
  --commit 110c7d5 --wait -- -m pytest -q -o addopts= \
  tests/test_stopping.py tests/test_p4s_analysis.py \
  tests/test_contraction.py tests/test_loop_invariants.py
```

The successful timing smoke used seed 17, all eight families and 200 validation episodes each. It ran in 48.39 application seconds (3713639). It printed only timing/status, and did not select thresholds or evaluate screening data.

```sh
scripts/submit.py --name smoke-p4s --queue regular-g --walltime 00:05:00 \
  --commit fc32c7a --wait -- scripts/run_p4s_stopping.py --smoke \
  --out /work/xg24i002/x10041/LoopLora/runs/p4s/smoke-v2
```

Full evaluation (3713666, exit 0):

```sh
scripts/submit.py --name eval-p4s --queue regular-g --walltime 00:15:00 \
  --commit fc32c7a --wait --tail 100 -- scripts/run_p4s_stopping.py \
  --out /work/xg24i002/x10041/LoopLora/runs/p4s/main-v1
```

Use a new empty output path when rerunning either command. The full run contains manifest.json, thresholds.json (including the pre-screening lock timestamp), traces/<checkpoint>/{val,test}.npz, latency.json, timing.json and summaries/. Per-seed summaries and all per-query traces remain there rather than in Git. The committed family tables in this directory are small exports.

Final audit/export (3714004, exit 0):

```sh
scripts/submit.py --name audit-p4s --queue regular-g --walltime 00:03:00 \
  --commit f743a59 --wait -- scripts/audit_p4s_stopping.py \
  --root /work/xg24i002/x10041/LoopLora/runs/p4s/main-v1 \
  --out /work/xg24i002/x10041/LoopLora-wt/core/reports/p4s \
  --reference-csv reports/p1s-f1/accuracy_tables.csv
```

The first GPU smoke exposed an overly component-sensitive state comparison at L16. A separate compute probe (3713620) found identical predictions/halts, FP32 vector drift 8.02e-6 and FP64 drift 1.42e-14. The audit now measures vector error (absolute 1e-6 + relative 1e-4), records its size, and retains exact predictions/halts/read-counter checks. Across the final latency matrix, maximum relative L2 drift was 1.543e-6. Neither thresholds, model weights nor scoring changed to resolve the check. The failed smoke and cancelled queued diagnostic remain in jobs.json.

Thresholds were locked after all validation traces and before the first test trace. The audit checked exact integer-equivalent tie-breaking across 48000 validation queries per family: six tied groups, zero mismatches. No confirmation episodes or r32 checkpoints were opened. True-hop stopping is retained as a label-informed control; it is not an accuracy upper bound for these learned update maps.
