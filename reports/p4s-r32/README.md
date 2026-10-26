# WP-C4 — released r32 stopping pilot

Evaluation only, screening seed 4242. The [design](design.json) was committed at
ce77e74 before evaluation. Driver, analysis and tests use 613376b, after merging
master 28d3668 (which includes b8e4584). Reader traces, rules, prediction convention,
threshold grids and validation-only family selection are reused from WP-C3.
The two shared helpers now allow certificate-free collection/loading; WP-C3's
default certificate contract is preserved. No latency or certificate computation
was performed, and no confirmation episodes were opened.

- [Tables](tables.md): all six families, fixed L=1/4/8/16, three stopping rules and
  true-hop reference; per-hop accuracy, mean steps/reads and G1-style deltas.
- [Accuracy](accuracy.csv) and [per-seed accuracy](accuracy_per_seed.csv).
- [Screening endpoints](endpoints.csv) and [per-seed endpoints](endpoints_per_seed.csv).
- [Selected thresholds](thresholds.csv), [full validation grid](threshold_grid.csv)
  and [frozen selection](selection.json).
- [Manifest](manifest.json), [audit](audit.json), [released-reference comparison](reference_check.csv)
  and [job ledger](jobs.json).

All deltas are stopped-minus-L1 accuracy; reach averages hops 2/3, far averages
6/7/8, and walk specificity is reach minus far. Families average the three training
seeds equally. A screening candidate has hop-1 delta ≥−1 pp, reach >0 and walk
specificity >0. This is the G1 preservation interpretation: improvement on hop 1
is allowed. These flags are point estimates, not formal gate acceptance. WP-C3
did not implement bootstrap intervals; none were added here. Selecting a candidate
after screening requires a fresh confirmation seed before making a formal claim.

All commands below must run via submit.py. Use explicit commits: its `--commit`
argument resolves references in the main checkout, so `HEAD` can select another
branch. The first attempt did so and stopped before running tests; its cost remains
in the ledger. The successful test+smoke job was 3714219 (`mg0013`): **77 passed**,
then all six families passed the 200-episode validation-only smoke. It ran:

```sh
scripts/submit.py --name t-c4-smoke --queue debug-g --walltime 00:05:00 \
  --commit 613376b --wait --tail 100 -- -c 'import subprocess,sys; subprocess.run([sys.executable,"-m","pytest","-q","-o","addopts=","tests/test_p4s_r32.py","tests/test_p4s_analysis.py","tests/test_stopping.py"],check=True); subprocess.run([sys.executable,"scripts/run_p4s_r32.py","--smoke","--out","/work/xg24i002/x10041/LoopLora/runs/p4s-r32/smoke-v1"],check=True)'
```

Full evaluation followed by audit/export (3714230):

```sh
scripts/submit.py --name eval-c4 --queue debug-g --walltime 00:03:00 \
  --commit 613376b --wait --tail 100 -- -c 'import subprocess,sys; subprocess.run([sys.executable,"scripts/run_p4s_r32.py","--out","/work/xg24i002/x10041/LoopLora/runs/p4s-r32/main-v1"],check=True); subprocess.run([sys.executable,"scripts/analyze_p4s_r32.py","--root","/work/xg24i002/x10041/LoopLora/runs/p4s-r32/main-v1","--out","/work/xg24i002/x10041/LoopLora-wt/core/reports/p4s-r32"],check=True)'
```

Use new output paths for reruns. Raw traces, timing records and a duplicate of
the summaries remain at `/work/xg24i002/x10041/LoopLora/runs/p4s-r32/main-v1/`.
Compact report files stay below 1 MB in Git; model files and per-query traces
are excluded. No training or dependency change.
