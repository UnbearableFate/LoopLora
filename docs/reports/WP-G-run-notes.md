# WP-G runner implementation notes

Owned paths: `llm/config.py`, `llm/run.py`, `configs/p1l/`, `tests/test_llm_run.py`.
No commits, submissions, training, evaluation, pilot or screening execution in this subtask.

- Configuration resolves the binding WP-G schema and TaskSpec. Formal defaults: 1000 steps,
  batch 8, E24/Q4, rank32/layer14, 2000 evaluation episodes. Smoke YAML uses 100 steps,
  200 validation episodes, seed20261027 and eval seed `derive_seed(20261027,"p1l-smoke-eval")`.
  It rejects non-val smoke splits, smoke confirmation permission, unapproved confirmation,
  counts >2000, and arbitrary start/episode-start fields. Non-smoke confirmation is only
  the evaluator's named `confirm` interval; ordinary split ranges cannot reach it.
- `smoke.yaml` resolves 14 runs: writer, five frozen readers sharing its `writer.pt`,
  five joint variants, context-trained B1 LoRA and B2 parallel nonlinear bottleneck,
  and one B0 run evaluating both protocols. B2 follows Addendum 1 with the same rank,
  position, data, optimizer and step count as B1; its config is `smoke-parallel.yaml`.
  Writer/B4/B1/B2/B0 evaluate only L1; other readers evaluate all seven registered depths.
  Writer trains/evaluates hop1. M1 frozen/joint and B0/B1/B2 include context-retained results.
  `fact_final` is a smoke configuration choice, without a scientific policy selection.
- `run_one` uses the symbolic config/record envelope and imports symbolic source metadata
  directly after WP-D is merged. The evaluator computes depth statistics once; the runner
  preserves that result without repeating the bootstrap. It adds model/tokenizer revisions,
  stage, protocols, adapter checkpoint, timing and failure records, closing adapters on exit.
  Record data_seed matches train.py's `llm/writer-data` or `llm/train-data` derivation.
  The completed record uses the actual training summary's data_seed. Reader records retain
  `consumed_writer_checkpoint` and `writer_source_record`; `pretrain_tokens/seconds=0`
  means no writer cost is allocated again to each reader, as stated explicitly in
  `pretrain_cost_accounting`. The source writer record carries its own training cost.
  Parameter accounting reports model/backbone/adapter/external-writer components and an
  active total. For B3 it adds external K/V/gates, counts shared A once, and excludes the
  unused external writer object's R/U/Q/B. `external_writer` is the additional unique count.
- Sequential suites load model/tokenizer once; `--procs 2` uses two persistent spawned
  workers on the first visible GPU. Each worker loads once and closes adapters between runs.
  Dependency failures skip consumers. All outputs reject reuse of nonempty directories.
  Per-run UTC time/name/start-complete-fail messages are flushed to the job log without scores.
  Duplicate variants in an existing comparison group (including different-LR runs) fail during
  suite loading before model allocation, and also fail in direct comparison calls, preventing
  silent replacement of one run's results by another.
- `--measure-scheduling` runs only two untrained B0 configurations per mode, 32 validation
  episodes, L1, counterfactuals disabled, comparing sequential and two-process execution.
  Evidence includes process and model loading/shutdown walltime, all run artifacts, failed
  mode costs, and node-hours for one allocated node (overlapping workers aren't summed).
  `--suite ... --procs auto` selects the faster successful system mode, retaining evidence
  under `<out>/scheduling` and the 14-run suite under `<out>/runs`. This is system smoke,
  never research selection. A fixed `--procs 1|2 --measure-scheduling` retains that mode.
- Suite comparisons filter only the declared memory-only protocol, then call
  `sym.stats.compare_runs` without intersecting missing episode/query labels. Frozen-reader
  Gate0 compares exact batch x/u/M/write_count hashes for the same checkpoint, task, seed,
  model/write policy and eval sets; joint variants do not claim a common writer. B4 pairs
  at L1 against M1 L4; B3/B5/M1-lin pair at L4. All conditions retain protocol labels.

Validation: `/usr/bin/python3` AST syntax parsing passed for the three owned Python files;
`git diff --check` passed for tracked changes. Runtime tests are deliberately unexecuted:
the current shell is login/control (`miyabi-g2`, no PBS allocation), and the parent requested
implementation only with no submissions yet. After WP-D merge, run on compute through
`scripts/submit.py --name t-wpg-run --walltime 00:10:00 --worktree . --wait -- -m pytest -q tests/test_llm_run.py`.
No measured scheduling preference or smoke timing is claimed before that execution.
