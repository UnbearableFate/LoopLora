# P1-L implementation and smoke

The formal design remains draft. These defaults are implementation choices for
the authorized WP-G smoke, not selected hyperparameters: layer 14, rank 32,
E=24, three chains of length 8, four training queries, batch 8, AdamW lr=3e-4,
1000 default steps and 100 smoke steps. `fact_final` is a smoke setting; no
comparison with `all_tokens` or scientific policy selection occurs here.

`smoke.yaml` runs 14 configurations with seed 20261027: one T1 writer, five
readers sharing its frozen checkpoint, five joint models, B1 LoRA and B2 parallel
bottlenecks trained with retained context, and untrained B0. Every trained stage
uses 10% of default steps. All smoke evaluation uses 200 validation episodes and
a separate derived smoke seed. It never touches screening or confirmation data.
M1, B0, B1 and B2 additionally evaluate with retained context. Writer, B4 and
the depth-independent baselines evaluate L=1; other readers use 0/1/2/3/4/6/8.

Submit only through `scripts/submit.py` on Miyabi-G. For example, after committing
the source and sizing walltime from measured evidence:

```sh
scripts/submit.py --name s-p1l --queue regular-g --walltime 00:40:00 \
  --commit wp/llm --wait -- scripts/smoke_llm.py \
  --suite configs/p1l/smoke.yaml --procs auto \
  --runs /work/xg24i002/x10041/LoopLora/runs/p1l/UNIQUE-RUN \
  --reports /work/xg24i002/x10041/LoopLora-wt/llm/reports
```

`--procs auto` first measures two untrained B0 runs (32 validation episodes,
L=1, no counterfactuals) with one versus two persistent model processes sharing
one GPU. Startup and model loads count in the choice; it is a bounded scheduling
measurement, not an accuracy comparison. Actual runs live under `runs/` within
the specified output, with scheduling evidence under `scheduling/`. This small
system benchmark does not establish the fastest mode for every future training
shape. Explicit `--procs 1`/`2` are also supported. Output directories must be new.

Each run writes `config.json`, `train_log.jsonl`, `metrics.json`, `samples.npz`,
adapter-only `checkpoint.pt`, `timing.json`, and `record.json`. Memory models also
write `writer.pt`; frozen readers embed their writer in the checkpoint for
self-contained reload. Writer pretraining costs belong to its source record,
without charging the same shared writer again to every reader.

Primary `correct` means full-vocabulary first-token exact match. Candidate
accuracy is secondary; its chance reference is inverse episode candidate count.
Flat samples retain WP-D's required fields for `looplora.sym.stats`; `condition`
includes protocol (`memory_only/original`, etc.), preventing cross-protocol
pooling. Counterfactual replacements preserve source episode/query labels;
swaps replace M but preserve source questions, labels and candidates (and source
KV in retained-context mode). Cross-variant comparisons explicitly use only the
common memory-only protocol. Gate 0 hashes x/u/M/write count across depth and
across readers with the same frozen writer. Model loading enables deterministic
Torch algorithms, a fixed cuBLAS workspace (`:4096:8`), and deterministic cuDNN;
these settings are recorded in every run. Unsupported nondeterministic operations
fail explicitly. Default GPU settings produced one process-dependent support
state in the initial smoke, so that suite was rejected rather than relaxing hashes.

`scripts/smoke_llm.py --audit-only` verifies artifacts, full-vocabulary labels,
sample-to-metric reproduction, unchanged frozen checkpoints, Gate 0 and paired
comparisons, then writes `reports/p1l_smoke_audit.json` and timing CSV. Run it on
compute too. Per-evaluation-set timings include configured depths and both
counterfactual controls; they are not single-depth latency measurements.

Future ordinary validation/test sets start at episode 0, up to 2000 episodes;
confirmation maps to test [100000,102000) and requires explicit
`allow_confirmation=true` with `smoke=false`, matching the current P1-L task.
Those options provide implementation support, not authorization to run them.

WP-L2 adds `model.input_site: residual` as a diagnostic alternative to the
default `o_proj`. It projects the pre-layer residual with A, while the frozen
attention projection and adapter output remain at that layer's o_proj. Adapter
input width is the residual hidden width; output width is unchanged. Use
`attach_adapters(..., input_site="residual")` for the same lower-level option.
Residual writer checkpoints record the site and input dimensions and cannot be
loaded into the default site. Existing default-site checkpoints retain their
signature and remain loadable. This option alone grants no experiment scope;
WP-L2's decision is recorded in `docs/reports/P1L-diagnosis.md`.
