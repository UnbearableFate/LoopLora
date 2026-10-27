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

WP-C6 adds `variant: B8`: four step-specific R/U/Q readers of the same memory
snapshot, with training and evaluation fixed at L=4 in writer, joint and reader
stages. Both input sites and evaluation protocols are supported. Writer
signatures remain independent of reader sharing, so ordinary M1 and B8 writers
can be consumed by either reader when their site, dimensions and writer settings
match. M1, B5 and B3 reader/joint training accepts `train.depths: [4]` without
resampling other depths. `train.prefetch_workers: 2` uses the existing CPU batch
stream; memory-only support writes use `scan_final`.

WP-L3's `configuration(arm, variant="B8")` and `WriterFixExperiment` support the
same seven interventions with B8. The addressing auxiliary uses Q[0] z⁰; oracle
addressing uses each step's own R/U. Standardization, paired-position writes,
training-only value heads and inference exports keep their existing behavior.
These diagnostic interventions retain WP-L3's truncated memory-only scope.
The default `configuration(arm)` and all existing variants remain unchanged.
This implementation support does not authorize a new scientific run.

# DPFP address features (WP-L5)

`looplora.llm.dpfp.configuration(arm, steps, E=16, smoke=False)` builds the three registered arms D32-3, D128-1 and
D128-3 of `docs/prereg/DPFP.md`: WP-L4's `read_fix.configuration('R', ...)` at rank r = r_mem = 32, 128 and 128, plus
`model.dpfp_nu` (3, 1, 3) and a descriptive `dpfp` block; nothing else differs. `AdapterConfig.dpfp_nu` (default 0 =
off) maps write keys and read queries through `core.memory.dpfp_features` and an L2 normalization, and the memory
becomes `[B, r_mem, key_dim]` with `key_dim = 2 * r_mem * dpfp_nu` (`docs/notation.md` section 9). Like the WP-L3 and
WP-L4 blocks, the new keys are added after `resolve_config`: they are not YAML or default fields, so existing
configs, config hashes and checkpoints are unchanged, and writer signatures gain `dpfp_nu` only when it is nonzero.
The option applies to the `fact_final` writers (W1-W3, W5, R), not to the paired W4 write. One arm per job:
`python -m looplora.llm.dpfp_run --arm D32-3 --out ROOT (--steps N --smoke | --schedule release.json)`.
`docs/reports/WP-L5-state.md` has the tests and smoke commands. This support alone authorizes no scientific run.

# Least-squares stores and the read-failure factorial (WP-L7)

`AdapterConfig.store` (`delta` default, `ridge`, `sweep`), `store_eta`, `store_sweeps` and `unit_query` (all off by
default; `docs/notation.md` section 10) add the two LSQ stores and unit-normalized queries. Like `dpfp_nu` they are read
from `config['model']` after `resolve_config`, so existing configs, hashes, signatures and checkpoints are unchanged and the
writer signature gains the fields only when non-default. `looplora.llm.lsq.configuration(arm, steps, E=16, smoke=False,
eta=None)` builds the four conditional arms L32-K4, L32-RG, L128-K4 and L128-RG of `docs/prereg/LSQ.md` (WP-L4's R at rank
32 or 128 plus the store fields and an `lsq` block; the RG arms need their registered `eta`), one arm per job:
`python -m looplora.llm.lsq_run --arm L32-K4 --out ROOT (--steps N --smoke [--eta X] | --schedule release.json)`.
`python -m looplora.llm.rfx_run --out DIR [--smoke]` is the evaluation-only read-failure factorial of
`docs/prereg/RFX.md` on the three frozen WP-L4 checkpoints (its `summary.json` names the ranks that trigger LSQ).
`docs/reports/WP-L5-state.md` has the step-2 order and commands. This support alone authorizes no scientific run.

# Fact-only oracle writer (WP-C7)

Set `writer: oracle_random`, `stage: reader`, `model.r: 32` (or `128`, P1-L B3) and
`protocol: memory_only`; no `writer_checkpoint` is consumed. `oracle_seed: 17`
is fixed independently of the reader training seed. Keys are global random
unit codes for `(entity, relation)` pairs (index `entity * n_rel + relation`);
values are exactly the symbolic `OracleWriter` entity codes at that seed and
memory dimension. Rank32 is the unchanged default. Checkpoint signatures record
the actual dimension and reject a writer of another rank.
Both tables are frozen fp32 buffers saved in the full checkpoint and
`writer.pt`. Each valid fact writes once, in fact-list order, with λ = β = 1.
No support text is rendered or forwarded through the LM. The learned writer's
`scan_final` path is bypassed because there are no support features to project;
the oracle uses the unchanged symbolic delta scan on already normalized codes.

M1 (including fixed `train.depths: [4]`), B8, B4, B5, B3 and M1-lin use their
ordinary LM question encodings and readers. Question-side `A` stays trainable;
unused learned K/V/gates are frozen. `train.prefetch_workers: 2` prepares only
question tokens and episode arrays in two CPU workers. Evaluation is restricted
to memory-only. Cross-reader comparisons assert identical memory/write counts;
the ordinary per-checkpoint depth check still asserts identical x/u/M/counts.

```yaml
writer: oracle_random
oracle_seed: 17
stage: reader
variant: B8
protocol: memory_only
model: {r: 32, layer: 14, input_site: o_proj}
task: {name: t2_chain_end, E: 12, Q: 4, n_rel: 4, head_sharing: 0.3, chain_lengths: [8, 4]}
train: {batch_size: 64, depths: [4], prefetch_workers: 2}
```
