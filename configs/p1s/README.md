Symbolic mechanism configurations use `config.resolve_config` defaults. `default.yaml`
records the 3000-step main configuration; it is preparation only, not authorization
for a pilot or screening run. `smoke.yaml` contains 24 runs: both T1 and T2 for all
eight variants, two fixed-L=4 M1 smokes, plus one shared T1 writer and five readers
using that exact artifact. B4 also supplies the fixed-L=1 iterative-reader smoke.
Every training/pretraining stage is 300 steps, one seed (17), with 32 evaluation
episodes per split/hop; training and pretraining keep the default batch size 32.
The 10% ratio applies to train/pretrain steps. T1 evaluates only hop 1. B4/B6/B7 fix L=1; B8 fixes L=4.
M1-lin sets the reader activation to identity.

Run only inside a Miyabi-G compute allocation, through the repository submit wrapper:

```bash
scripts/submit.py --name p1s-smoke --walltime 00:30:00 --commit <integrated-commit> --wait -- \
  -m looplora.sym.run --suite configs/p1s/smoke.yaml --procs 4 \
  --out /work/xg24i002/x10041/LoopLora/runs/p1s/smoke-17
```

Choose walltime/concurrency from measured timings and the current allocation.
Use an absolute persistent output path with `--commit`; its source archive lives
in temporary node storage. Development tests can use `--worktree .`. A
single run uses `--config configs/p1s/default.yaml --out runs/p1s/<unique-id>`.
Output directories must be empty. `--procs` bounds concurrent child processes;
all children share the first allocation-visible GPU. Child stdout/stderr is in
`_logs/<name>.log`; resolved inputs are in `_configs/<name>.json`; `suite.json`
records exit codes, skipped dependencies and elapsed time. Any failed child makes
the suite return nonzero. `depends_on: [name]` delays a run; `writer_from: name`
also sets `pretrain.writer_checkpoint` to `<suite-out>/<name>/writer.pt`.
Config paths in suite entries are relative to the suite YAML.

B3/B4/B5/M1/M1-lin use the same oracle writer and evaluation episodes for the
controlled mechanism comparison. The additional frozen-writer group shares one
learned writer checkpoint. B6/B7 use joint training with larger dimensions as
capacity controls; their memory is not held equal to the main comparison.

Each run stores `config.json` (resolved config, config hash, git commit, source
hash and dirty flag), `checkpoint.pt`, `train_log.jsonl`, `metrics.json`,
`samples.npz`, `timing.json` and `record.json`; learned frozen-writer runs also
store `writer.pt`. Source git metadata is read from `LL_SOURCE_ROOT` when set,
otherwise the package's repository root. Exported commit archives use the full
commit in submit.py's `LL_SPEC` because the archive contains no `.git`. Source
hashes cover code/configuration under `src`, `scripts`, `tests`, `configs`, plus
`pyproject.toml` and `uv.lock`. Hashes use blake2b. The record retains
the plan §9.4 identity, seeds, generator/model version, accounting, checkpoint,
sample paths, scores, latency, peak CUDA allocation and failure reason. Reader
training tokens/time and local pretraining tokens/time are recorded separately.
A reused writer records zero local pretraining cost, its consumed checkpoint path
and its source `record.json` path; the source record owns the pretraining cost. CPU peak
GPU allocation is null. `scores` and `latency` are JSON objects in registry CSV.

The following collection command uses only the standard library:

```bash
python scripts/collect_registry.py runs/p1s --out reports/experiment_registry.csv
```

CSV rows sort by experiment id and absolute run directory; nested values use
canonical JSON. The registry includes failed runs and preserves additional record
fields as deterministically ordered columns. Sample archives remain the source
for recomputing scores and paired statistics.

The WP-D acceptance wrapper runs this same smoke matrix, then checks all saved
sample tables, Gate 0 hashes, and cross-variant equality of controlled writer
states. It writes timing CSV, a registry, prespecified paired contrasts, and
test-split figures. It does not choose a method or launch further experiments:

```bash
scripts/submit.py --name s-sym --queue regular-g --walltime 00:15:00 \
  --commit <integrated-commit> --wait -- scripts/smoke_sym.py \
  --suite configs/p1s/smoke.yaml --procs 4 \
  --runs /work/xg24i002/x10041/LoopLora/runs/p1s/wp-d-smoke \
  --reports /work/xg24i002/x10041/LoopLora-wt/sym/reports
```

Use `--audit-only` to recreate reports from the completed suite without training.
For a new sweep of a saved checkpoint, use `python -m looplora.sym.evaluate
--checkpoint <run>/checkpoint.pt --out <new-empty-directory>` on compute.

Evaluation queries are deterministically constructed from each support graph:
one query per requested hop, keeping support unchanged. The original query hop
labels stratify counterfactual comparisons even when rewiring shortens a chain.
Swap-memory keeps original query features, labels and candidates. `oracle_orth`
uses one source-episode basis for factual/counterfactual contexts and assigns
new entities to unused orthogonal slots; their union must fit `r_mem`.
T3 missing-data evaluation requires generated query coverage for each requested
hop; answer-changing counterfactuals require answerable source queries, as in
the data-generator contract. These constraints do not affect the T1/T2 smoke.

WP-D2 adds `model.reference_r` (defaults to `model.r`): B6 requires an explicit
larger `model.r`, and B7 an explicit `model.r_mem` larger than both reader and
reference ranks. `pretrain.task` merges overrides into the T1 writer task;
`n_rel` and `V` must remain compatible with the reader backbone. Joint runs also
export `writer.pt`. Cross-stage reuse accepts
`writer_from: {checkpoint: /absolute/path/to/writer.pt}`. `runtime.threads`
defaults to one CPU thread per packed worker and is recorded in run provenance.

On compute, `python -m looplora.sym.pilot plan1 --out <plan-dir> --shards 3`
creates the validation-only Pilot-1 grid. Run each shard through `sym.run` and
collect it with `sym.pilot summarize1 --runs <runs-dir> --manifest
<plan-dir>/manifest.json --out <report-dir>`. The `plan2` action consumes
`--selection1 <report-dir>/selection.json`; `plan-replication` also consumes
`--selection2`. The corresponding summary actions retain every LR result and
loss curve, and selected hop/depth tables. Selection uses the best LR per E,
then E*, then the LR at E*; ties use the smaller LR. The loss plateau diagnostic
is not a selection criterion. All pilot training uses seed17 and all pilot
evaluation uses split val, generator seed4242.

Saved checkpoints support allowlisted evaluation overrides, for example
`--set eval.episodes=2000 --set 'eval.splits=[val]' --set eval.seed=4242`.
Effective inputs, the original checkpoint config, and overrides are retained.
No pilot command evaluates the screening or confirmation sets.

For pooled paired statistics, `python -m looplora.sym.stats --pooled-a a.json
--pooled-b b.json --depth-a 4 --depth-b 1 --hops 2 3 --split val --out stats.json`
accepts JSON maps from training-seed labels to sample archives (paths relative
to each manifest). Use the same manifest for a within-checkpoint depth contrast.
Episode draws are shared across seeds; seed-specific accuracy differences are
averaged with equal seed weight. The reported interval is conditional on the
provided training seeds.
