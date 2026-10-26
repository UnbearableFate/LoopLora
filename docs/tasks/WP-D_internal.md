# WP-D implementation contracts

Worker-local coordination record; WP-D and notation remain authoritative. All
execution/imports/tests happen in a Miyabi-G allocation via scripts/submit.py.
Do not change core/data, dependencies, or other branches. No agent commits.

## Ownership
- Root: backbone.py, writers.py, model.py, train.py, integration, GPU validation.
- evaluation agent: evaluate.py and tests/test_sym_evaluate.py.
- statistics agent: stats.py, figures.py, tests/test_sym_stats.py.
- runner agent: config.py, run.py, configs/p1s/, scripts/collect_registry.py,
  tests/test_sym_run.py.

## Shared Python contracts
Config is a JSON-compatible dict. `config.load_config(path) -> dict` resolves
YAML defaults and validates it. `config.resolve_config(mapping) -> dict` does
the same for programmatic input. Groups:
- name, seed, backbone_seed, device (cuda), writer (oracle_random/oracle_orth/
  t1_frozen/joint), variant (B3/B4/B5/M1/M1-lin/B6/B7/B8), activation (relu/identity).
- model: d_e=64, r=64, r_mem=64, rho=.25, alpha=1., lambda_init=.99,
  beta_init=.1, max_hop=16; B6 rank=96, B7 r_mem=96; oracle requires r=r_mem.
- task: TaskSpec fields (name, V=1024 etc., resolved by WP-B).
- train: steps=3000, batch_size=32, lr=.003, weight_decay=.01,
  grad_clip=1., warmup_steps=20, depths=[1,2,4], log_every=10.
- pretrain: steps=3000, batch_size=32, lr=.003, warmup_steps=20,
  writer_checkpoint=null; t1_frozen uses same writer initialization/pretrain
  seed independent of reader variant, then resets reader with a separate seed.
- eval: episodes=64, batch_size=32, depths=[0,1,2,3,4,6,8,12,16],
  splits=[val,test,test_unseen], hops=[1,2,3,4,5,6,7,8], seed=4242,
  counterfactuals=true; T1 only hop=1; B4/B6/B7 only L=1, B8 only L=4.
- stats: resamples=2000, seed=2718.

`train.train(config, out: Path) -> (model, summary_dict)` writes checkpoint.pt,
train_log.jsonl, writer.pt when applicable. Summary includes mean_L,
train_seconds, train_step_seconds, train_tokens, pretrain_seconds.

`model.SymbolicModel(config)` is an nn.Module; `.prepare(EpisodeArrays)` returns
PreparedBatch: x [B,Q,d_in], u [B,Q,r], memory [B,rm,rm],
answer [B,Q], candidate_mask [B,V+1], codes [B,V+1,d_out],
true_value [B,Q,rm], retrieval_valid [B,Q], write_count [B].
PreparedBatch tensors never change during read; `.read(prepared, L)` returns
dict with logits [B,Q,V+1], z [B,Q,r], first_read_cosine [B,Q]
(NaN when no read or no valid target). Model exposes `.device`, `.r_mem`,
`.backbone`, `.adapter`, `.writer_condition`. No query writes.
`model.make_task_spec(config, evaluation=False) -> TaskSpec`; evaluation ensures
one query per hop using deterministic generator-compatible stratification.
Evaluator may generate separately per hop and combine query fields on the same
facts only when verified consistent; prefer WP-B support for stratified queries.

`evaluate.evaluate(model, config, out: Path) -> (metrics, timing)` writes
samples.npz, metrics.json. Metrics contain `table` list rows {split, condition,
hop, L, n, accuracy, full_accuracy, chance_accuracy}; diagnostics and memory_hashes
separate keys. Samples are flat aligned arrays with split, condition, episode_id,
query_id, hop, L, prediction, answer, correct, full_prediction, full_correct,
candidate_size. Conditions: original, replace_key_fact, swap_memory. Add pairing
metadata, first-read diagnostic etc. as needed. Episode id = episode_seed.
Hash x/u/M + write_count and assert unchanged across L; each test dataset is
generated once, paired across variants via eval seed/spec independent of variant.

`stats.depth_comparisons(samples, resamples=2000, seed=...) -> list[dict]`;
`stats.compare_runs(samples_a, samples_b, depth_a, depth_b, ...)` verifies matching
episode/query/answer, returns episode-cluster paired bootstrap results. Across-run
comparison is explicit CLI/figure operation, never selects from test scores.
Statistics must not silently compare different data/splits/conditions.

Runner calls train then evaluate then depth_comparisons; writes config.json with
config/hash/git_commit/source_dirty, timing.json, record.json (plan §9.4).
Suite YAML: list of {config: relative_yaml, name: unique_name, overrides: mapping};
overrides use recursive merge, each child process gets its own resolved config
and output dir; --procs bounds subprocess concurrency. Smoke suite includes T1
and T2_chain_end for B3/B4/B5/M1/M1-lin/B6/B7/B8, 300 steps (=10% of 3000), one
seed, small fixed evaluation sets. B6/B7 use joint for dimensional change; primary
B3/B4/B5/M1/M1-lin use oracle_random for cheap controlled smoke. Include one
t1_frozen shared-writer path and oracle_orth coverage in tests. Do not execute any
pilot/screening. Config docs distinguish mechanism controls vs capacity controls.
