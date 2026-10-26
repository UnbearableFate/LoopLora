# WP-P4L preparation interface

CLI (compute only; parent submits):

```bash
python scripts/prepare_p4s_learned.py --root /work/.../p4s-l/preparation \
  --out /work/.../LoopLora-wt/sym/reports/p4s-l
```

Optional source overrides: `--c4-manifest`, `--c4-thresholds`, `--h-root`.
Defaults are committed `reports/p4s-r32/{manifest.json,thresholds.csv}` and the
released WP-H root from `run_p4s_r32.py`. Root must be new; output design.json
must not exist. No head/core learned-halting module is imported, and no head is
trained. Preparation collects only six oracle_random validation traces, with
2000 episodes, seed4242, hops1..8, maxsteps16, certificates=false. Selection calls
existing WP-C3 `select_thresholds` with unchanged grids/tie ordering.

`REPORTDIR/design.json` is the complete pre-head design lock:

- `schema_version:1`, `task:'WP-P4L'`, `study:'P4-S-L'`, `status:'prepared_for_commit'`;
  `runs`:24 rows, preserving the18 C4 names/paths/identities/blake2b seals and
  adding6 WP-H oracle_random/r32 readers. Row fields: name,writer,r,variant,seed,
  family,checkpoint,checkpoint_blake2b; C4 reference metadata is preserved.
- `label_free_thresholds`:24 rows with family,criterion,threshold,
  validation_accuracy,validation_mean_steps,`source` mapping containing kind,
  path,blake2b. C4's18 numeric rows are reused unchanged; only6 random rows are new.
- `best_label_free`: mapping family→complete selected threshold row, using
  validation_accuracy descending and criterion order read_norm,relative_change,
  output_stability to break ties. It is locked before head training.
- Top-level `max_steps:16`, `minimum_steps:1`, `hops:[1..8]`, `r:32`,
  `normalization_epsilon:1e-12`, `training_seeds:[17,29,43]`,
  `theta_grid:[.05,.10,..,.95]`, `theta_selection` with validation/tie semantics.
- `head`:input_dim51,hidden_dim64,hidden_activation'relu',output_dim1,
  output_activation'sigmoid', ordered features/normalization. Head output targets
  are candidate argmax correctness at post-update steps1..16.
- `training`:split'train',generator_seed31337,episodes20000,steps2000,
  optimizer'adam',lr.001,batch_episodes256,hops_per_episode8,steps_per_query16;
  every minibatch retains all8 queries and16 states for each sampled episode.
- `validation`:split'val',generator_seed4242,episodes2000;
  `confirmation`:split'test',generator_seed999002,episodes2000.
- `bootstrap`:resamples2000,confidence`1-.05/6`,seed2718, paired episode draws
  shared across seeds; `latency`:batch_size64,warmup5,repeats20.
- `seed_derivation`:head_init and episode_sampler rules exactly
  `derive_seed(seed,'p4s-l',family,'head_init'/'episode_sampler')`.
- `primary_families`:six non-joint families; `descriptive_families`:two joint
  families; source-file seals, source provenance, preparation paths/timestamp,
  and user clarification. No top-level self-hash is stored in design.json; the
  separate preparation_audit.json/preparation.json records stable_hash(full parsed design).

Raw preparation retains `manifest.json`, `traces/<random-run>/val.npz`,
`thresholds.json` (six choices and full grids), and `timing.json`. Reports also
include compact `label_free_thresholds.csv` and `preparation_audit.json`.
No source checkpoint is modified. All24 checkpoint seals are rechecked after
selection. Parent must commit design.json and its rows before any head training.
The clarification is explicit: no random/r32 thresholds existed; their validation
selection occurs now, using unchanged C3 rules, and no C4 family is reselected.
