# JS — spectral-radius regularization of the terminal reader state

Binding [preregistration](../../docs/prereg/JS.md), [task](../../docs/tasks/WP-JS_jacobian_stability.md).
Scientific design locked in `852d293` before training or model-score inspection. All numerical work uses
compute nodes via `scripts/submit.py`; prior runs and all reference checkpoints stay read-only.
Raw plans, checkpoints, logs, validation/confirmation traces and full statistics:
`/work/xg24i002/x10041/LoopLora-wt/symf/runs/js/`.

- `design.json`, `method.json`: exact ReLU Jacobian at z_L, eigenvalue backend/timing, gradient tests,
  gamma0.9/lambda1, random-depth distribution, reference checkpoint paths and BLAKE2b seals, statistics and
  stopping rules. CPU `torch.linalg.eigvals` is exact; neither singular values nor an unconverged power iteration
  substitute for spectral radius. Gradients include trainable memory; ReLU-gate derivatives are locally zero.
- `smoke_timing.json`, `cost_gate.json`: 1500-step/four-family smoke, 200 validation episodes, finite/active
  regularizers and measured job sizing. No smoke score was read. RS radius is unmeasured during training.
- `training_manifest.csv`, `training.csv`: all twelve 15000-step, batch64, lr0.003 runs, seeds17/29/43,
  final losses/penalties, activation counts and resource use. Main source `731a2a2`.
- `checkpoint_manifest.csv`: twelve new models and twelve unchanged M1-L4/mixed references.
- `threshold_grid.csv`, `thresholds.csv`, `thresholds.json`: WP-C3 relative-change thresholds selected solely on
  validation seed4242, 2000 episodes, all eight hops and three fixed seeds. Integer correct-count ties prefer
  fewer total steps, then smaller threshold. The selected file must match the committed source before confirmation.
- `accuracy.csv`: writer/condition/hop/L/seed counts, accuracy and mean reads. `stopped` means the first
  qualifying post-update state (or cap16); a saved full trace is used without inspecting labels for each stop.
- `endpoints.csv`, `tables.md`, `decision.json`: 97.5% pooled paired episode-bootstrap CIs, 2000 draws,
  per-seed effects and registered overshoot criterion. Primary DeltaS resamples the four-term depth-gain
  difference jointly across conditions and all three training seeds. Descriptive RS/reference contrasts
  use L4/16 and hops1–3,4–8,1–8.
- `rho.csv`, `rho_per_seed.csv`: measured spectral-radius distributions at z4/z8/z16; hop0 pools hops1–8.
  Fractions below gamma and1 describe reached states and are not contraction certificates or correctness proofs.
- `stopping.csv`: locked thresholds, mean steps, fraction halting before the cap and correct counts per seed.
- `accuracy_depth_{oracle_random,joint_fact}.{png,pdf}`: all eight hop curves for RS, RS+J and both references.
- `audit.json`, `jobs.csv`, `budget.json`: trace/file seals, paired labels, immutable inputs/memory,
  training/validation/confirmation barriers and actual resource use including failed development checks.

Confirmation is new test seed999003, 2000 episodes, one query per hop1–8, read depths1–16.
Every main run must finish first; every model's blind trace must exist before confirmation aggregation.
Internal statistics encode the stopped policy as L17; it always selects an actual z1..z16 and is displayed
as `stopped`. CIs condition on the three fixed training seeds; no gate or follow-up depends on JS.
