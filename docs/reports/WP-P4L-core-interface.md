# WP-P4L core interface proposal

Implementation lives only in `core/learned_halting.py`; training remains with
the parent. Inputs reuse `core.stopping.ReaderBatch`; no labels enter features.

- Constants: `MAX_STEPS=16`, `RANK=32`, `FEATURE_DIM=51`, `EPS=1e-12`.
- `HaltingHead(seed=0)`: deterministic 51→64→1 MLP with ReLU hidden layer.
  The seed is used directly without further derivation; the parent supplies the
  locked `derive_seed(runseed,'p4s-l',family,'head_init')` initialization seed.
  `forward(features)` returns sigmoid probabilities with the final singleton
  removed; `logits(features)` supports stable BCE-with-logits training. Leading
  dimensions are preserved, e.g. `[N,16,51]` → `[N,16]`.
- `trace_frozen_reader(adapter, batch, max_steps=16) -> LearnedReaderTrace`.
  This subclasses `ReaderTrace`: predictions/read_norm/relative_change are
  `[N,S+1]` for states z0..zS; states are `[N,S+1,32]`, features `[N,S,51]`.
  Initial relative change is NaN, matching the existing helper; all post-step
  features are finite. Feature indices are z0:32, norm32, relative_change33,
  post-state read-norm ratio34, and one-hot step1..16 at35:51. The denominator
  for read ratio is the norm at z0, clamped at EPS; relative change clamps the
  previous-state norm. The reader is never changed and tracing is no-grad.
- `learned_halting_steps(probabilities, theta) -> Tensor[N]`: probabilities
  `[N,S]`, S≤16; first inclusive p≥theta, otherwise earliest argmax, returning
  1-based selected steps. Finite probabilities in[0,1], theta in[0,1].
- `learned_adaptive_read(adapter,batch,head,theta,max_steps=16)` returns
  `LearnedAdaptiveResult`, extending existing `AdaptiveResult` with
  `executed_updates` and `threshold_hit`. Existing `.steps` means selected
  state. Hits at S execute S updates/S+1 memory reads; no hit executes the cap
  (16 updates/17 reads) while returning the earliest best-probability state,
  prediction and its selected step. Active queries compact immediately at hits;
  only best fallback state/step/probability are retained, not a full trajectory.

Reader validation enforces iterq/r32, compatible batch shapes/devices/dtypes,
finite inputs/states/readouts/features/head outputs, and 1≤max_steps≤16.
Invalid/nonfinite inputs fail explicitly. Empty batches return empty records.
No runtime execution on login; parent submits compute tests.
