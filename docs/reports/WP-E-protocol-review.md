# WP-E protocol review (static)

Reviewed `inject.py`, `protocols.py`, and `test_llm_protocols.py`, including the
later write-policy/count and bf16 tests. **No concrete correctness findings**
within the requested padding, KV-fork, gradient, causality, and reset scope.

Evidence checked:

- `answer_scores` gathers the preceding **real** token for each answer label;
  incremental teacher forcing keeps its previous logits when an answer slot is
  padding. Both assign positions by cumulative valid-token counts and mask
  physical padding positions, including within the retained support prefix.
- `_fork_cache` clones every retained K/V tensor; `AdapterBank.use` independently
  clones each memory tensor and restores scoped contexts in `finally`.
  Installed Transformers 4.57.6 `Qwen3Attention.forward` updates passed caches
  even when `use_cache=False`, so the explicit clone is necessary and present.
- Reset support attention combines same-segment, causal, and valid-key masks.
  Position offsets restart on reset, including reset-at-padding; retained
  attention exposes only the last segment. Installed `masking_utils.py` returns
  an explicit 4-D mask unchanged, preserving that isolation.
- The frozen backbone forwards and K/V/memory clones do not detach activations;
  loss construction preserves autograd. Current tests check nonzero A/B/R/U/Q/K/V
  gradients through the full backbone, absent backbone gradients, multi-layer
  attachment, padding/reset equivalence, cache immutability, read-only question
  forks, and base/write counts independent of L.

Validation limitation: read-only source inspection on login node `miyabi-g2`;
no Torch/Transformers imports or runtime tests executed. Provisional core API
names were deliberately excluded from findings; final core integration and
GPU numeric tolerances remain for the primary agent's compute tests.
