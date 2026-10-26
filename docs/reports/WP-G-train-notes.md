# WP-G training and batching implementation notes

Owned files: `llm/train.py`, `llm/batching.py`, `tests/test_llm_train.py`.
No core/data/sym/dependency edits, commits, or job submissions by this worker.

`LLMExperiment(config, model=..., tokenizer=...)` installs one adapter; `close()`
restores the original projection and permits sequential backbone/tokenizer reuse.
`train(config, out, model=..., tokenizer=...)` returns experiment plus summary;
`load_checkpoint(path, model=..., tokenizer=..., device=...)` restores a self-contained
adapter/writer checkpoint without reading its original writer path. Backbone
weights are never saved. Writer checkpoints validate exact signatures, parameter
names, shapes, and fp32 dtype. Signatures pin backbone revision, insertion,
rank/dimensions, gate, write policy, and entity vocabulary.

Writer training forces rendered T1, four relations, head sharing .3, L=1;
reader loads A/K/V/gates exactly and freezes them while freshly seeded R/U/Q/B
train; joint starts fresh; LoRA training forces context-retained; B0 writes an
empty log and adapter-only baseline checkpoint with zero optimization steps.
AdamW minimizes **full-vocabulary** one-token CE. All initialization, episode,
and sampled-depth seeds derive from `derive_seed`. Backbone stays in eval mode.
Reader training verifies writer tensors remain bitwise equal before/after fitting.

Preparation renders and pads support once per episode. Scoring flattens questions
in episode-major B,Q order and repeats state/KV rows Q times for one backbone
call. Returned input captures are cloned raw o_proj inputs [B,Q,T,d_in].
`PreparedBatch` additionally exposes support IDs/masks/fact-final masks and
`write_count[B]`: actual permitted memory writes, zero for B0/B1/B2. Memory-only
B0/B1/B2 skip irrelevant support forwards. B3 uses a separate memory writer sharing
A with its LoopAdapter; prepare temporarily swaps it into the same wrapper for
memory-only support encoding, then restores the reader, which ignores M.

Validation prepared: tiny real Qwen3 + pinned tokenizer integration tests for
flattened forks and one-token greedy agreement (both protocols), full-vocabulary
CE, adapter-only saves, writer forcing/freezing, all five reader variants,
self-contained reload, B3 same-M/shared-A/ignores-M, baseline model reuse, and
writer-signature rejection. AST syntax validation passed on login/control
miyabi-g2; Torch/Transformers/pytest have not been imported there. Compute tests
are coordinated by the primary; final evidence is in [WP-G.md](WP-G.md).

Addendum 1: B2 now maps to the core `parallel` adapter under `lora_ctx`, keeps
B1 as the default, trains with the same context-retained loop/rank/position/data,
and has no writer. Added B2 optimization/reuse tests and both-protocol checks,
including suppression of a negative bottleneck feature. AST validation passed;
the primary owns compute execution/results.

Read-only integration review found no blocking training/evaluator API mismatch.
The runner now avoids duplicate bootstrap computation and counts active external
B3 writer parameters separately; both review findings were resolved.
