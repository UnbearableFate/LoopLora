# WP-G implementation coordination

Authorized: implementation, compute tests and one 10%-step smoke suite; no pilots,
screening, or confirmation-set execution. WP-D was merged through master with
orchestrator authorization; its statistics and source metadata are reused.

Runner subagent owns `llm/config.py`, `llm/run.py`, configs/p1l and runner tests.
Primary owns `llm/protocols.py` truncation support and artifact audit. Training subagent owns `llm/train.py`,
`llm/batching.py`, training tests. Evaluation subagent owns `llm/evaluate.py` and
evaluation tests. Do not edit core/data/sym/dependencies. All runtime on compute
via scripts/submit.py, never Torch/pytest/Transformers on login. Do not commit
from subagents. Shared conventions: `derive_seed`, blake2b, fp32 adapters.

## Configuration (resolved dict; primary creates config.py)
- name, seed (smoke 20261027), device='cuda', stage=writer|reader|joint|lora_ctx|baseline,
  variant=M1|M1-lin|B5|B3|B4|B1|B2|B0; protocol=memory_only|context_retained;
  write_policy=fact_final|all_tokens (smoke fact_final, not a selected scientific policy).
- model: r=32, layer=14, rho=.25, alpha=1.0; backbone pinned as WP-E.
- task: TaskSpec dict, default t2_chain_end E=24 Q=4 V=1024 n_rel=4
  chain_lengths=[8,8,8], hops=[1,2,3]. writer forces t1 n_rel4 head_sharing.3.
- train: steps=1000 default /100 smoke, batch_size=8, lr=.0003,
  weight_decay=.01, grad_clip=1, warmup_steps=20 default/2 smoke,
  depths=[1,2,4] (writer/B4/B1/B2 [1]), log_every=10.
- writer_checkpoint: null or path to writer.pt; reader requires path.
- eval: episodes=2000 default /200 smoke, batch_size=16, depths=[0,1,2,3,4,6,8],
  splits=['val'] smoke (can val/test/confirm/test_unseen for later stages),
  hops=[1,2,3,4,5,6,7,8] (writer [1]), seed=4242,
  counterfactuals=true, protocols=['memory_only'] (+context_retained M1/B0/B1/B2),
  allow_confirmation=false. val/test start0; confirm maps split test start100000.
- stats: resamples=2000, seed=2718. smoke=true for smoke configs.

## Training API
`train(config, out:Path, *, model=None, tokenizer=None) -> (LLMExperiment, summary)`;
base HF model/tokenizer may be reused. `load_checkpoint(path, model=None, tokenizer=None,
device=None) -> LLMExperiment`; adapter-only checkpoint plus config (never backbone).
`LLMExperiment.close()` removes injected adapters so base model can be reused.
Properties: .model (HF), .tokenizer, .config, .adapter (reader or None),
.writer (memory writer or None), .bank, .device.
`prepare(arrays, protocol=None, *, truncate=True) -> PreparedBatch`:
- .snapshot: WP-E SupportSnapshot, support encoded once per episode;
- .question_ids/.question_mask: [B,Q,T], .answer_ids: [B,Q] Qwen token IDs;
- .candidate_ids/.candidate_mask: [B,C] episode name token IDs, plus Unknown if needed;
- .arrays: original EpisodeArrays, .support_tokens (count).
`score(prepared, L, *, capture_inputs=False) -> (logits[B,Q,V], inputs|None)`;
inputs if captured: x[B,Q,T,2048] exact input to layer14 o_proj before adaptation,
no mutation; flatten B*Q internally and repeat snapshots for efficient one-call scoring.
For B0 capture the unwrapped o_proj input at the same point.
`prepare`/`score` retain autograd by default; inference caller uses no_grad.
`writer_state()` returns dict of A,K,V,a_lambda,a_beta (+input gate vectors if present).
B3 frozen readers load same writer externally, share frozen A, construct the same
snapshot for Gate0 but use LoopAdapter which ignores it. Support encoding can swap
in the external writer only for memory_only; no multi-module complexity.

Primary adds `encode_support(..., truncate=False)` option: truncate=True is only
memory_only + one module, write_only mode, forward hook stops after that o_proj;
state must equal full-forward support. For context_retained always full encoding.
B0 installs an empty AdapterBank (no wrapped module), so WP-E protocols can run
unchanged; no-adapter support remains full model unless optimized by caller.

## Evaluator API
`evaluate(experiment, config, out:Path) -> (metrics,timing)` writes samples.npz
and metrics.json. Reuse WP-D `stratify_queries`, tensor_hash, stats after merge.
Flat required stats fields: split, condition, episode_id, query_id, hop, L, answer,
correct; LLM primary prediction/correct are full vocabulary token IDs/EM. Also
full_prediction/full_correct aliases for record readers, candidate_prediction/
candidate_correct/candidate_size, original_answer/original_prediction,
memory_episode_id, protocol. To avoid ambiguity for stats comparisons, condition
includes protocol (e.g. memory_only/original); document metric semantics.
Assert exact hashes x/u/M/write count across L, and run.py asserts same hashes
across frozen-writer variants with equal data/protocol. JSON table must reproduce
from samples; use sym.stats.depth_comparisons / compare_runs directly.
Counterfactual replace forks N*Q contexts as WP-B; swap replaces support snapshot
from donor while retaining original question/answer/candidates; record both labels.

Runner mirrors sym.run record schema, source_metadata and config envelope; adds
stage/protocol/model/tokenizer versions. Smoke suite includes writer, 5 frozen
readers, 5 joint variants, LoRA/parallel-context and B0 both protocols (14 run directories).
Runner measures sequential model reuse versus --procs 2 in a bounded system-only
smoke scheduling comparison (no model selection); chooses measured faster mode,
then complete authorized suite. Outputs retained under canonical runs/p1l/.
