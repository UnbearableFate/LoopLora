# WP-A — completed 2026-10-26

Implemented all seven adapter variants, validated configuration/initialization, functional delta memory and snapshot helpers, sequential and batched-reader sequence paths, an independent Appendix-A oracle, the frozen-linear context wrapper, parameter/MAC/state accounting, and exact spectral-norm contraction bounds. Implementation: `216febf`; tests: `7059246`, on `wp/core`.

All orchestrator addenda are included: `memory.delta_update(M, k, v, write_mask, lam, beta)` accepts normalized codes and underlies `write`; `AdapterConfig.activation` selects `relu` or `identity` wherever φ appears; `AdaptedLinear` keeps the base input unchanged, casts only adapter input to adapter precision, and casts the branch to base-output precision through differentiable operations.

Validation command:

```sh
PYTHONUNBUFFERED=1 scripts/submit.py --name t-core --queue regular-g \
  --walltime 00:05:00 --worktree . --wait -- -m pytest -q tests -rP
```

**304 tests passed; exit 0**, Miyabi-G compute host `mg0852`, job **3711460.opbs**, 19:10:16–19:10:33 JST. Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/t-core-20261026-190904-1290240.log`. Tests cover both activations, all memory variants, both gate types, L=0/1/2/4, rectangular memory, masks/resets, causality, independent forks, writer-inclusive gradcheck, gradient startup, frozen-base input gradients, cached fixed queries, tied unshared readers, both double-BA initializations, wrapper modes/incremental calls, accounting, and contraction bounds. The BF16-base/FP32-adapter test verifies output dtype, FP32-reference agreement, FP32 state, and nonzero adapter gradients. No torch imports or pytest execution occurred on the login node.

BF16-state versus FP32, on the fixed 64-token instance: maximum output absolute error **0.000124395**, relative error **0.000275469**, final-state absolute error **0.00472598**, identical for both sequence paths. The test justifies its loose output bounds (0.02 absolute, 5% relative) for that bounded instance.

Interface choices: use `make_adapter(cfg)` and `MemoryContext`/`set_context`; masks default to valid/write=true and reset=false. Statistics are per-sequence int64 tensors counting memory reads and writes; fixed-query reads count once when L>0. `base_call_count` counts wrapped linear invocations; `base_token_count` counts projected rows. `fork` groups n independent copies per batch row. The standalone `scan_prewrite` additionally takes keyword writer parameters `K,V,lam,beta,eps`; the adapter method binds them. Unshared readers require L=`unshared_steps`, including zero.

Deviation: saturated debug-g predicted a 35-minute wait, so queued job 3711418 was cancelled before execution and validation used regular-g. The initial 303-test suite also passed (3711433.opbs); the final rerun includes addendum 3. No dependencies changed. No open WP-A issues. The fast path retains a sequential writer scan and O(B T r_mem²) snapshots; no speedup or general low-precision stability claim is made.
