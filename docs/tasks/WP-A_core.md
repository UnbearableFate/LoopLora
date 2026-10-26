# WP-A — Core adapters, delta memory, Gate 0 tests (plan T-02, §10.3, §10.5)

Read first: `docs/README.md`, `docs/notation.md` (binding), plan §4, §5.1, §6.3, §10, Appendix A.
Branch/worktree: `wp/core` at `/work/xg24i002/x10041/LoopLora-wt/core`. Report: `docs/reports/WP-A.md`.

## Deliverables (`src/looplora/core/`)
1. `config.py` — `AdapterConfig` dataclass: `variant` (`lora|parallel|loop|iterq|fixedq|unshared|double_ba`),
   `d_in, d_out, r, r_mem (default r), rho, alpha, gate ("scalar"|"input"), lambda_init, beta_init, eps,
   init stds (notation §1), eta (double_ba), unshared_steps (B8), double_ba_form ("skip"|"gated")`.
   B6/B7 are configurations (larger r / r_mem with L=1), not separate classes.
2. `memory.py` — functional delta writer and state helpers:
   - `write(M, u, write_mask, *, K, V, lam, beta, eps) -> M_new` (one token, batched; mask per row; masked rows
     unchanged, no decay).
   - `zeros_state(batch, r_mem, dtype, device)`, `fork(M, n)` (repeat a snapshot for n independent readers).
   - `scan_prewrite(M0, u[B,T,r], valid, write, reset) -> (M_prev[B,T,r_m,r_m], M_final)`: the state each
     token reads (before its own write), honouring reset/valid/write exactly as notation §4.
3. `adapters.py` — one `nn.Module` per variant with a common interface:
   - `read(u, M_prev, loop_steps) -> z` (no write; `u = A x`; works for `[B, r]` or `[B, T, r]` with matching
     `M_prev` `[B, r_m, r_m]` or `[B, T, r_m, r_m]`; `loop_steps` is one int per call — per-token L for
     adaptive stopping is a later extension, do not build it now).
   - `branch(z) -> α B z`; `u(x) -> A x`; `write(M_prev, u, write_mask) -> M_new` (memory variants).
   - `process_sequence(x, M0, loop_steps, valid, write, reset) -> (branch_out[B,T,d_out], M_final, stats)`:
     the sequential per-token reference semantics (read with pre-write state, then write). `stats` counts
     reads and writes per sequence.
   - `process_sequence_fast(...)`: same semantics via `scan_prewrite` + one batched read over all tokens;
     must match `process_sequence` (it is valid because the writer does not depend on the reader output).
   - V2-FQ computes `m = M_prev Q u` once per token and caches it across the L updates.
4. `wrapper.py` — `AdaptedLinear(nn.Module)`: wraps a frozen `nn.Linear` (W, bias kept frozen, no `no_grad`)
   plus one adapter; `forward(x)` returns `W x + b + branch`. The caller sets per-call context through an
   explicit object (e.g. `ctx = MemoryContext(state, loop_steps, valid, write, reset, mode)`), where mode ∈
   `online | write_only | read_only`; after forward the updated state is available on the context. It counts
   base-projection calls. This is the hook the LLM stage (WP-E) uses on `o_proj`; keep it free of Qwen code.
5. `reference.py` — Appendix A `recurrent_adapter_step` (single sample, single token, unbatched) as the oracle.
6. `accounting.py` — `param_count(cfg)`, `macs_per_token(cfg, loop_steps, writes)`, `state_bytes(cfg,
   batch, n_modules, dtype)` per notation §7 (MAC counts are the dominant terms, not profiler numbers).
7. `contraction.py` — `contraction_upper_bound(R, U, Q, M, rho, lip=1.0)` (exact spectral norms via
   `torch.linalg.matrix_norm(ord=2)`), `residual(F, z)`, `fixed_point_error_bound(e_s, c)`.

## Tests (`tests/`, all must pass on a compute node; fp64 unless stated)
| file | must show |
| --- | --- |
| `test_reference_equivalence.py` | batched `process_sequence` and `_fast` equal the Appendix-A oracle token by token for every memory variant, L ∈ {0,1,2,4}, random masks/resets |
| `test_causality.py` | changing tokens t' > t leaves outputs and states ≤ t unchanged |
| `test_state_lifecycle.py` | padding never decays/writes; write=0 keeps state; reset zeroes before the token; packed sequences with reset equal separate runs; forked snapshots read independently (no cross-contamination across questions or L) |
| `test_gradient_flow.py` | B=0 ⇒ output equals base exactly; first backward: only B (and nothing else) gets nonzero grad, then after one optimizer step on B the internal params (A, R, U, Q, K, V, gates) get nonzero grads; grad reaches an upstream input through frozen W; W gets no grad; `torch.autograd.gradcheck` on a tiny V2 sequence (incl. writer) |
| `test_loop_invariants.py` | L does not change base-call count or write count; L=0 equals LoRA on the same params; FQ equals its definition; B8 with tied per-step params equals V2 |
| `test_precision.py` | bf16-state run vs fp32 reference on a fixed instance: report max abs/rel error in the test output; assert a loose bound you justify |
| `test_contraction.py` | on constructed contractive instances (c < 1), iterates satisfy the error bound at every step; a non-contractive instance is reported as such |
| `test_accounting.py` | param counts equal `sum(p.numel())`; MAC formula matches notation §7 for r = r_m and r_m ≠ r |

Run: `scripts/submit.py --name t-core --walltime 00:15:00 --worktree . --wait -- -m pytest -q tests`.

## Constraints
- FP32/FP64 reference first; no custom kernels, no torch.compile in this WP.
- No in-place updates on tensors autograd may have saved (state updates return new tensors).
- Keep the public interface above stable; WP-E (LLM) and the symbolic stage build on it.
