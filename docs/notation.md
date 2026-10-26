# T-01 Notation and binding semantics (v1, 2026-10-26)

Source: plan §4–§6, §10 and Appendix A. Where this file is more specific than the plan, this file binds the code.

## 1. Shapes and conventions
Math uses column vectors. Code uses row-major batches: `x: [B, T, d_in]`; parameter matrices are stored with
their math shape and applied as `x @ A.T`. Memory `M: [B, r_m, r_m]` maps key → value:
`m = M q` ⇔ `einsum("bij,bj->bi", M, q)`; a write `err kᵀ` ⇔ `einsum("bi,bj->bij", err, k)`.

| Symbol | Shape (math) | Meaning | Trainable |
| --- | --- | --- | --- |
| `x_t` | d_in | input of the wrapped linear at token t | – |
| `W, b_W` | d_out×d_in, d_out | frozen base projection (bias optional) | no |
| `A` | r×d_in | input bottleneck; init kaiming-uniform(a=√5) as in PEFT LoRA | yes |
| `B` | d_out×r | output bottleneck; init **zeros** | yes |
| `R` | r×r | recurrent core; init N(0, (σ_R/√r)²), σ_R=0.5 | yes |
| `U` | r×r_m | readout → core; init N(0, (σ_U/√r_m)²), σ_U=0.5 | yes |
| `Q` | r_m×r | query; init N(0, 1/r) | yes |
| `K, V` | r_m×r | writer key/value; init N(0, 1/r) | yes (frozen in mechanism stage) |
| `a_λ, a_β` | scalar | λ=σ(a_λ) init 0.99, β=σ(a_β) init 0.1 | yes |
| `α` | scalar | output scale, fixed 1.0 (no α/r scaling) | no |
| `ρ` | scalar | loop strength, fixed 0.25 | no |
| `φ` | – | ReLU | – |
| `ε` | – | key-norm clamp 1e-6 | – |
| `L` | int ≥ 0 | number of reader updates for the token | – |
| `r_m` | int | memory dimension; `r_m = r` unless a variant says otherwise (B7) | – |

## 2. Reader (V2, main candidate)
```
u = A x ;  z⁰ = u
for s in 0..L-1:  q = Q zˢ ;  m = M_prev q ;  zˢ⁺¹ = u + ρ φ(R zˢ + U m)
y = W x + b_W + α B z^L
```
- `L = 0` ⇒ `z = u`: the branch is exactly LoRA (`α B A x`) and memory is not read.
- All L updates read the same pre-write snapshot `M_prev`; never build `U M Q` explicitly.

## 3. Writer (decay-first delta rule, plan §4.5)
```
k = K u / max(‖K u‖₂, ε) ;  v = V u
M̄ = λ M_prev ;  M_new = M̄ + β (v − M̄ k) kᵀ
```
`u = A x` is the same `u` the reader uses (shared A). Gate options: `scalar` (default, λ, β per module) or
`input` (later variant: λ_t = σ(w_λ·u + a_λ), β_t = σ(w_β·u + a_β), w init 0).

## 4. Per-token order and masks (each sequence b independently)
1. `reset[t]` ⇒ `M ← 0` before anything else for this token.
2. `valid[t] = 0` (padding) ⇒ output `W x + b_W`, no read, no write, **no decay**; state unchanged.
3. Read: L updates with `M_prev` (state before this token's write) → `y_t`.
4. `write[t] = 1` ⇒ `M ← writer(M_prev, u_t)` exactly once; `write[t] = 0` ⇒ state unchanged (no decay).
Changing L never changes the number of base-projection calls (one per token) or writes (≤ 1 per token).

## 5. Protocols
- **online**: valid context tokens read, then write, in causal order.
- **memory QA** (mechanism default): support-context tokens write (their reads are not scored); question and
  answer tokens read only (write = 0) from the final context snapshot. Each question (and each L) forks
  from the same snapshot; no answer token is ever written before it is scored.
- **context-retained** (LLM only): the support text also stays in the backbone's attention/KV.
- **memory-only** (LLM only): the support text's backbone KV is dropped after writing; only `M` is kept.

## 6. Variants (all at the same insertion point; ids from plan §8.1)
| id | name | branch (y = W x + b_W + …) | memory |
| --- | --- | --- | --- |
| B1/V0 | LoRA | α B A x | none |
| B2 | parallel nonlinear adapter | α B φ(A x) | none |
| B3/V1 | no-memory loop | zˢ⁺¹ = u + ρ φ(R zˢ); α B z^L | none |
| B4 | V2 at L = 1 | §2 with L=1 | read 1× |
| B5/V2-FQ | fixed query | m = M_prev Q u once; zˢ⁺¹ = u + ρ φ(R zˢ + U m) | read 1×, cached |
| M1/V2 | iterative query | §2 | read L× |
| B6 | larger-rank single step | V2, L=1, larger r | read 1× |
| B7 | larger-state single read | V2, L=1, r_m > r | read 1× |
| B8 | unshared unrolled reader | §2 with per-step (R_s, U_s, Q_s), fixed L | read L× |
| B9 | double BA (d_in = d_out) | α B A φ(B A x); skip form α[B A x + η B A φ(B A x)], η=1 | none |
For B9 in the non-skip form, B = 0 makes every first-order gradient zero, so the branch never starts
(plan §1.3). B9 therefore uses either the skip form with A ≠ 0, B = 0 (the `B A x` term gives B a gradient),
or the non-skip form with nonzero A, B and an outer scalar gate `g` initialised to 0 (`y = W x + g α B A φ(B A x)`).

## 7. Accounting (plan §10)
- Params V2: `r(d_in + d_out) + r r_m (Q, K, V, U: 4 r r_m) + r² (R)` → `r(d_in+d_out) + 5r²` when r_m = r;
  +2 gate scalars (+2 r for input gates).
- MACs per token: `r(d_in + d_out)` + reader `L (r r_m [Q] + r_m² [Mq] + r r_m [U] + r² [R])`
  + writer `2 r r_m [K, V] + 2 r_m² [M̄k, outer]` when written; MAC conventions: 1 MAC = 2 FLOP.
- State bytes: `N_modules × batch × r_m² × bytes_per_element` (+ O(r) working state); excludes backbone KV.

## 8. Contraction (plan §6.3)
`F(z) = u + ρ φ(H z)`, `H = R + U M Q`; `c = |ρ| L_φ ‖H‖₂ < 1` ⇒ unique fixed point and
`‖z_s − z*‖ ≤ e_s / (1 − c)`, `e_s = ‖F(z_s) − z_s‖`. Use the bound `c ≤ |ρ| (‖R‖₂ + ‖U‖₂‖M‖₂‖Q‖₂)`
when H is not formed. The bound says nothing about task correctness.
