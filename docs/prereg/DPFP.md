# Study DPFP — DPFP key/query features for the learned LLM writer (WP-L5), pre-registered

Registered 2026-10-27 07:22 JST at the user's request ("预注册一轮DPFP writer研究，维护后运行"), before any implementation
or run. It is an explicit exception, made by the user after WP-L4's negative result, to WP-L4's rule "no further
writer screens after WP-L4". Runs start after the Miyabi maintenance (2026-10-28 22:00 JST).

Known then: WP-L4 (merged 39f7f0b, `docs/reports/P1L-read-fix.md`): best learned arm R at r = 128, val EM 33.5%
(W5 29.9%); r = 32 arms about 12%; oracle-addressing diagnostic OA 84.0% at r = 128. In R at r = 128 the
training-only clean-value head decodes 93.3% but the deployed read only 19.7%; the learned keys are highly
correlated (WP-L4 audit of W5 at r = 32: key Gram RMS 0.33, mean max |cos| 0.85). The bottleneck is read
distortion from overlapping keys, not the output path. Literature: Schlag, Irie & Schmidhuber 2021, "Linear
Transformers Are Secretly Fast Weight Programmers" (arXiv 2102.11174, §5.4): DPFP
φ_ν(k) = [ r(x) ⊙ roll(r(x), j) ]_{j = 1…ν}, x = [k; −k], r = ReLU, dimension 2·d_key·ν; with the delta rule, the
number of associations stored before errors accumulate grows with that dimension (their Setting 1).

## Question
Does passing keys and queries through DPFP before the delta-rule write and read (a parameter-free,
higher-dimensional, nonnegative feature map that makes stored keys closer to orthogonal) remove the read
distortion and give a learned writer that passes the WP-L4 gate (val EM ≥ 0.80)?

## Design
- T1 exactly as WP-L4: E = 16, n_rel 4, head sharing 0.3, seed 17, layer-14 `o_proj`, L = 1, batch 32 / Q 4,
  lr 3e-3 (no new search), val 1000 episodes at eval seed 4242, `scan_final`, `train.prefetch_workers: 2`.
  Base objective R (W5 = standardization + InfoNCE addressing + clean-value CE, plus the raw-read loss, λ = 1).
- DPFP change: keys k = K u and queries q = Q z are mapped to k̂ = φ_ν(k)/‖φ_ν(k)‖₂ and q̂ = φ_ν(q)/‖φ_ν(q)‖₂.
  Memory M ∈ R^{r_mem × 2·r_mem·ν}; write: the existing decay-first delta rule with k̂; read: M q̂. Every
  address-based loss and diagnostic uses φ-space similarities q̂·k̂_j; the raw-read loss uses M q̂; values are
  unchanged. Deviation from Schlag et al.: L2 instead of sum normalization, which keeps the existing delta-rule
  semantics (exact overwrite at β = 1).
- Arms, each in one 1-node job ≤ 2 h, step count = the largest multiple of 5 whose 10%-smoke-measured time × 1.5
  fits the job, fixed before training; val EM at five evenly spaced checkpoints:
  - **D32-3**: r = r_mem = 32, ν = 3 (key dimension 192);
  - **D128-1**: r = r_mem = 128, ν = 1 (256);
  - **D128-3**: r = r_mem = 128, ν = 3 (768).
- References: WP-L4 R at r = 32 (12.0%) and r = 128 (33.5%), not rerun, provided a compute test shows that the new
  code with DPFP off reproduces WP-L4 R's training losses bitwise for the first 200 steps at both ranks;
  otherwise both references are rerun under the same schedule rule.
- Descriptive diagnostics at the final checkpoint, as the WP-L4 audit: key Gram RMS and mean max |cos| in the
  original and φ space, read error, clean/read value-head EM, value SIR, EM/NLL with zeroed and shuffled memory.

## Decision rule (stated before results)
- Candidate order: r = 32 first (D32-3), then r = 128 (D128-1 < D128-3). The writer is the first candidate with
  val EM ≥ 0.80 at its final checkpoint.
- If none reaches 0.80 but the best reaches ≥ 0.50: retrain that arm once at E = 8 (same schedule rule) and accept
  it if val EM ≥ 0.80 (flagged).
- Otherwise no writer is selected: the report gives every curve, the diagnostics, and whether DPFP reduced key
  overlap and the clean/read gap; the paper reports the negative result; no further writer screen follows from
  this study.
- If a writer is selected: a new registration (P1L-D) runs the P1-L learned condition at the writer's rank (same
  readers, task, seeds, endpoints and Pilot-2 rule as the oracle condition; a test range not used by any earlier
  evaluation; 95% CIs for this single condition). The oracle P1-L conditions keep k = 2 (Amendment B5). P3 is
  reconsidered only if P1L-D passes G1-L and G2-L, through its own registration.

## Limits
Validation only; seed 17; no T2, reader or joint runs. Each job ≤ 1 node × 2 h via
`scripts/submit.py --scale-up docs/prereg/DPFP.md` (justification: WP-L4's curves were still rising at the end of
fixed budgets, and the user requested this study); tests and smoke ≤ 0.5 node·h; node·h recorded per job.
Executed by a Claude Sonnet-5.5 subagent (codex workers closed 2026-10-27). Report: `docs/reports/DPFP.md`.
