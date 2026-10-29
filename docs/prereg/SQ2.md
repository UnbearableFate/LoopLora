# Study SQ2: storage x read-coefficient supervision, continued from D128-3 (WP-L12, G5 of `plan/followup_2026-10-29.md`), pre-registered

Registered 2026-10-29 13:13 JST by the orchestrator, before any implementation, smoke or run. The user approved G5 with both designs on 2026-10-29
("OK，按你的建议启动G5，两个方案都试一下"). Drafted by sn-llm (Claude Sonnet 5.5; WP-L12 step 1, wp/llm `f67ea82`).
Source: `docs/review/rv-astra_2026-10-29.md` proposal 4 (a new hypothesis informed by Q-Delta, not its implementation). The
sister study LSQT is an independent registration. Both are validation screens; neither reads a test range. The
orchestrator's decisions are at the end.

## What was known at drafting
- D128-3 (DPFP, nu = 3, r = 128, key dimension 768, E = 16; `docs/reports/DPFP.md`): val EM 60.8 % (608 of 1,000), normalized read error 0.363, read-value head 51.0 %
  against a clean-value head of 90.6 %, value SIR 8.0, after 22,370 steps; the curve had flattened (+0.6 pp over the last interval).
- RFX (`docs/reports/RFX.md`, evaluation only, WP-L4 checkpoints): neither the exact (ridge) store alone nor the true key alone rescued the read; both together read the written value exactly.
  astra reads this as coupled storage and query alignment, and notes that a high key top-1 does not mean the coefficient mixture of the read is right. DPFP-X (r = 256: 66.2 %, read error 0.397)
  raised EM without lowering the read error.
- The exact coefficients of the deployed read, Mq = sum_i d_i v_i, are already computed by `read_fix.store_read` / `gram_read` (delta) and `memory.ridge_coefficients` (ridge) and logged as
  diagnostics (`target_d`, value SIR, parallel interference).

## Question
Continued from the same D128-3 weights, does (a) replacing the delta store by the ridge store on the DPFP features, together with (b) a loss on the direction of the read coefficients, lower the
read distortion and raise val EM, and is the combination needed?

## Design
- **Start.** The final D128-3 checkpoint (`runs/p1l/dpfp/arms/D128-3/checkpoint.pt`, step 22,370, config hash `2207a847e4967fb8e60d1663237fd979`; blake2b-128 of the file at drafting
  `b6acff8b772a692311e630293add553f`, of `writer.pt` `31d98c01c604c05a54bc20e4b4c4f436`; both digests are frozen in the release and checked by the runner). It holds the adapter, the training-only clean-value head
  and the AdamW state. Every arm starts from these weights, this head and this optimizer state.
- **Everything else fixed across arms and as D128-3** (Qwen3-0.6B-Base, E = 16, n_rel 4, head sharing 0.3, seed 17, layer 14, L = 1, r = r_mem = 128, nu = 3, key dimension 768, unit DPFP query,
  batch 32 / Q 4, AdamW lr 3e-3 constant (D128-3's lr after its 20-step warmup; no new warmup), weight decay 0.01, grad clip 1, `scan_final` support, 1,000 validation episodes at eval seed 4242,
  `train.prefetch_workers: 2`). The objective of every arm is D128-3's: W5 (InfoNCE addressing, clean-value CE) plus the raw-read loss with weight 1. The old raw-read loss stays in every arm.
- **The 2 x 2** (storage x loss; one change from the continued D128-3 per factor):

  | arm | storage | losses |
  | --- | --- | --- |
  | `S-DE` | delta, as trained (the continued control) | existing |
  | `S-DC` | delta | existing + coefficient loss |
  | `S-RE` | ridge on the DPFP features | existing |
  | `S-RC` | ridge on the DPFP features | existing + coefficient loss |

- **Ridge on DPFP features.** M = V W Φᵀ (Φ W Φᵀ + eta I)⁻¹ over the E = 16 unit DPFP keys (width 768) and the raw values (width 128), recency weights lam^(E-1-j) of the learned decay gate, float64 Cholesky,
  differentiable; M has shape [128, 768] as D128-3's state. The read is M q with the unit DPFP query. The store ignores the write-strength gate beta (it gets no gradient there). Last-write semantics for duplicate
  addresses are not defined by the ridge store (it averages the values of two facts with the same address); T1 has unique (head, relation) addresses per episode, so this does not occur in this study.
- **Coefficient loss.** With d [B, Q, E] the exact read coefficients of the deployed store (`store_read`, float64, keeping the graph) and e_t the basis vector of the true fact among the E = 16 written facts:
  L_coef = mean over queries of || d / (||d|| + eps) - e_t ||², eps = 1e-6; weight w_c (fixed below). Added to the existing loss.
- **Samples and schedule.** All four arms train the same N steps on the same samples: a fresh training stream `derive_seed(17, 'llm', 'sq2-continuation')`, batch 32 (not a replay of D128-3's first N
  batches), the same optimizer settings. Evaluation (1,000 validation episodes) at step 0 (before any update; the arm's own store on the D128-3 weights) and at the five evenly spaced checkpoints N/5, ..., N.
  Check at step 0: `S-DE` reproduces D128-3's recorded final validation exactly (608 correct, the same NLL), otherwise the continuation is void.

## Fixed before any validation result, from training-only numerics
One compute job at the D128-3 checkpoint on 64 batches (32 episodes, Q = 4) of the **train** split from a separate stream (`derive_seed(17, 'sq2', 'numerics')`); no validation or test episode is used. Its output is frozen in the release.
- **eta (ridge arms).** The ridge store over the checkpoint's own written DPFP keys and values and its learned query; eta = the value of RFX's grid (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10) with the lowest mean normalized
  read error on those episodes (ties: the larger eta), clipped to [1e-4, 1]. The curve is reported.
- **w_c (coefficient arms).** The median over the 64 batches of ||grad L_read|| / ||grad L_coef|| (gradients with respect to the trainable adapter parameters, delta store, at the checkpoint; L_read the raw-read loss),
  rounded to the nearest power of two and clipped to [1/16, 16]. The same w_c is used in `S-DC` and `S-RC`. The ratio under the ridge store is reported, not used.
- Also reported from this job (not decisions): the log10 condition numbers of the key Gram and of the weighted ridge system, the coefficient cancellation and the query norms.

## Continuation length
**N = min(6,000, the largest multiple of 5 that fits one 1 h job under the DPFP rule at 3,600 s)**, the same N in all four arms, from the smokes (600 steps per arm, 100-episode evaluations; timings and memory read
only; the slowest arm decides). 6,000 is astra's figure. Reasons: at the measured 0.20 to 0.22 s per step (WP-L4 R, D128-3, DPFP-X) an arm costs about 6,000 x 0.22 s + evaluations + load = about 0.4 node·h, four arms
about 1.6 node·h, which with tests, smokes, numerics and audit lands inside the task's 1.8 to 2.5; the rule's 1.5 margin fits 1 h for steps up to about 0.38 s per step, so the jobs need no `--scale-up`; the delta control
gained only 0.6 pp over D128-3's last 4,474 steps, so 6,000 further steps on new samples are a fair control for "more of the same". A longer continuation would not fit the budget with four arms and a reserve for a rerun.

## Primary contrast (registered before results)
**`S-RC` against `S-DE`** at the final checkpoint, 1,000 validation episodes. It **advances** only if all three hold:
1. val EM of `S-RC` is at least 10 pp above `S-DE` (at least 100 more correct of 1,000);
2. the normalized read error of `S-RC` (mean `relative_read_error` over the validation queries, as in the DPFP reports) is at most half of `S-DE`'s;
3. **value decoding is stable** in `S-RC`: no nonfinite loss or gradient during training, and the clean-value head accuracy at each of the five checkpoints is at least `S-DE`'s final clean-value head accuracy minus 5 pp.

"Advances" means: flagged for a separate three-seed confirmation registration on an untouched range with matched control and candidate training (astra's estimate 9 to 12 node·h); nothing is confirmed here. If `S-RC` also reaches
val EM >= 0.80 it is a writer **candidate**, flagged (validation was used to select D128-3), and nothing more. The other three cells, and the interaction (`S-RC` - `S-RE`) - (`S-DC` - `S-DE`) in pp of EM, are descriptive: one seed, standard
error of a difference of two EMs about 2.2 pp. No cell is promoted by being the best of four after the fact; a non-primary cell reaching 0.80 is reported, not promoted. If the primary contrast does not advance, the learned-writer line stays stopped.

Registered prediction (astra; reported, it decides nothing beyond the rule above): `S-RC` beats `S-DE` by >= 10 pp and halves the read error; neither change alone matches the combination (neither `S-RE` nor `S-DC` meets conditions 1 and 2 against `S-DE`).

## Diagnostics (descriptive, at every checkpoint and the final one)
As DPFP-X (read error, clean- and read-value head, value SIR, parallel interference, key Gram RMS and mean max |cos| in both spaces, the exactness of the Gram reconstruction), plus, logged at every logged step and reported:
the log10 condition numbers of the key Gram and the ridge system, query norms, the **coefficient cancellation** (sum_i |d_i| ||v_i|| / ||sum_i d_i v_i||, large when the coefficients cancel), the direction cosine d·e_t / ||d|| of the loss, and the
EM/NLL with zeroed and shuffled memory (the audit of DPFP, extended to these exports).

## Deviations from the task and open points (each flagged for review)
1. **Fresh samples instead of D128-3's stream.** The task says "the same samples" across arms; the same fresh stream is used for all four. Replaying D128-3's first N batches would repeat what the weights have already seen.
2. **Optimizer state is loaded for every arm and the lr stays constant 3e-3, no warmup.** `S-DE` is then exactly D128-3 continued. The ridge and coefficient arms start with AdamW moments from another objective (not reset);
   this is the same for every arm, and the step-0 and early checkpoints show the transient. The alternative, a fresh optimizer with warmup in all four arms, would make `S-DE` something other than a continuation.
3. **Step-0 evaluation** is added (not in the task) as the check of the continuation and as the descriptive "store swapped, nothing retrained" reading of the ridge arms.
4. **Ridge on DPFP features is new code**, not "run the existing LSQ config": `AdapterConfig` forbids a least-squares store with DPFP features and `ridge_final` has no feature map. K4 on DPFP features is not part of this study.
5. **eta range [1e-4, 1]** (LSQ clips to [1e-4, 1e-1]); the same grid as RFX. **w_c rule** (gradient-norm ratio) is a choice of this draft; the task only requires training-only numerics.
6. **N is at most 6,000** and may be smaller if a smoke shows a step time above about 0.38 s (then the same smaller N for every arm).
7. **Validation reuse.** The 1,000 validation episodes also served D128-3's selection and the DPFP and DPFP-X comparisons; the contrast here is within that set, so a result is a screen only.
8. **Unit query is part of the DPFP query map** (`unit_query` stays off in the config; the DPFP map normalizes). Nothing about the query or the feature width changes between arms.

## Implementation (step 2, after the orchestrator's OK; nothing written yet)
- `core/config.py` and `core/memory.ridge_final`/`adapters`: a ridge store with `dpfp_nu > 0` (a `feature_map` argument that defaults to none, so the existing paths are bitwise unchanged).
- `read_fix`: the coefficient loss behind an off-by-default config key (`read_fix.coefficient_weight`), the extra diagnostics (cancellation, direction cosine); the existing losses unchanged when it is off.
- A continuation runner (`sq2_run.py`): loads the checkpoint (adapter, value head, optimizer; `rfx.load_checkpoint` pattern), checks the config hash and file digests, takes a data seed and the step-0 evaluation,
  and a numerics job for eta and w_c. `read_fix_run.fit` gets optional arguments that default to its present behaviour.
- A release (`prepare_dpfp_release.py` pattern, `study: SQ2`): N, eta, w_c, the checkpoint digests, the science commit and the config hashes, with the reload check first.
- Tests on compute: the new configs; **every new option off reproduces the existing paths bitwise** (`tests/test_llm_dpfp_repro.py`, `tests/test_llm_dpfp.py`, `tests/test_lsq.py` and the config-hash pins of `tests/test_llm_dpfp_x.py`);
  ridge-on-DPFP exactness (the Gram reconstruction of the read < 1e-4 and the interpolation of E = 16 independent keys); the coefficient loss against a hand computation; the continuation (step-0 equality with the checkpoint, optimizer state loaded).
  `pyproject.toml` and `uv.lock` are not changed.

## Limits and budget
Validation only, seed 17, E = 16; no test or confirmation data; no reader or joint runs. Each arm job is 1 node x <= 1 h (no `--scale-up` needed at N <= 6,000 and 0.38 s per step; otherwise 2 h with `--scale-up docs/prereg/SQ2.md`).
The four arm jobs run in parallel. Estimates from the measured step times (0.196 to 0.22 s):

| item | node·h |
| --- | ---: |
| tests (new code, unchanged-path checks, continuation), one job | <= 0.15 |
| numerics job (64 train batches, gradients of two losses, the eta grid) | about 0.03 |
| smokes, 600 steps, four arms | about 0.2 |
| reload and step-0 checks | about 0.03 |
| four arm jobs, 6,000 steps (about 0.4 each) | about 1.6 |
| audit of the four final exports | about 0.05 |
| total | about 2.1 (task range 1.8 to 2.5) |

Node·h is recorded per job. Executed by sn-llm (Sonnet-5.5) on wp/llm; reports `docs/reports/SQ2.md` and `docs/reports/SQ2-state.md`.

## Decisions at registration (orchestrator)
1. **Deviations 1–8 are accepted as written.** They cover:
   - one fresh stream shared by all arms;
   - the loaded AdamW state with constant lr 3e-3;
   - the step-0 evaluation;
   - new code for ridge on the DPFP features;
   - η in [1e-4, 1] with the w_c rule;
   - N ≤ 6,000;
   - validation reuse;
   - the unit query as part of the DPFP map.
2. **The η rule and the w_c rule are accepted as written** (open points). The numerics job uses only the train split. Its
   outputs (η, w_c and the reported curves) are frozen in the release before any smoke output is read. Smokes are read for
   time and memory only.
3. **The stability definition (condition 3) is accepted as written:**
   - no nonfinite loss or gradient;
   - the clean-value head at each of the five checkpoints ≥ `S-DE`'s final value − 5 pp.
4. **Step-0 check.** If `S-DE` at step 0 does not reproduce D128-3's 608 correct and its NLL exactly, the continuation is void.
   STOP and report before any arm job runs. Run this check in the reload/step-0 job, before the arm jobs.
5. **Stops.** Step 2 ends with the release committed (N, η, w_c, checkpoint digests, the science commit, config hashes, this
   file's blake2b and the step-0 check). STOP there. Step 3 runs the four arms in parallel, then the audit and the decision,
   and STOPs.

## Clarification C1 (2026-10-29 13:43 JST, before the numerics job has run; from sn-llm's step-2 question)
- "Rounded to the nearest power of two" means rounding on the log2 scale: w_c = 2^round(log2(median ratio)), then clipped
  to [1/16, 16]. Between 2^n and 2^(n+1) the boundary is 2^n·√2.
- The reason is that the ratio of gradient norms is multiplicative.
- This is fixed before any numerics output exists. The release records the median ratio and w_c under this rule only.
