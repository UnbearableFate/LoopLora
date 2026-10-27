# Study LSQ — least-squares and repeated-scan stores for the learned LLM writer (WP-L7, F4 of the follow-up plan), pre-registered

Registered 2026-10-27 11:10 JST by the orchestrator, before any LSQ run (smoke included), any RFX or LSQ result and,
as DPFP Amendment A1 requires, any DPFP result. Drafted 09:56 JST by sn-llm (WP-L7 step 1, no compute). The study is
**conditional**: for each rank, RFX's verdict on that rank's R checkpoint (`docs/prereg/RFX.md`) decides which arms
run. Storage runs K4 and RG. Normalization runs U, which the orchestrator added at registration. Any other verdict
runs nothing. The plan's rules are kept (`plan/followup_2026-10-27.md` §3 F4, §6). Places where the plan left a choice
are marked "(operationalization)"; the draft's open points are decided in "Decisions at registration". Runs start
after the Miyabi maintenance (2026-10-28 22:00 JST), with code that implements this text (WP-L7 step 1b).

Known then: WP-L4 selected no writer (`docs/reports/P1L-read-fix.md`): best learned arm R at r = 128, val EM 33.5% (read
error 0.631, key top-1 88.5%, clean/read value-head accuracy 93.3% / 19.7%); R at r = 32 12.0% (read error 0.666);
OA 84.0% at r = 128 (read bypassed). With E = 16 <= r the written keys are linearly independent unless they collapse,
so an exactly interpolating store exists (rv-fable C3, W1; Wang, Shi & Fox 2025, test-time regression, arXiv 2501.12352;
Siems et al. 2025, DeltaProduct, arXiv 2502.10297 for repeated delta steps). The delta rule's single pass over
correlated keys leaves the cross-talk `(I + βC₊)⁻¹` (WP-L4 audit, Prop. 11); the DPFP study (`docs/prereg/DPFP.md`) tries
to remove it by changing the key geometry, LSQ by changing the store.

## Question
Does a store that solves for (or iterates towards) the least-squares interpolant of the E = 16 written pairs, read with
unit-normalized queries, remove the read distortion and give a learned writer that passes the WP-L4 gate
(val EM >= 0.80)? If RFX instead finds that normalizing the query alone rescues the deployed read: does the deployed
delta store, read with the unit query, give such a writer?

## Design
- T1 exactly as WP-L4 and DPFP: E = 16, n_rel 4, head sharing 0.3, seed 17, layer-14 `o_proj`, L = 1, batch 32 / Q 4,
  lr 3e-3 (no new search), val 1000 episodes at eval seed 4242 (indices 0..999, the WP-L4 validation), `scan_final`
  support, `train.prefetch_workers: 2`. Base objective R: W5 (pooled-input standardization, InfoNCE addressing at
  temperature 0.1, clean-value CE through the training-only head) plus the raw-read loss, all weights 1. The RFX data
  ranges are disjoint from LSQ's validation range, and RFX's η is fitted on train-split episodes only.
- **Query (every arm)**: the read uses the unit query q̂ = q / max(‖q‖₂, 1e-6), q = Q z; the deployed read is M q̂; the
  raw-read loss, the diagnostics and the exported model use it. The InfoNCE address loss already compares unit queries
  with unit keys and is unchanged.
- **Store**, M ∈ R^{r_mem × r_mem}, built once from the E = 16 final-token writes (unit keys k_j = K u_j / ‖K u_j‖,
  values v_j = V u_j, as the deployed `scan_final`); no new parameter is added (state-dict names and shapes equal WP-L4
  R's at the same rank):
  - **RG (ridge)**: M = V W Kᵀ (K W Kᵀ + η I)⁻¹ with recency weights w_j = λ^(E−1−j), W = diag(w), λ the learned scalar
    decay gate (the last write has weight one; this is the weighted least-squares minimizer, and for η → 0 with
    independent keys it interpolates every pair exactly whatever λ). Solved in float64 through the E x E system by a
    Cholesky factorization, differentiable; η is a constant, frozen below. The log10 condition numbers of the E x E key Gram
    KᵀK and of the weighted ridge system W^½ KᵀK W^½ + η I are logged at every logged step and reported at validation.
  - **K4**: the existing β = 1 delta `scan_final` repeated four times over the same ordered writes (Kaczmarz sweeps; every
    write overwrites exactly along its unit key; the learned λ decays the state before every one of the 64 writes). No
    new mathematics, no η. Four sweeps do not reach the interpolant for correlated keys (at λ = 1 the distance to it is
    non-increasing in the number of sweeps and vanishes only in the limit); this arm is the registered K = 4, not an
    approximation of RG.
  - **U (unit query only)**: the deployed decay-first delta `scan_final` of R, with its learned λ and β, unchanged. The
    only change from R is the unit query. It runs only under RFX's normalization verdict. It is not a least-squares store,
    but it is the deployable form of that verdict.
  - β (`a_beta`) is not used by the RG and K4 stores (it receives no gradient there). Off by default:
    `AdapterConfig.store` (`delta`, `ridge`, `sweep`), `store_eta`, `store_sweeps`, `unit_query`; with them off the code
    is bitwise the old path (tested: `tests/test_lsq.py`; real-model WP-L4 R reproduction: `tests/test_llm_dpfp_repro.py`).
- **η rule** (operationalization): for a rank ρ with a storage verdict, η = clip(η*, 1e-4, 1e-1). Here η* is RFX's
  fitting-set choice for the (ridge, unit q) cell of the R-r_ρ checkpoint (`rfx_eta_by_rank` of RFX's `summary.json`).
  It is copied into the release file before any LSQ training and enforced by the runner (`release_eta`). Smokes use an
  explicit `--eta` inside the bounds and are never interpreted.
- **Arms**, each one job on one node (≤ 2 h). The step count is the largest multiple of 5 whose 10%-smoke-measured
  time × 1.5 fits the job (WP-L4's `largest_schedule`: 1.5 margin, 90 s reserve, 7200 s, five evaluations, a
  1000-episode evaluation costing ten times the smoke's 100-episode maximum). It is fixed in a release file before
  training. Val EM is measured at five evenly spaced checkpoints. Only the arms triggered by RFX run:

  | arm | rank = r_mem | store | runs if RFX's verdict on R-r is | candidate order |
  | --- | ---: | --- | --- | ---: |
  | L32-U | 32 | delta (as R) | normalization | 1 |
  | L32-K4 | 32 | K4 | storage | 2 |
  | L32-RG | 32 | ridge, η from RFX R-r32 | storage | 3 |
  | L128-U | 128 | delta (as R) | normalization | 4 |
  | L128-K4 | 128 | K4 | storage | 5 |
  | L128-RG | 128 | ridge, η from RFX R-r128 | storage | 6 |

  For each rank, U and the K4/RG pair are mutually exclusive.

- References: WP-L4 R at r = 32 (12.0%) and r = 128 (33.5%), not rerun, provided `tests/test_llm_dpfp_repro.py` (the
  new code with every new option off reproduces WP-L4 R's training losses bitwise for the first 200 steps at both
  ranks) passes on the science commit; otherwise both references are rerun under the same schedule rule.
- Descriptive diagnostics at the final checkpoint, as WP-L4's audit and DPFP: read error, clean/read value-head accuracy,
  value SIR, key Gram RMS and mean max |cos|, EM/NLL with zeroed and shuffled memory, the exactness of the Gram
  reconstruction of the read (< 1e-4), and the log10 condition numbers.

## Decision rule (stated before results; DPFP's rule on LSQ's candidates)
- Candidate order: as in the arms table. Rank 32 comes before rank 128 (the order of DPFP). Within a rank, K4 comes
  before RG (operationalization: K4 has no hyperparameter and needs no RFX η). Only triggered arms are candidates.
  The writer is the first candidate with val EM >= 0.80 (>= 800 of 1000) at its final checkpoint.
- If none reaches 0.80 but the best reaches >= 0.50: retrain that arm once at E = 8 (same schedule rule, the audited
  selection lock `choice` = `retry_e8` for that arm) and accept it if val EM >= 0.80 (flagged). Ties keep the candidate
  order.
- Otherwise no writer is selected: the report gives every curve, the diagnostics and whether the exact store removed the
  read distortion and the clean/read gap. Together with DPFP: if neither DPFP nor LSQ selects a writer (or RFX did not
  trigger LSQ and DPFP selected none), the learned-writer line stops, the paper reports the negative result, and no
  further writer screen follows (plan §6; DPFP Amendment A1). This registration precedes every DPFP result and is
  triggered only by RFX's pre-stated verdict, so it is not a writer screen "after" the DPFP study.
- If a writer is selected: a new registration (as DPFP's P1L-D) runs the P1-L learned condition at the writer's rank; the
  oracle P1-L conditions keep k = 2. If DPFP and LSQ each select a writer, P1L-D uses the one at the lower rank. At the
  same rank it uses the one with the higher final val EM at E = 16, and a tie goes to DPFP, which was registered first.
  This is fixed here, before any DPFP or LSQ result. An E = 8 selection counts as its arm's rank and its E = 8 val EM.
- Registered predictions (rv-fable W1; reported, they decide nothing): at r = 128 the deployed read error is < 0.1, the
  read-head accuracy is within 5 pp of the clean head, and val EM >= 0.80 if top-1 addressing stays at about 88%; if the read
  error is < 0.1 while EM < 0.50 the fault is in the query path, not the store.

## Limits and cost
Validation only; seed 17; E = 16 (E = 8 only in the conditional retrain); no T2, reader or joint runs. Each arm job is
1 node × ≤ 2 h via `scripts/submit.py --scale-up docs/prereg/LSQ.md` (justification: WP-L4's curves were still rising at
the end of their fixed budgets, and the plan assigns DPFP and F4 arms 2 h). Estimates from WP-L4 (R at r = 128: 24,645
steps in 4,887 s, 0.196 s per step, 1000-episode evaluation 6-9 s; the ridge solve adds a 16 x 16 float64 Cholesky per
episode, K4 four times the 16 sequential writes: both a few percent of the step):

| item | node·h |
| --- | ---: |
| tests of all new code, one job (shared with the DPFP and RFX tests) | <= 0.1 |
| smokes (2400 steps, 100-episode evaluations; per arm ≈ 0.15) | 0.3 (one rank), 0.6 (both) |
| arm job (used ≈ 1.4; walltime reserved 2) | <= 2 |
| one rank with a storage verdict (K4 and RG) | ≈ 2.7 used, <= 4 reserved |
| one rank with a normalization verdict (U) | ≈ 1.4 used, <= 2 reserved |
| both ranks, storage | ≈ 5.5 used, <= 8 reserved |
| conditional E = 8 retrain of one arm | <= 2 |

Plan estimate for F4: 2-4 node·h. Smokes and tests together stay within 0.5 node·h per rank. Node·h is recorded per job.
Commands (after the release file is committed): the release holds `study: LSQ`, `E`, `arms` (RFX's `lsq_arms`, in
candidate order), `steps_by_arm`, `rfx_eta_by_rank` and `eta_by_rank` (both copied from RFX's `summary.json`, for the
ranks with RG arms) and `science_commit`. Then: `scripts/submit.py --name r-lsq-ARM --queue regular-g --nodes 1 --walltime 02:00:00
--scale-up docs/prereg/LSQ.md --commit <science_commit> -- -m looplora.llm.lsq_run --arm ARM --schedule RELEASE.json
--out RUNS/lsq/arms`. The conditional retrain adds `--e8-decision LOCK.json`, takes its own release (`E: 8`, every other
field as above, `steps_by_arm` holding only the retrained arm) and its own `--out RUNS/lsq/e8` (the E = 16 arm folder
already exists under `RUNS/lsq/arms`). Report: `docs/reports/LSQ.md` with every curve, node·h and `lsq_run.choose` over
the E16 records. Executed by a Claude Sonnet-5.5 subagent.

## Decisions at registration (orchestrator, on the draft's open points)
1. **K4 before RG within a rank: kept.** K4 is the smaller change to the deployed mechanism: it has no new operator and
   no regularizer.
2. **η from RFX: kept.** It is RFX's unweighted ridge fit on deployed WP-L4 keys, clipped to [1e-4, 1e-1]. The LSQ keys
   are learned jointly with the store and the weighted system is conditioned differently, so the log10 condition numbers
   are reported.
3. **DPFP and LSQ both select a writer:** decided above (lower rank, then higher val EM, then DPFP).
4. **Duplicate addresses: a stated limit.** Ridge averages the values of two facts with the same address instead of
   keeping the last. T1 has unique (head, relation) pairs per episode, so this never occurs here; it would for the T3
   update tasks.
5. **Arm U added** under RFX's normalization verdict (RFX "Decisions at registration" 2). Without it, a rescue by query
   normalization alone would leave the decision tree with no outcome. The learned-writer stop rule would then end the
   line without testing the indicated fix.
