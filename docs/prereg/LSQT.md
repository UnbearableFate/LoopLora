# Study LSQT: trained least-squares stores at r = 128, from scratch (WP-L12, G5 of `plan/followup_2026-10-29.md`), pre-registered

Registered 2026-10-29 13:13 JST by the orchestrator, before any implementation, smoke or run. The user approved G5 with both designs on 2026-10-29
("OK，按你的建议启动G5，两个方案都试一下"). Drafted by sn-llm (Claude Sonnet 5.5; WP-L12 step 1, wp/llm `f67ea82`).
Source of the design: `docs/review/rv-fable_2026-10-29.md` proposal W2. It reuses `docs/prereg/LSQ.md` and `src/looplora/llm/lsq.py`
(written and tested on a tiny model, never run on a real model; RFX triggered no LSQ arm). The sister study SQ2 is an independent
registration. Both are validation screens; neither reads a test range. The orchestrator's decisions are at the end.

## What was known at drafting
- WP-L4 (`docs/reports/P1L-read-fix.md`): R at r = 128 reaches val EM 33.5 % (normalized read error 0.631, clean/read value-head 93.3 % / 19.7 %); the oracle-addressing
  arm OA 84.0 %. With E = 16 <= r an exactly interpolating store exists.
- RFX (`docs/reports/RFX.md`, evaluation only, frozen WP-L4 checkpoints): at R-r128 the ridge store with the learned query gives EM 38.8 % (+5.0 pp), the delta store
  with the true key 39.9 %, and the ridge store with the true key 84.2 % (the clean-value level). Neither factor alone rescued; the verdict was "store+query", which
  triggered no LSQ arm. With the learned query the best ridge eta was 1 (a regularized, no longer interpolating store); with the true key 1e-6. The learned keys' Gram is badly
  conditioned (log10 condition number median 2.4 at r = 128).
- DPFP and DPFP-X (`docs/reports/DPFP.md`, `docs/reports/DPFP-X.md`): the best arm D128-3 (DPFP features, nu = 3, r = 128) reaches 60.8 %, read error 0.363, read-value
  head 51.0 %. DPFP-X's r = 256 arm reached 66.2 % with read error 0.397; neither DPFP-X arm reached 0.80. The learned-writer line is registered as stopped.
- RFX could not see co-adaptation: it swapped the store on checkpoints trained with the delta rule. LSQT trains the writer and the query with the exact store in the
  loop, which RFX could not do.

## Question
Trained jointly with the query, does a least-squares store (ridge at two regularization strengths, or four Kaczmarz sweeps) remove the read distortion and give a learned
writer that passes the WP-L4 gate (val EM >= 0.80), or at least clearly beat R at r = 128 (33.5 %)?

## Design (LSQ.md at r = 128, from scratch, with the changes listed below)
Everything is as `docs/prereg/LSQ.md` "Design" unless stated:
- T1 as WP-L4: E = 16, n_rel 4, head sharing 0.3, seed 17, layer-14 `o_proj`, L = 1, batch 32 / Q 4, lr 3e-3 (no new search), 1,000 validation episodes at eval seed 4242,
  `scan_final` support, `train.prefetch_workers: 2`, base objective R (W5 plus the raw-read loss, all weights 1), Qwen3-0.6B-Base. From scratch: no checkpoint is loaded.
  The training data stream is WP-L4's (`derive_seed(17, 'llm', 'writer-data')`).
- Query (every arm): unit L2 normalization of the learned query at read, as LSQ (`unit_query`); the deployed read, the raw-read loss, the diagnostics and the export use it.
- Stores (`lsq.py`, `core/memory.py::ridge_final`, `sweep_final`, unchanged): M in R^{128 x 128} built once from the E = 16 final-token writes (unit keys, values V u), no new
  parameter (state-dict names and shapes equal WP-L4 R's at r = 128):
  - **RG**: recency-weighted ridge M = V W Kᵀ (K W Kᵀ + eta I)⁻¹ with weights lam^(E-1-j) (lam the learned scalar decay gate), float64 Cholesky, differentiable; eta is a constant.
  - **K4**: the beta = 1 delta `scan_final` repeated four times (Kaczmarz sweeps), learned decay gate, no eta.
  - **U** (control only, below): R's deployed decay-first delta store with the unit query.
- Arms (candidate order is the order of `LSQ.md`: K4 before RG, then the smaller eta first):

  | arm | store | eta | role | candidate order |
  | --- | --- | ---: | --- | ---: |
  | `LT-K4` | four sweeps | none | arm | 1 |
  | `LT-RG001` | ridge | 1e-2 | arm | 2 |
  | `LT-RG1` | ridge | 1 | arm | 3 |
  | `LT-U` | delta (as R), unit query | none | control, optional (see "Budget") | - |

- **eta, fixed here, before any LSQT run** (fable W2): 1 and 1e-2. They bracket the two regimes that RFX measured on R-r128: eta = 1 is RFX's own best value with the learned query
  (a regularized store, a shrunk read that the learned V can rescale; the weighted ridge system has condition number at most about 17 whatever the keys), and 1e-2 is a near
  interpolating store (condition number up to about 1,600, for the badly conditioned learned keys). The smokes use the same values. The log10 condition numbers of the key Gram and of the
  weighted ridge system are logged at every logged step and reported.
- Step counts: the DPFP budget rule at r = 128 (WP-L4's `largest_schedule`): the largest multiple of 5 whose 10%-smoke-measured time x 1.5 fits one 1-node 2 h job (1.5 margin,
  90 s reserve, 7200 s, five evaluations, a 1,000-episode evaluation costing ten times the smoke's 100-episode maximum). Smokes: 2,400 steps per arm, 100-episode evaluations, timings
  and memory read only. The step count of each arm is frozen in a release file before training (`prepare_dpfp_release.py` pattern; reload check `scripts/check_smoke_reload.py` first).
  Arms may get different step counts (K4 does four times the sequential write work of the other stores); the report then also gives val EM at matched fractions of training.
  Val EM is recorded at five evenly spaced checkpoints.
- References (descriptive): R-r128 33.5 % (WP-L4, not rerun, provided `tests/test_llm_dpfp_repro.py` passes at the science commit) and D128-3 60.8 % (DPFP).
- Diagnostics at the final checkpoint (descriptive, as DPFP-X): read error, clean- and read-value head accuracy, value SIR, key Gram RMS and mean max |cos|, the exactness of the Gram
  reconstruction of the read (< 1e-4), the log10 condition numbers, and EM/NLL with zeroed and shuffled memory (the audit script of DPFP, extended to the LSQ exports).

## Decision rule (stated before results)
Outcomes, each arm judged at its own final checkpoint, 1,000 validation episodes (standard error of one EM about 1.5 pp, of a difference about 2.2 pp):
1. **Writer candidate (flagged):** any arm with val EM >= 0.80. If several, the highest EM (ties in candidate order). Flagged because validation was used to select earlier arms on this line.
   No E = 8 retrain follows. A confirmation needs its own registration.
2. **Otherwise, by the best RG arm (`LT-RG001` or `LT-RG1`), with R-r128 = 33.5 %:**
   - >= 48.5 % (R + 15 pp): flagged as fable's trigger for W3 (ridge + DPFP, learned feature map). W3 is not part of this registration and needs its own.
   - 40 % to < 48.5 %: neither trigger nor closure; the report gives every curve and diagnostic.
   - **< 40 %: falsified (fable's falsifier). The trained-LSQ line closes on trained evidence.**
3. The comparisons of RG with R-r128 and with D128-3 (60.8 %) are descriptive. The learned-writer line stays stopped unless outcome 1 holds.

Registered predictions (fable W2, LSQ.md; reported, they decide nothing): RG >= 50 % with read error <= 0.35; K4 between R-r128 and RG; at r = 128 read error < 0.1 and read-value
head within 5 pp of the clean head; if the read error is < 0.1 while EM < 0.50, the fault is in the query path, not the store.

## Deviations from LSQ.md and from the task (each flagged for review)
1. **Unconditional arms.** LSQ's trigger (RFX's `lsq_arms`) is replaced by the user's approval. RFX's verdict was store+query, which triggered nothing; LSQT tests the case that RFX could
   not see.
2. **eta from the task, not from RFX.** LSQ takes eta = clip(RFX eta*, 1e-4, 1e-1) with RFX's value in the release. RFX produced none (no storage verdict). eta = 1 and 1e-2 are fixed here
   (fable). eta = 1 lies outside LSQ's clip `ETA_BOUNDS`; the bounds are widened to [1e-4, 1] for the LSQT arms only (the LSQ arms and their config hashes stay as they are). The release names the etas.
3. **Rank 128 only, rank 32 not run.** The LSQ registry has six arms; LSQT adds the arms above with their own names (`lsq_release`/`lsq` registry extension, the old arms untouched).
4. **The E = 8 retrain rule of LSQ is dropped** (outcome 1 flags at 0.80 and nothing is retrained), as in DPFP-X; the task says a candidate's confirmation needs its own registration.
5. **Selection by P1L-D tie rules is dropped** (there is no writer selection here, only a flag).
6. **Optional U control.** LSQ registered U under RFX's normalization verdict. Here it is a control for the unit query (every LSQT arm uses it and R-r128 does not): without it, a gain over
   R-r128 cannot be split between the store and the unit query. RFX measured the unit query on frozen delta stores (EM at most +0.2 pp) but not in training. U runs only if the
   orchestrator confirms the budget at release time, before any LSQT result (see "Budget"). If it does not run, the report says so and makes no store-versus-query attribution beyond K4 against RG.
7. **Release and runner** carry `study: LSQT`, the three (or four) arms, their etas and steps (new check, `lsq_run` keeps its LSQ checks).
8. **Extra checks (not in LSQ.md):** the frozen eta is enforced by the runner as `release_eta` was; the audit script is extended to LSQ exports.

## Implementation (step 2, after the orchestrator's OK; nothing written yet)
Reuse `lsq.configuration` (R plus `unit_query`, `store`, `store_eta`, `store_sweeps`, the `lsq` block) and `LSQExperiment` for the new arm names; eta bounds as above; `lsq_run` and a DPFP-style
release (`scripts/prepare_dpfp_release.py` pattern, steps from smokes, science commit, config hashes) with `--scale-up docs/prereg/LSQT.md`; `scripts/check_smoke_reload.py` and
`scripts/audit_dpfp.py` take the LSQ exports. Tests on compute: the new configs, LSQ's existing configs and the WP-L4 and DPFP config hashes unchanged (hash pins as in
`tests/test_llm_dpfp_x.py`), `tests/test_lsq.py` and `tests/test_llm_dpfp_repro.py` (every new option off reproduces WP-L4 R bitwise for the first 200 steps at both ranks).
`pyproject.toml` and `uv.lock` are not changed.

## Limits and budget
Validation only, seed 17, E = 16; no test or confirmation data is read; no reader or joint runs. Each arm job is 1 node x <= 2 h, `scripts/submit.py --scale-up docs/prereg/LSQT.md`
(justification: DPFP's arms used 78 to 82 min and DPFP-X's 77 to 82 min of the 2 h cap, and WP-L4's curves were still rising at the end of their budgets). Estimates from the
measured step times (R at r = 128: 0.196 s per step; D128-3: 0.215 s; DPFP-X: 0.199 and 0.220 s; 1,000-episode evaluation 6 to 9 s; the ridge solve is a 16 x 16 float64 Cholesky per
episode; K4 repeats the 16 sequential writes four times, so its step time is unknown until its smoke):

| item | node·h |
| --- | ---: |
| tests of the new code and the unchanged-path checks, one job | <= 0.15 |
| smokes, 2,400 steps (about 0.12 to 0.15 per arm) | 0.4 (three arms), 0.55 (with U) |
| reload checks (about 25 s each) | 0.03 |
| arm job (the step rule targets about 80 min used; 2 h reserved) | about 1.35 used, <= 2 reserved |
| two RG arms | 3.2 with smokes and tests (fable's figure), <= 5 reserved |
| three arms (K4, RG001, RG1) | about 4.6 with smokes and tests, <= 6.5 reserved |
| optional U control | + about 1.5 |
| audit of the final exports (about 1 min per arm) | 0.05 |

The registered set is the three arms (about 4.6 node·h). Tests, smokes, reload checks and audit together stay <= 0.8 node·h. Node·h is recorded per job. Jobs of the arms run in parallel
(6-job cap). Executed by sn-llm (Sonnet-5.5) on wp/llm; reports `docs/reports/LSQT.md` and `docs/reports/LSQT-state.md`.

## Decisions at registration (orchestrator)
1. **Deviations 1–8 are accepted as written.**
2. **The U control is conditional.** It runs only after the three registered arms (`LT-K4`, `LT-RG001`, `LT-RG1`) have
   finished, and only if the best of them reaches val EM ≥ 40 %. Below 40 % the trained-LSQ line closes, so there is no gain
   to attribute.
   - If U runs, it uses the same release rule (its own smoke, frozen steps, at most 2 h) and the same audit.
   - Its result is descriptive only: it splits a gain over R-r128 between the store and the unit query.
   - It enters neither outcome 1 nor outcome 2. If U itself reaches 0.80, that is reported and the orchestrator decides.
   - If U does not run, the report says so and makes no store-versus-query attribution beyond K4 against RG.
3. **Stops.**
   - Step 2 ends with the release committed: three arms, η, steps, the science commit, config hashes, this file's blake2b
     and the reload checks. STOP there for the orchestrator's OK.
   - Step 3 runs the three arms in parallel, then the audit and the decision, and STOPs before U.
