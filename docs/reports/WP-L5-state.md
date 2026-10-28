# WP-L5 and WP-L7 state: DPFP option, least-squares stores and the read-failure factorial (steps 1, 1b, 1c and 1d done; nothing executed)

Branch `wp/llm`. WP-L5 step 1 (DPFP) is `34028dd`, `742d3bc`, `8eb24ab` on master `6da6330`; master was merged in as
`77373ea` (P1-L pipeline from wp/p1l, follow-up plan), `cbc8389` (follow-up tasks) and `1485bd9` (the registered RFX and
LSQ, the step-1b task); step 1b is `17b27c2`. Step 1c (task section "Step 1c" of `docs/tasks/WP-L7_rfx_lsq.md` on master
`9c6364b`, from the static review `docs/review/static_2026-10-27/rfxlsq.md`, which found no defect in `17b27c2`) is
`0a3048a`. Step 1d (section "Step 1d" on master `4f22c3c`, from the second review `docs/review/static_2026-10-27/release.md`,
which found no defect in the release path) is the `WP-L7: step 1d ...` commit on top of `0a3048a`; neither master commit
is merged into wp/llm. **Every step-2 command must use the full hash of a commit that contains the registrations and the
step-1b, 1c and 1d code: the last `WP-L7: step 1d ...` commit on wp/llm or its merge into master (`0a3048a` breaks
entity-restricted ties by entity index and has no reload check; `17b27c2` also lacks the entity-EM guard and the two
release scripts). The step-1 commits (up to `9588998`) implement the drafts' decision rules, not the registered ones.**
After the merges the DPFP files still compile (`py_compile`, `pyflakes`), and master's `src/` changes since `6da6330` touch
only the `oracle_random` rank-128 allowance (`llm/config.py`, `llm/oracle.py`, `llm/train.py`, oracle branches only), not
the learned-writer path that the DPFP-off reproduction test covers.

Step 1 of `docs/tasks/WP-L5_dpfp_writer.md` is implemented and statically checked (`py_compile` and `pyflakes` on every
changed Python file, a torch-free dry run of the repro test's CSV parsing, and two independent read-only reviews of the
code and of the tests by Sonnet subagents; their one likely test defect and two hardening items are fixed in the second
commit). Step 1 of `docs/tasks/WP-L7_rfx_lsq.md` (RFX, LSQ, below) is statically checked the same way and was reviewed
independently twice more (LSQ core and tests; RFX hook, runner and tests) by read-only Sonnet subagents; the findings
that held (a wrong einsum in an LSQ test helper, two test tolerances, a missing digest guard, NaN-blind instrumentation
maxima, a too small default evaluation budget of the L-BFGS decoder fit, wording of the Gram notation and of the E8
release) are fixed in the `WP-L7: review fixes` commit. Step 1b (the code and tests rewritten to the registered rules, task
section "Step 1b") is statically checked the same way, and its pure decision logic (verdict tree, arm sets, release checks,
the hand-off from the RFX summary to the LSQ release) was also executed torch-free against the functions extracted from the
source. Step 1c (a fail-closed entity-restricted EM guard in the RFX run, O1; the stdlib release scripts of DPFP and LSQ,
O2; the registry and release checks they need moved into torch-free modules) is statically checked the same way and its
pure tests (`tests/test_release_scripts.py`, the guard test of `tests/test_rfx.py`) were executed torch-free with a
stand-in test runner on the login node (80 cases passed, the one test that needs torch was skipped), and killed all 27 and
all 9 deliberate mutants of the release code and the guard. Step 1d (the restricted argmax breaks ties like the full-vocabulary
one, F1; the reload check of the smoke exports, F2) is statically checked the same way; the new pure tie tests of
`tests/test_rfx.py` and the control flow of `scripts/check_smoke_reload.py` were executed torch-free against stand-ins for
torch and for the compute modules (the tie tests kill the step-1c rule and six other mutants of `entity_hits`; every check
of the script is load-bearing), which tests the logic and not torch's or the loaders' behavior.
**No torch code, `pytest` or run has been executed**: the login node forbids it and compute is unavailable until
2026-10-28 22:00 JST. Not merged, not pushed. Binding designs:
`docs/prereg/DPFP.md` and, registered 2026-10-27 11:05 JST before any RFX, LSQ or DPFP result, `docs/prereg/RFX.md` and
`docs/prereg/LSQ.md`.

## Step 2 status (compute from 2026-10-28 23:40 JST; finished 2026-10-29; commits on wp/llm)
Source commit of the runs: `efadc36436e4d29514a4ea570596925b779b8b2d` (= b79e0f8 plus one test-fixture commit); the E8 retrain ran at `72ed469` (adds the audited lock).
- **Tests** (`t-dpfp`, job 3720636, 6m19s): everything passed except three tests of `tests/test_llm_dpfp.py` (the unit-phi-keys test for D32-3
  and D128-1; the zero/shuffle controls). Cause, measured in jobs 3720796 and 3720880: at the fixture's adapter rank 4 the nonzero entries
  of phi(query) and of the stored phi keys did not overlap (3 to 6 nonzeros of key_dim 8 to 24), so `M q_hat` was exactly zero: no gradient, and
  the zero-memory control equal to the original. At rank 8 the read is nonzero. Fix: `RANK = 8` in the test file (efadc36), no assertion changed.
  `t-dpfp2` (3720890) ran the file at efadc36: all 16 passed. DPFP-off reproduction of WP-L4 R (`t-dpfp-repro`, 3720637): passed.
- **DPFP smokes** (3720940-42, 8m40s to 9m09s each; timings only): steady step 0.204 / 0.198 / 0.210 s (D32-3 / D128-1 / D128-3), peak CUDA
  allocated 3.1 / 3.5 / 4.4 GB. Reload checks (`scripts/check_smoke_reload.py`, 3721044-46, 23 s each): config equal, reload identical,
  0 prediction mismatches for all three. Release `reports/p1l-dpfp/release.json` (committed 1dd1869): E16, steps 23015 / 23690 / 22370.
- **RFX** (smoke 3720943; science 3721047, 4m52s): entity-EM guard passed, `full_hit_entity_miss` 0 in all 42 cells. Verdicts R-r128
  store+query, R-r32 store+query, W5-r32 none; summary `reports/p1l-rfx/summary.json` (379fccd); report `docs/reports/RFX.md`.
- **LSQ did not run.** `lsq_arms` in the RFX summary is empty: the registered rule triggers the least-squares arms only for a storage verdict
  (K4 and RG) or a normalization verdict (U), and neither checkpoint of R got one (store+query; W5-r32 triggers nothing). Per the
  orchestrator's decision there is no LSQ release and no LSQ run. The LSQ code, `scripts/prepare_lsq_release.py` and the LSQ tests stay in the
  tree unused; the LSQ commands and release text below are not to be executed.
- **DPFP arms** (3721087-89, about 80 min each, complete at the released steps): final val EM D32-3 39.0 %, D128-1 58.0 %, D128-3 60.8 %.
  None reaches 0.80; the best (D128-3, 608 of 1,000) is at least 0.50, so the registered rule triggered **one retrain of D128-3 at E = 8**
  (audited lock `reports/p1l-dpfp/e8-lock.json`). E8 smoke 3721493, release `reports/p1l-dpfp/release-e8.json`, reload check 3721520, science
  job 3721535 (78 min, 26065 steps): **730 of 1,000 (73.0 %), below 0.80, not accepted.**
- **Final DPFP decision (orchestrator-confirmed 2026-10-29): stop. No writer selected, P1L-D not run, negative result.** With RFX triggering no LSQ arm,
  the learned-writer line stops. Registered diagnostics: zero and shuffled memory at the final checkpoints (`scripts/audit_dpfp.py`, jobs 3721521
  and 3721773): EM at most 0.2 % in both conditions for all four runs; the originals reproduce the records exactly.
- Cost: DPFP about 6.14 node·h, RFX 0.10. Tests plus smokes used about 0.78 node·h against the registered 0.5 cap (recorded overrun, causes in
  `docs/reports/DPFP.md`). Reports: `docs/reports/DPFP.md`, `docs/reports/RFX.md`; records in `reports/p1l-dpfp/` and `reports/p1l-rfx/`; tables by
  `scripts/analyze_dpfp.py` and `scripts/analyze_rfx.py`. Nothing remains to run in this work package.

## What is implemented (WP-L5: DPFP)
DPFP is `AdapterConfig.dpfp_nu` (default 0 = off). With nu >= 1 the write key and the read query are mapped to
`phi_nu(.)/max(||phi_nu(.)||_2, eps)` and the memory is `[B, r_mem, key_dim]`, `key_dim = 2 * r_mem * nu`.

| File | Change |
| --- | --- |
| `src/looplora/core/config.py` | `dpfp_nu` (last field, memory variants only), `key_dim` property |
| `src/looplora/core/memory.py` | `dpfp_features(k, nu)`; keyword `feature_map` in `write`, `scan_prewrite`, `scan_final` (applied before the unchanged unit normalization); rectangular states (`zeros_state(..., key_dim)`, `_check_state`, `delta_update` shape checks; both scans raise `ValueError` when the state's key width differs from the written key's) |
| `src/looplora/core/adapters.py` | `key_dim` in state shapes; `MemoryAdapter.query_code` (identity when off) and `_key_features` (passes `feature_map` only when on); `_query` is now an instance method |
| `src/looplora/core/accounting.py` | MACs and state bytes use `r_mem * key_dim` (identical values when off) |
| `src/looplora/llm/inject.py`, `train.py` | zero states use `key_dim`; `dpfp_nu` read from `config['model']`; writer signature gains `dpfp_nu` only when nonzero |
| `src/looplora/llm/writer_fix.py` | recorded keys are the unit phi keys actually written (`FixBatch.keys`), original-space keys in `FixBatch.base_keys`; InfoNCE queries `query_codes` are `q_hat`; W4 + DPFP raises |
| `src/looplora/llm/read_fix.py` | `question_terms` reads `M q_hat`; `diagnostics` keeps its columns (now phi space for DPFP arms) and adds `key_gram_rms_base`, `key_gram_max_abs_base` |
| `src/looplora/llm/dpfp.py` (new) | arms `D32-3`, `D128-1`, `D128-3` (`ARMS`, `SPECS` live in `dpfp_release.py` since step 1c); `configuration(arm, steps, E=16, smoke=False)` = `read_fix.configuration('R', ...)` at rank 32/128/128 plus `model.dpfp_nu` and a `dpfp` block; `DPFPExperiment(ReadFixExperiment)` |
| `src/looplora/llm/dpfp_run.py` (new) | one arm per job: `python -m looplora.llm.dpfp_run --arm ARM --out ROOT (--steps N --smoke \| --schedule release.json) [--e8-decision lock.json]`; reuses `read_fix_run.fit` unchanged; record adds rank, nu, key_dim, state bytes, peak CUDA memory; `choose` (registered rule), `load_inference`; `scheduled_steps` (an E16 release must fix all three arms at once; an E8 release names the retrained arm) and `eligible_e8` are imported from `dpfp_release.py` |
| `src/looplora/llm/dpfp_release.py`, `scripts/prepare_dpfp_release.py` (new, step 1c) | torch-free: the arm registry, `scheduled_steps` and `eligible_e8` (moved verbatim out of `dpfp.py` and `dpfp_run.py`, so the runner and the login-node script share them); the script writes the DPFP release from the three smokes (see the commands below) |
| `scripts/check_smoke_reload.py` (new, step 1d; compute only, DPFP and LSQ) | the reload gate of one smoke arm, run before the release is committed (below): reloads `ARM/writer.pt` with the study's `load_inference`, compares configs, validates 100 episodes and compares the predictions with the smoke's last evaluation |
| `tests/test_dpfp.py`, `tests/test_llm_dpfp.py`, `tests/test_llm_dpfp_repro.py`, `tests/assets/memory_before_dpfp.py` (new) | see below |
| `docs/notation.md` (section 9), `configs/p1l/README.md` | option documented; this file |

DPFP-off is meant to be bitwise the old path: with `feature_map=None` and `dpfp_nu=0` the same tensor ops run in the
same order (`query_code` returns its input; adapters add no keyword; the key recompute is the same expression).
`llm/config.py` DEFAULTS and `resolve_config` are untouched; the DPFP keys are added after it, like `writer_fix` and
`read_fix`, so WP-L4 config hashes, signatures, checkpoints and re-analysis are unchanged. DPFP adds no parameters:
state-dict names and shapes equal WP-L4 R at the same rank (only M changes shape).

## Tests (compute only; none run yet)
- `tests/test_dpfp.py` (core, no model): `phi_nu` equals a loop reference (d in 1/5/32, nu in 1/2/3), nonnegative,
  degree-2 homogeneous, literal examples fix block order and roll direction; unit phi keys overwrite exactly at beta = 1
  (nu 1/3, lambda 1/.7); `write` with a feature map; fast scan vs reference scan vs explicit delta updates with
  gradients (cpu/cuda, random/all/initial-reset/interior-reset); **DPFP-off bitwise equality** with a frozen copy of
  the pre-DPFP `memory.py` (forward cpu+cuda, gradients cpu, fp32/fp64, with and without `feature_map=None`);
  `AdapterConfig` validation; key-width rejection in both scans; `process_sequence` and `process_sequence_fast` vs
  token-by-token feature-space equations; Gram read in phi space; accounting.
- `tests/test_llm_dpfp.py` (tiny Qwen3 on cuda, inputs standardized as the runner does): config equals WP-L4 R plus
  only the DPFP fields (3 arms x E16/E8); inconsistent arm/config rejected; W4 guard; per arm (nu 3 and 1): state/key/query shapes, unit nonnegative phi keys,
  memory = ordered delta rule over the recorded keys, InfoNCE and alignment in phi space, `M q_hat` read, gradients of
  the raw-read loss, Gram reconstruction, original- and phi-space key overlap; DPFP-off experiment wiring unchanged;
  zero/shuffle controls on the rectangular state; 5-step `fit`, `validate` (16 episodes), export and exact reload
  (and WP-L4's loader refuses a DPFP export); `choose`, `scheduled_steps`, `eligible_e8` against the prereg rule.
- `tests/test_llm_dpfp_repro.py` (real model): **200-step DPFP-off reproduction of WP-L4 R** at rank 32 and 128.
  It compares exactly (`==`) every logged field except step time at steps 1/50/100/150/200 with the committed
  `reports/p1l-read-fix/loss_curves.csv` (steps of record 23590 and 24645; the first 200 steps do not depend on
  them). Each rank trains in a fresh subprocess; run the file alone in its own job. A step-200 match implies
  the whole trajectory matched. WP-L4's rank-128 source (`be07f7f`) has `src/` identical to the base `6da6330`;
  the rank-32 source (`0b03d60`) differs only by the WP-C7 oracle writer and the A2 rank parameter (diff read, not
  run), which leave R's learned-writer path unchanged. If either rank differs, first check for an environment
  change since WP-L4 (torch/CUDA/driver, maintenance); per the prereg both references are then rerun under the
  same schedule rule.

## Commands for step 2 (after 2026-10-28 22:00 JST; run from `/work/xg24i002/x10041/LoopLora`)
`REF` is any committed ref (`wp/llm`, or the commit after the orchestrator merges); submit.py snapshots it by `git archive`.
Measured WP-L4 comparables: 69 tests 182 s; one rank-128 smoke job (2400 steps, two arms in parallel) 510 s. The
WP-L7 tests are not measured yet (about 100 more tests, most on the tiny model; a few `run`/`run_checkpoint` smokes; the
81 cases of `tests/test_release_scripts.py` write and read small JSON files, run two subprocesses and import torch once,
so they take seconds): the 40-minute walltime of command 1 is an estimate with margin, shrink it from the first
measurement.
```sh
REF=wp/llm; RUNS=/work/xg24i002/x10041/LoopLora/runs/p1l/dpfp

# 1. DPFP and WP-L7 (LSQ, RFX, release scripts) tests plus regression tests of every touched module
scripts/submit.py --name t-dpfp --queue regular-g --nodes 1 --walltime 00:40:00 --commit "$REF" --wait -- \
  -m pytest -q tests/test_dpfp.py tests/test_llm_dpfp.py tests/test_lsq.py tests/test_llm_lsq.py tests/test_rfx.py \
  tests/test_llm_rfx.py tests/test_release_scripts.py tests/test_llm_read_fix.py tests/test_llm_writer_fix.py \
  tests/test_llm_train.py tests/test_llm_prefetch.py tests/test_llm_b8.py tests/test_accounting.py \
  tests/test_scan_final.py tests/test_reference_equivalence.py tests/test_state_lifecycle.py tests/test_precision.py \
  tests/test_gradient_flow.py tests/test_loop_invariants.py tests/test_causality.py tests/test_stopping.py \
  tests/test_learned_halting.py tests/test_contraction.py tests/test_theory_examples.py

# 2. DPFP-off reproduction of WP-L4 R (own job: fresh CUDA state; ~2 x 2-3 min)
scripts/submit.py --name t-dpfp-repro --queue regular-g --nodes 1 --walltime 00:25:00 --commit "$REF" --wait -- \
  -m pytest -q tests/test_llm_dpfp_repro.py

# 3. 10% smokes, time and memory only: 2400 steps, 100-episode evals (WP-L4 A2 convention); one job per arm
for ARM in D32-3 D128-1 D128-3; do
  scripts/submit.py --name s-dpfp-$ARM --queue regular-g --nodes 1 --walltime 00:25:00 --commit "$REF" -- \
    -m looplora.llm.dpfp_run --arm $ARM --steps 2400 --smoke --out $RUNS/smoke
done
```
Do not read smoke scores (the smoke evaluates 100 episodes of the validation range the science arms use). Print only timings:
```sh
python3 -c "
import json
for a in ('D32-3','D128-1','D128-3'):
    r=json.load(open('$RUNS/smoke/%s/record.json'%a)); t=r['training']
    print(a,r['status'],t['steady_step_mean_seconds'],t['step_p95_seconds'],
          max(v['seconds'] for v in t['metrics'].values()),r['memory'])"
```
Release and reload check (later, after the three smokes have completed; nothing is committed before the check passed).
**1. Release, on the login node** (no torch, no GPU: the script is standard library only, so the system `python3` with
`PYTHONPATH=src` is enough). It writes the file; do not commit it yet:
```sh
PYTHONPATH=src python3 scripts/prepare_dpfp_release.py --root $RUNS/smoke --out reports/p1l-dpfp/release.json
```
It reads only identity and timing fields of `$RUNS/smoke/ARM/record.json` (never a score or an em), requires complete,
clean 10% smokes of the right arm at one common full 40-hex commit (the `science_commit`), and writes `study: DPFP`,
`E: 16`, `steps_by_arm` (WP-L4's rule: the largest multiple of 5 that fits the 2 h job, from the smoke's step time and the
ten-fold 100-episode evaluation time), the five evaluation steps, the sizing and the smoke timings. Before writing, it
checks the release with the runner's own `dpfp_release.scheduled_steps`, and it refuses to overwrite an existing file
(`FileExistsError`; a committed release is replaced by nobody).
**2. Reload check, one short compute job per smoke arm** (step 1d; about 3 minutes and 0.05 node-hours each, not
measured; `scripts/check_smoke_reload.py`, choice 10 below): the release script cannot load an export, and three 2 h jobs
must not depend on one that cannot be reloaded.
```sh
REL=/work/xg24i002/x10041/LoopLora/reports/p1l-dpfp/release.json
SC=$(python3 -c "import json;print(json.load(open('$REL'))['science_commit'])")
for ARM in D32-3 D128-1 D128-3; do
  scripts/submit.py --name c-dpfp-$ARM --queue regular-g --nodes 1 --walltime 00:10:00 --commit "$SC" -- \
    scripts/check_smoke_reload.py --release $REL --root $RUNS/smoke --arm $ARM
done
# when the jobs have ended (no score is printed; a missing file is a failed check, the job log says why):
python3 -c "
import json
for a in ('D32-3','D128-1','D128-3'):
    print(json.load(open('$RUNS/smoke/export-check-%s/reload-check.json'%a)))"
```
**3. Commit the release only if all three passed.** If one failed, do not commit: delete the uncommitted release file (and
`$RUNS/smoke/export-check-ARM` before a rerun), fix the cause (a smoke whose export does not reload is rerun in a new smoke
root and the release is written again). Science, per arm: `scripts/submit.py --name r-dpfp-ARM --queue regular-g --nodes 1
--walltime 02:00:00 --scale-up docs/prereg/DPFP.md --commit <science_commit> -- -m looplora.llm.dpfp_run --arm ARM
--schedule RELEASE.json --out $RUNS/arms`.
The conditional E8 retrain adds `--e8-decision LOCK.json` (the audited lock whose `choice` is `retry_e8` for that arm) to
four things: the arm's own E8 smoke (`... dpfp_run --arm ARM --steps 2400 --smoke --e8-decision LOCK.json --out
$RUNS/smoke-e8`; the release script requires the smoke's `task.E` to equal the release's `E`), its own release
(`prepare_dpfp_release.py --root $RUNS/smoke-e8 --out reports/p1l-dpfp/release-e8.json --e8-decision LOCK.json`: `E: 8`,
`steps_by_arm` naming that arm only), its own reload check (`check_smoke_reload.py --release .../release-e8.json --root
$RUNS/smoke-e8 --arm ARM`) and its own `--out` root for the science run (the E16 arm folder `$RUNS/arms/ARM` already exists
and the runner refuses to reuse it).

## Choices where the prereg was silent
1. Normalization is `x / max(||x||_2, eps)` with the existing eps = 1e-6, at write and read. phi_nu is quadratic, so a
   feature vector below eps (keys of norm about 1e-3 or less) is shrunk instead of unit and loses the exact overwrite for
   that key. The arms avoid it through WP-L4's input standardization (unit-scale keys); the tiny-model tests standardize
   first, because the raw activations of the tiny random model are about 0.03.
2. Diagnostic columns keep their names and are phi space for DPFP arms (`key_gram_rms`, `key_gram_max_abs`, key top-1
   and margin, Gram read); the original space (unit projected key before phi) is added as `*_base`. WP-L4's audit table
   "Mean max abs cos" is the mean of `key_gram_max_abs`, so the same column is compared.
3. DPFP applies to the fact_final writers (W1-W3, W5, R); the paired W4 write raises. B8/unshared reads work but are
   unused here.
4. The DPFP arm label is the experiment's `read_arm`: it names the `arm` sample column, export, and signature
   (`read_fix`). Base objective, standardization, lambda/beta init, value head, InfoNCE temperature .1, weights 1 and
   the optimizer are exactly WP-L4 R's.
5. `core/accounting.py` was generalized to `key_dim` (values unchanged when off). `core/stopping.py` and
   `core/contraction.py` still require square memory and raise on DPFP states; the DPFP path does not use them.

## Open questions and risks for the orchestrator
- Budget: prereg caps tests plus smokes at 0.5 node-hours. Estimated from WP-L4: tests 0.06, repro about 0.1, three
  smokes 0.3-0.4, total about 0.5-0.6. If it binds, shorten the smokes (e.g. 1200 steps) or the test list, and record
  the smoke fraction in the release as WP-L4 did; step time, not accuracy, is what they measure.
- The step-count rule is WP-L4's `read_fix_run.largest_schedule` (1.5 margin, 90 s reserve, 7200 s, five evaluations,
  1000-episode evaluation time = 10 x the smoke's 100-episode maximum). `scripts/prepare_dpfp_release.py` and
  `prepare_lsq_release.py` must run on the login node, where `read_fix_run` cannot be imported (it imports torch), so
  `looplora.llm.release.largest_schedule` is a stdlib copy of the rule; `test_the_sizing_rule_is_wp_l4s` (compute only)
  compares the copy with WP-L4's function and `tests/test_release_scripts.py` checks the sizing against an independent
  transcription of the rule.
- Comparing key overlap with WP-L4: phi features are nonnegative, so phi-space cosines have a positive floor. For random
  Gaussian keys (E = 16, numpy check) the phi-space RMS cosine is about 0.14 (D32-3), 0.12 (D128-1), 0.11 (D128-3)
  against 0.18, 0.09, 0.09 in the original space, and mean pairwise cosine is about 0.10-0.11 versus 0. Compare WP-L4's
  key Gram numbers with the `*_base` columns; the phi-space columns measure what the memory actually addresses.
- D128-3 stores 128 x 768 fp32 states (393 KB per episode, 128 query copies per batch); memory should be fine on a
  GH200 but is measured in the smoke record (`memory`).

## WP-L7 steps 1, 1b, 1c and 1d: RFX and LSQ (implemented to the registered rules and statically checked; nothing executed)
Task `docs/tasks/WP-L7_rfx_lsq.md`; designs `docs/prereg/RFX.md` and `docs/prereg/LSQ.md` (registered; they differ from the
step-1 drafts, which the step-1b commits supersede). **RFX** is evaluation only: on the three frozen WP-L4 checkpoints
(R-r128, R-r32, W5-r32) it replaces the read at each question's final token by delta/ridge x raw/unit/true-key reads (each as
is and with a fitted scalar gain), the clean value and zero, decodes through the LM path (full-vocabulary and entity-restricted
EM), the clean-value head and a refit linear decoder, and tabulates normalized read error, head accuracy, EM, NLL, write age,
write order (four permutations), conditioning, every rescue flag and the registered verdict, first that applies: **storage**
(ridge/raw/fit rescues, or ridge/unit/fit rescues while delta/unit/fit does not), **normalization** (delta/unit/fit),
**query** (delta/true/fit), **store+query** (ridge/true/fit), **decoder** (refit decoder against the entity-restricted baseline
EM), **none**. **LSQ** is the conditional training study, per rank by the verdict on that rank's R checkpoint: storage runs
K4 (four beta = 1 sweeps) and RG (recency-weighted ridge), normalization runs U (R's delta store with the unit query), any other
verdict runs nothing; the arms RFX triggers are `lsq_arms` in `summary.json`.

| File | Change |
| --- | --- |
| `src/looplora/core/config.py` | `AdapterConfig.store` (`delta` default, `ridge`, `sweep`), `store_eta` (1e-3), `store_sweeps` (4), `unit_query` (False): last fields, validated (a non-delta store needs a memory variant, scalar gates and no DPFP; `unit_query` and `dpfp_nu` are exclusive) |
| `src/looplora/core/memory.py` | `ridge_final` (recency-weighted ridge least squares, float64 Cholesky through the n x n system, differentiable), `sweep_final` (beta = 1 `scan_final` repeated), exact read coefficients `ridge_coefficients`, `sweep_coefficients`, `gram_conditions` (finite condition numbers; smallest eigenvalue floored at 1e-30) |
| `src/looplora/core/adapters.py` | `query_code` returns the unit query when `unit_query` (the same object when off); `scan_final` dispatches on `store` (the delta branch is the old call); `write` raises for a non-delta store unless nothing is written and `scan_prewrite` always raises (final-only support writes) |
| `src/looplora/llm/train.py` | `store`, `store_eta`, `store_sweeps`, `unit_query` read from `config['model']` with `get`; the writer signature gains them only when non-default |
| `src/looplora/llm/read_fix.py` | `store_read` (exact-coefficient dispatch; delta is the old `gram_read`), `diagnostics` uses it and adds `key_gram_log10_cond` (and `ridge_log10_cond` for RG) for every LSQ arm: a non-delta store or `unit_query` (arm U) |
| `src/looplora/llm/read_fix_run.py` | `fit` adds the optional `experiment.log_extras()` columns to each log row (none for every existing arm) |
| `src/looplora/llm/lsq.py`, `lsq_run.py` (new) | `configuration`, `LSQExperiment` (log extras: key-Gram and ridge condition numbers; arms U: R's delta store and learned lambda, beta with `unit_query` on and no store field; K4, RG: R plus the store fields); runner `python -m looplora.llm.lsq_run --arm ARM --out ROOT (--steps N --smoke [--eta X] \| --schedule release.json) [--e8-decision lock.json]`; `choose(records, arms)` (prereg rule over the release's arms), `load_inference` (the WP-L4 and DPFP loaders refuse an LSQ export and vice versa) |
| `src/looplora/llm/lsq_release.py` (new, step 1c; the parts moved verbatim out of `lsq.py` and `lsq_run.py`) | torch-free: arms in candidate order L32-U, L32-K4, L32-RG, L128-U, L128-K4, L128-RG, `SPECS`, `ETA_BOUNDS`, `lsq_eta` (the clip), `check_arms` (per rank nothing, U alone, or K4 with RG), `release_arms` (the release's `arms` are RFX's `lsq_arms`), `release_eta` (RG arms only; enforces the clipped RFX eta), `scheduled_steps`, `eligible_e8`; the runner and `scripts/prepare_lsq_release.py` share them |
| `src/looplora/llm/release.py` (new, step 1c) | torch-free helpers of both release scripts: `smoke_timing` (checks the identity of a 10% smoke record and reads its timings, never a score), `largest_schedule` (stdlib copy of WP-L4's rule), `schedule_fields` (step counts, five evaluation marks, sizing, `science_commit`, smoke block), `write_new` (refuses to overwrite) |
| `scripts/prepare_lsq_release.py`, `scripts/prepare_dpfp_release.py` (new, step 1c) | the immutable schedule releases, standard library only (login node: `PYTHONPATH=src python3 scripts/...`); commands below |
| `src/looplora/llm/rfx.py`, `rfx_run.py` (new) | `FinalRead` (instance-level hook: records, and optionally replaces, the final-token read; no change to shared code), `observe`, ridge/gain/eta fits on the fitting set, refit decoder, entity-restricted EM (`entity_hits`: argmax over the name tokens `load_name_pool().token_ids[:V]` with every other token masked to -inf, so an exact tie goes to the lowest token id as in the full-vocabulary argmax and the winner is mapped back to its entity index (step 1d), per variant; each cell also records `full_hit_entity_miss`, the rows with a full-vocabulary hit and a restricted miss; `entity_em_failures` is the guard, step 1c), write age and order, conditioning, `classify` (the registered verdict tree and every rescue flag), `lsq_arms`, `storage_ranks`, `load_checkpoint`; runner `python -m looplora.llm.rfx_run --out DIR [--smoke] [--labels L ...]` (one job, three checkpoints, refuses a checkpoint whose blake2b-128 digests differ from `rfx_run.DIGESTS` before creating anything; `summary.json` has `verdicts`, `rescue`, `lsq_arms`, and `rfx_eta_by_rank` and `eta_by_rank` for the ranks with a storage verdict only, built by `decisions`, and `entity_em_guard` = `{slack, passed: true}`; if the guard fails the per-label JSON and samples are written and the job raises before `summary.json`, so a summary exists only for a run that passed it) |
| `tests/test_lsq.py`, `tests/test_llm_lsq.py`, `tests/test_rfx.py`, `tests/test_llm_rfx.py` (new) | see below |
| `docs/prereg/RFX.md`, `docs/prereg/LSQ.md` (registered by the orchestrator, not edited here), `docs/notation.md` (section 10), `configs/p1l/README.md` | decision rules and node-hour estimates; options documented; this file |

**Off path.** Every new flag defaults off and the old tensor ops run in the same order: the adapter's delta branch is the
unchanged `memory.scan_final` call, `query_code` returns its input, the training builder reads the new keys with `get`,
signatures and `diagnostics` change only for non-default stores, and `fit` adds no column. `tests/test_lsq.py` checks the
adapter's `scan_final`, `scan_prewrite` and read bitwise against the frozen pre-DPFP `memory.py`
(`tests/assets/memory_before_dpfp.py`); the real-model WP-L4 R reproduction `tests/test_llm_dpfp_repro.py` (first 200 steps,
ranks 32 and 128) runs through the same flags-off path, so one pass covers DPFP-off and LSQ-off.

### WP-L7 tests (compute only; none run yet)
- `tests/test_lsq.py` (core, cpu and cuda): ridge store vs the primal weighted reference (random, all-valid and
  initial-reset layouts), exact interpolation of independent keys as eta -> 0 whatever lambda, normal equations for n above
  the key dimension, error cases, finite-difference gradients; K4 equals the repeated `scan_final(beta = 1)` (1-3 sweeps, one
  reset, interior reset rejected), sweeps approach the interpolant monotonically while four are far from it; gradients
  flow and stay finite with padding; coefficient reconstruction of both stores; `gram_conditions`; adapter dispatch,
  final-only guards, unit-query behavior; arm U's adapter (delta store with the unit query: same parameters, bitwise the
  same write, only the read normalizes); **default adapter bitwise equals the pre-LSQ write and read**; config
  validation matrix; parameter counts equal the delta rule's.
- `tests/test_llm_lsq.py` (tiny Qwen3 on cuda): config equals WP-L4 R plus only the registered fields (6 arms x E16/E8; U adds
  `unit_query` alone); rejected arms, etas and arm/config mismatches; clipped eta rule; off-path experiment signature; per arm
  (U, K4, RG) store and unit-query reads against explicit references (U: the delta rule with the learned lambda and beta),
  `M q_hat` read, gradients (`a_beta` unused by K4 and RG, trained for U), Gram reconstruction and conditioning logs; 5-step
  `fit`, `validate`, export and exact reload for all three kinds (WP-L4's and DPFP's loaders refuse an LSQ export);
  `store_read`; `choose`, `release_arms` (every arm set RFX can trigger is a release, nothing else), `release_eta` (RG only),
  `scheduled_steps` (E16 names every release arm, E8 only the retrained one), `eligible_e8` against the prereg rule; the
  hand-off from `rfx_run.decisions` to a release for every pair of R verdicts.
- `tests/test_rfx.py` (synthetic tensors): ridge reads vs the formula, interpolation and order invariance, equality with
  `ridge_final` at lambda = 1; normalized error; fitted gain; eta/gain fitted on the fitting set only; the 14 variants;
  refit decoder (scale free, chance level); per-episode permutations; delta-state rebuild; table helpers; **the
  registered verdict tree** (hand-written cases from the registration, all 64 combinations of the five decisive rescue
  flags and the decoder flag against a transcription of the rule, every rescue flag recorded, both halves of the rescue
  test, exact 10 pp and 0.5 boundaries, baselines = better of as-is and gain-fitted, the decoder against the
  entity-restricted EM); `lsq_arms` (all verdict pairs, candidate order, W5-r32 inert) and `storage_ranks`;
  `entity_hits` and **its tie rule** (exact ties between the answer and an entity with a lower entity index and a higher token
  id, the mirror image, a three-way tie, a tie with a non-entity token, all-equal logits; a random four-level grid in which
  almost every row ties: the winner is the lowest token id at the entity maximum, a full-vocabulary hit is always a
  restricted hit, and the entity-index rule of step 1c would decide some rows differently); **the entity-EM guard**
  (`entity_em_failures`: a deficit of the restricted EM of 20 rows passes and of 21
  fails at 4,000 rows, 1 row passes and 2 fail at the 64-row smoke, the 200-row boundary where 0.5 pp equals one row,
  queries per episode count as rows, a wrong restricted set and NaN fail, the failure carries the counts); registered
  data ranges and seeds.
- `tests/test_llm_rfx.py` (bare adapters and the tiny model): the hook records and replaces exactly each row's last real
  token (ragged lengths, one-token question, padding untouched, token counter restart, unread rows flagged, loop depth
  guard), construction guards, cleanup and double interception; observation vs the deployed read, cached store, true key and
  answer; re-inserting the recorded read is bitwise the identity; only the replaced rows' last position of the adapted
  projection changes; ragged question masks; `run_checkpoint` smoke (finite JSON, instrumentation checks, deployed EM, NLL
  and entity-restricted EM (names sorted by token id, mapped back) equal the model's own forward, `full_hit_entity_miss` 0
  in every cell, rescue flags, evaluation only: parameters unchanged, no gradients); `load_checkpoint` for both export formats (and its refusals, including an arm-U export); `checkpoint_identity`,
  the registered digests and the refusal of a mismatch; `rfx_run.decisions` (arms, and etas only for storage ranks);
  `rfx_run.run` end to end on two tiny exports (records `entity_em_guard`); the entity-restricted EM against scripted
  logits (the tiny model hits entities only at chance, so the glue that builds `entity_ids` and the answer entity is
  checked with logits whose top token is the answer, a non-entity token with the answer second, another entity, or an exact
  tie between the answer and an entity of a lower index and a higher token id, which the entity-index rule of step 1c would
  take in 1,018 of the 1,024 answer entities: the counts must be (1, 1), (0, 1), (0, 0) and (1, 1), `full_hit_entity_miss`
  0); a failed guard writes every per-label record and no `summary.json`.
- `tests/test_release_scripts.py` (stdlib and pytest; only the parity test imports torch; step 1c): both scripts for every
  arm set RFX can trigger (six for LSQ) and for E16 and E8: the release holds the arms, both eta maps and the step
  counts the runner accepts, the step counts equal an independent transcription of WP-L4's sizing rule (largest
  multiple of 5, one more step does not fit); no `correct` or `em` field is read; a doubtful RFX summary (a smoke, dirty
  source, a failed instrumentation check, no or failed `entity_em_guard`, an unclipped eta, the wrong eta ranks) is
  refused; the release checks catch every edit of a release away from the summary; 20+ smoke-record defects per study
  (wrong arm, rank, store or nu, not a smoke, dirty source, hash, short commit, step count, evaluation marks or
  episodes, E, non-finite or non-positive timing, an eta on a non-RG arm) and smokes at different commits or a
  missing smoke are refused; an existing release is never overwritten (and refuses before reading any input); an E8
  arm RFX did not trigger is refused; both scripts run in a subprocess with `torch` and `numpy` unimportable; every
  arm set that `check_arms` accepts is exactly what RFX can trigger; the stdlib `largest_schedule` equals
  `read_fix_run.largest_schedule` (compute only); the registries are the registered candidate orders.

### Choices the registrations left to the code (the rest was decided at registration)
1. RFX applies the fitted scalar gain to every cell; the baseline is the better of the as-is and gain-fitted deployed
   reads (registered). `rescue` records a flag for all 14 variants plus `decoder_refit` (a superset of the five decisive
   cells), so the report can show every rescue.
2. The entity-restricted EM takes the argmax of the logits over the 1,024 name tokens with every non-entity token masked to
   -inf and maps the winning token id back to its entity index, so an exact tie breaks as in the full-vocabulary argmax (the
   lowest token id). **Step 1d (review F1).** Step 1c broke ties by entity index, which is a seeded shuffle of the token ids
   (511 of 1,023 adjacent pairs of the name pool are inversions): with exact bf16 ties a row could count for the
   full-vocabulary EM and not for the restricted one even with a correct glue, and the 64-row smoke, which is deterministic,
   could trip the guard of item 8 with a probability of a few percent. The registered metric (`docs/prereg/RFX.md`: "argmax
   of the same logits over the 1,024 task-entity name tokens") fixes no tie rule, so no registered quantity changes; no RFX
   number existed before this change. A row that the full-vocabulary argmax hits is now always a restricted hit (the answer
   is an entity token and the lowest-id maximum of the whole vocabulary), for any logits.
3. LSQ eta = clip(RFX fitting-set eta of the (ridge, unit q) cell of R-r_rank, 1e-4, 1e-1), copied into the release;
   RFX reports it for the ranks with a storage verdict only. RFX's ridge is unweighted; LSQ's has recency weights
   lambda^(E-1-j) with the learned decay.
4. `lsq_run.choose(records, arms)` takes the release's arms (required) and rejects any set RFX cannot trigger. The
   candidate order is the one of `lsq_release.ARMS`: U, K4, RG at rank 32, then at rank 128.
5. K4 = four sweeps of the learned-decay beta = 1 scan (64 writes); four sweeps are not the interpolant for correlated keys.
   Neither least-squares store uses beta (arm U trains it); ridge averages duplicate addresses (none occur in T1).
6. RFX's "zero" replaces only the final-position read, so it is not WP-L4's zero-memory control (all positions).
7. If DPFP and LSQ both select a writer, P1L-D uses the one at the lower rank, then the higher E16 val EM, then DPFP
   (registered in `docs/prereg/LSQ.md`; an E8 selection counts as its arm's rank and its E8 val EM). No code applies this
   yet; the audit or table script of step 2 must (the release scripts do not).
8. **Entity-EM guard (step 1c, O1).** `rfx_run.run` raises `RuntimeError` before writing `summary.json` (the per-label
   JSON and samples are already on disk) if any variant of any checkpoint has `lm_em - lm_em_entity` above the allowance
   `max(0.005, 1/rows) + 1e-9`, `rows` = episodes x queries per episode, NaN counted as a failure. The task text says
   0.005; at the registered 4,000 rows the two are identical (20 rows), the 64-row smoke tolerates one row (1.56 pp), and
   the difference matters under 200 rows only. That one row was added in step 1c to absorb tie rows; with the tie rule of
   item 2 nothing needs it any more and step 1d left the guard unchanged as instructed (`max(0.005, 1/rows)` can go back to
   the flat registered 0.005 if the orchestrator prefers; the smoke would then tolerate no row at all). A correct glue now has
   `full_hit_entity_miss` = 0 in every cell, so a nonzero count is a defect even where the guard tolerates it, and a wrong
   slice, axis or pairing in the glue costs about EM x rows.
   Each cell records `full_hit_entity_miss` (rows with a full-vocabulary hit and a restricted miss), and the error
   message lists label, variant, rows, both EMs and that count. `summary.json` carries `entity_em_guard` = `{slack: 0.005,
   passed: true}`, and the LSQ release script refuses a summary without it. The decoder verdict compares the refit decoder
   with the entity-restricted baseline EM, so a doubtful restricted EM can change a verdict: the verdicts, arms and etas of
   `summary.json` therefore exist only for a run that passed (the per-label JSON of a failed run holds its own `verdict`;
   do not use it).
9. **Release scripts (step 1c, O2).** `scripts/prepare_lsq_release.py` and `prepare_dpfp_release.py` import only the
   standard library and the torch-free `looplora.llm` modules (`release`, `lsq_release`, `dpfp_release`, `infra.seeds`),
   so they run on the login node with the system Python 3.9 (`PYTHONPATH=src python3 scripts/...`); `looplora/__init__.py`,
   `looplora/llm/__init__.py` and `looplora/infra/__init__.py` import nothing heavy. To make that possible, `ARMS`,
   `SPECS`, `lsq_eta`, `check_arms`, `release_arms`, `release_eta`, `scheduled_steps` and `eligible_e8` moved verbatim
   out of the torch-importing `lsq.py`, `lsq_run.py`, `dpfp.py` and `dpfp_run.py` into `lsq_release.py` and
   `dpfp_release.py` (no aliases; every importer and test was re-pointed), so the script checks its release with the
   runner's own functions. The LSQ release is `study: LSQ`, `E`, `arms` (= `summary['lsq_arms']`), `steps_by_arm`,
   `primary_eval_steps_by_arm`, `sizing_by_arm`, `rfx_eta_by_rank` and `eta_by_rank` (copied from the summary),
   `science_commit`, the `smoke` timing rows and `rfx` provenance (`summary_blake2b`, the summary's `git_commit`,
   `job_id` and `verdicts`; a blake2b-128 digest); extra fields do not disturb the runner. Before writing, the script
   asserts `arms == summary['lsq_arms']`, both eta maps equal to the summary's and that `release_arms`, `release_eta`
   and `scheduled_steps` accept the release. It reads only identity and timing fields of the smokes, refuses a mixed
   or dirty commit, and never overwrites an existing release. The E8 release needs an E8 smoke of the retrained arm
   (`task.E` must match) and, for LSQ, an arm that RFX triggered. The DPFP E8 check is the runner's `eligible_e8`, not
   tightened further.
10. **Reload check (step 1d, review F2).** The release scripts check identity and timing only; WP-L4's A2 gate also reloaded
   the export and compared a fresh validation, which needs torch. `scripts/check_smoke_reload.py` is that part, one short
   compute job per smoke arm before the release is committed (commands below). It reads the uncommitted release (study,
   `E`, smoke steps, the arm's smoke row) and the arm folder, and fails unless: the record is the one the release was sized from
   (`job_id`, `git_commit`, `source_hash`); the record's config equals the code's `configuration(arm, smoke steps, E of the
   release, smoke=True)` (LSQ RG: with the smoke's own `store_eta`, not the release's; JSON-normalized, since tuples become
   lists); the study's `load_inference` reloads `ARM/writer.pt` (signature and strict state-dict checks inside it) to an
   experiment with that same config, no value head and finite parameters; and a fresh 100-episode `validate` of the reloaded
   writer has the smoke's episodes and exactly the predictions of the smoke's last evaluation (`ARM/val-STEPS`). It never
   prints or compares a score, writes `ROOT/export-check-ARM/` (the folder must not exist: `validate` refuses an existing
   one) and, only on success, `reload-check.json` there. It does not require the check to run at the smoke's commit (the
   result records both), though the commands use `science_commit`. It has not been run and has no compute test: its first
   run is the gate, and a failure blocks the commit of the release and changes nothing else.

## Step-2 order and worklist (after 2026-10-28 22:00 JST; WP-L5 and WP-L7)
Order given by the orchestrator: (1) tests, (2) DPFP smokes, (3) RFX, (4) DPFP arms, (5) LSQ only if RFX triggers it.
Sketch with the reasons:
1. **Tests**: command 1 above (DPFP, LSQ, RFX, release-script and regression tests) and command 2 (DPFP-off and LSQ-off reproduction of
   WP-L4 R). If a test fails, find out whether the code or the test is wrong before changing either; the WP-L7
   tests were never run.
2. **DPFP smokes**: command 3 above (timing and memory only; no scores read).
3. **RFX**, one smoke then one science job (about 0.2 node-hours); `docs/prereg/RFX.md` is registered:
```sh
REF=<full hash of the last WP-L7 step-1d commit on wp/llm, or its master merge>; RUNS=/work/xg24i002/x10041/LoopLora/runs/p1l
scripts/submit.py --name rfx-smoke --queue regular-g --nodes 1 --walltime 00:20:00 --commit "$REF" --wait -- \
  -m looplora.llm.rfx_run --smoke --out $RUNS/rfx/smoke
# the smoke runs the entity-EM guard (below): it has a summary.json only if the guard passed. Check it before the science
# job, with the count that a correct glue never has (step 1d: the restricted argmax breaks ties like the full one): it must be 0
python3 -c "
import json
s=json.load(open('$RUNS/rfx/smoke/summary.json'))
miss=[c['full_hit_entity_miss'] for l in s['labels'] for c in json.load(open('$RUNS/rfx/smoke/%s.json'%l))['cells'].values()]
print(s['entity_em_guard'], s['instrumentation_failures'], 'max full_hit_entity_miss', max(miss))"
scripts/submit.py --name rfx --queue regular-g --nodes 1 --walltime 00:30:00 --commit "$REF" --wait -- \
  -m looplora.llm.rfx_run --out $RUNS/rfx/science
python3 -c "
import json
s=json.load(open('$RUNS/rfx/science/summary.json'))
print(s['verdicts'], s['lsq_arms'], s['rfx_eta_by_rank'], s['eta_by_rank'], s['instrumentation_failures'])"
```
   The smoke evaluates fresh `train` episodes (never the val or test ranges), so its numbers may be read to test the
   code, not for results. **Entity-EM guard (step 1c)**: both the smoke and the science run raise before writing
   `summary.json` (the per-label `LABEL.json` and samples stay) if any variant of any checkpoint has an entity-restricted
   EM below the full-vocabulary EM by more than 0.5 pp or one row, whichever is more (64 rows in the smoke, so one row is
   1.56 pp; 4,000 rows in the science run, so 20 rows). Run the smoke first so that a glue bug in `rfx.evaluate` (the
   entity ids, the axis, the answer entity) costs a smoke and not the science job. Since step 1d the restricted argmax
   breaks ties as the full-vocabulary one does, so a correct glue has no row with a full-vocabulary hit and a restricted
   miss: the per-variant count `full_hit_entity_miss` is 0 in every cell (the guard's slack of 1 row at 64 rows and 20 at
   4,000 is not meant to be used), and a nonzero count after the smoke is a bug to find before the science job even when
   the guard passed. If the guard raises, diagnose from the per-label JSON before anything else: a wrong slice or pairing
   costs about EM x rows:
```sh
python3 -c "
import json
for label in ('R-r128','R-r32','W5-r32'):
    r=json.load(open('$RUNS/rfx/smoke/%s.json'%label)); e=r['data']['eval']
    for name,c in r['cells'].items():
        print(label,name,e['episodes']*e['queries_per_episode'],c['lm_em'],c['lm_em_entity'],c['full_hit_entity_miss'])"
```
   The science run also raises after writing its files if an instrumentation check fails; fix the
   code and rerun everything (the registered rule does not change; the guard is not a threshold to tune). The job refuses
   checkpoints whose digests differ from the ones in `docs/prereg/RFX.md` (`rfx_run.DIGESTS`), before creating its output.
   Then `docs/reports/RFX.md` (a table script like `analyze_p1l_read_fix.py` is still to be written).
4. **DPFP arms**: `scripts/prepare_dpfp_release.py` on the login node, the three reload checks, then commit the release
   and run three jobs, as described above.
5. **LSQ, only the arms in `summary.json['lsq_arms']`** (`docs/prereg/LSQ.md` is registered): 10% smokes of those arms
   (RG arms need an `--eta` inside 1e-4..1e-1; U and K4 take none; smokes are never interpreted), then the release, the
   reload checks, the commit of the release and the arms:
```sh
RUNS=/work/xg24i002/x10041/LoopLora/runs/p1l
# one smoke per ARM in summary.json['lsq_arms'], e.g. L32-K4 and L32-RG (RG: add --eta 1e-3; U and K4: no --eta)
scripts/submit.py --name s-lsq-L32-K4 --queue regular-g --nodes 1 --walltime 00:25:00 --commit "$REF" -- \
  -m looplora.llm.lsq_run --arm L32-K4 --steps 2400 --smoke --out $RUNS/lsq/smoke
scripts/submit.py --name s-lsq-L32-RG --queue regular-g --nodes 1 --walltime 00:25:00 --commit "$REF" -- \
  -m looplora.llm.lsq_run --arm L32-RG --steps 2400 --smoke --eta 1e-3 --out $RUNS/lsq/smoke
# release (login node, standard library only), after every smoke has completed; written, not committed yet:
PYTHONPATH=src python3 scripts/prepare_lsq_release.py --summary $RUNS/rfx/science/summary.json --root $RUNS/lsq/smoke \
  --out reports/p1l-lsq/release.json
# reload check (step 1d), one short compute job per arm of the release, before the release is committed:
REL=/work/xg24i002/x10041/LoopLora/reports/p1l-lsq/release.json
SC=$(python3 -c "import json;print(json.load(open('$REL'))['science_commit'])")
for ARM in $(python3 -c "import json;print(*json.load(open('$REL'))['steps_by_arm'])"); do
  scripts/submit.py --name c-lsq-$ARM --queue regular-g --nodes 1 --walltime 00:10:00 --commit "$SC" -- \
    scripts/check_smoke_reload.py --release $REL --root $RUNS/lsq/smoke --arm $ARM
done
# every arm must have left $RUNS/lsq/smoke/export-check-ARM/reload-check.json (print them as for DPFP); only then commit the
# release and run, per arm of the release:
scripts/submit.py --name r-lsq-L32-RG --queue regular-g --nodes 1 --walltime 02:00:00 --scale-up docs/prereg/LSQ.md \
  --commit <science_commit> -- -m looplora.llm.lsq_run --arm L32-RG --schedule RELEASE.json --out $RUNS/lsq/arms
```
   The release holds `study: LSQ`, `E: 16`, `arms` (= `summary['lsq_arms']`, so per rank U alone or K4 with RG, in
   candidate order), `steps_by_arm` (WP-L4's sizing rule on the smokes' timings, five intervals), `rfx_eta_by_rank` and
   `eta_by_rank` (copied from the summary; they exist for the ranks with RG arms), `science_commit` and the RFX
   provenance. The script needs one complete, clean smoke per arm of `lsq_arms` at one commit (other folders under
   `--root` are not read); it refuses a summary that is a smoke, from dirty source, has an instrumentation failure or lacks
   the passed entity-EM guard; before writing it asserts that `arms` and both eta maps equal the summary's and that the
   runner's `release_arms`, `release_eta` and `scheduled_steps` accept the release; it reads only identity and timing
   fields of the smokes and never overwrites an existing file. It prints the study, `E`, arms, step counts and
   `science_commit` only.
   The conditional E = 8 retrain adds `--e8-decision LOCK.json` to three things: the arm's own E8 smoke (`lsq_run --arm
   ARM --steps 2400 --smoke [--eta X] --e8-decision LOCK.json --out $RUNS/lsq/e8-smoke`), its own release (`prepare_lsq_release.py
   --summary ... --root $RUNS/lsq/e8-smoke --out reports/p1l-lsq/release-e8.json --e8-decision LOCK.json`: `E` 8, every
   other field as above, `steps_by_arm` holding only the retrained arm, which must be one of the RFX arms) and its own
   science `--out $RUNS/lsq/e8`: the E = 16 arm folder already exists in `$RUNS/lsq/arms`, and the runner refuses to
   reuse it; its reload check is `check_smoke_reload.py --release .../release-e8.json --root $RUNS/lsq/e8-smoke --arm ARM`.
   Node-hours (`docs/prereg/LSQ.md`): about 2.7 used and at most 4 reserved for a rank with a storage verdict (K4 and RG),
   about 1.4 and at most 2 for a normalization verdict (U); the reload checks add about 0.05 per smoke arm (an estimate).

Still to write in step 2 (not needed before; the two release scripts are done): `scripts/audit_dpfp.py` and an LSQ
counterpart (final checkpoints: zero/shuffle memory, key overlap, read error, value SIR, clean/read value-head EM;
`controlled_score` and `diagnostics` already support rectangular and LSQ states; load LSQ exports with
`lsq_run.load_inference`), table scripts for `docs/reports/DPFP.md`, `RFX.md` and `LSQ.md` (every curve, node-hours,
`dpfp_run.choose` / `lsq_run.choose(records, lsq_release.release_arms(release))` on the E16 records, and the P1L-D writer pick
between DPFP and LSQ: lower rank, then higher E16 val EM, then DPFP).

## Open questions and risks for the orchestrator (WP-L7)
- The registrations decided the drafts' open points; nothing in the step-1b, 1c and 1d code is pending a decision.
  Choices the code made where the text was silent are listed above (every rescue flag for all variants, the entity-EM tie
  rule, `choose` takes the release's arms, the guard allowance, the release scripts' checks, the reload check). One departs
  from the task text and is worth a look: the guard allows `max(0.005, 1/rows)`, not 0.005 (identical at the registered
  4,000 rows); since the tie rule of step 1d nothing needs the extra row at the 64-row smoke any more (item 8).
- Step 1c moved the LSQ and DPFP registries and release checks out of the torch-importing modules into new stdlib
  modules (item 9); it is a pure move, but it touches the two runners and four existing test files, so the first
  compute test run must include `tests/test_llm_lsq.py` and `tests/test_llm_dpfp.py`, which command 1 does. The scripts
  and their tests need Python 3.9-compatible syntax (the login node's Python) and `PYTHONPATH=src`.
- Step 1d changed `rfx.entity_hits` (the only RFX code that decides a metric), which command 1 tests through
  `tests/test_rfx.py` and `tests/test_llm_rfx.py` (the latter on the tiny model, where the reference now sorts the names by
  token id). `scripts/check_smoke_reload.py` has no compute test; if the first check job fails, find out whether the check or
  the smoke is wrong before touching either.
- The rescue rule needs the LM path to accept a substituted read. A "none" or "decoder" verdict on a checkpoint whose
  clean head decodes well can be a distribution-shift artifact of the downstream path (stated as a caveat in the prereg).
- Budget: tests, smokes and RFX together are about 1 node-hour for WP-L5 and WP-L7 (tests 0.1-0.2, DPFP smokes 0.3-0.4,
  RFX smoke and science 0.2-0.25); LSQ adds 2.7-4 node-hours per rank with a storage verdict and 1.4-2 per rank with a
  normalization verdict.
- All new code is untested on a GPU. The riskiest parts are the RFX hook (checked by the identity and ragged-mask tests),
  the float64 Cholesky solve in the training loop (finite-difference and gradient tests), and the tiny-model comparisons
  that depend on bf16 rounding (tolerances were set from the formulas, not from runs).
