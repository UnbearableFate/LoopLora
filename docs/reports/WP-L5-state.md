# WP-L5 state: DPFP option for the learned writer (step 1 done; nothing executed)

Branch `wp/llm`, base master `6da6330`. Step 1 of `docs/tasks/WP-L5_dpfp_writer.md` is implemented and statically
checked (`py_compile` and `pyflakes` on every changed Python file, plus a torch-free dry run of the repro test's CSV
parsing). **No torch code, test or run has been executed**: the login node forbids it and compute is unavailable until
2026-10-28 22:00 JST. Not merged, not pushed. Binding design: `docs/prereg/DPFP.md`.

## What is implemented
DPFP is `AdapterConfig.dpfp_nu` (default 0 = off). With nu >= 1 the write key and the read query are mapped to
`phi_nu(.)/max(||phi_nu(.)||_2, eps)` and the memory is `[B, r_mem, key_dim]`, `key_dim = 2 * r_mem * nu`.

| File | Change |
| --- | --- |
| `src/looplora/core/config.py` | `dpfp_nu` (last field, memory variants only), `key_dim` property |
| `src/looplora/core/memory.py` | `dpfp_features(k, nu)`; keyword `feature_map` in `write`, `scan_prewrite`, `scan_final` (applied before the unchanged unit normalization); rectangular states (`zeros_state(..., key_dim)`, `_check_state`, `delta_update` shape checks) |
| `src/looplora/core/adapters.py` | `key_dim` in state shapes; `MemoryAdapter.query_code` (identity when off) and `_key_features` (passes `feature_map` only when on); `_query` is now an instance method |
| `src/looplora/core/accounting.py` | MACs and state bytes use `r_mem * key_dim` (identical values when off) |
| `src/looplora/llm/inject.py`, `train.py` | zero states use `key_dim`; `dpfp_nu` read from `config['model']`; writer signature gains `dpfp_nu` only when nonzero |
| `src/looplora/llm/writer_fix.py` | recorded keys are the unit phi keys actually written (`FixBatch.keys`), original-space keys in `FixBatch.base_keys`; InfoNCE queries `query_codes` are `q_hat`; W4 + DPFP raises |
| `src/looplora/llm/read_fix.py` | `question_terms` reads `M q_hat`; `diagnostics` keeps its columns (now phi space for DPFP arms) and adds `key_gram_rms_base`, `key_gram_max_abs_base` |
| `src/looplora/llm/dpfp.py` (new) | arms `D32-3`, `D128-1`, `D128-3`; `configuration(arm, steps, E=16, smoke=False)` = `read_fix.configuration('R', ...)` at rank 32/128/128 plus `model.dpfp_nu` and a `dpfp` block; `DPFPExperiment(ReadFixExperiment)` |
| `src/looplora/llm/dpfp_run.py` (new) | one arm per job: `python -m looplora.llm.dpfp_run --arm ARM --out ROOT (--steps N --smoke \| --schedule release.json) [--e8-decision lock.json]`; reuses `read_fix_run.fit` unchanged; record adds rank, nu, key_dim, state bytes, peak CUDA memory; `choose` (registered rule), `scheduled_steps`, `eligible_e8`, `load_inference` |
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
  `AdapterConfig` validation; `process_sequence` and `process_sequence_fast` vs token-by-token feature-space
  equations; Gram read in phi space; accounting.
- `tests/test_llm_dpfp.py` (tiny Qwen3 on cuda): config equals WP-L4 R plus only the DPFP fields (3 arms x E16/E8);
  inconsistent arm/config rejected; W4 guard; per arm (nu 3 and 1): state/key/query shapes, unit nonnegative phi keys,
  memory = ordered delta rule over the recorded keys, InfoNCE and alignment in phi space, `M q_hat` read, gradients of
  the raw-read loss, Gram reconstruction, original- and phi-space key overlap; DPFP-off experiment wiring unchanged;
  zero/shuffle controls on the rectangular state; 5-step `fit`, `validate` (16 episodes), export and exact reload
  (and WP-L4's loader refuses a DPFP export); `choose`, `scheduled_steps`, `eligible_e8` against the prereg rule.
- `tests/test_llm_dpfp_repro.py` (real model): **200-step DPFP-off reproduction of WP-L4 R** at rank 32 and 128.
  It compares exactly (`==`) every logged field except step time at steps 1/50/100/150/200 with the committed
  `reports/p1l-read-fix/loss_curves.csv` (steps of record 23590 and 24645; the first 200 steps do not depend on
  them). Each rank trains in a fresh subprocess; run the file alone in its own job. A step-200 match implies
  the whole trajectory matched. If either rank differs, first check for an environment change since WP-L4
  (torch/CUDA/driver, maintenance); per the prereg both references are then rerun under the same schedule rule.

## Commands for step 2 (after 2026-10-28 22:00 JST; run from `/work/xg24i002/x10041/LoopLora`)
`REF` is any committed ref (`wp/llm`, or the commit after the orchestrator merges); submit.py snapshots it by `git archive`.
Measured WP-L4 comparables: 69 tests 182 s; one rank-128 smoke job (2400 steps, two arms in parallel) 510 s.
```sh
REF=wp/llm; RUNS=/work/xg24i002/x10041/LoopLora/runs/p1l/dpfp

# 1. DPFP tests plus regression tests of every touched module
scripts/submit.py --name t-dpfp --queue regular-g --nodes 1 --walltime 00:20:00 --commit "$REF" --wait -- \
  -m pytest -q tests/test_dpfp.py tests/test_llm_dpfp.py tests/test_llm_read_fix.py tests/test_llm_writer_fix.py \
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
Science (later, per arm, after the three step counts are committed in a release JSON with `study: DPFP`, `E: 16`,
`steps_by_arm`, `science_commit`): `scripts/submit.py --name r-dpfp-ARM --queue regular-g --nodes 1 --walltime 02:00:00
--scale-up docs/prereg/DPFP.md --commit <science_commit> -- -m looplora.llm.dpfp_run --arm ARM --schedule RELEASE.json --out $RUNS/arms`.
The conditional E8 retrain adds `--e8-decision LOCK.json` (the audited lock whose `choice` is `retry_e8` for that arm).

## Choices where the prereg was silent
1. Normalization is `x / max(||x||_2, eps)` with the existing eps = 1e-6, at write and read; a feature vector below eps
   would not be unit, which cannot occur for generic nonzero keys (phi_nu is quadratic).
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
- The step-count rule is not coded: reuse WP-L4's `read_fix_run.largest_schedule` (1.5 margin, 90 s reserve, 7200 s,
  five evaluations, 1000-episode evaluation time = 10 x the smoke's 100-episode maximum) unless you fix another.
- D128-3 stores 128 x 768 fp32 states (393 KB per episode, 128 query copies per batch); memory should be fine on a
  GH200 but is measured in the smoke record (`memory`).

## Step-2 worklist
Run commands 1-3 and fix failures; `scripts/prepare_dpfp_release.py` after `prepare_p1l_read_fix_a2.py` (timing-only
read, frozen per-arm step counts, `science_commit`); submit three arms; `scripts/audit_dpfp.py` after
`audit_p1l_read_fix.py` (final checkpoints: zero/shuffle memory, original and phi key overlap, read error, value SIR,
clean/read value-head EM; `controlled_score` and `diagnostics` already support the rectangular state); analysis and
`docs/reports/DPFP.md` with every curve, node-hours, and `dpfp_run.choose` on the three E16 records.
