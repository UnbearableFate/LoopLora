# WP-L12 state: SQ2 (storage x coefficient-loss 2x2 continued from D128-3), done

Branch `wp/llm`, master `a3013ff` (SQ2 Clarification C1) merged in. Not merged into master, not pushed. Task: `docs/tasks/WP-L12_storage_studies.md`; registration `docs/prereg/SQ2.md` (blake2b-128 with C1 `9f725a62...`); results `docs/reports/SQ2.md`.
**Final registered decision: the primary contrast S-RC against S-DE does not advance (EM +4.1 pp, needs +10; read-error ratio 0.554, needs at most 0.5; stable decoding met). The learned-writer line stays stopped; no cell reaches 0.80.**
Correct of 1,000: S-DE 654, S-DC 630, S-RE 723, S-RC 695 (parent D128-3 608). Nothing remains to run for SQ2.

## Commits (wp/llm)
| commit | content |
| --- | --- |
| `ecbe5e5` | code and tests: ridge store on DPFP features, coefficient loss, continuation loader, numerics, runner, release and check scripts, LSQT code |
| `5cff944`, `b637e75`, `dd5e4c8` | test fixes; two product defects in `sq2_numerics.py` found by its test (unpacking, `torch.sqrt` of a float). `dd5e4c8` is the LSQT science commit |
| `7b51ccd`, `0a4b96f` | merge of master a3013ff (Clarification C1); the prereg digest in the tests |
| `3d543b4` | `reports/p1l-sq2/numerics.json` (eta 1e-4, w_c 1). **3d543b4 is the science commit** of the SQ2 smokes, arm jobs and hashes |
| `81284e1` | `check_sq2.py reload` also checks the science config hash and the prereg digest |
| `aeb988a`, `a5cb772` | `reports/p1l-sq2/release.json` (N 6,000), the orchestrator approval record |
| the report commit | records, step-0, audit, decision, `scripts/analyze_sq2.py`, `tests/test_analyze_sq2.py`, `docs/reports/SQ2.md`, this file |

## What the code does (every new option off reproduces the old paths)
- `core/config.py`, `core/memory.ridge_final`, `core/adapters.py`: the ridge store takes DPFP features (`feature_map`, solved through the n x n system; the sweep store still refuses them; `AdapterConfig` rule updated, `tests/test_lsq.py` expectations changed accordingly).
- `llm/read_fix.py`: `coefficient_loss` (directional, float64, eps 1e-6), `coefficient_statistics`, `ReadFixExperiment.coefficient_loss` (None unless `read_fix.coefficient_weight` is set). `llm/read_fix_run.fit` takes optional `data_seed`, `optimizer_state`, `warmup`, `time_limit` (defaults reproduce the old loop) and calls `observe` / `log_extras` when present.
- `llm/sq2_release.py` (torch-free: arms, parent digests, eta and w_c rules, step rule, prereg digest), `sq2.py` (config, `SQ2Experiment`, `load_continuation`, `load_inference`), `sq2_numerics.py`, `sq2_run.py` (step-0 evaluation, fresh data stream, constant lr).
- Scripts: `sq2_numerics.py` (compute), `prepare_sq2_release.py` (login), `dpfp_config_hashes.py --study SQ2`, `check_sq2.py` (`step0`, `reload`), `audit_dpfp.py` (loader per study: DPFP, LT-*, S-*), `analyze_sq2.py` (tables and decision).

## Tests (compute)
- `tests/test_sq2.py` (41, all pass at dd5e4c8): registry and prereg digest; eta and w_c rules at their boundaries; configs equal D128-3 plus only the registered changes; ridge on DPFP features equals the dual and primal ridge solutions and interpolates; coefficient loss against a hand computation and exact stubs; the coefficient and statistics on the tiny model for all four arms;
  the continuation (step-0 equality with the parent, optimizer state, digest / hash refusals); constant lr and fresh stream; numerics on a tiny model; `run_one` end to end; the real D128-3 checkpoint (digests, config hash, S-DE config equality, recorded 608 and NLL).
- `tests/test_lsqt.py`, `tests/test_release_scripts.py` (LSQT and SQ2 release modes), `tests/test_audit_dpfp.py` (LT and S exports), `tests/test_lsq.py` and the DPFP, DPFP-X and repro tests (`tests/test_llm_dpfp_repro.py` is the bitwise DPFP-off check): green; the rest of the suite 2,440 passed, 1 skipped (job 3724076).
- `tests/test_analyze_sq2.py` (3): the decision rule at its exact thresholds and the decision re-derived from the committed records.
- Test defects: duplicate keyword in a helper, a wrong Adam-step expectation (a parameter without gradient does not step: `a_beta` under the ridge store), cuda/cpu tensors, one rounding-boundary expectation (5.7 against 5.657). No assertion weakened.

## How to reproduce the tables
`cd scripts && python3 analyze_sq2.py ../reports/p1l-sq2` (stdlib; reads `reports/p1l-sq2/`). Large files (`checkpoint.pt`, `writer.pt`, `samples.npz`) stay under `/work/xg24i002/x10041/LoopLora/runs/p1l/sq2/`
(`smoke`, `arms`, `step0`, `audit`, `numerics.json`, `config-hashes.json`, `release.json`). The release is byte-identical to `reports/p1l-sq2/release.json`; science jobs ran with `--schedule` on the runs-dir copy.

## Cost
SQ2 jobs 1.740 node·h (numerics 0.021, step-0 0.008, smokes 0.186, hashes 0.005, reload 0.026, four arm jobs 1.466, audit 0.027); the shared WP-L12 test jobs 0.438; with them 2.178 node·h (prereg estimate about 2.1). Job table in `docs/reports/SQ2.md`.

## Not done / open
Nothing is open for SQ2. Descriptive facts for the paper table: the ridge store swap alone gives 68.7 % at step 0 (D128-3 weights, nothing retrained); S-RE is the best cell (72.3 %); the coefficient loss at w_c = 1 changed neither the read error nor the direction cosine;
the gradient clip (1.0) is active at every logged step in all four arms and in D128-3's own log (pre-clip norm about 8 to 10). LSQT (3724516 to 3724518) is separate and still running at the time of writing.
