# WP-L12 state: LSQT (LT-K4, LT-RG001, LT-RG1 at r = 128, from scratch) and the U control, done

Branch `wp/llm`, master `a3013ff` merged in. Not merged into master, not pushed. Task: `docs/tasks/WP-L12_storage_studies.md`; registration `docs/prereg/LSQT.md` (blake2b-128 `94ed7364...`); results `docs/reports/LSQT.md`.
**Final registered decision: no arm reaches 0.80 (K4 383, RG001 461, RG1 426 of 1,000), no writer candidate. The best ridge arm LT-RG001 (46.1 %) is in the 40 % to < 48.5 % band: neither the W3 trigger nor the falsifier. The trained-LSQ line stays open; the learned-writer line stays stopped.**
All arms are above R-r128 (33.5 %) by +4.8, +12.6 and +9.1 pp and below D128-3 (60.8 %) by 22.5, 14.7 and 18.2 pp; read errors 0.554, 0.512, 0.499.
The conditional U control (descriptive; enters neither outcome) ran after approval: **LT-U 360 of 1,000 (36.0 %)** at 24,560 steps, 2.5 pp above R-r128 and 10.1 pp below LT-RG001. RG001's +12.6 pp over R-r128 splits, in point estimates, into **+2.5 pp for the unit query (U - R-r128, standard error 2.1)** and
**+10.1 pp for the store (RG001 - U, standard error 2.2)**; one seed, descriptive, not a registered test. U's final checkpoint is its registered selection (each arm at its own final checkpoint) and also its highest. The registered decision is unchanged.

## Commits (wp/llm)
| commit | content |
| --- | --- |
| `ecbe5e5` | LSQT and SQ2 code and tests (`lsqt_release.py`, `lsqt.py`, `lsqt_run.py`, release / hash / reload script modes) |
| `5cff944`, `b637e75`, `dd5e4c8` | test fixes. **dd5e4c8 is the science commit** of the LSQT smokes, hashes and arm jobs |
| `ec91b3a`, `1ac2ee1` | `reports/p1l-lsqt/release.json` (steps 17,970 / 28,005 / 28,245), the orchestrator approval record |
| the report commit | records, audit, decision, `scripts/analyze_lsqt.py`, `tests/test_analyze_lsqt.py`, `docs/reports/LSQT.md`, this file |
| `c07452b` | the LSQT-U release path (study `LSQT-U`, LT-U alone), tests; the **science commit of the U smoke, hashes and arm job** |
| `71a6f1c`, `a0eaddd` | `reports/p1l-lsqt-u/release.json` (LT-U 24,560 steps), the orchestrator approval record (committed before the U science job) |
| `8d701e8`, `19ba57c` | U records (`arms/LT-U`), `audit.json`, `split.json`, `scripts/analyze_lsqt_u.py`, `tests/test_analyze_lsqt_u.py`; the matched-fraction output of the script |
| the U report commit | `docs/reports/LSQT.md` ("U control" section, cost, attribution) and this file |

## What the code does (nothing in the LSQ, DPFP or WP-L4 paths changed)
- `lsqt_release.py` (torch-free): `ARMS = (LT-K4, LT-RG001, LT-RG1)`, `CONTROL = LT-U`, `SPECS`, `ETA_BOUNDS = (1e-4, 1)`, `prereg_digest`, `scheduled_steps` (the three arms from an `LSQT` release; the control only from an `LSQT-U` release).
- `lsqt.py`: `configuration` (R at rank 128 + `unit_query` + the store fields, an `lsqt` block; `lsq.STORE_RULES` extracted, output unchanged) and `LSQTExperiment` (checks arm, rank, store, eta, no `lsq` block). `lsqt_run.py`: runner, `load_inference`, `choose_t`.
- `scripts/prepare_dpfp_release.py --study LSQT|LSQT-U`, `dpfp_config_hashes.py --study LSQT|LSQT-U`, `check_smoke_reload.py` (LSQT, LSQT-U; frozen science hash and prereg digest), `audit_dpfp.py` (loader per study), `analyze_lsqt.py`, `analyze_lsqt_u.py` (the U record check against the release with a strict finiteness scan, the split with standard errors, matched-fraction differences, `--split-out`).

## Tests (compute)
- `tests/test_lsqt.py` (27 at dd5e4c8, plus the U release test): registry and prereg digest; configs are R + the registered store fields (and equal LSQ's L128-K4 / L128-RG training config up to name); LSQ config hashes unchanged (24 literals computed on compute at 2396e54, job 3724024);
  rejected configs and mixed arms; scheduled steps; `choose_t` at its thresholds; tiny train, export, reload for all four stores; no other loader accepts an LSQT export.
- `tests/test_llm_dpfp_repro.py` at dd5e4c8 with `tests/test_lsqt.py`: 29 passed, 0 failed (job 3725434). `tests/test_release_scripts.py` (LSQT and LSQT-U modes), `tests/test_audit_dpfp.py` (LT exports), `tests/test_lsq.py`, `tests/test_llm_lsq.py`, DPFP tests: green. The rest of the suite: 2,440 passed, 1 skipped (job 3724076).
- `tests/test_analyze_lsqt.py` (3): the rule at its thresholds and the decision re-derived from the records (jobs 3725511 and 3724910).
- U: `tests/test_lsqt.py`, `tests/test_release_scripts.py`, `tests/test_analyze_lsqt.py`, `tests/test_analyze_sq2.py`, `tests/test_llm_lsq.py` and `tests/test_audit_dpfp.py` at c07452b: 238 passed, 0 failed (job 3725518, before the smoke).
  `tests/test_analyze_lsqt_u.py` (3: the split arithmetic and standard errors, the refusal of NaN / infinity / overflow in the record and the train log, the committed records re-deriving `split.json`) with `tests/test_analyze_lsqt.py`: 6 passed at `8d701e8` (job 3727496) and again at `19ba57c`, after the matched-fraction output was added (job 3727554).

## How to reproduce the tables
`cd scripts && python3 analyze_lsqt.py ../reports/p1l-lsqt ../reports/p1l-dpfp` (stdlib); U: `python3 analyze_lsqt_u.py ../reports/p1l-lsqt-u ../reports/p1l-lsqt ../reports/p1l-dpfp`. Large files stay under `/work/xg24i002/x10041/LoopLora/runs/p1l/lsqt/` and `.../lsqt-u/` (`smoke`, `arms`, `audit`, `config-hashes.json`, `release.json`;
the releases are byte-identical to the committed ones, the U records and audit to `reports/p1l-lsqt-u/`).

## Cost (node·h from the job logs; job table in `docs/reports/LSQT.md`)
| item | node·h |
| --- | ---: |
| LSQT, three arms (smokes 0.433, hashes 0.005, reload 0.019, three arm jobs 4.119, audit 0.022, repro 0.057, analysis tests 0.005) | 4.659 |
| U control: preparation (tests 0.038, smoke 0.140, hashes 0.005, reload 0.007) | 0.189 |
| U control: arm job 3725694 (4,487 s) | 1.246 |
| U control: audit 0.011, analysis tests 0.005 | 0.016 |
| **U control total** | **1.452** |
| **LSQT jobs including U** | **6.111** |
| shared WP-L12 test jobs (counted in full in SQ2 and in LSQT) | 0.438 |
| **LSQT including U plus the shared test jobs** (prereg estimate about 6.1 = 4.6 + 1.5) | **6.549** |
| SQ2 jobs (1.740; with the shared tests 2.178) | 1.740 |
| **WP-L12 cumulative** (SQ2 + LSQT including U + the shared tests once) | **8.289** |

## U control (docs/prereg/LSQT.md decision 2; descriptive only; `reports/p1l-lsqt-u/`)
- Release `71a6f1c`: LT-U 24,560 steps (evaluations 4,912 / 9,824 / 14,736 / 19,648 / 24,560), science commit `c07452b`, config hash `a88fe73a30cec96112ee21f7fa45aa9c`, prereg digest `94ed7364...`; reload check 3725676 passed; approval `a0eaddd` committed before the job.
- Job 3725694 (regular-g, 02:00:00, `--scale-up docs/prereg/LSQT.md`, `--schedule`): exit 0, 74.3 min of training, 0.180 s per step, peak CUDA 3.05 GiB.
- Record check: 24,560 steps; the five released marks; config hash and commit equal the release's; 6,834 numbers and 493 train-log rows finite, no NaN or infinity token; the train gradient clip is active at all logged steps (as in every other arm).
- Audit (job 3727489): original 360 / NLL 3.382 (equals the record), zero memory 2 / 8.697, shuffled memory 0 / 10.795, repeated-logit drift 0; Gram reconstruction relative error 7.4e-08.
- Result: 360 of 1,000; does not reach 0.80; read error 0.589 (R-r128 0.631, RG001 0.512); curve 15.9, 24.5, 30.3, 32.8, 36.0 %.
- Split of RG001's +12.6 pp over R-r128: unit query +2.5 pp (standard error 2.1), store +10.1 pp (standard error 2.2). For RG1 +9.1 = +2.5 + 6.6, for K4 +4.8 = +2.5 + 2.3. Matched-fraction differences, caveats (one seed, the split goes through U, different step counts, no ridge-with-learned-query arm): `docs/reports/LSQT.md` "U control".
- Selected checkpoint: the final one (step 24,560), as the registered rule for every arm; it is also U's highest. U enters neither outcome and does not change `reports/p1l-lsqt/decision.json`.

## Not done / open
Nothing is open for LSQT or for the U control. Not run: a second seed, the ridge store with the learned query from scratch, E = 8 (registered: none).
