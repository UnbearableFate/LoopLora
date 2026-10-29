# WP-L12 state: LSQT (LT-K4, LT-RG001, LT-RG1 at r = 128, from scratch), done; U control prepared

Branch `wp/llm`, master `a3013ff` merged in. Not merged into master, not pushed. Task: `docs/tasks/WP-L12_storage_studies.md`; registration `docs/prereg/LSQT.md` (blake2b-128 `94ed7364...`); results `docs/reports/LSQT.md`.
**Final registered decision: no arm reaches 0.80 (K4 383, RG001 461, RG1 426 of 1,000), no writer candidate. The best ridge arm LT-RG001 (46.1 %) is in the 40 % to < 48.5 % band: neither the W3 trigger nor the falsifier. The trained-LSQ line stays open; the learned-writer line stays stopped.**
All arms are above R-r128 (33.5 %) by +4.8, +12.6 and +9.1 pp and below D128-3 (60.8 %) by 22.5, 14.7 and 18.2 pp; read errors 0.554, 0.512, 0.499.
The conditional U control is proposed (best arm >= 40 %) and prepared (below); its science job is not submitted.

## Commits (wp/llm)
| commit | content |
| --- | --- |
| `ecbe5e5` | LSQT and SQ2 code and tests (`lsqt_release.py`, `lsqt.py`, `lsqt_run.py`, release / hash / reload script modes) |
| `5cff944`, `b637e75`, `dd5e4c8` | test fixes. **dd5e4c8 is the science commit** of the LSQT smokes, hashes and arm jobs |
| `ec91b3a`, `1ac2ee1` | `reports/p1l-lsqt/release.json` (steps 17,970 / 28,005 / 28,245), the orchestrator approval record |
| the report commit | records, audit, decision, `scripts/analyze_lsqt.py`, `tests/test_analyze_lsqt.py`, `docs/reports/LSQT.md`, this file |
| `c07452b` | the LSQT-U release path (study `LSQT-U`, LT-U alone), tests; the science commit of the U smoke, hashes and arm job |

## What the code does (nothing in the LSQ, DPFP or WP-L4 paths changed)
- `lsqt_release.py` (torch-free): `ARMS = (LT-K4, LT-RG001, LT-RG1)`, `CONTROL = LT-U`, `SPECS`, `ETA_BOUNDS = (1e-4, 1)`, `prereg_digest`, `scheduled_steps` (the three arms from an `LSQT` release; the control only from an `LSQT-U` release).
- `lsqt.py`: `configuration` (R at rank 128 + `unit_query` + the store fields, an `lsqt` block; `lsq.STORE_RULES` extracted, output unchanged) and `LSQTExperiment` (checks arm, rank, store, eta, no `lsq` block). `lsqt_run.py`: runner, `load_inference`, `choose_t`.
- `scripts/prepare_dpfp_release.py --study LSQT|LSQT-U`, `dpfp_config_hashes.py --study LSQT|LSQT-U`, `check_smoke_reload.py` (LSQT, LSQT-U; frozen science hash and prereg digest), `audit_dpfp.py` (loader per study), `analyze_lsqt.py`.

## Tests (compute)
- `tests/test_lsqt.py` (27 at dd5e4c8, plus the U release test): registry and prereg digest; configs are R + the registered store fields (and equal LSQ's L128-K4 / L128-RG training config up to name); LSQ config hashes unchanged (24 literals computed on compute at 2396e54, job 3724024);
  rejected configs and mixed arms; scheduled steps; `choose_t` at its thresholds; tiny train, export, reload for all four stores; no other loader accepts an LSQT export.
- `tests/test_llm_dpfp_repro.py` at dd5e4c8 with `tests/test_lsqt.py`: 29 passed, 0 failed (job 3725434). `tests/test_release_scripts.py` (LSQT and LSQT-U modes), `tests/test_audit_dpfp.py` (LT exports), `tests/test_lsq.py`, `tests/test_llm_lsq.py`, DPFP tests: green. The rest of the suite: 2,440 passed, 1 skipped (job 3724076).
- `tests/test_analyze_lsqt.py` (3): the rule at its thresholds and the decision re-derived from the records (jobs 3725511 and 3724910).

## How to reproduce the tables
`cd scripts && python3 analyze_lsqt.py ../reports/p1l-lsqt ../reports/p1l-dpfp` (stdlib). Large files stay under `/work/xg24i002/x10041/LoopLora/runs/p1l/lsqt/` (`smoke`, `arms`, `audit`, `config-hashes.json`, `release.json`; the release is byte-identical to `reports/p1l-lsqt/release.json`).

## Cost
LSQT jobs 4.659 node·h (smokes 0.433, hashes 0.005, reload 0.019, three arm jobs 4.119, audit 0.022, repro 0.057, analysis tests 0.005); with the shared WP-L12 test jobs (0.438) 5.098 node·h (prereg estimate about 4.6). The U control is not included.

## U control (docs/prereg/LSQT.md decision 2; approved for preparation by the orchestrator, descriptive only)
Smoke job `lsqtu-smoke`, release and reload check: see `docs/reports/LSQT.md` "Next" and `reports/p1l-lsqt-u/` once committed. The U science job waits for the orchestrator's `reports/p1l-lsqt-u/approval.json`.

## Not done / open
The U science job (awaits approval). Nothing else is open for LSQT.
