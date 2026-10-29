# WP-L8 state: DPFP-X (X256 and X17), done

Branch `wp/llm`, master `21254c9` merged in. Not merged into master, not pushed. Task: `docs/tasks/WP-L8_dpfp_x.md`; registration `docs/prereg/DPFP-X.md`; results `docs/reports/DPFP-X.md`.
**Final registered decision: neither arm reaches 0.80 (X256 66.2 %, X17 59.3 %); the learned-writer line stays stopped.** X17 against D128-3 (60.8 %): -1.5 pp, not distinguished. Nothing remains to run.

## Commits (wp/llm)
| commit | content |
| --- | --- |
| `803076d` | X256 / X17 in the DPFP runner and release scripts, their tests |
| `3f150a8`, `9214790` | test fixes (below). **9214790 is the science commit** of every smoke, run, reload check and audit |
| `61e2a25` | `reports/p1l-dpfp-x/release.json` (differs from 9214790 only by this file) |
| the report commit | records, audit, decision, `scripts/analyze_dpfp_x.py`, `docs/reports/DPFP-X.md`, this file |

## What the code does (nothing in the DPFP-off or DPFP path changed)
- `dpfp_release.py` (torch-free): `XARMS = ('X256', 'X17')`, `XSPECS` (rank, nu, backbone), `arm_spec`, `study_of`, `prereg_digest`; `scheduled_steps` takes the study from the arm
  (DPFP-X: both arms, E16 only). `eligible_e8` still rejects X arms.
- `read_fix.configuration(..., extra_runs=())`: the rank guard `(rank, arm) in RUNS` is lifted only for `(256, 'R')` through the new keyword, used by X256 alone. `RUNS` and `read_fix_run.choose` are untouched.
- `dpfp.configuration`: X17 adds the `backbone` key `{id, revision}` (P5L's optional top-level key, outside `DEFAULTS`) to the config and to the `dpfp` block; X256 and the D arms have none, so their hashes are unchanged.
  `DPFPExperiment` checks arm, nu, rank (X arms) and that the config's backbone equals the arm's. Shapes (d_in, d_out, key dimension 2·r·nu) come from the loaded model and the config.
- `dpfp_run.py`: arms `ARMS + XARMS`; `choose_x` is its own function (`choose` is untouched; E8 retrain forbidden for DPFP-X); with a DPFP-X schedule the runner checks the release's prereg
  blake2b-128 and the arm's frozen config hash before training (`--e8-decision` is refused).
- `scripts/prepare_dpfp_release.py --study DPFP-X` (`--steps-only`, `--config-hashes`), `scripts/dpfp_config_hashes.py` (compute: the science config hashes, since `configuration` imports torch),
  `scripts/check_smoke_reload.py` (DPFP-X study: also the frozen science hash and the prereg digest), `scripts/analyze_dpfp_x.py` (tables of the report).
- The release (`reports/p1l-dpfp-x/release.json`): steps X256 22,780, X17 21,980 (five checkpoints at 20 to 100 %), science commit, config hashes `4fa871ca...` and `abbb0e87...`, prereg `54334bca...`.

## Tests (compute; all pass at the science commit)
- `tests/test_llm_dpfp_x.py` (14): the three DPFP configs and the E8 config keep their recorded hashes; the stable_hash of every WP-L4 run and every DPFP arm (E8/E16, smoke or not) equals the value
  computed at master `21254c9` before any DPFP-X code (job 3722600; literals in the file); the WP-L4 reference records R-r32 / R-r128 are reproduced in full including their stored hashes; the default
  backbone is the pinned 0.6B; the rank guard is lifted only for X256; X configs are D128-3 plus only the registered change; inconsistent arm/config/backbone is rejected; X256 trains, exports and reloads
  at rank 256 on a tiny model ([256, 1536] memory); the backbone option works through the DPFP runner (`run_one`, tiny backbone from a cache directory, shapes from the loaded model, reload without a
  passed model, the frozen-hash refusal); `choose_x` (thresholds at exactly 800, 5.0 pp, error cases) and `choose` untouched; step counts from the release; prereg digest.
- `tests/test_llm_dpfp.py`, `test_llm_dpfp_repro.py` (the DPFP-off reproduction of WP-L4 R), `test_audit_dpfp.py` (now with an X256 export), `test_release_scripts.py` (DPFP-X release mode: hashes,
  refused smoke records, no E8), `test_p5l.py`: passed in jobs 3722584 and 3722607 (every failure in those jobs was in `tests/test_llm_dpfp_x.py`, listed below).
- Test defects found (none a product defect, no assertion weakened):
  1. the X vs D128-3 config comparison ignored the label `writer_fix.value_head` (`training_only_{rank}_to_1024_affine`, rank-dependent by the existing code);
  2. the test read a `backbone` key in `writer_signature()`, which names the pin as `model` and `revision`;
  3. the R-r32 / R-r128 reference check: R-r32 (commit 0b03d60) predates the config defaults `writer` and `oracle_seed`, R-r128 (be07f7f) holds them. First fix stripped the keys for both records (wrong for R-r128, test
     job 3722607 failed); the final test strips them only where the record lacks them and then requires full config equality and the stored hash for both.

## How to reproduce the tables
`python3 scripts/analyze_dpfp_x.py` (stdlib; reads `reports/p1l-dpfp-x/arms/*` and D128-3 from `reports/p1l-dpfp/`). Large files (`checkpoint.pt`, `writer.pt`, `samples.npz`) stay under
`/work/xg24i002/x10041/LoopLora/runs/p1l/dpfp-x/` (smoke, arms, audit, config-hashes.json, decision.json; `arms/X256` and `arms/X17` hold the exports the audit reloaded).

## Cost
Step 1 (tests, hash jobs, smokes, reload) 0.384 node·h; step 2 (two science runs 2.647, audit 0.017, decision 0.005) 2.669; total 3.053 node·h. Job table in `docs/reports/DPFP-X.md`.
Tests, smokes, reload checks and audits together 0.406 node·h (cap 0.6).

## Not done / open
Nothing is open. Descriptive fact for the paper table: both curves were still rising at the final checkpoint (X256 +3.5 pp, X17 +2.3 pp over the last interval); the registered decision uses the final checkpoint.
A reader study with a flagged candidate would have needed its own registration; none was flagged.
