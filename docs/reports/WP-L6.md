# WP-L6 step 1 — oracle code designs in the LLM pipeline (2026-10-27)

**Implemented and committed on `wp/oracle` (from master dc659e0; not merged, not pushed). Nothing has run on a GPU.**
The login node allows no torch, pytest or run, and Miyabi is in maintenance until 2026-10-28 22:00 JST. Checked so
far: `py_compile` and `pyflakes` on all 9 changed or new Python files (login Python 3.9; the project targets 3.13),
both pilot configs resolve, and the runner's `decide` and `size` rules ran on synthetic records (stdlib only). All
other code, every test included, is desk-checked and first runs in step 2.

| Piece | Where |
| --- | --- |
| `oracle_rot`, `oracle_shared` (key = normalize(Π_ρ g_h) or normalize(Π_ρ C_h); value = g_t or normalize(C_t); definitions in the prereg); `oracle_random` untouched | `src/looplora/llm/oracle.py` |
| `writer` ∈ {learned, oracle_random, oracle_rot, oracle_shared}; same memory-only, r = r_mem ∈ {32, 128}, no-checkpoint rules | `llm/config.py` |
| Writer signatures (oracle_random's verbatim); C in its own AdamW group (reader lr and warmup, weight decay 0), clipped jointly with the reader; export and reload (C saved; shape, fp32, key set checked) | `llm/train.py` |
| Records; C counted as `external_writer` (reader counts unchanged: M1 at r = 32 has 101,376 trainable, 103,426 stored; C adds 32,800); `compare_suite` skips the per-reader shared table | `llm/run.py` |
| Runner `smoke`, `pilot`, `size`, `decide` (stdlib at import) | `scripts/run_wp_l6.py` |
| Pilot configs: `configs/p1l/main/M1-L4-s17.json` except name, writer, steps 5,666, eval depths 1–4 | `configs/wp_l6/` |
| Preregistration drafts (the orchestrator commits them before any run) | `docs/prereg/WP-L6.md`, `P1L-S.md` |
| Tests | `tests/test_llm_oracle_codes.py`, `test_llm_oracle_offpath.py`, `test_wp_l6_runner.py`, `tests/assets/oracle_offpath_probe.py` |

**What the tests cover (step 2 runs them).** Rotation orthogonality, seed and QR sign; unit keys; value→key map is the
fixed Π_0 for rot and not linear for random; float64 delta-rule reference; exact overwrite at β = 1; invalid slots never
addressed or written; gradient reaches C only for shared; config rules; tiny-model train, export and reload for both;
optimizer group and joint clipping; parameter counts; run records; `compare_suite`; checkpoint rejection; the
registered decision rule, divergence versus infrastructure failure, and the packing rule. **`oracle_random` bitwise
unchanged:** the recorded CPU code hash `55179f86…` is reproduced, and the four changed files, put back to dc659e0 in a
copy of `src`, must give identical code tables, memory, resolved hashes of all committed P1-L configs, tiny-model
training of M1/B8/B4/B5/B3, `run_one` records and `compare_suite` (needs dc659e0 reachable; `LOOPLORA_REPO`
overrides).

**Choices the plan left open** (conservative; in the prereg). C trains at the reader's lr and warmup, weight decay 0,
clipped jointly. A non-finite loss, evaluation value or gradient norm fails that arm and the other still runs;
infrastructure failures rerun the unchanged config. The 567-step smoke uses a derived seed and validation seed and
times a full 1,000-episode evaluation; its scores are never read. The WP-L6 run is not a P1L-S Pilot-2 candidate.

**Step-2 commands.** `W=/work/xg24i002/x10041/LoopLora-wt/oracle`, `R=/work/xg24i002/x10041/LoopLora/runs/wp-l6`,
`S=$W/scripts/submit.py`, `H` = full hash of a commit that contains this branch and the committed preregs.
```
# 1-2 tests, then regression of the existing LLM/oracle tests (debug-g, blocking)
$S --name wp-l6-tests --walltime 00:30:00 --worktree $W --wait -- -m pytest -q tests/test_llm_oracle_codes.py tests/test_llm_oracle_offpath.py tests/test_wp_l6_runner.py
$S --name wp-l6-regr --walltime 00:30:00 --worktree $W --wait -- -m pytest -q tests/test_llm_oracle.py tests/test_llm_run.py tests/test_llm_train.py tests/test_llm_evaluate.py tests/test_llm_determinism.py tests/test_sym_writer_compatibility.py tests/test_p1l_main_gates.py tests/test_p1l_main_readout.py
# 3 time-only smoke, both arms
$S --name wp-l6-smoke --walltime 00:20:00 --commit $H --wait -- scripts/run_wp_l6.py smoke --out $R/smoke
# 4 login node: packed or separate (packed only if 1.5 x (summed expected seconds + 90 s) <= 1 h)
$W/scripts/run_wp_l6.py size --smoke $R/smoke --out $W/reports/wp-l6/sizing.json
# 5a packed: walltime = packed.walltime from sizing.json (regular-g above 30 min; no --wait)
$S --name wp-l6-pilot --queue regular-g --walltime <packed.walltime> --commit $H -- scripts/run_wp_l6.py pilot --out $R/pilot
# 5b separate: one job per arm, walltime = arms.<arm>.walltime
$S --name wp-l6-rot --queue regular-g --walltime <w> --commit $H -- scripts/run_wp_l6.py pilot --arms oracle_rot --out $R/pilot-rot
$S --name wp-l6-shared --queue regular-g --walltime <w> --commit $H -- scripts/run_wp_l6.py pilot --arms oracle_shared --out $R/pilot-shared
# 6 login node: the registered decision (packed: $R/pilot/<arm>; separate: $R/pilot-rot/oracle_rot, $R/pilot-shared/oracle_shared)
$W/scripts/run_wp_l6.py decide --oracle_rot <dir> --oracle_shared <dir> --out $W/reports/wp-l6/decision.json
```
Cost from P1-L's measured step times: ≈ 1.3 node·h in all (tests ≈ 0.5, smoke ≈ 0.1, pilot ≈ 0.65); every job ≤ 1 node × 1 h.

**Open questions.**
1. P1L-S needs the P1-L scripts parameterized (they hard-code `oracle_random`, `configs/p1l/`, `reports/p1l/` and the
   confirmation start 100000 in `llm/evaluate.py`); sn-p1l's P1L-C needs the same mechanism for [102000, 104000). One
   shared change, owner to be named. Under O-shared the finalize check must record a per-run C hash and compare write
   counts, not memory snapshots.
2. B3-L4 under O-shared never reads memory, so C stays at its initialization but is counted. Say so before P1L-S if B3
   should use a fixed code instead.
3. The dc659e0 overlay test is a one-time proof; retire it if a later change edits the four files, keep the hash test.
4. `configs/p1l/README.md` is not edited (append conflict with sn-p1l); usage of the new writers is in the prereg.
