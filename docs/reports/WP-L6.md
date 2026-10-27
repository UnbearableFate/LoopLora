# WP-L6 step 1 — oracle code designs in the LLM pipeline (2026-10-27)

**Implemented and committed on `wp/oracle` (step 1: e4a13cd, from master dc659e0; not merged into master, not
pushed). Nothing has run on a GPU.** The orchestrator reviewed the preregs (0b1c109) and registered them on master
(4ac2410); master is merged into this branch (e5c618d, no conflict). The login node allows no torch, pytest or run, and
Miyabi is in maintenance until 2026-10-28 22:00 JST. Checked so far: `py_compile` and `pyflakes` on all 9 changed or
new Python files (login Python 3.9; the project targets 3.13), both pilot configs resolve, and the runner's `decide`
and `size` rules ran on synthetic records (stdlib only). All other code, every test included, is desk-checked and
first runs in step 2.

| Piece | Where |
| --- | --- |
| `oracle_rot`, `oracle_shared` (key = normalize(Π_ρ g_h) or normalize(Π_ρ C_h); value = g_t or normalize(C_t); definitions in the prereg); `oracle_random` untouched | `src/looplora/llm/oracle.py` |
| `writer` ∈ {learned, oracle_random, oracle_rot, oracle_shared}; same memory-only, r = r_mem ∈ {32, 128}, no-checkpoint rules | `llm/config.py` |
| Writer signatures (oracle_random's verbatim); C in its own AdamW group (reader lr and warmup, weight decay 0), clipped jointly with the reader; export and reload (C saved; shape, fp32, key set checked) | `llm/train.py` |
| Records; C counted as `external_writer` (reader counts unchanged: M1 at r = 32 has 101,376 trainable, 103,426 stored; C adds 32,800); `compare_suite` skips the per-reader shared table | `llm/run.py` |
| Runner `smoke`, `pilot`, `size`, `decide` (stdlib at import) | `scripts/run_wp_l6.py` |
| Pilot configs: `configs/p1l/main/M1-L4-s17.json` except name, writer, steps 5,666, eval depths 1–4 | `configs/wp_l6/` |
| Preregistrations (orchestrator-reviewed 0b1c109, registered on master 4ac2410) | `docs/prereg/WP-L6.md`, `P1L-S.md` |
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
`S=$W/scripts/submit.py`, `H` = full hash of a commit that contains this branch (the preregs are already registered).
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

**Decisions (orchestrator, 2026-10-27).**
1. Preregs: the O-rot interpretation is softer (the pilot does not separate why O-rot would work). P1L-S does not
   retrain B3-L4: it never reads the memory, so P1-L's r = 32 B3-L4 runs (seeds 17/29/43, lr 1e-3, S = 17,000) are
   evaluated on [110000, 112000); 12 new runs, about 14.5 node·h. No registered run trains B3 under the new writers;
   the B3 case of the tiny-model test stays as a code-path check.
2. I own the P1L-S pipeline parameterization (writer, config and report roots, lock paths, confirmation range, CI
   level, O-shared finalize checks): step 3 only if WP-L6 selects a code, and only after sn-p1l's step-A changes are
   merged into master. Defaults must reproduce P1-L exactly; P1-L's confirmation runs from its pinned source commit
   either way. For step 3: `finalize_p1l_main.py` requires one `writer_hash` across all primary runs and equal
   snapshots/write counts across readers. P1L-S mixes writers (new runs under W, reused B3-L4 under oracle_random) and
   O-shared trains a table per run, so the check becomes per group, with a per-run C hash and write counts and fact
   lists compared instead of memory snapshots.
3. Agreed: the dc659e0 overlay test is a one-time proof (retire it if a later change edits the four files, keep the
   hash test); `configs/p1l/README.md` stays unedited (append conflict with sn-p1l; usage is in the prereg and here).

Step 2 starts when the maintenance ends (2026-10-28 22:00 JST).
