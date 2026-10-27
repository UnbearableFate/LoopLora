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

## Step 1b — static review fixes (2026-10-27)

Review `docs/review/static_2026-10-27/oracle.md` (head 3a788b1): no defect in the WP-L6 code, runner, configs or tests.
**F1 fixed:** `scripts/run_wp_l6.py` is git mode 100755 (`git ls-files -s`), shebang kept, module-level imports still
stdlib only; `tests/test_wp_l6_runner.py::test_runbook_runs_the_file_directly_on_the_login_node` runs it as a command
(step 2). The `size` and `decide` commands above stand as written. **F2 is the plan below; no P1L-S code is written.**

## P1L-S implementation plan (step 3a; not started)

Starts when the orchestrator says wp/p1l is merged into master (its compute tests passed) and runs while the WP-L6
pilot runs. Base and baseline of every default-equality test: master at that merge (it then holds sn-p1l's
`memory_use_check` and the removed launcher deadline). P1L-S itself runs only if `reports/wp-l6/decision.json`
selects a code. Sources: review F2.1–F2.8 and "Pipeline work that precedes the smoke" in `docs/prereg/P1L-S.md`.
Functions, not line numbers, because wp/p1l moves lines.

**Rule.** New behaviour is selected by `--study p1l-s` (`run_p1l_main.py`, `launch_p1l_main.py`,
`analyze_p1l_main.py`) or by `study: "p1l-s"` in the manifest (`evaluate_p1l_main.py`, `finalize_p1l_main.py`; the
authorization must name the same study). Absent means P1-L and that path runs today's statements.
`check_training(manifest, authorization, *, check_all=True)` keeps its signature (sn-p1l's `evaluate_p1l_causal.py`
calls it).

1. **Study profile (writer, config root, lock paths).** `scripts/run_p1l_main.py`: a stdlib `STUDIES` table. `p1l`
   holds today's literals (`configs/p1l`, `reports/p1l`, lock names, `docs/prereg/P1L.md`, job prefix `p1l`,
   `LABELS`); `p1l-s` holds `configs/p1l-s`, `reports/p1l-s`, `docs/prereg/P1L-S.md`, prefix `ps`, trained readers
   B8, M1-L4, B4, B5-L4 (+ reused B3-L4), condition `p1l-s-r32`, `gated_conditions` 1, window `p1l-s`. `config()`,
   `labels_for_rank()` and the `main` branch of `train_stage()` read it. New `scripts/make_p1ls_configs.py` writes
   `configs/p1l-s/main/{B8,M1-L4,B4,B5-L4}-s{17,29,43}.json` from the P1-L files, changing only `name` and `writer`
   (= `decision.json` `selected`, asserted); committed with the S/lr lock before the smoke. `fit()` adds
   `parameters`, `writer_trainable_parameters` and, for `oracle_shared`, the drift of C (tensor-level core of
   `run_wp_l6.code_drift`) for writers other than `oracle_random` only.
2. **One locked confirmation start, 110,000 (F2.2).** The start is a registered window name, not a free integer
   (`tests/test_llm_run.py` keeps rejecting `eval.start` and `eval.episode_start`).
   `src/looplora/llm/config.py`: `CONFIRMATION_WINDOWS = {"p1l": 100000, "p1l-s": 110000}`, `split_window(eval, split)`
   (`("test", start)` for `confirm`, default `p1l`; `(split, 0)` otherwise), and an optional `eval.confirmation` key in
   `resolve_config` (allow-listed, not in `DEFAULTS`, only with a `confirm` split) so existing resolved configs and
   hashes stay identical. `llm/evaluate.py::evaluate`: generator split/start and `episode_ranges` from `split_window`
   (the two literals). `scripts/evaluate_p1l_main.py`: `swap_successors` and `readout` (they repeat the literal) use
   `split_window`; `main()` merges `{"eval": {"confirmation": <authorization.confirmation_window>}}` into the primary
   config only when the authorization has the field (typed inherits it). `finalize_p1l_main.py::finalize`: the range
   check compares `metrics["episode_ranges"]` and `cfg["eval"].get("confirmation")` with the authorization's window
   (default `p1l`). `check_training` requires `confirmation_window: "p1l-s"` for P1L-S and rejects any window in a P1-L
   authorization. The authorization is the single source, so neither side can be parameterized alone: an evaluation
   that ignores the window fails finalize, a finalize that ignores it fails the lock.
3. **Per-group writer and memory checks (F2.1).** `finalize()` groups primary runs by the training record's
   `config["writer"]`: the 12 new runs under W and the 3 reused B3-L4 runs under `oracle_random` (named in
   `manifest["reused"]`; a mixed set without that entry fails). Fixed-code groups (`oracle_random`, `oracle_rot`): one
   `writer_hash` and identical `(memory, write_count)` snapshots within the group (today's check, per group).
   `oracle_shared`: `global_codes` and `rotations` (hashed from `checkpoint.pt["writer"]`, whose file hash `finalize`
   already verifies) identical across runs and equal to a writer rebuilt from `oracle_seed`; each run's `C` hash and
   `writer_trainable_parameters` (32,800) recorded; no memory comparison. Across groups: equal `write_count` per
   `(split, protocol, condition, batch_start)`; the fact lists need no new check because `pooled_compare_runs`
   already demands identical episode/query keys, hops and answers on every compared pair. `run-manifest.json` records
   the groups, the fixed-code hashes and each run's trained-table hash.
4. **Writer labels (F2.3).** `endpoints(primary, typed, confidence, writer="oracle_random")` and the `writer` field
   of the `accuracy.csv` rows: P1L-S writes W for the new readers and `oracle_random` for B3-L4 (from each run's
   recorded config); `endpoints.json` also records `study`, `confirmation_window`, `writer_by_reader`. P1-L keeps its
   literals (`oracle_random`, `none` for B0/B1/B2). The `results.md` headline and `plot()` titles take the study name.
5. **P1L-S memory-use check (F2.6).** Reuse sn-p1l's `memory_use_check(primary, confidence)` unchanged, called with
   the finalize confidence (0.95 once `gated_conditions` is 1, as `finalize` already derives). `memory_use_lines`
   gets a `study` argument: P1-L keeps its Rule 2/3 sentences, P1L-S prints the decision text of its prereg (B7
   failure: the readers did not learn to use the oracle memory at r = 32 with W; G1-L, G2-L and B6 reported, not
   interpreted). Added for P1L-S only, derived from values `endpoints()` already computes: a `B6_equal_parameter`
   flag (`M1_vs_B4` meets the G1-L criterion: ≥ 3 pp, CI > 0, positive in every seed).
6. **`check_training` for 12 new + 3 reused runs, k = 1 (F2.4).** `scripts/evaluate_p1l_main.py::check_training`,
   `study == "p1l-s"`: expected names are B8, M1-L4, B4, B5-L4 and B3-L4 × seeds 17/29/43, no B0/B1/B2;
   `gated_conditions == 1` and `condition_manifests == {"p1l-s-r32": …}`; the authorization names the study, the
   window and the committed WP-L6 `decision.json` (path and hash; `status: selected`, `selected == manifest["writer"]`);
   new runs: `cfg["writer"] == manifest["writer"]`, `oracle_seed` 17; reused runs: label B3-L4,
   `cfg["writer"] == "oracle_random"`, S and lr as in P1-L's lock, and a `manifest["reused"]` entry whose source path,
   manifest hash and checkpoint hash match P1-L's `training-manifest.json`. The shared checks (complete record,
   checkpoint hash, rank, no smoke or confirmation access, S/lr lock) are unchanged; P1-L's k ≥ 2 and two-condition
   rules stay on the P1-L path.
7. **Launcher (F2.5).** `scripts/launch_p1l_main.py`: `walltime()` keeps sn-p1l's removal of the deadline (if absent at
   the merge, remove it: it refuses every launch after 2026-10-28 00:00 UTC); `submit()` takes `scale_up`
   (`docs/prereg/P1L.md` / `docs/prereg/P1L-S.md`); `launch()` takes lock path, condition, job prefix and groups from
   the study (pilot: the four readers; main: one single-run job per reader and seed, as the prereg's cost table
   assumes; no B4+B3 packing, no B1/B2/B0); its `training-manifest.json` carries `study`, `writer`, `condition`,
   `reused` and the lock hash. Job names are `ps-<stage>-<tag>`: none of the ledger prefixes (`pf-p1l`, `size-p1l`,
   `sm-p1l`, `t-p1l`, `p1l-`, `ev-p1l`, `fin-p1l`) matches, so P1-L's ledger never counts P1L-S jobs;
   `p1l_job_ledger.py` gets `--study` (prefix `ps-`, task label, since-date) for P1L-S's own node·h.
   `scripts/analyze_p1l_main.py`: `timings`, `size`, `select` take the label set from the study (S = 17,000, the pilot
   length and the lr rule unchanged).
8. **C in `inference.json` (F2.7).** `evaluate_p1l_main.py::inference_timing` adds `writer_parameters`
   (`_parameter_counts(experiment)["external_writer"]`) only when the writer has trainable parameters, so P1-L files are
   unchanged; `finalize` adds the column to `inference.csv` for P1L-S only (32,800 per memory reader under O-shared, 0
   for the reused B3-L4).
9. **Evaluation templates (F2.8).** `configs/p1l/main/eval-confirmation.json` and `eval-typed-replication.json` stay
   as they are and `main()` keeps loading them from `configs/p1l/main` for both studies (same design, no start inside
   them); the window comes only from the authorization (item 2). `--directions` is P1-L's cache: the name-gradient
   probe runs on the frozen backbone without adapters, so it does not depend on the writer.

**Tests (compute node; written first, checked with py_compile and pyflakes on the login node).**
- `tests/test_p1ls_defaults.py`, P1-L defaults unchanged, differential against the baseline commit as in
  `tests/test_llm_oracle_offpath.py` (baseline files from `git show` in a temp tree, both versions on identical
  synthetic inputs): `resolve_config` hash of every committed P1-L config (the off-path probe, with `evaluate.py`
  added to `CHANGED`); `evaluate()` with `generate` stubbed to record `(split, start)` and the `episode_ranges`
  (default confirm = `("test", 100000)`; no confirmation episode is generated); `check_training` on one accepted and a
  battery of damaged manifests and authorizations (same exception text); `endpoints` on the cohort fixtures of
  `tests/test_p1l_main_gates.py`; the launcher with `subprocess.run` stubbed (command lines, receipts,
  `training-manifest.json`, pilot and main); `finalize` end to end on a synthetic 22-run matrix (byte-identical
  `endpoints.json`, `accuracy.csv`, `inference.csv`, `run-manifest.json`, `results.md`).
- `tests/test_p1ls_pipeline.py`, new behaviour: window (110000 recorded; finalize rejects 100000 under a `p1l-s`
  authorization and 110000 under a P1-L one; `resolve_config` rejects an unknown window and one without a `confirm`
  split); `check_training` accepts 12 + 3 and rejects a missing or extra run, B0, a reused run with
  `writer != oracle_random`, a new run under another writer, k = 2, a missing window, a decision that is not
  `selected`; finalize groups (O-rot accepts; O-shared accepts differing `C` and rejects differing `global_codes`,
  `rotations` or a write-count mismatch; mixed writers without `reused` reject); labels; memory use at 95% on passing
  and failing cohorts; C in `inference.json`; launcher names, `--scale-up`, lock path, no deadline.
- Real artifacts, compute: one reused P1-L B3-L4 checkpoint loads under the new code and scores a few **validation**
  episodes (never a confirmation range); the existing `tests/test_p1l_main_gates.py`, `tests/test_p1l_main_readout.py`
  and the WP-L6 tests pass unchanged.

**Order.** (1) wp/p1l merged. (2) Items 1–9 and the tests, login static checks, then compute tests during the pilot.
(3) Only if `decision.json` selects W: make the 12 configs, commit them with the lock, smoke (1,700 steps per reader),
`size`, Pilot-2, lock, main, authorization (committed once), evaluation (`ps-ev-<reader>` jobs; the evaluation map of
P1L-state §4 covers all 15 names, B3-L4 into P1L-S's own root), finalize, `docs/reports/P1L-S.md`.

**Not touched.** `scripts/split_p1l_packed_jobs.py` (handover warning), the P1-L locks and manifests, and sn-p1l's
P1L-C code: `p1lc.py` has its own [102000, 104000) window, so the prereg's "the same mechanism P1L-C needs" holds only
as a need, not as shared code. Review observation, no action: the pilot exercises Π_0 only (`t2_chain_end` writes
relation 0); Π_1–Π_3 first act on the LM path in P1L-S's typed replication, so a pilot pass says nothing about them.
