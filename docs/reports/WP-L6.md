# WP-L6 step 1 — oracle code designs in the LLM pipeline (2026-10-27)

**Implemented and committed on `wp/oracle` (step 1: e4a13cd, from master dc659e0; not merged into master, not
pushed). Nothing has run on a GPU.** The orchestrator reviewed the preregs (0b1c109) and registered them on master
(4ac2410); master is merged into this branch (e5c618d, no conflict). The login node allows no torch, pytest or run, and
Miyabi is in maintenance until 2026-10-28 22:00 JST. Checked so far: `py_compile` and `pyflakes` on all 9 changed or
new Python files (login Python 3.9; the project targets 3.13), both pilot configs resolve, and the runner's `decide`
and `size` rules ran on synthetic records (stdlib only). All other code, every test included, is desk-checked and
first runs in step 2. Step 1b (F1 fix, the P1L-S plan) and step 1c (the P1L-S pipeline, implemented early on top of
wp/p1l 02fc561) have their own sections below; **the WP-L6 tests, smoke and pilot stay pinned to 7175e09**.

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
unchanged:** the recorded CPU code hash `55179f86…` is reproduced, and the four changed files (five since step 1c, which
adds `evaluate.py`), put back to dc659e0 in a copy of `src`, must give identical code tables, memory, resolved
hashes of all committed P1-L configs, tiny-model
training of M1/B8/B4/B5/B3, `run_one` records and `compare_suite` (needs dc659e0 reachable; `LOOPLORA_REPO`
overrides).

**Choices the plan left open** (conservative; in the prereg). C trains at the reader's lr and warmup, weight decay 0,
clipped jointly. A non-finite loss, evaluation value or gradient norm fails that arm and the other still runs;
infrastructure failures rerun the unchanged config. The 567-step smoke uses a derived seed and validation seed and
times a full 1,000-episode evaluation; its scores are never read. The WP-L6 run is not a P1L-S Pilot-2 candidate.

**Step-2 commands (pinned to 7175e09 since step 1c).** `W=/work/xg24i002/x10041/LoopLora-wt/oracle`,
`C=/work/xg24i002/x10041/LoopLora`, `R=$C/runs/wp-l6`, `S=$W/scripts/submit.py`,
`H0=7175e09471fc548c5f376af38cd6a2d2f8661ec3` (the step-1b commit, the last one before the P1L-S pipeline). The tests,
the smoke and the pilot run from H0 as `--commit` snapshots, so the pilot does not depend on the P1L-S pipeline code
that later commits add: `llm/config.py`, `llm/evaluate.py` and the P1-L scripts differ after H0, while
`scripts/run_wp_l6.py`, `llm/oracle.py`, `llm/run.py`, `llm/train.py`, `configs/wp_l6/` and the WP-L6 tests are
byte-identical in every later commit. The test jobs therefore use `--commit $H0` instead of `--worktree $W` (which would
test whatever the worktree holds), and the two login-node steps run from a plain export of H0 (`P0`). The preregs are
already registered.
```
P0=$(mktemp -d) && git -C $C archive $H0 | tar -x -C $P0
# 1-2 tests, then regression of the existing LLM/oracle tests (debug-g, blocking)
$S --name wp-l6-tests --walltime 00:30:00 --commit $H0 --wait -- -m pytest -q tests/test_llm_oracle_codes.py tests/test_llm_oracle_offpath.py tests/test_wp_l6_runner.py
$S --name wp-l6-regr --walltime 00:30:00 --commit $H0 --wait -- -m pytest -q tests/test_llm_oracle.py tests/test_llm_run.py tests/test_llm_train.py tests/test_llm_evaluate.py tests/test_llm_determinism.py tests/test_sym_writer_compatibility.py tests/test_p1l_main_gates.py tests/test_p1l_main_readout.py
# 3 time-only smoke, both arms
$S --name wp-l6-smoke --walltime 00:20:00 --commit $H0 --wait -- scripts/run_wp_l6.py smoke --out $R/smoke
# 4 login node: packed or separate (packed only if 1.5 x (summed expected seconds + 90 s) <= 1 h)
$P0/scripts/run_wp_l6.py size --smoke $R/smoke --out $W/reports/wp-l6/sizing.json
# 5a packed: walltime = packed.walltime from sizing.json (regular-g above 30 min; no --wait)
$S --name wp-l6-pilot --queue regular-g --walltime <packed.walltime> --commit $H0 -- scripts/run_wp_l6.py pilot --out $R/pilot
# 5b separate: one job per arm, walltime = arms.<arm>.walltime
$S --name wp-l6-rot --queue regular-g --walltime <w> --commit $H0 -- scripts/run_wp_l6.py pilot --arms oracle_rot --out $R/pilot-rot
$S --name wp-l6-shared --queue regular-g --walltime <w> --commit $H0 -- scripts/run_wp_l6.py pilot --arms oracle_shared --out $R/pilot-shared
# 6 login node: the registered decision (packed: $R/pilot/<arm>; separate: $R/pilot-rot/oracle_rot, $R/pilot-shared/oracle_shared)
$P0/scripts/run_wp_l6.py decide --oracle_rot <dir> --oracle_shared <dir> --out $W/reports/wp-l6/decision.json
# 7 commit the evidence: P1L-S reads the decision from the commit (git show) and its authorization hashes it
cd $W && git add reports/wp-l6/sizing.json reports/wp-l6/decision.json && git commit -m 'WP-L6: smoke sizing and the registered decision'
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
(step 2). The `size` and `decide` commands stand as written (step 1c runs them from an export of 7175e09).
**F2 is the plan below; step 1b wrote no P1L-S code (step 1c implements it).**

## P1L-S implementation plan (step 3a; implemented in step 1c, see the last section)

*Status: the orchestrator moved this up (step 1c): it is implemented on wp/p1l 02fc561 (merged into wp/oracle, 0fd1550),
which already holds sn-p1l's `memory_use_check` and the removed launcher deadline; the baseline of the default-equality
tests is that commit until wp/p1l is in master. The text below is the plan as registered in 7175e09; where the code
differs, "Where the implementation differs from the plan text" in the last section lists it.* P1L-S itself runs only if
`reports/wp-l6/decision.json` selects a code.
Sources: review F2.1–F2.8 and "Pipeline work that precedes the smoke" in `docs/prereg/P1L-S.md`.
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

## Step 1c — the P1L-S pipeline, implemented (2026-10-27)

**Commits on `wp/oracle`.** `0fd1550` merges wp/p1l 02fc561 (parents 7175e09 and 02fc561; no conflict). `47949f0` is the
pipeline and its tests (15 files, +2,194 −91). `0c00af6` holds the fixes from an independent review (see below; 3 files,
+112 −13). The commit that carries this section is the head, **H1** below (`git rev-parse wp/oracle`). No merge into
master, no push, no GPU run, no job submitted. WP-L6's pinned files (`scripts/run_wp_l6.py`, `llm/oracle.py`,
`llm/run.py`, `llm/train.py`, `configs/wp_l6/`, `tests/test_llm_oracle_codes.py`, `tests/test_wp_l6_runner.py`) are
byte-identical to 7175e09 (`git diff 7175e09 HEAD -- <paths>` is empty), and `docs/prereg` is untouched (`git diff
0fd1550 HEAD -- docs/prereg` is empty). The code is dormant unless `reports/wp-l6/decision.json` selects an oracle code.

**Rule (as planned).** `--study p1l-s` (`run_p1l_main.py`, `launch_p1l_main.py`, `analyze_p1l_main.py`,
`p1l_job_ledger.py`) or `study: "p1l-s"` in the manifest (`evaluate_p1l_main.py`, `finalize_p1l_main.py`, with the
authorization naming the same study) selects the new behaviour; absent means P1-L, whose statements run unchanged.
Signatures: an AST comparison with 02fc561 of the eight changed modules (47 surviving functions) shows no positional
parameter changed or removed, and every added parameter has a default. The only top-level names that disappeared are
`labels_for_rank` in `launch_p1l_main` and `analyze_p1l_main` (each now imports `trained_labels`; nothing imports the
old name from there; `run_p1l_main.labels_for_rank` stays, with a defaulted `study`).

| Plan item | Implemented in |
| --- | --- |
| 1 Study profile | `run_p1l_main.py`: `STUDIES` (`p1l`, `p1l-s`), `labels_for_rank(rank, study)`, `trained_labels`, `lock_file`, `config(label, seed, rank, study)`; `--study`, with an error for r ≠ 32 or a label the study does not train. `fit()` adds `writer_trainable_parameters` and `parameters` for `oracle_rot`/`oracle_shared` and `code_drift` for `oracle_shared` only (P1-L records keep their fields). New `scripts/make_p1ls_configs.py` (mode 755, stdlib): the 12 configs, P1-L's files with only `name` (`p1l-s-<reader>-s<seed>`) and `writer` replaced, from `reports/wp-l6/decision.json` (status `selected`), never overwriting a differing file, all-or-nothing |
| 2 One locked window | `llm/config.py`: `CONFIRMATION_WINDOWS = {"p1l": 100000, "p1l-s": 110000}`, `split_window(eval, split)`, optional allow-listed `eval.confirmation` (only with a `confirm` split; absent keeps every resolved P1-L config and hash; `eval.start`/`episode_start` still rejected). `llm/evaluate.py` and `evaluate_p1l_main.py` (`swap_successors`, `readout`) read the window through `split_window`; `main()` merges the authorization's `confirmation_window` into the confirmation template (typed replication inherits it); `finalize` compares `metrics["episode_ranges"]` and `cfg["eval"]["confirmation"]` with the authorization's window |
| 3 Per-group code audit | `finalize_p1l_main.py`: `audit_writer_groups` (the retrained runs under W and the reused runs under `oracle_random` must be exactly the manifest's split; write counts equal across all runs), `audit_fixed_code` (one `writer_hash` and identical snapshots per fixed-code group), `audit_shared_code` (`global_codes` and `rotations` of every run's checkpoint equal to a writer rebuilt from `oracle_seed`; the recorded trained-parameter count equals the size of C (32,800 at r = 32); each run's C hashed and recorded, not compared; drift recorded); all of it in `run-manifest.json` |
| 4 Writer labels | `endpoints(..., writer)`; `accuracy.csv` `writer` from each run's recorded config; `endpoints.json` adds `study`, `confirmation_window`, `writer_by_reader`; headline and plot titles name the study |
| 5 B7 and B6 | `memory_use_check` unchanged (95% at k = 1); `memory_use_lines(rank, memory, study, gates)` prints, after a B7 pass, the line of `docs/prereg/P1L-S.md` that matches the (G1-L, G2-L) outcomes, and the B7-fail text otherwise; `meets_g1_criterion` and `B6_equal_parameter` in `endpoints.json` and `results.md` |
| 6 `check_training` | `evaluate_p1l_main.py`: `check_study_training` (12 + 3 runs, k = 1, window, committed `decision.json` by path and hash with `selected == manifest["writer"]`, reused entries re-verified against P1-L's `training-manifest.json`), `check_run` shared with the P1-L path; signature unchanged |
| 7 Launcher, analysis, ledger | `launch_p1l_main.py`: `submit(..., scale_up)`, `selected_writer` (the decision and the 12 configs are read with `git show` at the commit the jobs run from), `reused_runs`, pilot = four readers at seed 17, main = 12 single-run jobs, names `ps-<p\|m>-<tag>`, `--scale-up docs/prereg/P1L-S.md`, manifest with `study`, `writer`, `condition`, `reused`; `analyze_p1l_main.py` takes the labels from the study and reads `<root>/smoke/<label>-s17`; `p1l_job_ledger.py --study p1l-s` (prefix `ps-`, which no existing job name has) |
| 8 C in `inference.json` | `inference_timing` adds `writer_parameters` only when nonzero; `inference.csv` gets the column for P1L-S |
| 9 Evaluation templates | unchanged files in `configs/p1l/main`, used by both studies; the window comes only from the authorization |

**Where the implementation differs from the plan text.**
1. `fit()` imports `code_drift` from `run_wp_l6.py` (lazily, `oracle_shared` only) instead of a shared tensor-level core:
   extracting it would edit a WP-L6-pinned file.
2. `writer_parameters` is `sum(experiment.writer_trainable_parameters())`, not `_parameter_counts(...)["external_writer"]`
   (same value: 32,800 for `oracle_shared`, absent otherwise).
3. `launch_p1l_main.py` has `--reuse <P1-L r = 32 training-manifest.json>` (required by, and only valid for, the P1L-S
   main stage, and it must be an absolute path because every later job re-reads it from its own working directory);
   its entries (`source`, `manifest`, `manifest_hash`, `checkpoint_hash`) are what `check_study_training` re-verifies.
4. The WP-L6 decision lives at `reports/wp-l6/decision.json` (profile key `decision`); the authorization names it as
   `decision: {path, hash}`.
5. `check_training` on the P1-L path now also refuses an authorization that carries `confirmation_window` or names
   another study. P1-L's authorization has neither (`reports/p1l/confirmation-lock.json` keys; the other callers,
   `compare_p1l_ranks.py` and sn-p1l's `evaluate_p1l_causal.py`/`finalize_p1l_causal.py`, pass P1-L's own); this is the
   plan's "rejects any window in a P1-L authorization".
6. In `launch()` the P1-L group-building block is indented one level deeper (diff noise; the differential test compares
   its command lines, receipts and manifests).
7. `memory_use_lines` takes `gates = (G1-L, G2-L)` as well as `study`: after a B7 pass, `results.md` states what the
   outcomes allow (the prereg's decision table: both pass, G1-L fails, G1-L passes and G2-L fails) instead of one
   sentence for every pass.
8. The default-equality test takes `src`, `scripts` and `configs` of the baseline whole from `git archive` (a consistent
   tree) instead of single `git show` files.

**Independent review.** A fresh read-only subagent was given the diff of 47949f0 and the specifications, not my
conclusions. It found no wrong run accepted, no valid run rejected, no change on the P1-L path and no pytest or torch
pitfall, and reported three items, all fixed in `0c00af6`: the launcher checked the working-tree configs although jobs
run from the commit (now `git show`, test added); the B7-pass sentence read as "composition shown" even when G1-L
failed (item 7 above; cohorts with G1-L failing and with G2-L failing added); no test loaded a real reused checkpoint
(added, see below). It ran no GPU test, `evaluate_part` or real pytest, as the login-node rules forbid.

**Tests (compute node; their first run is the H1 jobs below).**
- `tests/test_p1ls_defaults.py`: P1-L defaults unchanged. `src`, `scripts` and `configs` of the baseline commit
  `BASELINE` (wp/p1l 02fc561 for now) come from `git archive`; `tests/assets/p1ls_defaults_probe.py` runs in a fresh
  process on that tree and on this checkout over the same synthetic inputs, and every part of its record must be equal:
  resolved hashes of the 40 committed P1-L configs and both confirmation templates, `run_p1l_main` labels and configs,
  the launcher's command lines, receipts and manifests (pilot and main at r = 32 and 128; 7/7/19/5/15 commands),
  `analyze` size and select, `check_training` on 23 inputs (2 accepted, 21 rejected; the same outcome and exception
  text on both trees, at least 10 distinct texts), `endpoints` and the memory-use text, `finalize`'s output files on a synthetic r = 32 matrix, and
  `evaluate()`'s confirmation window and `episode_ranges` (generator stubbed, no test episode generated).
- `tests/test_p1ls_pipeline.py` (60 test functions, 236 cases; `tests/assets/p1l_matrix.py` builds the synthetic P1-L
  and P1L-S matrices from files only): windows (registered, disjoint from P1-L's and P1L-C's, templates carry none,
  one-key config difference, rejections); `check_training` (16 authorization, 15 manifest, 14 record and 6 reuse
  cases, lock-check semantics, decision rule, cross-study refusal); `evaluate_p1l_main.main` with stubs (window in both
  confirmation configs for P1L-S, absent for P1-L); `finalize` (writers, B6 and B7 text for all three outcomes,
  `writer_parameters`, range and study checks before any evaluation is opened, missing evaluation, group and
  fixed-code damage, `meets_g1_criterion` boundaries, `audit_writer_groups`, `audit_shared_code`); the control plane on a
  stand-in repository (launcher main and pilot, receipts, idempotence, job names, argument rules, decision, committed
  configs, reuse, sizing and lock errors, existing output; run profile; `make_p1ls_configs` write, atomicity and
  refusal; analyze; ledger per study; import without torch). Eleven cases need the GPU model or real artifacts:
  tiny-model `fit` code fields for the three writers (3), `inference_timing` (3), `evaluate()` with a recording
  `generate` that returns validation episodes only (4), and the reuse test (1): P1-L's real r = 32 B3-L4 seed-17
  checkpoint (the run P1L-S reuses) loads under this code, the `oracle_random` code this tree builds from its config
  has the `writer_hash` P1-L recorded, and 16 validation episodes score (no value is asserted; it skips if
  `runs/p1l/main/training-manifest.json` is absent).
- `tests/test_llm_oracle_offpath.py` and `tests/assets/oracle_offpath_probe.py`: `evaluate.py` joins `CHANGED` (five
  files; it imports `split_window` from the changed `config.py`, so a dc659e0 `config.py` cannot serve it).

**Checked here and not.** Static: pyflakes on the 21 Python files changed since 02fc561 shows only the 7 unused-import
warnings that 02fc561's copies of the six P1-L scripts have too (same messages); `py_compile` passes on Python 3.9 and
3.13 (bytecode redirected out of the worktree; `__pycache__` count 0); the five stdlib control-plane scripts
(`launch_p1l_main`, `p1l_job_ledger`, `make_p1ls_configs`, `run_p1l_main`, `analyze_p1l_main`) import and print `--help`
under the login node's Python 3.9. Executed on the login node with stand-ins, **not pytest, not torch**:
`tests/test_p1ls_pipeline.py` under a stdlib mini-pytest and a numpy-only fake torch, on the final code: 225 passed,
11 skipped (the GPU or real-artifact cases above), 0 failed; the existing `tests/test_p1l_main_gates.py` (5) and
`tests/test_p1l_memory_use.py` (8) pass under the same shim; the probe's CPU parts, run on a `git archive` of 02fc561 and
on this tree after the review fixes, give equal records. Against the real artifacts (hashes and JSON only):
`make_p1ls_configs.make` gives 12 configs for both codes that differ from P1-L's files in `name` and `writer` only
and serialize identically; `launch_p1l_main.reused_runs` and the per-run reuse checks hold for P1-L's three real
r = 32 B3-L4 runs (complete, checkpoint hash matches, r = 32, no smoke or confirmation access, `oracle_random`, oracle
seed 17, S = 17,000, lr as locked, manifest map and condition as `check_study_training` expects). **Not run:** any real
pytest; the 11 GPU or real-artifact cases of the pipeline module and the probe's evaluate part; the existing torch
tests (`test_llm_*`, `test_p1lc*`, the off-path overlay with `evaluate.py` added); `audit_shared_code` on checkpoints
written by the real `train()` (its tests rebuild the real writer class's state under torch on compute, but no trained
run has been audited); every compute job.

**Commands.** `W=/work/xg24i002/x10041/LoopLora-wt/oracle`, `C=/work/xg24i002/x10041/LoopLora`, `S=$W/scripts/submit.py`.
*WP-L6 step 2 (tests, smoke, pilot, decision): pinned to 7175e09, see "Step-2 commands" above (H0).* *P1L-S tests, at
the new head H1 (separate from step 2; they may run while the WP-L6 pilot runs; job names start with `ps-`, so the
P1L-S ledger counts them):*
```
H1=$(git -C $W rev-parse HEAD)     # full hash: submit.py resolves refs in the central checkout
$S --name ps-t-pipeline --queue regular-g --walltime 01:00:00 --commit $H1 -- -m pytest -q tests/test_p1ls_pipeline.py
$S --name ps-t-defaults --queue regular-g --walltime 01:00:00 --commit $H1 -- -m pytest -q tests/test_p1ls_defaults.py
$S --name ps-t-oracle --walltime 00:30:00 --commit $H1 -- -m pytest -q tests/test_llm_oracle_codes.py tests/test_llm_oracle_offpath.py tests/test_wp_l6_runner.py
$S --name ps-t-regr --walltime 00:30:00 --commit $H1 -- -m pytest -q tests/test_llm_oracle.py tests/test_llm_run.py tests/test_llm_train.py tests/test_llm_evaluate.py tests/test_llm_determinism.py tests/test_llm_b8.py tests/test_llm_prefetch.py tests/test_sym_writer_compatibility.py
$S --name ps-t-p1l --walltime 00:30:00 --commit $H1 -- -m pytest -q tests/test_p1l_main_gates.py tests/test_p1l_main_readout.py tests/test_p1l_memory_use.py tests/test_p1lc.py tests/test_p1lc_stats.py
```
Five single-node jobs, at most 3.5 node·h if every request were used. The wall times are guesses (nothing here has run
on a GPU), and the defaults test needs `git` and the central repository on the compute node, as the off-path test does.
The last three jobs re-run the existing tests that call `resolve_config`, `llm.evaluate` or the P1-L scripts directly
(the step-2 regression list, `test_llm_b8`, `test_llm_prefetch`, and the P1-L tests that came with the merge). The other
`test_llm_*` suites reach `config.py` only through `llm/run.py` and `llm/train.py` and are not re-run; add
`tests/test_llm_*.py` to `ps-t-regr` for a full sweep. Once wp/p1l is in master after its compute tests: merge master into wp/oracle, set
`BASELINE` in `tests/test_p1ls_defaults.py` to that merge commit (reachable from this checkout or the central one, or
`LOOPLORA_REPO`), and rerun `ps-t-defaults`.

*P1L-S step 3, only if `reports/wp-l6/decision.json` is committed with status `selected` (step 2 above commits it) and
P1-L's r = 32 runs have finished; `P1LM` is P1-L's raw `training-manifest.json`, identical today to the committed
`reports/p1l/training-manifest.json`. All paths absolute; compute jobs run from a commit snapshot, so every file they
read must be committed first.*
```
R=$C/runs/p1l-s; P1LM=$C/runs/p1l/main/training-manifest.json; A=$W/reports/p1l-s/confirmation-authorization.json
cd $W && python3 -B scripts/make_p1ls_configs.py && git add configs/p1l-s && git commit -m 'P1L-S: configs under the WP-L6 code'
H2=$(git rev-parse HEAD)
# smoke: 4 readers x 1,700 steps, time only (about 0.35 node.h); never read its scores
$S --name ps-s-smoke --queue regular-g --walltime 00:45:00 --commit $H2 --wait -- scripts/run_p1l_main.py smoke --out $R/smoke --steps 1700 --study p1l-s
$S --name ps-size --walltime 00:05:00 --commit $H2 --wait -- scripts/analyze_p1l_main.py size --root $R --out $W/reports/p1l-s/sizing.json --study p1l-s
# Pilot-2: one job per reader, lr 1e-3 and 3e-3 at 5,666 steps (login node; the launcher submits the jobs and records receipts)
python3 -B scripts/launch_p1l_main.py pilot --root $R --sizing $W/reports/p1l-s/sizing.json --study p1l-s
# when all four pilot jobs have finished:
$S --name ps-select --walltime 00:05:00 --commit $H2 --wait -- scripts/analyze_p1l_main.py select --root $R/pilot --sizing $W/reports/p1l-s/sizing.json --out $W/reports/p1l-s/lr-S-lock.json --study p1l-s
git add reports/p1l-s && git commit -m 'P1L-S: sizing and lr/S lock'     # main needs the lock committed at HEAD
python3 -B scripts/launch_p1l_main.py main --root $R --sizing $W/reports/p1l-s/sizing.json --study p1l-s --reuse $P1LM
```
After all 12 trainings have finished, the orchestrator writes the authorization once (never rewritten), commits it, and
takes `P1LS`:
```
python3 -B - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from run_p1l_main import STUDIES, digest
decision = STUDIES["p1l-s"]["decision"]
auth = dict(study="p1l-s", status="allowed", orchestrator_decision="<who, when, why>", gated_conditions=1,
            confirmation_window="p1l-s", decision=dict(path=decision, hash=digest(Path(decision))),
            condition_manifests={"p1l-s-r32": "/work/xg24i002/x10041/LoopLora/runs/p1l-s/training-manifest.json"})
out = Path("reports/p1l-s/confirmation-authorization.json"); assert not out.exists()
out.write_text(json.dumps(auth, indent=2) + "\n")
PY
git add reports/p1l-s/confirmation-authorization.json && git commit -m 'P1L-S: release the confirmation after the 12 trainings'
P1LS=$(git rev-parse HEAD)
$S --name ps-t-preconfirm --walltime 00:10:00 --commit $P1LS --wait -- -c '
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from evaluate_p1l_main import check_training
a = json.loads(Path("'$A'").read_text())
check_training(json.loads(Path(a["condition_manifests"]["p1l-s-r32"]).read_text()), a)
print("P1L-S training matrix verified")'
for label in B8 M1-L4 B4 B5-L4 B3-L4; do     # one job per reader, three seeds; test ! -e before each
  case $label in M1-L4|B8) wall=01:00:00; queue=regular-g;; *) wall=00:15:00; queue=debug-g;; esac
  test ! -e $R/confirmation/$label
  $S --name ps-ev-$label --queue $queue --walltime $wall --commit $P1LS -- scripts/evaluate_p1l_main.py \
    --manifest $R/training-manifest.json --authorization $A --out $R/confirmation/$label \
    --runs $label-s17 $label-s29 $label-s43 --directions $C/runs/p1l/main/jlens/directions.pt | tee -a $R/confirmation-submissions.jsonl
done
```
Then the evaluation map and the finalizer, as in P1L-state §4 (15 names; B3-L4 is evaluated from P1-L's checkpoints into
this study's own root), the ledger, and the reports:
```
python3 -I - <<'PY'
import json
from pathlib import Path
root = Path("/work/xg24i002/x10041/LoopLora/runs/p1l-s")
runs = {name: str(root / "confirmation" / name.rsplit("-s", 1)[0] / name) for name in json.loads((root / "training-manifest.json").read_text())["runs"]}
for name, path in runs.items():
    assert json.loads((Path(path) / "record.json").read_text())["status"] == "complete", name
(root / "evaluation-manifest.json").write_text(json.dumps({"runs": runs}, indent=2) + "\n")
PY
$S --name ps-fin --walltime 00:10:00 --commit $P1LS --wait -- scripts/finalize_p1l_main.py --manifest $R/training-manifest.json --evaluations $R/evaluation-manifest.json --authorization $A --out $W/reports/p1l-s
python3 -B scripts/p1l_job_ledger.py --study p1l-s --out reports/p1l-s/jobs.json --require-final
```
Then `docs/reports/P1L-S.md` and, for a negative result, `reports/decisions/P1L-S.md` (the prereg's decision rules).
The Pilot-2 and main walltimes come from the launcher (1.5 × the measured time, at most 2 h) and the main jobs use
`--scale-up docs/prereg/P1L-S.md`. Margin: `analyze size` refuses S = 17,000 if the slowest reader needs more than
about 0.278 s per step (2 h / 1.5 less the load reserve); P1-L's M1-L4 measured 0.197 s at r = 32.

**Open questions (conservative choice made where the plan was silent).**
1. **Baseline.** The differential baseline is wp/p1l 02fc561; re-point it as above once wp/p1l is in master.
2. **Reuse source.** P1L-S records the blake2b of the P1-L manifest it is given and every later job re-checks it, so a
   rewrite of that file after the P1L-S main launch stops evaluation until the launch is repeated. The committed copy
   `reports/p1l/training-manifest.json` is the stable choice if the raw manifest may still change (identical today).
3. **`code_drift` import** from `run_wp_l6.py` (difference 1): move it into `src/` after the pilot if you want it shared.
4. **`audit_shared_code` compares tensors bitwise** with a CPU rebuild of the seeded writer. The training and the
   finalizer both run with `OPENBLAS_NUM_THREADS=1`, so the QR-built rotations should reproduce exactly; if the first
   real finalize rejects them, the check is cheap to relax and finalize cheap to rerun. Unverified until a compute run.
5. **Off-path overlay test kept.** Decision 3 above allowed retiring part 2 of `tests/test_llm_oracle_offpath.py` once a
   later change edits the four files. Step 1c edits `config.py` and adds `evaluate.py`, but the test still holds for
   `oracle_random` and the P1-L paths by design, so it is kept with five files; retire part 2 (the docstring says how)
   if you prefer: `ps-t-defaults` already covers the same ground against 02fc561.
6. **Test wall times** are guesses; the pipeline module is finalize-heavy and the defaults test runs two probes (each
   limited to 3,000 s inside the test).
7. **Smoke and sizing commands** above are this section's design (the plan did not spell them out): analyze reads
   `<root>/smoke/<reader>-s17` for P1L-S, so the smoke is run with `--out $R/smoke`; `analyze size` and `select` call
   `require_compute()`, hence they are jobs.
