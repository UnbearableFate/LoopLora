# P1L-C state — steps A and A2 (B7 check, P1L-C implementation, static-review fixes) done; nothing has run

As of 2026-10-27 16:52 JST (step A written 09:46, step A2 added). Author: sn-p1l (Claude Sonnet 5.5). Branch
`wp/p1l`, worktree `/work/xg24i002/x10041/LoopLora-wt/p1l`. Task: `docs/tasks/WP-P1LC_causal.md`; registration:
`docs/prereg/P1L.md` Amendment B7 and `docs/prereg/P1L-C.md` (with Clarification C1).

This is an implementation hand-off, not a result. No test or confirmation episode was generated or scored, no
job was submitted (Miyabi is in dedicated time and maintenance until 2026-10-28 22:00 JST), and nothing was
executed on the login node except static checks. All tests below are written but have never run.

## Status

| Commit | Content |
| --- | --- |
| `a5144d9` | B7 memory-use check in `scripts/finalize_p1l_main.py` + `tests/test_p1l_memory_use.py` |
| `30f4db1` | P1L-C library `src/looplora/llm/p1lc.py`, `scripts/evaluate_p1l_causal.py`, `scripts/finalize_p1l_causal.py` |
| `cdde6af` | `tests/test_p1lc.py` (CPU and tiny-Qwen GPU), `tests/test_p1lc_stats.py` (CPU, synthetic samples) |
| `c5de5c2` | this report, first version |
| `2f22f84` | step A2: merge of master (`712b284`) into wp/p1l |
| `c9b202f` | step A2: the finalizer requires one source tree for all nine runs (+ test) |
| `3a7f111` | step A2: hard-coded 2026-10-28 deadline removed from `launch_p1l_main.walltime` |
| (this file) | step A2: report update |

Pending, in order: step B tests (below), the P1-L confirmation and analysis (with the B7 flag), then, only if
r = 128 passes B7, the P1L-C smoke, the three registered jobs and the finalizer (step C commands below).

## Step A2: fixes from the static review (`docs/review/static_2026-10-27/p1l.md`)

The review found no defect in the registered computations (B7, P1L-C including Clarification C1); two low findings
and notes. Done, still without compute:

1. **Merge (L2, `2f22f84`).** master at `712b284` is merged into wp/p1l, so the tree that `P1L_SOURCE` and
   `P1LC_SOURCE` pin carries P1L-C Clarification C1 and the P1-L B7 wording note. `git diff --stat dc659e0 712b284
   -- src scripts configs tests pyproject.toml uv.lock` lists only the new `scripts/summarize_p1l_training_logs.py`
   (69 lines, not on the confirmation path); no file had changed on both sides, so the merge had no conflicts.
2. **One source for all nine runs (L1, `c9b202f`).** `finalize_p1l_causal.audit` now also compares `git_commit` and
   `source_hash` across the nine records (the episode, world and memory hashes do not depend on the code). New test
   `test_audit_requires_one_source_for_all_nine_runs`: a per-seed job rerun from another commit or source tree is
   rejected, one different source for all nine is accepted.
3. **Launcher (`3a7f111`).** The hard-coded 2026-10-28 00:00 UTC deadline is removed from
   `launch_p1l_main.walltime`: after the maintenance it refused every launch, including a needed resubmission of a
   P1-L training, and `test_job_margin_is_enforced_before_submission` would have failed from 2026-10-28 07:30 JST on
   (now + 1.5 h past the deadline). PBS holds a job that would cross a scheduled stop and `submit.py` enforces the
   per-job cap; the 2 h check in `walltime()` stays. The ban on rerunning the launcher also stays: its packing would
   recreate the cancelled groups (`P1L-state.md`).
4. **This report.** The wording of the Analysis A spec check (N1) and the interrupted-run protocol (N2) are below;
   the open questions are cut down to what Clarification C1 does not settle. Notes N3–N5 need no change; N4 (the B4
   path at L = 1 is not run end to end in the GPU tests) is covered by the smoke, which runs B4-s17 on a real
   checkpoint.
5. `py_compile` (system and venv Python) and `pyflakes` are clean on the edited files (the unused `LABELS` import in
   `launch_p1l_main.py` was already there). The tests run in step B.

## What was built

- **B7 (rule 1).** `memory_use_check(primary, confidence)` in `finalize_p1l_main.py`: per rank, confirmation set,
  hop-1 full-vocabulary EM of B4@1, B8@4 and M1-L4@4 each minus B3-L4@4, through the existing `endpoint()`
  (pooled paired episode bootstrap, 2000 draws, the gate confidence, 97.5%), per-seed values, and
  `passed = every ci_low > 0`. `finalize()` adds `memory_use`/`memory_use_passed` to `endpoints.json`, a
  `memory-use.csv` table and a B7 section in `results.md` (failure sentence, Rule 2 note at r = 32, Rule 3 note at
  r = 128 when passed, worded as in the amendment). `endpoints()` and every gate computation are untouched; a test
  checks that `endpoints()` keys and values are unchanged and disjoint from the B7 contrast names.
- **P1L-C hook.** `p1lc.read_control(adapter, last, last_only=, capture=, donor=, where=)` is a context manager
  that patches ONE adapter instance for ONE scoring call (`unittest.mock.patch.object` on `process_sequence`,
  `_masked_branch` and, for capture/donor only, `_read`) and restores it on exit, also after errors. The adapter
  classes, `core/`, `llm/inject.py` and every other `llm/` module are not edited, so the confirmation path and the
  pinned source behave as before by construction: `git diff dc659e0 --stat -- src/looplora` lists only the new file
  `src/looplora/llm/p1lc.py`, and nothing outside the P1L-C scripts and tests imports it. Idle, the wrapper only
  counts positions; this is tested for bitwise equality on CPU adapters (M1 `iterq`, B8 `unshared`) and on a
  tiny Qwen, and the run itself raises if the idle wrapper, the captured-state replay of each world or an identity
  patch (final and all positions) does not reproduce the plain logits exactly.
- **Analysis A** (`read_position_samples`). Every query of the 2000 episodes in the confirmation batch shape
  (`eval.batch_size` 16, 8 queries per episode). The config is built as `evaluate_p1l_main.py` builds the
  confirmation config (the checkpoint config merged with `configs/p1l/main/eval-confirmation.json`), and that is
  what fixes the task spec and every episode seed; the spec-equality check in `evaluation_config` cannot fail,
  because the two configs differ only in the eval splits and `allow_confirmation`, so it guards nothing by itself.
  Conditions
  `memory_only/deployed` (the plain path) and `memory_only/final_only` (M = 0 at every real question token except
  each row's last, whose state predicts the answer; the module still runs its memory-free branch there), at the
  reader's trained L (B4: 1, M1-L4 and B8: 4). Analysis reads hops 1, 2, 3 and pooled 1–3: final-only minus
  deployed, per seed and pooled, 95% paired episode-bootstrap CI, 2000 draws. Descriptive; gates nothing.
- **Analysis B** (`select_bridge_worlds`, `bridge_patch_samples`; M1-L4 and B8, L = 4). Episodes: the first 256, in
  generation order, whose stratified hop-2 query starts on the long (8-edge) chain. World A is the original
  (s → b_A → t_A). World B replaces only the tail of the first edge by b_B, the head of the other chain's final
  edge b_B → t_B. World C replaces it by c, a seen-pool entity outside the episode that has no outgoing fact (seed
  `derive_seed(4242, "p1lc", "unrelated-entity", absolute episode index)`). Both continuations (b_A → t_A and
  b_B → t_B) stay in every memory. On world A's memory, z_1 after the first read is replaced at the final question
  position by z_1 of world B, of world C, or by z_1^A plus a random direction of norm ‖z_1^B − z_1^A‖ (steps 2–L
  continue on world A's memory); secondary: the same at every real question position. Saved per episode: the
  predictions of the nine conditions (`world_A/B/C`, `{B,C,R}_{final,all}`), the t_A/t_B/b/c tokens and the state
  norms.
- **Statistics** (`finalize_p1l_causal.py`). shift = rate(t_B | B-patch) − max(rate(t_B | C-patch), rate(t_B |
  random patch)) per seed, pooled as the unweighted mean over seeds; 95% percentile bootstrap over the 256 episodes
  shared by all seeds (2000 draws, the max recomputed in each draw). Causal bridge claim for a reader iff pooled
  shift ≥ 10 pp, CI lower bound > 0 and a positive shift in every seed, using the final-position patch; the
  all-position patch is secondary. Also: t_A rates, state norms, and the descriptive replication M1-L4@4 − B4@1
  and B8@4 − B4@1 on hops {2, 3} of this range, per seed, with 95% and 97.5% CIs.
- **Gating and provenance.** The registered run (`evaluate_p1l_causal.py` without `--smoke`) requires
  `--authorization` and `--endpoints`, refuses unless the endpoints file says rank 128, `memory_use_passed` true and
  carries the same authorization record, then runs `check_training(manifest, authorization, check_all=False)`, all
  before the first test episode is generated. Each run writes `record.json` (checkpoint, manifest, authorization and
  endpoints hashes, window, spec/episode/world/memory-snapshot hashes, timings, git commit and source hash, host,
  job id). The finalizer re-checks all nine records, that the hashes shared by readers and seeds agree (episodes,
  worlds, oracle memory snapshots, and the `git_commit` and `source_hash` of the code), the window
  `test [102000, 104000) seed 4242`, and that the nine run directories exist; it then writes `results.json`,
  `results.md`, `read-position.csv`, `replication.csv`, `bridge-rates.csv`, `bridge-shift.csv`, `run-manifest.json`.

## Verification

Done (static only, login node): `py_compile` with `/usr/bin/python3` and the project venv's Python and `pyflakes`
on every new or edited file (clean; `finalize_p1l_main.py` still shows two unused-import warnings that were
already in `dc659e0`); every cross-module name the new code imports was checked to exist; every call into
`core/adapters.py`, `core/wrapper.py`, `llm/train.py`, `llm/batching.py`, `sym/evaluate.py`, `sym/stats.py` and
`data/relations.py` was checked by reading the source (signatures, shapes, call order, the `_masked_branch` row
selection, the deterministic-algorithms mode); the tiny-Qwen and end-to-end tests were traced by hand against the
config defaults, the `prepare`/`score` contracts and the layer-0 hook placement.

Not done: executing anything. In particular no pytest run, no smoke, no timing. The bitwise `torch.equal`
assertions in the GPU tests rest on GEMM rows being independent of other rows' values and on deterministic
kernels; if one fails by a last-bit difference, investigate before relaxing it.

## Open questions

Clarification C1 (`docs/prereg/P1L-C.md`, in this tree through the master merge) registers what the first version of
this section left open; the code was written to that reading and matches it. Settled:

- b_B is the head of the other chain's final edge (no draw; t_B is that chain's end, reached in two hops from s as t_A
  is in world A); only c is drawn with the episode-index seed, a seen-pool entity outside the episode with no
  outgoing fact.
- The patch replaces z_1 only (the recipient's `u` and world A's memory are kept for steps 2–L; the code continues B8
  with its per-step R/U/Q); the random control is one direction per episode and position from the episode-index
  seed.
- P1L-C runs under the P1-L confirmation authorization (it has no record of its own; `evaluate_p1l_causal.py` and
  the finalizer require the r = 128 `endpoints.json` to carry the same record), and a validation-only smoke
  (val [0, 64), timing only) may run before the confirmation.

Still open (conservative choices are implemented; the orchestrator can confirm or redirect):

1. **Which hop-2 query.** The stratified hop-2 query of each episode (the one the confirmation scores at hop 2);
   selection depends on the episode only, so all readers and seeds see the same 256 episodes and worlds.
2. **Control details.** The random direction is isotropic, drawn from a fixed 128-position table per episode (so it
   does not depend on batching) and scaled to the position's ‖z_1^B − z_1^A‖. The C-patch is the registered
   unrelated-entity control and is not norm-matched; state norms are reported.
3. **Analysis A scope.** All 8 hops are scored in the confirmation batch shape and the file check requires 2000
   queries per condition and hop; the analysis reads hops 1–3 as registered. B4 (L = 1) is included because A is
   stated "for each reader at its trained L".
4. **Pooling and CI.** Unweighted seed mean (as the P1-L pooled contrasts), bootstrap over episodes shared by seeds.
   The bridge bootstrap seed is `derive_seed(2718, "p1lc", "bridge-shift", reader, seeds)`, independent of the
   confidence level. The 95% level follows P1L-C.md, not the 97.5% of the P1-L gates.
5. **Replication.** Reported at both 95% and 97.5% (descriptive; P1-L's G1-L and B6 are unchanged by it).
6. **Ledger.** Job names below start with `t-p1l`, `sm-p1l`, `ev-p1l` and `fin-p1l`, which `scripts/p1l_job_ledger.py`
   already lists, so P1L-C node·h land in `reports/p1l/jobs.json` (separable by the `-c` marker). I left the ledger
   script unchanged.
7. **Runtime is not measured.** My inference (no measurement; unverified) is a few minutes per job, far below the
   0.6 node·h registered. Walltime must come from the smoke.

## Step B additions (before the authorization commit)

The pinned `P1L_SOURCE` is `git rev-parse HEAD` after the authorization commit, so everything on `wp/p1l` up to
then is inside it and must be tested first. Run the tests on the exact commit (the authorization commit adds only
a JSON file under `reports/`, so the code tree is the same), with a clean tree:

```bash
P1L_W=/work/xg24i002/x10041/LoopLora-wt/p1l
cd "$P1L_W" && git status --short          # must print nothing
/usr/bin/python3 scripts/submit.py --name t-p1l-c-tests --walltime 00:20:00 --commit "$(git rev-parse HEAD)" --wait -- \
  -m pytest -q tests/test_p1l_memory_use.py tests/test_p1lc.py tests/test_p1lc_stats.py tests/test_p1l_main_gates.py
```

Require exit 0. Fix failures (production or test, after investigating which) and commit before the authorization
commit; do not weaken assertions. If a change is unavoidable after the authorization commit it must not touch the
confirmation path (`scripts/evaluate_p1l_main.py`, `scripts/readout_p1l.py`, `scripts/run_p1l_main.py`, `core/`,
other `llm/` modules).

## Step C commands (only if r = 128 `memory_use_passed` is true)

```bash
P1L_W=/work/xg24i002/x10041/LoopLora-wt/p1l
P1L_R=/work/xg24i002/x10041/LoopLora/runs/p1l/main
P1L_AUTH="$P1L_W/reports/p1l/confirmation-authorization.json"
P1L_ENDPOINTS="$P1L_W/reports/p1l/r128/endpoints.json"      # written by finalize_p1l_main.py (P1L-state.md step 4)
cd "$P1L_W" && git status --short                            # clean
/usr/bin/python3 -I -c "import json,sys; e=json.load(open('$P1L_ENDPOINTS')); sys.exit(0 if e['rank']==128 and e['memory_use_passed'] is True else 1)" \
  || echo "B7 not passed at r = 128: do not run P1L-C; the report says so"
P1LC_SOURCE="$(git rev-parse HEAD)"                          # explicit full hash; submit.py resolves HEAD in the central checkout

# 1. Smoke: validation only, no authorization. Needs the three seed-17 r128 checkpoints complete. It also runs B4
#    (L = 1) end to end, which the tiny-Qwen tests do not. Run directories are never overwritten: for a rerun use a
#    new --out (e.g. causal-smoke-2) and keep the old one.
/usr/bin/python3 scripts/submit.py --name sm-p1l-c-smoke --walltime 00:20:00 --commit "$P1LC_SOURCE" --wait -- \
  scripts/evaluate_p1l_causal.py --smoke --manifest "$P1L_R/r128/training-manifest.json" \
  --out "$P1L_R/r128/causal-smoke" --runs M1-L4-s17 B8-s17 B4-s17

# 2. Walltime per job = 1.5 x measured: 31.25 x read_position_seconds + 16 x bridge_patch_seconds (smoke
#    record.json, summed over the job's three runs) + model load; round up. At most 1 h per job (submit.py
#    refuses more), debug-g if <= 30 min else regular-g. If one job would exceed 1 h, split it by reader.
WALL=00:30:00 QUEUE=debug-g                                  # replace from the smoke measurement

# 3. Registered runs: test [102000, 104000), seed 4242, one job per training seed, one node each.
for seed in 17 29 43; do
  /usr/bin/python3 scripts/submit.py --name "ev-p1l-c-s$seed" --queue "$QUEUE" --walltime "$WALL" \
    --commit "$P1LC_SOURCE" -- scripts/evaluate_p1l_causal.py --manifest "$P1L_R/r128/training-manifest.json" \
    --authorization "$P1L_AUTH" --endpoints "$P1L_ENDPOINTS" --out "$P1L_R/r128/causal" \
    --runs "M1-L4-s$seed" "B8-s$seed" "B4-s$seed" | tee -a "$P1L_R/r128/causal-submissions.jsonl"
done

# 4. After all three exit 0: audit and registered statistics (absolute --out: the job's cwd is scratch).
/usr/bin/python3 scripts/submit.py --name fin-p1l-c --walltime 00:10:00 --commit "$P1LC_SOURCE" --wait -- \
  scripts/finalize_p1l_causal.py --root "$P1L_R/r128/causal" --manifest "$P1L_R/r128/training-manifest.json" \
  --authorization "$P1L_AUTH" --endpoints "$P1L_ENDPOINTS" --out "$P1L_W/reports/p1lc"

# 5. Accounting (login-safe), then docs/reports/P1L-C.md (<= 1 page: results from reports/p1lc/results.md,
#    node·h per job, deviations).
/usr/bin/python3 scripts/p1l_job_ledger.py --out reports/p1l/jobs.json
```

Notes: the registered run directories are created without `exist_ok`, so a repeated submission cannot overwrite
an earlier output. Keep the `--out` root and `P1LC_SOURCE` fixed across the three jobs: the finalizer compares the
episode, world and oracle-memory hashes and the `git_commit` and `source_hash` of all nine runs and rejects a
mixture. This is a recipe, not an idempotent controller. Do not read any P1L-C score before the finalizer has audited
the nine runs. If B7 fails at r = 128, P1L-C is not run and `docs/reports/P1L-C.md` says so in one paragraph.

**Interrupted run (killed at walltime, node failure).** A run is complete only if its directory holds `record.json`
(written last, atomically); a directory without it is interrupted. Move it aside unread and never delete it:

```bash
cd "$P1L_R/r128/causal" && mv "M1-L4-s29" "M1-L4-s29.interrupted-1"    # k = 1, 2, ... ; open nothing inside
```

The job log may be read for the cause (it carries no scores), the run directory not. Then resubmit only the runs of
that job that lack a complete directory (the interrupted one and any that never started; `evaluate_p1l_causal.py`
refuses an existing directory, so a complete run is never rerun), at the same `P1LC_SOURCE` and with the same
arguments, e.g. `--name ev-p1l-c-s29-r1 --runs M1-L4-s29 B8-s29 B4-s29`. The finalizer ignores the moved-aside
directories. Record the event under "Interrupted-run log" below. A job that failed with an exception (for example a
run-time audit) is not an interruption: diagnose it first, and a code change means a new `P1LC_SOURCE` for all nine
runs (the finalizer rejects mixtures), so report to the orchestrator before moving anything.

## Interrupted-run log

None (step C: nine runs complete, 2026-10-29). One line per event: run, job id, cause, moved-aside path, resubmission job id, `P1LC_SOURCE` unchanged.

## Step C job record (2026-10-29, `P1LC_SOURCE` 395e30914149898006d63cd289b20dac15d732fe)

`sm-p1l-c-smoke` 3721352 (0.008 node·h, exit 0); `ev-p1l-c-s17/s29/s43` 3721357/3721358/3721359 (debug-g, 15 min
requested, 0.041/0.041/0.043 node·h, exit 0); `fin-p1l-c` 3721377 (0.007, exit 0, status complete). Receipts in
`runs/p1l/main/r128/causal-submissions.jsonl`. Results and discussion: `docs/reports/P1L-C.md`.

## Constraints that stayed binding

Never rerun `scripts/split_p1l_packed_jobs.py` or `scripts/launch_p1l_main.py`; never target the closed codex
session; no lock or manifest was rewritten; `pyproject.toml`, `uv.lock` and every config are unchanged; hashes are
blake2b; seeds come from `derive_seed`; no test or confirmation score was read or produced.
