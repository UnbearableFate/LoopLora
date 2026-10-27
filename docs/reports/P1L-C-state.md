# P1L-C state — step A (B7 check and P1L-C implementation) done; nothing has run

As of 2026-10-27 09:46 JST. Author: sn-p1l (Claude Sonnet 5.5). Branch `wp/p1l`, worktree
`/work/xg24i002/x10041/LoopLora-wt/p1l`. Task: `docs/tasks/WP-P1LC_causal.md`; registration:
`docs/prereg/P1L.md` Amendment B7 and `docs/prereg/P1L-C.md`.

This is an implementation hand-off, not a result. No test or confirmation episode was generated or scored, no
job was submitted (Miyabi is in dedicated time and maintenance until 2026-10-28 22:00 JST), and nothing was
executed on the login node except static checks. All tests below are written but have never run.

## Status

| Commit | Content |
| --- | --- |
| `a5144d9` | B7 memory-use check in `scripts/finalize_p1l_main.py` + `tests/test_p1l_memory_use.py` |
| `30f4db1` | P1L-C library `src/looplora/llm/p1lc.py`, `scripts/evaluate_p1l_causal.py`, `scripts/finalize_p1l_causal.py` |
| `cdde6af` | `tests/test_p1lc.py` (CPU and tiny-Qwen GPU), `tests/test_p1lc_stats.py` (CPU, synthetic samples) |

Pending, in order: step B tests (below), the P1-L confirmation and analysis (with the B7 flag), then, only if
r = 128 passes B7, the P1L-C smoke, the three registered jobs and the finalizer (step C commands below).

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
  (`eval.batch_size` 16, 8 queries per episode, task spec equal to the confirmation spec, asserted), conditions
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
  job id). The finalizer re-checks all nine records, that the hashes shared by readers and seeds agree, the window
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

Conservative choices were made and implemented; the orchestrator can confirm or redirect.

1. **b_B.** The registration says b_B is "the head of an edge b_B → t_B on the other chain" drawn with a seed
   derived from the episode index. I use the other chain's FINAL edge (its only such candidate that keeps t_B the
   chain-end answer, which the reader is trained to output); only c is actually drawn. A random mid-chain edge would
   make "answer = t_B" ambiguous with the chain-end answer in world B.
2. **c.** A seen-pool entity outside the episode (appears in no fact except the patched edge in world C), not an
   in-episode chain terminal. Uniform draw from the seen pool minus the episode's entities.
3. **Which hop-2 query.** The stratified hop-2 query of each episode (the one the confirmation scores at hop 2);
   selection depends on the episode only, so all readers and seeds see the same 256 episodes and worlds.
4. **Patch semantics.** Only the reader state z_1 is replaced; steps 2–L use the recipient token's own `u` and world
   A's memory. For B8 the per-step R/U/Q are used for steps 2–4. Patching u as well would change the question the
   reader answers and is not what the registration describes.
5. **Random control.** One isotropic random direction per episode and position (fixed 128-position table seeded by
   the absolute episode index, so it does not depend on batching), scaled to the position's ‖z_1^B − z_1^A‖. The
   C-patch is the registered unrelated-entity control and is not norm-matched; state norms are reported.
6. **Analysis A scope.** All 8 hops are scored in the confirmation batch shape and the file check requires 2000
   queries per condition and hop; the analysis reads hops 1–3 as registered. B4 (L = 1) is included because A is
   stated "for each reader at its trained L".
7. **Pooling and CI.** Unweighted seed mean (as the P1-L pooled contrasts), bootstrap over episodes shared by seeds.
   The bridge bootstrap seed is `derive_seed(2718, "p1lc", "bridge-shift", reader, seeds)`, independent of the
   confidence level. The 95% level follows P1L-C.md, not the 97.5% of the P1-L gates.
8. **Replication.** Reported at both 95% and 97.5% (descriptive; P1-L's G1-L and B6 are unchanged by it).
9. **Authorization.** P1L-C has no authorization record of its own; it reuses the P1-L confirmation authorization
   and the r = 128 `endpoints.json`, so it cannot run before that confirmation was authorized and analyzed. Add a
   separate record if you want an independent release.
10. **Smoke.** `--smoke` uses val [0, 64) (16 patch episodes), needs no authorization, prints only run names and
    seconds, and may therefore run before the confirmation; its predictions must not be read as results. If you
    prefer to run it only after the confirmation, nothing changes.
11. **Ledger.** Job names below start with `t-p1l`, `sm-p1l`, `ev-p1l` and `fin-p1l`, which `scripts/p1l_job_ledger.py`
    already lists, so P1L-C node·h land in `reports/p1l/jobs.json` (separable by the `-c` marker). I left the ledger
    script unchanged.
12. **Runtime is not measured.** My inference (no measurement; unverified) is a few minutes per job, far below the
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

# 1. Smoke: validation only, no authorization. Needs the three seed-17 r128 checkpoints complete.
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
an earlier output; keep the `--out` root and source fixed across the three jobs, as the finalizer compares hashes
across all nine runs. This is a recipe, not an idempotent controller: after an interruption, inspect the receipts
and existing run directories and submit only the missing seeds. Do not read any P1L-C score before the finalizer
has audited the nine runs. If B7 fails at r = 128, P1L-C is not run and `docs/reports/P1L-C.md` says so in one
paragraph.

## Constraints that stayed binding

Never rerun `scripts/split_p1l_packed_jobs.py` or `scripts/launch_p1l_main.py`; never target the closed codex
session; no lock or manifest was rewritten; `pyproject.toml`, `uv.lock` and every config are unchanged; hashes are
blake2b; seeds come from `derive_seed`; no test or confirmation score was read or produced.
