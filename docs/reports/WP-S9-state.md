# WP-S9 step 1 — symbolic follow-ups IVD, CVH, TSM, UNS and the neighbour-confusion rate (sn-sym, 2026-10-27)

Step 1 only. Worktree `/work/xg24i002/x10041/LoopLora-wt/sym`, branch `wp/sym` (base `dc659e0`). Nothing was run: no
PBS job, no pytest, no `import torch`, no training or evaluation, on the login node or anywhere. **All code and all
tests below are written but have never been executed**; the only checks are static (compilation under Python 3.9 and
3.13, `/usr/bin/pyflakes`, `ast` link checks), a read-only review by a subagent (§8) and the file-level facts listed in
§7. Expect the first compute-node test run to find defects; fix them before any lock. Step 2 (running) waits for the
orchestrator's go after 2026-10-28 22:00 JST.

```bash
SYM=/work/xg24i002/x10041/LoopLora-wt/sym     # worktree; step-2 outputs go to $SYM/runs/<study> (git-ignored)
                                              # and compact results to $SYM/reports/<study> (committed)
CENTRAL=/work/xg24i002/x10041/LoopLora        # central repo: submit.py's REPO, venv, run data of P1-S
```

## 1. What exists

| study | prereg draft | library | scripts | tests |
| --- | --- | --- | --- | --- |
| shared | — | `src/looplora/sym/followup_stats.py` | `scripts/s9_common.py`, `s9_torch.py` | `tests/test_followup_stats.py` |
| IVD (F8) | `docs/prereg/IVD.md` | `ivd.py` | `run_ivd.py` (lock, smoke, run), `analyze_ivd.py` | `test_ivd.py`, `test_ivd_scripts.py` |
| CVH (F9) | `docs/prereg/CVH.md` | `cvh.py` | `run_cvh.py` (lock, smoke, train, confirm), `analyze_cvh.py` | `test_cvh.py`, `test_cvh_scripts.py` |
| TSM (F10) | `docs/prereg/TSM.md` | `tsm.py` | `run_tsm.py` (lock, smoke, train, evaluate), `analyze_tsm.py` | `test_tsm.py`, `test_tsm_scripts.py` |
| UNS (F11) | `docs/prereg/UNS.md` | `uns.py` | `audit_uns.py` (Part A, and Part B with `--extension-root`), `run_uns.py` (Part B: lock, run) | `test_uns.py`, `test_uns_scripts.py` |
| NCR (C4) | `docs/prereg/NCR.md` | `ncr.py` | `analyze_neighbour_confusion.py` | `test_neighbour_confusion.py` |

All paths are under `src/looplora/sym/`, `scripts/`, `tests/` in the worktree. No tracked file was modified (§7), so every
existing code path is unchanged; where a new function replaces an existing loop for a new option, a test compares it with
the existing code (`fit_tsm` without stay loss and gate equals `looplora.sym.train._fit` bitwise; `fit_head` with the
P4-S-L seed label reproduces the P4-S-L head fit; `ivd` cells with no extras equal the plain `model.prepare` memory).
Every lock is a JSON with blake2b seals of the checkpoints; `run` stages refuse to start unless the design file equals its
Git-committed copy (`git show <commit>:<path>`); every stage re-verifies checkpoint files and model state hashes after use.

## 2. Inventory of existing checkpoints (paths, seeds, writers)

All r = r_mem = 32, seeds 17/29/43, task `t2_chain_end` E = 12, chains (8, 4), V = 1024, backbone seed 31415.

| study | checkpoints | where |
| --- | --- | --- |
| IVD | 12: writers `oracle_random`, `joint_fact` × readers M1 (mixed-depth {1, 2, 4}), B4 | `oracle_random`: `$CENTRAL/runs/p1s/main/20261026/runs/primary-oracle_random-{M1,B4}-r32-s{17,29,43}/checkpoint.pt`; `joint_fact`: `$SYM/../symf/runs/p1s/f2/evaluation/f2-main-joint_fact-{M1,B4}-s{17,29,43}/checkpoint.pt` (symbolic links into `.../f2/main-readers/shard-0N/<name>/checkpoint.pt`; the lock resolves and seals the real files) |
| CVH | 9 M1 readers (`oracle_random`, `joint_fact`, `t1_frozen_fact`) + their 9 exported P4-S-L heads + 3 locked P4-S-L thetas (0.25, 0.45, 0.25) | readers as in P4-S-L `reports/p4s-l/design.json` (`oracle_random`: P1-S main as above; `joint_fact`, `t1_frozen_fact`: `.../symf/runs/p1s/f2/main-readers/shard-0N/f2-main-<writer>-M1-s<seed>/checkpoint.pt`); heads `reports/p4s-l/heads/<name>.pt` (24 files, all 24 blake2b seals verified against `head_manifest.json` in this session); thetas `reports/p4s-l/theta.json` |
| TSM | no reuse of trained weights: 6 released M1 configurations as templates (`oracle_random` P1-S, `joint_fact` F2; training settings checked at the lock) and 6 B4 references (same files as IVD's B4) | as IVD. The P2-S-F1 M1-L4 runs are not reused; their configurations were compared with the released M1 ones (§7) |
| UNS A | 18 screening evaluation directories (`screening/samples.npz` + `metrics.json`): writers `joint`, `joint_fact`, `oracle_random` (control) × M1, B4 | `joint`, `oracle_random`: `$CENTRAL/runs/p1s/main/20261026/runs/primary-<w>-<reader>-r32-s<seed>/screening`; `joint_fact`: `.../symf/runs/p1s/f2/evaluation/f2-main-joint_fact-<reader>-s<seed>/screening` |
| UNS B | 9 P2-S-F1 M1-L4 (`joint`, `joint_fact`, `oracle_random`) + the 9 B4 above | `.../symf/runs/p2s-f1/main/shard-0{0,1,2}/main-<w>-M1-r32-m32-s<seed>-lr0.003/checkpoint.pt`, sealed in `reports/p2s-f1/checkpoint_manifest.csv` (source `p2s-f1`) |
| NCR | 72 saved sample sets: 6 writers (`t1_frozen_fact`, `joint_fact`, `joint`, `t1_frozen`, `oracle_random`, `oracle_orth`) × M1, B4 × screening, confirmation × 3 seeds | P1-S main and F2 layouts as above (`<run>/screening/`, `<run>/confirmation/`); all 72 exist |

The F2 and P2-S-F1 data live under the closed wk-symf worktree `$SYM/../symf/runs/` (untracked). **Do not remove that
worktree** before step 2 ends. IS (54 E12 reference checkpoints, `reports/is/design.json`, lock `fb62cf1`; runs under
`$SYM/../symf/runs/is/`) and JS (12 runs under `$SYM/../symf/runs/js/`) are cost references only; the follow-ups do not reuse
their weights.

## 3. Evaluation seeds and confirmation sets used so far, and the new allocation

| seed | role | used by |
| --- | --- | --- |
| 17 / 29 / 43 | training seeds (and derivation labels) | every symbolic study |
| 31415 | frozen backbone seed (not an evaluation seed) | all |
| 31337 | head-training generator, split `train`, 20000 episodes | P4-S-L (CVH reuses it for A2 training) |
| 4242 | split `val` selection and pilots; screening evaluation (`test`, `test_unseen`, 2000 episodes) | P1-S, F1, F2, P2-S, P2-S-F1 pilots, halting C3/C4 and P4-S-L validation, IS/JS validation |
| 777001 | C777: split `test`, 2000 episodes (the WP-H set) | P1-S main confirmation, F1, F2 |
| 888001 | C888: split `test`, 2000 episodes (the P2-S set) | P2-S, P2-S-F1 (reused, stated in its report) |
| 999001 | IS confirmation | IS |
| 999002 | P4-S-L confirmation | P4-S-L |
| 999003 | JS confirmation | JS |
| **999004** | **IVD** (split `test`, 2000 episodes) | new |
| **999005** | **CVH** (split `test`, 2000 episodes) | new |
| **999006** | **UNS Part B** (splits `test`, `test_unseen`, 2000 episodes; optional) | new |
| **999007** | **TSM** (split `test`, 2000 episodes) | new |
| none | UNS Part A (re-analysis of the existing seed-4242 screening samples) and NCR (existing samples) | no new seed |

The LLM side uses index ranges of the split `test` ([100000, 102000), [102000, 104000), [110000, 112000)), not these seeds.
Whole-word search for 999004–999007 across `docs`, `plan`, `configs`, `scripts`, `src`, `tests`, `reports` of the central
repo and of every other worktree found no use (run trees under `runs/` were not scanned).

## 4. Committed unseen-pool rows (binding instruction after WP-T9 open question (a))

Metadata only; no `difference`, interval or per-seed value was read or printed. `dataset=unseen` rows in the committed
`reported_contrasts.csv` exports (each row at confidence 0.95 and 0.975; training seeds pooled by the rows: 17, 29, 43 for
P1-S primary and replication and for all F1 and F2 rows, seed 17 only for the 96 P1-S secondary rows):

| export | rows total (screening / unseen / confirmation) | unseen rows by writer | by reader | named endpoints |
| --- | --- | --- | --- | --- |
| `reports/p1s` | 888 (344 / 344 / 200); groups primary 224, replication 24, secondary (r = 16) 96 | oracle_random 96, oracle_orth 88, t1_frozen 64, joint 96 | M1 162, M1-lin 122, B5 14, B5-lin 14, B3 8, B4 24 | reach, hop1, far, specificity, vs_fixed 28 each; vs_loop 16; replication_{reach, far, specificity} 16 each; L8 − L4 hops 4–8: 84; M1 − M1-lin: 20 + 20; B4@1 − M1@1: 8 + 8 |
| `reports/p1s-f1` (r = 64, 128) | 264 (120 / 120 / 24) | oracle_random 60, joint 60 | M1 64, M1-lin 56 | reach, hop1, far, specificity, vs_fixed, vs_loop 16 each; H6 L8 − L4: 16; M1 − M1-lin: 8 |
| `reports/p1s-f2` (r = 32) | 144 (64 / 64 / 16) | t1_frozen_fact 32, joint_fact 32 | M1 36, M1-lin 28 | reach, hop1, far, specificity, vs_fixed, vs_loop 8 each; H6 L8 − L4: 8; M1 − M1-lin: 4; B4@1 − M1@1: 4 |

The committed `screening` rows (the seen-pool rows of the same exports) pool seed 17 only in all three exports, so they are
not like-for-like with the three-seed unseen rows; Part A recomputes the seen-pool values from the `test` split of the same
three seeds, and its `context.csv` lists the pooled training seeds beside every transcribed row (found while checking the
exports' seed metadata; `uns.load_committed` and `uns.context_table` were extended accordingly).
The exports carry no evaluation-seed or episode-count columns. From the runs' own `metrics.json` (config block, read in this
session for the 18 Part A directories, no metric values) the evaluation is split `test_unseen`, seed 4242, 2000 episodes,
hops 1–8, depths 0, 1, 2, 3, 4, 6, 8, 12, 16; the sample file hashes of all 18 equal the recorded ones. For `joint`,
`joint_fact` and `oracle_random` exactly one committed unseen row exists per (M1, endpoint ∈ {hop1, reach, far}, confidence),
in the P1-S primary group (r = 32) and in F2 respectively.
Coverage of the F11 question: the committed rows cover the released mixed-depth M1 (depth contrasts) and, as contrasts only,
B4; they do not cover the P2-S-F1 M1 trained at L = 4 (no unseen evaluation exists) or absolute unseen accuracies of B4.
Consequently UNS is **Part A, a zero-evaluation audit** of the 18 existing screening evaluations (protocol audit, exact
recomputation of the 18 committed M1 depth rows, seen-versus-unseen tables, context transcription), plus **Part B**,
registered only for what is missing (M1-L4 and B4 on new episodes, seed 999006, optional). `docs/prereg/UNS.md` states
that these rows were seen before registration (committed values and the qualitative sentences of the F1 and F2 reports),
that the author read their metadata only, that the paper makes no unseen claim and that no gate depends on UNS.

Disclosure (review stage). After the drafts were complete, the independent reviewer (§8), a read-only subagent, loaded the
`prediction` and `correct` arrays of both pools of the 18 Part A sample files for structural checks, and replayed with its
own code 12 of the 18 reproduction comparisons (`joint` and `joint_fact`; L4 − L1 on hops {1}, {2, 3}, {6, 7, 8}; both
confidence levels; `oracle_random` was not replayed) against the committed unseen rows. Its report says that the checks
passed and that the reproduction was exact; it contains no value, and the author has seen none. `docs/prereg/UNS.md` and
`docs/prereg/NCR.md` state this. No registered quantity or rule changed because of it. It does mean that Part A's
reproduction, which was never blind to the committed rows, has been looked at once already by a delegate of the author.

## 5. Cost basis and node·h estimates

Measured costs (PBS elapsed, one node per job; ledgers `reports/{is,js,p4s-l,p2s-f1}/`):

| study | evidence | node·h |
| --- | --- | --- |
| P2-S-F1 | main: 3 jobs of 8 concurrent runs (M1-L4 and B-variants, 15000 steps) 935–950 s each; final evaluation 67 s; pilot 0.293; total 1.099 | 0.806 main + final |
| JS | training: 3 jobs of 4 concurrent runs (random depth 4–8, half with the Jacobian penalty) 1117–1130 s each; smoke 123 s; total | 1.011 |
| IS | training: 3 jobs of 12 concurrent runs 841–843 s each; confirmation statistics 216 s; total | 0.819 |
| P4-S-L | 24 heads with traces and validation 337 s application (351 s allocated); confirmation, timing and analysis 72 s; total including a failed test job | 0.144 |
| screening evaluation (P1-S, F2 `timing.json`) | `joint` 129–145 s, `oracle_random` 154 s, `joint_fact` 75–79 s per checkpoint, 2 splits, 2000 episodes, nine depths, counterfactual conditions | — |

| study (job) | basis | estimate | upper bound |
| --- | --- | --- | --- |
| tests of all new code (one debug job) | P4-S-L 159 tests 60 s, JS 16 tests 15 s, IS 23 tests 18 s | 0.02 | 0.05 |
| NCR | CPU, 72 sample sets, row-by-row verification plus bootstraps: minutes | 0.03 | 0.1 |
| UNS Part A | CPU bootstraps (about 200 intervals) and 18 file reads | 0.05 | 0.1 |
| UNS Part B (optional) | 18 evaluations without counterfactuals; screening cost per checkpoint above | 0.3 | 0.75 |
| IVD | lock, smoke (4 checkpoints × 128 episodes), 12 checkpoints × 7–9 cells at loads up to 36 facts: about one to one-and-a-half screening evaluations per checkpoint (plan: 0.2–0.4) | 0.3 | 0.8 |
| CVH | 9 readers: P4-S-L cost scaled from 24 to 9 heads, plus continuation labels, confirmation with latency timing | 0.1 | 0.3 |
| TSM | 6 T0 runs at about one M1-L4 run each (0.033 node·h in P2-S-F1) + 12 T1/T2 runs at about twice that (depth 8, per-step decode) = 0.2 + 0.8–1.2; evaluation 24 × 4 depths; smoke; tests | 1.2–1.5 | 1.8 |
| **all** | | about 2.1–2.4 | about 3.9 |

The IVD and TSM figures are the least certain; their smoke jobs fix the walltimes (walltime = measured × 1.5 from the smoke
reports' `suggested_walltime_seconds`; TSM's `smoke.json` also lists `extrapolated_run_seconds` per run so any subset of
runs can be costed). Per job: one node, at most 1 h; `--scale-up docs/prereg/<NAME>.md` (at most 2 h) only where the prereg's
Limits allows it: IVD `run` (no subset option exists), TSM (only if one seed's six runs cannot be split further). The
`--scale-up` document must exist in the central repo, so the orchestrator commits the preregs to master first.

## 6. Step 2: gates and commands

Order (plan): IVD, CVH, UNS (A, then optionally B) and NCR first; TSM after. They are independent and may run in parallel
nodes. Before any lock: (1) the orchestrator reviews and commits the five preregs to master (that commit is the registration);
(2) the compute-node test job below passes (fix defects, commit, repeat; defects found at this stage are not scientific
changes unless they alter a registered quantity, which must then be flagged); (3) the code commit to snapshot is chosen
(`CODE` below, a full 40-character SHA; `--commit` snapshots it with `git archive`, so the run uses exactly that code).
Output roots are absolute, new, and contain no path component with the word "confirmation".

```bash
cd $SYM
SUB="/usr/bin/python3 scripts/submit.py"
CODE=<full SHA>

# 0. tests of all new code (development; runs the working tree)
$SUB --name s9-tests --queue debug-g --walltime 00:15:00 --worktree $SYM --wait -- \
  -m pytest -q -o addopts= tests/test_followup_stats.py tests/test_ivd.py tests/test_ivd_scripts.py \
  tests/test_cvh.py tests/test_cvh_scripts.py tests/test_tsm.py tests/test_tsm_scripts.py \
  tests/test_uns.py tests/test_uns_scripts.py tests/test_neighbour_confusion.py

# 1. IVD   (lock file inside the worktree; commit it: its full SHA is IVD_LOCK)
$SUB --name ivd-lock  --queue debug-g   --walltime 00:05:00 --commit $CODE --wait -- scripts/run_ivd.py lock --out $SYM/reports/ivd/design.json
git add reports/ivd/design.json && git commit -m "IVD: design lock (blake2b seals of the 12 checkpoints)"; IVD_LOCK=$(git rev-parse HEAD)
$SUB --name ivd-smoke --queue regular-g --walltime 00:10:00 --commit $CODE --wait -- scripts/run_ivd.py smoke --design $SYM/reports/ivd/design.json --out $SYM/runs/ivd/smoke
#    read suggested_walltime_seconds in runs/ivd/smoke/smoke.json; status must be "passed" (mean overlap ordered low < nat < high)
$SUB --name ivd-run   --queue regular-g --walltime <smoke suggestion> [--scale-up docs/prereg/IVD.md if > 01:00:00] --commit $CODE -- \
  scripts/run_ivd.py run --design $SYM/reports/ivd/design.json --design-commit $IVD_LOCK --smoke-dir $SYM/runs/ivd/smoke --out $SYM/runs/ivd/main
$SUB --name ivd-analyze --queue debug-g --walltime 00:15:00 --commit $CODE --wait -- scripts/analyze_ivd.py --run $SYM/runs/ivd/main --out $SYM/reports/ivd/results

# 2. CVH   (the lock reads the committed P4-S-L files of the central repo; the confirm stage needs the Git lock of theta.json)
$SUB --name cvh-lock  --queue debug-g   --walltime 00:05:00 --commit $CODE --wait -- scripts/run_cvh.py lock --repo $CENTRAL --out $SYM/reports/cvh/design.json
git add reports/cvh/design.json && git commit -m "CVH: design lock"; CVH_LOCK=$(git rev-parse HEAD)
$SUB --name cvh-smoke --queue regular-g --walltime 00:10:00 --commit $CODE --wait -- scripts/run_cvh.py smoke --design $SYM/reports/cvh/design.json --root $SYM/runs/cvh/smoke
$SUB --name cvh-train --queue regular-g --walltime <suggested_train_walltime_seconds> --commit $CODE -- \
  scripts/run_cvh.py train --design $SYM/reports/cvh/design.json --design-commit $CVH_LOCK --root $SYM/runs/cvh/main
#    theta.json (validation only) -> $SYM/reports/cvh/theta.json, committed (CVH_THETA) before confirmation; a registered failure stops here
$SUB --name cvh-confirm --queue regular-g --walltime <suggested_confirm_walltime_seconds> --commit $CODE -- \
  scripts/run_cvh.py confirm --design $SYM/reports/cvh/design.json --design-commit $CVH_LOCK \
  --theta $SYM/reports/cvh/theta.json --theta-commit $CVH_THETA --root $SYM/runs/cvh/main
$SUB --name cvh-analyze --queue debug-g --walltime 00:15:00 --commit $CODE --wait -- scripts/analyze_cvh.py --root $SYM/runs/cvh/main --out $SYM/reports/cvh/results

# 3. UNS Part A (no new seed) and NCR (no new seed)
$SUB --name uns-a --queue debug-g --walltime 00:20:00 --commit $CODE --wait -- scripts/audit_uns.py --out $SYM/reports/uns/part-a
$SUB --name ncr   --queue debug-g --walltime 00:20:00 --commit $CODE --wait -- scripts/analyze_neighbour_confusion.py --out $SYM/reports/ncr

# 3b. UNS Part B (optional; orchestrator decision)
$SUB --name uns-lock --queue debug-g --walltime 00:05:00 --commit $CODE --wait -- scripts/run_uns.py lock --repo $CENTRAL --out $SYM/reports/uns/design.json
git add reports/uns/design.json && git commit -m "UNS Part B: design lock"; UNS_LOCK=$(git rev-parse HEAD)
#    timing run (partial, never analysed, delete afterwards):
$SUB --name uns-time --queue regular-g --walltime 00:10:00 --commit $CODE --wait -- scripts/run_uns.py run --design $SYM/reports/uns/design.json \
  --design-commit $UNS_LOCK --root $SYM/runs/uns/timing --names joint-M1L4-r32-s17,joint-B4-r32-s17
$SUB --name uns-b --queue regular-g --walltime <measured x 1.5> --commit $CODE -- scripts/run_uns.py run --design $SYM/reports/uns/design.json \
  --design-commit $UNS_LOCK --root $SYM/runs/uns/main
$SUB --name uns-b-audit --queue debug-g --walltime 00:20:00 --commit $CODE --wait -- scripts/audit_uns.py --extension-root $SYM/runs/uns/main --out $SYM/reports/uns/part-b

# 4. TSM   (after the others)
$SUB --name tsm-lock  --queue debug-g --walltime 00:05:00 --commit $CODE --wait -- scripts/run_tsm.py lock --out $SYM/reports/tsm/design.json
git add reports/tsm/design.json && git commit -m "TSM: design lock"; TSM_LOCK=$(git rev-parse HEAD)
$SUB --name tsm-smoke --queue regular-g --walltime 00:20:00 --commit $CODE --wait -- scripts/run_tsm.py smoke --design $SYM/reports/tsm/design.json --root $SYM/runs/tsm/smoke --procs 6
for S in 17 29 43; do   # one job per seed (6 runs each); walltime from smoke.json extrapolated_run_seconds x 1.5 / procs
  $SUB --name tsm-train-$S --queue regular-g --walltime <from smoke> --commit $CODE -- scripts/run_tsm.py train --design $SYM/reports/tsm/design.json \
    --design-commit $TSM_LOCK --root $SYM/runs/tsm/main --procs 6 --seeds $S --label s$S
done
$SUB --name tsm-eval  --queue regular-g --walltime <about 00:20:00> --commit $CODE -- scripts/run_tsm.py evaluate --design $SYM/reports/tsm/design.json --design-commit $TSM_LOCK --root $SYM/runs/tsm/main
$SUB --name tsm-analyze --queue debug-g --walltime 00:15:00 --commit $CODE --wait -- scripts/analyze_tsm.py --root $SYM/runs/tsm/main --out $SYM/reports/tsm/results
```

Jobs longer than the Bash tool's limit are submitted without `--wait` and followed with `qstat` (or the
`miyabi-pbs-goal-wakeup` workflow). The lock files are written into the worktree because the `run` stages verify them with
`git show <commit>:<path>` in the file's own repository; a commit of the same file on master at the same relative path
is equally valid. A failed training job leaves its siblings' partial `runs/<name>` directories: TSM's `train-one` refuses to
overwrite them, so move the incomplete directories aside (their `record.json` status is not `complete`) before resubmitting.

## 7. Verification done in step 1 (all static or file-level)

- Syntax and lint: all 27 new Python files compile under the login node's Python 3.9 (`py_compile` before the review fixes of
  §8, `compile()` without bytecode after them; so no 3.10-only syntax at runtime) and under the project's Python 3.13.13, and
  pass `/usr/bin/pyflakes` (re-run after the review fixes: exit 0).
- Link check by `ast` only (no import of project code), re-run after the review fixes: the 229 names of the 116
  project-internal `from` imports, 584 module-attribute references and 921 calls (keyword names, positional counts,
  required arguments) between the new files and the existing library resolve, with no problem (the signature checker's 4
  hits are `require_compute` and `load_frozen` re-exports, which it does not follow); the
  `argparse` destinations of the nine command-line scripts equal their `args.*` uses; every option in the §6 commands exists in
  `submit.py` or in the called script and subcommand; the pytest fixtures of the ten test files (93 test functions; the
  first commit message says 92, counted before the last test was added) are builtin or defined in the file; all
  third-party imports (numpy, torch, pytest) exist in the venv (checked with `find_spec`, nothing imported).
- Structural check of real data by the author, numpy only: the label arrays (split, condition, L, hop, episode_id, query_id,
  answer, candidate_size; the author read no `prediction` or `correct`; for what the reviewer read see §4 and §8) of the 18
  Part A screening sample files agree with `uns.audit_samples`
  (2000 episodes per pool, depths 0–16, hops 1–8, 2000 queries per hop and depth, identical query labels across depths,
  answers inside the pool, candidate size 14, no episode id shared by the pools): zero problems.
- The numbers quoted in the preregs' "Known then" sections (IS, P4-S-L, JS, P2-S-F1) were re-checked against the committed
  reports, the counts of the committed unseen rows against the three CSVs, and every repository path named in the five
  preregs and in this report (other than files that a run will create) exists.
- Layout and configuration facts read from JSON, CSV and `ls`, never from torch: all 18 Part A screening directories exist with
  `evaluation_seed`, `config_hash`, `samples_hash`, `config` keys, matching sample hashes (blake2b) and the registered protocol;
  all 72 NCR sample sets exist; the 24 P4-S-L head files match `head_manifest.json`; the P2-S-F1 M1-L4 configurations of
  both TSM writers (six runs) differ from the released M1 configurations only in `train.depths` (outside the eval and pretrain
  blocks); `reports/p2s-f1/checkpoint_manifest.csv` hashes of the nine M1-L4 checkpoints were verified in an earlier part of
  this task; seeds 999004–999007 are unused; the committed unseen rows have exactly one row per (writer, endpoint, confidence)
  needed by the reproduction check.
- Synthetic test expectations (for example the full-vocabulary IVD test) were derived by hand from the formulas, not by running.
- Read-only independent review of the code by one subagent (replays of the audits on real data included): see §8.
- `git diff --name-status dc659e0 HEAD` lists only added files (no tracked file modified); see the last line of the final reply.

Not verified: any execution of the new code (tests, shapes, devices, GPU memory and time); the exact recomputation of the
committed unseen rows by `uns.reproduce_reported` (the reviewer's own implementation reproduced 12 of the 18 comparisons
exactly, which shows that the registered recomputation is feasible but does not test this code); the A0 reproduction of the
P4-S-L validation accuracy; the smoke-based walltimes; the CUDA-only paths (CVH latency benchmark, `torch.cuda.set_device`,
`synchronize`); the concurrent child processes of the TSM smoke and train stages; the bitwise equalities that the tests assert
(`fit_tsm` without stay loss and gate against `looplora.sym.train._fit`, the CVH fit against the P4-S-L head fit); state-hash
stability across evaluations; the IVD smoke ordering check (mean overlap low < nat < high) on real checkpoints.

## 8. Independent review

One read-only subagent (maximum effort; 308 tool calls over 68 minutes; instructed to change nothing, and its report states
that nothing in the worktree changed, its scratch files being in the session scratchpad; it imported no torch and ran no test)
reviewed the work after it was complete, on the tree as it stood before the fixes below. Coverage: the 6 library modules,
11 scripts, 10 test files, the 5 prereg drafts, plan F8–F11, the WP-S9 task file, `_common_rules.md` and the command sheet
of §6, with two full passes over the high-risk files; the tracked code that the new code depends on was read where needed.
Mechanical checks: compilation, pyflakes, 907 calls against their signatures, 576 module-attribute references, artifact keys
between writers and readers, no SHA-256/SHA-1/MD5, each new seed only in its own study, no tracked file modified. Replays on
real data with its own stdlib/numpy code (the repo modules were not imported): the Part A audit of the 18 sample directories
and the reproduction of the committed unseen M1 L4 − L1 rows for `joint` and `joint_fact` (see the disclosure at the end of
§4), existence and identity of the 12 IVD checkpoints, of the 9 UNS Part B M1-L4 checkpoints (hash, identity, trained depths
[4]) and of the 6 TSM templates, and the consistency of the CVH design inputs between the central checkout and this
worktree.

Verdict: no defect with a crash, wrong-number, protocol or test-defect failure mode. Eight low-severity observations; none
was a failing scenario. Disposition (the fixes are covered by the static checks of §7 and, where stated, by a new or changed
test, which like all tests has not been run):

| # | observation | disposition |
| --- | --- | --- |
| O1 | IVD, `oracle_random`: the constructed per-episode code tables also served as the decode table, so the full-vocabulary exact match (a reported second view, not gating) depended on the overlap arm; the candidate-restricted decisions were not affected because the extras' heads never are candidates | fixed: `ivd.prepare_cell(..., decode_codes=)` and `evaluate_cells` decode every cell against the unmodified global codes; new test `test_constructed_codes_change_the_write_keys_but_not_the_decode_table`; IVD.md states it |
| O2 | `--out` / `--root` of the IVD smoke and run, CVH smoke, train and confirm, TSM smoke, train and evaluate stages are documented as absolute paths but were not checked | fixed: `s9_common.absolute_path` refuses relative paths in those stages; test in `test_ivd_scripts.py` |
| O3 | `run_uns.py lock` did not call `require_compute()` (the other studies' lock stages do) although it loads and hashes 18 checkpoints | fixed: `require_compute()` in `main()` for every stage; the §6 sheet already ran it through `submit.py` |
| O4 | the CVH CUDA latency path is not exercised by the tests (they pass `--skip-latency`) | no change: the smoke stage calls the benchmark on CUDA, so a failure shows up there and not in a registered result; listed under "Not verified" |
| O5 | `cvh.collect_traces` asserts that the trace's candidate argmax at depths 1, 4, 16 equals `model.read` exactly; it could raise if the two paths differed numerically | no change: it is the same guard as the P4-S-L collection uses on these readers; a failure is loud, and the smoke runs it |
| O6 | `analyze_neighbour_confusion.py` caught every `ValueError` of `ncr.summarize` as an "undefined group" | fixed: only the registered case (a seed with no errors) is `ncr.UndefinedShares` and is reported as undefined; anything else stops the analysis; test changed |
| O7 | the smoke stages do not verify a Git lock, and CVH and TSM `train` do not require a passed smoke report (IVD `run` does) | no change: it is what the preregs say (the smoke reads validation data only, selects nothing, and CVH.md and TSM.md register no refusal rule for it); smoke-before-train is by convention |
| O8 | the TSM state-hash check around each evaluation covers the reader but not the T2 gate's two scalars | no change: the evaluation attaches no optimizer and only reads the gate |

The self-found fixes of this session were made before the review reported and were not prompted by it: the UNS and state-report
statements on the training seeds that the committed rows pool (and the matching change of `uns.load_committed` and
`uns.context_table`), the IVD decay-only overlap (reported as NaN, with its test), the CVH cost figures aligned to the P4-S-L
ledger, and the corrected IS/JS path in §2.

## 9. Deviations, interpretations and open questions

Every item is also flagged in the corresponding prereg ("Flagged interpretations"). None changes a decision rule of the plan
as to its thresholds; each fills something the plan leaves open.

| study | item |
| --- | --- |
| IVD | decision load L24 (L6 reported); per-writer decision with a 97.5% interval; `nat` arm and second load level added (reported only); manipulation check (overlap gap ≥ 0.2, label `inconclusive`) added, same action as `not_supported`; frozen readers see 18 or 36 stored facts, outside their training distribution |
| CVH | the plan's three heads: A0 is a reference, the rule applies to A1 and A2 (six comparisons); margin 0.25 nats fixed before any data; the hop-1 constraint is applied to the threshold choice of both A1 and A2; "mean executed depth" read as mean executed updates (fallback = 16), stricter than the mean selected step; a family whose A1 and A2 are both infeasible is not traced on the confirmation set; wall time is the primary latency clock; the A0 recomputation must reproduce the P4-S-L validation accuracy within 1e-3 |
| TSM | T0 = released recipe at the fixed depth 4 (P2-S-F1 recipe), retrained; stay uses L4 and L16, far uses L8 for arm and T0 (L16 reported), gain uses L4 against B4 at L1; an arm passes only if both writers pass (rv-astra), intervals corrected for four comparisons; stay-loss weighting (mean over steps s ≥ hop, equal over queries); T2 without PonderNet step cost, with explicit init (a, b) = (0, 4); same released lr (0.003) for every arm, no retry; a run that fails to train blocks the evaluation and is reported, with no protocol change before an explicit amendment |
| UNS | split into Part A (zero-evaluation audit of the committed rows) and optional Part B (what is missing); "M1-L4 and B4" covered by Part A (released M1 at L4/L1, B4) and Part B (M1 trained at L = 4); `oracle_random` added as a control; a 3 pp wording rule for the scope row (changes no gate); disclosure that the unseen rows were seen before registration |
| NCR | single read = L = 1 of M1 and B4; both rate and share reported; category priority and the h = 2 case fixed before any neighbour-confusion quantity is computed; the already analysed C777 confirmation samples are re-read, not re-evaluated |

Open questions for the orchestrator:
1. UNS Part B: go or no-go (0.3 node·h expected, 0.75 upper bound). Part A alone is the zero-evaluation reading of the binding instruction.
2. TSM: is "both writers must pass" (rv-astra) the intended arm-level rule, and should a registered failure to train (non-finite loss) stay a stop-and-amend event, as drafted?
3. Where the lock commits go: the commands commit the design and `theta.json` locks on `wp/sym`; the orchestrator may instead commit identical files on master at the same paths.
4. The `--scale-up` allowances written into the IVD and TSM preregs (up to 2 h, only if the smoke requires it) need the orchestrator's acceptance when committing them.
5. Master has advanced since the base (`f7be34a` versus `dc659e0`) without touching F8–F11 (the hunks of `plan/followup_2026-10-27.md` on master lie in its header, §2, §3 and the F2–F4 sections, none in F8–F11 or §6); the new files do not overlap any file changed on master (`git diff --name-only dc659e0 master`), so a plain merge should be clean.
