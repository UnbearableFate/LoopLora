# WP-S9 step 1 — symbolic follow-ups IVD, CVH, TSM, UNS and the neighbour-confusion rate (sn-sym, 2026-10-27)

Step 1 only. Worktree `/work/xg24i002/x10041/LoopLora-wt/sym`, branch `wp/sym` (base `dc659e0`). Nothing was run: no
PBS job, no pytest, no `import torch`, no training or evaluation, on the login node or anywhere. **All code and all
tests below are written but have never been executed**; the only checks are static (compilation under Python 3.9 and
3.13, `/usr/bin/pyflakes`, `ast` link checks), a read-only review by a subagent (§8) and the file-level facts listed in
§7. Expect the first compute-node test run to find defects; fix them before any lock. Step 2 (running) waits for the
orchestrator's go after 2026-10-28 22:00 JST.

Step 1b (2026-10-27, §10): the five preregs were registered on master at 40ddf08 and master was merged into `wp/sym`; the TSM
scripts now implement the registered training-failure rules, and the §6 command sheet stops after each lock stage and
reports the file path (the orchestrator commits it on `wp/sym`) and runs UNS Part B. An independent review of the first
version found gaps in the rerun rules; they are fixed (§10). Still nothing was executed on a compute node: the new tests have
never run with torch or pytest, and §10 states exactly what was checked and how.

```bash
SYM=/work/xg24i002/x10041/LoopLora-wt/sym     # worktree; step-2 outputs go to $SYM/runs/<study> (git-ignored)
                                              # and compact results to $SYM/reports/<study> (committed)
CENTRAL=/work/xg24i002/x10041/LoopLora        # central repo: submit.py's REPO, venv, run data of P1-S
```

## 1. What exists

| study | prereg (registered 40ddf08) | library | scripts | tests |
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
| **999006** | **UNS Part B** (splits `test`, `test_unseen`, 2000 episodes; registered, run) | new |
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
registered only for what is missing (M1-L4 and B4 on new episodes, seed 999006; registered as run, not optional). `docs/prereg/UNS.md` states
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
| UNS Part B (registered) | 18 evaluations without counterfactuals; screening cost per checkpoint above | 0.3 | 0.75 |
| IVD | lock, smoke (4 checkpoints × 128 episodes), 12 checkpoints × 7–9 cells at loads up to 36 facts: about one to one-and-a-half screening evaluations per checkpoint (plan: 0.2–0.4) | 0.3 | 0.8 |
| CVH | 9 readers: P4-S-L cost scaled from 24 to 9 heads, plus continuation labels, confirmation with latency timing | 0.1 | 0.3 |
| TSM | 6 T0 runs at about one M1-L4 run each (0.033 node·h in P2-S-F1) + 12 T1/T2 runs at about twice that (depth 8, per-step decode) = 0.2 + 0.8–1.2; evaluation 24 × 4 depths; smoke; tests | 1.2–1.5 | 1.8 |
| **all** | | about 2.1–2.4 | about 3.9 |

The IVD and TSM figures are the least certain; their smoke jobs fix the walltimes (walltime = measured × 1.5 from the smoke
reports' `suggested_walltime_seconds`; TSM's `smoke.json` also lists `extrapolated_run_seconds` per run so any subset of
runs can be costed). Per job: one node, at most 1 h; `--scale-up docs/prereg/<NAME>.md` (at most 2 h) only where the prereg's
Limits allows it: IVD `run` (no subset option exists), TSM (only if one seed's six runs cannot be split further). The
`--scale-up` document must exist in the central repo: the preregs are registered on master (40ddf08), where the IVD and TSM
allowances were accepted.

## 6. Step 2: gates and commands

Order (plan): IVD, CVH, UNS (Part A, then Part B, which is registered) and NCR first; TSM after. They are independent and may
run in parallel nodes. Before any lock: (1) done: the five preregs are registered on master at 40ddf08; (2) the compute-node
test job below passes (fix defects, commit, repeat; defects found at this stage are not scientific changes unless they alter a
registered quantity, which must then be flagged); (3) the code commit to snapshot is chosen (`CODE` below, a full 40-character
SHA; `--commit` snapshots it with `git archive`, so the run uses exactly that code). Locks: the sheet **stops** after each
lock stage and after the CVH `theta.json` stage and reports the file path; the orchestrator checks the file against the
registered text and commits it on `wp/sym`, and the sheet continues with that commit's full SHA (`IVD_LOCK`, `CVH_LOCK`,
`CVH_THETA`, `UNS_LOCK`, `TSM_LOCK`). Output roots are absolute, new, and contain no path component with the word "confirmation".

```bash
cd $SYM
SUB="/usr/bin/python3 scripts/submit.py"
CODE=<full SHA>

# 0. tests of all new code (development; runs the working tree)
$SUB --name s9-tests --queue debug-g --walltime 00:15:00 --worktree $SYM --wait -- \
  -m pytest -q -o addopts= tests/test_followup_stats.py tests/test_ivd.py tests/test_ivd_scripts.py \
  tests/test_cvh.py tests/test_cvh_scripts.py tests/test_tsm.py tests/test_tsm_scripts.py \
  tests/test_uns.py tests/test_uns_scripts.py tests/test_neighbour_confusion.py

# 1. IVD   (the lock file is written inside the worktree; the orchestrator commits it: its full SHA is IVD_LOCK)
$SUB --name ivd-lock  --queue debug-g   --walltime 00:05:00 --commit $CODE --wait -- scripts/run_ivd.py lock --out $SYM/reports/ivd/design.json
#    STOP: report $SYM/reports/ivd/design.json; the orchestrator commits it on wp/sym. IVD_LOCK = that commit's full SHA
$SUB --name ivd-smoke --queue regular-g --walltime 00:10:00 --commit $CODE --wait -- scripts/run_ivd.py smoke --design $SYM/reports/ivd/design.json --out $SYM/runs/ivd/smoke
#    read suggested_walltime_seconds in runs/ivd/smoke/smoke.json; status must be "passed" (mean overlap ordered low < nat < high)
$SUB --name ivd-run   --queue regular-g --walltime <smoke suggestion> [--scale-up docs/prereg/IVD.md if > 01:00:00] --commit $CODE -- \
  scripts/run_ivd.py run --design $SYM/reports/ivd/design.json --design-commit $IVD_LOCK --smoke-dir $SYM/runs/ivd/smoke --out $SYM/runs/ivd/main
$SUB --name ivd-analyze --queue debug-g --walltime 00:15:00 --commit $CODE --wait -- scripts/analyze_ivd.py --run $SYM/runs/ivd/main --out $SYM/reports/ivd/results

# 2. CVH   (the lock reads the committed P4-S-L files of the central repo; the confirm stage needs the Git lock of theta.json)
$SUB --name cvh-lock  --queue debug-g   --walltime 00:05:00 --commit $CODE --wait -- scripts/run_cvh.py lock --repo $CENTRAL --out $SYM/reports/cvh/design.json
#    STOP: report $SYM/reports/cvh/design.json; the orchestrator commits it on wp/sym. CVH_LOCK = that commit's full SHA
$SUB --name cvh-smoke --queue regular-g --walltime 00:10:00 --commit $CODE --wait -- scripts/run_cvh.py smoke --design $SYM/reports/cvh/design.json --root $SYM/runs/cvh/smoke
$SUB --name cvh-train --queue regular-g --walltime <suggested_train_walltime_seconds> --commit $CODE -- \
  scripts/run_cvh.py train --design $SYM/reports/cvh/design.json --design-commit $CVH_LOCK --root $SYM/runs/cvh/main
#    STOP: report $SYM/runs/cvh/main/theta.json (validation only; a registered failure stops the study here); the orchestrator copies it
#    byte for byte to $SYM/reports/cvh/theta.json and commits it on wp/sym before confirmation. CVH_THETA = that commit's full SHA
$SUB --name cvh-confirm --queue regular-g --walltime <suggested_confirm_walltime_seconds> --commit $CODE -- \
  scripts/run_cvh.py confirm --design $SYM/reports/cvh/design.json --design-commit $CVH_LOCK \
  --theta $SYM/reports/cvh/theta.json --theta-commit $CVH_THETA --root $SYM/runs/cvh/main
$SUB --name cvh-analyze --queue debug-g --walltime 00:15:00 --commit $CODE --wait -- scripts/analyze_cvh.py --root $SYM/runs/cvh/main --out $SYM/reports/cvh/results

# 3. UNS Part A (no new seed) and NCR (no new seed)
$SUB --name uns-a --queue debug-g --walltime 00:20:00 --commit $CODE --wait -- scripts/audit_uns.py --out $SYM/reports/uns/part-a
$SUB --name ncr   --queue debug-g --walltime 00:20:00 --commit $CODE --wait -- scripts/analyze_neighbour_confusion.py --out $SYM/reports/ncr

# 3b. UNS Part B (registered, not optional; after Part A)
$SUB --name uns-lock --queue debug-g --walltime 00:05:00 --commit $CODE --wait -- scripts/run_uns.py lock --repo $CENTRAL --out $SYM/reports/uns/design.json
#    STOP: report $SYM/reports/uns/design.json; the orchestrator commits it on wp/sym. UNS_LOCK = that commit's full SHA
#    the two-checkpoint timing run is optional (partial, never analysed, delete afterwards):
$SUB --name uns-time --queue regular-g --walltime 00:10:00 --commit $CODE --wait -- scripts/run_uns.py run --design $SYM/reports/uns/design.json \
  --design-commit $UNS_LOCK --root $SYM/runs/uns/timing --names joint-M1L4-r32-s17,joint-B4-r32-s17
$SUB --name uns-b --queue regular-g --walltime <measured x 1.5> --commit $CODE -- scripts/run_uns.py run --design $SYM/reports/uns/design.json \
  --design-commit $UNS_LOCK --root $SYM/runs/uns/main
$SUB --name uns-b-audit --queue debug-g --walltime 00:20:00 --commit $CODE --wait -- scripts/audit_uns.py --extension-root $SYM/runs/uns/main --out $SYM/reports/uns/part-b

# 4. TSM   (after the others; failure rules: docs/prereg/TSM.md, "Decisions at registration")
$SUB --name tsm-lock  --queue debug-g --walltime 00:05:00 --commit $CODE --wait -- scripts/run_tsm.py lock --out $SYM/reports/tsm/design.json
#    STOP: report $SYM/reports/tsm/design.json; the orchestrator commits it on wp/sym. TSM_LOCK = that commit's full SHA
$SUB --name tsm-smoke --queue regular-g --walltime 00:20:00 --commit $CODE --wait -- scripts/run_tsm.py smoke --design $SYM/reports/tsm/design.json --root $SYM/runs/tsm/smoke --procs 6
for S in 17 29 43; do   # one job per seed (6 runs each); walltime from smoke.json extrapolated_run_seconds x 1.5 / procs
  $SUB --name tsm-train-$S --queue regular-g --walltime <from smoke> --commit $CODE -- scripts/run_tsm.py train --design $SYM/reports/tsm/design.json \
    --design-commit $TSM_LOCK --root $SYM/runs/tsm/main --procs 6 --seeds $S --label s$S
done
#    After every job read $SYM/runs/tsm/main/train_manifest-<label>.json (exit status 0 = every selected run is final):
#      "training_failures" lists registered failures (a non-finite loss in a T1/T2 run): no retry, no re-tuning; report them, the evaluation goes on;
#      status "stopped": a T0 run had a non-finite loss, the study stops: report to the orchestrator and do not evaluate;
#      status "incomplete" (the job ended before its runs did: walltime, node or I/O fault): once qstat shows the job has ended, submit the
#        identical command again at the same --commit with a new --label (s17-r2) and --rerun-incomplete, restricted to the runs of that job
#        (--seeds ...); at most twice per run (3 attempts). `train` refuses a run marked "running" that was written to in the last 5 minutes
#        (its job may be alive): wait and submit again. A rerun job that itself ends before its runs started is submitted again the same way.
#      All 18 runs must come from the same $CODE and every attempt must equal the one before it (configuration and source_hash); `train`,
#      `evaluate` and the analysis refuse otherwise (a different code version is a decision for the orchestrator, §9 (e)).
$SUB --name tsm-eval  --queue regular-g --walltime <about 00:20:00> --commit $CODE -- scripts/run_tsm.py evaluate --design $SYM/reports/tsm/design.json --design-commit $TSM_LOCK --root $SYM/runs/tsm/main
$SUB --name tsm-analyze --queue debug-g --walltime 00:15:00 --commit $CODE --wait -- scripts/analyze_tsm.py --root $SYM/runs/tsm/main --out $SYM/reports/tsm/results
```

Jobs longer than the Bash tool's limit are submitted without `--wait` and followed with `qstat` (or the
`miyabi-pbs-goal-wakeup` workflow). The lock files are written into the worktree because the `run` stages verify them with
`git show <commit>:<path>` in the file's own repository; a commit of the same file on master at the same relative path
is equally valid. TSM's rerun rules are in the comments of block 4: the `train` stage itself moves an unfinished run to `<root>/attempts/`
(with `--rerun-incomplete`), never touches a complete run or a registered failure, and `evaluate` refuses anything else.
An `evaluate` job that is killed leaves `evaluation/manifest.json` at "running" and the stage refuses to start again (§9 (d)).

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
| TSM | T0 = released recipe at the fixed depth 4 (P2-S-F1 recipe), retrained; stay uses L4 and L16, far uses L8 for arm and T0 (L16 reported), gain uses L4 against B4 at L1; an arm passes only if both writers pass (rv-astra), intervals corrected for four comparisons; stay-loss weighting (mean over steps s ≥ hop, equal over queries); T2 without PonderNet step cost, with explicit init (a, b) = (0, 4); same released lr (0.003) for every arm, no retry; training failures as registered at 40ddf08 and implemented in step 1b (§10): a non-finite loss in T1/T2 fails that arm for that writer (completed seeds reported descriptively, other arms evaluated), in T0 it stops the study, an infrastructure failure is rerun identically at most twice, a single-writer pass is a writer-specific result, not a remedy |
| UNS | split into Part A (zero-evaluation audit of the committed rows) and Part B (what is missing; registered as run, not optional); "M1-L4 and B4" covered by Part A (released M1 at L4/L1, B4) and Part B (M1 trained at L = 4); `oracle_random` added as a control; a 3 pp wording rule for the scope row (changes no gate); disclosure that the unseen rows were seen before registration |
| NCR | single read = L = 1 of M1 and B4; both rate and share reported; category priority and the h = 2 case fixed before any neighbour-confusion quantity is computed; the already analysed C777 confirmation samples are re-read, not re-evaluated |

Answered at registration (orchestrator; 40ddf08 and the step-1b message), no longer open:
1. UNS Part B runs; it is not optional (only its two-checkpoint timing run stays optional).
2. TSM: both writers must pass; a single-writer pass is reported as writer-specific, not as a remedy. Failures: infrastructure
   failure, identical rerun, at most twice; non-finite loss in T1/T2, that arm fails for that writer and the other arms are still
   evaluated; non-finite loss in T0, the study stops.
3. The orchestrator commits each design lock and `theta.json` on `wp/sym`; the command sheet stops after each lock stage and
   reports the path.
4. The `--scale-up` allowances for IVD and TSM are accepted.
5. Master was merged into `wp/sym` (a928ea0).

Open after step 1b (for the orchestrator; none blocks the lock stages):
a. A TSM run whose three attempts all end without completing (infrastructure failures): `train` refuses a fourth and `evaluate`
   refuses (such a run is neither complete nor a registered training failure). The registration has no rule for it, so it needs
   a decision (an amendment) before the confirmation set is opened.
b. A non-finite gradient norm with a finite loss (the `RuntimeError` of `clip_grad_norm_(..., error_if_nonfinite=True)`) is
   classified as a numerical failure like a non-finite loss, because an identical rerun reproduces it. Confirm, or say that only
   `FloatingPointError` counts.
c. Any other end of a run before completion (for example a CUDA out-of-memory error or an I/O error) is treated as an
   infrastructure failure and may be rerun twice. The registration defines infrastructure failure as walltime, node or I/O faults.
   Confirm this reading.
d. An `evaluate` job that ends before it completes (walltime, node fault) leaves `evaluation/manifest.json` at "running"; the
   stage then refuses to start ("already started") and the registration has no rule for an interrupted evaluation. The runs and
   episodes are fixed, so an identical re-evaluation (the old `evaluation/` moved aside by hand) would be the natural rule, but it
   is a decision about the confirmation set and is left to the orchestrator.
e. "T0 is retrained under the same code version" is implemented as one `source_hash` (blake2b of `src`, `scripts`, `tests`, `configs`,
   `pyproject.toml`, `uv.lock`; not `docs/` or `reports/`) for all 18 records, complete or failed, checked before `evaluate` opens the
   confirmation set and again by the analysis; the sheet submits every TSM job at one `CODE`. A change to any of those files between
   two jobs, even one unrelated to TSM, makes `evaluate` refuse until the orchestrator decides. Confirm this reading, or name a
   weaker one (for example, the TSM files only).

## 10. Step 1b (2026-10-27): aligned with the registrations

Master (6dcf271: the preregs registered at 40ddf08 and the step-1b task of b3ba3b4) was merged into `wp/sym` (merge commit
a928ea0, no conflict). The five prereg files on `wp/sym` are byte-identical to master's 40ddf08 and were not edited. Commits:
a928ea0 (merge), b291c55 (TSM scripts and tests), 2b36cde (fixes after the independent review, 10.2) and the commit of this
report. Only the TSM scripts and their tests changed (plus this report); the IVD, CVH, UNS and NCR code did not.

| registered rule (`docs/prereg/TSM.md`, "Decisions at registration") | implementation |
| --- | --- |
| infrastructure failure: identical rerun, at most twice | `train --rerun-incomplete` (the job that started the run must have ended; a run still marked "running" must also have been silent for 300 s): the unfinished run's directory and log move to `<root>/attempts/<run>-attempt<k>`, the run is trained again with the same configuration hash (so the same seeds and data stream) and the same `source_hash` as the latest earlier attempt, at most three attempts per run (then `train` refuses and the case goes to the orchestrator); every attempt is compared with the one before it, also when a rerun job stopped before its child started; each record carries `attempt` and `source` |
| non-finite loss in T1/T2: registered failure | the record is `failed`, the siblings go on, the job exits 0 when every selected run is final, the manifest `train_manifest-<label>.json` lists the failure under `training_failures`; never retried, not even with `--rerun-incomplete`; `evaluate` evaluates the completed runs and the six B4 references and lists the failures in its manifest; `analyze_tsm.py` marks the (writer, arm) cell `training_failed`, `passes` false, no registered endpoint, reports the stay, far, gain and four other contrasts of its completed seeds as descriptive (`descriptive_contrasts.csv`, with `seeds`; T0 and B4 restricted to the same seeds) and lists the failure in `summary.md` and `summary.json` |
| non-finite loss in T0: the study stops | the job ends with an error (manifest `status: stopped`, siblings terminated); every later `train` and `evaluate` refuses, and `evaluate` does so before the confirmation episodes are generated |
| `evaluate` no longer needs all 18 runs | it needs every T0 run complete and every T1/T2 run complete or a registered failure, each on the registered configuration and steps, with the checkpoint seal intact, and one `source_hash` for all 18 records; an unfinished run is refused with the rerun hint; all of this is checked before `evaluation/manifest.json` is written |
| an arm passes only for both writers; a single-writer pass is writer-specific | `summary.md` has a "Registered reading" table (per arm, one column per writer: passes / fails / training failure, and the outcome `both`, `writer_specific` = "writer-specific result, not a remedy", or `none`) and one line per writer ("no arm passes", or the arms that pass; a training failure counts as not passing), so the "no arm passes" statement can be worded per writer; `summary.json` has `arm_outcome`, `passing_writers`, `failed_cells`, `training_source_hash` |

Interpretations made in step 1b (none changes a registered threshold or statistic):
- A run's state is read from its `record.json`: `complete`; `training_failure` when the status is `failed` and the failure text
  is a `FloatingPointError` (the non-finite loss that `train` and `tsm.fit_tsm` raise) or says "non-finite" (the `RuntimeError`
  of `clip_grad_norm_(..., error_if_nonfinite=True)`); `incomplete` for anything else (still `running` after its job ended,
  failed with another error, or an unreadable record); `missing` without a run directory. Open questions (b) and (c).
- `--rerun-incomplete` is deliberate, not automatic: several jobs share one root and the code cannot see whether the job that
  started a run is still alive, so the operator confirms with `qstat` that it has ended. As a second guard (10.2, F1) `train`
  refuses a run that is still marked "running" when anything in its directory or log was written in the last 300 s: a live
  run appends to `train_log.jsonl` every `log_every` = 10 steps (the released configuration), seconds apart, so 300 s of silence
  means the run is dead; a record that says `failed` is exempt. Without the flag an unfinished run stops `train` before
  anything starts. Every job needs its own `--label`; a manifest is never overwritten.
- "Same code" is the `source_hash` of `looplora.sym.run.source_metadata` (blake2b over `src`, `scripts`, `tests`, `configs`,
  `pyproject.toml`, `uv.lock`; `docs/` and `reports/` are outside it, so committing a lock does not change it): submit a rerun at
  the same `--commit`. A different hash is refused as "not identical code". For the registered T0 ("retrained under the same
  code version") the same hash is required across all 18 records: open question (e).
- A child that died before it created its run directory has trained nothing: its log is set aside as
  `attempts/<run>-notstarted<k>.log` and no attempt is counted.
- A registered failure does not fail the job, because the registered study goes on; the stdout line and the manifest name it, and
  the command sheet says to read them after every job.
- A T0 failure stops the job in which it happens (its running siblings are terminated) and every job started later; jobs of other
  seeds that are already running finish their own runs unless the orchestrator `qdel`s them. `evaluate` refuses either way.

### 10.2 Independent review of b291c55 and the fixes

A second read-only subagent (82 tool calls over about 49 minutes; it changed nothing in the worktree; scratch files in the session
scratchpad) reviewed `a928ea0..b291c55` (`run_tsm.py`, `analyze_tsm.py`, `test_tsm_scripts.py`). Its method: the tests' own bodies
against torch-free fakes (21 of 21 passed), 37 one-line mutants of the two scripts run against the tests (31 killed), a random
walk over kill points of the train stage (30 seeds, no violation), and real subprocess scenarios with a stub torch and sleeping
stand-ins for training (SIGTERM to the process group as the walltime kill). It found no defect that changes an endpoint, a verdict or a
registered statement, and found that attempts never exceed three, registered failures are never restarted, `evaluate` succeeds iff
every T0 run is complete and every T1/T2 run is complete or a registered failure, and the audit checks every seal before any
statistic. Its findings and what was done (2b36cde):

| # | finding on b291c55 | disposition |
| --- | --- | --- |
| F2 | the identity check ran only for a run still in place; after a `--rerun-incomplete` job moved attempt 1 aside and ended before its child wrote a record, the run was `missing` and the next job trained attempt 2 on different code unchecked (reproduced with processes) | fixed: `check_rerun` compares with the latest earlier record, in place or in `attempts/`, for every run that is not a first attempt, in `stage_train` (also `missing` runs) and in `train_one`; tests `test_an_aborted_rerun_does_not_loosen_the_identity_rule`, `test_an_attempt_must_be_identical_...`; the process scenario now refuses the changed code and trains the identical rerun as attempt 2 |
| F1 | `--rerun-incomplete` treated every "running" record as dead: a rerun job whose selection overlapped a live job moved the live run aside and started a second child, whose files the first child then overwrote in the new directory (reproduced with processes) | fixed: the 300 s silence rule above; test `test_a_run_that_may_still_be_running_is_not_moved_aside` (a fresh file in an old directory, in the run or in its log, is refused); the process scenario refuses the overlapping job while the first job completes untouched, and accepts the rerun after a kill once the (shortened) silence has passed. The command sheet restricts a rerun job to the runs of the failed job |
| F3 | an `OSError` from `start_child` (for example its log existing because another job runs the same run) escaped `schedule`; the train manifest stayed "running" with no reason | fixed: `schedule` raises a `ValueError`, the existing handler writes a `failed` manifest; tests in `test_schedule_...` and `test_a_child_that_cannot_be_started_...` |
| F4 | `stage_train` skipped a complete or failed record of another configuration or steps; `evaluate` refused it later, after the job slots were spent | fixed: `registered_record` is checked for every final run of the selection at the start of `train`, and by `final_states` |
| F5 | `evaluation/manifest.json` ("running") was written before the checkpoint seals were checked; a missing or corrupt checkpoint, or a kill during `evaluate`, blocked every rerun | partly fixed: the digests of all complete runs are verified in `final_states`, before the manifest is written (a missing file is a refusal, not a crash); a killed `evaluate` still blocks a rerun: open question (d) |
| F6 | the audit did not require one code version for the 18 runs although T0 is "retrained under the same code version" | fixed as an interpretation: one `source_hash` across all 18 records, checked by `evaluate` before the confirmation set is opened and by the analysis, recorded as `training_source_hash` in `summary.json`; open question (e) |
| F7 | `final_states` stopped at the first unfinished run in plan order, hiding a later T0 failure's "the study stops" message | fixed: T0 failures are checked first; test `test_an_unfinished_run_earlier_in_the_plan_does_not_hide_a_t0_failure` |
| F8 | the outcome "passes for neither writer" was also printed for arms with a training failure, and the per-writer "no arm passes" wording existed only as a pointer to the table | fixed: the outcome says a training failure counts as not passing, and one line per writer is printed |
| T1 | the gate-profile audit check was never exercised (deleting it left every test green) | new assertion in the end-to-end test: a missing T2 profile is refused |
| T2 | the configuration/steps check of `final_states` was untested | `test_evaluate_needs_...` rewrites `steps` and `config_hash` of a record |
| T3 | no stage-level identity test, no aborted-rerun test, no live-run test | the tests of F1 and F2 above, and a changed `source_hash` refused at the stage |
| T4 | the stand-in children ran synchronously, so "running siblings are stopped" was never exercised | a `hang` outcome: a sibling still running when a T0 run fails is stopped |
| T5 | the attempt guard of `train_one` and the pre-load digest check of `evaluate_runs` had no test | `test_train_one_refuses_a_fourth_attempt_and_creates_nothing`, `test_the_evaluation_itself_refuses_a_checkpoint_that_changed_after_training` |

### 10.3 Verification (all static or torch-free; the tests have never been run with torch or pytest)

- `python3 -m pyflakes` (2.4.0, Python 3.9.25) on the three changed Python files: clean. `compile()` of the three files under
  Python 3.9.25 and 3.13.13 (`-I`, no bytecode written): ok. The `ast` checks of step 1 re-run on the final tree: 660
  module-attribute references with none unresolved, 1002 calls with the same 4 re-export hits as in §7, no hit in the TSM files
  (the checker lists 10 hits elsewhere in the repository, in files this work did not touch), 108 test functions in the ten files
  (93 at step 1; `test_tsm_scripts.py` went from 5 to 20) whose fixtures are builtin or defined in the file.
- The repository's own test bodies of `tests/test_tsm_scripts.py`, run through a stand-in for pytest's fixture machinery (module
  fixtures, `tmp_path`, `monkeypatch`, `raises`, `parametrize`) on the login node with the venv python, a stub `torch` package, and
  `SymbolicModel`, `train`, `train_arm`, `load_tsm`, `evaluate_arm`, `gate_profile`, `load_frozen` and `evaluation_arrays`
  replaced by file and numpy stand-ins (the stage code, the state handling, the manifests, the audit, `tsm.arm_endpoints` and
  the `followup_stats` bootstraps are the repository's): 27 of 27 cases passed (20 functions; the nine cases of
  `test_run_state_...` count separately; `test_t0_is_the_unchanged_training_function` is skipped: it needs the real `train()`).
  Run against b291c55's scripts (with the one constant the tests set added), 11 of the 27 cases fail there: the tests of F1, F2, F3
  and F7, the new assertions for F4, F5, F6 and F8 in the end-to-end, partial-failure, evaluate and wording tests, and the
  fourth-attempt guard (whose message differs); the other 16 pass there too, among them the stage-level identity test, the T0 stop
  with a running sibling and the evaluation's own checkpoint guard, which only add coverage.
- Mutation check of the new logic, by the same stand-in run: 27 one-line mutants of `run_tsm.py` and `analyze_tsm.py` (no identity
  check for missing or incomplete runs, none in `train_one`, the earlier record read only in place, the silence rule off, failed
  records not exempt from it, no registered-record check in `train` or `final_states`, no checkpoint digest, no single
  `source_hash`, the T0 failure checked after the loop, no gate-profile check, `reruns` of the manifest counted by the job that
  moved the run, `OSError` not converted, siblings not stopped, the T0 stop check removed or never fatal, the attempt count off
  by one, the configuration or the code not compared, no per-writer line, no `training_source_hash`, no stale-log set aside,
  a rerun without the flag, the silence rule looking at the directory only): all 27 were killed by at least one test. One
  survived the first version of the live-run test (the directory-only rule) and led to its second half: a fresh file in an old
  directory. The reviewer's mutants of b291c55 had left six survivors; they correspond to T1, T2, T3 and T5 above (T4 was a gap
  that its process scenario showed).
- Real subprocess scenarios (real `Popen` children, logs, SIGTERM to the process group, stub torch, sleeping stand-ins for training;
  the reviewer's scripts adapted): an overlapping rerun job is refused while the first job completes untouched; after a kill the
  rerun is refused within the silence period and accepted after it (period shortened to 3 s for the run); an aborted rerun
  followed by changed code is refused, the same code trains attempt 2 without the flag and sets the stale log aside; a T0
  non-finite loss ends the job as `stopped`, terminates the siblings (no process left), and later jobs refuse; a T1 non-finite
  loss lets the other runs finish, exits 0, and `--rerun-incomplete` leaves the failed record alone.
- Not verified: any execution with real torch or real pytest (including the fixtures and `monkeypatch` of the new tests, the one
  real-torch test, `train_one` with a real `FloatingPointError` from `fit_tsm`), a PBS walltime kill or I/O fault on a compute
  node, filesystem timestamps across nodes on the shared filesystem (the 300 s rule assumes clock differences of seconds), and
  the actual `log_every` cadence of a T1/T2 run on a GPU. The first compute-node run of `tests/test_tsm_scripts.py` is part of the
  step-0 test job of §6.
