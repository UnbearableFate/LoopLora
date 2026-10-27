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

Step 1c (2026-10-27, §10.4): the orchestrator's Clarification C1 of `docs/prereg/TSM.md` (master 3477526, identical on `wp/sym`
at 68c19da) answered the questions that were open after step 1b (§9). Two answers changed the TSM scripts: only CUDA
out-of-memory and `OSError` are infrastructure failures, any other exception is a code defect that stops the study like a T0
failure, and an interrupted `evaluate` is restarted from scratch with identical inputs, at most twice, with its partial output
moved aside unread. The §6 sheet states the same operational rule for the evaluation jobs of IVD, CVH, UNS and NCR. As before,
nothing was run with torch, pytest or on a compute node; §10.5 says what was checked and how.

Step 1d (2026-10-27, §10.6): Clarification C2 of `docs/prereg/TSM.md` (master d488b78, identical on `wp/sym` at a5c093a) answered the
six points left open after step 1c (§9, item 7), and the TSM failure protocol is closed: any further edge case is decided by the
orchestrator when it occurs and recorded in this report. Three changes to the TSM scripts: `train --max-attempts N` (default 3, recorded in
the manifest) grants further identical attempts; an allocation failure (CUDA out-of-memory, a cuBLAS or cuDNN allocation failure, a host
`MemoryError`) is an infrastructure failure, and any other CUDA runtime error stays a code defect; `evaluate --restart-interrupted
--new-evaluation-code` lets a restart run at another evaluation `source_hash`, and the evaluation manifest, `summary.json` and `summary.md`
record the code of every attempt. The §6 sheet states all three. As before, nothing was run with torch, pytest or on a compute node;
§10.7 says what was checked and how.

Step 1e (2026-10-27, §10.8): Clarification C3 of `docs/prereg/TSM.md` (master d335fe7, identical on `wp/sym` at 3ca4d89) and the independent static
review of the failure rules (`docs/review/static_2026-10-27/tsm.md`, findings F1 to F8; it found no break of a registered quantity or decision rule)
reopened the TSM failure protocol once, and the scripts changed accordingly. The first evaluation attempt runs at the training code (F1), and `train`
refuses a code that differs from any record of its root (F2). torch's checkpoint-serialization errors, its CPU-allocator failure and
`CUBLAS_STATUS_NOT_INITIALIZED` reported by `cublasCreate` are infrastructure failures (F3, F4). A stopping failure starts no pending run and lets the
running ones finish (F5). The pause after three attempts and the orchestrator's recorded decision (`train --max-attempts N --decision FILE`) are enforced
in code (F6), a live evaluation writes a heartbeat file (F7), and the §6 sheet words the check of an evaluation fix at code level (F8). As before,
nothing was run with torch, pytest or on a compute node; §10.9 says what was checked and how.

Step 1f (2026-10-27, §10.10): Clarification C4 of `docs/prereg/TSM.md` (master d6ceb6a, identical on `wp/sym` at 76a00b3) and the second static review
(`docs/review/static_2026-10-27/tsm2.md`: the process management F5, the pause F6, the one-code rules and the heartbeat behave as specified; one
classification defect and one dead end in the evaluation rules) changed the TSM scripts once more. The two texts of torch's checkpoint writer that say a file
cannot be created, "File <path> cannot be opened." and "Parent directory <dir> does not exist.", are I/O errors, hence infrastructure failures (finding 1).
A first evaluation attempt at another code than the training code, when neither `evaluation/` nor `attempts/evaluation-attempt*` exists, is accepted
with `evaluate --new-evaluation-code --decision FILE`, and the manifest records the file's path and blake2b digest (finding 2). Three notes of the review are
done: a stopped `train` manifest records the stopping child's run, exit code and log, the sheet says that a manifest left at "running" means the job was
killed, and the heartbeat thread's `join` waits at most 2 x `HEARTBEAT_SECONDS`. As before, nothing was run with torch, pytest or on a compute node; §10.11
says what was checked and how.

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

# 4. TSM   (after the others; failure rules: docs/prereg/TSM.md, "Decisions at registration" and Clarifications C1 to C4)
$SUB --name tsm-lock  --queue debug-g --walltime 00:05:00 --commit $CODE --wait -- scripts/run_tsm.py lock --out $SYM/reports/tsm/design.json
#    STOP: report $SYM/reports/tsm/design.json; the orchestrator commits it on wp/sym. TSM_LOCK = that commit's full SHA
$SUB --name tsm-smoke --queue regular-g --walltime 00:20:00 --commit $CODE --wait -- scripts/run_tsm.py smoke --design $SYM/reports/tsm/design.json --root $SYM/runs/tsm/smoke --procs 6
for S in 17 29 43; do   # one job per seed (6 runs each); walltime from smoke.json extrapolated_run_seconds x 1.5 / procs
  $SUB --name tsm-train-$S --queue regular-g --walltime <from smoke> --commit $CODE -- scripts/run_tsm.py train --design $SYM/reports/tsm/design.json \
    --design-commit $TSM_LOCK --root $SYM/runs/tsm/main --procs 6 --seeds $S --label s$S
done
#    After every job read $SYM/runs/tsm/main/train_manifest-<label>.json (exit status 0 = every selected run is final):
#      "training_failures" lists registered failures (a non-finite loss or gradient norm in a T1/T2 run): no retry, no re-tuning; report them, the evaluation goes on;
#      status "stopped": the study stops. Report it to the orchestrator; do not evaluate and submit no further TSM job (jobs that are already running
#        may finish: the stopped job itself starts no pending run, lets the runs that were already running finish, keeps their records and then ends
#        with an error; the manifest's "run_states" says what each selected run ended as, "missing" for a run that was never started, and
#        "stopping_child" gives the run, the exit code and the log (logs/<run>.log) of the first child whose exit stopped the job). Two causes,
#        named in the manifest's "reason" (which names the first run in the plan's order, a code defect before a T0 failure): a non-finite loss or
#        gradient norm in a T0 run, or a code defect, that is a run that ended with any exception other than those, an I/O error and an allocation
#        failure. An I/O error is an OSError, or a failure of torch's checkpoint writer or reader to create, open, write or read a file (a RuntimeError
#        "PytorchStreamWriter failed ...", "PytorchStreamReader failed ...", "File <path> cannot be opened." or "Parent directory <dir> does not
#        exist."; the same condition through Python's open is an OSError: Clarifications C3 and C4). An allocation failure is CUDA
#        out-of-memory, a cuBLAS or cuDNN allocation failure, CUBLAS_STATUS_NOT_INITIALIZED reported by cublasCreate, torch's CPU-allocator failure
#        ("DefaultCPUAllocator: not enough memory") or a host MemoryError (Clarification C3). Any other CUDA runtime error, an illegal memory access
#        for example, and CUBLAS_STATUS_NOT_INITIALIZED from any other call, is a code defect (its record.json says failure_kind code_defect; the
#        traceback is in logs/<run>.log). A code defect that was misread as one of the failures above repeats identically and reaches the pause
#        below (C3); if the diagnosis then finds a code defect, the stop applies. After a code defect the fixed code must retrain every TSM run, in
#        a new --root (one root holds one source_hash, and every later train of the stopped root refuses), before the confirmation set is opened: the
#        orchestrator decides, nothing continues by itself (TSM.md, Clarification C1). If the orchestrator's diagnosis finds a hardware fault behind
#        the error it records that, and the run is retried as an infrastructure failure (C2): the scripts have no option for it, so the operator moves
#        runs/<run> to attempts/<run>-attempt<k> and logs/<run>.log to attempts/<run>-attempt<k>.log (k = the attempts of that run already moved + 1: what
#        --rerun-incomplete does) and submits the identical train job again, which counts as an attempt;
#      status "incomplete" (the job ended before its runs did: walltime or node fault, an I/O error or an allocation failure as above; failure_kind
#        infrastructure, and for an allocation failure the remedy may be a lower --procs): once qstat shows the job has ended, submit the
#        identical command again at the same --commit with a new --label (s17-r2) and --rerun-incomplete, restricted to the runs of that job
#        (--seeds ...); at most twice per run (3 attempts). A run that is not final and has used its 3 attempts, the last one ended, pauses the study
#        (Clarification C1): `train` then refuses every job, whatever it selects. Report it: the orchestrator diagnoses. After the orchestrator's recorded
#        decision, written to a file, the same command with --max-attempts N --decision FILE allows N attempts in all (--max-attempts 4 permits a
#        fourth attempt; the default is 3, a smaller N is refused, an N above 3 is refused without --decision, FILE is an absolute path of a non-empty
#        file, the manifest records N and the file's path and blake2b digest, and every attempt is still compared with the one before it). The pause is
#        read at each job's own N: until the paused run is final, every later `train` job, whatever it selects, passes the same --max-attempts N --decision
#        FILE (a run that fails again at its N-th attempt pauses the study again). `train` refuses a
#        run marked "running" that was written to in the last 5 minutes
#        (its job may be alive): wait and submit again. A rerun job that itself ends before its runs started is submitted again the same way.
#        A run reported as unfinished that has no record.json (an error before its run directory exists: its child died, or the design, the
#        configuration or the code version could not be read) is not rerun blindly: read logs/<run>.log and report the traceback; the orchestrator
#        classifies it (C1, C2);
#      status "running" in a manifest whose job has ended (qstat shows it): the job was killed (walltime or a node fault) before it could write its end,
#        possibly in the middle of the drain after a stop, so the manifest says nothing about how its runs ended: read the run states. Each run's
#        runs/<run>/record.json has a status ("complete", "running" or "failed") and, for a failure, a failure_kind: numerical (a registered failure of a
#        T1/T2 run, a stop for T0), code_defect (a stop) or infrastructure. A stop is also enforced from these records by every later `train` and by
#        `evaluate`, which refuse. A run that is neither complete nor a registered failure is unfinished: rerun it as under status "incomplete".
#      All 18 runs must come from the same $CODE and every attempt must equal the one before it (configuration and source_hash): `train` refuses a job
#      whose source_hash differs from that of any record under runs/ or attempts/ of its root, before it starts anything, and `evaluate` and the
#      analysis refuse otherwise (a different code version is a decision for the orchestrator, §9; a retraining at another code starts in a new --root).
$SUB --name tsm-eval  --queue regular-g --walltime <about 00:20:00> --commit $CODE -- scripts/run_tsm.py evaluate --design $SYM/reports/tsm/design.json --design-commit $TSM_LOCK --root $SYM/runs/tsm/main
#    tsm-eval runs at the training $CODE: a first attempt (neither evaluation/ nor attempts/evaluation-attempt* exists) at another source_hash is refused,
#    unless it is submitted with --new-evaluation-code --decision FILE on the orchestrator's recorded decision (below, Clarification C4).
#    A live evaluation writes evaluation/heartbeat every 30 s, so that the silence rule below cannot mistake it for an ended one.
#    tsm-eval ends before it completes (walltime or node fault; evaluation/manifest.json says "running", or "failed" if the job could write its own end):
#      once qstat shows the job has ended, submit the identical command again (same --commit, same $TSM_LOCK) with --restart-interrupted, at most twice
#      (3 attempts in all). `evaluate` moves the partial evaluation/ to attempts/evaluation-attempt<k> without opening anything in it, requires identical
#      inputs (design lock, source_hash, the 18 run checkpoints) and records `attempt` and `restarts` in the new manifest; a "running" manifest must have
#      been silent for 5 minutes (any file written in evaluation/ counts, the heartbeat included). Never open the moved files; report each restart. The
#      analysis audits the record and prints it in summary.md and summary.json.
#      A defect in the evaluation code (not in the training code) is for the orchestrator (C2, C3). It checks the fix at code level, not at file level,
#      and records the check with its decision: the fix must change no function, class or module-level statement that the training path executes (the
#      training path is `train`, `train-one` and everything they call, shared helpers included; evaluation and training share scripts/run_tsm.py and
#      src/looplora/sym/tsm.py, so a check by file could never pass). For example: a diff of the training CODE and the fixed commit over src/, scripts/ and
#      configs/, read hunk by hunk, where every hunk must lie in code that the training path does not execute. It names the fixed commit. The evaluation
#      is then submitted at that commit (--commit <fixed>, the same $TSM_LOCK) with `--restart-interrupted --new-evaluation-code`: the 18 trained runs
#      are kept, the partial evaluation/ is moved aside unread, and the restart runs at the new source_hash. Without --new-evaluation-code a restart at
#      another source_hash is refused, and the flag without --restart-interrupted is refused unless it makes a first attempt under --decision (below): an
#      evaluation in place, or moved aside by an aborted restart, is an earlier attempt. A restart that has no readable manifest of an earlier attempt to
#      compare with runs at the training $CODE, unless the flag says otherwise. A restart needs no decision file.
#      A defect that ends `evaluate` before its first evaluation/manifest.json exists (in the checks, the seals or the metadata that come before it) leaves
#      neither evaluation/ nor attempts/evaluation-attempt*: report it. The orchestrator handles it as the same evaluation-code defect (Clarification C4): it
#      records its decision, with the code-level check above, in a file, and the first attempt is then submitted at the fixed commit (--commit <fixed>, the same
#      $TSM_LOCK) with `--new-evaluation-code --decision FILE` (FILE: the absolute path of a non-empty file; no --restart-interrupted, there is nothing to
#      restart). Without both options a first attempt at another source_hash is refused. The manifest records the file's path and blake2b digest
#      (`decision`) and both codes (`source`, `training_source_hash`); the rows of `restarts` carry the decision of their attempt (`restarts[].decision`, null
#      for none), so that a later restart at the same code keeps it on record; summary.md names it ("on the orchestrator's recorded decision (Clarification C4;
#      decision file ..., blake2b ...)") and summary.json has evaluation_decision. A first attempt at another code with no decision recorded (a manifest edited
#      by hand, say) is printed as outside the rule. The scripts do not check that FILE is the orchestrator's decision, or what the code-level check found.
#      Nothing else may differ (the design, its lock commit --design-commit, the checkpoints), the 18 training records must still share one source_hash (checked),
#      and three attempts in all remain the limit. The manifest records the source_hash of this attempt (`source`), of every moved one
#      (`restarts[].source_hash`) and the one the 18 runs were trained at (`training_source_hash`); summary.json adds evaluation_source_hash, and
#      summary.md always prints both hashes ("the same code" or "a DIFFERENT code"), says so when the first attempt ran at another code (outside the
#      rule, unless a decision is recorded for it) and names the change of code at a restart with the source_hash of each attempt.
$SUB --name tsm-analyze --queue debug-g --walltime 00:15:00 --commit $CODE --wait -- scripts/analyze_tsm.py --root $SYM/runs/tsm/main --out $SYM/reports/tsm/results
```

Jobs longer than the Bash tool's limit are submitted without `--wait` and followed with `qstat` (or the
`miyabi-pbs-goal-wakeup` workflow). The lock files are written into the worktree because the `run` stages verify them with
`git show <commit>:<path>` in the file's own repository; a commit of the same file on master at the same relative path
is equally valid. TSM's rerun rules are in the comments of block 4: the `train` stage itself moves an unfinished run to `<root>/attempts/`
(with `--rerun-incomplete`), never touches a complete run or a registered failure, and `evaluate` refuses anything else; an `evaluate` job that
was interrupted is started again with `--restart-interrupted` as described there (without the flag it refuses: "already started"); `--max-attempts N
--decision FILE`, `--new-evaluation-code` and `evaluate --decision FILE` are described there too.

**Interrupted evaluation jobs of IVD, CVH, UNS and NCR.** The same operational rule as for TSM (Clarification C1) applies: a job that ends before
it completes (walltime, node or I/O fault) is restarted from scratch with identical inputs (same `--commit $CODE`, same lock commits and
arguments), at most twice (3 attempts in all); its partial output is moved aside without being opened, and each restart is reported to the
orchestrator. These scripts refuse an existing output directory (IVD `run`, UNS `run`: `mkdir(exist_ok=False)`; CVH `confirm`: "the confirmation
has already started") or overwrite files in it (NCR, UNS Part A), and they have no restart option, so after `qstat` shows that the job has ended the
operator moves the partial output by hand and submits the same job again unchanged. Nothing in the moved output is opened (not even `cat`
or `head`; `ls -l` is enough), it is never analysed, and it is not committed (`runs/` is git-ignored, `reports/` is not: the output of
NCR and UNS Part A is moved out of `reports/`). Their manifests record nothing about it: the `*-attempt<k>` directories and the
report to the orchestrator are the record.

```bash
K=1                                  # 1 for the first restart, 2 for the second; a third restart is not allowed (report it instead)
AT=$SYM/runs/attempts; mkdir -p $AT
# IVD run:      mv $SYM/runs/ivd/main $AT/ivd-main-attempt$K          then submit ivd-run again
# UNS Part B:   mv $SYM/runs/uns/main $AT/uns-main-attempt$K          then submit uns-b again (the optional timing run is no confirmation evaluation)
# NCR:          mv $SYM/reports/ncr $AT/ncr-attempt$K                 then submit ncr again
# UNS Part A:   mv $SYM/reports/uns/part-a $AT/uns-part-a-attempt$K   then submit uns-a again
# CVH confirm:  its outputs share the training root, which must stay: move only the confirmation outputs
R=$SYM/runs/cvh/main; C=$AT/cvh-confirm-attempt$K; mkdir -p $C/traces
for f in manifest.json latency.json confirmation_timing.json; do [ -e $R/$f ] && mv $R/$f $C/; done
for f in $R/traces/*/test.npz; do [ -e "$f" ] && mkdir -p $C/traces/$(basename $(dirname $f)) && mv $f $C/traces/$(basename $(dirname $f))/; done
#               (train_manifest.json, theta.json, heads/ and traces/*/val.npz stay), then submit cvh-confirm again
```

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
| TSM | T0 = released recipe at the fixed depth 4 (P2-S-F1 recipe), retrained; stay uses L4 and L16, far uses L8 for arm and T0 (L16 reported), gain uses L4 against B4 at L1; an arm passes only if both writers pass (rv-astra), intervals corrected for four comparisons; stay-loss weighting (mean over steps s ≥ hop, equal over queries); T2 without PonderNet step cost, with explicit init (a, b) = (0, 4); same released lr (0.003) for every arm, no retry; training failures as registered at 40ddf08 and in Clarifications C1 to C4, implemented in steps 1b to 1f (§10): a non-finite loss or gradient norm in T1/T2 fails that arm for that writer (completed seeds reported descriptively, other arms evaluated), in T0 it stops the study, an infrastructure failure (walltime, node fault, `OSError`, a failure of torch's checkpoint writer or reader to create, open, write or read a file, an allocation failure: CUDA out-of-memory, a cuBLAS or cuDNN allocation failure, `CUBLAS_STATUS_NOT_INITIALIZED` reported by `cublasCreate`, torch's CPU-allocator failure, a host `MemoryError`) is rerun identically at most twice (more with `train --max-attempts N --decision FILE` on the orchestrator's recorded decision; a run that is not final and has used its attempts pauses the study and `train` refuses to start), any other exception is a code defect that stops the study (the job starts no pending run and lets the running ones finish), one root holds one `source_hash`, an interrupted `evaluate` is restarted from scratch with identical inputs at most twice (the first attempt runs at the training code; another evaluation code only at a restart, with `--new-evaluation-code`, or at a first attempt when neither `evaluation/` nor `attempts/evaluation-attempt*` exists, with `--new-evaluation-code --decision FILE`), a single-writer pass is a writer-specific result, not a remedy |
| UNS | split into Part A (zero-evaluation audit of the committed rows) and Part B (what is missing; registered as run, not optional); "M1-L4 and B4" covered by Part A (released M1 at L4/L1, B4) and Part B (M1 trained at L = 4); `oracle_random` added as a control; a 3 pp wording rule for the scope row (changes no gate); disclosure that the unseen rows were seen before registration |
| NCR | single read = L = 1 of M1 and B4; both rate and share reported; category priority and the h = 2 case fixed before any neighbour-confusion quantity is computed; the already analysed C777 confirmation samples are re-read, not re-evaluated |

Answered (orchestrator; 40ddf08, the step-1b message, for item 6 Clarification C1 of `docs/prereg/TSM.md` (master 3477526, identical on
`wp/sym` at 68c19da), for item 7 Clarification C2 (master d488b78, identical on `wp/sym` at a5c093a) and for item 8 the step-1e message with
Clarification C3 (master d335fe7, identical on `wp/sym` at 3ca4d89) and for item 9 the step-1f message with Clarification C4 (master d6ceb6a, identical on
`wp/sym` at 76a00b3)), no longer open:
1. UNS Part B runs; it is not optional (only its two-checkpoint timing run stays optional).
2. TSM: both writers must pass; a single-writer pass is reported as writer-specific, not as a remedy. Failures: infrastructure
   failure, identical rerun, at most twice; non-finite loss in T1/T2, that arm fails for that writer and the other arms are still
   evaluated; non-finite loss in T0, the study stops.
3. The orchestrator commits each design lock and `theta.json` on `wp/sym`; the command sheet stops after each lock stage and
   reports the path.
4. The `--scale-up` allowances for IVD and TSM are accepted.
5. Master was merged into `wp/sym` (a928ea0).
6. The five questions left open after step 1b, answered by Clarification C1 (the step-1c message; the code follows it, §10.4):
   (a) three failed attempts of a run pause the study; the orchestrator diagnoses, and a remedy that changes no registered quantity
   permits further identical attempts, each reported (the scripts stop at three unless `train --max-attempts N` grants more: item 7);
   (b) a non-finite gradient norm is a numerical failure like a non-finite loss (as implemented in step 1b);
   (c) only CUDA out-of-memory and I/O errors (`OSError`) are infrastructure failures (the remedy may lower the concurrent runs per
   job); any other exception is a code defect: the study stops, as after a T0 failure, and the fixed code retrains every TSM run before
   the confirmation set is opened (step 1c, change 1);
   (d) an interrupted `evaluate` is restarted from scratch with identical inputs, at most twice, the partial output moved aside unread
   (step 1c, change 2); the same operational rule applies to an interrupted evaluation job of IVD, CVH, UNS and NCR (the §6 sheet);
   (e) one `source_hash` across all 18 runs, and every TSM training and evaluation job at one `CODE` commit (smoke exempt), as
   implemented in step 1b.

7. The six points left open after step 1c, answered by Clarification C2 (the step-1d message; the code follows it, §10.6):
   (1) after the orchestrator's recorded decision under C1, `train --max-attempts N` (default 3) grants further identical attempts (step 1d,
   change 1); (2) a restart reads only the interrupted attempt's `manifest.json` (status and inputs), never its samples or statistics (as in
   step 1c; no further code); (3) a defect in the evaluation code (not the training code): the 18 trained runs are kept, the orchestrator checks that
   the fix changes no file the training path uses, and the evaluation restarts from scratch at the fixed commit with `evaluate --restart-interrupted
   --new-evaluation-code`, the partial output moved aside unread and both code hashes recorded (change 3; the orchestrator's check is not
   automated); (4) the evaluation job's `source_hash` is recorded and compared with the training hash by no script: C2 asks for no check, the training
   jobs share one `CODE` commit and the evaluation job may use a later commit only under C2 (no further code; superseded in step 1e: item 8, F1, the
   first evaluation attempt now has its `source_hash` compared with the training hash by `evaluate`); (5) allocation failures (CUDA
   out-of-memory, cuBLAS or cuDNN allocation failures, host `MemoryError`) are infrastructure failures and other CUDA runtime errors are code defects
   (change 2); if the orchestrator's diagnosis finds a hardware fault, it records that and the run is retried as an infrastructure failure (a
   manual move in the §6 sheet, no option in the scripts); (6) an error before a run directory exists leaves no record: the operator reports the
   traceback and the orchestrator classifies it (the sheet; no further code).

8. The findings of the independent static review of the failure rules (`docs/review/static_2026-10-27/tsm.md`) that the step-1e message and Clarification C3
   answered (the code follows them, §10.8): (F1) the first evaluation attempt runs at the training code, and `--new-evaluation-code` needs an earlier
   attempt (the second half is superseded in step 1f: item 9); (F2) `train` refuses a `source_hash` that differs from any record under `runs/` or `attempts/` of its root, in `stage_train` and in `train-one`
   before `mkdir`; (F3, F4) torch's checkpoint-serialization errors ("PytorchStreamWriter failed", "PytorchStreamReader failed"), its CPU-allocator failure
   ("DefaultCPUAllocator: not enough memory") and `CUBLAS_STATUS_NOT_INITIALIZED` reported by `cublasCreate` are infrastructure failures, and a code defect
   that is misread as one of them repeats identically and reaches the pause (C3, bounded exposure); (F5) a stopping failure starts no pending run, the
   running runs of the same job finish and their records are kept; (F6) `train` refuses to start while a run that is not final has used its attempts, and
   `--max-attempts N` above 3 needs `--decision FILE` (the manifest records the file's path and blake2b digest); (F7) an evaluation heartbeat file, and a test
   that pins `QUIET_SECONDS == 300`; (F8) the orchestrator's check of an evaluation fix works at code level and is recorded with its decision (C3; the sheet
   words it, the check is not automated). The `--rerun-incomplete` help text names the allocation failures and `--max-attempts`.

9. The findings of the second static review (`docs/review/static_2026-10-27/tsm2.md`) that the step-1f message and Clarification C4 answered (the code follows
   them, §10.10): (1) the writer's texts "File <path> cannot be opened." and "Parent directory <dir> does not exist." are I/O errors like the C3 texts, so the
   same condition is one classification whether it goes through torch or through Python's `open`; the bounded exposure of C3 applies; (2) a defect that ends
   `evaluate` before its first manifest exists is an evaluation-code defect of C2: the orchestrator records its decision, with the code-level check, and the
   first evaluation attempt runs at the fixed commit with `--new-evaluation-code --decision FILE`, both code hashes recorded; and three notes: the stopped
   `train` manifest records the stopping child, a manifest left at "running" means that the job was killed (the sheet), and the heartbeat's `join` has a timeout.

Nothing is open for the orchestrator after step 1f. The TSM failure protocol is closed again (Clarification C4): a further edge case is decided by the
orchestrator when it occurs and recorded in this report.

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
  failed with another error, or an unreadable record); `missing` without a run directory. Open questions (b) and (c) of step 1b;
  superseded in step 1c (§10.4): the kind of a failure is recorded when it happens, and an exception that is neither numerical, an
  I/O error nor CUDA out-of-memory is a code defect, not `incomplete` (CUDA out-of-memory became "an allocation failure" in step 1d, §10.6).
- `--rerun-incomplete` is deliberate, not automatic: several jobs share one root and the code cannot see whether the job that
  started a run is still alive, so the operator confirms with `qstat` that it has ended. As a second guard (10.2, F1) `train`
  refuses a run that is still marked "running" when anything in its directory or log was written in the last 300 s: a live
  run appends to `train_log.jsonl` every `log_every` = 10 steps (the released configuration), seconds apart, so 300 s of silence
  means the run is dead; a record that says `failed` is exempt. Without the flag an unfinished run stops `train` before
  anything starts. Every job needs its own `--label`; a manifest is never overwritten.
- "Same code" is the `source_hash` of `looplora.sym.run.source_metadata` (blake2b over `src`, `scripts`, `tests`, `configs`,
  `pyproject.toml`, `uv.lock`; `docs/` and `reports/` are outside it, so committing a lock does not change it): submit a rerun at
  the same `--commit`. A different hash is refused as "not identical code". For the registered T0 ("retrained under the same
  code version") the same hash is required across all 18 records (open question (e) of step 1b; confirmed by C1).
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
| F5 | `evaluation/manifest.json` ("running") was written before the checkpoint seals were checked; a missing or corrupt checkpoint, or a kill during `evaluate`, blocked every rerun | partly fixed: the digests of all complete runs are verified in `final_states`, before the manifest is written (a missing file is a refusal, not a crash); a killed `evaluate` still blocked a rerun (open question (d) of step 1b; answered by C1 and implemented in step 1c, §10.4) |
| F6 | the audit did not require one code version for the 18 runs although T0 is "retrained under the same code version" | fixed as an interpretation: one `source_hash` across all 18 records, checked by `evaluate` before the confirmation set is opened and by the analysis, recorded as `training_source_hash` in `summary.json`; open question (e) of step 1b, confirmed by C1 |
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

### 10.4 Step 1c (2026-10-27): Clarification C1 in the code

Task: `docs/tasks/WP-S9_symbolic_followups.md`, section "Step 1c" (master f94cf52). Rules: Clarification C1 of `docs/prereg/TSM.md`
(master 3477526, byte-identical on `wp/sym` at 68c19da; not edited). Commits: 55f0499 (`scripts/run_tsm.py`, `scripts/analyze_tsm.py`,
`tests/test_tsm_scripts.py`) and the commit of this report. Nothing else changed: the IVD, CVH, UNS and NCR code, the five preregs,
`pyproject.toml` and `uv.lock` are as they were; for those four studies the change is a rule in the §6 sheet.

| rule (Clarification C1) | implementation |
| --- | --- |
| only CUDA out-of-memory and I/O errors are infrastructure failures | `train_one` records every exception that ends a run after its directory exists (the setup, the training, and the seals that follow it) as `status: failed` with `failure_kind` from `classify_failure(error)`: `infrastructure` for an `OSError` (any subclass), `torch.cuda.OutOfMemoryError`, a `RuntimeError` whose text says "cuda" and "out of memory" (the CUDA runtime's own "CUDA error: out of memory"), or an interruption that is not an `Exception` (`KeyboardInterrupt`, `SystemExit`); `numerical` for a `FloatingPointError` and for the `RuntimeError` of the gradient clipping that says "non-finite"; `code_defect` for everything else. `run_state` reads the kind, not the text: `training_failure` (numerical), `code_defect`, `incomplete` (still running, infrastructure, unreadable record) |
| any other exception is a code defect: the study stops, as after a T0 failure; every later `train` or `evaluate` refuses | `stops_study(root, run)` is true for a code defect in any run and for a numerical failure in a T0 run; `study_stop(root)` words the reason (a code defect first: "the fixed code must retrain every TSM run before the confirmation set is opened; report it to the orchestrator"). `schedule` ends the job at the first such run (running siblings are terminated; the manifest is `stopped` with the reason; exit status 1), `train` refuses before it writes a manifest, and `final_states`, hence `evaluate`, refuses before the confirmation episodes are generated and before the evaluation manifest exists |
| an interrupted `evaluate` is restarted from scratch with identical inputs, at most twice; the partial output is moved aside unread; the restart is recorded | `evaluate --restart-interrupted`, see the next paragraph. The evaluation manifest has `attempt`, `restarts` (one `{attempt, moved_to}` per moved evaluation) and `checkpoints` (the blake2b seal of every complete run's checkpoint); the analysis audits both and prints the restart in `summary.md` and `summary.json` (`evaluation_attempt`, `evaluation_restarts`) |
| the same operational rule for IVD, CVH, UNS and NCR | the §6 sheet: restart from scratch with identical inputs, at most twice, the partial output moved by hand into `$SYM/runs/attempts/` without being opened, each restart reported (the commands were run on a dummy tree) |

The evaluation restart. An evaluation directory in place is `complete`, `failed` (its manifest says so: the job caught the exception
that ended it, wrote it as `failure` and `finished_at`, and let it propagate), or `running` (the manifest a killed job leaves; also a
directory without a manifest, or with a cut-off one). A complete evaluation is never repeated. Anything else is refused unless the job
is started with `--restart-interrupted`; then, after `final_states` has accepted the 18 runs (a stopped study refuses also a restart) and
before any confirmation episode is generated, `evaluation_attempt` requires (i) fewer than three attempts used, the interrupted one in
place and those moved aside counting (a fourth is refused and goes to the orchestrator), (ii) for `running`, 300 s without a write
anywhere in the evaluation directory (its job may be alive; `failed` has recorded its own end), and (iii) identical inputs: the design
hash and its lock commit, the `source_hash` and the checkpoint seals, compared with the manifest of the latest earlier attempt, in place
or moved aside (an earlier restart that stopped before it wrote its manifest is still compared). `move_evaluation_aside` then renames
`evaluation/` to `attempts/evaluation-attempt<k>`; nothing in it is opened, and the new manifest carries `attempt = k + 1`.

Interpretations made in step 1c (none changes a registered threshold, statistic or sample):
- The kind of a failure is decided when the exception happens, from its type, and written into the record; the failure text is never
  parsed afterwards. A `failed` record without a `failure_kind` (this code writes none) counts as a code defect: nothing shows it was not one.
- CUDA out-of-memory is `torch.cuda.OutOfMemoryError` (in the pinned torch 2.14 `torch._C.OutOfMemoryError`, a `RuntimeError`; read in the
  installed source, not instantiated) or a `RuntimeError` that says both "cuda" and "out of memory". A host `MemoryError`, a host allocation failure and
  every other CUDA error were code defects in step 1c, as the registered text then had it; Clarification C2 made the cuBLAS and cuDNN allocation
  failures and the host `MemoryError` infrastructure failures (step 1d, §10.6), and every other CUDA error stays a code defect.
- An interruption that is not an exception of the code (`KeyboardInterrupt`, `SystemExit`) is infrastructure: the run was ended from outside. A
  walltime kill or a node fault leaves the record at "running", as before; the silence rule then decides when a rerun may start.
- Only the manifest of an interrupted evaluation is read, for its status and its inputs; it is written before any confirmation episode is
  generated and holds no sample or statistic. The samples and gate profiles are not opened (confirmed by C2: §9, item 7).
- `evaluate` marks its manifest `failed` for any exception, whatever its kind, and allows identical restarts only; it does not apply the
  code-defect rule inside the evaluation; C2 settled the case of a defect in the evaluation code (§9, item 7; `--new-evaluation-code`, §10.6).
- `MAX_ATTEMPTS` = 3 also caps the evaluation (the first attempt and two restarts). Since step 1d `train --max-attempts N` lifts the cap of the
  training runs (§10.6); the evaluation keeps its three attempts.

### 10.5 Verification of step 1c (all static or torch-free; nothing was run with torch, pytest or on a compute node)

- `python3 -m pyflakes` (2.4.0, Python 3.9.25) on the three changed Python files: clean. `py_compile` with the bytecode written outside the
  worktree under Python 3.9.25 and 3.13.13 (`-I -B`): ok; no `__pycache__` in the worktree. The `ast` checks of step 1 on the final tree:
  730 module-attribute references, none unresolved; 1075 calls with 5 hits, the 4 re-exports of §7 and `run_tsm.digest_checkpoint` in a new
  test (`from s9_common import digest_checkpoint`, `run_tsm.py:51`; it resolves at run time). 118 test functions in the ten files (108 after
  step 1b); `test_tsm_scripts.py` went from 20 to 30: ten new functions (classification; a code defect in a T0, T1 and T2 run;
  a code defect reported before a T0 failure and an unfinished run; out-of-memory and I/O errors rerun, even for T0; every error after the run
  directory exists is recorded with its kind; the restart with the partial output unreadable, audited and tampered with; at most two
  restarts; the restart waits for the silence; identical inputs, also after an aborted restart; a directory without a readable manifest)
  and `test_run_state_...` extended with the failure kinds.
- The repository's own test bodies of `tests/test_tsm_scripts.py` through the stand-in for pytest's fixture machinery of 10.3 (venv python,
  stub `torch` package whose `torch.cuda.OutOfMemoryError` derives from `RuntimeError` as in the pinned torch, file and numpy stand-ins for
  the model, `train`, `train_arm`, `evaluate_arm` and the loaders): 56 of 56 cases passed (29 functions; 16, 10, 3 and 2 cases for the four
  parametrized ones; `test_t0_is_the_unchanged_training_function` is skipped as before). In the first run one new test failed because it needed
  the silence threshold that the `tiny` fixture sets to 0 (fixed in the test, not in the code).
- Mutation check of the logic by the same stand-in: 72 one-line mutants of `run_tsm.py` and `analyze_tsm.py`, the 27 of 10.3 adapted to the step-1c
  text and 45 new ones (OSError, out-of-memory, interruption, "any other exception" read as numerical or infrastructure, the message and
  gradient-norm cases of the numerical rule, the run state of a failed record with and without a kind, which runs stop the study, the order of the
  reasons, the manifest status of a stop, no kind recorded, a setup or seal error not recorded, a complete evaluation restartable, no flag
  needed, a third restart, moved attempts not counted, no silence rule, `failed` held to the silence rule, the silence rule on the directory
  only, no identity check, each of the four inputs not compared, only the evaluation in place read, nothing moved, the restarts or the seals
  not recorded, a failed end not recorded, the partial output read or deleted, a directory without a manifest or with a cut-off one not an
  evaluation, the attempt not advanced, the three audit checks and the summary field, the restart not named in `summary.md`): all 72 were killed by
  at least one test, none by a crash of the run. Writing the mutants showed one gap in the first classification test (a loose text rule for
  out-of-memory would have passed it); it led to the host out-of-memory case.
- Real subprocess scenarios (real `Popen` children and parents, logs, SIGTERM to the process group, stub torch, fake training and evaluation
  with sleeps; scripts in the session scratchpad): (A) a `ValueError` in a T1 run while two siblings train: the job exits 1 with the code-defect
  reason, the siblings are terminated (no process left), the manifest is `stopped`, two later jobs (another selection; `--rerun-incomplete`) and `evaluate`
  refuse, and no `evaluation/` exists; (B) an `OSError` in a T0 run and a CUDA out-of-memory in a T2 run: the other seven runs complete, the job is
  `incomplete`, and `--rerun-incomplete` trains both as attempt 2 with attempt 1 moved to `attempts/`; (C) an `evaluate` killed by SIGTERM
  (manifest `running`, 5 partial samples): a restart without the flag is refused, with the flag inside the silence period refused ("written to 1 s
  ago"), after it accepted: attempt 2, 24 samples, the 5 partial ones in `attempts/evaluation-attempt1`, and the analysis audits it and reports
  `evaluation_attempt` 2; (D) three evaluate attempts killed in turn: the fourth start is refused ("used 3 of 3 attempts"), the third attempt and the
  two moved directories untouched. The four process scenarios of 10.2 and 10.3 were run again on the final code with the same outcomes (the first one ends
  in an error of its own last line, which reads a manifest that a refused job never writes; the same on 68c19da).
- Not verified: any execution with real torch or real pytest (including the fixtures and `monkeypatch` of the new tests and the real
  `torch.cuda.OutOfMemoryError` and `clip_grad_norm_` errors, only read in the source), a PBS walltime kill or an I/O fault on a compute node,
  the clock and file timestamps across nodes of the shared filesystem (the 300 s rule), and the manual `mv` rules of the sheet beyond the dummy
  tree (the file lists come from reading the IVD, CVH, UNS and NCR scripts, not from running them). The first compute-node run of
  `tests/test_tsm_scripts.py` is part of the step-0 test job of §6.

### 10.6 Step 1d (2026-10-27): Clarification C2 in the code

Task: `docs/tasks/WP-S9_symbolic_followups.md`, section "Step 1d" (master 32e3b2e). Rules: Clarification C2 of `docs/prereg/TSM.md`
(master d488b78, byte-identical on `wp/sym` at a5c093a; not edited, and C1 is as it was). Commits: 382dbc5 (`scripts/run_tsm.py`,
`scripts/analyze_tsm.py`, `tests/test_tsm_scripts.py`) and the commit of this report. Nothing else changed: the IVD, CVH, UNS and NCR
code, the five preregs, `pyproject.toml` and `uv.lock` are as they were.

| rule (Clarification C2) | implementation |
| --- | --- |
| after the orchestrator's recorded decision, `train --max-attempts N` grants further identical attempts; the default is 3 | `train --max-attempts N` (default `MAX_ATTEMPTS` = 3; an N below 3 is refused before anything is read: "--max-attempts must be at least 3"). `stage_train` applies N in `check_rerun` (the attempts used, those moved aside and the run in place, must be fewer than N) and hands it to every child: `schedule` and `start_child` put `train-one ... --max-attempts N` on the child's command line, and the child's own `check_rerun` in `train_one` applies the same N. The manifest records `max_attempts`, and the refusal message names the option. Each attempt is still compared with the one before it (configuration hash and `source_hash`), a rerun still needs `--rerun-incomplete` and an ended job, and a stopped study still refuses every later `train` with or without a larger N |
| allocation failures are infrastructure failures: CUDA out-of-memory, cuBLAS or cuDNN allocation failures, host `MemoryError`; other CUDA runtime errors are code defects | `is_allocation_failure` (replacing `is_cuda_out_of_memory`) is true for `torch.cuda.OutOfMemoryError`, any `MemoryError` (subclasses too, such as numpy's), and a `RuntimeError` whose text says both "cuda" and "out of memory" or names one of the four statuses of `ALLOCATION_STATUSES`: `CUBLAS_STATUS_ALLOC_FAILED`, `CUDNN_STATUS_ALLOC_FAILED` and the cuDNN 9 `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED` and `CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED`. `classify_failure` makes these `infrastructure`, as it does an `OSError` or an interruption; nothing else in the classification changed |
| a defect in the evaluation code: the 18 trained runs are kept, the evaluation restarts from scratch at the fixed commit with `evaluate --restart-interrupted --new-evaluation-code`, the partial output is moved aside unread and both code hashes are recorded | `--new-evaluation-code` exempts the `code` entry (the evaluation job's `source_hash`) from the identical-inputs comparison of `evaluation_attempt`; the design hash, the design lock commit and the checkpoint seals must still equal those of the interrupted attempt, and `final_states` still requires one `source_hash` across the 18 training records. The flag without `--restart-interrupted` is refused. The new manifest records its own `source`, the hash the runs were trained at (`training_source_hash`) and, in `restarts`, `{attempt, moved_to, source_hash}` for every moved attempt (`restart_rows`). `analyze_tsm.py` requires the code of every attempt (`does not record the code`) and a `training_source_hash` equal to the run records, writes "The evaluation code changed at a restart ... source_hash by attempt: 1: ..., 2: ...; the 18 runs were trained at ..." into `summary.md` when the known hashes differ, and adds `evaluation_source_hash` to `summary.json` next to `training_source_hash` |
| a restart reads only the interrupted attempt's `manifest.json` | as in step 1c: the status and the inputs; `restart_rows` reads the same manifest as that comparison (`earlier_evaluation`) and no other file, and the samples and gate profiles are never opened (the tests make the partial samples unreadable) |
| an error before a run directory exists: the operator reports the traceback, the orchestrator classifies it | no code: as in step 1c such an error leaves no record and counts no attempt; the sheet says to read the log and report the traceback |
| a hardware fault that the orchestrator's diagnosis finds behind an error read as a code defect is recorded, and the run is retried as an infrastructure failure | no option in the scripts: the sheet describes the move of the run and its log into `attempts/` that `--rerun-incomplete` makes for an unfinished run; it lifts the stop (the stop is read from the runs in place) and the moved attempt counts as an attempt (a test) |

Interpretations made in step 1d, stated as they are implemented (none changes a registered threshold, statistic or sample):
- `--max-attempts` applies to the training runs. The evaluation keeps three attempts (the first and two restarts): C2 does not extend them. N = 3 is the registered number, so a smaller N is refused rather than accepted as a stricter cap.
- A larger cap moves only the point at which `check_rerun` refuses. It lifts no stop, no silence rule (300 s) and no identity rule.
- An allocation failure is recognised by type for `torch.cuda.OutOfMemoryError` and `MemoryError`, and by the library's status name in the text of a `RuntimeError` for cuBLAS and cuDNN; the text is read in lower case and the exception must be a `RuntimeError`. A `RuntimeError` of PyTorch's CPU allocator (`DefaultCPUAllocator: can't allocate memory`), another status of these libraries (`CUBLAS_STATUS_NOT_INITIALIZED`, `CUBLAS_STATUS_EXECUTION_FAILED`, `CUDNN_STATUS_NOT_SUPPORTED`, `CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED`), an illegal memory access and the allocation failure of any other library are code defects, as the registered text has it.
- The cuDNN names are those of the pinned cuDNN 9. Its header calls `CUDNN_STATUS_ALLOC_FAILED` a deprecated alias of the host status, and the status strings in `libcudnn_graph.so.9` are the host and device `..._ALLOCATION_FAILED` ones, so an allocation failure of that cuDNN names one of them; the legacy name stays for cuDNN 8. The exact text of a real cuBLAS or cuDNN allocation failure was not observed.
- "Both code hashes" are read as the evaluation `source_hash` of the interrupted attempt and that of the restart; the manifest also records the hash the 18 runs were trained at, and the audit compares it with the run records. The evaluation job's hash is compared with the training hash by no script, and the orchestrator's check that the fix changes no file the training path uses is not automated either; the sheet names it.
- `--new-evaluation-code` permits another hash and does not require one: with unchanged code it is accepted, the manifest then records identical hashes and `summary.md` reports no change. It needs `--restart-interrupted`, also after an aborted restart that left no evaluation in place.
- The code of a moved attempt is read from the latest readable manifest, which carries the rows of its own predecessors. An attempt whose manifest was cut off or never written has `source_hash: null`: the audit accepts it and `summary.md` shows "unknown".

### 10.7 Verification of step 1d (all static or torch-free; nothing was run with torch, pytest or on a compute node)

- `python3 -m pyflakes` (2.4.0, Python 3.9.25) on the three changed Python files: clean. `py_compile` with the bytecode written outside the worktree
  under Python 3.9.25 and 3.13.13 (`-I -B`): ok; no `__pycache__` in the worktree. The `ast` checks of step 1 on the final tree: 786 module-attribute
  references, none unresolved; 1133 calls with 6 hits, the 4 re-exports of §7 and two uses of `run_tsm.digest_checkpoint` in tests (re-exported by
  `from s9_common import digest_checkpoint`, `run_tsm.py:55`; it resolves at run time). No added line is longer than 120 columns.
- 127 test functions in the ten files (118 after step 1c); `test_tsm_scripts.py` went from 30 to 39: nine new functions (the cap on the child's command
  line through the real `start_child` with `Popen` replaced, the child parsing it; cuBLAS, cuDNN and host allocation failures rerun, even for T0;
  `train_one` with a larger cap; four allocation failures and the cap granted at 4 and 5, with the refusals, the manifests and the identity rule;
  a run moved aside by hand after a recorded hardware fault; a fixed evaluation code restarted only with the flag, both hashes recorded and audited;
  the flag relaxing nothing else; the code of every attempt across restarts, two cases; an attempt whose manifest was cut off). Existing tests changed:
  the classification now has 27 cases (16 before; `MemoryError()` moved from code defect to infrastructure; every allocation status, a
  `torch.cuda.OutOfMemoryError` whose text does not say CUDA, the other CUDA statuses, a `ValueError` with CUDA text), the `check_rerun` test with a larger cap, the schedule test (the cap reaches `start_child`), the code-defect
  stop test (a larger cap does not lift it), and every assertion on `restarts` and on the summary (rows carry `source_hash`).
- The repository's own test bodies of `tests/test_tsm_scripts.py` through the stand-in for pytest's fixture machinery of 10.3 (venv python, stub
  `torch` package, file and numpy stand-ins for the model, `train`, `train_arm`, `evaluate_arm` and the loaders; `monkeypatch.setenv` added to the
  stand-in): 77 of 77 cases passed (38 functions; `test_t0_is_the_unchanged_training_function` is skipped as before).
- Mutation check of the logic by the same stand-in: 122 one-line mutants of `scripts/run_tsm.py` and `scripts/analyze_tsm.py`, each applied to a copy
  of the tree (the repository is not touched) and run against the whole test file (4 workers). 50 were written for step 1d, 72 are those of steps 1b and 1c
  run against the step-1d code (13 of them re-written for lines that changed). The step-1d ones change: the cap not handed to the children (stage, command
  line, `schedule`, `train_one`, `check_rerun`), a cap below the registered number accepted, no cap in the manifest, a wrong default, a count off by one;
  each part of the allocation rule dropped or widened (cuBLAS, the cuDNN names, host `MemoryError`, the class `torch.cuda.OutOfMemoryError`, any CUDA
  error, the CPU allocator, any exception class); the flag accepted without a restart, not handed on, on by default, or relaxing the lock or the
  checkpoints; rows without `source_hash`, the chain of codes not carried, the code of the wrong attempt; the audit, `summary.md` and `summary.json`
  fields; the training hash. Result of the full run on the final tests: 122 of 122 killed, each by at least one failing test and none by a crash of
  the run. The first full run, before the last test case was added, killed 121 of the 122: the survivor removed the class check of
  `torch.cuda.OutOfMemoryError`, which the text rule shadows for every text that torch writes; a case with an out-of-memory error whose text says HIP
  (known by its class alone) was added to the classification, and the full run was repeated on the final tests. One more pattern, the flag relaxing the
  design hash as well, was dropped as equivalent: the design is checked against its lock commit, and the lock commit is compared.
- Real subprocess scenarios (real `Popen` children and parents, logs, SIGTERM to the process group, stub torch, fake training and evaluation with
  sleeps; scripts in the session scratchpad; the harness now runs the real `start_child`, replacing only the script in its command): (A) one run
  fails four times with a cuBLAS allocation failure, a `MemoryError`, a CUDA out-of-memory and a cuDNN allocation failure: without
  `--max-attempts` the fourth attempt is refused ("used 3 of 3 attempts"), `--max-attempts 2` is refused, `--max-attempts 4` starts attempt 4 (the
  child's log shows `train-one ... --max-attempts 4`), a second job at 4 is refused ("used 4 of 4"), and `--max-attempts 5` completes attempt 5; the
  manifests record 3, 3, 3, 4 and 5; (B) an illegal memory access is a code defect that stops the study, and `--rerun-incomplete --max-attempts 9`
  is refused ("the study stops"); (C) an `evaluate` killed by SIGTERM at one code is restarted at another `source_hash`: refused without
  `--new-evaluation-code`, refused with it but without `--restart-interrupted`, and with both flags it completes as attempt 2 with the partial
  output in `attempts/evaluation-attempt1`, its own `source_hash` and the interrupted attempt's in `restarts`; the analysis audits it, `summary.json` has
  both hashes and `summary.md` the sentence naming the change. The seven process scenarios of 10.2 to 10.5 were run again on the final code: the outputs
  are identical to the earlier ones except for the new `source_hash` field of the restart rows and for times, process ids and the seals that contain
  them (scenario 1 still ends in the error of its own last line, as before).
- Read statically in the installed packages, not run: `torch.cuda.OutOfMemoryError` is `torch._C.OutOfMemoryError`, declared `class
  OutOfMemoryError(RuntimeError)` (torch 2.14.0); the string `CUBLAS_STATUS_ALLOC_FAILED` is in `libtorch_cuda.so`; the installed cuDNN 9 header
  `cudnn_graph.h` defines `CUDNN_STATUS_ALLOC_FAILED` as a deprecated alias of `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED` and the two
  `..._ALLOCATION_FAILED` strings are in `libcudnn_graph.so.9`, while `CUDNN_STATUS_ALLOC_FAILED` is in none of the cuDNN libraries. The first version
  of the rule matched only `CUDNN_STATUS_ALLOC_FAILED`; reading these files showed that a cuDNN 9 allocation failure would not carry that name, and the
  rule was corrected before the commit.
- Not verified: any execution with real torch or real pytest, a real cuBLAS or cuDNN or CUDA allocation failure and the text of its error, a PBS
  walltime kill or an I/O fault on a compute node, the clock and file timestamps across nodes of the shared filesystem (the 300 s rule), and the
  `--max-attempts` and `--new-evaluation-code` jobs on a compute node. The first compute-node run of `tests/test_tsm_scripts.py` is part of the
  step-0 test job of §6.

### 10.8 Step 1e (2026-10-27): the static review of the failure rules and Clarification C3 in the code

Task: `docs/tasks/WP-S9_symbolic_followups.md`, section "Step 1e" (master 45060c7). It covers the independent static review
`docs/review/static_2026-10-27/tsm.md` (findings F1 to F8; no break of a registered quantity or decision rule) and Clarification C3 of
`docs/prereg/TSM.md` (master d335fe7, byte-identical on `wp/sym` at 3ca4d89; not edited, and C1 and C2 are as they were). Commits: fab4006
(`scripts/run_tsm.py`, `scripts/analyze_tsm.py`, `tests/test_tsm_scripts.py`) and the commit of this report. Nothing else changed: the IVD, CVH, UNS
and NCR code, the five preregs, `pyproject.toml` and `uv.lock` are as they were.

| finding (rule) | implementation |
| --- | --- |
| F1, Medium: the first evaluation attempt runs at the training code (C1: every TSM training and evaluation job runs at one `CODE` commit; C2: another evaluation code only under the `--new-evaluation-code` rule) | `evaluation_attempt`, before anything is created or generated. `--new-evaluation-code` needs an earlier attempt (an evaluation in place or moved aside) and, as before, `--restart-interrupted`: the first attempt refuses it. An attempt that has no readable earlier manifest to compare with must run at `training_source_hash` ("the evaluation runs at source_hash ... but the 18 runs were trained at ..."); a restart whose earlier attempt has a readable manifest is compared with it as before, `--new-evaluation-code` exempting only the code. `analyze_tsm.py` always prints "Code (source_hash): the 18 runs were trained at ...; the evaluation ran at ... (the same code, or a DIFFERENT code)". If the first attempt of a record ran at another code (a manifest that did not pass this check), a sentence says that this is outside the rule, to be reported before the result is used, with the code of each attempt; a restart at another code is worded as in step 1d |
| F2, Low: `train` compares the `source_hash` across the runs of a root | `require_one_code(root, source)` in `stage_train` (after the stop check, before the pause check, the run states, any log and the manifest: a refused job writes neither a manifest nor a log) and in `train_one` (before `check_rerun` and `mkdir`: the child creates no run directory). It compares the job's `source_hash` with that of every readable `runs/*/record.json` and `attempts/*/record.json` (`recorded_source_hashes`; a cut-off record, or one without a source, is no evidence). The message says that a retraining at another code starts in a new `--root` |
| F3, Low-Medium (C3: I/O errors include torch's checkpoint-serialization errors) | `is_serialization_failure`: a `RuntimeError` whose lower-case text contains "pytorchstreamwriter failed" or "pytorchstreamreader failed". `classify_failure` makes it `infrastructure`, as it does an `OSError` |
| F4, Low (C3: allocation failures include torch's CPU-allocator failure and `CUBLAS_STATUS_NOT_INITIALIZED` reported by `cublasCreate`) | `is_allocation_failure` also accepts a `RuntimeError` that contains "defaultcpuallocator: not enough memory" (lower case), or both "cublas_status_not_initialized" and "cublascreate". The same status from another call (`cublasSetStream`, or none named) and another status of `cublasCreate` stay code defects. The bounded exposure of C3 is not coded: a code defect that was misread as one of these repeats identically and reaches the pause (F6) |
| F5, Low (C1: running jobs may finish, and their outputs are kept) | `schedule` with `fatal` (the training stage): the first stopping failure (a T0 numerical failure, or a code defect) clears the pending runs, and the loop goes on until the running children have ended. Nothing is terminated and the records of the finishing runs are kept. `stage_train` then reads the records on disk: `study_stop` names the first run that stopped the study (a code defect before a T0 failure, as before), and the `stopped` manifest carries `run_states` (what each selected run ended as, `missing` for a run that was never started). Without `fatal` (the smoke stage) the first non-zero exit still stops everything at once, and a child that cannot be started still does |
| F6, Low (C1: three failed attempts pause the study; C2: `--max-attempts N` after the recorded decision) | `paused_runs(root, N)`: a run of the 18 that is not final (`missing` or `incomplete`), has used N attempts (`attempts_used`: those moved aside and the one in place) and whose last attempt has ended (`has_ended`: its record says `failed`, or nothing was written for 300 s; always the case for a run with no run directory) pauses the study. `stage_train` then refuses every job, whatever it selects, before it writes anything ("... the study is paused ..."). `--max-attempts N` above 3 is refused without `--decision FILE` (an absolute path of an existing, non-empty file), and the manifest records `decision` = `{path, blake2b}` (blake2b-128 of the file; `None` without `--decision`). Nothing reads the content of the file |
| F7, Low | `heartbeat(path)` in `stage_evaluate`: a daemon thread rewrites `evaluation/heartbeat` (the current time) every `HEARTBEAT_SECONDS` = 30 s, with a first beat before the evaluation starts, until the evaluation completes or fails; a beat that cannot be written (`OSError`) is skipped. The silence rule counts it (any file written in `evaluation/` does). A test pins `QUIET_SECONDS == 300` and `0 < HEARTBEAT_SECONDS <= 60` |
| F8, Low (C3: the check of an evaluation fix is made at code level and recorded with the decision) | the sheet only (§6, block 4): the check works on the functions, classes and module-level statements that the training path executes, not on files; it stays the orchestrator's and is not automated; the example is a diff of the training `CODE` and the fixed commit, read hunk by hunk |
| the `--rerun-incomplete` help | names walltime or a node fault, an I/O or serialization error and an allocation failure, and says "at most twice per run, or `--max-attempts` attempts in all". The `--max-attempts` help names `--decision`, and the `--new-evaluation-code` help says that the first attempt refuses it |

Interpretations made in step 1e, stated as they are implemented (none changes a registered threshold, statistic or sample):
- "The training code" is the `source_hash` that the 18 run records share (`training_source_hash`). `--commit`, `git_commit` and `source_dirty` are compared by no script, as before.
- "No earlier attempt" is no evaluation in place and none moved aside. An earlier attempt whose manifest is not readable (a job killed before it wrote one) still counts as an attempt, and a restart then runs at the training code unless `--new-evaluation-code` says otherwise: there is no earlier manifest to compare with, and the flag is the orchestrator's decision (C2). The code of such an attempt is `null` in `restarts` and "unknown" in `summary.md`, as in step 1d.
- The pause is read at the job's own `--max-attempts`. A job at N = 4 passes a run that has used 3 attempts and a job at the default refuses it: until the paused run is final, every later `train` job passes the same `--max-attempts N --decision FILE` (the sheet says so), and a run that fails again at its N-th attempt pauses the study again.
- A run whose last attempt may still be alive (marked "running", written to in the last 5 minutes) is not paused: it has not failed yet. A run with attempts moved aside and no run directory (a rerun whose child died at once) is paused once its attempts are used.
- `--max-attempts 3` with `--decision FILE` is accepted and the decision is recorded; the registered number needs none. The file must exist and not be empty. Whether it is the orchestrator's decision is not checked; the digest ties the manifest to the file.
- A stopping failure stops the job the way C1 lets a study stop: nothing new starts, the running children finish and none is terminated. The job lives until its running children have ended or its walltime does (a child that the walltime ends first is an unfinished run like any other). A child that fails after the first stopping failure is recorded as well, and `study_stop` names the first code defect in the plan's order, else the first T0 failure. Failures that do not stop the study (a numerical failure of a T1 or T2 run, an infrastructure failure) are recorded as before and stop nothing.
- The three C3 texts are read in a `RuntimeError`, as are the C2 library statuses (the libraries raise a `RuntimeError`); the same text in a `ValueError` is a code defect.
- The heartbeat shows that the evaluation process is alive, not that it progresses: an evaluation that hangs while its interpreter still runs goes on beating until its walltime ends. The interval (30 s) is half the required 60 s. The silent gap of a live evaluation was not measured (no compute node); the heartbeat makes it irrelevant to the silence rule.
- Statements of 10.1 to 10.7 that step 1e supersedes: the CPU allocator "can't allocate memory" and `CUBLAS_STATUS_NOT_INITIALIZED` as code defects (torch 2.14 says "not enough memory", and under C3 both are infrastructure failures; `CUBLAS_STATUS_NOT_INITIALIZED` reported by any call other than `cublasCreate` stays a code defect), "the evaluation job's hash is compared with the training hash by no script" (F1), the running siblings of a stopping failure being terminated (they finish), and the 300 s rule being untested (F7).

### 10.9 Verification of step 1e (all static or torch-free; nothing was run with torch, pytest or on a compute node)

- `python3 -m pyflakes` (2.4.0, Python 3.9.25) on the three changed Python files: clean. `py_compile` with the bytecode written outside the worktree
  under Python 3.9.25 and 3.13.13 (`-I -B`): ok; no `__pycache__` in the worktree. The `ast` checks of step 1 on the final tree: 846 module-attribute
  references, none unresolved; 1194 calls with 6 hits, the 4 re-exports of §7 and two uses of `run_tsm.digest_checkpoint` in tests (re-exported by
  `from s9_common import ... digest_checkpoint ...`, `run_tsm.py:64`; it resolves at run time). No added line of code or tests is longer than 120 columns.
- 139 test functions in the ten files (127 after step 1d); `test_tsm_scripts.py` went from 39 to 51: twelve new functions. F5: a stopping failure through the
  real `schedule` with stub children (the default path stops everything at once; with the predicate no pending run starts, the running children end, the error
  names the first stopping failure; a failure the predicate does not name stops nothing). F2: `train_one` refuses another code and creates nothing (records
  in place and moved aside count, a cut-off or source-less record does not), and a training job at another code is refused by the real `main` before it
  starts, logs or writes anything (two selections). F6: a larger cap needs a decision that exists (absent, empty, a directory and a relative path are
  refused with nothing created; the registered number with a decision records it; a manifest without one says `None`), the pause rule on a bare directory
  (a registered training failure is final, an unfinished run pauses, a missing run with three moved attempts pauses), and the pause across real jobs
  (three killed attempts; the last attempt still alive pauses nothing; after it ended five different selections are refused with nothing written; a decision at 4
  lets another job go on and then the paused run's fourth attempt completes). F1: the summary states both codes (8 cases of attempt codes, the unknown
  one included), the first evaluation attempt runs at the training code (refused at another code with or without `--restart-interrupted`, the flag refused
  with no earlier attempt, nothing created; a manifest edited to say another code is audited and flagged), and a restart with no readable earlier manifest runs
  at the training code unless the flag says otherwise. F7: `QUIET_SECONDS == 300` and `0 < HEARTBEAT_SECONDS <= 60`, a live evaluation writes a heartbeat
  while its work writes nothing else (several different beats, one thread, the first beat before the work, nothing after the end), and the heartbeat stops
  at a failure, skips a beat that cannot be written and lets an error of its block propagate. Existing tests changed: the classification has 36 cases (27
  before): the real torch 2.14 texts of the three C3 failures, their near misses (`CUBLAS_STATUS_NOT_INITIALIZED` from `cublasSetStream` or with no call named,
  another status of `cublasCreate`, another text of the CPU allocator, the C3 texts in a `ValueError`, a stream error with neither marker); the old
  `DefaultCPUAllocator: can't allocate memory` text is gone. The end-to-end allocation test (renamed) has six outcomes (adds a serialization error, the CPU
  allocator and the `cublasCreate` initialisation), the fixture's children have `serialize`, `cpualloc`, `cublasinit` and `slow` (a child still running for a
  few polls, replacing `hang`), the two study-stop tests expect the sibling to finish with its record kept and `run_states` in the manifest, the larger-cap
  tests pass a decision file and check its recorded path and digest, the rerun test has a configuration-mismatch leg, the restart-waits test has a heartbeat leg,
  and the pipeline and restart tests check the code sentence of `summary.md`.
- The repository's own test bodies of `tests/test_tsm_scripts.py` through the stand-in for pytest's fixture machinery of 10.3 (venv python, stub
  `torch` package, file and numpy stand-ins for the model, `train`, `train_arm`, `evaluate_arm` and the loaders): 105 of 105 cases passed (50 functions;
  `test_t0_is_the_unchanged_training_function` is skipped as before). The heartbeat tests run on the real clock.
- Mutation check of the logic by the same stand-in: 190 one-line mutants of `scripts/run_tsm.py` and `scripts/analyze_tsm.py`, each applied to a copy of the
  tree (the repository is not touched) and run against the whole test file (4 workers). 123 are those of steps 1b to 1d (16 of them re-written for code
  that step 1e changed), 67 were written for step 1e: F1 13 (the first attempt allowed at any code, no way round for a restart with no manifest, the flag
  accepted with no earlier attempt or refused for a first restart, the two codes compared with themselves or swapped, an earlier manifest not compared, and
  six changes of the code line and the sentences of `summary.md`), F2 7 (no check in the stage or in the child, only the records of `runs/` or of
  `attempts/`, any record or one without a code counted, only the first code compared), F3 and F4 10 (the CPU allocator, the cuBLAS status and the
  `cublasCreate` condition dropped or widened, the writer or the reader marker alone, no `RuntimeError` requirement, serialization read as a code defect), F5 8
  (starts not stopped, running children terminated, the stop not reported or the last one reported, the default path draining too, the predicate inverted,
  no `run_states`, a stopping failure that drains without clearing the pending runs), F6 17 (no decision required, the checks of the file dropped one by one,
  the digest, the manifest, the pause dropped, widened or narrowed, the floor of the cap, the attempts counted), F7 12 (the interval above 60 s, the silence rule
  changed, the file elsewhere, no first beat, the thread not stopped or not joined, a failed beat ending the evaluation, no heartbeat, the silence rule not reading
  it, an interval as long as the silence rule, a single beat, an unnamed thread). Result of the full
  run on the final tests: 188 of 190 killed, each by at least one failing test and none by a crash of the run. The two survivors are not gaps in the tests. D12
  (the loop of `stage_train` that compares an attempt with the one before it allowing one attempt more than the cap) is equivalent: a run for which that cap matters
  has been refused earlier, by the pause check (its last attempt has ended, or it has no run directory) or by `require_ended` (it may be alive), and the child's own
  `check_rerun` applies the true cap. D20 (a step-1d mutant that adds the phrase "allocate memory" to the library statuses, so that the invented CPU-allocator text of
  step 1d counted) is obsolete: the case that killed it pinned that invented text, which F4 replaced (no library of torch 2.14 contains it), and I01 and I07 check
  the registered text. The first full run, on the tests as they were before three changes, killed 187: the survivors were M01 (the identity comparison of an
  unfinished run's configuration, shadowed for the code by `require_one_code`; a configuration-mismatch leg was added to the rerun test), L10 (a heartbeat interval as
  long as the silence rule; the heartbeat test now sets the silence rule to an hour, so the interval is tested on its own) and D12; the three were run again after
  the two test changes (M01 and L10 killed), and the full run was repeated after the third change, the replacement of the invented text, which is the run
  reported here.
- Real subprocess scenarios (real `Popen` children and parents, logs, SIGTERM to the process group, stub torch, fake training and evaluation with sleeps;
  scripts in the session scratchpad; the harness runs the real `start_child`, replacing only the script in its command). The seven scenarios of 10.2 to
  10.5 and scenario 8 of 10.7 were run again on the final code: the outputs of 1, 4, 6 and 7 are identical to the earlier ones except for times, process ids
  and the seals that contain them (scenario 1 still ends in the error of its own last line, as before); in 2 the refusal of the changed code now comes from the
  stage ("the code differs from the earlier TSM records"); in 3 and 5 the siblings of a stopping failure now end `complete` where they were left `running`
  after the termination; in 8 the fourth attempt is refused as the pause ("used 3 of 3 attempts and is not final: the study is paused") and the larger caps
  pass a decision file. New: (9, F5) a non-finite loss in a T0 run while two T0 siblings train (6 s each) and three T1 runs are pending: the siblings end
  `complete` with their checkpoints, the pending runs have no log and are `missing` in the `stopped` manifest's `run_states`, no process is left, a later job
  of any selection and `evaluate` refuse, and with a T0 failure and a code defect in one batch the manifest names the code defect; (10, F7) an evaluation
  that takes 5 s per checkpoint under a silence rule of 2 s: with the heartbeat (every 0.5 s) the restart is refused while the first evaluation is alive
  (the newest file is 0.1 s old, the newest sample 3 s old) and accepted after it was killed and 2.5 s had passed, with the heartbeat at an interval of 3600 s
  (the control) the same restart is accepted under the live job; (11, F6, F2) after three failed attempts of one run (cuBLAS, host memory, CUDA out-of-memory) a job for
  another run is refused with nothing started or written, and so is `--max-attempts 4` without a decision, with a relative path and with an empty file;
  `--max-attempts 4 --decision FILE` lets that other run train, the manifest's `decision` equals the path and an independently computed blake2b-128 of the
  file, and the paused run completes as attempt 4; a job at another code (`LL_SOURCE_ROOT` at a second snapshot) is refused with nothing created, and so is the `train-one` child run by hand
  (no run directory), while at the first code it runs; (12, F1) at another code the first evaluation attempt is refused with and without
  `--restart-interrupted`, `--new-evaluation-code` is refused without an earlier attempt and without `--restart-interrupted`, nothing is created, at the
  training code it completes, and the analysis prints "the evaluation ran at ... (the same code)".
- Read statically in the installed torch 2.14.0+cu132, not run: `libtorch_cpu.so` contains "PytorchStreamWriter failed ", "PytorchStreamReader failed " and
  "file write failed"; `libc10.so` contains "DefaultCPUAllocator: not enough memory: you tried to allocate " next to `alloc_cpu.cpp` and the enforce
  expression `data`; none of the torch libraries contains "can't allocate memory"; `libtorch_cuda.so` contains `CUBLAS_STATUS_NOT_INITIALIZED` and " when calling `cublasCreate(handle)`". The test texts are assembled from these
  strings; the enforce prefix and its line number (`alloc_cpu.cpp:121`), the file name of the stream error and the byte count are placeholders.
- Not verified: any execution with real torch or real pytest; the text of a real serialization failure (a quota or full file system during `torch.save`), of a real
  CPU-allocator failure and of a real `cublasCreate` failure (the strings of the libraries were read, no failing call was made); the heartbeat and the
  silence rule on a compute node, across nodes of the shared file system, and the silent gap of a live evaluation; a PBS walltime kill; the `--max-attempts`
  with `--decision`, pause and `--new-evaluation-code` jobs on a compute node. The first compute-node run of `tests/test_tsm_scripts.py` is part of the step-0 test
  job of §6.

### 10.10 Step 1f (2026-10-27): the second static review and Clarification C4 in the code

Task: `docs/tasks/WP-S9_symbolic_followups.md`, section "Step 1f" (master d518596). It covers the second independent static review
`docs/review/static_2026-10-27/tsm2.md` (finding 1, finding 2 and the notes a, b and f; it found no defect of consequence in the process management F5, the pause F6,
the one-code rules F1 and F2 or the heartbeat F7, and no change of a registered quantity) and Clarification C4 of `docs/prereg/TSM.md` (master d6ceb6a,
byte-identical on `wp/sym` at 76a00b3; not edited, and C1 to C3 are as they were). Commits: 6404f8d (`scripts/run_tsm.py`, `scripts/analyze_tsm.py`,
`tests/test_tsm_scripts.py`) and the commit of this report. Nothing else changed: the IVD, CVH, UNS and NCR code, the five preregs, `pyproject.toml`
and `uv.lock` are as they were.

| finding (rule) | implementation |
| --- | --- |
| 1, Medium-Low (C4: C3's I/O rule covers every error that torch's checkpoint writer or reader raises because a file cannot be created, opened, written or read) | `OPEN_FAILURE`, a regular expression over the lower-case message: `\bfile .* cannot be opened\.` or `\bparent directory .* does not exist\.`. `is_serialization_failure` accepts a `RuntimeError` that contains a C3 marker ("pytorchstreamwriter failed", "pytorchstreamreader failed") or matches it, wherever the text stands in the message. `classify_failure` makes it `infrastructure`, as it does an `OSError`. The same text in another exception (a `ValueError`) is a `code_defect`, and so are "Directory ... does not exist." (no "Parent") and "Archive ... cannot be opened." (no "File"). C3's bounded exposure is not coded, as for F3 and F4: a code defect that was misread as an I/O error repeats identically and reaches the pause |
| 2, Low (C4: a defect before the first evaluation manifest is an evaluation-code defect of C2; the orchestrator records its decision, with C3's code-level check; the first attempt then runs at the fixed commit; both code hashes recorded) | `evaluate --decision FILE` (an absolute path of an existing, non-empty file, validated by the same `decision_record` as `train --decision`, before the design is read). In `stage_evaluate` a first attempt is one with no evaluation in place and none moved aside (`first`); there `--new-evaluation-code` is accepted without `--restart-interrupted`, and at any later attempt it still needs `--restart-interrupted`. `evaluation_attempt`, at a first attempt: the evaluation `source_hash` must equal `training_source_hash`, or `--new-evaluation-code` must come with `--decision`; neither alone is enough, and without `--decision` the refusal of step 1e stays (its message now names `--new-evaluation-code --decision FILE`). A restart is as before: compared with the latest readable earlier manifest, another code needs `--new-evaluation-code`, no decision file is needed. The manifest records `decision` (`{path, blake2b}` or `null`) next to `source` and `training_source_hash`; each row of `restarts` carries the `decision` of its attempt (taken from the earlier manifest's own `decision` and from its `restarts[].decision`; `null` for none or not known), so that a restart at the same code keeps the first attempt's decision on record. `analyze_tsm.py`: the audit wants `decision` in the manifest and in every row, as a path and a digest or `null`; for a first attempt at another code `summary.md` says "on the orchestrator's recorded decision (docs/prereg/TSM.md, Clarification C4; decision file ..., blake2b ...)" when its decision is recorded, and "no decision of the orchestrator is recorded for it: this is outside the rule" when not; `summary.json` has `evaluation_decision` |
| note a: the stopped `train` manifest records the stopping child | `schedule` raises `ChildFailed` (a `ValueError`) carrying the run, the exit code and the log path of the first child whose exit stopped the job; a child that could not be started raises a plain `ValueError` (no child ended). `stage_train` writes `stopping_child = {run, exit_code, log}` into the `stopped` (or `failed`) manifest, `null` when no child ended. `reason` is still the `study_stop` text |
| note b: a train manifest left at "running" | the sheet (§6, block 4): the job was killed before it wrote its end; read the run states (each `runs/<run>/record.json`), which every later `train` and `evaluate` also read to enforce a stop |
| note f: the heartbeat thread's `join` | `thread.join(2 * HEARTBEAT_SECONDS)`; the thread is a daemon and is left behind if its beat hangs inside the write |
| help texts and docstrings | `--rerun-incomplete` names a checkpoint file that cannot be created or opened; `--new-evaluation-code` states both uses; `evaluate --decision` is new; the docstrings of the module, `classify_failure`, `is_serialization_failure`, `schedule`, `stage_train`, `decision_record`, `evaluation_attempt`, `restart_rows`, `heartbeat` and `stage_evaluate` follow |

Interpretations made in step 1f, stated as they are implemented (none changes a registered threshold, statistic or sample):
- The two texts were read, not observed. `libtorch_cpu.so` of the installed torch 2.14.0+cu132 contains "Parent directory ", " does not exist.", "File " and
  " cannot be opened." (with their full stops) among the strings of `inline_container.cc`, next to "opening archive " and "initializing archive ", and
  `torch/serialization.py` hands an ASCII path to `torch._C.PyTorchFileWriter(path, ...)` (a non-ASCII one is opened by Python's `open`), so `torch.save` reaches the library's writer. Whether the raised
  message carries a prefix or a C++ trace was not observed: the expression finds the text anywhere in the lower-case message, and the texts are matched with
  their closing full stops, as C4 quotes them. No failing call was made.
- C4's first sentence is general ("every error ... because a file cannot be created, opened, written or read"; the quoted texts are examples); the code lists
  the texts it knows: the two C3 markers and the two C4 texts. The reader has two more by-name open texts in the library ("open file failed because of errno ... on
  fopen: ..., file path: ...", "open file failed with strerror: "); they are not named by C4 and not recognized. As read in the installed
  `torch/serialization.py`, `torch.load(path)` opens the file with Python's `open` first (`_open_file_like`), so a missing or unreadable file at load is an
  `OSError`, which is infrastructure already. A text of the same kind that the code does not know is a code defect (C1's default) and stops the study; the
  orchestrator decides such a case when it occurs.
- A first attempt is what C4 and the task say: neither `evaluation/` nor `attempts/evaluation-attempt*` exists. An `evaluation/` that is there (also one with no
  readable manifest, cut off before the manifest was written) is an attempt, and so is one moved aside: a decision does not take the place of
  `--restart-interrupted` there. At a first attempt `--new-evaluation-code` is accepted at the training code too (it changes nothing there), `--decision` alone is
  accepted and recorded, and `--restart-interrupted` with nothing to restart is accepted as it was.
- A restart needs no decision (the C2 contract of step 1d is unchanged); a decision that is given at a restart is recorded for that attempt. The decision under
  which the first attempt ran is carried in `restarts[0].decision` of every later manifest, and the audit and the summary read the first attempt's own row, not the
  final attempt's: a restart at the code of the interrupted first attempt needs neither flag nor decision (identical inputs).
- The decision file is checked for existence and non-emptiness only: the scripts do not read what it says, whether it is the orchestrator's, or whether the
  code-level check behind it was made (as for `train --decision`). The digest ties the manifest to the file as it was at the start; the analysis does not hash the
  file again. A defect before the first manifest leaves no record of its own: the operator reports it (the sheet).
- `stopping_child` is the first child whose non-zero exit made the predicate true. `reason` names the first run in the plan's order by the study-stop rule (a
  code defect before a T0 failure). When two stopping failures end in one drain they can name different runs (`reason` by the plan's order, `stopping_child` by the order of the exits); both are recorded.
- The heartbeat thread is a daemon: a beat that hangs inside its write is left behind when the block ends, at most `2 * HEARTBEAT_SECONDS` = 60 s after the
  evaluation's work has ended (the manifest says "complete" or "failed" before that, inside the block). The existing heartbeat tests now wait for the end of the
  thread (`_heartbeat_gone`) instead of assuming that it has ended when `join` returns: with an interval of 0.01 s the timeout is 0.02 s, which a loaded machine
  can exceed. The new test of a hung beat pins both bounds of the wait (at least 90 % of 2 x `HEARTBEAT_SECONDS`, and well under the time the beat is held).
- Statements of 10.1 to 10.9 that step 1f supersedes: the first attempt refuses `--new-evaluation-code` (10.8, F1, and the old help text), the rows of `restarts`
  having three fields (they have four: `decision` is new), the F3 row naming the two serialization markers only (C4 adds the two open texts), and the
  "outside the rule" sentence of `summary.md` for any first attempt at another code (it now applies when no decision is recorded).

### 10.11 Verification of step 1f (all static or torch-free; nothing was run with torch, pytest or on a compute node)

- `python3 -m pyflakes` (2.4.0, Python 3.9.25) on the three changed Python files: clean. `py_compile` with the bytecode written outside the worktree
  under Python 3.9.25 and 3.13.13 (`-I -B`): ok; no `__pycache__` in the worktree. The `ast` checks of step 1 on the final tree: 871 module-attribute
  references, none unresolved; 1212 calls with the same 6 hits as in 10.9 (the 4 re-exports of §7 and two uses of `run_tsm.digest_checkpoint` in tests, which
  resolve at run time). No added line of code or tests is longer than 120 columns.
- 143 test functions in the ten files (139 in 10.9); `test_tsm_scripts.py` went from 51 to 55: four new functions, and the existing ones that the new rules touch
  were changed. Finding 1: the classification table has 43 cases (36 before): the two texts of the writer in a `RuntimeError` are infrastructure failures (one of
  them with an enforce prefix and a trailing C++ trace line, as the library may word them), the same two texts in a `ValueError` are code defects, and so are two
  near misses ("Directory ... does not exist." without "Parent", "Archive ... cannot be opened." without "File"). The end-to-end test (renamed
  `test_library_allocation_serialization_and_open_failures_are_infrastructure_failures_too`) has eight failing runs, two more than before (the open failure and
  the missing parent directory): each is recorded as infrastructure, the job ends `incomplete`, and `--rerun-incomplete` trains them again as attempt 2.
  Finding 2: a first attempt at another code is refused without `--new-evaluation-code` and `--decision FILE` together (six combinations of the flags,
  `--restart-interrupted` included, each refused with a message that names both, nothing created), a bad decision (absent, empty, a directory, a relative path) is refused, and with both the
  evaluation completes and its manifest records the decision's path and blake2b digest (computed independently in the test) and both codes; the summary then names the
  decision, and a manifest edited to have none is printed as outside the rule; a malformed or missing decision (in the manifest or in a row of `restarts`) is refused by the
  audit; a three-attempt chain (the first attempt under a decision is interrupted, the flags are refused while its directory is in place, the restart needs
  none and keeps the decision on record in `restarts`, a second interruption and a completing third attempt) and seven combinations of codes and decisions through
  `render` (decided, restarted at the same code or at the training code, a decision of a later attempt, a restart at another code with or without a decision, a decision at
  the training code). Existing tests changed: the first-attempt refusals (no flags, the restart flag, both flags without a decision, then the run at the training
  code), the two tests of the new-code flag (the flag without `--restart-interrupted` is still refused with an evaluation in place or moved aside, also with a decision;
  a restart needs no decision), and every assertion of `restarts` rows (they carry `decision`). Note a: `schedule` raises `ChildFailed` with the run, the exit code
  and the log (the cannot-be-started error is not one), and the two study-stop tests and the cannot-be-started test check `stopping_child` in the manifest. Note f:
  the hung-beat test patches the beat's clock call to block the heartbeat thread, and requires the end of the block to wait for at least 90 % of 2 x `HEARTBEAT_SECONDS`
  and to come well before the beat is released; the three heartbeat tests that assumed the thread gone at the end of the block now wait for it (`_heartbeat_gone`).
- The repository's own test bodies of `tests/test_tsm_scripts.py` through the stand-in for pytest's fixture machinery of 10.3 (venv python, stub
  `torch` package, file and numpy stand-ins for the model, `train`, `train_arm`, `evaluate_arm` and the loaders): 122 of 122 cases passed on the final tree (55
  functions; 105 cases in 10.9; `test_t0_is_the_unchanged_training_function` is skipped as before). The stand-in needed one change: its `pytest.raises` now keeps
  `.value`, which the new tests read. The heartbeat tests run on the real clock.
- Mutation check of the logic by the same stand-in: 234 one-line mutants of `scripts/run_tsm.py` and `scripts/analyze_tsm.py`, each applied to a copy of the
  tree (the repository is not touched) and run against the whole test file (4 workers). 188 are those of steps 1b to 1e (19 of them re-written for code that step 1f changed; G03 and G04 dropped, the lines they
  mutated being gone), one (G14) is the restart message with its two hashes swapped, and 45 were written for step 1f: finding 1 8 (the open texts not
  recognized, either of them alone, any directory, any object that cannot be opened, the match anchored at the start of the message, no lower-casing, no
  closing full stop), note a 8 (the exit code or the log path not carried or wrong, the child not recorded or another run, a start failure counted as a
  child), finding 2 15 (a first attempt at another code with no decision or with a decision alone, another code never allowed, the first attempt read as a
  restart, a restart that needs a decision, a moved-aside or an in-place evaluation not counted as an attempt, the flag taken as a first attempt's whenever it
  is given, the decision not recorded, no digest, not validated, the decision of the interrupted attempt or of the earlier restarts not carried into
  `restarts`), the analysis 11 (a decided first attempt still outside the rule, the final attempt's decision read, a decision of any attempt opening the first,
  the file or the digest not named, no `evaluation_decision`, the audit not checking the structure, a missing key, a path alone, this attempt's decision, a
  decision at the training code flagged) and note f 3 (no timeout, the silence rule as the timeout, the join given up at once). Result: 231 of 234 killed,
  each by at least one failing test and none by a crash of the run. The survivors are D12 and D20, as in 10.9 (equivalent and obsolete, for the reasons given
  there), and N08 (a file text without its closing full stop is read as an open failure too): the library's text ends with the full stop, as C4 quotes it, so
  a test would have to invent a text, and no case pins either choice. The full run (233 mutants) was made before two last changes, and L05 and R03 survived
  it: the heartbeat tests had just been changed to wait for the thread instead of assuming it gone at the end of the block, so nothing required the join. The
  hung-beat test now requires the wait, and one line of the audit was wrapped; the heartbeat and the analysis series (26 mutants, Q11 added) were run again
  on the final tree: 26 of 26 killed, among them L05 and R01 to R03.
- Real subprocess scenarios (real `Popen` children and parents, logs, SIGKILL and SIGTERM, stub torch, fake training and evaluation with sleeps; scripts in the
  session scratchpad; the harness runs the real `start_child`, replacing only the script in its command). Scenarios 1 to 12 (10.2 to 10.8) were run again on the final
  code: the outputs of 1 to 5, 9 and 11 are identical to the step-1e ones except for times, process ids and the seals that contain them (scenario 1 still ends in the
  error of its own last line); in 6, 7, 8 and 10 the only differences are the new `decision: null` of the `restarts` rows (and, in 8, the new wording of one refusal);
  in 12, the scenario of step 1e's F1, the three refusals at another code are the refusals of the first attempt with the new wording, and its steps 4 to 6 end
  differently as C4 intends: `--new-evaluation-code` alone at the training code with no earlier attempt, refused in 1e, now runs the first attempt (the flag
  changes nothing at the training code), and the two later calls find the evaluation complete. New, scenario 13: (A, note a) a code defect in one real child of 18, with
  17 siblings finishing: the job exits 1, the `stopped` manifest's `stopping_child` is the run, exit code 1 and the real path of `logs/<run>.log`, and that log holds
  the child's own error line; (B, finding 1) two children end with the writer's open failure and with the missing parent directory (a `RuntimeError` each): both
  are recorded `failed` with kind `infrastructure`, the job ends `incomplete` (no `stopping_child`), `--rerun-incomplete` completes them as attempt 2 with the first
  attempts moved aside, and `evaluate` at the training code completes; (C, finding 2) at another code (a second snapshot of the code through `LL_SOURCE_ROOT`) the first
  attempt is refused with no flag, with the restart flag, with it and the new-code flag, with the new-code flag alone, with `--decision` alone and with the restart flag
  and `--decision`, and a relative `--decision` is refused; with `--new-evaluation-code --decision FILE` the evaluation completes as attempt 1, the manifest's
  `decision` equals the path and an independently computed blake2b-128 of the file, its code differs from `training_source_hash`, and the analysis prints the sentence
  of C4 with the decision and not the "outside the rule" one, and `summary.json` carries the decision; (D, finding 2) a first attempt under a decision killed with SIGKILL
  (manifest "running"): the flags again are refused while its directory is in place, `--restart-interrupted` alone completes it as attempt 2 at the same code, the
  manifest's `restarts[0].decision` is the sealed record, and the analysis still names the first attempt's decision.
- Read statically in the installed torch 2.14.0+cu132, not run: `libtorch_cpu.so` contains "Parent directory ", " does not exist.", "File ", " cannot be opened.",
  "opening archive ", "initializing archive " and "open file failed with strerror: " in one cluster after the path of `inline_container.cc` (and
  "PytorchStreamWriter failed " just before it); `torch/serialization.py` gives an ASCII name to
  `torch._C.PyTorchFileWriter(` (`_open_zipfile_writer_file`, line 828; a non-ASCII name is opened by Python's `open`, line 819) and opens the file of
  `torch.load(path)` with Python's `open` (`_open_file`, line 776). The two C4 texts of the tests are assembled from these strings; the path, the enforce prefix
  and the trailing trace of the third case are placeholders.
- Not verified: any execution with real torch or real pytest; the text of a real writer failure (a missing directory, a quota, a full or unwritable file system
  during `torch.save`), in particular whether the raised message is the bare text or carries a prefix or a trace (the strings of the library were read, no
  failing call was made); a compute node; a PBS walltime kill (SIGKILL stands in for it); the heartbeat and the 300 s silence rule across nodes of the shared file
  system; a decision file on the shared file system. The first compute-node run of `tests/test_tsm_scripts.py` is part of the step-0 test job of §6.
