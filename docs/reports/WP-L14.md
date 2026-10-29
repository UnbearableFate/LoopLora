# WP-L14 Step 1: S1B addendum draft

Worker wk-loop, branch `wp/loop`, login/control plane `miyabi-g2` (aarch64), no PBS allocation. Merged master **once**, as instructed: fast-forward `2f7abbc` → `8f4bb1da18d146d5979a6ae0fe9ea5c5dac27e1f`; only the task file/worker registry changed. No push.

Draft: [SLOT-LOOP-v1-S1B](../prereg/SLOT-LOOP-v1-S1B.md). It specifies all three S1A outcome branches, E12/(8,4) T2 with relation 0, a common S-RE start, frozen writer, stage-local L1 calibration, equal 6,000 updates, paired whole-episode decisions, descriptive depth sweep, conditional mechanism controls, complete data/open-record/source-lock contracts and measured-cost release requirements. Flagged S1A ablation admits all four arms with identical `A_query` copies. Main design departures are numbered in the addendum, including the third primary contrast and stricter Bonferroni intervals.

Evidence: all WP-L14 read-first documents, native and slot loop paths, writer signatures/loaders, generator/stratifier/renderer, old P1-L L4 configs/results and real S-RE metadata were inspected. E12 is structurally supported but frozen-writer T2 performance is unverified; old successful L4 runs were oracle writers. Current S1A helpers explicitly assume T1/L1 and will require separate S1B behavior after registration. Codegraph was unavailable in this session's tool catalog, so source inspection used rg and bounded reads.

Login validation: stdlib AST/JSON/blake2b metadata derivation reproduced the parent task digest `b7f93c1ca56e5dd1f79a9dccfe2f6054` and S-RE record digest `816570a0882ed6952b4f851b67d4ef9f`; derived S1B task digest `f556c1b91252b0b74d3ff674717be37c` and all registered purpose/train/bootstrap seeds without application imports or episode generation. Document link/digest/arithmetic checks and `git diff --check` are recorded at completion. No torch, pytest, training, evaluation, job submission, sleep or polling; **0 new node·h**.

Deferred by the Step-1 gate: all implementation, compute tests, actual TaskSpec/disjointness checks, preflight/lock, calibration/pilot measurements, formal runs and confirmation access. The cost table is explicitly conditional on future measured S1A/S1B timings. Next action belongs to the orchestrator: register or amend the addendum on master, then authorize Step 2. Worker stops here; no additional merge is authorized.

## Step 2: implementation and compute validation

Authorization: registered addendum on master `a9178d5b3b9aa191b730e2dcabc91f0a605d33bc`, blake2b-128 `6f2e776f2c6cfde131380fb343854784`, decisions 1–6. Merged master **once** into wp/loop; the add/add conflict in the addendum was resolved by taking the exact registered master file, not editing its decisions. Merge commit `5d5bec5`. No further merge or push. Plane rechecked: login/control, miyabi-g2/aarch64, no allocation.

Implementation is isolated in new `src/looplora/llm/s1b_*.py` modules. S0/S1A source/configs/prereg/locks are unchanged. `s1b_data/rules` encode the T2 layout, eleven seeds, full stream-prefix disjointness using the old layout for old identities, all three S1A branches, fixed N and the calibrated-parent hop-1 stop. `s1b_model/train` preserve the inherited tensor path, support matched L1/L4 training, actual-forward trajectory verification, owner/read accounting, fixed-query caching and identical-initialization untied controls, writer/buffer verification and model-only exports. `s1b_artifacts/preflight/run/analysis` provide source-bound locks, branch evidence, calibration/pilot/smoke/reload/release, exact-release approval, dev selection, independent-opening checkpoint maps, paired episode inference, descriptive sweeps/unseen diagnostics and post-gain controls. Scripts: `prepare_s1b.py`, `analyze_s1b.py`.

Cost estimation uses measured load, steady step, evaluation, checkpoint/support-verification and export overhead; it refuses an indivisible run above 16 node·h. The benchmark separates uninstrumented end-to-end timings from synchronized backend/CUDA-event timings. Controls need this backend's confirmed gain and a separate extension bound to measured control timings; they cannot change the main verdict. Main formal runs require S1A frozen decision/dev/ablation evidence and a ready approved release.

Tests target the registered data/seal boundaries, three branches, source/release mutations, whole-episode bootstrap and continuation screens, native Matrix-L4 logits/gradients, actual trajectory/label isolation, control initialization/read counts/gradient reachability, frozen-writer training and export reload. A synthetic-parent integration exercises the real 20-step smoke→JSON builder→candidate→reload CLI→ready path. Existing slot regression tests additionally check the unchanged T1/S1A/golden paths. No real parent run, preflight, lock, calibration, pilot, formal training/evaluation or sealed data access is performed in this step.

Login validation: AST parsing of all new Python files without importing application code, registered-addendum blake2b check, and `git diff --check`. Compute test results and exact snapshot/job ledger follow below. Dependencies and old source files remain unchanged. The next gate after passing tests is the single core-code cross-review; preflight/lock remain deferred until that review/fix validation.

### Compute test ledger (initial snapshot)

Snapshot **`a731d6dcbb94f06cf5fe0999b522f872653b6fef`** contains implementation, tests and this report's implementation record. Each command uses `/usr/bin/python3 -I scripts/submit.py --queue debug-g --nodes 1 --commit a731d6dcbb94f06cf5fe0999b522f872653b6fef`; no `--wait`, sleep or polling. The three payloads are `-m pytest -q` followed by the files below. Walltime basis: earlier slot metadata tests took 83–84 s and experiment tests 167–177 s; new actual-forward depth-eight tracing and the four-arm synthetic smoke merit the longer model shard. Requests include approximately 1.5× margin and remain below the per-job cap. No S1B science output is created.

| job | payload files | requested walltime / node·h | outcome / actual node·h |
| --- | --- | --- | --- |
| 3728953.opbs | test_s1b_data_analysis.py, test_s1b_artifacts.py | 00:05:00 / 0.083333 | exit 1, 34 passed / 1 failed, mg0002, 24 s / 0.006667 |
| 3728954.opbs | test_s1b_model.py, test_s1b_integration.py | 00:08:00 / 0.133333 | exit 1, 9 passed / 3 failed, mg0014, 27 s / 0.007500 |
| 3728955.opbs | test_slot_experiment.py, test_slot_backend.py, test_slot_data.py | 00:07:00 / 0.116667 | exit 0, 113 passed, mg0028, 235 s / 0.065278 |

Persistent logs under `/work/xg24i002/x10041/LoopLora/logs/pbs/`: `s1b-t-meta-20261029-230438-3532112.log`, `s1b-t-model-20261029-230438-3532117.log`, `s1b-t-legacy-20261029-230438-3532120.log`. Combined requested ceiling 0.333333 node·h; actual total **286 / 3600 = 0.079444 node·h**. Sources/argv, CUDA availability, complete progress, full failure tracebacks and `.exit` markers checked. No real-parent preflight or manifest lock before review.

### Initial failure and skip audit

The watcher message reported metadata “2 failed, 1 error” and model “6 failed, 3 errors, 15 skipped”. These counts do **not** match the named canonical `.log` files or their PBS `.log.pbs` copies. Actual pytest progress is metadata `...............................F...` (35 cases: 34 passed / 1 failed), model `........FF.F` (12 cases: 9 passed / 3 failed), and legacy 113 dots (113 passed). None contains `s`, `SKIPPED`, a fixture-error section or a skipped-test reason. The wrapper's `[ERROR] job.pbs failed at line 69` records pytest's nonzero process exit; it is not another pytest error.

**Skipped tests: 0 in each of these three jobs.** There are no individual skip reasons to enumerate. In particular, the reported fifteen skips are not present in either model log copy, and no S1B test contains a skip/xfail marker. The three training cases failed at the same initialization line; they were not skipped. Retries explicitly request pytest `-ra` and persistent JUnit XML so every actual skip, if any, has a named reason in the log and XML.

Diagnoses and focused corrections:

- Metadata: `test_calibration_writes_sanity_evidence_before_stopping` raises `NameError: np is not defined` before calling production code. Restored the test's NumPy import. Its <10% sanity-stop assertion is unchanged.
- Model: both parameterized matched-training/export round trips and the smoke→builder→CLI-reload integration raise `ValueError: too many values to unpack (expected 5)` in `s1b_train.fit`: five history variables received six lists. Corrected the initializer to five lists. This fixes a production startup error that the existing behavior tests exposed; no assertion, registered rule or training schedule changed.
- Detached the native-versus-Matrix logit-gap tensor before converting the diagnostic maximum to a float, resolving the observed autograd scalar-conversion warning without changing logits, gradients or the bitwise assertion.

Retry the two affected S1B shards in full. Existing S0/S1A source imports and test files did not change; the passed 113-case legacy shard remains applicable. Review stop still waits for successful training/export/integration execution. Current checks on login: AST parsing and diff whitespace only; all runtime work remains in PBS.

### Retry submissions

Snapshot `5518661a4b8491daa99a48eac4730ced2b739899`, one node per job, debug-g, no `--wait` or polling. Both payloads use `-m pytest -o addopts= -q -ra --junitxml=ABS_PATH` and the same file sets as their initial shards; registered assertions remain unchanged.

| job | shard | requested walltime / node·h | outcome / actual node·h |
| --- | --- | --- | --- |
| 3729032.opbs | metadata/statistics/artifacts | 00:01:00 / 0.016667 | exit 0, 35 passed, mg0018, 25 s / 0.006944 |
| 3729033.opbs | reader/trainer/integration | 00:06:00 / 0.100000 | exit 0, 12 passed, mg0044, 57 s / 0.015833 |

Metadata request is >1.5× its measured 24 s. The model's 27 s failed before training, so sizing used the related 167–177 s experiment measurements plus margin for the full four-arm smoke. Logs: `s1b-r1-meta-20261029-231206-3547267.log`, `s1b-r1-model-20261029-231206-3547278.log` under the same persistent PBS log root. JUnit XML: `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b-tests/5518661-{meta,model}.xml`. Both complete logs and exit markers confirm the intended full snapshot/argv, CUDA and success. XML independently confirms 35 / 12 cases, **zero failures, errors and skips**; neither XML contains a skipped testcase, so neither has an individual skip reason. Retry actual **82 / 3600 = 0.022778 node·h**; cumulative actual **368 / 3600 = 0.102222 node·h**, including initial failures. No source, script, config or test changed after the passing retry snapshot.

### Combined validation required before review

Latest orchestrator instruction requires one green invocation of the **entire S1B test set plus the legacy slot shard** on the review candidate. The separate green runs above do not complete this new gate. The combined payload explicitly includes all four discovered S1B test files and the three legacy files: `tests/test_s1b_data_analysis.py tests/test_s1b_artifacts.py tests/test_s1b_model.py tests/test_s1b_integration.py tests/test_slot_experiment.py tests/test_slot_backend.py tests/test_slot_data.py` (expected 160 cases). Use `-m pytest -o addopts= -q -ra --junitxml=ABS_PATH`, one node on debug-g. Measured job durations sum to 25 + 57 + 235 = 317 s; ×1.5 = 475.5 s, so request **00:08:00 (0.133333 node·h)**. All tests and synthetic training remain on compute. Preflight, source lock and all real-parent science remain deferred; review stop follows verified combined success, with every actual skip reason recorded here.
