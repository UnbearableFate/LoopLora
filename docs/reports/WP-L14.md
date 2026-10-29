# WP-L14: S1B implementation and validation

**Confirmation and dev depth evaluations completed; frozen CONFIRMATION analysis is submitted as 3731365.** Both evaluation jobs exited 0 and produced all thirty expected entries each. Frozen twelve-checkpoint selection `e3f269db39cd9b431d27e0a5eb528d6e`, orchestrator [open.json](../../reports/slot-loop-v1/s1b/open.json) `b698f7ce5e31b2b43da5c210134588b0`, release `d1ccbb7ff4c31fa53abfd7a3cc9ad572`, science `258f5f5a24eae684f640a9fd62679bb643470a76` and lock `2309d5a8ebae91332f32bbd85e5eeb8a` remain unchanged. Completed worker actual **17.922222 node·h** across 31 jobs; independent rerun **0.009444 node·h** separately. Pending analysis requests **0.333333 node·h**; verdict and descriptive dev tables await that compute result. Unseen remains off; mechanism controls await a later orchestrator release extension.

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

Completed **3729079.opbs**, `s1b-full-review`, commit **`4bd1a5bb154052470bc112dda4c0211b16323fd9`**, one node/debug-g/00:08:00. Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-full-review-20261029-231847-3564355.log`; JUnit: `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b-tests/4bd1a5b-full.xml`. **Exit 0, 160 passed, zero failures/errors/skips**, mg0003, 2026-10-29 23:18:54–23:23:29 JST: **275 s / 0.076389 node·h** (pytest time 267.92 s). Full source/argv, CUDA availability, 100% progress, explicit pytest summary and exit marker checked. No `--wait`, sleep or polling.

JUnit independently confirms every expected module: S1B data/statistics 23, artifacts 12, model 11, integration 1; legacy experiment 48, backend 23, data 42. Total **47 S1B + 113 legacy = 160**; **no skipped testcase elements**, so there are no individual skip reasons. JUnit file blake2b-128: `1cf3b3f63250b7ec954297e7467f9062`. No warning or fixture-error section appears in the combined log. Cumulative actual **(286 + 82 + 275) / 3600 = 0.178611 node·h**; all six jobs are accounted for, including failed attempts.

Review scope is the S1B implementation/tests/scripts introduced after merge `5d5bec5` plus the focused fixes. `git diff --exit-code 4bd1a5bb154052470bc112dda4c0211b16323fd9 HEAD -- src scripts tests configs` confirms the review code exactly matches the combined green snapshot; only report/ledger commits follow. Registered addendum blake2b-128 remains `6f2e776f2c6cfde131380fb343854784`. Final static whitespace check passes. Working tree is clean after the report commit. No merge beyond the authorized one, push, real-parent execution, sealed-data access, preflight or source lock. Next: `STOP: review <full hash>`; orchestrator arranges the single cross-review before preflight/lock.

## Cross-review fixes before preflight (main cba968b)

User authorization: read the new WP-L14 review section and both `s1b-rv-{code,opus}.md` reviews directly from `/work/xg24i002/x10041/LoopLora`, main `cba968b5d5b8ae9e94bdf58e31b2b7ad9230cb62`; **do not merge master**. All three documents read. No further review round. Fix items 1–9 with tests, run the full S1B + legacy set green, then execute real-parent preflight and commit the lock. Calibration and later science still wait for S1A's decision record.

First regression: `tests/test_s1b_review.py::test_review_zero_boundary_keeps_storage_null_and_selects_matrix`, rv-code's exact synthetic 2,000-episode fixture, registered seed/10,000 draws/three-contrast interval. Initial snapshot adds only the regression and this report to reviewed `67deeee`; production source remains byte-identical to the reviewed implementation. Submit it before changing the bootstrap, expecting the exact-zero lower-bound assertion to fail for the documented numerical reason. Subsequent fix validation will run the identical regression and the entire combined suite.

RED job **3729223.opbs**, snapshot `fd6f5c3efd4869bcd891d4f90d1aa7d22e5ebc2a`, one node/debug-g/00:01:00 (0.016667 requested node·h), payload `-m pytest -o addopts= -q -ra --junitxml=/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b-tests/fd6f5c3-review-red.xml tests/test_s1b_review.py -k zero_boundary`. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-review-red-20261029-235216-3758012.log`. Verified **exit 1, exactly one expected failure, zero errors/skips**, mg0005, 23:52:22–23:52:29 JST: **7 s / 0.001944 node·h**. The assertion is precisely `2.2204460492503132e-17 == 0`; no setup failure. The watcher reported “2 failed, 1 error”, but the complete log and JUnit agree on one failure and no errors. JUnit blake2b-128 `87fb205112d0137257e80e8de53c4d48`. No source edits in this RED snapshot; fixes below land afterward.

Implemented the nine requested changes (now verified in the full green run):

1. Integer bootstrap numerators; exact rational linear quantiles and lower-bound sign, float conversion only for reports. The original RVC-01 regression also checks the Matrix continuation choice.
2. Calibrated smoke samples are captured from the original step-20 in-memory instance before closing it. Both the first load and CLI reload must reproduce that record; the ready release binds its sample digest. Deterministic B-altered reload regressions fail at both boundaries.
3. Retain every scheduled export and original dev sample file, including step 0, under each run's `checkpoints/`. Every dev row carries its export/sample digests and step. `selected.pt` is an exact copy of the retained winner. Selection audits verify file bytes, sample/row binding, selected digest and exported arm/seed/step/config metadata. A forced-multiple-improvement test verifies retention and rejects a stale selected file.
4. Persist final-valid-token q/m/z in `[episode,hop,step,feature]` order for every evaluated depth, with ordered loop indices and row provenance. Analysis validates shape/coverage, and tests compare saved NPZ tensors to actual captured forward rows at L=1/2/4/8.
5. Frozen confirmation JSON now contains explicitly descriptive arm/seed/hop counts, EM, NLL, all depth-sweep rows, S1−M1 and the Memory×Depth point interaction. Frozen `verdict.md` renders those tables separately from the decisions. Positive and null outcomes both require complete tables.
6. Branching refuses untriggered ablation records and requires one explicit matching S1A release digest across decision/dev/ablation. Existing positive fixtures now supply this required provenance; mismatch/missing/stray-record regressions are added. No S1A source or schema is modified here; absent release fields fail closed before calibration.
7. Nonfinite calibration candidates are recorded with `nll:null`/error and excluded using the inherited tie rule. All-nonfinite cases write a failed record before STOP; lock/calibration validation handles excluded candidates consistently.
8. Real preflight now includes T1 Native–Matrix identity at L1/2/4 for both projection branches, owner checks at L1/2/4/8, and explicit `[B,12,768]` keys / `[B,12,128]` values / 12 writes assertions. The inherited S1A implementation remains unchanged.
9. A two-update T2 training test uses the actual `BatchStream` with two spawned workers and no `source=` override. It checks CPU-prefetched `token_fact_positions` on both 32-episode/E12 batches, finite reader updates and frozen support/checkpoint evidence.

Login validation: AST parsing and whitespace checks only; no application imports. Next submit all five S1B test files (including review regressions) and the same three legacy slot files in one invocation, with explicit skip summary/JUnit. Prior full run cost 275 s; added prefetch/trajectory/owner/negative-reload cases justify 00:10:00, one node/debug-g, below the cap. Tests must pass before any real preflight. Master has not been merged; registered preregs and inherited source remain unchanged.

GREEN validation **3729273.opbs**, snapshot `4a76e16d3ae25a2fc4d3268d7cb8ff8531461b89`, one node/debug-g/00:10:00 (0.166667 requested node·h). One pytest invocation includes all five S1B files plus the three legacy slot files, with `-o addopts= -q -ra` and `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b-tests/4a76e16-review-green.xml`. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-review-green-20261030-000433-3793774.log`. Verified **exit 0, 179 passed, zero errors/skips**, mg0038, 2026-10-30 00:06:32–00:11:38 JST: **306 s / 0.085000 node·h** (pytest 299.41 s). Source/argv, CUDA, complete progress, exit marker and JUnit checked. Module counts: S1B 23+12+11+2+18 = **66**, legacy 48+23+42 = **113**. All 18 review regressions pass, including the identical RED case, T2 `BatchStream(workers=2)`, both altered-reload refusals, retained checkpoints and saved trajectories. JUnit blake2b-128 `d662139cd2cb871149e804eb224af6cc`. No skipped testcase elements, so no individual skip reasons. Both RED and GREEN are immutable snapshots; no sleep/polling. Cumulative actual **(643+7+306)/3600 = 0.265556 node·h**. Only this report changes after GREEN.

### Real-parent preflight and lock preparation

Authorization remains the main-cba968b review section, read directly without merging. Full green validation and the required RED regression are inspected, so the preflight gate is now open. Run `-m looplora.llm.s1b_run --mode preflight --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/preflight-4a76e16` on an immutable commit with exactly the tested source. The output path was confirmed absent. The preflight uses only ordinary S1B smoke and authorized legacy identity replays: both projection branches, T1 L1/2/4 identity, T2 Matrix-L4 gradients, all-depth owner/trajectory checks, capture shapes, eight 20-update real-parent freeze runs and reloads. It neither needs nor reads S1A's decision/confirmation, S1B calibration or new dev/confirmation sets.

Request one node/debug-g/00:15:00 (0.250000 node·h). Planning anchors: prior real-parent S0 preflight 53 s, S-RE 0.199 s/step and 6.84 s/1,000 one-question evaluations. S1B adds 160 updates, both projection branches' T1 three-depth Native/Matrix comparisons and repeated strict parent/reload instances; allow roughly ten minutes for these expanded checks/loading, ×1.5 walltime. Actual S1B preflight duration will replace this estimate. No source edits after the passed tests. If all eleven checks pass and source/inventory/manifest match, create the S1B-specific lock, archive its preflight report/inventory in `reports/slot-loop-v1/s1b/`, commit locally and STOP locked. Any failed check blocks the lock. Calibration and all later stages remain held for S1A's frozen decision record.

Completed **3729312.opbs**, `s1b-preflight`, immutable snapshot **`09044cea3a642d64b6b0470051e5f562045fb1e2`**, one node/debug-g/00:15:00. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-preflight-20261030-001531-3859454.log`; persistent output as above. Verified **exit 0, status complete, passed true**, host mg0034, 2026-10-30 00:22:23–00:25:14 JST: **171 s / 0.047500 node·h**. The log binds the intended commit/argv, confirms CUDA and ends with exit 0; the separate exit marker is 0. Its sole warning is the inherited `torch_dtype` deprecation. No checks skipped. Measured equivalent-preflight sizing is now 171×1.5 = 256.5 s, rounded to a five-minute request if a future source change requires a rerun. Cumulative actual for all nine jobs, including failures/retries: **(643+7+306+171)/3600 = 0.313056 node·h**.

Inspected all eleven mandatory checks and their nested evidence in the completed report:

| check | observed evidence |
| --- | --- |
| bytes | both projection branches: capacity 12; Slot 43,028 B/episode, Matrix 393,216 B/episode; measured Slot bytes equal formula |
| capacity | both branches: E=S=12 succeeds; E>S raises |
| features | both branches: actual `[4,12,768]` keys, `[4,12,128]` values, exactly 12 writes per episode; unit keys and no base-key substitution; key gap 5.96e-8 ≤ 1e-5, value gap 0, reconstruction relative error 1.10e-7 ≤ 1e-6 |
| frozen | eight real-parent 20-update runs, advance/query-copy × M1/M4/S1/S4: frozen tensors and support unchanged, trainable tensors changed; exact trainable counts 180,224 / 442,368 respectively |
| isolation | both branches pass label isolation and immutable-memory checks |
| legacy | S-RE replay exactly 723/1,000, NLL 1.642127275466919, relative-read error 0.19612866640090942; all saved per-sample fields bitwise equal and no differences; T1 L1/2/4 on both branches, 1,000 episodes and 455,808,000 elements per branch, zero unequal elements / max gap 0 |
| matrix_L4 | both branches: full-logit max gap 0 and Q/R/U/B gradient gaps 0 versus native tensor loop |
| owner_mask | both branches: every L1/2/4/8 owner/padding/reorder check passes for all three backends |
| registry | registered T2 layout `f556c1b91252b0b74d3ff674717be37c`, all 22 registered ID groups checked disjoint |
| reload | all eight smoke exports reload with complete matching metadata and evidence |
| trajectory | both branches at L1/2/4/8 agree with actual forward; fixed-query and identical-start untied checks pass |

### Committed S1B lock

Archived the original preflight JSON byte-for-byte and created the lock with the reviewed metadata command `scripts/prepare_s1b.py --mode lock --preflight reports/slot-loop-v1/s1b/preflight/preflight.json --out reports/slot-loop-v1/s1b/manifest.lock.json`. The builder rechecked the registered prereg, actual S-RE inventory, full ID manifest, all mandatory checks and current source before writing `status: locked`. This bounded lock preparation ran on the login/control node; no torch, pytest, training or evaluation ran there. The standalone inventory follows the existing inventory format and records S-RE's role, loader, real file digests, legacy replay and preflight provenance. S0/S1A inventories and locks are unchanged.

| committed artifact | blake2b-128 |
| --- | --- |
| [manifest.lock.json](../../reports/slot-loop-v1/s1b/manifest.lock.json) | `a91da220da5fee75d5b26f923116083f` |
| [checkpoint_inventory.json](../../reports/slot-loop-v1/s1b/checkpoint_inventory.json) | `aec34c268f7c66991dca466e34b8255c` |
| [preflight/preflight.json](../../reports/slot-loop-v1/s1b/preflight/preflight.json) | `f7b14c225bc72d9712cab3183d7de665` |

Final bounded stdlib metadata verification checks archived/original report equality, embedded report/manifest/inventory binding, all eleven passed flags, prereg digest `6f2e776f2c6cfde131380fb343854784`, and the actual sorted-source digest `bb60b89c78ee840a940c5ec855455604` (101 files, 1,220,063 bytes; excludes bytecode). `src/`, scripts, tests and configs remain identical to the 179-test green snapshot and preflight snapshot; only report and lock metadata follow. `git diff --check` passes. No additional merge, push, sleep or polling.

Stop at the committed lock. **Calibration, pilot, later smoke/release and formal execution remain held for S1A's frozen decision record**, per registered decision 4 and the review instruction. Branch preparation will fail closed unless the required S1A records carry an identical explicit release digest; no S1A schema/source was changed here. Any later `src/` edit invalidates this lock and requires another preflight.

## Advance-branch release preparation after S1A

Orchestrator authorization: S1A completed and merged on master `650dc3e`; verdict ADVANCE, dev no_trigger, no ablation. **Do not merge master.** Copy the two supplied frozen records unchanged, construct branch evidence from those copies, then calibration → calibrated 20-update smoke → equal LR pilot → candidate release → CLI reload checks → ready release. Stop for exact-release `approval.json` before main training. Every job stays within 16 node·h and requests measured time ×1.5. The existing S1B lock/source are retained.

Read the records directly from `/work/xg24i002/x10041/LoopLora/reports/slot-loop-v1/` and verified their supplied blake2b-128 digests with stdlib metadata operations. Both explicitly identify S1A release `523d11d57a08a2491893ebe7cedc86a7`:

| unchanged input | blake2b-128 | relevant fields |
| --- | --- | --- |
| decision_s1a.json | `d9f00ee8b208a9e1f84edb6ebe935616` | stage S1A, verdict ADVANCE |
| s1a-selection.json | `b6e5d452089045e27b22d4d94a57aacc` | stage S1A-DEV, outcome no_trigger, empty reasons |

Byte-identical copies are at `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/release-prep-bd4e24f/s1a/`. The same new preparation root contains byte-identical copies of the committed S1B lock and checkpoint inventory. No ablation record is fabricated or supplied. The reviewed branch builder will read only these copies and write `branch.json`, requiring `advance`: M1/M4/S1/S4, shared frozen A, common unadapted S-RE. The immutable science snapshot for calibration, smoke and pilot is **`bd4e24fc72223c404d6c7e605970cecf35ffe612`**; subsequent report-only commits leave it and its source digest unchanged.

First job runs the reviewed branch builder then the existing calibration CLI in the same compute allocation. Calibration is once per active backend at L1 on 2,048 ordinary calibration episodes × hops1–3: Matrix's three gains and SoftSlot's twelve gain/temperature settings. Nothing later is submitted until the completed calibration passes the **selected Matrix-L1 hop-1 EM ≥10%** sanity gate. A below-threshold result stops for the orchestrator; no retuning or reduced budget.

Walltime basis: the real S1B preflight's advance-M1 L1 evaluation of eight episodes × three hops took at most 0.181889112 s (advance-S1: 0.128287220 s). Conservatively use the slower L1 figure for all fifteen settings: `0.181889112 × (2048/8) ×15 = 698.454 s`; allow 30 s for parent/metadata/data startup, then ×1.5 = 1,092.681 s, round to **00:20:00**, one node/debug-g (**0.333333 requested node·h**). Calibration's own per-candidate measurements will replace this bound. Output `release-prep-bd4e24f/calibration/` was absent before submission. No torch, pytest, training or evaluation runs on the login/control node. No sleep/polling or master merge.

Submitted **3729493.opbs**, `s1b-calibration`, snapshot `bd4e24fc72223c404d6c7e605970cecf35ffe612`, one node/debug-g/00:20:00. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-calibration-20261030-003936-3994587.log`; its persistent spec captures the exact branch-builder and calibration payload. Outcome/actual cost pending. The next action after completion notification is to inspect the copied-input/branch bindings, all fifteen candidate records and the registered sanity gate, then proceed to smoke only on success.

### Calibration sanity stop and diagnostic submission

Verified **3729493.opbs exit 1**, mg0036, 2026-10-30 00:49:06–00:53:06 JST: **240 s / 0.066667 node·h**. The branch builder accepted the unchanged copied records and wrote `advance`, branch-record blake2b-128 `180f43948eb3bc7613e8cbe2ff3cd55a`. All fifteen calibration candidates are finite and recorded before the precise registered `StopForOrchestrator: calibrated Matrix parent hop-1 calibration EM below 10%`. The calibration file's `status: complete` means the grid finished; it does **not** override that failed sanity gate. Calibration-file blake2b-128 **`6e4f7346376cb010820a84832f6d7d91`**. Matrix selected g=1: hop-1 104/2048 (5.078125%), NLL 9.898808733679743; g=2 reaches 116/2048 (5.6640625%) but is not the registered NLL winner. SoftSlot selected g=1, tau=.3: hop-1 26/2048 (1.26953125%), NLL 8.523176905902801. These are calibration diagnostics, not a reading-limit result. Cumulative actual **(1127+240)/3600 = 0.379722 node·h**.

Read the new **“Calibration stop (2026-10-30 00:56 JST)”** task section directly from main checkout `/work/xg24i002/x10041/LoopLora`, master `7b247dad8d911073a67e6273afbc0de1e5babbed`, without merging. It authorizes one debug-g diagnostic job ≤30 minutes, S1B calibration and non-sealed T1 reference data only. No S1B pilot/dev/confirmation or optimization. The orchestrator decides any design amendment.

Diagnostic implementation: [scripts/diagnose_s1b_transfer.py](../../scripts/diagnose_s1b_transfer.py), with no `src/` edits or changes to the existing lock. It runs the S1B evaluator on the S1A T1 pilot reference (recorded Matrix step-0 **733/1000, NLL 1.4211350753479493**), compares all per-sample fields to both inherited native and explicit Matrix evaluators, and separately reproduces the two selected calibration settings on the original three-hop batch layout. It then evaluates T1 and T2 hop-1 with selected Matrix/SoftSlot settings, S0's clean-final oracle, a separately labeled all-token true-value oracle, and zero-memory reads (`m=0`, same L1 reader retained). Captured actual queries give key top-1, signed true-key margin and read error at **every valid token**, including an explicit final-position table. Query/owner/position/target/write-span and frozen-adapter invariants are checked on compute. A training-only S-RE value head describes whether the written true value still decodes the target; it never participates in model reads.

Five fixed T2 calibration prompts and five T1 reference prompts, with separate support/read question text, target token IDs, answer-position token and full-vocabulary top-five next tokens, are saved for both selected backends. A diagnostic prompt-only intervention additionally puts T1 wording on the **same T2 hop-1 supports, queried edges, answers and IDs**; this is evidence for distinguishing wording/position transfer from writer failure, not a proposed change to the registered task. All samples and per-token diagnostic arrays persist in the new diagnostic output; no sealed set is generated.

Walltime model uses the measured maximum calibration-candidate evaluation **16.092340426 s** for 2048×3 questions. Two exact reproduction passes, five one-hop conditions on each of two T2 wordings, and five T1 reference conditions scale to approximately 99 s; allow another 30 s for native/reference comparisons, 100 s for actual per-token capture/invariant checks and 30 s startup/input verification: approximately 259 s ×1.5 = 389 s, rounded up to **00:10:00**, one node/debug-g (**0.166667 requested node·h**), within the task's 30-minute limit. Only AST/whitespace/source-diff metadata checks run on login. No sleep/polling or additional master merge; existing calibration/lock files remain untouched.

Submitted **3729676.opbs**, `s1b-diagnosis`, immutable snapshot **`685504858921ff3dc2b6f5bf29051ca15aaee6a1`**, one node/debug-g/00:10:00. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-diagnosis-20261030-010221-630526.log`; output `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/diagnosis-6855048`. Outcome/actual cost pending. No other job is submitted. After completion notification, inspect the replay, prompts/targets, per-token keys and oracle/off comparisons; archive evidence, recommend (a), (b) or (c) with measured support, and stop at a diagnosis commit.

### Completed diagnosis and recommendation

**3729676.opbs completed with exit 0**, host **mg0009**, 2026-10-30 01:09:08–01:12:17 JST: **189 s / 0.052500 node·h** (diagnostic application 171.642913 s). Requested 600 s; no timeout, retry, skipped condition or extra job. The canonical log and exit marker confirm the intended snapshot/argv, CUDA, complete status and `t1_reference_matches: true`; only the inherited `torch_dtype` deprecation appears. Cumulative actual for all eleven jobs is **(1127+240+189)/3600 = 0.432222 node·h**.

**Recommendation (b): prompt/position mismatch is the main demonstrated cause of the sanity failure.** The evidence does not show a general S1B evaluator bug: on the non-sealed T1 reference the S1B path gives **733/1000, NLL 1.4211350753479493**, exactly the recorded step 0, with zero NLL gap. Its IDs, predictions, correctness, targets and NLL agree with inherited native/Matrix paths; the four per-sample scoring fields are bitwise equal. Both selected three-hop T2 calibration results reproduce **exactly**, including every recorded hop summary. All adapter parameter/buffer digests remain unchanged, and the original source lock still matches.

The diagnostic conditions below use the same prepared support within each set, L1, full-vocabulary next-token EM, and the unadapted parent. T1 uses its recorded Matrix g=1 / SoftSlot g=1,tau=.1; both T2 wordings use the stopped calibration's Matrix g=1 / SoftSlot g=1,tau=.3 with no reselection. T2's third column changes **only the read-question wording/positions** to T1's template on the same 2,048 hop-1 supports, queried edges, targets and IDs. No new T2 data or training is introduced.

| diagnostic condition | T1 reference: correct/1000; EM; NLL (nats) | T2 original: correct/2048; EM; NLL (nats) | same T2 with T1 wording: correct/2048; EM; NLL (nats) |
| --- | --- | --- | --- |
| selected Matrix | 733; 73.30%; 1.4211 | 104; 5.0781%; 9.8989 | 645; 31.4941%; 4.7496 |
| selected SoftSlot | 770; 77.00%; 1.3486 | 26; 1.2695%; 8.5227 | 200; 9.7656%; 5.7588 |
| S0 clean-final oracle | 924; 92.40%; 0.3574 | 1628; 79.4922%; 0.8338 | 1695; 82.7637%; 0.7794 |
| additional all-token oracle | 908; 90.80%; 0.4893 | 1595; 77.8809%; 0.9377 | 1648; 80.4688%; 0.8863 |
| memory off | 3; 0.30%; 9.9666 | 4; 0.1953%; 13.2631 | 4; 0.1953%; 10.0705 |

Clean-final supplies the **actual written value of the true hop-1 next edge as m**, gain 1, at the final valid question token; earlier tokens retain the native Matrix g=1 read, exactly S0's clean condition. Thus it is one common oracle comparison, not a SoftSlot-specific earlier-token path. The all-token oracle supplies that same value at every valid question token. Memory off means **m=0 at every token**, retaining the same L1 reader R/U/B path; it does not remove the reader branch. The hop-1-only diagnostic has a different question-batch shape from the original three-hop calibration: its NLLs differ slightly (Matrix 9.89894329 versus calibration 9.89880873; SoftSlot 8.52272525 versus 8.52317691), while counts match. The original three-hop replay is exact, so these two layouts are explicitly kept separate in the artifacts.

Actual captured query diagnostics support a query-transfer failure. For each token, scores are dot products against the **written** unit DPFP keys; signed margin is true-edge score minus the largest other-edge score. At L1 the query is computed before the read, so Matrix and SoftSlot have the same query/key results. Per-token owner/index coverage and trace/forward consistency passed, with no substitution of a recomputed query for these metrics.

| actual-read statistic | T1 reference | T2 original | same T2 with T1 wording |
| --- | --- | --- | --- |
| final token key top-1 | 894/1000 = 89.40% | 303/2048 = 14.7949% | 1218/2048 = 59.4727% |
| final token mean signed margin | +0.246934 | −0.177465 | +0.040025 |
| final token median signed margin | +0.227712 | −0.184593 | +0.048921 |
| all valid token key top-1 | 19.4813% over 16,000 reads | 9.3598% over 59,392 reads | 16.3147% over 32,768 reads |
| all valid token mean signed margin | −0.107138 | −0.166334 | −0.123559 |
| final Matrix relative read error (mean) | 0.178568 | 0.575341 | 0.350337 |
| final SoftSlot relative read error (mean) | 0.077224 | 0.470142 | 0.358762 |

The archive includes key top-1 and margin distributions at **every actual valid read position**. Near the T2 answer, key top-1 at offsets −3/−2/−1 is 10.4980% / 12.4023% / 14.7949%; there is no nearby earlier position with T1-like retrieval. The maximum over all 29 T2 positions is only 16.1621%, at offset −27. Read-owner expansion, true-edge→answer matching, 12 writes and original writer spans all pass. This provides no evidence for an off-by-one answer-index implementation bug.

The five requested original T2 and five T1 prompts, including both backends' top-five next tokens and exact target token IDs, are in the committed [prompts.md](../../reports/slot-loop-v1/s1b/diagnosis-6855048/prompts.md); [diagnosis.json](../../reports/slot-loop-v1/s1b/diagnosis-6855048/diagnosis.json) additionally includes every question token and fact-write position. T2 uses `Starting from …, keep following … until there is no further fact … Which person is reached?\nAnswer:` (29 tokens); T1 uses `Which person is linked from … by the relation …?\nAnswer:` (16 tokens). Both score after the final **`:`** at zero-based positions **28 / 15**, respectively. Targets round-trip as one leading-space token, and writes align to each fact's final `.\n` token, matching the inherited rule. The first original T2 example targets ` Nigel` (token 65816), but Matrix's top five are ` Jo`, ` Ann`, ` Chan`, ` Joan`, ` Allen`; the paired T1-worded version makes ` Nigel` top-1. The first five paired examples have identical support text, target tokens and recorded write positions.

The paired change raises Matrix EM by **26.4160 pp** and key top-1 by **44.6777 pp** without changing the writer. Meanwhile, the oracle raises original-prompt T2 EM from 5.08% to 79.49%. These observations localize most of the collapse to prompt-conditioned query/read transfer. They do **not** establish that wording alone, rather than its associated token positions/representations, causes it: the intervention changes both. They also do not establish perfect writer transfer. The diagnostic training-only S-RE value head decodes true written targets at **934/1000 = 93.40%, NLL 0.5317** on T1 versus **1713/2048 = 83.6426%, NLL 2.1421** on T2, identical under both T2 wordings; reworded T2 Matrix remains below T1 (31.49% versus 73.30%). There is residual graph/writer/query domain shift, but the evidence does not support (c), a writer-only failure, as the principal diagnosis.

**Disposition:** retain the sanity stop and the stopped calibration unchanged. The T1-worded hop-1 counterfactual is diagnostic only; its one-edge question cannot simply replace the registered chain-end question at higher hops. Recommend that the orchestrator decide an amendment/transfer strategy addressing query wording/positions while retaining the intended chain-end semantics before authorizing more science. No source fix, new calibration, smoke, LR pilot, candidate release, confirmation or main training is performed or implied by this report.

Archived metadata/prompts/inputs/job evidence are byte-identical to their persistent outputs; all **22** produced-file blake2b digests were checked. The 17 NPZ sample/position files remain at the persistent diagnostic output, named and hashed in [provenance.json](../../reports/slot-loop-v1/s1b/diagnosis-6855048/provenance.json) and [artifacts.json](../../reports/slot-loop-v1/s1b/diagnosis-6855048/artifacts.json). No sample files were deleted.

| evidence | blake2b-128 |
| --- | --- |
| diagnosis.json | `2ad225aa70f09940ff68b56c214ae0d3` |
| prompts.md | `63bde55472147b9957b4a5d3135abc18` |
| provenance.json | `92bdec3c3572aa5d61dae7cd066ecd71` |
| diagnostic script | `c84e3d88970dbfb9879241b3a6803aa3` |

Final metadata checks confirm byte-identical stopped calibration, branch/lock bindings, archived inputs, original registered prereg, unchanged sorted-source digest and no diff in `src/` or the lock relative to `bd4e24f`. The only executable added is the authorized diagnostic under `scripts/`, and its exact committed snapshot ran successfully on compute. `git diff --check` passes. No additional master merge, push, sleep or polling. Stop at the diagnosis commit for the orchestrator's design decision.

## Registered A1 implementation and release preparation

Read task section **“Amendment A1 (2026-10-30 01:20 JST)”**, main `777dc7fd9ee08302be139c9e020dc0b798a4a205`, and [SLOT-LOOP-v1-S1B-A1.md](../prereg/SLOT-LOOP-v1-S1B-A1.md), registered at `4178808`, exact blake2b-128 **`341d3c71f9334a1384a102b097e7ad75`**. Step 1 explicitly authorizes **one merge of master**. Completed it once, merge **`76d66e1e55535b954d3fc2ed2d7993401c2e7eb4`**, without conflicts. `git diff --name-only master -- src` lists only the eight S1B modules: all `src/` outside `s1b_*` is byte-identical to merged master, including the completed S1A path. No push. Plane: login/control miyabi-g2/aarch64, no allocation; all runtime remains on compute.

A1 implementation:

- Bind the unchanged original addendum and A1 separately in registration checks, manifests, new locks, science configs, model export/load, release and confirmation opening records. An altered/missing A1 refuses execution.
- Replace the active T2 calibration contract with `prepare_s1b.py --mode read-settings`: load S1A's **ready** release by exact file digest **`523d11d57a08a2491893ebe7cedc86a7`**, verify it matches the frozen branch evidence, take `arms.matrix_L1.read` / `arms.soft_slot_L1.read`, and change only capacity 16→12. Matrix g=1; SoftSlot g=1,tau=.1; all-valid-question scope, no recency. The resulting record binds its source files, new lock, branch and A1; smoke/pilot/release recheck that provenance and L1/L4 read equality.
- The original stopped calibration remains immutable, exact digest **`6e4f7346376cb010820a84832f6d7d91`**. Its path/digest and descriptive-only disposition are carried into the read-settings record and release. No calibration grid is repeated and decision 3 no longer gates S1B. Removed the calibration runtime mode and superseded sanity/nonfinite-grid tests; replaced them with A1 transferred-read, stopped-evidence and no-calibration-execution checks.
- Set **N=18,000**, batch 32, **train [0,576,000)** for each registered main seed; keep the 200-update/6,400-episode LR pilot unchanged. The manifest explicitly lists each stream's start/count/base seed; full old/new ID disjointness uses all enlarged prefixes and each old entry's original layout. A collision at episode 575,999 must fail. Main and conditional control checkpoints are 1,800,3,600,…,18,000; step 0 is nonselectable. Selection, tasks, seeds, contrasts and seals are unchanged. Resource estimation uses N=18,000 and refuses an indivisible job above 16 node·h.

New A1 regressions cover registered-document mutation at the lock boundary, source-release/stopped-evidence byte changes, missing amendment at opening, read retuning/release provenance, extended-window collision detection, full stream windows, unchanged pilot budget and the amended schedule/science hash. Existing smoke→original samples→first reload→candidate→CLI reload→ready integration now uses the transferred S1A reads and A1-bearing exports. All original exact-bootstrap/trajectory/owner/selection/real-prefetch regressions remain, as do the legacy slot tests. No further cross-review is requested, per the registered task.

Submit one combined invocation: all **six** `tests/test_s1b_*.py` modules, plus `test_slot_experiment.py`, `test_slot_backend.py`, `test_slot_data.py` and merged master's `test_slot_runbook.py`, with `-o addopts= -q -ra` and persistent JUnit XML. Prior combined run took **306 s**; allow about 40 s for the enlarged full-prefix registry and added A1/runbook cases, ×1.5 ≈519 s, rounded to **00:10:00**, one node/debug-g (**0.166667 requested node·h**). Current checks on login are AST parsing, registered file digests, source-isolation diff and whitespace only. Preflight, new lock, smoke, pilot and release remain downstream of the complete green run.

Submitted **3729793.opbs**, `s1b-a1-tests`, immutable snapshot **`84575879f4b9b6c7f93a9c2c27203e55c89905b8`**, one node/debug-g/00:10:00. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-tests-20261030-012351-840120.log`; JUnit `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b-tests/8457587-a1-full.xml`. Full ten-file argv is retained in the persistent job spec. Outcome/actual cost pending; no dependent job submitted and no waiting/polling. Inspect any actual skips individually before proceeding.

Verified **3729793.opbs exit 0, 191 passed, no failures/errors/skips**, host mg0035, 2026-10-30 01:23:56–01:29:12 JST: **316 s / 0.087778 node·h** (pytest 307.95 s). Source/argv, CUDA, complete progress and separate exit marker checked. JUnit independently agrees: S1B A1/artifacts/data/model/integration/review **9+12+23+11+2+19 = 76**, legacy experiment/backend/data/runbook **48+23+42+2 = 115**. No skipped testcase elements, so no individual skip reasons. XML blake2b-128 **`6ee2ace56ca2c4d2bdac70131f3bf563`**. Cumulative actual **(1556+316)/3600 = 0.520000 node·h**. Source/scripts/tests/configs are identical to the green snapshot; only report commits follow. Bounded stdlib hashing gives sorted-source blake2b-128 **`d7e7ff0fb5987d5e5abfc312c82c449c`**.

### A1 real-parent preflight and lock

Next one-node compute job runs the existing real-parent preflight on the exact tested snapshot, then the reviewed lock builder, branch builder and A1 read-settings builder after successful completion. These builders do not train or evaluate. The old lock/calibration are preserved. New persistent root **`/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587`** contains newly copied, byte-identical S1A decision/dev/release records and stopped-calibration evidence; all four pinned digests checked before copying. Preflight output `preflight/` was absent. No S1B pilot/dev/confirmation or new calibration is accessed.

Walltime uses the measured real-parent preflight **171 s**, plus approximately 40 s for the larger full-prefix registry and repeated lock/read-settings reconciliation, ×1.5 ≈316.5 s; round up to the platform's ten-minute request unit: **00:10:00**, one node/debug-g (**0.166667 requested node·h**). All eleven engineering checks remain mandatory; failed/incomplete/stale preflight refuses the new lock. Once the output is inspected, archive and commit the new A1 lock before any calibrated-read smoke/LR pilot. No source edits after green tests, another master merge, push, sleep or polling.

Submitted **3729827.opbs**, `s1b-a1-preflight`, immutable green snapshot **`84575879f4b9b6c7f93a9c2c27203e55c89905b8`**, one node/debug-g/00:10:00. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-preflight-20261030-013227-1082665.log`; persistent spec retains the exact preflight/builders payload. Outcome/actual cost pending; no downstream smoke or pilot submitted.

Verified **3729827.opbs exit 0**, mg0008, all builders completed, **all eleven preflight checks and nested passed flags true**. Log interval 2026-10-30 01:34:08–01:37:06 JST is 178 s; finished PBS history (`qstat -H -f 3729827.opbs`, one read, not polling) reports **00:02:59 = 179 s / 0.049722 node·h**, including the one-second allocation-accounting difference. Use the PBS duration here. CUDA/source/argv, complete status and separate exit marker checked; only inherited `torch_dtype` deprecation. Cumulative actual **(1872+179)/3600 = 0.569722 node·h**.

Evidence inspected: T1/native identity at L1/2/4 for both branches has max gap 0 and zero unequal elements over 455,808,000 elements per branch; T2 Matrix-L4 full logits and Q/R/U/B gradient gaps are 0. Captures are exactly `[4,12,768]` / `[4,12,128]`, writes `[12,12,12,12]`, keys unit and base keys unused, with the same measured feature tolerances as the prior preflight. Eight 20-update runs preserve frozen bytes/support and change trainables; all eight reloads pass. Both branches pass owner checks and trace fidelity at L1/2/4/8, label isolation, capacity and byte checks. The registry verifies all **22** ID groups, including **576,000** training episodes for each of 17/29/43 and unchanged **6,400** pilot episodes. S-RE replay matches every per-sample field bitwise, 723/1000 with unchanged NLL. No checks skipped.

Archived the original preflight and compute-produced new lock/inventory/branch/read-settings records **byte-for-byte** under [reports/slot-loop-v1/s1b/a1/](../../reports/slot-loop-v1/s1b/a1/manifest.lock.json), along with all four pinned input copies and the 191-test JUnit XML. Rechecked embedded preflight/manifest/inventory equality, current source, frozen/support flags, extended windows and exact transferred-read settings. Source and A1 digest match the green snapshot; stopped calibration remains descriptive, unchanged. The historical S1B lock and S0/S1A locks are untouched.

| A1 artifact | blake2b-128 |
| --- | --- |
| manifest.lock.json | `2309d5a8ebae91332f32bbd85e5eeb8a` |
| preflight/preflight.json | `c7f0b5de30ca77c598046bbb1ae5eeb7` |
| checkpoint_inventory.json | `657ecba2a9af49c6db58eb3841a5d529` |
| branch.json | `6ded1c79c96b267aff4498c120b57b09` |
| read-settings.json | `c0eec1c6d461a394eb1f8517c5784057` |

### A1 smoke and first reload

Commit the verified new lock before the next submission; the lock commit becomes the common science snapshot for smoke/pilot/release, with source identical to the tested/preflight snapshot. Run the existing four-arm smoke with the persistent A1 lock/branch/read-settings files and output `a1-8457587/smoke/`. Every arm starts from unadapted S-RE, trains 20 updates on ordinary smoke only, checks frozen bytes/support and exact optimizer reachability, saves original in-memory step-20 samples, then physically reloads its export and compares predictions/correctness/NLL/IDs to the original record. This first reload must pass before the LR pilot; the separate candidate-bound CLI reload checks remain after pilot/release construction.

Measured advance-arm preflight training totals approximately **28.13 s** for the four 20-update paths; its maximum eight-episode evaluation is **0.188187 s**. Scaling four sample/evaluation passes per arm to the 64-episode smoke gives approximately 24.09 s, plus about 48 s for parent/reload instances, expanded-registry/input reconciliation and support/export checks: approximately **100 s ×1.5 = 150 s**, rounded to **00:10:00**, one node/debug-g (**0.166667 requested node·h**). Smoke's actual per-arm step/evaluation/load overhead will refine the LR-pilot request. Main-run walltimes remain fixed-N estimates from the later 200-step pilot, with the 16 node·h refusal intact. No LR pilot or main job submitted yet, no calibration repeated, no sleep/polling, no further merge or push.

New A1 lock committed at **`258f5f5a24eae684f640a9fd62679bb643470a76`** before submission. This is the common A1 science commit for smoke/pilot/release; it differs from the 191-test/preflight snapshot only in report and archived metadata. Submitted **3729928.opbs**, `s1b-a1-smoke`, that immutable commit, one node/debug-g/00:10:00. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-smoke-20261030-014340-1263423.log`; output `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/smoke`. Outcome/actual cost pending. After completion, inspect all four first-reload/freeze/sample bindings and use the measured timings to size the registered LR pilot.

Verified **3729928.opbs exit 0, smoke complete**, mg0019, log 2026-10-30 01:43:44–01:45:02 JST (78 s); reported PBS walltime **79 s / 0.021944 node·h**, with the same one-second allocation-accounting difference as preflight. Full source/argv, CUDA and separate exit marker checked. Only inherited `torch_dtype` warning. Every arm completed and selected step 20, all finite; exact trainables Q/R/U/B (**180,224** parameters), **294,914 frozen parameters** plus buffers, frozen tensor bytes and fixed support unchanged, trainables changed. All four physical first reloads reproduce the original in-memory sample record and restore complete A1/projection/task/backend/depth metadata. No skips or omitted arm. Verified actual selected-export/original-sample file digests and all lock/branch/read/source bindings. Full [smoke record](../../reports/slot-loop-v1/s1b/a1/smoke/smoke.json), per-arm training histories and a persistent-export/sample inventory are archived under `reports/slot-loop-v1/s1b/a1/smoke/`; exports and original sample NPZs remain at the persistent smoke root. Smoke JSON blake2b-128 **`556491f738dc5d7344d3dac1d88ee878`**. Cumulative actual **(2051+79)/3600 = 0.591667 node·h**.

| arm | steady step (s) | step p90 (s) | max 64-episode smoke evaluation (s) | first reload/freeze |
| --- | --- | --- | --- | --- |
| M1 | 0.208764 | 0.211953 | 0.615752 | pass |
| M4 | 0.345650 | 0.349622 | 0.732273 | pass |
| S1 | 0.234230 | 0.235165 | 0.544820 | pass |
| S4 | 0.415785 | 0.421825 | 0.797899 | pass |

### A1 LR pilot

Run the registered grid once: four arms × LR {3e-4,1e-3,3e-3} ×200 updates, each from unadapted S-RE; same pilot training stream and 1,000-episode hop1–3 pilot shard, including step 0. No main dev/confirmation access. Per arm, choose lowest final pilot NLL; candidates within .01 nats tie to lower LR. Nonfinite candidates are recorded/excluded, and all failed or all worse than step 0 stops. Pilot weights never initialize main runs. The source/read settings/lock stay fixed at common science commit `258f5f5a24eae684f640a9fd62679bb643470a76`.

Walltime from measured smoke parts: `3 ×200 ×sum(arm step_seconds)` = **722.657 s**; two pilot evaluations per LR, scaling each arm's slowest 64-episode evaluation by 1000/64, gives **252.257 s**; measured setup/final/export/support overhead scaled to three candidates adds **16.560 s**. Allow **120 s** for twelve parent instances, fresh prefetched-worker startup and one-time registry/module overhead. Expected **1,111.474 s**, ×1.5 = **1,667.212 s**, rounded to **00:30:00**, one node/debug-g (**0.500000 requested node·h**), well below 16 node·h. The pilot's own 200-step measurements, including the slowest finite candidate per arm, will determine the N=18,000 main-run release walltimes; an indivisible run above the cap stops. Output `a1-8457587/pilot/` is new. No main training submitted and no source changes, extra merge, push, sleep or polling.

Submitted **3729954.opbs**, `s1b-a1-pilot`, immutable science commit **`258f5f5a24eae684f640a9fd62679bb643470a76`**, one node/debug-g/00:30:00. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-pilot-20261030-014906-1343462.log`; output `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/pilot`. Outcome/actual cost pending. Inspect every candidate and the per-arm registered choice on completion, then build the exact candidate release and perform its independent CLI reload checks before ready/STOP.

Verified **3729954.opbs exit 0, pilot complete**, mg0008, log 2026-10-30 01:49:10–02:08:20 JST (1,150 s); reported PBS walltime **00:19:11 = 1,151 s / 0.319722 node·h**, including the one-second allocation-accounting difference. Source/argv, CUDA, completion and separate exit marker checked; only inherited `torch_dtype` warning. **All twelve candidates are finite, none excluded/skipped**, each exactly 200 updates / 6,400 episodes, same registered pilot stream and 1,000-episode shard. All preserve frozen bytes/support and change trainables. All four LR choices independently reproduced with the registered .01-nat tie rule, and every winning NLL is below step 0. Pilot JSON blake2b-128 **`5a1c2b090772bb4348b1d5d6566e4a9f`**. Full [pilot record](../../reports/slot-loop-v1/s1b/a1/pilot/pilot.json) and per-candidate histories archived; all twelve actual pilot-export digests verified and retained on persistent storage, never used to initialize main runs. Cumulative actual **(2130+1151)/3600 = 0.911389 node·h**.

| arm | step-0 NLL | LR .0003 NLL | LR .001 NLL | LR .003 NLL | selected LR |
| --- | --- | --- | --- | --- | --- |
| M1 | 10.542903 | 6.782494 | 6.440166 | 6.366551 | .003 |
| M4 | 22.516058 | 8.551318 | 6.933272 | 6.663716 | .003 |
| S1 | 9.778330 | 6.620500 | 6.428880 | 6.347167 | .003 |
| S4 | 22.312346 | 8.215011 | 6.788721 | 6.641147 | .003 |

The release uses each arm's **slowest finite candidate** for step/evaluation/load/overhead. Main request is `ceil_to_600s(1.5 × (load + 18000×step + 11×evaluation + overhead))`; N and the ten dev checkpoints are not adapted to pilot scores. Every indivisible one-node run fits the 16 node·h cap; no budget is reduced.

| arm | step (s) | evaluation (s) | load (s) | overhead (s) | time with 1.5× margin (s) | one-node request |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | 0.192734 | 8.040816 | 1.361697 | 5.460390 | 5346.711 | 01:30:00 (1.500000 node·h) |
| M4 | 0.335047 | 10.966011 | 0.430269 | 3.450730 | 9233.038 | 02:40:00 (2.666667 node·h) |
| S1 | 0.214682 | 8.672139 | 0.666512 | 3.415585 | 5945.638 | 01:40:00 (1.666667 node·h) |
| S4 | 0.391021 | 13.159213 | 0.435913 | 3.422765 | 10780.477 | 03:00:00 (3.000000 node·h) |

Requested total for the twelve main runs, **if later approved**, is **26.5 node·h**. They remain held. No main dev/confirmation/unseen payload is generated or read during pilot/release preparation.

### A1 candidate and independent CLI reloads

Next one compute job invokes the existing `prepare_s1b.py --mode release` builder against the exact lock/branch/transferred reads/pilot/smoke and common science commit. Candidate is `pending_reload`, never trainable. Run **four separate Python CLI processes** (`-m looplora.llm.s1b_run --mode reload-check`), each loading its real smoke export and comparing all original IDs/predictions/correctness/NLL plus complete buffers/backend/projection/depth metadata against that exact candidate. Only after all four pass, rebuild the release with their four reports and require `status: ready`; run the reviewed release validator. No approval record is created. Optional unseen is fixed **false** for this release, using the registered default and avoiding extra evaluation; all mandatory main/depth-sweep contrasts remain unchanged.

Walltime model: prior 79-second smoke includes eight model instances, 80 training updates, repeated metadata checks and sixteen small evaluations. The release job removes training but adds repeated full-prefix lock/read/source reconciliation and four independent interpreter startups; allow approximately **180 s** for these measured loading/evaluation and metadata paths, ×1.5 = **270 s**, rounded up to **00:10:00**, one node/debug-g (**0.166667 requested node·h**). New paths `a1-8457587/candidate.json`, `reload/{M1,M4,S1,S4}/` and `release.json` were absent. No source changes after tests/preflight, no extra merge, push, sleep or polling. Stop at the committed ready-release artifacts for orchestrator `approval.json`.

Submitted **3730094.opbs**, `s1b-a1-release`, common immutable science commit **`258f5f5a24eae684f640a9fd62679bb643470a76`**, one node/debug-g/00:10:00. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-release-20261030-021236-1562889.log`; the persistent spec captures candidate builder → four independent CLI processes → ready builder → release-validator payload. Outcome/actual cost pending. On completion, inspect all four reports and exact candidate/ready bindings, archive the release and evidence, update the ledger, commit and STOP release; main training stays held for approval.

### Ready release: verified and held for approval

Verified **3730094.opbs exit 0, status ready, four reloads, twelve configs**, host **mg0013**. Log interval 2026-10-30 02:30:10–02:32:50 JST is 160 s; reported PBS walltime **00:02:41 = 161 s / 0.044722 node·h** includes the one-second allocation-accounting difference. The log shows candidate construction, each of the four separate CLI processes completing, ready construction and the reviewed `require_release` validator passing. Its final digest matches the actual file bytes: **`d1ccbb7ff4c31fa53abfd7a3cc9ad572`**. Source/argv, CUDA and separate exit marker checked; the four warnings are the inherited `torch_dtype` deprecation, one per reload process. No failed or skipped reload, no source change or extra test run needed. Cumulative actual for all sixteen jobs is **(3281+161)/3600 = 0.956111 node·h**, including earlier failures/retries and superseded calibration/diagnosis.

Archived [candidate.json](../../reports/slot-loop-v1/s1b/a1/candidate.json), [release.json](../../reports/slot-loop-v1/s1b/a1/release.json), all four `reload/ARM/reload.json` records, the canonical job log/spec and [release-artifacts.json](../../reports/slot-loop-v1/s1b/a1/release-artifacts.json) under the committed A1 directory. Every copy is byte-identical to its persistent original below `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/`. The original candidate remains `pending_reload` and unchanged; ready differs only by `status` and the four added report digests. Recomputed its canonical blake2b reload basis **`ede5da36d6621f5d99f682fe3f38f362`**, which all four reports share.

| exact artifact | blake2b-128 |
| --- | --- |
| candidate.json | `377912afbbec7b804af21bcb3ca8ae9a` |
| release.json | **`d1ccbb7ff4c31fa53abfd7a3cc9ad572`** |
| reload/M1/reload.json | `e700412e78ecb1b1094ba46820aa1705` |
| reload/M4/reload.json | `6293b3e231890ba4742f799bb991606a` |
| reload/S1/reload.json | `d676516984d7f6daf337b57976d777fc` |
| reload/S4/reload.json | `1e671699eaaf00405b61cdcf10bfccf4` |
| release-artifacts.json | `673dbef997ff969e28715e2dadd34491` |

Final bounded stdlib metadata audit verifies the ready/candidate delta and basis, actual reload-report bytes, all four real smoke export and original-sample byte digests, complete matching metadata/read/depth/source, all twelve distinct config labels for seeds 17/29/43, selected LR .003 and exact pilot-derived request seconds. It also verifies frozen branch-input bytes, the embedded transferred-read record and source release, lock/inventory binding, both original preregs and A1, unchanged stopped-calibration evidence and the frozen analysis-script digest. The current sorted `src/` digest exactly matches the new lock and the tested/preflight science source; source/scripts/tests/configs remain identical to science commit `258f5f5a24eae684f640a9fd62679bb643470a76`. Only report/artifact commits follow it. All four CLI reloads compare every sample field against the original step-20 in-memory smoke; their physical export digests equal the recorded smoke exports, not pilot weights.

Release contract for approval:

- **Science commit:** `258f5f5a24eae684f640a9fd62679bb643470a76`; **source:** `d7e7ff0fb5987d5e5abfc312c82c449c`; **A1:** `341d3c71f9334a1384a102b097e7ad75`.
- **Exact ready file:** persistent `a1-8457587/release.json` or the byte-identical committed `reports/slot-loop-v1/s1b/a1/release.json`; **file digest:** `d1ccbb7ff4c31fa53abfd7a3cc9ad572`.
- **Advance design:** M1/M4/S1/S4, common unadapted S-RE `3994a5631021de03b2039974ac934e56`, shared frozen A; all-valid-question scope, capacity 12, no recency; Matrix g=1 and SoftSlot g=1,tau=.1 from S1A release `523d11d57a08a2491893ebe7cedc86a7`. Stopped calibration `6e4f7346376cb010820a84832f6d7d91` is descriptive only and was not repeated.
- **Budget:** 18,000 updates for all four arms × three continuation seeds; 576,000 training episodes per run; dev checkpoints 1,800 through 18,000; same selection/contrasts/continuation rules. Per-run requests 5,400 / 9,600 / 6,000 / 10,800 seconds; all ≤16 node·h, twelve-run requested total 26.5 node·h. Optional unseen is fixed false. Frozen analysis script blake2b-128 `78f001a232595acce6d38091f0e4895b`.
- **Next gate:** orchestrator `approval.json` with `status: approved` and `release_blake2b_128: d1ccbb7ff4c31fa53abfd7a3cc9ad572`. No A1 approval file was created or reused. No main training, dev selection or confirmation has been submitted or accessed.

The original S1B calibration/lock and all S0/S1A artifacts remain preserved. There was exactly the one A1-authorized master merge, no further merge or push. All runtime ran on compute; this final turn performs metadata reads/hashing/copying and documentation only on login/control. `git diff --check` passes. Stop at the committed ready release for the orchestrator's approval.

## Approved twelve-run main execution and frozen dev selection

Read **“Approval (02:42)”** directly at the end of main checkout's task file, master `dc4ef34406d9ddb891a9021e2f57ef9c09b96cbd`; no merge or pull. The orchestrator already committed [approval.json](../../reports/slot-loop-v1/s1b/a1/approval.json) in this worktree: **`0b5973b3a8577eaca6c06b56395f4763abc2dbbb`**, with scope amended by **`547a5e454c1e9cd72e29ce6dd0f15b6c4c7c8f66`** to permit all twelve jobs queued at once. These commits are retained unchanged. Approval file blake2b-128 **`27f254b499b302ed932e5f3260e08034`**; verified status approved, exact release digest, science commit and all-at-once scope. Copied its bytes unchanged to persistent `a1-8457587/approval.json`, and verified both release copies and the frozen analysis script before submission. The existing ready release/source/configs are unchanged; no new approval was fabricated or edited.

Plane confirmed login/control miyabi-g2/aarch64, no allocation. Initial `qstat --limit` shows project **0/128 accepted, 0/16 running**; twelve jobs fit with four running slots left for wk-slot. Every submission uses **`scripts/submit.py --commit 258f5f5a24eae684f640a9fd62679bb643470a76 --nodes 1 --queue regular-g`**, no `--scale-up`, and `-m looplora.llm.s1b_run --mode train` with the same persistent A1 lock/release/approval. Outputs are new `a1-8457587/main/ARM/sSEED/`, all confirmed absent before submission. Every run starts from the same unadapted S-RE; pilot/smoke weights are not used. Fixed 18,000 updates, complete train [0,576,000), dev step 0 and 1,800,…,18,000, registered LR/read settings and exact science hashes apply.

Submitted all twelve on 2026-10-30 **02:44:38–02:44:40 JST**. Receipts/specs are verified against each label, source, arm, seed, approval/release path, output and walltime; no job was omitted or submitted twice. The durable persistent ledger is `a1-8457587/main-jobs.json`, with a byte-identical committed [copy](../../reports/slot-loop-v1/s1b/a1/main-jobs.json). This is a submission receipt state, not a claim that any job has completed.

| arm | seed 17 job | seed 29 job | seed 43 job | one-node walltime | total requested node·h |
| --- | --- | --- | --- | --- | --- |
| M1 | 3730250.opbs | 3730251.opbs | 3730252.opbs | 01:30:00 | 4.5 |
| M4 | 3730253.opbs | 3730254.opbs | 3730255.opbs | 02:40:00 | 8.0 |
| S1 | 3730256.opbs | 3730257.opbs | 3730258.opbs | 01:40:00 | 5.0 |
| S4 | 3730259.opbs | 3730260.opbs | 3730261.opbs | 03:00:00 | 9.0 |

All exact log/spec/output paths are in the ledger; logs use prefix `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-ARM-sSEED-20261030-0244...`. Requested main total **26.5 node·h**; actual node·h and exit/freeze status pending for every run. Completed preparation actual remains **0.956111 node·h**; running allocations are not represented as zero-cost completed work. There is no new aggregate budget reduction. No sleeping, status polling or independent wake watcher: the orchestrator supplies completion notifications under the worker's standing WAIT protocol.

On each completion notification, inspect exit 0/complete status, exact release/science/config/seed/read bindings, finiteness, frozen tensor/buffer/support evidence and all eleven dev rows (0 plus ten checkpoints through 18,000), with retained checkpoint/sample provenance. A failure stops for diagnosis; do not resubmit a changed run. Archive completed records/histories, retain model exports needed for selection, and update per-job actual costs.

**Only after all twelve pass**, submit the unchanged frozen `scripts/analyze_s1b.py --stage DEV` on compute from the science commit, using the same lock/release and the complete `a1-8457587/main/` records root. Its retained-export/sample audit and exact registered checkpoint selection produce the twelve-label selection; preserve the script digest **`78f001a232595acce6d38091f0e4895b`**. Then archive selection/source index, report and STOP selected for the independent re-run and orchestrator opening record. This approval does **not** cover confirmation, unseen, depth-sweep evaluation or mechanism controls; none is accessed or submitted here.

### A1 main completion and frozen DEV selection

Completion notification received for **3730250–3730261**, all exit 0. Independently read every canonical log and separate exit marker, each complete record and retained-artifact file metadata. All twelve records bind the exact approved release, A1 lock, immutable science commit, arm/seed/config hash, transferred read, LR .003 and frozen dev ID digest. Each completed **18,000 updates / 576,000 episodes / 2,304,000 questions**, with all-finite flags true, Q/R/U/B reachable with finite nonzero gradient evidence, trainables changed, frozen parameters/buffers unchanged and identical support before/after. Every record has the exact eleven dev rows **0,1800,…,18000**, 1,000 episodes each, finite NLL and step 0 nonselectable. All **132 export/sample pairs** exist, total **258,295,338 bytes**; selected exports and every scheduled candidate remain on persistent storage. No run changed or resubmitted, no arm or seed omitted, no skipped check represented as passed. Full-array and artifact-byte verification is pending the compute audit below.

Archived each original `record.json`, `train_log.jsonl`, `train_steps.npz`, canonical job log/exit marker and submission spec under [main/](../../reports/slot-loop-v1/s1b/a1/main/). Copies were checked byte-for-byte with BLAKE2b. [main-completion.json](../../reports/slot-loop-v1/s1b/a1/main-completion.json) records the completed metadata checks and pending compute audit. [main-history.json](../../reports/slot-loop-v1/s1b/a1/main-history.json) contains the twelve finished PBS accounting rows, normalized against ledger IDs by removing the `.opbs` server suffix; the history helper returns numeric IDs. This reconciliation is metadata inspection following completion notification, with no waiting or status polling. The submission requested `regular-g`; PBS reports its routed execution queue as `small-g` for all twelve. Preserve both fields rather than relabel the request.

| arm | seed 17 PBS seconds | seed 29 PBS seconds | seed 43 PBS seconds | actual node·h |
| --- | ---: | ---: | ---: | ---: |
| M1 | 3397 | 3322 | 3468 | 2.829722 |
| M4 | 5482 | 6322 | 5674 | 4.855000 |
| S1 | 3637 | 3652 | 3815 | 3.084444 |
| S4 | 6663 | 6799 | 7013 | 5.687500 |

All per-job elapsed times fit their approved requests and the 16 node·h cap. Actual main total **59,244 / 3600 = 16.456667 node·h** versus **26.5 requested**. Completed cumulative cost, including all prior failures/retries and superseded calibration/diagnosis, is **(3442+59244)/3600 = 17.412778 node·h**, across 28 jobs. Pending analysis cost is accounted separately.

Submitted **3730962.opbs**, `s1b-a1-dev`, one node/debug-g/**00:10:00**, with `scripts/submit.py --commit 258f5f5a24eae684f640a9fd62679bb643470a76`. No source/scripts/tests/config changes since the 191-test green snapshot; frozen analysis script digest remains **`78f001a232595acce6d38091f0e4895b`**. The job first checks all **216,000** saved updates for finite losses/norms/positive times, reconciles all **4,332** logged update rows and their LR/clipping fields against arrays, verifies registered stream seeds, and checks all 132 saved dev sets' IDs/hops/counts/prediction correctness and summary values. This extra read-only audit writes `dev-audit/main-audit.json`; it neither selects weights nor changes the registered rule. Then a separate process invokes the unchanged frozen CLI:

```text
scripts/analyze_s1b.py --stage DEV
  --manifest /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/manifest.lock.json
  --release /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/release.json
  --records /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/main
  --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/dev-selection
```

The frozen script physically loads all 132 exports to check metadata, audits retained checkpoint/sample bytes and original NLLs, reproduces the registered nonzero-step lowest-NLL selection with 1e-6/earlier-step rule, verifies selected-file equality, complete label coverage and twelve distinct selected digests, then writes `selection.json` and `source_index.json`. Both DEV and audit outputs were absent before submission. The [job receipt](../../reports/slot-loop-v1/s1b/a1/dev-job.json), full spec and exact inline audit payload are archived under `dev-selection/`; payload digest **`5320f8838aa5b37f6f177d7f78ba9115`**. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-dev-20261030-050214-2397168.log`.

Walltime basis: the preceding S1A six-run/66-row frozen DEV job used **7 s** in PBS history; S1B has twelve runs/132 exports plus the explicit update/sample audit and A1 extended-prefix release reconciliation. The measured A1 release/reload job took **161 s** including multiple release validations and four model instances. Allow **180 s** for this CPU artifact/metadata path, ×1.5 = **270 s**, rounded to the existing ten-minute request unit: **600 s / 0.166667 requested node·h**. Actual remains pending. No training, inference, test rerun or model loading on login; login work was bounded stdlib metadata/copy/hash inspection. No confirmation or unseen payload opened, no depth sweep/control submitted, no source edits, further merge, push, sleep or polling. After completion, inspect the audit/selection/source index, archive and update this ledger, then STOP selected for the independent rerun and orchestrator opening.

### Frozen DEV selection verified: STOP selected

Verified **3730962.opbs exit 0**, host **mg0003**, unchanged immutable science snapshot. Canonical `.log` and `.log.pbs` are byte-identical, the separate exit marker is 0, all twelve audit receipts say passed, and both frozen CLI and wrapper report complete. Log interval **2026-10-30 05:02:19–05:02:54 JST** is 35 s; the single finished-job PBS history query reports **36 s / 0.010000 node·h**, including allocation overhead. The inline audit took **0.389686 s** and the audit plus frozen CLI **27.760642 s**; accounting uses PBS elapsed, not those partial timings. This resolves all pending audit/selection/cost entries above. Completed cumulative actual is **(62686+36)/3600 = 17.422778 node·h**, across **29 jobs**; none pending.

All **216,000** updates have finite losses/norms and positive durations; all **4,332** training-log rows exactly match saved arrays, LR/warmup and clipping flags. All **132** dev sample sets pass registered IDs/count/hops/value checks and reproduce their original correctness/NLL summaries. The frozen script then successfully checked export metadata and byte digests for all **132** scheduled candidates and original samples, exact release/config/seed/depth/read provenance, frozen-writer/support evidence, the registered nonzero-step selection, selected-file equality and twelve unique checkpoint digests. **No failed, omitted or skipped audit/run.** No tests were rerun: source/scripts/tests/configs are still identical to the tested science snapshot; the complete **191 passed / zero skips** compute result remains applicable.

Archived the original compute-produced [selection.json](../../reports/slot-loop-v1/s1b/a1/dev-selection/selection.json), [source_index.json](../../reports/slot-loop-v1/s1b/a1/dev-selection/source_index.json), [main-audit.json](../../reports/slot-loop-v1/s1b/a1/dev-selection/main-audit.json), job log/exit marker and finished PBS history under `reports/slot-loop-v1/s1b/a1/dev-selection/`, each byte-identical to its persistent original. Reconciled every receipt with both the persistent and already archived run histories by BLAKE2b. Updated `dev-job.json` and `main-completion.json` to complete in both locations. The [selected-artifacts.json](../../reports/slot-loop-v1/s1b/a1/dev-selection/selected-artifacts.json) inventory binds each winning step, export/sample path and digest, selected-file size and run-record digest to the compute audit. All scheduled exports, original dev samples and selected files are retained in `a1-8457587/main/` for the independent rerun and later approved opening.

| artifact | blake2b-128 |
| --- | --- |
| selection.json | `e3f269db39cd9b431d27e0a5eb528d6e` |
| source_index.json | `c9e51be2a280962ab47cc6bf335b7336` |
| main-audit.json | `689bf3d8a24a685eb94e20df413f80dd` |
| frozen scripts/analyze_s1b.py | `78f001a232595acce6d38091f0e4895b` |
| approved release.json | `d1ccbb7ff4c31fa53abfd7a3cc9ad572` |
| manifest.lock.json | `2309d5a8ebae91332f32bbd85e5eeb8a` |

Selected steps and descriptive **DEV-only, hops 1–3** summaries follow. These are checkpoint-selection records, not confirmation outcomes; no gain or continuation verdict is made here. Lowest NLL excludes step 0 and uses the registered 1e-6/earlier-step rule. Every chosen NLL is below its own step-0 value. The full twelve-checkpoint digest map is in the frozen selection artifact, without changes to its schema or bytes.

| arm | seed | selected step | dev NLL (nats) | dev EM (%) |
| --- | ---: | ---: | ---: | ---: |
| M1 | 17 | 18000 | 5.618181 | 12.9333 |
| M1 | 29 | 18000 | 5.618711 | 12.8667 |
| M1 | 43 | 18000 | 5.617210 | 12.0333 |
| M4 | 17 | 18000 | 5.478263 | 14.3333 |
| M4 | 29 | 18000 | 5.506691 | 14.9667 |
| M4 | 43 | 16200 | 5.434525 | 15.4667 |
| S1 | 17 | 18000 | 5.510304 | 14.5667 |
| S1 | 29 | 14400 | 5.521522 | 14.0333 |
| S1 | 43 | 16200 | 5.511117 | 13.5000 |
| S4 | 17 | 18000 | 5.266448 | 17.3667 |
| S4 | 29 | 14400 | 5.282213 | 17.8000 |
| S4 | 43 | 18000 | 5.272094 | 18.3667 |

The advance branch and contrasts **S4−S1, M4−M1, S4−M4** are unchanged. Stop at this selected checkpoint map: the independent rerun and orchestrator's `s1b-open.json` are next. No opening record was created, confirmation/unseen payload opened, depth sweep/control run, additional merge/pull/push performed, or sleep/polling used. Login/control work in this finalization was bounded stdlib receipt/hash/copy inspection and report edits; all application execution occurred in the completed compute job.

### Opened (05:18): authorized evaluation submissions

Read the new task section **“Opened (05:18): confirmation, the depth sweeps and the frozen verdict”** directly from the main checkout, as instructed (registered on master `0635178`); **no merge or pull**. Independent reviewer rv-code's [report](../../reports/slot-loop-v1/s1b/a1/independent-selection-rerun.md), archived verbatim from main, confirms job **3730976.opbs** reproduced both `selection.json` and `source_index.json` byte-for-byte and found no discrepancies in all twelve runs. Its PBS actual is **34 s / 0.009444 node·h**: record this external review separately from wk-loop's existing **62,722 s / 17.422778 node·h**. The known combined completed S1B worker + independent-rerun total is **62,756 s / 17.432222 node·h**, before the two new pending jobs.

The orchestrator committed `reports/slot-loop-v1/s1b/open.json` in this worktree at **`9f68575e27f8c14f2a03ab9b78ff5f98f8a6ddc5`**. Verified BLAKE2b of its bytes against the read-only copy at **`/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/open-root/reports/slot-loop-v1/s1b/open.json`**: both **`b698f7ce5e31b2b43da5c210134588b0`**. All selected-label/digest entries match frozen selection, release bytes match, and the opening-root analysis script and three prereg files have their registered digests. The open root and record are untouched. Current source/scripts/tests/configs remain identical to science `258f5f5`; no new tests or source edits are needed for these artifact-only steps.

Created [selected-entries.json](../../reports/slot-loop-v1/s1b/a1/selected-entries.json), with exactly twelve labels mapped to `a1-8457587/main/ARM/sSEED/selected.pt`; persistent and archived bytes match, digest **`1a355810c35058d8bc9519aa13bd596c`**. Bounded byte hashing rechecked every actual selected export against its opened digest. Both output directories below were absent before submission. Each job uses one node and the **unchanged CLI from the full science commit**, via `scripts/submit.py --commit 258f5f5a24eae684f640a9fd62679bb643470a76`, with the existing A1 lock/release/selection, this entries file and **`--open-root /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/open-root`**:

| job | CLI purpose | evaluation | output below a1-8457587 | queue / walltime / requested node·h |
| --- | --- | --- | --- | --- |
| 3731051.opbs | `--mode evaluate --set confirmation` | 2,000 paired episodes × hops 1–8; all twelve selected checkpoints; M4/S4 at L=1/2/4/8 | `confirmation/` | regular-g / 01:30:00 / 1.500000 |
| 3731052.opbs | `--mode evaluate --set dev` | 1,000 dev episodes × hops 1–8; same checkpoints/depths; descriptive only | `dev-depth-sweep/` | regular-g / 00:50:00 / 0.833333 |

Each output has **30 expected sample files**: six L1-trained checkpoints at L1, plus six L4-trained checkpoints at four depths. Evaluation checks the exact export mapping, runs the existing trace/forward equality checks, validates all episode IDs/hops/counts and `[episode,hop,step,feature]` trajectories, and saves original q/m/z, correctness, predictions, NLL and provenance. The same L4 evaluation supplies its primary arm row and the depth-four sweep row; no replacement of independently trained L1 controls. Stdout reports status/counts rather than metrics. DEV sweep results do not alter the already frozen checkpoint selection or any confirmation decision. Unseen is **false** in the approved release and is not submitted; mechanism controls are not authorized yet.

Walltime model uses the largest observed main evaluation at each trained depth, over all seeds and all eleven evaluations, for **1,000 episodes × three hops**:

| arm | measured maximum evaluation seconds |
| --- | ---: |
| M1 | 7.914513 |
| M4 | 11.308367 |
| S1 | 8.607215 |
| S4 | 11.899191 |

For each backend, estimate depth `L` by `T1 + (L−1)×max(0,(T4−T1)/3)`, preserving the measured non-recurrent cost rather than multiplying the whole L4 forward by depth. Scale question rows by **8/3**, sum the thirty rows and paired seed counts: ordinary evaluation **856.986777 s** at 1,000 episodes and **1713.973553 s** at 2,000. Add one further full ordinary-evaluation allowance for diagnostic capture/trace replay and serialization; this is an explicit conservative estimate, not a newly measured diagnostic timing. Add **60 s** for twelve model/export setups and release reconciliation, based on measured maximum setup ≤2.551 s per model and the 27.761-second frozen DEV audit/CLI path. Expected totals: dev **1773.973553 s**, confirmation **3487.947106 s**. Apply **1.5×** and round up to 600 s: **3000 / 5400 s**, respectively. Both exceed debug-g's 30-minute limit, so both request regular-g; each is well below 16 node·h. Reassess against actual evaluation/serialization times once finished.

Submission receipts, exact source/argv/spec/log/output paths, timing inputs/formula, requested resources and baseline cost ledger are in [opened-evaluation-jobs.json](../../reports/slot-loop-v1/s1b/a1/opened-evaluation-jobs.json), byte-identical to the persistent ledger. Full job specs are archived in `confirmation/` and `dev-depth-sweep/`. Logs: `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-confirm-20261030-052205-2428093.log` and `s1b-a1-devdepth-20261030-052205-2428102.log`. Both outcomes/actual costs are pending; none is represented as zero-cost completed work.

After completion notifications, inspect every expected evaluation entry and saved-provenance receipt on compute as needed, without changing/repeating a failed job. Then invoke the **frozen** `scripts/analyze_s1b.py --stage CONFIRMATION` with this selection/open root and complete confirmation records, keep the descriptive dev sweep separate, archive `verdict.json`/`verdict.md`, update verdict tables and actual node·h, and STOP verdict. No sleep, polling, independent watcher, further merge/pull/push or login-side application execution. The orchestrator supplies completion notifications under the worker WAIT protocol.

### Evaluation completion and frozen CONFIRMATION analysis submission

Verified **3731051.opbs and 3731052.opbs exit 0**, complete status, immutable science commit and intended evaluation arguments in both canonical logs. Each log matches its `.log.pbs` copy byte-for-byte, each separate exit marker is 0, and each has only the inherited `torch_dtype` deprecation warning. Each saved index contains **exactly all 30 expected labels/depths**, the approved release, and the correct purpose; confirmation additionally binds the exact open record, while the descriptive dev index has its registered null opening field. Every indexed file exists with positive size and finite positive recorded evaluation duration. These are completed evaluation checks; the frozen analysis will independently rehash confirmation files and validate saved metadata, IDs, trajectories and pairing before any verdict is accepted. No failed job or changed resubmission.

| job | compute host | log interval (JST) | PBS elapsed / actual node·h | saved sample files / compressed bytes | index blake2b-128 |
| --- | --- | --- | --- | --- | --- |
| 3731051.opbs, confirmation + sweep | mg0715 | 06:21:11–06:40:52 | 1181 s / 0.328056 | 30 / 2,689,399,287 | `401aa6180ad213a7f75a54dd07674e4a` |
| 3731052.opbs, descriptive dev sweep | mg0804 | 05:42:28–05:52:45 | 617 s / 0.171389 | 30 / 1,344,907,790 | `d520239db3acc135d660234eda04953f` |

The single finished-job history query agrees with both notified walltimes and log intervals. Combined evaluation actual **1798 / 3600 = 0.499444 node·h**, versus **2.333333 requested**. Completed worker actual is **(62722+1798)/3600 = 17.922222 node·h**, across **31 jobs**. Including rv-code's separately recorded 34-second rerun, known combined actual is **64554/3600 = 17.931667 node·h** across 32 jobs. Do not add the pending analysis request as actual cost. The evaluation estimate deliberately allowed a complete extra forward for capture/replay/serialization; actual whole-job timings were lower and now replace that estimate for accounting and later artifact-analysis sizing.

Archived original indexes and canonical job logs/exit markers under [confirmation/](../../reports/slot-loop-v1/s1b/a1/confirmation/index.json) and [dev-depth-sweep/](../../reports/slot-loop-v1/s1b/a1/dev-depth-sweep/index.json), with byte equality checks. Updated the durable [evaluation ledger](../../reports/slot-loop-v1/s1b/a1/opened-evaluation-jobs.json) to complete in both persistent and repository locations; [finished PBS history](../../reports/slot-loop-v1/s1b/a1/opened-evaluation-history.json) and [sample inventory](../../reports/slot-loop-v1/s1b/a1/opened-evaluation-artifacts.json) preserve paths, sizes, expected digests, durations and pending independent content-audit state. The **4,034,307,077** compressed sample bytes remain at the persistent evaluation outputs; no bulk sample arrays are read on login or copied into Git.

Submitted **3731365.opbs**, `s1b-a1-verdict`, one node/debug-g/**00:20:00**, common immutable science commit **`258f5f5a24eae684f640a9fd62679bb643470a76`**. A separate Python process runs the unchanged frozen entry point:

```text
scripts/analyze_s1b.py --stage CONFIRMATION
  --manifest /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/manifest.lock.json
  --release /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/release.json
  --selection /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/dev-selection/selection.json
  --open-root /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/open-root
  --records /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/confirmation
  --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/confirmation-analysis
```

It checks all confirmation sample hashes, exact checkpoint/provenance/depth/read mapping, registered 2,000-episode IDs, eight-hop coverage, finite q/m/z trajectories, trace agreement and complete paired arm/seed rows. Then it executes the registered **10,000 whole-episode bootstrap resamples**, exact integer/rational interval boundary tests, three Bonferroni-adjusted contrasts, per-seed/hop-1 guards and the same-checkpoint L4−L1 continuation screens, writing frozen `verdict.json`, `verdict.md` and `source_index.json`. Analysis script digest remains **`78f001a232595acce6d38091f0e4895b`**. No wrapper changes its results or schema.

After the frozen process completes, the same compute job separately audits all **30 descriptive dev** files by digest, saved provenance/checkpoint/depth/read/task, IDs/hops, prediction correctness and finite ordered q/m/z shapes. It retains only compact correctness/NLL/ID arrays for the unchanged frozen `descriptive_tables` helper, and writes `descriptive-dev/summary.json` and `audit.json`. It does **not** invoke the inferential decision function on dev, change selection, or feed dev scores into the already written verdict. The wrapper finally records output digests and audit counts in `verdict-job/analysis-audit.json`. All output directories were absent before submission. Full inline payload, job spec and [receipt](../../reports/slot-loop-v1/s1b/a1/verdict-job.json) are archived under `confirmation-analysis/`; payload digest **`071d94cd3cd42d99bffde3dbb9635b37`**. Stdout remains status/count/digest only.

Walltime derives from completed measurements: confirmation's timed prepare/forward total **748.639458 s** and dev's **378.075258 s** leave **(1181−748.639458)+(617−378.075258) = 671.285284 s** of generation, model setup, diagnostics, validation, compression and artifact handling. Conservatively allow that entire measured residual for reading/checking the two saved datasets; add the prior **36 s** frozen DEV job for module/release reconciliation/statistics. Expected **707.285284 s**, ×1.5 = **1060.927927 s**, rounded up to **1200 s / 0.333333 requested node·h**. This fits debug-g and the 16 node·h cap; actual remains pending. Log `/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-a1-verdict-20261030-064703-2584073.log`.

No scientific source/config/checkpoint/selection/open-record change since opening, no additional test run required, no unseen evaluation or mechanism control submitted, no merge/pull/push, sleep or polling. Login work was bounded stdlib JSON/log/metadata/archive inspection; all NPZ decoding, trajectory validation, statistics and metric summaries run on compute. On completion, verify and archive both frozen verdict outputs and descriptive dev evidence, report all verdict tables and actual costs, then STOP verdict for the orchestrator.
