# S1A frozen dev-analysis rerun

**Verdict: selection reproduced.** No discrepancies.

Reviewer: rv-code, independent of selection worker wk-slot. Reviewed on 2026-10-30 JST under [the rerun brief](s1a_selection_rerun_brief.md), from clean master `f60cdb5d2373477c3f7de2e1d2f704e7c3dc799d`.

All paths below use `R = /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1`. The worker's output is `R/s1a-selection/` (job 3729237); this rerun is `R/s1a-selection-rerun/`.

**Compute execution.** Exactly one job was submitted from the main checkout with the brief's command:

```bash
/usr/bin/python3 -I scripts/submit.py --name s1a-dev-rerun --queue debug-g --walltime 00:01:00 --commit 1523b19c7949673ddcd1d89cc1e5fe518736805c -- scripts/analyze_slot_study.py --stage S1A-DEV --manifest /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/manifest.lock.json --records /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a-selection-rerun
```

PBS job **3729280.opbs**, one node, `debug-g`, requested walltime 00:01:00; compute host **mg0005**. Evidence: [job log](/work/xg24i002/x10041/LoopLora/logs/pbs/s1a-dev-rerun-20261030-000730-3798819.log), [exit marker](/work/xg24i002/x10041/LoopLora/logs/pbs/s1a-dev-rerun-20261030-000730-3798819.log.exit), and [submission spec](/work/xg24i002/x10041/LoopLora/logs/pbs/specs/s1a-dev-rerun-20261030-000730-3798819.json). The spec and log both identify science commit `1523b19c7949673ddcd1d89cc1e5fe518736805c` and the prescribed S1A-DEV arguments. The wrapper's compute-allocation guard precedes application execution.

Submitted at 00:07:30 JST. The only history poll, `qstat -H 3729280`, at approximately 00:09:14 JST returned `FINISH`, start 00:08:06 and elapsed 00:00:07. The application log spans 00:08:09–00:08:15 JST and records `stage S1A-DEV`, `outcome no_trigger`, `runs 6`, and `exit=0`; the exit marker contains `0`. Scheduler completion is corroborated by successful output readback.

**Frozen inputs.** Independently computed blake2b-128 values:

| Input | Verified digest |
| --- | --- |
| [Ready release](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a-release.json), status `ready` | `523d11d57a08a2491893ebe7cedc86a7` |
| `scripts/analyze_slot_study.py` bytes from the science commit via `git show` | `843d2ff4d1a03224ccd16e68720ba344` |
| [Manifest lock](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/manifest.lock.json) | `5af7a9872827630280fa34218ffabde6` |
| Registered preregistration | `56e8c611c01e0ef9b7ae32db97b0e334` |

The script, manifest and preregistration digests equal the release's respective fields. The rerun's source index also records the same frozen script digest and the same manifest digest.

**Comparison with the worker.** Direct byte comparisons, followed by blake2b-128 hashing, passed for all three files:

| File | Bytes in each copy | Comparison | blake2b-128 |
| --- | ---: | --- | --- |
| [s1a-selection.json](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a-selection-rerun/s1a-selection.json) | 14315 | byte-identical | `b6e5d452089045e27b22d4d94a57aacc` |
| [s1a-selection.md](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a-selection-rerun/s1a-selection.md) | 3878 | byte-identical | `843d0657f90fab3ed6e8a4711bee8ea5` |
| [source_index.json](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a-selection-rerun/source_index.json) | 821 | byte-identical | `d711832ed2bba6147bc40236904ca154` |

`source_index.json` is also byte-identical: it contains input paths and digests, the script name/digest, and stage, but no output-directory field. There is therefore no path difference to explain. Its seven input digests (manifest plus six `record.json` files) were independently checked against disk.

**Independent checks from the six run directories.** These checks used separate inline Python with stdlib and NumPy-only artifact reads; they did not import or call the analysis script or project selection helpers. Small JSON/JSONL/NPZ files were checked directly, and `selected.pt` was streamed through blake2b-128 without deserializing it.

1. Every `R/s1a/<arm>/s<seed>/record.json` has the ready-release digest `523d11d57a08a2491893ebe7cedc86a7`, science commit `1523b19c7949673ddcd1d89cc1e5fe518736805c`, `source_dirty=false`, `status=complete`, and `confirmation_accessed=false`. Each complete config equals the corresponding release `science_configs[arm/seed]`; its recorded config hash equals the release hash.
2. `record.steps`, `record.config.steps`, `training.json.steps`, and the release all give **N = 6000**. Both the run config and `training.json.lr` equal the released LR: **matrix_L1 = 0.001**, **soft_slot_L1 = 0.003**, for seeds 17, 29 and 43.
3. All six dev histories have exactly 11 rows, steps `0, 600, …, 6000`, with `n=1000`. Step 0 is nonselectable; the ten subsequent rows are selectable. All 66 rows agree exactly between `record.json` and `training.json`. Their `step`, `selectable`, `correct`, `em`, `n`, and `nll` fields equal the worker/rerun JSON tables. Every Markdown table row also matches independently formatted record values, including its displayed rounding. Every EM equals `correct/n`; all NLLs are finite. All six records share dev-ID digest `6bf2f42455e065ce309af6cd7fc79541`.
4. Following [the registered selection rule](/work/xg24i002/x10041/LoopLora/docs/prereg/SLOT-LOOP-v1.md:111), I excluded step 0, found the minimum NLL among the remaining ten rows, and chose the earliest step within 1e-6 of that minimum. The result equals `selected_step` in each record, training summary, and selection table. Each run has a unique minimum within this tolerance. The smallest gap to a runner-up is 0.00008904017385613638 (matrix_L1/s29), greater than 1e-6.
5. Each on-disk `selected.pt` digest equals `selected_checkpoint_blake2b_128` in its `record.json`, `training.json`, and both selections. The six independently hashed exports are listed below.
6. Completion and stability for the ablation rule were checked from the underlying run artifacts: every run is complete at 6000 steps; every selected NLL is below its own step-0 NLL; all `all_finite` flags are true. In each `train_steps.npz`, both `loss` and `grad_norm` have shape `(6000,)` and all values are finite: 72,000 finite values across six runs. Each `train_log.jsonl` contains 121 rows (step 1 and every 50 steps through 6000); every logged loss and gradient norm is finite and exactly matches its indexed NPZ value. The full-array maximum gradient norm also matches the training summary, record, and selection.

| Run (record evidence) | N | LR | Independently selected step | Step-0 NLL | Selected NLL | Selected correct / 1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [matrix_L1/s17](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a/matrix_L1/s17/record.json) | 6000 | 0.001 | 5400 | 1.421839259186323 | 0.866040755549865 | 834 |
| [matrix_L1/s29](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a/matrix_L1/s29/record.json) | 6000 | 0.001 | 6000 | 1.421839259186323 | 0.894967002192947 | 833 |
| [matrix_L1/s43](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a/matrix_L1/s43/record.json) | 6000 | 0.001 | 5400 | 1.421839259186323 | 0.887063703588373 | 838 |
| [soft_slot_L1/s17](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a/soft_slot_L1/s17/record.json) | 6000 | 0.003 | 5400 | 1.282060354058383 | 0.736074387708088 | 867 |
| [soft_slot_L1/s29](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a/soft_slot_L1/s29/record.json) | 6000 | 0.003 | 6000 | 1.282060354058383 | 0.760171110831361 | 868 |
| [soft_slot_L1/s43](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a/soft_slot_L1/s43/record.json) | 6000 | 0.003 | 4200 | 1.282060354058383 | 0.774393162836116 | 875 |

**Selected checkpoint digests.** blake2b-128 of the actual `selected.pt` file in each run directory:

```text
matrix_L1/s17: 00a1535965a0f2813b4a1b552b6e2ee2
matrix_L1/s29: b44f13443a4ce44e3d22d9f1ac77685d
matrix_L1/s43: 07e1ab7460f12615735fccb180d2dc3b
soft_slot_L1/s17: d71e06da709ee4c588b5f008d68c0d15
soft_slot_L1/s29: 136226b18beb45c5ca218451e9c13ea1
soft_slot_L1/s43: 52d43da424025388a8756bacb03f3058
```

**§7.4 trigger.** The [registered rule](/work/xg24i002/x10041/LoopLora/docs/prereg/SLOT-LOOP-v1.md:122) requires completed, finite, stable runs and a mean selected-checkpoint dev EM difference strictly below +3.0 pp. Completion, finiteness and stability pass as above. Independently subtracting the selected dev counts gives:

| Seed | Matrix correct | SoftSlot correct | SoftSlot − Matrix | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| 17 | 834 | 867 | 33 | +3.30 |
| 29 | 833 | 868 | 35 | +3.50 |
| 43 | 838 | 875 | 37 | +3.70 |
| Total / mean | 2505 | 2610 | 105 | +3.50 |

`100 × 105 / (3 × 1000) = +3.5 pp`. The below-3.0-pp condition would require fewer than 90 total additional correct answers; 105 does not satisfy it. Therefore **no_trigger** is the registered outcome, with no instability stop and no ablation triggered. This matches both selections and their empty `reasons` lists.

**Scope.** The login/control host was `miyabi-g2`; only the prescribed PBS job ran application compute. Login checks used bounded file reads, stdlib, or the venv's NumPy-only reads, with `PYTHONPYCACHEPREFIX=/tmp/claude-30041/-work-xg24i002-x10041-LoopLora/7ae01788-4591-4489-abeb-1056ba03ea57/scratchpad/rv-code` set and `-B` for independent reads. No login-side torch import or pytest run occurred. No confirmation or unseen dataset was read or generated; the manifest was hashed as an input artifact. No checkpoint reload or new model evaluation was performed: export identity here is verified by file digest. This review determines dev selection and its trigger only. The requested submission wrapper produced its normal PBS spec/log/ledger artifacts; the only review-authored file is this report. Nothing was committed, and no opening record was written.

