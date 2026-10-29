# S1A selection: independent re-run of the frozen dev analysis

This is a registered step, not a code review. `docs/prereg/SLOT-LOOP-v1.md` (line 125 and open question 4) requires it: at the `S1A selected` stop, a reviewer who is not the worker re-runs the frozen analysis script on the S1A dev records and checks the selection. Only after that does the orchestrator write `s1a-open.json`, which opens the confirmation set.

**Reviewer.** rv-code. The worker wk-slot produced the selection; you check it.

## Inputs

All paths are absolute. `R=/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1`.

| item | value |
| --- | --- |
| science commit | `1523b19c7949673ddcd1d89cc1e5fe518736805c` (fixed by the ready release) |
| ready release | `$R/s1a-release.json`, blake2b-128 `523d11d57a08a2491893ebe7cedc86a7`; frozen analysis script digest `843d2ff4d1a03224ccd16e68720ba344` |
| six runs | `$R/s1a/{matrix_L1,soft_slot_L1}/s{17,29,43}/`: `record.json`, `training.json`, `selected.pt` and others |
| worker's selection | `$R/s1a-selection/`: `s1a-selection.json`, `s1a-selection.md`, `source_index.json` (job 3729237) |
| rules | prereg "Checkpoints and selection" (lowest dev NLL; a tie within 1e-6 goes to the earlier step; step 0 is not selectable) and §7.4 (the ablation trigger) |

## Steps

1. Re-run the frozen analysis on compute, from the main checkout `/work/xg24i002/x10041/LoopLora`, with exactly one job:

   ```bash
   /usr/bin/python3 -I scripts/submit.py --name s1a-dev-rerun --queue debug-g --walltime 00:01:00 --commit 1523b19c7949673ddcd1d89cc1e5fe518736805c -- scripts/analyze_slot_study.py --stage S1A-DEV --manifest /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/manifest.lock.json --records /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a-selection-rerun
   ```

   It takes about 10 s once it starts. Wait for it with `qstat -H <id>` at intervals of at least 60 s; do not loop faster.
2. Compare the re-run's three files with the worker's. Expect `s1a-selection.json` and `s1a-selection.md` to be byte-identical. Explain any difference, for example an output path in `source_index.json`.
3. Check the selection yourself from the six run directories, without the analysis script:
   - Every record carries the ready release's digest `523d11d57a08a2491893ebe7cedc86a7`. Each run has N = 6000 and its released LR: matrix_L1 1e-3, soft_slot_L1 3e-3.
   - The dev rows in each record match the selection tables.
   - The selected step follows the rule.
   - The selected export's file digest on disk (blake2b-128 of `selected.pt`) equals the selected checkpoint digest.
   - The trigger outcome (`no_trigger`, dev gap 105 correct over three seeds) follows §7.4.
4. Write `/work/xg24i002/x10041/LoopLora/docs/review/static_2026-10-29/s1a-selection-rerun.md` with:
   - the job id and its log;
   - the comparison result;
   - each check, with its evidence;
   - the six selected checkpoint digests as `arm/seed: digest`;
   - a verdict: either "selection reproduced" or a list of discrepancies.

## Rules

- **Login node.** Do not import torch or run pytest on the login node. The one PBS job above is the only compute.
- **Reads.** Read dev records only. Do not read or generate confirmation or unseen data.
- **Python.** Use `/usr/bin/python3 -B` (stdlib) or the venv's python for numpy-only reads. Set `PYTHONPYCACHEPREFIX` to `/tmp/claude-30041/-work-xg24i002-x10041-LoopLora/7ae01788-4591-4489-abeb-1056ba03ea57/scratchpad/rv-code`.
- **Hashing.** Use blake2b only.
- **Writes.** Write only your report and the job's output directory. Commit nothing.

Your final reply: the report path and the verdict.
