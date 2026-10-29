# S1B selection: independent re-run of the frozen dev analysis

This is a registered step, not a code review. `docs/prereg/SLOT-LOOP-v1-S1B.md` line 77 requires it: before confirmation, after dev selects checkpoints, a reviewer who is not the worker re-runs the **frozen** dev analysis. Only after that does the orchestrator write the S1B open record, which opens the confirmation set.

**Reviewer.** rv-code. The worker wk-loop produced the selection; you check it.

## Inputs

All paths are absolute. `R=/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587`.

| item | value |
| --- | --- |
| science commit | `258f5f5a24eae684f640a9fd62679bb643470a76` (fixed by the ready release) |
| ready release | `$R/release.json`, blake2b-128 `d1ccbb7ff4c31fa53abfd7a3cc9ad572`; frozen analysis script `scripts/analyze_s1b.py`, digest `78f001a232595acce6d38091f0e4895b` |
| manifest lock | `$R/manifest.lock.json`, blake2b-128 `2309d5a8ebae91332f32bbd85e5eeb8a` (the release's `lock_blake2b_128`) |
| twelve runs | `$R/main/{M1,M4,S1,S4}/s{17,29,43}/`: `record.json`, `selected.pt` and others (jobs 3730250-3730261) |
| worker's selection | `$R/dev-selection/`: `selection.json` (blake2b-128 `e3f269db39cd9b431d27e0a5eb528d6e`) and `source_index.json`, from job 3730962. That job first ran an audit (`$R/dev-audit/payload.py`, result `$R/dev-audit/main-audit.json`), then the frozen script with the arguments of step 1 and `--out $R/dev-selection` |
| rules | prereg line 89 (lowest mean hop 1-3 dev NLL; a tie within 1e-6 goes to the earlier step; step 0 is not selectable; a selected row whose NLL exceeds its step-0 row is unstable and stops); amendment `docs/prereg/SLOT-LOOP-v1-S1B-A1.md` (N = 18,000, dev every 1,800 steps) |

## Steps

1. Re-run the frozen analysis on compute, from the main checkout `/work/xg24i002/x10041/LoopLora`, with exactly one job:

   ```bash
   /usr/bin/python3 -I scripts/submit.py --name s1b-dev-rerun --queue debug-g --walltime 00:05:00 --commit 258f5f5a24eae684f640a9fd62679bb643470a76 -- scripts/analyze_s1b.py --stage DEV --manifest /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/manifest.lock.json --release /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/release.json --records /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/main --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/dev-selection-rerun
   ```

   The worker's whole job, including its audit, took 36 s. Wait for the job with `qstat -H <id>` at intervals of at least 60 s; do not loop faster.
2. Compare the re-run's files with the worker's. Expect `selection.json` and `source_index.json` to be byte-identical. Explain any difference.
3. Check the selection yourself from the twelve run directories, without the analysis script or project selection helpers:
   - Every record carries the ready release's digest `d1ccbb7ff4c31fa53abfd7a3cc9ad572`, status complete, all finite, N = 18,000 and the released LR (3e-3 in all four arms).
   - The dev schedule is step 0 plus 1,800 to 18,000 every 1,800, and each row has the registered dev count.
   - The selected step follows the rule above, and its NLL does not exceed the step-0 row.
   - The freeze evidence shows the frozen parameters and the support unchanged.
   - The selected export's file digest on disk (blake2b-128 of `selected.pt`, streamed, not deserialized) equals the selected checkpoint digest, and the twelve digests are distinct.
4. Write `/work/xg24i002/x10041/LoopLora/docs/review/static_2026-10-29/s1b-selection-rerun.md` with:
   - the job id and its log;
   - the comparison result;
   - each check, with its evidence;
   - the twelve selected checkpoint digests as `arm/seed: digest`;
   - a verdict: either "selection reproduced" or a list of discrepancies.

## Rules

- **Login node.** Do not import torch or run pytest on the login node. The one PBS job above is the only compute.
- **Reads.** Read dev records only. Do not read or generate confirmation or unseen data.
- **Python.** Use `/usr/bin/python3 -B` (stdlib) or the venv's python for numpy-only reads. Set `PYTHONPYCACHEPREFIX` to `/tmp/claude-30041/-work-xg24i002-x10041-LoopLora/7ae01788-4591-4489-abeb-1056ba03ea57/scratchpad/rv-code`.
- **Hashing.** Use blake2b only.
- **Writes.** Write only your report and the job's output directory. Commit nothing.

Your final reply: the report path and the verdict.
