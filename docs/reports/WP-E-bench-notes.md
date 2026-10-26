# WP-E throughput pilot implementation

`src/looplora/llm/bench.py` and `configs/pilot_throughput.yaml` implement the full
20-cell rank × L × batch grid. Default synthetic lengths are support=64,
question=16, answer=1; these timings make no task-quality claims. Each phase uses
2 warmups and 5 synchronized repeats, logging raw seconds, tokens/s, absolute
allocated/reserved CUDA peaks and allocated growth. Reader throughput counts
the actual `question + answer[:-1]` backbone input tokens.

Support writing and question reading use inference `no_grad`; the reader SGD
step preserves activation autograd through frozen upper layers. One untimed
SGD activation step moves zero-initialized B before training warmup. Support is
then re-encoded at those weights and detached. Every measured/warmup training
step starts from identical cloned parameters and a fixed independent snapshot;
parameter restoration is outside timing. SGD has no momentum or saved state.
No training occurs beyond these timing/activation steps. The final step checks
nonzero adapter gradients and absent backbone gradients.

One atomic JSON document is saved before loading and after each phase/cell at
`runs/pilot/throughput-<UTC stamp>.json`. It includes raw config, source-file
BLAKE2b digests, Git revision/status, derived cell seeds, exact pinned model,
software/GPU/PBS metadata, and partial results on failure. A soft 1080-second
cell-admission budget leaves margin within a 20-minute PBS request; completing
the full grid is marked explicitly, never inferred from a partial document.

Validation: focused tests cover complete-grid/repeat constraints, exclusion of
parameter restoration/warmup from timing, synchronization/peak accounting, and
incremental JSON replacement. Runtime tests and the pilot have not been run by
this subagent (login plane: miyabi-g2, no PBS allocation). Primary agent should
include `tests/test_llm_bench.py` in its compute-node tests, then submit:

```
scripts/submit.py --name p-llm-tput --walltime 00:20:00 --worktree . --wait -- -m looplora.llm.bench --config configs/pilot_throughput.yaml
```

No dependencies changed; no full pilot submitted; no commit made.
