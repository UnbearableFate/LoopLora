# WP-E — completed 2026-10-26

Implemented pinned offline Qwen3-0.6B-Base loading, frozen-backbone adapter attachment/detachment (default layer 14, multiple layers supported), scoped FP32 memory contexts, support writing with all-token/fact-final policies, and both memory-only/context-retained QA protocols. Teacher forcing and incremental greedy decoding fork independent memory/KV snapshots, preserve gradient flow through frozen upper layers, and handle padding and packed-episode resets. Rendering covers every WP-B task and T3 base, exact fact spans, single-token answers/EM, and a versioned 1,024-name asset (776 given names, 248 surnames; deterministically mixed across entity splits).

Implementation commits: `1726c18`, `7379489` on `wp/llm`; master `dbead6f` was merged with orchestrator authorization. WP-A supplied the mixed-precision wrapper correction; WP-E did not edit core or dependencies.

**Validation: 54 tests passed, none skipped, exit 0**, job **3711502.opbs**, Miyabi-G **mg0170**, 19:15:03–19:15:31 JST. Full 28-layer model tests use FP32, plus a BF16-backbone/FP32-adapter backward test. Covered B=0 baseline equality, teacher-forced/incremental logits for both protocols and L=0/1/4, projection/write invariants, padding/reset equivalence, question/L independence, adapter-only gradients, multi-layer attachment, every name token and rendered span, and actual generated episodes. All runtime imports/tests ran on compute through `scripts/submit.py`.

```sh
PYTHONUNBUFFERED=1 scripts/submit.py --name t-llm --queue regular-g \
  --walltime 00:20:00 --worktree . --wait -- -m pytest -q tests/test_llm_*.py
```

[Test log](/work/xg24i002/x10041/LoopLora/logs/pbs/t-llm-20261026-191352-1340422.log). Earlier rendering/generator integration also passed 32 tests in job `3711392.opbs` (`mg0007`).

**Throughput pilot: all 20 cells complete, exit 0**, job **3711529.opbs**, **mg0264**, one NVIDIA GH200, 19:17:43–19:19:26 JST; benchmark elapsed **85.65 s**. Grid: r=16/32, L=0/1/2/4/8, batch=16/64; synthetic support/question/answer lengths=64/16/1, BF16 backbone, FP32 adapters/state, SDPA, two warmups and five CUDA-synchronized repeats per phase. Each training repeat restores identical parameters; only timing/warmup SGD steps ran. No backbone gradients occurred.

| Phase | Median time range | Tokens/s range | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Support encoding + writing | 111.4–246.9 ms | 4,247–31,084 | 1.340 |
| Question reading/scoring | 53.5–95.0 ms | 2,723–16,836 | 1.268 |
| Reader forward/backward/SGD | 91.6–196.5 ms | 1,332–10,727 | 1.987 |

```sh
PYTHONUNBUFFERED=1 scripts/submit.py --name p-llm-tput --queue regular-g \
  --walltime 00:20:00 --worktree . --wait -- -m looplora.llm.bench \
  --config configs/pilot_throughput.yaml \
  --output /work/xg24i002/x10041/LoopLora/runs/pilot/throughput-20261026-101620.json
```

[Raw per-cell JSON, provenance and software versions](/work/xg24i002/x10041/LoopLora/runs/pilot/throughput-20261026-101620.json); [pilot log](/work/xg24i002/x10041/LoopLora/logs/pbs/p-llm-tput-20261026-191701-1378836.log). The pilot recorded clean implementation commit `7379489` and source BLAKE2b digests.

**Choices/deviations:** `regular-g` replaced congested `debug-g` following the orchestrator's guidance; queued smoke `3711443.opbs` was cancelled before execution. Missing labels use reserved single token ` Unknown`; fact-final writes include the line's trailing newline token. Synthetic fixed-length pilot timings measure implementation cost, not task accuracy or long-sequence performance. No required checks were skipped and no WP-E issues remain.
