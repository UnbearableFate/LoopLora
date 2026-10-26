# LoopLora

Research code for *Iterative Low-Rank Adaptation under a Fixed Context-Memory Budget*
(plan: `plan/recurrent_lora_research_plan_2026-10-26.md`). Working conventions: `docs/README.md`;
binding definitions: `docs/notation.md`.

Environment: `uv sync` on a Miyabi-G login node (torch 2.14.0+cu132, aarch64). All tests and experiments run
on compute nodes via `scripts/submit.py` (per-experiment cap: ≤ 8 nodes × 1 h).
