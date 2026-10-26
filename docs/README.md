# LoopLora working conventions (read before any task)

Authoritative inputs: `AGENTS.md`, `plan/recurrent_lora_research_plan_2026-10-26.md` (reference plan, not
authoritative; the orchestrator updates it), `docs/notation.md` (T-01, binding definitions), your task file in
`docs/tasks/`.

## Roles and communication
- Orchestrator (Claude Code) writes task files `docs/tasks/WP-*.md`, reviews and merges branches.
- Workers are long-lived, with a fixed role (`docs/workers.md`); they receive successive task files in that
  role on their own branch/worktree. Each task ends with `docs/reports/<task>.md` (≤ 1 page: what was built
  or run, test command + result with job id, deviations from the task file and why, open issues) and a
  one-line reply pointing to that report. After the reply, stay idle in the same session for the next task.
- Ask only when blocked on a decision the task file and notation cannot resolve; otherwise choose, record the
  choice in the report, and proceed.

## Repository layout
```
src/looplora/            python package (PYTHONPATH=src; not installed)
  core/                  adapters, delta memory, reference semantics, accounting (WP-A)
  data/                  symbolic episode generators (WP-B), text rendering (WP-E)
  sym/                   symbolic mechanism experiments (later WP)
  llm/                   Qwen3 injection and protocols (WP-E)
  infra/                 seeds, hashing, run records (orchestrator)
tests/                   pytest; one file per concern
configs/                 YAML experiment configs
scripts/submit.py        the only way to submit compute jobs (enforces caps)
scripts/pbs/job.pbs      generic job script used by submit.py
reports/                 experiment_registry, decisions/, figures/
docs/                    notation, tasks, worker reports
```

## Execution rules (Miyabi-G)
- Login node (`miyabi-g*`): edit files, git, `uv` from wheels, qsub/qstat only. Never import torch, run
  pytest, train or evaluate on the login node.
- Run tests on a compute node, blocking until done (debug-g, usually starts in < 1 min):
  `scripts/submit.py --name t-<wp> --walltime 00:15:00 --worktree . --wait -- -m pytest -q tests/<file>`
  It prints the log tail and returns the job's exit code. Keep test runs short (< 10 min).
- Environment: shared venv `/work/xg24i002/x10041/LoopLora/.venv` (python 3.13, torch 2.14.0+cu132,
  transformers 4.57.6). Do not modify `pyproject.toml`/`uv.lock`; ask the orchestrator for new dependencies.
- Resource cap (user, current stage; "experiment" = one PBS job, study totals may be relaxed, 2026-10-27): one job ≤ 8 nodes × 1 h; up to 8 nodes × 2 h when results
  already obtained support scaling up (`submit.py --scale-up <decision/prereg>`; the orchestrator decides). `submit.py` enforces it.
- HF models: `HF_HOME=/work/xg24i002/x10041/hf_home`, offline in jobs (`HF_HUB_OFFLINE=1`).

## Code rules
- Do not use SHA-256 anywhere (AGENTS.md). For content hashes use `hashlib.blake2b(digest_size=16)`.
- Seeds: derive every RNG seed from (base seed, purpose label) via `looplora.infra.seeds.derive_seed`.
- Reference implementations are FP32/FP64, functional (no in-place updates of tensors autograd saved).
- Keep modules small and typed; comments only where intent is not obvious; no speculative options.
- Tests check behaviour and invariants from `docs/notation.md` and plan §10.5, not implementation details.

## Git
- Each worker works in its own worktree `/work/xg24i002/x10041/LoopLora-wt/<wp>` on branch `wp/<wp>`.
- Commit your own paths only, small logical commits, message `WP-<id>: <summary>`. Do not merge, rebase or
  push other branches; the orchestrator merges into `master`.
