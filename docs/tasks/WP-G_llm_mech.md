# WP-G — P1-L pipeline: single module in Qwen3-0.6B-Base (plan T-04/T-05 for the LLM stage)

Read first: `docs/README.md`, `docs/notation.md`, `docs/prereg/P1L.md` (design), `docs/prereg/P1S.md`
(endpoints reused), `docs/reports/WP-E.md` (your protocols), `docs/tasks/WP-D_sym.md` (the symbolic pipeline
whose structure, records and statistics this mirrors; reuse `looplora.sym.stats` and the record format).
Branch/worktree: `wp/llm` (continue). Report: `docs/reports/WP-G.md`.

## Deliverables (`src/looplora/llm/`)
1. `train.py` — three stages, all in the **memory-only** protocol unless stated:
   - `writer`: rendered `t1` (n_rel = 4, head_sharing = 0.3, E from config), one `iterq` module at layer 14
     (r = 32), L = 1, all adapter params trainable; saves the writer (A, K, V, gates).
   - `reader`: loads the writer frozen (A, K, V, gates), re-initialises R, U, Q, B from `derive_seed`, trains on
     rendered `t2_chain_end` (hops 1–3), L ~ U{1,2,4} per batch (B4: L = 1); variants M1, M1-lin, B5, B3, B4.
   - `joint`: all adapter params from scratch on `t2_chain_end`, mixed L.
   - `lora_ctx` baseline: B1 LoRA (same position, same rank, same steps) trained in the **context-retained**
     protocol; B0 = no adapter (both protocols) needs no training.
   Answer loss = CE on the single answer token. Several queries per episode fork one support snapshot.
   Support encoding for memory-only may stop at layer 14 (x for writes is fixed by layers ≤ 14); must be
   verified equal to the full-forward path in a test.
2. `evaluate.py` — depth sweep L ∈ {0,1,2,3,4,6,8} on fixed val / test [0,2000) / confirm [100000,102000)
   episode sets (one query per hop 1–8, same as P1-S); EM = greedy full-vocabulary argmax equals the answer
   token (primary); candidate-restricted accuracy (episode's name tokens) secondary; per-sample `samples.npz`
   and `metrics.json` in the P1-S format; blake2b hashes of the support snapshots and question-token adapter
   inputs per batch, asserted identical across L (and across variants sharing a frozen writer); counterfactual
   `replace_key_fact` and `swap_memory`; context-retained evaluation for B0, B1 and M1.
3. `run.py` — `python -m looplora.llm.run --config configs/p1l/<file>.yaml --out runs/p1l/<run_id>`; `--suite` runs a
   config list sequentially in one process (model loaded once) or as N parallel processes on one GPU, whichever
   is faster (measure). Each run dir has the same files as P1-S (`config.json`, `train_log.jsonl`, `metrics.json`,
   `samples.npz`, adapter `checkpoint.pt`, `timing.json`, `record.json`).
4. `configs/p1l/` — smoke configs (10% of default steps) for every stage and variant.

## Tests (compute node)
Truncated support encoding equals full forward; frozen-writer stages leave writer params bitwise unchanged;
snapshot/activation hashes identical across L; teacher-forced EM equals greedy decode EM for one-token answers;
B0 memory-only accuracy is at chance on a small set; per-sample records reproduce `metrics.json`.

## Deliverable boundary
Build, test, and run the smoke suite (all stages and variants, 10% steps, one seed) with timing; report measured
time per 1k steps and per eval set so jobs can be sized (≤ 8 nodes × 1 h each). Do not run pilots or screening;
the orchestrator finalises `docs/prereg/P1L.md` first.
