# WP-D — Symbolic mechanism pipeline (plan T-04, T-05; §5.2, §7.1–§7.2, §8.4, §9)

Read first: `docs/README.md`, `docs/notation.md`, `docs/tasks/WP-A_core.md` and `WP-B_data.md` (interfaces you
use; both are being built in parallel — code against the interfaces, integrate when the orchestrator says they
are merged into `master`). Branch/worktree: `wp/sym` at `/work/xg24i002/x10041/LoopLora-wt/sym`.
Report: `docs/reports/WP-D.md`.

## Question this pipeline must answer (plan §15)
With one module, a fixed input projection and writer, and the same memory snapshot, does the evolving-query
multi-read (M1) beat the single read (B4), the fixed-query loop (B5) and the no-memory loop (B3)?

## Symbolic backbone (`src/looplora/sym/backbone.py`, frozen, from a fixed `backbone_seed`)
- Embeddings N(0, 1/d_e), d_e = 64: entities (V = 1024, + one row for `INSUFFICIENT`), relations, hop
  (index 0 = "not stated").
- Fact token `x = [E[head]; Rel[rel]; E[tail]]` (d_in = 192). Query token `x = [E[start]; Rel[rel] +
  Hop[hop_given]; 0]`. Base output is zero (`W = 0`: the symbolic base model has no task knowledge), so
  `y = α B z`, d_out = d_e, logits `= E y`.
- Readout restricted to the episode's candidate set (entities occurring in its facts, plus `INSUFFICIENT` when the
  task can produce it) is the primary metric; full-vocabulary accuracy is reported as secondary.

## Writer conditions (`src/looplora/sym/writers.py`)
| id | keys/values | trainable |
| --- | --- | --- |
| `oracle_random` | fixed random unit codes `c[e] ∈ R^{r_m}`; `k = c[head]`, `v = c[tail]`; query `u = c[start]`; readout logits `c[j]·y` (d_out = r_m) | none (λ = β = 1 fixed, `core.memory.delta_update`) |
| `oracle_orth` | as above with per-episode orthonormal codes (only when #entities ≤ r_m) | none |
| `t1_frozen` | adapter writer (A, K, V, gates) pretrained on `t1` together with a V2 L=1 reader, then frozen; the T2 reader is re-initialised | reader only |
| `joint` | writer and reader trained together on the T2 task (mixed L) | all |
Memory is written from the support facts only (memory-QA protocol, notation §5); queries never write.

## Training (`src/looplora/sym/train.py`)
- Reader variants: B3 `loop`, B4 `iterq` trained and evaluated at L=1, B5 `fixedq`, M1 `iterq`; also B6/B7 as
  `iterq` L=1 with larger r / r_mem, and B8 `unshared` (fixed L=4) when configured.
- Depth: fixed-L smokes (L=1, L=4); main runs sample L ~ Uniform{1,2,4} per batch. Record mean L.
- AdamW, grad clip 1.0, weight decay 0.01, batch of episodes, constant LR after a short warmup; every field in the
  YAML config. All randomness from `derive_seed(seed, purpose)`.
- `t1_frozen`: stage 1 trains the writer on `t1`, saves it; stage 2 loads it frozen for every reader variant
  (identical writer weights ⇒ identical M for every variant and every L on the same episodes).

## Evaluation (`src/looplora/sym/evaluate.py`)
- Depth sweep on one checkpoint: L ∈ {0, 1, 2, 3, 4, 6, 8, 12, 16} on val/test/test_unseen, test episodes with
  one query per hop 1..8 (fixed eval episode sets, generated once per spec and seed).
- Per-sample records (episode id, query id, hop, L, prediction, answer, correct, candidate-set size) to
  `samples.npz`; aggregated hop × L table to `metrics.json`; a blake2b hash of each eval memory tensor batch,
  asserted identical across L (Gate 0: changing L changes neither x/u/M nor the write count).
- Counterfactual evals on test: `replace_key_fact` (does the prediction move to the new answer?), `swap_memory`
  (accuracy against the original answers should fall to chance).
- Diagnostics per L: ‖z‖, ‖M‖_F, retrieval cosine of the first read to the true value code (oracle and learned).

## Statistics (`src/looplora/sym/stats.py`)
Paired cluster bootstrap by episode (≥ 2000 resamples, fixed seed): accuracy differences with 95% CI for
(same checkpoint) L=4 − L=1, L=8 − L=4 and (across variants on identical episodes) M1 − B4, M1 − B5, M1 − B3.

## Runner and records
- `python -m looplora.sym.run --config configs/p1s/<file>.yaml --out runs/p1s/<run_id>`; a `--suite` mode runs a
  list of configs as parallel processes on one GPU (`--procs N`), each with its own output dir.
- Each run dir: `config.json` (with blake2b config hash and git commit from `LL_SOURCE_ROOT`/git), `train_log.jsonl`,
  `metrics.json`, `samples.npz`, `checkpoint.pt`, `timing.json` (writer pass, reader pass per L, train step),
  `record.json` (plan §9.4 fields). `scripts/collect_registry.py` builds `reports/experiment_registry.csv` from
  all `record.json` files.
- Figures (`python -m looplora.sym.figures`): accuracy–L curves per hop for one checkpoint, hop × L heatmap,
  paired differences; PNG under `reports/figures/p1s/`.

## Tests (`tests/test_sym_*.py`, compute node)
Backbone determinism; oracle writer with orthonormal codes and λ=β=1 retrieves every stored value exactly;
memory hash identical across L; per-sample records reproduce `metrics.json`; bootstrap CI covers a known effect
on synthetic data; a 300-step T1 smoke on GPU gets clearly above chance.

## Deliverable boundary
Build and validate the pipeline, then run only: (1) the test suite; (2) one smoke suite (T1 + T2 at 10% of the
default steps, all variants, one seed) with timing. Do not run pilots or the screening suite — the orchestrator
pre-registers those. Report measured times per run to size jobs (≤ 8 nodes × 1 h per experiment).
