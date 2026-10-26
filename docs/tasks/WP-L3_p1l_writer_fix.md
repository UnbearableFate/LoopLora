# WP-L3 — P1-L writer fix screen (validation only)

Read first: `docs/reports/P1L-diagnosis.md` (WP-L2), `docs/theory.md` Prop. 10 (10.6), `reports/decisions/P1S.md`
§7 (F2: factorized symbolic writers reach 91–92% T1 val accuracy vs 87–88% unfactorized),
`src/looplora/sym/writers.py` (`FactorizedWriter`). Branch/worktree: `wp/llm` (merge `master` first). Report:
`docs/reports/P1L-writer-fix.md`.

## Evidence so far (WP-L2, T1, seed 17, val)
- D1 (rank-32 probes on standardized features, layer-14 o_proj input): tail-name token: tail 97.6%, head 89.8%,
  relation 100%; fact_final: tail 84.2%, head 80.9%; question final: start 99.4%, relation 100%.
- D2: the output path works when z carries the answer: val EM 0.20 / 0.85 / 0.99 at lr 3e-4 / 1e-3 / 3e-3
  after 1500 steps. D3 used only lr ≤ 1e-3.
- D3/D4: no learning in 6400 steps at either input site: loss plateaus at ≈ 7.3 (the name prior, ln 1024 ≈ 6.9),
  candidate accuracy at chance (3.2–3.9% vs 3.6%), key top-1 ≈ chance; at lr 3e-4 query–key cosines are
  ≈ 0.94 for every key (collapsed keys).
- Note: even without addressing, the average of an episode's values could tell the model which 16 tails are
  present (loss ≈ ln 16); the model did not learn that either, so the value path may fail before addressing does.

## Question
Which minimal change makes the LLM writer store and retrieve single-hop facts in the memory-only protocol, and is
the failing link addressing (keys/queries) or the value path?

## Design (T1 as WP-L2: n_rel 4, head sharing 0.3, E = 16, seed 17, r = 32, layer 14, L = 1, batch 32,
## S = 4800 steps, val 1000 episodes at eval seed 4242, EM at steps 1200 / 2400 / 4800; lr 3e-3 unless stated)
Arms (one run each; training-only additions are discarded at evaluation):
- OA oracle addressing (diagnostic): the read returns the queried fact's own written value (replace M q by its
  value code as stored), so only the value and output paths must learn.
- W0 current writer at lr 3e-3.
- W1 input standardization: x̃ = (x − μ) / σ per feature with μ, σ from 256 training episodes at the write and
  read positions (fixed buffers saved with the checkpoint), u = A x̃.
- W2 auxiliary addressing loss: + λ_a · CE over the episode's E keys of ⟨normalize(Q z⁰), k_j⟩ / τ, target = the
  queried fact; λ_a = 1, τ = 0.1.
- W3 auxiliary value loss: + λ_v · CE over the 1024 names of a training-only linear head on each written value
  v_j, target = fact j's tail; λ_v = 1.
- W4 factorized writer (Prop. 10.6 / F2 analog): key code from the module input at the token before the tail
  name, value code from the module input at the tail-name token (separate A_k, A_v as in `FactorizedWriter`;
  one write per fact), read unchanged. First add the D1 probes for head and relation at the pre-tail position.
- W5 = W1 + W2 + W3 (does anything work at all).
Log per arm: main and auxiliary losses every 50 steps; val EM, candidate accuracy, fraction of predictions that are
one of the episode's tails, key top-1 / margin / mean pairwise key cosine, at each evaluation step.

## Decision rule (stated before results)
Order of simplicity: W0 < W1 < W4 < W3 < W2 < W5. The P1-L writer is the simplest arm with val EM ≥ 0.80 at 4800
steps. If none reaches 0.80 but one reaches ≥ 0.30, run that arm once more for the longest ≤ 1 h schedule and apply
the same threshold. If none reaches 0.30, P1-L stops: report the negative result with OA's outcome identifying the
failing link. Auxiliary losses (W2, W3, W5) become part of the P1-L training objective for every memory variant
if chosen, and are reported as supervision beyond the answer.

## Limits
Val only; seed 17; no T2, reader comparison or joint runs. Two runs per node as in D3. Each PBS job ≤ 1 node × 1 h;
WP-L3 total ≤ 4 node·h; record node·h per job. If WP-C5's exact fast write path lands on `master` before you
submit, use it for every arm (same code for all arms). Final line: `WP-L3 done: docs/reports/P1L-writer-fix.md`.
