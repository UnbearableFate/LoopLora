# WP-B — Symbolic relation-episode generators (plan T-03, §7.1–§7.4)

Read first: `docs/README.md`, plan §7.1–§7.4 and §9.2. Branch/worktree: `wp/data` at
`/work/xg24i002/x10041/LoopLora-wt/data`. Report: `docs/reports/WP-B.md`.

## Purpose
Episodes for the symbolic mechanism stage and, after text rendering (WP-E), the LLM stage. An episode is a set
of facts `(head, rel, tail)` presented as context, plus queries with a unique answer. The generator is numpy
only (no torch), deterministic, and splittable by episode.

## Interface (`src/looplora/data/relations.py`)
- Global ids: entities `0..V-1` (default V = 1024); relations `0..n_rel-1`; `INSUFFICIENT = V` (answer id for
  "information missing"). Entity pools: `seen = [0, 768)` for train/val/test, `unseen = [768, 1024)` for split
  `test_unseen`. Pools and V are spec fields.
- `TaskSpec` (frozen dataclass, hashable via `looplora.infra.seeds.stable_hash`) with the task name and knobs
  below; `generate(spec, split, start, count, base_seed) -> EpisodeArrays`. Episode `i` of a split depends only
  on `(base_seed, spec hash, split, i)` via `derive_seed`, so any slice can be generated independently.
- `EpisodeArrays` (numpy, fixed shapes per spec): `facts int32[N,E,3]` in presentation order, `fact_valid
  bool[N,E]`, `q_start int32[N,Q]`, `q_rel int32[N,Q]`, `q_hop_given int32[N,Q]` (−1 if the query does not
  state h), `q_hop_true int32[N,Q]` (stratification/oracle only), `answer int32[N,Q]`, `episode_seed
  int64[N]`, plus `meta` (generator version, spec, split, base seed).
- Throughput: report episodes/s per process for each task; aim for ≥ 20k/s for T2 at E = 24.

## Tasks (all keep the total number of facts E constant within a spec, independent of query hop)
| name | construction | query → answer |
| --- | --- | --- |
| `t1` | E facts with distinct `(head, rel)` keys; knob `head_sharing` ∈ [0,1] reuses existing heads with another relation (key similarity) | `(head, rel)` → tail; h = 1 |
| `t2_chain_end` | relation 0 only; disjoint chains with lengths `chain_lengths` (sum = E), all nodes distinct, no cycles; facts shuffled | start at distance h from its chain's terminal (h sampled from `hops`, chain chosen among those long enough) → the terminal; `q_hop_given = −1` |
| `t2_walk` | as above, chains may continue past the answer | start at position p, h stated in query (`q_hop_given = h`) → node p+h |
| `t2_typed` | `t2_chain_end` chains in relation 0 (sum = E − n_distract) plus `n_distract` edges in relations 1..n_rel−1 whose heads are chain nodes (same-head keys) | as `t2_chain_end`, following relation 0 only |
| `t3_update` | `t1` or `t2_chain_end` base; `n_updates` keys appear twice with different tails, the later one valid; temporal order of each pair kept, rest shuffled | answer uses the latest fact; queries over updated keys at rate `update_query_rate` |
| `t3_missing` | `t2_walk` or `t1` base with one required fact removed and replaced by a distractor fact (E constant) at rate `missing_rate` | → `INSUFFICIENT` |

## Counterfactuals (`counterfactual(arrays, kind, seed) -> (arrays_cf, answer_cf)`)
- `replace_key_fact`: rewire one edge on the query's path to a node that changes the answer; recompute; the new
  answer must differ from the old one and stay unique (no cycles).
- `remove_required_fact`: delete one path edge, add a distractor edge elsewhere (E constant). For `t1`/`t2_walk`
  the label becomes `INSUFFICIENT`; for `t2_chain_end` the answer becomes the cut node (sensitivity check).
- `swap_memory`: pair each episode's queries with another episode's facts; return the pairing (scored as
  accuracy against the original answers, expected to fall to chance).

## Tests (`tests/test_relations.py`; run on compute via `scripts/submit.py … -- -m pytest -q tests/test_relations.py`)
- An independent brute-force solver (dict walk) reproduces every answer for every task; answers are unique.
- Same `(base_seed, spec, split, i)` ⇒ identical episode; different splits never share an episode seed; no
  identical fact set across splits (blake2b signature of sorted facts); `test_unseen` uses only unseen entities.
- Fact count E and node distinctness hold for every hop; empirical hop distribution matches the spec.
- Chain tasks have no cycles; in `t2_chain_end` the start is exactly h steps from a node without an outgoing
  relation-0 edge; in `t2_typed` distractors never create a relation-0 edge.
- `t3_update`: the later fact wins and its pair order is preserved. Counterfactual answers recomputed by the
  brute-force solver match.

## Constraints
No torch; no SHA-256 (use blake2b); no files written at import; generation must not depend on global RNG state.
