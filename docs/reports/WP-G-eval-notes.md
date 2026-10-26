# WP-G evaluation implementation notes

Owned files: `src/looplora/llm/evaluate.py`, `tests/test_llm_evaluate.py`.

`evaluate(experiment, config, out)` writes flat `samples.npz` and `metrics.json`
and returns metrics plus timing. Full-vocabulary first-token EM is primary
(`prediction/correct` and `full_prediction/full_correct`); candidate token
accuracy is secondary. The table and depth statistics are reproducible from
the sample archive. `chance_accuracy` is the mean inverse candidate count and
applies to the secondary metric. Statistics directly reuse `sym.stats`;
evaluation query stratification and tensor hashing reuse `sym.evaluate`.

Every generated episode has one fixed query per requested hop. Validation and
screening use their named split starting at 0; confirmation uses `test` starting
at 100000 and requires `allow_confirmation=true` plus `smoke=false`. This
implementation does not authorize screening or confirmation execution.
B4 has only L=1. All protocols reuse the same generated episode set.

Conditions include protocol (`memory_only/original`, etc.) so paired statistics
cannot silently combine protocols. Replace forks one support per source query,
retains original episode/query/hop labels and token answers/predictions, and
scores its new token answer. Swap substitutes donor M and actual write counts
but retains the source questions, answers and candidate set. In context-retained
mode it deliberately retains the source KV. `memory_episode_id` identifies
the donor (or the edited support for replacement).

`metrics.invariant_hashes` has one record per condition/batch with exact keys
`split,protocol,condition,episode_start,x,u,M,write_count`, no L. The runner can
compare these across frozen-reader variants in memory-only mode.
`memory_hashes` additionally records each L, `batch_start`, `attention_mask`,
`backbone_kv`, and `memory` (same digest as invariant M). Every score captures x,
computes u with the installed adapter's FP32 A, and asserts x/u/M/write counts
and snapshot KV unchanged at every L. B0 hashes an empty u projection.
Timing includes synchronized support preparation, scoring per L, total duration
and per protocol/split duration (including configured counterfactuals).

Validation prepared: unit coverage for archive aggregation, primary/secondary
metric distinction, counterfactual labels and source KV preservation, pairing,
B4, confirmation guards and deliberately violated Gate 0. Pinned-backbone tests
cover teacher-forced versus greedy first-token EM in both protocols with a
nonzero adapter, true-model Gate 0, and B0 memory-only support independence and
accuracy within a three-standard-error chance bound on 64 validation episodes.

Static validation: both files parse via Python stdlib AST. Runtime tests have
not been submitted by this subagent; parent coordinates compute validation.
Plane established with the Miyabi PBS skill: login/control `miyabi-g2`, aarch64,
no allocation. No Torch/pytest/Transformers import or workload ran on login.
No core/data/sym/dependency edits and no commits.
