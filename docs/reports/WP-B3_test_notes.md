# WP-B3 MuSiQue independent test notes

Authored `tests/test_musique.py` against `docs/tasks/WP-B3_contract.md`.
The suite covers raw schema/duplicate validation, immutable records, disjoint
deterministic splits, independently calculated proportional quotas and exact
blake2b-128 ranks/modulo, own distractors, paragraph-order randomization,
real-tokenizer spans/write masks/answer boundaries, official metric handcases,
substitution eligibility/replacement/donors, support removal, JSONL roundtrips,
and confirmation access guards before file access.

The tokenizer fixture reads the literal model/revision/cache pins from
`llm/inject.py` using `ast` and loads `AutoTokenizer` with
`local_files_only=True`; it does not import `llm/inject.py`, load weights, or
download anything. Unicode text, punctuation, numeric and multi-token answers
exercise exact token boundaries. The fixed shuffle sample uses 384 example IDs
for each of 2/3/4 hops; a four-standard-deviation binomial upper allowance avoids
asserting that a fair shuffle must beat its mean. Matching the decomposition
order must occur sometimes, so an implementation that forcibly excludes it also
fails.

Metric expectations were checked against the cached official
`metrics/answer.py`, revision `922ac98f19a201998dbdae6d7f2887a5258dbdeb`:
ASCII punctuation is removed; curly apostrophes are retained; two empty
normalized token lists have F1=1. Alias maxima and repeated-token overlap have
explicit hand-computed expected scores.

Validation: parent reports the initial suite passed **56 tests** on compute
host `mg0846`, PBS job `3712871.opbs`. A subsequent focused regression rejects
substitute gold `USA` for original gold `United States`/alias `U.S.A.` and
filters donor aliases `USA`/`the USA`: the official metric treats those strings
as the original alias despite their distinct literal spelling. This new case
is pending a compute rerun with the parent's production fix.

No runtime checks were executed by this author. The shell is login/control
(`miyabi-g2`, aarch64; no PBS allocation), checked with the `miyabi-pbs-cmd`
context helper. Parent worker runs the suite through
`scripts/submit.py --name t-b3 --walltime 00:15:00 --worktree . --wait -- -m pytest -q tests/test_musique.py`
and record the compute job/result in `docs/reports/WP-B3.md`. No runtime imports,
dependency changes, downloads, or commits were performed by this test author.

Final MuSiQue-only validation: **57 passed in 10.87s**, job `3712970.opbs`,
host `mg0851`, exit 0, at implementation commit `8bc909a`. The normalized-alias
regression is included. Full-corpus preparation also passed at the same commit
(job `3712971.opbs`, `mg0919`, exit 0). Retention validation remains pending.
