# WP-B3 — MuSiQue and retention data

**Status: MuSiQue complete; retention preparation awaits orchestrator decisions.**
Code committed on `wp/data` (`15c8bfe`, `8bc909a`) after merging master `0bba591`.

Implemented offline MuSiQue schema validation, deterministic stratified/hash splits,
given-support rendering with exact Qwen token spans, answer EM/F1, same-kind answer
substitution, final-support removal and guarded confirmation loading. Existing
symbolic generators and rendering are unchanged. Retention code and tests are
authored; its Parquet dependency and actual prepared subsets remain pending.

Pinned MuSiQue files match [paper Table 2](https://aclanthology.org/2022.tacl-1.31.pdf):
19,938 train and 2,417 dev. File blake2b-128 values:
train `4c35d71fae509e0d689429941416e36f`, dev `da564a3ccdf6f1d4a9132b6e410e2a18`.

| Split | Before cap | Retained | Dropped | Substitution | Answer budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 18,938 | 18,263 | 675 | 17,949 | 26 |
| val | 1,000 | 965 | 35 | 944 | 19 |
| test | 808 | 736 | 72 | 725 | 24 |
| confirm | 1,609 | 1,492 | 117 | 1,479 | 18 |

Context cap: **640 tokens**, training-context P95 rounded up to 128; model outputs
were never used. Answer budgets are per-split maximum retained gold/alias token
lengths: greedy generation stops at newline/EOS. Val can share single-hop questions
with train (1,449 shared IDs); MuSiQue restricts leakage only across official splits.
Metrics match the [official answer algorithm](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py),
with unrounded per-example scores; the official CLI rounds aggregates to 3 decimals.
Final-support removal is sensitivity-only; closed-book means question only.

**Compute validation:** 57 MuSiQue tests passed in 10.87s, exit 0, job
`3712970.opbs`, host `mg0851`. Full preparation: job `3712971.opbs`, `mg0919`,
exit 0, 60.85s. Both used `scripts/submit.py --queue regular-g --commit 8bc909a --wait`;
test argv `-m pytest -q -o addopts= tests/test_musique.py`; preparation argv
`scripts/prepare_musique.py --section musique`. Logs are under the shared
`LoopLora/logs/pbs/` directory. One incorrectly resolved `HEAD` submission
(`3712964.opbs`) was cancelled before execution and resubmitted with the exact commit.

All IDs, per-hop proportions/length distributions/drop counts, shuffle audits,
substitution coverage and file hashes are in
`/work/xg24i002/x10041/LoopLora/cache/musique/manifest.json`.

**Pending decisions:** approve `pyarrow==21.0.0` binary wheel in the shared venv
(`docs/README.md` requires asking for new dependencies); choose ARC-Challenge
299 validation items versus 500 test items (validation only has 299); approve
the complete MMLU validation shard at the cached revision versus a restricted
medical-subject test suite (only six subjects were initially cached).
Retention tests and real-cache preparation have not run. WP-B3 is not complete.
