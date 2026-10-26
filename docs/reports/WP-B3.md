# WP-B3 — MuSiQue and retention data

Completed on `wp/data` (`15c8bfe`, `8bc909a`, `bf95246`); merged master through
`3dbb597`. Implemented offline MuSiQue parsing, splits, shuffled support rendering,
Qwen paragraph spans, metrics/counterfactuals, confirmation guard, and six retention
subsets. Existing generators/rendering are unchanged; no model scoring was run.

Pinned MuSiQue source counts match [paper Table 2](https://aclanthology.org/2022.tacl-1.31.pdf):
19,938 train and 2,417 dev. File blake2b-128:
train `4c35d71fae509e0d689429941416e36f`; dev `da564a3ccdf6f1d4a9132b6e410e2a18`.

| Split | Before cap | Retained | Dropped | Substitution | Answer budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 18,938 | 18,263 | 675 | 17,949 | 26 |
| val | 1,000 | 965 | 35 | 944 | 19 |
| test | 808 | 736 | 72 | 725 | 24 |
| confirm | 1,609 | 1,492 | 117 | 1,479 | 18 |

Context cap: **640 tokens**, training-context P95 rounded up to 128, independent
of model output. Generation is greedy, stopping at newline/EOS; budgets include
retained gold/alias lengths. Val can share single-hop sub-questions with train
(1,449 shared IDs before filtering); MuSiQue limits leakage across official splits.
Metrics follow the [official answer algorithm](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py),
with unrounded scores versus the official CLI's three-decimal aggregate rounding.
Final-support removal is sensitivity-only: unchanged target, refusal is not an
answer error. Closed-book means question only, without support KV or memory.

**Retention:** validation only. ARC-Challenge **n = 299** (approved exception);
ARC-Easy, PIQA, HellaSwag, WinoGrande and MMLU **n = 500 each**. MMLU uses all
1,531 validation rows across 57 subjects at cached revision, downloaded on login
with `timeout 300`. PyArrow 21.0.0 came from master/shared venv; no `uv sync` ran.
HellaSwag's repeated source indices required content-qualified IDs; a compute
probe established the collision and a regression covers it. All subsets rank
stable IDs by blake2b-128 and record input/output hashes and selected IDs.

**Compute validation:** MuSiQue **57 passed** (`3712970.opbs`, `mg0851`, 10.87s,
commit `8bc909a`); retention **26 passed** (`3713131.opbs`, `mg0841`, 2.02s,
commit `bf95246`). Test argv: `-m pytest -q -o addopts= tests/test_musique.py`
and `tests/test_retention.py`, respectively. Full MuSiQue preparation succeeded
in `3712971.opbs` on `mg0919` (60.85s). Final retention preparation succeeded in
`3713132.opbs` on `mg0856`. All final jobs exited 0 and used:

```text
scripts/submit.py --name <name> --queue regular-g --walltime 00:03:00 --commit <commit> --wait -- <argv>
```

Preparation argv: `scripts/prepare_musique.py --section musique` (`8bc909a`),
then `--section retention --allow-smaller-retention arc_challenge` (`bf95246`).
Detailed [statistics and cached revisions](WP-B3_statistics.md) include per-hop
proportions, lengths, drops, shuffle audits and substitution coverage. Full IDs
and hashes are in `/work/xg24i002/x10041/LoopLora/cache/{musique,retention}/manifest.json`;
job logs are under `LoopLora/logs/pbs/`. No unresolved issues or skipped checks.
