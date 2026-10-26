# WP-B3 retention module handoff

Implemented `src/looplora/data/retention.py` and `tests/test_retention.py` only;
main MuSiQue preparation and its CLI belong to the parent worker. No dependencies,
downloads, submissions, or commits were made by this subworker.

## Interface and scoring

- Frozen `MCItem(id, context, choices, label, source_dataset, source_config,
  source_split)`; `label` is zero-based and `choices` is a tuple.
- `prepare_retention(hf_home, output_dir, *, count=500, split_overrides=None,
  allow_smaller=()) -> dict`; `load_retention(name, output_dir) -> tuple[MCItem,...]`.
- Six names: `arc_easy`, `arc_challenge`, `piqa`, `hellaswag`, `winogrande`, `mmlu`.
  The function expects HF_HOME itself, with its `hub/` child.
- Choices are complete continuations: score `context + choice`, masking context.
  ARC/PIQA/MMLU context is `Question: {question}\nAnswer:` and choices start with
  a space. HellaSwag uses its stored `ctx`; choices begin with a space. WinoGrande
  preserves the sentence prefix before `_`; each choice is its option plus the
  complete suffix, so concatenation reconstructs the sentence.
- ARC maps `answerKey` through actual choice labels (including numeric labels);
  WinoGrande maps answers 1/2 to 0/1; remaining datasets use stored indices.

## Reproducibility and rejection rules

Cached `refs/main` selects the revision when present; without it exactly one
snapshot is required. Manifest records the revision, raw relative Parquet paths
and blake2b-128 hashes, source and unique counts, duplicate counts, selected IDs,
configuration names, explicit policy overrides, and output hashes. Selection is
ascending blake2b-128 of each UTF-8 ID, with an ID tie-break. ARC IDs are retained.
HellaSwag uses `ind:blake2b(context,choices,config)` because its raw `ind` repeats
for distinct examples. Missing IDs use canonical context/choices/config content;
MMLU IDs also carry the subject. Identical duplicate rows collapse; conflicting
labels/content for one ID fail. The loader verifies artifact hashes and IDs.

Splits use validation, then dev, then test, based on cached availability. An
insufficient count does not change the split. `allow_smaller` authorizes only
count shortfalls; explicit `split_overrides` are separately recorded. Missing
Parquet shards or incomplete MMLU fail even with `allow_smaller`. MMLU prefers
`all/<split>*.parquet` over subject directories; aggregate rows must cover all
57 standard subjects. Otherwise every subject needs the chosen split cached.
All six suites validate before output writes, so source/policy errors leave no
partially prepared new output directory. No network or model calls are present;
PyArrow is imported lazily during preparation.

## Evidence and validation

Inspected cache revisions: ARC `210d026faf9955653af8916fad021475a3f00453`;
PIQA `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1`;
HellaSwag `218ec52e09a7e7462a5400043bb9a69a41d06b76`;
WinoGrande `01e74176c63542e6b0bcb004dcdea22d94fb67b5`;
MMLU `c30699e8356da336a370243923dbaf21066bb9fe`. ARC-Challenge's card
states 299 validation rows. MMLU initially has only six subject directories;
default preparation correctly rejects it. The orchestrator subsequently approved
the complete validation shard at that revision and all 299 ARC-Challenge
validation items, with no test items. The contracted API required no extension.

Login/control plane: `miyabi-g2`, no allocation. Static checks passed:
`python -m py_compile src/looplora/data/retention.py tests/test_retention.py`
and `git diff --check`. No runtime imports, pytest, or real-data preparation
ran here. Parent must run `tests/test_retention.py` through `scripts/submit.py`
after approved PyArrow installation. Tests cover deterministic order-independent subsets, all six
formats, label mapping, exact continuations, semantic IDs, manifest/raw hashes,
split policies, MMLU completeness, missing shards, duplicates, and corruption.

Parent validation: PyArrow 21.0.0 was installed by the orchestrator and brought
into this branch through master; no worker-side `uv sync` ran. The 25 retention
tests passed in 2.11s on Miyabi-G `mg0981`, job `3713065.opbs`, exit 0, using
`scripts/submit.py --queue regular-g --walltime 00:03:00 --commit c12f7db --wait -- -m pytest -q -o addopts= tests/test_retention.py`.
Real-cache preparation exposed repeated HellaSwag `ind` values: compute probe
`3713105.opbs` on `mg0841` found 10,042 rows but 9,609 unique indices, including
distinct contexts for index 180. The content-qualified identity fixes this without
including labels in IDs or suppressing conflicting annotations. A focused
regression and repeated-index integration fixtures cover this boundary.
The final real-cache result is recorded in `WP-B3.md` and both cache manifests.
