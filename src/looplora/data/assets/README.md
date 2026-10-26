# Qwen3 entity-name asset v1

`qwen3_names.v1.json` pins 1024 entity labels and exact leading-space token IDs
for `Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. There are 776 given names and 248
surnames. Entity ID is the zero-based index in the stored arrays; never sort or
reshuffle the selected list at runtime. `Unknown` is excluded from entity labels
and reserved for `INSUFFICIENT` (entity vocabulary size V). Selected labels are
deterministically permuted before assigning entity IDs to avoid aligning the
given-name/surname category with seen/unseen entity pools.

The explicit candidate corpus `name_candidates.v1.json` comes from Faker
v37.12.0 person providers for en_US, en_GB, de_DE, fr_FR, it_IT and en_IN. The
corpus records the exact source URLs and 128-bit blake2b digests. Names are
extracted from literal `first_names_male`, `first_names_female`, and `last_names`
assignments, without importing/executing the provider. Faker's MIT license is
included in `FAKER-LICENSE.txt`. Source files were obtained before submission;
compute jobs use local files and the pinned tokenizer cache only.

Selection is versioned in `build_name_pool.py`: unique ASCII names matching
`[A-Z][a-z]{1,19}`, given names before surnames, lexical order within each group;
require `encode(' ' + name, add_special_tokens=False)` to return one token and
decode exactly to the original text, require distinct token IDs, exclude
`Unknown`, and retain the first 1024. Order these selected records by
`(derive_seed(0, 'name-pool-order', name), name)` using the repository seed helper.
This permutation fixes ID assignment independently of candidate category.
Other spellings, names split into multiple tokens, and labels beyond the
selected 1024 are excluded. The missing-answer
label must independently pass the same single-token round trip.

Rebuild on a compute node without network or Faker installed:

```sh
scripts/submit.py --name names-llm --walltime 00:05:00 --worktree . --wait -- \
  src/looplora/data/assets/build_name_pool.py
```

To regenerate the candidate corpus too, stage the six recorded source files as
`looplora-faker-person-en-us.py` and `looplora-faker-person-<locale>.py` in a shared
directory, then pass `--extract-faker <directory>`. Runtime rendering does not
write files. `validate_name_pool(tokenizer)` verifies every recorded token ID.

Text templates are `relations-text-v1` in `render.py`. A fact is one line ending
in a newline; half-open character/token spans include that newline, and the
final token of the span is the `fact_final` write position. Invalid fact slots
are omitted while spans retain their original slot index. Questions never use
`q_hop_true`; answers contain their leading space. EM compares the greedy first
answer token to the recorded target token, including `Unknown` for missing facts.
