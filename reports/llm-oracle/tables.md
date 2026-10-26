# WP-C7 oracle writer training smoke

100 train-only steps per variant, batch 64, one GPU process, two CPU prefetch workers; five warmups discarded. Pinned Qwen3-0.6B, layer14/o_proj, r32, E12/Q4, chains (8,4), four relations, head sharing 0.3. Full-vocabulary CE; no support text/LM forward or evaluation.

| Reader | L | Mean ms/step | Median ms/step | Episodes/s | Peak GiB |
| --- | --- | --- | --- | --- | --- |
| M1-L4 | 4 | 189.29 | 189.23 | 338.10 | 6.683 |
| B8 | 4 | 188.48 | 188.75 | 339.55 | 6.683 |
| B4 | 1 | 131.98 | 131.98 | 484.92 | 6.675 |
