# WP-C7 oracle writer training smoke

10 train-only steps per variant, batch 64, one GPU process, two CPU prefetch workers; five warmups discarded. Pinned Qwen3-0.6B, layer14/o_proj, r32, E12/Q4, chains (8,4), four relations, head sharing 0.3. Full-vocabulary CE; no support text/LM forward or evaluation.

| Reader | L | Mean ms/step | Median ms/step | Episodes/s | Peak GiB |
| --- | --- | --- | --- | --- | --- |
| M1-L4 | 4 | 198.05 | 198.36 | 323.15 | 6.683 |
| B8 | 4 | 199.95 | 199.95 | 320.08 | 6.683 |
| B4 | 1 | 132.72 | 132.75 | 482.20 | 6.675 |
