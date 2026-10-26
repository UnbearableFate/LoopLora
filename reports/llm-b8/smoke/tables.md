# WP-C6 B8 versus M1 at fixed L=4

One process, two prefetch workers, 1 discarded warmups and 2 measured steps per cell. Pinned Qwen3-0.6B-Base, layer14/o_proj, r32, E16/Q4 T2 train-only data, fact_final writes. CUDA-synchronized production training steps; process/prefetch startup and export excluded from steady ms. These are short systems measurements, not quality comparisons.

| Variant | Batch | Mean ms/step | Median ms/step | Episodes/s | Peak GiB | Fast / reference scan calls |
| --- | --- | --- | --- | --- | --- | --- |
| B8 | 32 | 261.20 | 261.20 | 122.51 | 3.960 | 3 / 0 |
| M1 | 32 | 254.52 | 254.52 | 125.73 | 3.960 | 3 / 0 |
| B8 | 128 | 361.71 | 361.71 | 353.88 | 12.250 | 3 / 0 |
| M1 | 128 | 355.70 | 355.70 | 359.85 | 12.243 | 3 / 0 |
