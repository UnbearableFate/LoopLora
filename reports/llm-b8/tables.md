# WP-C6 B8 versus M1 at fixed L=4

One process, two prefetch workers, 5 discarded warmups and 20 measured steps per cell. Pinned Qwen3-0.6B-Base, layer14/o_proj, r32, E16/Q4 T2 train-only data, fact_final writes. CUDA-synchronized production training steps; process/prefetch startup and export excluded from steady ms. These are short systems measurements, not quality comparisons.

| Variant | Batch | Mean ms/step | Median ms/step | Episodes/s | Peak GiB | Fast / reference scan calls |
| --- | --- | --- | --- | --- | --- | --- |
| B8 | 32 | 263.52 | 264.14 | 121.43 | 3.961 | 25 / 0 |
| M1 | 32 | 273.55 | 267.37 | 116.98 | 3.961 | 25 / 0 |
| B8 | 128 | 358.73 | 358.67 | 356.82 | 12.250 | 25 / 0 |
| M1 | 128 | 357.16 | 357.29 | 358.39 | 12.243 | 25 / 0 |
