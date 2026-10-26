# WP-B4 prefetch benchmark

Batch and prefetch worker counts are per training process. Throughput includes process barriers; startup and close are separate. GPU memory is the sum of per-process peaks.

| Processes | Batch | Policy | Workers | ms/step | episodes/s | Speedup | First batch (s) | Close (s) | Peak allocated (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | all_tokens | 0 | 458.60 | 279.11 | 1.000 | 0.010 | 0.000 | 7.819 |
| 1 | 128 | all_tokens | 8 | 303.22 | 422.14 | 1.512 | 18.696 | 0.484 | 7.819 |
| 1 | 128 | fact_final | 0 | 345.68 | 370.28 | 1.000 | 0.014 | 0.000 | 7.744 |
| 1 | 128 | fact_final | 8 | 190.14 | 673.19 | 1.818 | 18.620 | 0.472 | 7.744 |
| 2 | 128 | all_tokens | 0 | 932.61 | 274.50 | 1.000 | 0.011 | 0.000 | 15.639 |
| 2 | 128 | all_tokens | 8 | 469.60 | 545.15 | 1.986 | 18.368 | 0.568 | 15.639 |
| 2 | 128 | fact_final | 0 | 836.94 | 305.88 | 1.000 | 0.011 | 0.000 | 15.487 |
| 2 | 128 | fact_final | 8 | 364.48 | 702.37 | 2.296 | 18.384 | 0.565 | 15.487 |
