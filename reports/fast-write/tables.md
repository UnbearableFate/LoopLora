# WP-C5 full P1-L T1 benchmark

Pinned Qwen3-0.6B-Base, E=16, Q=4, r=32, layer 14, o_proj input. Frozen BF16 backbone, FP32 adapter/state, AdamW 3e-4. Five warmups and 20 measured steps. Both methods start from identical parameters, optimizer and episode streams. Each step generates and renders fresh training episodes; no accuracy selection.

Wall ms/step includes synchronization barriers in two-process achieved throughput. Worker durations without barriers remain in the JSON files. Memory is the sum of per-worker peak allocated GiB, not an observed simultaneous peak. Every requested cell fit.

## Before / after

| Processes | Policy | Batch/worker | ms/step before / after | Episodes/s before / after | Sum peak GiB before / after | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | fact_final | 32 | 389.8 / 189.4 | 82.1 / 168.9 | 2.823 / 2.819 | 2.06× |
| 1 | fact_final | 64 | 432.3 / 228.5 | 148.1 / 280.1 | 4.470 / 4.463 | 1.89× |
| 1 | fact_final | 128 | 554.3 / 343.6 | 230.9 / 372.5 | 7.763 / 7.749 | 1.61× |
| 1 | all_tokens | 32 | 407.4 / 295.2 | 78.5 / 108.4 | 2.857 / 2.841 | 1.38× |
| 1 | all_tokens | 64 | 447.2 / 333.7 | 143.1 / 191.8 | 4.536 / 4.503 | 1.34× |
| 1 | all_tokens | 128 | 561.8 / 450.1 | 227.8 / 284.4 | 7.892 / 7.828 | 1.25× |
| 2 | fact_final | 32 | 518.7 / 286.4 | 123.4 / 223.4 | 5.641 / 5.633 | 1.81× |
| 2 | fact_final | 64 | 693.3 / 460.6 | 184.6 / 277.9 | 8.934 / 8.920 | 1.51× |
| 2 | fact_final | 128 | 1100.7 / 843.6 | 232.6 / 303.5 | 15.520 / 15.492 | 1.30× |
| 2 | all_tokens | 32 | 556.7 / 403.7 | 115.0 / 158.6 | 5.710 / 5.677 | 1.38× |
| 2 | all_tokens | 64 | 722.9 / 575.2 | 177.1 / 222.5 | 9.067 / 9.001 | 1.26× |
| 2 | all_tokens | 128 | 1114.4 / 962.0 | 229.7 / 266.1 | 15.778 / 15.651 | 1.16× |

## Separate profiling steps

Three extra steps synchronize at phase boundaries. They diagnose remaining cost and do not replace the unprofiled timings above. Support forward includes the frozen prefix, A projection and support bookkeeping, excluding gates/K/V/state updates in Write. Question forward/backward includes the backward pass through the writer; optimizer includes gradient clipping. Render means support tokenization and remaining preparation.

| Processes | Policy | Batch | Method | Data | Render | Support fwd | Write | Question fwd/bwd | Optimizer | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fact_final | 32 | reference | 3.37 | 39.59 | 24.23 | 86.28 | 233.08 | 1.13 | 387.67 |
| 1 | fact_final | 32 | fast | 3.13 | 39.17 | 23.46 | 8.91 | 112.65 | 0.96 | 188.28 |
| 1 | fact_final | 64 | reference | 5.46 | 76.22 | 24.37 | 88.44 | 235.23 | 1.07 | 430.79 |
| 1 | fact_final | 64 | fast | 5.57 | 75.36 | 23.84 | 8.89 | 113.91 | 0.97 | 228.54 |
| 1 | fact_final | 128 | reference | 10.44 | 146.23 | 40.39 | 89.35 | 261.01 | 1.09 | 548.50 |
| 1 | fact_final | 128 | fast | 10.48 | 143.88 | 39.94 | 8.72 | 138.71 | 0.98 | 342.72 |
| 1 | all_tokens | 32 | reference | 3.08 | 39.56 | 23.80 | 92.41 | 241.72 | 1.08 | 401.64 |
| 1 | all_tokens | 32 | fast | 3.32 | 39.38 | 23.65 | 42.91 | 181.18 | 1.09 | 291.54 |
| 1 | all_tokens | 64 | reference | 5.46 | 76.23 | 24.22 | 94.44 | 247.97 | 1.10 | 449.42 |
| 1 | all_tokens | 64 | fast | 5.61 | 75.12 | 23.72 | 42.76 | 184.71 | 1.04 | 332.96 |
| 1 | all_tokens | 128 | reference | 10.37 | 145.29 | 40.46 | 94.17 | 270.59 | 1.10 | 561.98 |
| 1 | all_tokens | 128 | fast | 10.59 | 145.09 | 40.29 | 42.95 | 209.67 | 1.09 | 449.68 |
| 2 | fact_final | 32 | reference | 3.29 | 120.39 | 27.12 | 102.90 | 252.14 | 1.24 | 507.09 |
| 2 | fact_final | 32 | fast | 3.27 | 119.55 | 26.88 | 9.47 | 123.80 | 1.18 | 284.15 |
| 2 | fact_final | 64 | reference | 5.97 | 239.72 | 45.96 | 104.01 | 290.96 | 1.25 | 687.88 |
| 2 | fact_final | 64 | fast | 5.99 | 237.76 | 45.26 | 9.51 | 160.21 | 1.23 | 459.96 |
| 2 | fact_final | 128 | reference | 11.12 | 475.87 | 83.78 | 128.51 | 383.15 | 1.29 | 1083.72 |
| 2 | fact_final | 128 | fast | 11.04 | 472.03 | 82.00 | 10.71 | 265.12 | 1.45 | 842.35 |
| 2 | all_tokens | 32 | reference | 3.27 | 120.12 | 27.02 | 127.35 | 265.88 | 1.23 | 544.88 |
| 2 | all_tokens | 32 | fast | 3.59 | 120.16 | 27.03 | 43.51 | 201.74 | 1.13 | 397.15 |
| 2 | all_tokens | 64 | reference | 5.91 | 239.68 | 45.87 | 126.47 | 302.43 | 1.28 | 721.63 |
| 2 | all_tokens | 64 | fast | 6.04 | 238.69 | 45.21 | 44.56 | 239.16 | 1.10 | 574.76 |
| 2 | all_tokens | 128 | reference | 11.27 | 476.58 | 83.77 | 124.09 | 399.09 | 1.28 | 1096.08 |
| 2 | all_tokens | 128 | fast | 11.48 | 474.06 | 80.37 | 45.83 | 339.88 | 1.16 | 952.78 |

## Fixed-batch real-model equivalence

Nonzero B, batch 2, same episode data. Required loss relative tolerance 1e-5; state checks use rtol 1e-5, atol 1e-6. Core tests also compare u/K/V/gate gradients.

| Input | Policy | Reference CE | Fast CE | Maximum state error |
| --- | --- | --- | --- | --- |
| o_proj | fact_final | 11.856364250 | 11.856364250 | 1.49e-08 |
| o_proj | all_tokens | 11.840630531 | 11.840630531 | 1.49e-08 |
| residual | fact_final | 12.167141914 | 12.167141914 | 8.94e-08 |
| residual | all_tokens | 12.249484062 | 12.249484062 | 2.38e-07 |
