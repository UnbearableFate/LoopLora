# WP-C5 — final-only support writes (2026-10-27)

**Implemented and enabled `memory.scan_final` in both write-only wrappers;
P1-L T1 throughput improves 1.16–2.06× across all requested settings.** It packs
each row's valid writes and gates, preserves token order and inactive rows, and
iterates masked updates without per-step nonzero/index_select or host sync.
No explicit prefix snapshot stack. Token-zero resets stay fast; interior resets
use the unchanged reference. Online mode keeps scan_prewrite. No dtype,
dependency or torch.compile changes.

**150 tests passed** on `mg0002`, PBS **3714480**: randomized CPU/CUDA FP32
states and gradients for u/K/V/scalar and input gates, padding, zero/all writes,
resets, empty sequences and undefined inactive inputs; rtol 1e-5, atol 1e-6.
Both input sites pass fixed-batch memory-only loss checks. On the pinned 0.6B
model, all four nonzero-B site/policy losses match exactly (required relative
1e-5); maximum state error is 2.38e-7.

Full benchmark **3714489**, `mg0002`, exit 0: Qwen3-0.6B-Base, BF16 frozen
backbone, FP32 adapter/state, T1 E=16/Q=4, r=32, layer 14/o_proj input, AdamW.
Five warmups, 20 measured steps; batch below is per worker. Values are **ms per
step before → after**, with two-process barriers included in achieved throughput.

| Policy | Batch | One process | Two processes |
| --- | --- | --- | --- |
| fact_final | 32 | 389.8 → 189.4 | 518.7 → 286.4 |
| fact_final | 64 | 432.3 → 228.5 | 693.3 → 460.6 |
| fact_final | 128 | 554.3 → 343.6 | 1100.7 → 843.6 |
| all_tokens | 32 | 407.4 → 295.2 | 556.7 → 403.7 |
| all_tokens | 64 | 447.2 → 333.7 | 722.9 → 575.2 |
| all_tokens | 128 | 561.8 → 450.1 | 1114.4 → 962.0 |

[Complete tables](../../reports/fast-write/tables.md) include episodes/s and
peak memory for every cell. All fit. Best observed throughput is one process,
batch 128: fact_final **230.9→372.5 episodes/s**, peak **7.763→7.749 GiB**;
all_tokens **227.8→284.4**, **7.892→7.828 GiB**. Whole-step memory reductions
are modest; autograd still needs writer intermediates. Two-worker peaks are
reported as sums of worker peaks, not simultaneous peak readings.

Separate three-step profiles locate remaining cost. At batch 128/one process,
fast fact_final spends approximately **40 ms support forward, 9 write,
139 question forward/backward, 1 optimizer, 154 generation/rendering** (343 total).
all_tokens spends **40/43/210/1/156 ms** (450 total). Question backward includes
writer backward. Two-process preparation reaches about 474 ms at batch 128,
limiting throughput. Profile synchronization is excluded from headline timings.

An initial bulk projection changed FP32 rounding enough to fail residual/all_tokens
loss equivalence through the BF16 backbone. Retaining per-token projections
fixed it; tolerances were unchanged. A diagnostic-only padded gather also failed
and was corrected. [Reproduction notes](../../reports/fast-write/README.md) preserve
these investigations, test commands and source commits. Final [artifact audit](../../reports/fast-write/audit.json)
**3714538**, `mg0016`, passed all 24 cells, 720 measured worker steps and phase
partitions. [Total cost](../../reports/fast-write/jobs.json): **572 s = 0.15889/0.5
node·h**, including failures and probes; all runtime used compute via submit.py.
No open checks. Interior resets deliberately retain reference performance.
