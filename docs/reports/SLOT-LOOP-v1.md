# SLOT-LOOP-v1 (WP-L13): S1A complete

**Decision: ADVANCE.** SoftSlot-L1 achieved 86.00% mean confirmation EM against Matrix-L1's 82.27% on 2,000 registered episodes: **+3.733 pp, paired 95% interval [2.700, 4.817]**. The registered +3 pp threshold, positive lower bound and three positive seed differences all pass. H1 is supported at the registered level.

| continuation seed | Matrix EM | SoftSlot EM | difference (paired 95% interval), pp |
| --- | ---: | ---: | --- |
| 17 | 82.50% | 86.00% | +3.50 [2.20, 4.80] |
| 29 | 81.95% | 86.15% | +4.20 [2.90, 5.50] |
| 43 | 82.35% | 85.85% | +3.50 [2.10, 4.95] |
| mean | 82.27% | 86.00% | +3.733 [2.700, 4.817] |

Confirmation mean NLL/read error: Matrix 0.908315/0.1503; SoftSlot 0.792593/0.0770. Native and Matrix step 0 both reached 72.35% EM, NLL 1.473178, read error 0.1767; calibrated SoftSlot step 0 reached 76.35%, 1.313163, 0.0808. Unseen-name results are descriptive: Matrix 10.45%, SoftSlot 13.43%, +2.983 pp [2.083, 3.883]; absolute accuracy remains low.

Both arms continued the same S-RE writer for 6,000 steps at L1, T1/E16, full-vocabulary EM and all valid question reads; Q/R/U/B trained while the writer, shared A, backbone and buffers remained frozen. Equal pilot effort selected LR 0.001 for Matrix and 0.003 for SoftSlot, with g=1 for both and SoftSlot tau=0.1. Checkpoints were selected by dev NLL before opening. The stable dev mean gap was +3.5 pp, so the registered below-3 pp ablation trigger did not fire. No optional common-fixed-LR run was made.

S0 selected R2 (full adaptation): frozen Matrix/SoftSlot/HardSlot dev EM was 72.0/74.5/78.1%; all nine preflight checks passed, including bitwise Matrix–Native logits and gradients. The source has remained frozen since preflight. [S0 decision](../../reports/slot-loop-v1/s0/decision_s0.md), [release report](SLOT-LOOP-v1-release.md), [dev selection](SLOT-LOOP-v1-selected.md).

**Validation.** Compute tests passed before release: jobs 3728741 (158 analysis, release, runbook and round-2 tests) and 3728742 (48 experiment tests); smoke, timing/LR pilot and both reload checks passed. Six main runs completed with finite losses/norms, correct frozen/support/config/release digests and selected-export hashes. Independent job 3729280 reproduced dev selection byte for byte before opening. Jobs 3729313/3729314 evaluated nine entries per set; frozen analysis job 3729369 validated locked counts/IDs and provenance, then used 10,000 paired whole-episode resamples (seed 9056755664197379988). All 21 analysis-input and three output hashes matched; exact samples and outputs are archived. No science rule deviation; written budget caps were lifted by the user.

| final jobs, one node each | outcome | actual node·h |
| --- | --- | ---: |
| 3729313 confirmation, mg0036 | nine × 2,000 episodes; 141 s | 0.039167 |
| 3729314 unseen, mg0009 | nine × 2,000 episodes; 140 s | 0.038889 |
| 3729369 analysis, mg0018 | ADVANCE; 7 s | 0.001944 |

S1A total including preparation, tests and independent re-run: **2.030556 node·h**; whole study **2.808833 node·h**, using inherited rounded accounting of 0.503. Worker-only total is 2.807167; every job is below 16 node·h. Full ledger and log paths: [state](SLOT-LOOP-v1-state.md).

The result is conditional on one parent writer and three data-order seeds, this task, name pool and calibrated read interface. It does not establish interference removal as the cause, loops, multi-hop reasoning or broad unseen-name generalization. ADVANCE makes the full S1B Matrix/SoftSlot × L1/L4 study eligible under a separately registered addendum with new data. No S1B work has started.

Evidence: [decision](../../reports/slot-loop-v1/decision_s1a.md), [frozen verdict and full metrics](../../reports/slot-loop-v1/s1a-summary/verdict.md), [archive inventory](../../reports/slot-loop-v1/s1a-summary/artifacts.json). Science commit `1523b19c7949673ddcd1d89cc1e5fe518736805c`; release blake2b-128 `523d11d57a08a2491893ebe7cedc86a7`; frozen analysis `843d2ff4d1a03224ccd16e68720ba344`.
