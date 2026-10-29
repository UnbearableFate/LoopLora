# SLOT-LOOP-v1 S0 (WP-L13): frozen read replacement

**Decision: R2, full S1A adaptation.** HardSlot improves EM by 6.1 pp over Matrix with lower read error on the registered 1,000-episode dev set. Calibrated SoftSlot improves EM by 2.5 pp, below the exploratory 5 pp threshold. S1A choices: **Matrix-L1 g=1; SoftSlot-L1 g=1, tau=0.1**, all valid question tokens. HardSlot remains diagnostic. Formal S1A runs are unstarted; preparation is complete and waits for approval.json.

Calibration used the registered 2,048 episodes, all 18 candidates and fixed NLL selection. Matrix and HardSlot selected g=1; SoftSlot selected g=1/tau=0.1. Uncalibrated SoftSlot (tau=1) reached 2.1% EM, so temperature is consequential. No dev setting or recency variant replaced the registered selection.

| S-RE condition, all valid question tokens | EM | NLL | Mean relative read error | EM difference vs Matrix (descriptive paired 95% interval) |
| --- | ---: | ---: | ---: | --- |
| Native / Matrix, g=1 | 72.0% | 1.577648 | 0.183978 | reference |
| SoftSlot, g=1/tau=0.1 | 74.5% | 1.505075 | 0.092405 | +2.5 pp [0.6, 4.4] |
| HardSlot, g=1 | 78.1% | 2.434278 | 0.137009 | +6.1 pp [3.8, 8.4] |
| SoftSlot recency-matched, same settings | 74.6% | 1.504875 | 0.092405 | +2.6 pp [0.7, 4.5] |

The separate final-only table gives Matrix 72.0%, SoftSlot 75.5%, HardSlot 80.3%, and Clean-final oracle 92.6%. HardSlot's NLL is worse despite higher EM; S0 uses the registered EM/read-error rule. D128-3's descriptive recheck, at S-RE's settings without recalibration, gives Matrix 63.3%, SoftSlot 73.7%, HardSlot 74.8%: the same direction. S0 is exploratory, conditional on one parent and task, and does not establish interference removal as the cause. The 10,000-resample paired intervals are descriptive.

**Validation.** All nine preflight checks passed. S-RE reproduced the old 723/1,000 and NLL 1.642127275466919 bitwise; D128-3 reproduced 608/1,000. Matrix–Native logits and gradients were bitwise identical (maximum differences 0). Owner/mask, isolation, frozen set, feature consistency, exact bytes and capacity errors passed. Analysis checked locked counts/IDs, per-sample Native/Matrix correctness/NLL, calibration selection and decision thresholds. Input/script/output blake2b digests matched before archiving. No source change after preflight and no sealed S1A data opened.

| job | result | actual node·h |
| --- | --- | ---: |
| 3728339.opbs, preflight, mg0031 | all required checks passed, 53 s | 0.014722 |
| 3728354.opbs, S0, mg0046 | 18 calibration candidates, 13 dev conditions and D128-3 recheck, 218 s | 0.060556 |
| 3728547.opbs, analysis, mg0022 | R2, persistent decision artifacts, 6 s | 0.001667 |

Execution adds 0.076944 node·h; cumulative including tests/retries is 0.778278 node·h (inherited accounting 0.503). Written stage budgets were lifted; every job is below 16 node·h. Preflight commit: `7febf58c3440483fe081e99c263f68ad62512b10`; lock/S0/analysis commit: `2f22aba2f6e93029198d1126ede2b9528cd3eef4`; source blake2b-128: `fe548da4c06d77b7f4a8ae65349650f9`.

Artifacts: [decision](../../reports/slot-loop-v1/s0/decision_s0.md), [metrics and samples inventory](../../reports/slot-loop-v1/s0/artifacts.json), [lock](../../reports/slot-loop-v1/manifest.lock.json), [preflight](../../reports/slot-loop-v1/preflight/preflight.json), [detailed state](SLOT-LOOP-v1-state.md). Persistent originals: `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/`. Deferred S1A follow-ups are task Review round 3: analysis-only release-file digest binding and complete S1A runbook/parser tests; no further review round. No deviation from registered S0 science rules.

S1A preparation is recorded separately in [the release report](SLOT-LOOP-v1-release.md); the costs above describe the earlier S0 stop.
