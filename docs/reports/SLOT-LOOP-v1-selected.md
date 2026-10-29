# SLOT-LOOP-v1 S1A dev selection (WP-L13)

**S1A selected; ablation not triggered. Confirmation and unseen sets remain sealed.** Frozen S1A-DEV returned `no_trigger`: all six runs are stable, selected dev NLL is below step 0 for every run, and the mean SoftSlot–Matrix dev EM gap is +3.5 pp. The registered ablation requires a stable gap below +3.0 pp, so neither ablation training nor S1A-ABL ran. This is dev selection, not the confirmatory H1 verdict.

| arm | seed | selected step | dev correct / 1,000 | dev NLL |
| --- | ---: | ---: | ---: | ---: |
| Matrix-L1 | 17 | 5,400 | 834 | 0.866041 |
| Matrix-L1 | 29 | 6,000 | 833 | 0.894967 |
| Matrix-L1 | 43 | 5,400 | 838 | 0.887064 |
| SoftSlot-L1 | 17 | 5,400 | 867 | 0.736074 |
| SoftSlot-L1 | 29 | 6,000 | 868 | 0.760171 |
| SoftSlot-L1 | 43 | 4,200 | 875 | 0.774393 |

Mean selected dev EM: Matrix 83.5%, SoftSlot 87.0%; per-seed differences +3.3/+3.5/+3.7 pp. Selection used the fixed dev-NLL rule (earlier checkpoint within 1e-6), not EM. All arms trained 6,000 updates from the same S-RE parent, using the approved LRs and S0 read settings.

**Validation.** Six jobs exited 0 after 6,000 steps, eleven evaluations and 121 expected log rows each. Every loss/norm and evaluation was finite; frozen writer/buffer and fixed-support digests were unchanged; Q/R/U/B gradients were finite and nonzero. All six records carry the approved release digest, correct science/config hash and identical registered dev IDs. Each selected checkpoint's actual file digest matches both its run record and selection output. Analysis verified shared release identity, stability and trigger arithmetic. Its input/script/output hashes were checked before archival. No source or analysis changes, and no confirmation access.

Main jobs 3729048–3729053 used 6,301 seconds = 1.750278 node·h, each below its 30-minute request. S1A-DEV job 3729237 ran on mg0008 at 00:01:46–00:01:52 JST, 6 seconds = 0.001667 node·h. S1A including preparation totals **1.948889 node·h**; whole-study cumulative is **2.727167 node·h** (inherited accounting 0.503). Every job is within 16 node·h.

[Frozen selection](../../reports/slot-loop-v1/s1a-selection.json), [selection table](../../reports/slot-loop-v1/s1a-selection.md), [selected checkpoint paths/digests](../../reports/slot-loop-v1/s1a-selected-checkpoints.json), [archived records/logs/step arrays](../../reports/slot-loop-v1/s1a-main/artifacts.json), [job ledger](../../reports/slot-loop-v1/s1a-main-jobs.json). Six weights remain under persistent runs with retention paths recorded in the checkpoint manifest.

Science commit: `1523b19c7949673ddcd1d89cc1e5fe518736805c`; ready-release blake2b-128: `523d11d57a08a2491893ebe7cedc86a7`; frozen analysis: `843d2ff4d1a03224ccd16e68720ba344`; dev IDs: `6bf2f42455e065ce309af6cd7fc79541`. Next: the orchestrator's independent re-run of this frozen dev analysis and selection check, then an authorized s1a-open.json before any sealed evaluation. The worker created no opening record.
