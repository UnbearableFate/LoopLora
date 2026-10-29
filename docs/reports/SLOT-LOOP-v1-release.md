# SLOT-LOOP-v1 S1A release preparation (WP-L13)

**Ready for approval.json; formal runs are unstarted.** S0 R2 settings are retained. The real smoke, six-candidate LR/timing pilot, candidate builder, both exact reloads and ready-release builder passed. No S1A dev selection, ablation, confirmation or unseen evaluation has run.

| fixed item | released value |
| --- | --- |
| science commit | `1523b19c7949673ddcd1d89cc1e5fe518736805c` |
| steps, all arms/seeds | 6,000; ten dev checkpoints plus unselectable step 0 |
| Matrix-L1 | LR 0.001, g=1 |
| SoftSlot-L1 | LR 0.003, g=1, tau=0.1 |
| scope / trainables | all valid question tokens / Q,R,U,B; writer/backbone/buffers frozen |
| per-run walltime | 1,800 s, debug-g, one node (0.5 requested node·h) |
| configs | six main arm/seed hashes plus two seed-17 query-copy ablation hashes |

All six pilot candidates completed 200 finite steps. NLLs at LRs 0.0003/0.001/0.003 were Matrix 1.161793/1.139500/1.162964 and SoftSlot 1.036309/0.970089/0.945848. The registered 0.01-nat tie rule selected the released rates. Timing maxima S=0.158955944 s, Ev=8.320275826 s, first load Ld=1.368659242 s give N=6,000 and 1,800 s under the unchanged registered formulas. Expected unpadded time is about 17.4 minutes per formal run; the six main jobs request 3 node·h total and project about 1.74 actual node·h from pilot timings.

**Validation.** Before the smoke, analysis-only follow-ups bound the exact ready-release bytes to the opening record and required one shared release digest across six training records. RED reproduced both defects; complete submission/target parser checks replaced the incomplete S1A runbook. GREEN: 158 analysis/runbook/integration cases and 48 experiment cases passed. No src change invalidated the real-parent preflight.

Both real 20-step smoke arms passed all frozen-tensor/buffer/support and gradient checks (180,224 trainable, 294,914 frozen parameters). Reloads on separate nodes exactly matched the persisted candidate configs and saved episode IDs, predictions, correctness and NLL. The ready builder repeated calibration/LR/schedule/freeze/reload self-checks. All eight config hashes and exact input digests were independently verified. Smoke weights remain only in persistent runs; their digests and retention paths are archived.

| job | phase | seconds | node·h |
| --- | --- | ---: | ---: |
| 3728718 | expected RED failures | 18 | 0.005000 |
| 3728741 / 3728742 | GREEN tests | 37 / 170 | 0.057500 |
| 3728762 | real smoke | 36 | 0.010000 |
| 3728825 | LR/timing pilot | 384 | 0.106667 |
| 3728950 | candidate builder | 6 | 0.001667 |
| 3728988 / 3728990 | exact reloads | 26 / 25 | 0.014167 |
| 3729009 | ready builder | 7 | 0.001944 |

S1A preparation: 709/3,600 = **0.196944 node·h**. Whole-study cumulative including S0/tests/retries: **0.975222 node·h**, based on inherited 0.503 accounting. Written stage budgets are superseded; every job is below 16 node·h.

[Ready release](../../reports/slot-loop-v1/s1a-release.json): blake2b-128 `523d11d57a08a2491893ebe7cedc86a7`. Frozen analysis: `843d2ff4d1a03224ccd16e68720ba344`; prereg unchanged: `56e8c611c01e0ef9b7ae32db97b0e334`; manifest: `5af7a9872827630280fa34218ffabde6`. [Preparation artifacts](../../reports/slot-loop-v1/s1a-preparation/artifacts.json) and [state/runbook](SLOT-LOOP-v1-state.md) retain evidence and job logs. After approval, the six runs must use the released science commit and exact ready-release file. Later dev selection and independent re-run precede opening; the orchestrator checks all six training release digests before issuing s1a-open.json. No approval or opening record was created by the worker.
