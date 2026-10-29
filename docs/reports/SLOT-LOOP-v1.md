# SLOT-LOOP-v1: final study report

**Study result: the registered T1/L1 read-operator benefit is supported; the registered T2 loop benefit is not established.** S0 selected R2, S1A returned **ADVANCE**, and S1B A1 returned **NOT ADVANCED**, with no S2 primary backend. S1B's independently trained loop contrasts were **+2.575 pp** for SoftSlot and **+2.091667 pp** for Matrix, below the registered **+3 pp** requirement. S2 is not entered; the conditional S1B §6 mechanism controls do not train, and S3 is not entered. This report makes no S2 mechanism or S3 cost/practicality claim.

| stage | comparison and observed result | registered disposition | frozen evidence |
| --- | --- | --- | --- |
| S0, frozen T1/E16 reader | Matrix 72.0%, calibrated SoftSlot 74.5%, HardSlot 78.1%; 1,000 dev episodes | R2: full read adaptation; HardSlot remains diagnostic | [S0 decision](../../reports/slot-loop-v1/s0/decision_s0.md) |
| S1A, independently continued T1/E16 L1 readers | Matrix 82.266667%, SoftSlot 86.000000%; +3.733333 pp, paired 95% interval [2.700000, 4.816667] | ADVANCE: conditional H1 supported | [S1A frozen verdict](../../reports/slot-loop-v1/s1a-summary/verdict.md) |
| S1B A1, T2/E12 Memory×Depth | S4−S1 +2.575000 pp; M4−M1 +2.091667 pp; S4−M4 +0.466667 pp, with storage interval spanning zero | NOT ADVANCED; no loop-qualified backend | [S1B frozen verdict](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.json) |

### Answers to the six questions in plan §15.3

The questions follow the [execution plan](../../plan/looplora_diagnosis_and_execution_plan_2026-10-29.md). Interpretations below are identified explicitly; measured results and registered decisions come from the linked frozen records.

1. **Improvement against a same-start, same-information baseline.** On S1A, both readers continued the same frozen S-RE writer for **6,000 updates**, with identical information access, all valid question tokens and L1. SoftSlot improved over the independently continued Matrix baseline by **+3.733333 pp** (86.00% versus 82.266667%), meeting all registered H1 criteria on 2,000 confirmation episodes. The earlier S0 swap gave a descriptive **+2.5 pp** for calibrated SoftSlot without training. On S1B, independently trained L4 improved over independently trained L1 by **+2.575 pp** for SoftSlot and **+2.091667 pp** for Matrix, failing the required +3 pp size; the L4 storage contrast was **+0.466667 pp**, interval **[−0.491667, 1.450000]**. [S0](../../reports/slot-loop-v1/s0/decision_s0.md), [S1A](../../reports/slot-loop-v1/decision_s1a.md), [S1B](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.json).

2. **Attribution to operator, cross-step query, supervision or resources.** S0 held weights fixed: changing SoftSlot from uncalibrated τ=1 to calibrated τ=0.1 changed EM from **2.1% to 74.5%**, while Matrix was 72.0%. S1A's read error fell from **0.1503 to 0.0770** alongside its EM gain, but the operators also differ in normalization; equal LR-search effort selected **0.001 versus 0.003**, so optimization trajectories were not identical. S1B used the same **180,224 Q/R/U/B trainable parameters**, frozen shared A and writer, answer-only loss, and **18,000 updates** in every arm. Actual three-seed main cost nevertheless rose from **2.829722 to 4.855000 node·h** for M1→M4, and **3.084444 to 5.687500** for S1→S4. **Interpretation:** the evidence supports a calibrated operator-plus-adaptation benefit in T1; it does not isolate interference removal, cross-step query semantics or a compute-independent loop mechanism. There were **zero** S2 transition-supervision runs and **zero** conditional S1B §6 mechanism-control runs. [Frozen S0 conditions](../../reports/slot-loop-v1/s0/decision_s0.md), [S1A metrics](../../reports/slot-loop-v1/s1a-summary/verdict.md), [S1B freeze record](../../reports/slot-loop-v1/s1b/a1/main/M1/s17/record.json), [main cost ledger](../../reports/slot-loop-v1/s1b/a1/main-jobs.json).

3. **Negative results and failed thresholds.** S0's SoftSlot gain **+2.5 pp** missed its exploratory +5 pp priority signal; HardSlot's +6.1 pp selected R2. S1A unseen-name EM remained **10.45%/13.433333%** for Matrix/SoftSlot despite a positive relative difference. The initial S1B Matrix-L1 calibration was **5.1% hop-1 EM**, below its 10% sanity stop; diagnosis and amendment A1 preceded later runs. In final S1B, both loop point estimates missed **+3 pp**, while storage failed both positive interval lower bound and strictly positive differences in every seed (**seed 43: 0 pp**). Absolute hop-2/3 confirmation EM was only **8.541667–11.100000%** across the four arms. [S0 decision](../../reports/slot-loop-v1/s0/decision_s0.md), [S1A unseen metrics](../../reports/slot-loop-v1/s1a-summary/verdict.md), [stopped calibration](../../reports/slot-loop-v1/s1b/a1/stopped-calibration.json), [A1](../prereg/SLOT-LOOP-v1-S1B-A1.md), [frozen S1B decision](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.json).

4. **Agreement among independent L1, same-checkpoint sweeps and interventions.** The L4-trained checkpoints' confirmation L4−L1 sweep differences averaged **+6.233333 pp Matrix / +7.750000 pp SoftSlot**, with positive differences in all three seeds. Both sweep screens pass, yet the independent L4−L1 comparisons above fail their +3 pp criterion. At L2/L8, hop-2/3 EM falls to **0.208333%/0.091667%** for M4 and **0.691667%/0.516667%** for S4. Saved trajectories agree with the actual forward, but **zero** registered S2 intermediate-state intervention runs were entered. **Interpretation:** depth-specific reader adaptation and trace fidelity do not establish causal multi-hop transitions or the registered loop effect. [Frozen contrasts/continuation](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.json), [all sweep rows](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.md), [accepted gate consequences](../tasks/WP-L14_slot_s1b.md).

5. **Parents, names, tasks and budgets conditioning the conclusion.** Confirmatory results use **one S-RE writer parent** and continuation seeds **17/29/43**, which vary data order rather than writer identity. Seen names are **[0,768)**; S1A's descriptive unseen names are **[768,1024)**. S1A used T1/E16, four relations, L1 and 6,000 updates; S1B used T2/E12, chains of length **8 and 4**, relation 0, training/checkpoint-selection hops **1–3**, confirmation hops **1–8**, and 18,000 updates per arm/seed. Each stage used its own 2,000-episode confirmation namespace. S0's D128-3 recheck (**63.3% Matrix, 73.7% SoftSlot**) is descriptive evidence from an earlier checkpoint, not an additional confirmatory writer replication. S1B unseen was fixed **off**. There is no broad writer, capacity, unseen-name or real-text generalization claim. [S0 recheck](../../reports/slot-loop-v1/s0/decision_s0.md), [S1A frozen decision](../../reports/slot-loop-v1/decision_s1a.md), [S1B manifest](../../reports/slot-loop-v1/s1b/a1/manifest.lock.json), [A1 release](../../reports/slot-loop-v1/s1b/a1/release.json).

6. **State and end-to-end evidence for practical use.** S1B preflight verified **43,028 B/episode** for E12 slots versus **393,216 B/episode** for the matrix state. This accounts for the backend state, not complete deployment memory, preparation, KV/cache or peak residency. S1B's recorded execution cost was **17.951389 node·h**, including its independent rerun; the documented S0/S1A/S1B ledger totals approximately **20.760222 node·h**. These are execution costs, not a matched end-to-end latency or fixed-byte utility comparison. There were **zero** S3 runs, and the S1B gate prevents entering S2/S3. **Interpretation:** smaller measured backend state alone does not establish a cost advantage or practicality; no such claim is made. [Verified byte checks](../../reports/slot-loop-v1/s1b/a1/preflight/preflight.json), [S1B final ledger](../../reports/slot-loop-v1/s1b/a1/final-summary.json), [S0/S1A accounting](SLOT-LOOP-v1-state.md), [stage entry rules](../prereg/SLOT-LOOP-v1.md).

The original S1A completion report follows **unchanged**, including its historical stage-status statements. Those statements describe the point when S1A finished; the study-level status above and the S1B section below record the subsequent completed stage.

---

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

---

# S1B A1 (WP-L14): NOT ADVANCED

The orchestrator accepted the frozen verdict at `d5c6b64f24938141c1bd5d2e6658ea4b9f980717` and merged it into master at `2a27f57`. This report follows the newly authorized master merge, a fast-forward to **`4f98aae8498a7126b353f99edcae202588a7afe5`**. No experiment or analysis was rerun to write it. Frozen source: **`258f5f5a24eae684f640a9fd62679bb643470a76`**; ready-release blake2b-128 **`d1ccbb7ff4c31fa53abfd7a3cc9ad572`**; verdict blake2b-128 **`dca8d4e9777a8aefb85b79d8a8cdd758`**. [Full frozen JSON](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.json), [full frozen tables](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.md), [execution/audit report](WP-L14.md).

## Registered comparison and decisions

The S1A ADVANCE branch admitted M1, M4, S1 and S4, with the same frozen shared A, S-RE writer and backbone. M/S denote Matrix/SoftSlot; 1/4 denote independently trained loop depth. All twelve arm/seed runs started from the unadapted parent and completed **18,000 updates**, batch 32: **576,000 episodes / 2,304,000 training questions per run**. LR was 0.003 for every arm; Matrix used g=1, SoftSlot g=1/τ=0.1, capacity 12 and all valid question tokens. Checkpoints were selected by hop-1/2/3 dev NLL before opening. The independent DEV rerun reproduced selection byte-for-byte. [Preregistration](../prereg/SLOT-LOOP-v1-S1B.md), [A1](../prereg/SLOT-LOOP-v1-S1B-A1.md), [release](../../reports/slot-loop-v1/s1b/a1/release.json), [selection](../../reports/slot-loop-v1/s1b/a1/dev-selection/selection.json), [independent rerun](../review/static_2026-10-29/s1b-selection-rerun.md), [open record](../../reports/slot-loop-v1/s1b/open.json).

Primary differences average hops **2 and 3**, over seeds **17/29/43**, on **2,000 paired episode IDs**. The frozen analysis used **10,000 whole-episode resamples**, bootstrap seed **5799536610111230359**, and **98⅓%** Bonferroni intervals for the three-contrast family. Seeds/questions from one episode stay together. Exact count/rational comparisons determine pass/fail; displayed rounding does not change the decision.

| contrast | difference (pp) | 98⅓% interval (pp) | seed 17 (pp) | seed 29 (pp) | seed 43 (pp) | hop-1 difference (pp) | decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| S4−S1 | +2.575000 | [1.733333, 3.391667] | +2.300 | +2.450 | +2.975 | +5.066667 | fail: < +3 pp |
| M4−M1 | +2.091667 | [1.191667, 3.000000] | +1.275 | +2.350 | +2.650 | +1.183333 | fail: < +3 pp |
| S4−M4 | +0.466667 | [−0.491667, 1.450000] | +1.350 | +0.050 | 0.000 | +7.633333 | fail: lower bound ≤0; seed 43 is zero |

Both loop contrasts have positive intervals, positive differences in every seed and acceptable hop-1 changes, but fail the registered minimum point difference. Storage has a positive point difference without the required positive lower bound or unanimous strictly positive seeds. **There is no loop gain at the registered size and no confirmed L4 storage advantage.** This does not estimate the true loop effect as exactly zero; it records the failed registered conjunction. Frozen decision evidence: [verdict.json](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.json).

## Continuation

| backend | loop contrast passes | same-checkpoint L4−L1, seeds 17 / 29 / 43 (pp) | sweep screen | S2 qualified |
| --- | --- | --- | --- | --- |
| Matrix | no | +7.200 / +6.300 / +5.200 | pass | no |
| SoftSlot | no | +8.300 / +7.700 / +7.250 | pass | no |

The frozen verdict is **NOT ADVANCED**, `s2_primary: null`. The sweep screen cannot replace the independently trained L1 control. Neither backend qualifies; **S2 is not entered**, and the conditional S1B §6 mechanism controls are untriggered. **S3 is not entered**, because it needs the unestablished S2 mechanism evidence. The [accepted task disposition](../tasks/WP-L14_slot_s1b.md) and [S2 task closure](../tasks/WP-L15_slot_s2.md) preserve these gate consequences. No S2 mechanism evidence or S3 end-to-end cost/practicality claim follows from this study.

## Absolute confirmation performance

These EM percentages pool the three selected checkpoints at each arm's **trained depth**, directly from frozen per-seed correct counts. Each hop has 2,000 shared episode IDs evaluated under three continuation seeds (6,000 responses); the inferential episode count remains 2,000. Predictions use the **full vocabulary**.

For orientation, the nominal reference for choosing among **12 support entries** is **1/12 ≈8.333%**; repeated values or task structure can change an actual random-choice baseline. This support-choice reference is distinct from full-vocabulary LM chance and was not evaluated as a random baseline. **†** marks values within **±1 pp** of 8.333%; that band is a descriptive display rule, not a significance test or a preregistered decision threshold.

| trained arm | hop 1 EM (%) | hop 2 | hop 3 | hop 4 | hop 5 | hop 6 | hop 7 | hop 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 22.633 | 8.833† | 8.250† | 8.133† | 8.717† | 8.300† | 8.350† | 8.250† |
| M4 | 23.817 | 11.183 | 10.083 | 9.217† | 9.183† | 9.533 | 9.133† | 9.717 |
| S1 | 26.383 | 8.467† | 8.583† | 8.250† | 9.067† | 8.200† | 8.567† | 8.567† |
| S4 | 31.450 | 12.617 | 9.583 | 9.600 | 9.433 | 9.383 | 9.333† | 9.933 |

| trained arm | all-hop correct / 48,000 | all-hop EM (%) | hop-2/3 EM (%) | all-hop NLL (nats) |
| --- | ---: | ---: | ---: | ---: |
| M1 | 4888 | 10.183333 | 8.541667 | 5.864634 |
| M4 | 5512 | 11.483333 | 10.633333 | 5.786496 |
| S1 | 5165 | 10.760417 | 8.525000 | 5.806749 |
| S4 | 6080 | 12.666667 | 11.100000 | 5.730412 |

**Interpretation:** the relative contrasts coexist with low absolute multi-hop accuracy. M1/S1 hops 2–8 sit near the nominal support-choice level; most longer-hop L4 values also remain around 9–10%. These results do not demonstrate reliable chain traversal. The [frozen tables](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.md) retain every arm/seed/hop count and NLL. Descriptive hop-2/3 S1−M1 is **−0.016667 pp** and Memory×Depth is **+0.483333 pp**; neither changes the registered verdict.

## Descriptive same-checkpoint sweeps

All selected M4/S4 checkpoints were evaluated at **L=1/2/4/8** on identical graphs/memory, without retuning reads. The following values are pooled hop-2/3 EM. Confirmation uses 2,000 episode IDs; dev uses its unchanged 1,000. The descriptive dev sweep was produced after selection and affects no decision.

| set | L4-trained arm | EM@1 (%) | EM@2 (%) | EM@4 (%) | EM@8 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| confirmation | M4 | 4.400000 | 0.208333 | 10.633333 | 0.091667 |
| confirmation | S4 | 3.350000 | 0.691667 | 11.100000 | 0.516667 |
| dev | M4 | 3.816667 | 0.050000 | 9.733333 | 0.016667 |
| dev | S4 | 2.883333 | 0.516667 | 11.233333 | 0.333333 |

Actual final-question-token q/m/z trajectories are saved in `[episode,hop,step,feature]` order, and all trace/forward agreement checks passed. **Interpretation:** L4-trained readers are strongly depth-specific; additional loops do not give a monotonic accuracy improvement. Trace agreement establishes that the saved trace matches the executed recurrence, while causal intermediate-state semantics remain untested. [Confirmation sweep records](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/verdict.json), [dev summary](../../reports/slot-loop-v1/s1b/a1/descriptive-dev/summary.json), [dev audit](../../reports/slot-loop-v1/s1b/a1/descriptive-dev/audit.json), [complete sample inventory](../../reports/slot-loop-v1/s1b/a1/opened-evaluation-artifacts.json).

## Calibration stop and amendment A1

Before S1B pilot/dev/confirmation, the original calibration hit its registered stop: calibrated Matrix-L1 hop-1 EM **5.1% <10%**. A single diagnosis on calibration/non-sealed T1 data reproduced the T1 reference **733/1000**, found T2 oracle-read EM **79.5%**, and measured actual final-token key top-1 **14.8%** under chain-end wording. On the same T2 graphs, T1 wording raised key top-1 to **59.5%** and EM to **31.5%**. **Interpretation registered at that time:** this supported a prompt/read-position mismatch acting on the unadapted query Q; it did not motivate a claim that the values were unusable. [Stopped calibration](../../reports/slot-loop-v1/s1b/a1/stopped-calibration.json), [frozen diagnosis](../../reports/slot-loop-v1/s1b/diagnosis-6855048/diagnosis.json).

The orchestrator registered [A1](../prereg/SLOT-LOOP-v1-S1B-A1.md), blake2b-128 **`341d3c71f9334a1384a102b097e7ad75`**, before later stages. It cleared this stop, retained the stopped evidence as descriptive, inherited S1A's released reads (Matrix g=1; SoftSlot g=1/τ=0.1; capacity adjusted to 12), and increased **N=6,000→18,000** with dev every 1,800 updates. No new T2 read-temperature search was run. Branch, seeds, confirmation namespace, contrasts and continuation criteria stayed fixed. This final result is the **amended** protocol's result; the original stopped calibration remains in the record.

The new source passed **191 tests, zero skips**, eleven real-parent preflight gates, smoke/freeze checks, twelve finite LR-pilot candidates and four independent CLI reloads. All twelve main runs completed; the frozen DEV selection and independent rerun agreed before opening. Confirmation and descriptive dev each produced 30 sample files; all **60** were audited on compute. The frozen final verdict and Markdown equal their persistent originals, and the orchestrator independently recomputed the three point estimates with matching values. [Validation ledger](WP-L14.md), [A1 preflight](../../reports/slot-loop-v1/s1b/a1/preflight/preflight.json), [final analysis audit](../../reports/slot-loop-v1/s1b/a1/confirmation-analysis/analysis-audit.json), [accepted verdict](../tasks/WP-L14_slot_s1b.md).

## Resource ledger and practical limits

| recorded phase | actual node·h | accounting source |
| --- | ---: | --- |
| S0 implementation/tests/retries, preflight and diagnostic | 0.778278 | [S0/S1A state ledger](SLOT-LOOP-v1-state.md) |
| S1A preparation, six main runs, selections/review, confirmation/unseen and analysis | 2.030556 | [S1A final ledger](SLOT-LOOP-v1-state.md) |
| S1B preparation/tests/retries/stopped calibration/diagnosis through ready release | 0.956111 | [WP-L14 ledger](WP-L14.md) |
| S1B twelve main runs | 16.456667 | [per-run PBS ledger](../../reports/slot-loop-v1/s1b/a1/main-jobs.json) |
| S1B worker DEV audit/selection | 0.010000 | [DEV receipt](../../reports/slot-loop-v1/s1b/a1/dev-job.json) |
| S1B independent DEV rerun | 0.009444 | [independent report](../review/static_2026-10-29/s1b-selection-rerun.md) |
| S1B confirmation + descriptive dev evaluations | 0.499444 | [evaluation ledger](../../reports/slot-loop-v1/s1b/a1/opened-evaluation-jobs.json) |
| S1B frozen verdict + descriptive dev audit/summary | 0.019722 | [final job receipt](../../reports/slot-loop-v1/s1b/a1/verdict-job.json) |
| **S1B subtotal, including independent rerun** | **17.951389** | **64,625 node seconds**; [final summary](../../reports/slot-loop-v1/s1b/a1/final-summary.json) |
| **Documented S0/S1A/S1B execution total** | **≈20.760222** | prior S0+S1A 2.808833 plus S1B 17.951389 |

The inherited S0 basis includes a rounded **0.503 node·h** handover; S0/S1A use their recorded log-duration accounting, and the S1B final ledger uses PBS elapsed times where available. The combined figure preserves those bases rather than implying greater accounting precision. It includes retries, superseded calibration/diagnosis and the two independent selection reruns. Later master-integration test jobs and parallel S2 software preparation are outside these stage ledgers; their costs are not silently treated as zero. This documentation task submitted no jobs and adds **0 node·h**.

All jobs respected the user-authorized **≤16 node·h per-job** limit; older aggregate stage caps had been lifted. Slots' verified E12 persistent state is **43,028 B** against **393,216 B** for the matrix, but no equal-byte capacity sweep, matched deployment peak memory study or end-to-end latency comparison was performed. Node-hours and training times do not provide those missing comparisons. **S3 practicality is untested.** [Byte evidence](../../reports/slot-loop-v1/s1b/a1/preflight/preflight.json), [registered scope](../prereg/SLOT-LOOP-v1-S1B.md).

Unseen S1B evaluation remains **unrun** because the release fixed it false. The conditional §6 controls and S2/S3 studies are **not entered** under the failed loop gate; there is no S2 mechanism evidence or S3 cost/practicality claim. Table aggregation and formatting used frozen JSON counts; no bootstrap, model evaluation or source change was needed for this report. Frozen records and original sample/checkpoint paths remain in the linked inventories for reproduction.
