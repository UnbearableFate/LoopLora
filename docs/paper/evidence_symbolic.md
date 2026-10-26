# Symbolic claims and evidence (WP-T4)

Evidence cutoff: `master` at `30f86e20f4cd3b275a71e06e3138579e6f66a1f0`, merged into
`wp/theory`. This is a synthesis of committed results, not a new experiment or a reanalysis of
sample traces. The defensible result is conditional: evolving queries over a fixed memory can
improve composition, but the benefit depends on the writer, training protocol, and comparison
budget. The strongest joint-writer efficiency claim in plan §14.1 has not been established.

## Reading the tables

Each quoted claim is a candidate paper sentence; its status applies to that sentence with the
conditions in the same row. **Supported** means supported within those conditions;
**narrowed** requires the stated restriction; **not supported** identifies a sentence to reject;
**open** means the required measurement is absent. These are evidence judgments, not new tests.

All accuracies below are **candidate-restricted**, not full-vocabulary exact match. `%` denotes
accuracy; `pp` denotes a difference of accuracies. Values are converted from committed fractions
and rounded to **six decimal places**; the linked files retain the exact stored precision.
Brackets are **97.5% paired episode-bootstrap confidence intervals for the displayed contrast**,
unless explicitly marked **CI unavailable**. No interval is supplied for an absolute accuracy
merely because its contrast has one. Confirmation intervals use the same fixed training seeds
`17/29/43`, with `2,000` paired episode-bootstrap draws and bootstrap seed `2718`; they condition
on those training seeds and do not quantify uncertainty over new training seeds. They are the
registered endpoint intervals, not simultaneous intervals for every sentence in this synthesis.

Set labels used throughout:

| Label | Set and selection | Scope |
| --- | --- | --- |
| C777 | Confirmation, split `test`, evaluation seed `777001`, `2,000` episodes. | P1-S, rank follow-up P1-S-F1, factorized-writer follow-up P1-S-F2. Reusing this set across follow-ups is not independent replication. The P1-S registered replication changes the layout; it is distinct from rerunning the primary layout. |
| C888 | Confirmation, split `test`, evaluation seed `888001`, `2,000` episodes. | P2-S and P2-S-F1. The follow-up explicitly reuses the P2-S confirmation set as registered; it is not a fresh confirmation sample. |
| S4242 | Screening, split `test`, evaluation seed `4242`. | A5 diagnostics and stopping pilots. Stopping uses `2,000` episodes per checkpoint. No stopping CI was computed. |
| V4242 | Validation, split `val`, evaluation seed `4242`. | Stopping thresholds selected here, then frozen before S4242. No confirmation inference follows from the validation optimum. |

For P1-S/F1/F2, let **D** = accuracy at `L=4` minus `L=1` on hops `{2,3}`;
**F** = the same difference on far hops `{6,7,8}`; **W** = `D−F` (walk specificity);
**H** = `L=4−L=1` on hop `1`; **Q** = evolving-query reader minus matched fixed-query
reader B5 at `L=4` on hops `{2,3}`. In P2-S/F1, **reach** is absolute accuracy averaged
over hops `{2,3}` at the listed evaluation depth. **Long** always means hops `{4,5,6,7,8}`,
and must not be substituted for the far set. In the r32 stopping table, D/F/H instead compare
the selected stopping rule with `L=1`; W remains D−F.

Common task is `t2_chain_end`, primary layout `E=12`, chains `(8,4)`, entity dimension `64`,
and `rho=0.25`. The primary readers share their step matrices R/U/Q; M1 uses ReLU and
M1-lin uses identity. B5 caches one memory read, B3 has no memory, B6 increases rank, B7
increases memory dimension, and B8 has step-specific R/U/Q at fixed depth. A depth sweep of
one checkpoint holds x/u/M fixed. A comparison of separately trained models does not.

Oracle codes are fixed across reader variants. `t1_frozen` and `t1_frozen_fact` share one
T1-trained frozen writer per training seed across reader variants. `joint` and `joint_fact`
train a separate writer together with each reader variant: their cross-variant contrasts are
comparisons of trained systems, not interventions holding the writer fixed. `joint_fact` and
`t1_frozen_fact` separate head/relation keys from tail-only values; this removes the direct
head-subtraction statistic identified in Proposition 10, without proving the absence of every
possible global shortcut.

## Source keys and provenance

The commit in each row below is the introducing commit for the numerical artifact, rather
than the later merge commit. All were present at the evidence cutoff. Table entries use these
keys plus a unique writer/reader/rank/endpoint selector. Protocol numbers were checked in
the corresponding committed JSON/CSV manifests or configuration snapshots, not inferred
from rounded report prose.

| Key | Committed source file(s) | Commit |
| --- | --- | --- |
| P1G | [p1s/gates.json](../../reports/p1s/gates.json), including `statistics.<metric>.gate` and replication fields | `98e70eb` |
| P1A | [p1s/accuracy_tables.csv](../../reports/p1s/accuracy_tables.csv), [plan.json](../../reports/p1s/plan.json), [artifact_audit.json](../../reports/p1s/artifact_audit.json) | `98e70eb` |
| P1S | [p1s/swap_successors.csv](../../reports/p1s/swap_successors.csv) | `98e70eb` |
| F1G | [p1s-f1/gates.json](../../reports/p1s-f1/gates.json), [accuracy_tables.csv](../../reports/p1s-f1/accuracy_tables.csv), [evaluation_plan.json](../../reports/p1s-f1/evaluation_plan.json) | `960a84f` |
| F1C | [p1s-f1/manifest.json](../../reports/p1s-f1/manifest.json), [configs/f1-joint-r64-M1-s17.json](../../reports/p1s-f1/configs/f1-joint-r64-M1-s17.json) and corresponding family configs | `8798483` |
| F2E | [p1s-f2/gate_endpoints.csv](../../reports/p1s-f2/gate_endpoints.csv), [gates.csv](../../reports/p1s-f2/gates.csv), [gate_criteria.csv](../../reports/p1s-f2/gate_criteria.csv), [cross_family_contrasts.csv](../../reports/p1s-f2/cross_family_contrasts.csv) | `a898c87` |
| F2A | [p1s-f2/accuracy_summary.csv](../../reports/p1s-f2/accuracy_summary.csv), [swap_successors.csv](../../reports/p1s-f2/swap_successors.csv), [swap_successors_new_path_length.csv](../../reports/p1s-f2/swap_successors_new_path_length.csv), [writer_t1_scores.csv](../../reports/p1s-f2/writer_t1_scores.csv), [artifact_audit_summary.json](../../reports/p1s-f2/artifact_audit_summary.json) | `a898c87` |
| F2M | [p1s-f2/main_manifest.csv](../../reports/p1s-f2/main_manifest.csv) | `a2d637f` |
| P2E | [p2s/endpoints.csv](../../reports/p2s/endpoints.csv), [accuracy.csv](../../reports/p2s/accuracy.csv), [decision.json](../../reports/p2s/decision.json), [checkpoint_manifest.csv](../../reports/p2s/checkpoint_manifest.csv) | `bb101a6` |
| P2C | [p2s/cost.csv](../../reports/p2s/cost.csv), [practical.csv](../../reports/p2s/practical.csv) | `bb101a6` |
| P2FE | [p2s-f1/endpoints.csv](../../reports/p2s-f1/endpoints.csv), [accuracy.csv](../../reports/p2s-f1/accuracy.csv), [decision.json](../../reports/p2s-f1/decision.json), [checkpoint_manifest.csv](../../reports/p2s-f1/checkpoint_manifest.csv), [audit.json](../../reports/p2s-f1/audit.json) | `ae7bda6` |
| P2FM | [p2s-f1/training_manifest.csv](../../reports/p2s-f1/training_manifest.csv), [selected_lrs.csv](../../reports/p2s-f1/selected_lrs.csv) | `44c1fd5` |
| P4 | [p4s/headline.csv](../../reports/p4s/headline.csv), [thresholds.csv](../../reports/p4s/thresholds.csv), [latency.csv](../../reports/p4s/latency.csv), [contraction.csv](../../reports/p4s/contraction.csv), [manifest.json](../../reports/p4s/manifest.json), [audit.json](../../reports/p4s/audit.json) | `9dd0e04` |
| P4D | [p4s/design.json](../../reports/p4s/design.json) | `7888cf0` |
| P4R | [p4s-r32/endpoints.csv](../../reports/p4s-r32/endpoints.csv), [endpoints_per_seed.csv](../../reports/p4s-r32/endpoints_per_seed.csv), [accuracy.csv](../../reports/p4s-r32/accuracy.csv), [thresholds.csv](../../reports/p4s-r32/thresholds.csv), [manifest.json](../../reports/p4s-r32/manifest.json), [audit.json](../../reports/p4s-r32/audit.json) | `3e15f07` |
| P4RD | [p4s-r32/design.json](../../reports/p4s-r32/design.json) | `ce77e74` |

For endpoint CSVs select `confidence=0.975`; both confidence levels occur in the P2 files.
For accuracy tables select the stated set, `original` condition, writer, reader, rank, L and hop;
the reported absolute points are equal-seed means. A5 rates below pool the committed counts
within the explicitly named stratum. These count ratios and unit conversions introduce no
new uncertainty estimates.

## Composition, writer dependence, and extrapolation

P1-S uses `r=r_mem=32`, `15,000` training steps, training hops `{1,2,3}`, and mixed
training depth `{1,2,4}` (B4 only `L=1`). Its registered learned-writer replication uses
typed chains `(8,1)`, `E=12`, with distractors. Both primary oracle rows use the
**random-code**, `E=18`, chains `(8,8,2)` replication; an oracle_orth gate pass therefore
does not imply a second orthogonal-code experiment. F1 uses ranks `64/128` and **20,000**
steps with no replication. F2 uses rank `32`, **15,000** steps with no replication.

| ID and paper sentence | Status | Numbers, interval, and set | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| S1. “With orthogonal oracle writing, increasing the reads of one fixed checkpoint improved reachable-hop accuracy and passed the registered P1-S gate.” | supported | C777: M1 D **+92.883333 pp [92.233333,93.550000]**, W **+92.872222 [92.208299,93.563924]**, H **+17.783333 [15.216667,20.150208]**; `passed=true`. | Primary oracle_orth, shared M1, r32, mixed training depth; gate includes the random-code replication described above. | P1G `98e70eb`: `oracle_orth/M1`, reach/specificity/hop1. |
| S2. “The evolving-query system exceeded the fixed-query and memory-free controls under orthogonal oracle writing.” | supported | C777: M1−B5 Q **+92.891667 pp [92.149896,93.541667]**; M1−B3 **+85.233333 [84.233229,86.183333]**. | Same primary oracle codebook; separately trained reader variants, evaluated L4, r32. | P1G `98e70eb`: `oracle_orth/M1`, vs_fixed/vs_loop. |
| S3. “The unfactorized learned joint writer gave a smaller, seed-dependent depth benefit at rank32.” | narrowed | C777: M1 D **+8.291667 pp [7.383125,9.291771]**, W **+2.491667 [1.244410,3.733333]**, H **+4.683333 [3.616458,5.800208]**; Q **+7.766667 [6.516458,9.050000]**, but seed-specific Q is **+9.300000/−17.700000/+31.700000 pp**. `passed=true`. | joint/M1, mixed depth. D is within checkpoint; Q compares jointly trained systems with different writers. The pooled CI is not evidence of consistency across training seeds. | P1G `98e70eb`: `joint/M1`, statistics and per-seed Q. |
| S4. “The unfactorized joint M1-lin reader passed the primary rank32 depth gate.” | not supported | C777: D **+2.441667 pp [1.716667,3.191667]**, W **−0.408333 [−1.475000,0.753125]**; `passed=false`. | joint/M1-lin, r32; fails the registered minimum reach effect and specificity, including replication specificity. | P1G `98e70eb`: `joint/M1-lin`. |
| S5. “The tested unfactorized T1-pretrained frozen writers did not pass the replication-specificity criterion.” | supported | C777 replication W: M1 **+0.675000 pp [−0.069618,1.494583]**; M1-lin **+0.200000 [−0.647326,1.050069]**. Primary M1 D was **+8.125000 [7.358333,9.000000]**. | t1_frozen/r32, frozen writer shared across variants; typed learned-code replication. This does not show that joint training is necessary for every composable writer. | P1G `98e70eb`: t1_frozen, primary reach and replication specificity. |
| S6. “Random oracle codes exhibited positive depth effects but did not preserve hop1.” | supported | C777: M1 D **+33.258333 pp [32.016250,34.483542]**, H **−18.933333 [−21.050417,−16.799583]**; M1-lin D **+47.325000 [45.808125,48.833438]**, H **−29.000000 [−31.316667,−26.666458]**. Both fail only hop1. | oracle_random/r32, shared readers, mixed depth. Random codes with delta writes are not interference-free oracle memory. | P1G `98e70eb`: oracle_random, reach/hop1. |
| S7. “The trained orthogonal-code linear reader implements exact one-hop-per-read terminal following.” | not supported | C777 point accuracies, CI unavailable: oracle_orth/M1-lin hop4 at L4 **2.483333%**, hop8 at L8 **0.000000%**; hop1 L1/L4/L8 **99.566667/96.550000/26.300000%**. Yet hop4/5/6 at L8 are **98.650000/99.933333/97.066667%**. | r32, trained hops≤3 and L≤4. These support selected longer-hop successes and late-depth damage, not exact alignment or universal extrapolation. Prop.3's full alignment/gain/decoder assumptions were not established. | P1A `98e70eb`: oracle_orth/M1-lin, primary C777 original cells. |
| S8. “The rank follow-up found larger learned-code depth effects in the tested larger-rank training regimes.” | narrowed | C777: joint/M1 r64 D **+30.533333 pp [29.224896,31.900000]**, W **+20.927778 [19.052361,22.930556]**, Q **+24.016667 [22.575000,25.575208]**; seed Q **+22.575000/+25.075000/+24.400000 pp**. r128/M1-lin D **+40.591667 [39.133333,41.983542]**; hop8 at L8 **63.783333%** (absolute point, CI unavailable). | All joint r64/r128 reader rows pass the follow-up's unreplicated gate. F1 changes rank **and** training duration, 20,000 versus P1-S's 15,000 steps; do not attribute the contrast solely to rank or claim replication/new-seed robustness. | F1G `960a84f`: joint rows; F1C `8798483` for training duration. |
| S9. “Factorized learned writing yielded positive, walk-specific depth effects, including with a frozen T1 writer, but failed hop1 preservation.” | supported | C777, order joint_fact M1/M1-lin; frozen_fact M1/M1-lin. D pp: **+22.975000 [22.016667,23.975104] / +27.608333 [26.533333,28.708333] / +11.683333 [10.975000,12.433333] / +11.575000 [10.825000,12.350000]**. W pp: **+20.158333 [19.058333,21.191875] / +24.813889 [23.694375,25.964167] / +8.944444 [8.158194,9.722257] / +9.125000 [8.344306,9.897222]**. H pp: **−35.616667 [−37.217083,−33.933125] / −37.466667 [−39.100000,−35.850000] / −29.816667 [−31.367083,−28.199375] / −27.933333 [−29.483542,−26.416667]**. All G1 false, G2 true. | F2/r32, shared-step readers, mixed depths, no replication. Frozen rows are `t1_frozen_fact`. Positive reach/specificity do not constitute a full gate pass. | F2E `a898c87`: confidence .975, reach/specificity/hop1 and gate criteria. |
| S10. “Factorization improved the depth contrast without improving final L4 reachable-hop accuracy over the unfactorized joint system.” | supported | C777 joint_fact−joint, M1: L4 reach **−19.108333 pp [−20.533646,−17.599896]**; difference of D **+14.683333 [13.333333,16.025000]**. | r32, 15,000 steps, mixed depth, jointly trained writers per variant. The changed L1 baseline is part of the depth-effect difference; a larger D need not mean a better L4 system. | F2E `a898c87`: cross_family_contrasts.csv, joint_fact/joint M1. |
| S11. “Successor-swap diagnostics are consistent with path-sensitive behavior, but do not identify a unique learned algorithm.” | narrowed | S4242, CI unavailable. P1-S L4, `new_hop_le_L=true`, pooling original hops1–8: oracle_orth/M1-lin follows the new terminal **65.594446%** versus unswapped **0.172541%** (`15967/24342`, `42/24342`); joint/M1 **44.219867%/35.292909%** (`10764/24342`, `8591/24342`); joint/M1-lin **48.336209%/47.913072%** (`11766/24342`, `11663/24342`). F2 joint_fact/M1, **new_path_length=1** only: **47.985348%/4.426129%** (`1572/3276`, `145/3276`). | Screening counterfactuals, not confirmation. F2 and P1-S strata differ, so their rates do not estimate a controlled effect of factorization. Responses to local path changes can coexist with global terminal statistics. | P1S `98e70eb`: swap_successors.csv; F2A `a898c87`: swap_successors_new_path_length.csv, primary/L4 strata above. |
| S12. “Factorized T1 validation performance alone establishes multihop terminal retrieval.” | not supported | V4242, CI unavailable: factorized frozen-writer source T1 accuracies **91.800000/91.200000/92.400000%** (seeds17/29/43). With those writers frozen, F2 t1_frozen_fact/M1 C777 h1/h2/h3/h4 at L1: **85.100000/1.983333/1.033333/1.166667%**; at L4: **55.283333/18.833333/7.550000/4.333333%**. | T1 writer validation and T2 reader confirmation are different tasks. The matched frozen-writer profile supports some multihop use but still fails the full gate (S9); T1 loss/accuracy alone does not certify it. | F2A `a898c87`: writer_t1_scores.csv (`writer=joint_fact`, names f2-writer-s17/s29/s43, the frozen-writer sources) and accuracy_summary.csv (`t1_frozen_fact/M1`). |

## Larger single reads, sharing, and measured budgets

P2-S primary endpoint models and all P2-S-F1 models use `15,000` steps and the same
training seeds. The P2-S M1 reference is r32, trained at mixed depths and evaluated at L4;
B6/B7 are trained/evaluated at L1, B8 at fixed L4. The follow-up adds shared M1 trained
only at L4 and factorized B8/B6/B7, with unchanged mixed-depth references re-evaluated on
C888. Independently selected protocols are part of these comparisons. The selected numeric
M1 learning rate is nevertheless **0.003 for both fixed and mixed training** for every
writer; differing actual learning-rate values must not be invented as an observed confound
(P2FM; P1A plan; F2M). All F-E3 contrasts below use **mixed-depth M1**, not M1-L4.

| ID and paper sentence | Status | Numbers, interval, and set | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| B1. “A larger single read could not replace the shared four-read system with unfactorized learned writing.” | not supported | C888 M1−B6/r64 reach **+14.875000 pp [13.658333,16.183542]**, but −B6/r128 **−3.233333 [−4.675000,−1.833125]**; −B7/r_mem64 **−1.458333 [−2.858438,−0.125000]**, −B7/r_mem128 **−2.283333 [−3.591771,−0.891562]**. r64 contrast by seed **+33.125000/−20.675000/+32.175000 pp**. | joint, M1 r32/L4/mixed-depth; larger one-read models. Seed29 is negative in every E1/E2 contrast. Gate3 capability is false. | P2E `bb101a6`: E1/E2, joint, reach. |
| B2. “With random oracle codes, four reads outperformed both tested larger-rank single-read systems on reachable hops.” | supported | C888 M1 reach **34.775000%** versus B6/r64 **0.025000%** and r128 **0.000000%**; differences **+34.750000 pp [33.466667,36.075000]** and **+34.775000 [33.491667,36.083438]**. All tested-seed effects positive. | oracle_random; M1 r32 mixed-depth/L4 versus B6 L1. Only the tested B6 ranks, not arbitrary one-read functions. Absolute accuracies have no separate CI. | P2E `bb101a6`: E1, oracle_random, reach, corresponding accuracy.csv cells. |
| B3. “Step-specific four-read systems exceeded shared four-read systems even after matching training depth.” | supported | C888 B8−M1-L4 reach: joint **+13.216667 pp [11.983333,14.450000]**; joint_fact **+5.458333 [4.241667,6.750000]**; oracle_random **+3.200000 [1.958229,4.516771]**. Every tested-seed reach effect positive. | Both trained/evaluated fixed L4, r32. Matching depth does not match parameter count: P2C joint B8/M1 counts are **22,530/13,314**. Joint writers are trained separately. This is a trained-system advantage, not a parameter-neutral isolation of sharing. | P2FE `ae7bda6`: F-E1/reach; P2C `bb101a6`: joint B8/M1 counts. |
| B4. “The selected fixed-depth training protocol improved shared M1's reachable-hop accuracy over the mixed-depth reference.” | supported | C888 M1-L4−M1-mixed: joint **+20.516667 pp [19.208333,21.783333]**; joint_fact **+3.483333 [2.249792,4.758333]**; oracle_random **+8.241667 [7.116667,9.400000]**. Joint B8−M1 gap shrinks from **33.733333** to **13.216667 pp**. | Shared r32, evaluation L4; separately trained/selected protocols, all 15,000 steps. Descriptive gap reduction does not identify a unique causal percentage attributable to depth alone. | P2FE `ae7bda6`: F-E2/reach; P2E `bb101a6`: E3/joint/reach. |
| B5. “With factorized joint writing, four reads exceeded every tested larger single-read rank/state configuration on reachable hops.” | supported | C888 M1-mixed−B6/r64 **+24.016667 pp [23.000000,25.108333]**; −B6/r128 **+24.241667 [23.241563,25.333333]**; −B7/r_mem64 **+23.875000 [22.858333,24.975104]**; −B7/r_mem128 **+23.550000 [22.533333,24.658542]**. Every tested-seed contrast positive. M1 reach **24.483333%** versus **0.466667/0.241667/0.608333/0.933333%**, respectively (absolute points, CI unavailable). | joint_fact; M1 r32 mixed-depth/L4; B6 L1 at r=r_mem64/128, B7 L1 at r32/r_mem64/128. Capability at these budgets, not universal single-read impossibility or a measured efficiency frontier. | P2FE `ae7bda6`: F-E3/reach and matching accuracy.csv cells. |
| B6. “The unshared reader was better on every hop, or demonstrated superior depth extrapolation.” | not supported | C888 B8−M1-L4 **long** accuracy: joint **−5.533333 pp [−6.533625,−4.466583]**, joint_fact **−0.753333 [−1.076750,−0.433333]**, oracle_random **−0.536667 [−0.796708,−0.279958]**. Pooled hop1 gains are **+4.450000 [3.300000,5.750208] / +6.466667 [4.500000,8.316667] / +8.216667 [6.466458,10.017083]**, respectively; joint seed43 hop1 is **−0.850000 pp**. | Fixed training/evaluation L4, r32. Long averages are over untrained hop lengths, not larger evaluation L; B8 has no legal shared-weight depth scan. Pooled hop1 improvement is not every-seed improvement. | P2FE `ae7bda6`: F-E1, subsets long/hop1, per_seed. |
| B7. “The symbolic joint-writer study established a general practical accuracy–latency advantage for repeated reads.” | not supported | C888 descriptive cost points, CI unavailable: joint M1 r32/L4 hops1–3 **48.016667%**, **0.847758 ms/batch64**; joint B6 r128/L1 **57.244444%**, **0.352139 ms/batch64**. Every eligible joint M1 L1/2/4/8 point is matched by an equal-or-faster one-step point for batch1 and batch64. `capability_passed=false`, `practical_passed=false`. | P2-S Gate3 fails. M1 has **13,314** trained parameters and **4,096** persistent FP32 bytes/episode versus B6/r128 **114,690** and **65,536**: the points trade different static budgets. Timings measure prepared-query batches, not end-to-end inference. | P2C `bb101a6`: cost.csv, joint/source wph M1 and source p2s B6; practical.csv; P2E decision.json. |
| B8. “Factorized learned writing gives a favorable accuracy–latency frontier.” | open | C888 supplies the positive **accuracy** contrasts in B5. **No P2-S-F1 latency benchmark or per-model cost CSV was committed; timing CI unavailable.** | Static architecture budgets and capability contrasts cannot supply missing factorized timings, writer/preparation cost, per-model comparative training cost, or end-to-end cost. Aggregate stage budget ledgers exist. The oracle M1/L4 point is unmatched in P2-S's latency test, but that does not pass the joint-writer/global gate; the factorized capability result does not retroactively pass it either. | P2FE `ae7bda6`: audit/decision and released inventory; P2C `bb101a6`: practical.csv, oracle_random M1 L4. |
| B9. “The follow-up selected a reader protocol for the next stage, without establishing its LLM performance.” | supported | C888: `reader_for_P1L=B8`, fixed training L4; selection uses F-E1 sharing and F-E2 training-depth outcomes (claims B3/B4). No LLM effect or CI is supplied by these symbolic files. | B8 fixed L4 with shared M1-L4 comparator; protocol choice, not a symbolic same-checkpoint depth scan or an end-to-end LLM result. | P2FE `ae7bda6`: decision.json. |

P2-S timings use FP32 on GH200, `5` warmups and `20` repeats. They include reader,
output projection, and candidate decoding of prepared inputs, excluding context/writer
preparation and input projection. State bytes are per episode, not the whole batch.
The primary gate excludes the descriptive F1 `20,000`-step references. Parameter/state,
analytic FLOPs, measured prepared-batch latency, and total training cost are different
budgets; success against one must not be reported as success against all.

## Stopping and certificates

P4-S reuses the mixed-depth F1 checkpoints (r64/r128, `20,000` training steps).
P4-S-r32 reuses the mixed-depth P1-S/F2 checkpoints (r32, `15,000` steps). There is
no retraining. All are shared-step readers. Each family/rule threshold is selected on
V4242 by mean accuracy over hops1–8 and the three training seeds; ties prefer fewer
steps, then the smaller threshold. That objective is not hop1 preservation. Stopping
occurs at the first qualifying post-update state, with minimum `L=1` and cap `L=16`.
Read-norm stopping uses **L+1** memory reads;
relative-change and output-stability stopping use **L**. Every number in this table is
descriptive: **CI unavailable**, with no confirmation set opened.
P4-S timings use FP32 on GH200, with `5` warmups and `20` repeats (P4 manifest/latency,
P4D); spectral component norms for the sufficient certificate use FP64.

| ID and paper sentence | Status | Numbers and set (CI unavailable throughout) | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| H1. “Validation-selected relative-change stopping improved the screening long-hop accuracy of the tested r128 random-oracle readers, while retaining a hop1 loss relative to L1.” | narrowed | S4242 M1: long L8 **45.746667%** → stopped **70.690000%**; hop1 L1 **99.983333%**, stopped **87.883333%**; mean updates **8.301167**. M1-lin: long L8 **44.846667%** → **64.410000%**; hop1 **100.000000%** → **96.616667%**; mean updates **7.147146**. V4242 selected thresholds **0.7/1**, respectively. | oracle_random/r128. No full recovery of hop1, no formal gate pass, no CI; no inference that other ranks/writers receive the same benefit. | P4 `9dd0e04`: headline.csv and thresholds.csv, oracle_random/r128/relative_change. |
| H2. “Fewer adaptive updates made the tested stopping implementation faster than fixed-depth evaluation.” | not supported | Prepared batch64, CUDA mean ms, fixed L16 versus relative-change: oracle_random/r128/M1 **2.912083 → 11.058517**; M1-lin **2.691596 → 10.293335**. All **48** adaptive family×rule×batch cells are slower than fixed L16 in both CUDA and wall means. | S4242 pilot implementation, batch1/64; active-query bookkeeping and candidate decoding included, preparation/backbone/writing excluded. This is not a claim about every query or a future optimized implementation. | P4 `9dd0e04`: latency.csv, eight families × three rules × two batch sizes; P4D `7888cf0`. |
| H3. “The sufficient factorized norm test certified none of the tested P4-S episode evaluations.” | supported | V4242+S4242: **0/96,000** bounds below one; minimum **4.571932**, maximum **37.074453**. | Bound is `abs(rho)*(norm(R)+norm(U)*norm(M)*norm(Q))`, with FP64 spectral component norms. It is an upper bound on the exact contraction expression, not its measurement. Repeated checkpoint evaluations are not independent held-out episodes. No r32 certificates were computed. Failure to certify is not proof of noncontraction/divergence. | P4 `9dd0e04`: contraction.csv, audit.json; P4D `7888cf0`. |
| H4. “The tested label-free stopping policies repaired rank32 factorized overshoot while preserving hop1.” | not supported | S4242: all **12** factorized family/rule choices fail hop1 preservation. Even the least-loss rule **descriptively chosen on screening** in each family gives H: joint_fact/M1 **−33.033333 pp**, joint_fact/M1-lin **−41.733333**, t1_frozen_fact/M1 **−29.683333**, t1_frozen_fact/M1-lin **−24.900000**. Their D/W are respectively **+21.350000/+17.905556**, **+21.458333/+15.102778**, **+10.233333/+7.327778**, **+9.516667/+6.988889 pp**. | r32. Rules in the same order: relative_change .3, output_stability 4, output_stability 4, read_norm 1. D/W positive in every tested seed still do not compensate for H failure. This does not exclude every possible halting method. | P4R `3e15f07`: endpoints.csv, endpoints_per_seed.csv, thresholds.csv; P4RD `ce77e74`. |
| H5. “The unfactorized r32 joint/M1 screening flag establishes a benefit from adaptive stopping.” | not supported | S4242: all three rules use exactly **16** mean updates and reproduce fixed-L16 endpoints: H **+5.150000 pp**, D **+8.666667**, F **+6.027778**, W **+2.638889**; minimum seed W **−0.125000 pp**. Read-norm costs **17** reads. | joint/M1/r32. Aggregate `screening_candidate=true` is a point-estimate flag, not confirmation or a formal G1 pass. There is no stopping benefit over fixed L16 here. | P4R `3e15f07`: endpoints.csv, endpoints_per_seed.csv, accuracy.csv. |
| H6. “Stopping at the true graph distance is an accuracy upper bound for a learned reader.” | not supported | S4242 oracle_random/r128/M1 overall accuracy: true-hop **23.564583%**, relative-change **75.302083%**. | True-hop stopping is a label-informed diagnostic reference, not an upper bound. Learned update count need not equal graph distance. | P4 `9dd0e04`: headline.csv, oracle_true_hop/relative_change. |

## Theory-to-measurement ledger

The propositions are conditional mathematical results, not fitted explanations of these
checkpoints. Their proofs and counterexamples are in [theory.md](../theory.md): Props.1–9
introduced in `9d2100b`, Prop.10 in `8818756`, Prop.11 in `628ba19`. The table distinguishes
compatible observations, unmeasured assumptions, and contradictions of an **unconditional
experimental interpretation**. No theorem is declared empirically refuted merely because
its sufficient assumptions were not verified. The CIs/sets and numerical source commits
for referenced claim IDs remain those above.

| Proposition used | Prediction under its assumptions | Corresponding measurement and evidence status | Paper use / boundary |
| --- | --- | --- | --- |
| 1: fixed output subspace and frozen-memory input bottleneck | The branch stays in col(B); at fixed M its local input-Jacobian rank is at most r. Depth may change the function inside that bottleneck. Sequence history derivatives need not have this global rank bound. | S1–S3 show accuracy can change within a fixed-r checkpoint; P1A's audit verifies identical snapshots across L. Compatible, **not** a measurement of output-subspace or Jacobian rank; neither was estimated in these symbolic exports. | supported as a theorem, open as a direct rank measurement. Fixed col(B) does not imply useless depth; do not apply the frozen-memory rank bound to the stacked sequence map. |
| 2: linear matrix polynomial | Shared linear iteration gives `sum_{j=0}^L T^j u`, `T=rho*(R+UMQ)`; memory composition degree can increase with L. At fixed snapshot/L a linear map may be precomputed. | M1-lin depth profiles S4/S6/S7/S8 are compatible with depth-dependent composition and failure. They do not verify a particular polynomial coefficient/gain or show that more degree guarantees correctness. B8 has different per-step matrices. | supported algebraically; accuracy alone does not identify the polynomial or prove serial reads computationally unavoidable. No amortized precomputation benchmark was measured here. |
| 3: exact chain-terminal decoding | Orthonormal codes, exact oracle writes, nilpotent chains, aligned linear transition with gain greater than one, and matched decoder imply the correct visited node and terminal stability after enough reads. | S7 contradicts the blanket assertion that a **trained** oracle reader automatically has exact one-hop timing or terminal stability. Alignment/gain/decoder assumptions were not measured, so it does **not** contradict the proposition. H6 also shows learned update count is not graph distance. | supported as an existence construction; narrowed empirical claim to the measured hop/depth cells. Exact oracle storage alone is insufficient. |
| 4: interference amplification | Delta writes with nonorthogonal keys can erase earlier associations; an operator perturbation bound permits depth-amplified error. Growth is an upper-bound possibility, not universal monotonic damage. | S6's random-code hop1 damage and H1's incomplete recovery are compatible. S8's larger-rank gains are compatible but also change training duration. These tables do not measure edge/terminal residuals or isolate the bound's tightness/causal contribution. | use as a mechanism to investigate, not a proved explanation of every loss or a universal `sqrt(E/r)` law for learned chain values. |
| 5: conditional single-read information limit | B4/B5 receive memory only through one `MQu`; identical full observations force identical predictions. The chain Bayes ceiling additionally needs exact aligned first-hop access and the stated side-information restrictions. | S2 and B2/B5 support limits of **tested** one-read/fixed-query systems. They do not measure indistinguishable-observation pairs or verify a universal Bayes ceiling for arbitrary trained Q/decoders. The proposition explicitly supplies a counterexample to an unrestricted one-read impossibility interpretation. | narrow the capability claim to the tested architectures/ranks/writers. Candidate labels, a general learned query, and nonorthogonal leakage matter. |
| 6: bias-free ReLU code transmission | A width-r bias-free through-map cannot transmit a positively spanning signed code set exactly; this is not a theorem of universal classification failure for the residual reader. | S1's ReLU success alongside linear hop1 failure is compatible with the theorem: the full reader has reinjection, Rz, and a learned decoder. No exported measurement isolates value-to-query identity transmission. | no claim that the symbolic results disprove the obstruction, or that identity must outperform ReLU for every trained reader. |
| 7: contraction and stopping error | A valid constant below one bounds distance to a fixed point; answer correctness additionally requires task/decoder information. A sufficient norm upper bound failing does not prove divergence. | H3 supplies **no successful certificate**; H1/H4 stopping policies are empirical. H2 contradicts inferring runtime speedup from fewer updates. Neither a small relative change nor output stability supplies the missing certificate or a correctness proof. | use the negative certificate result and descriptive stopping tradeoff; do not claim certified halting or convergence implies correct answers. |
| 8: exact low-dimensional elimination | A recurrence whose entire update lies in a fixed col(B), with reproducible gates, can be represented exactly in low-dimensional coordinates. Extra high-dimensional operations need a separate proof. | No original-versus-eliminated high-dimensional recurrence equivalence/latency comparison is in P1-S/P2-S/P4-S. Their registered V2 recurrence is already low-dimensional. P2C/H2 timings do not measure this elimination. | supported as an algebraic equivalence; open as a measured speedup in this evidence set. Do not claim a new function class or that any full transformer loop can be eliminated. |
| 9: repeated BA and initialization | Shared linear powers factor through AB; a pure double-BA branch with zero B and zero-at-zero locally Lipschitz activation has zero first derivative. Skip/gated alternatives require their stated nondegeneracy. | No primary symbolic B9 startup/optimization endpoint is included in these CSV/JSON results. Existing theory examples are constructed checks, not an empirical comparison of the trained symbolic readers. | supported analytically; open as a benchmark claim. Do not transfer the pure double-BA dead start to V2's single output factor. |
| 10: writer shortcut and factorization | Under the explicit common-key/EMA assumptions, endpoint statistics can expose terminals without path following; tail-only factorization removes the direct head-subtraction term, with other shortcuts still possible. | S3/S4's weak/flat depth behavior, S9/S10/S12's changed factorized profiles, S11's swaps, and B1 versus B5 are consistent with this account. They do not measure the exact common-key representation or uniquely identify “pure terminal guessing.” | narrowed mechanistic interpretation. Neither a positive depth contrast/A5 rate proves exact pointer chasing, nor factorization proves all shortcuts absent. |
| 11: exact delta read and value-to-answer path | Signed read coefficients depend on later-write key erasure; SIR depends on the value Gram matrix. Normalized addressing loss does not set raw read gain or guarantee downstream decoding. | These symbolic exports do not supply the signed coefficients, value-Gram SIR, raw query/residual scales, or gradient routing needed to test those mechanisms in the trained models. T1-versus-T2 in S12 illustrates the gap in sufficiency, but is not a fitted validation of Prop.11 or of an LLM diagnosis. | supported derivations; open empirical attribution. Do not infer a training run's failure cause from key cosines, addressing loss, or auxiliary value decodability alone. |

## Discrepancies and interpretation corrections

The numerical audit found **no material disagreement at the reports' stated rounding** in
the selected gate/endpoints, stopping tables, or timing examples. It did find the following
scope/interpretation discrepancies; the paper should use the qualified versions above.

1. [P1-S decision](../../reports/decisions/P1S.md) §3's “walk one hop per read” is too exact
   for the trained models (S7). Its “pure terminal guessing” labels are a plausible mechanism,
   not an identified algorithm (Prop.10/S11). The oracle B6 multihop value rounded to zero in
   the [P2-S decision](../../reports/decisions/P2S.md) is actually **0.025000%** for r64 reach,
   versus **0.000000%** for r128 (B2).
2. “Composable codes need joint training” in the P1-S decision is broader than its tested
   unfactorized T1 failure. Frozen factorized writers have positive D/W in F2 (S9), though
   they still fail hop1 preservation. “Robust at larger rank” is limited to the fixed tested
   seeds and the longer, unreplicated F1 protocol (S8).
3. F2 “removes the terminal shortcut” / P2-S-F1 “shortcut-free” must mean the particular
   head-subtraction statistic removed by the input partition, not every possible shortcut.
   A larger D is not higher L4 accuracy (S10), and A5 rates from different strata are not a
   causal factorization comparison (S11).
4. The P1-S decision §7 explicitly records a **preregistration deviation**: the F2 no-go rule
   would have stopped P1-L after both factorized conditions failed; the orchestrator instead
   continued using the earlier joint/M1 pass and let F2 decide writer form. That deviation
   was recorded before P1-L T2 runs. This synthesis does not relabel F2 as a gate pass.
5. P2-S-F1's gap reduction is a comparison of selected training protocols (B4), not a unique
   causal percentage “due to training depth.” Numeric M1 learning rates actually agree;
   independent selection is not evidence that the selected values differed. Unsharing also
   increases parameters (B3), so matched depth is not matched parameter budget.
6. “Shared reads extrapolate further” in the P2-S decision §6 means the pooled **long-hop
   average at evaluation L4** (B6); it is not an extrapolated B8 depth scan. Likewise pooled
   hop1 gains are not gains in every training seed. The registered follow-up reuses C888.
7. Gate3's joint practical failure remains a failure (B7). Factorized F-E3 is a capability
   comparison with no factorized latency benchmark (B8). Static parameter/state ratios are
   not measurements of end-to-end efficiency.
8. P4-S contraction counts span validation **and** screening, including checkpoint reuse
   (H3). They are not independent confirmation episodes. The r32 `screening_candidate` flag
   has no CI and can coexist with negative seed-specific specificity and no actual early
   halting (H5). Fewer updates did not give lower measured latency (H2).
9. Candidate-restricted and full-vocabulary metrics must not be mixed. For example F2
   joint_fact/M1 C777 hop1/L1 is **86.083333% candidate** but **40.516667% full-vocabulary**;
   hop2/L4 is **32.333333% candidate** but **7.600000% full-vocabulary** (F2A `a898c87`,
   matching accuracy-table fields). No candidate accuracy above is a full-vocabulary EM claim.

## Statements the paper must not make

These exclusions incorporate [plan §14.3](../../plan/recurrent_lora_research_plan_2026-10-26.md)
and the empirical/theory limits above.

- “This is the first recurrent LoRA,” “the first interpretation of linear attention as dynamic
  weights,” or “the first evolving-query inner loop over fixed fast weights.” The dated
  [literature audit](../reports/lit_check.md) (`dbead6f`) identifies prior fixed-memory inner
  loops and limits any novelty claim. An incomplete search does not prove absence.
- “RecLoRA lacks nonlinearity/token mixing,” or a verified page-level comparison or withdrawal
  claim about it. The literature audit could not open its full text; the withdrawal label was
  only a search snippet. Preserve that uncertainty rather than treating it as verified.
- “Increasing loop count alone gives reasoning,” “all oracle readers walk exactly one hop per
  read,” “more reads always improve accuracy,” or “symbolic reader selection proves an LLM
  reasoning gain.” The evidence supports specific composition contrasts and several failures.
- “Convergence/residual stability implies a correct answer,” “these stopping rules are
  certified,” or “fewer updates imply faster inference.” State error, correctness, and measured
  latency require different evidence (Prop.7, H1–H5).
- “An r² state stores r² arbitrary facts losslessly,” “fixed col(B) makes depth useless,” or
  “few extra parameters make extra computation almost free.” Capacity, function composition,
  state size, and runtime are separate quantities (Props.1–4, B7, H2).
- “No single-read model can compose,” “joint training is necessary for every composable
  writer,” “factorization removes every shortcut,” or “the learned reader has been proved to
  use pure pointer chasing/terminal guessing.” Theorems and measured systems have narrower
  assumptions (Props.5/10, S5/S9/S11, B1/B5).
- “The method passes the general efficiency gate,” “the factorized follow-up measured an
  accuracy–latency advantage,” or “prepared-query timings establish end-to-end/training
  savings.” P2-S Gate3 fails and the missing measurements remain open (B7/B8).
- “Every seed improves,” “episode CIs establish new-training-seed robustness,” “the reused
  confirmation sample is independent replication,” or “screening stopping flags are
  confirmed gate passes.” Report the actual set, seed effects, selection, and CI scope.
- “Low normalized addressing/value auxiliary loss guarantees a useful raw memory read or
  final answer.” The missing scale, erasure, interference, and decoding conditions are explicit
  in Prop.11 and are not measured by this symbolic evidence table.

## Audit boundary

All displayed empirical numbers were read from the committed CSV/JSON sources identified
above, using bounded streaming CSV/JSON extraction and checking file provenance with git.
Only fraction-to-percent conversion, equal-seed means already defined by the exports, and
the explicitly disclosed A5 count pooling were used; no bootstrap was rerun and no new CI
was invented. The decision/report Markdown was used to find claims and discrepancies,
not as the numerical source. This task loaded no models or tensor traces, ran no training,
evaluation, tests, or numerical experiments, and submitted no compute jobs, as required by
the WP-T4 no-new-runs instruction. Earlier theory checks are historical evidence only.
