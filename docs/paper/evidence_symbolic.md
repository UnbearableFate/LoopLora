# Symbolic claims and evidence (WP-T4)

Evidence cutoff: `master` at `5d2efd3273ba62fd691219af5f20fdc01cd7bf12`, merged into
`wp/theory` for WP-T7. The original WP-T4 audit covered `30f86e2`; WP-T6 added IS
(`e144246`, merge `641911d`), and WP-T7 adds P4-S-L (`95736ec`, merge `9c84f4c`)
and JS (`96ff2e9`). This is a synthesis of committed results, not a new experiment or a reanalysis of
sample traces. The defensible result is conditional: evolving queries over a fixed memory can
improve composition, but the benefit depends on the writer, training protocol, and comparison
budget. The current comparison policy asks for improvement over the same-rank single read,
with additional inference latency reported as a cost (FR1). The earlier symbolic P2-S/P2-S-F1
larger-rank comparisons retain their original findings; the original joint practical gate failed.
WP-T9 (2026-10-27) adds scope statements, reuse and B8-overhead annotations, a pending-measurement row,
and further exclusions after two external reviews; it adds no numerical source and changes no status.
WP-T10 part 1 (2026-10-29, on `master` `abc2fa1`) adds the registered results of the follow-up studies IVD, CVH, TSM,
UNS and NCR (rows IV1–IV3, CV1–CV4, TS1–TS2, UN1–UN4, PM1 and NC2; sources IVE, CVE, TSE, UNA/UNB and NCE below),
fills PM1, and updates the scope, exclusions and audit boundary. It changes the status of no earlier row.

## Reading the tables

Each quoted claim is a candidate paper sentence; its status applies to that sentence with the
conditions in the same row. **Supported** means supported within those conditions;
**narrowed** requires the stated restriction; **not supported** identifies a sentence to reject;
**open** means the required measurement is absent. These are evidence judgments, not new tests. **Descriptive** marks a row whose preregistration attaches no gate or
decision rule: it reports numbers, and the paper may use them only for the wording that the preregistration attaches.

All accuracies below are **candidate-restricted**, not full-vocabulary exact match. `%` denotes
accuracy; `pp` denotes a difference of accuracies. Values are converted from committed fractions
and rounded to **six decimal places**; the linked files retain the exact stored precision.
Brackets are **97.5% episode-bootstrap confidence intervals for the displayed contrast**,
unless explicitly marked as an absolute-rate CI, a P4-S-L **99.166667% adjusted CI**,
or **CI unavailable**. Comparisons within
a load/set are paired; IS1's change across loads uses independent load bootstrap streams.
No interval is supplied for an absolute accuracy merely because its contrast has one.
IS4 explicitly uses the separately exported absolute-rate intervals, not contrast intervals.
Confirmation intervals use the same fixed training seeds
`17/29/43`, with `2,000` episode-bootstrap draws using purpose-derived streams from base
seed `2718`, with pairing/independence as specified above; they condition
on those training seeds and do not quantify uncertainty over new training seeds. They are the
registered endpoint intervals, not simultaneous intervals for every sentence in this synthesis.

Set labels used throughout:

| Label | Set and selection | Scope |
| --- | --- | --- |
| C777 | Confirmation, split `test`, evaluation seed `777001`, `2,000` episodes. | P1-S, rank follow-up P1-S-F1, factorized-writer follow-up P1-S-F2. Reusing this set across follow-ups is not independent replication. The P1-S registered replication changes the layout; it is distinct from rerunning the primary layout. NCR re-reads its saved predictions (descriptive, no new evaluation). |
| C888 | Confirmation, split `test`, evaluation seed `888001`, `2,000` episodes. | P2-S and P2-S-F1. The follow-up explicitly reuses the P2-S confirmation set as registered; it is not a fresh confirmation sample. |
| C999 | Confirmation, split `test`, evaluation seed `999001`, `2,000` episodes per load. | IS, new relative to C777/C888. Within-load contrasts pair episodes across readers/depths and the fixed seeds; the E36−E12 depth-effect change uses independent load-specific bootstrap streams. IS4 resamples whole episodes within its update-task set. |
| C999002 | Confirmation, split `test`, evaluation seed `999002`, `2,000` episodes. | P4-S-L; frozen readers and learned heads, validation-selected thresholds locked before confirmation. 99.166667% endpoint intervals. |
| C999003 | Confirmation, split `test`, evaluation seed `999003`, `2,000` episodes. | JS; 12 newly trained RS/RS+J and 12 unchanged reference models, evaluated on common fresh episodes. 97.5% endpoint intervals. |
| C999004 | Confirmation, split `test`, evaluation seed `999004`, `2,000` episodes per checkpoint. | IVD; 12 frozen r32 checkpoints (oracle_random, joint_fact × M1, B4 × seeds 17/29/43), new relative to every earlier set. 97.5% endpoint intervals. |
| C999005 | Confirmation, split `test`, evaluation seed `999005`, `2,000` episodes per reader. | CVH; nine frozen P4-S-L M1 readers (three families × seeds 17/29/43) with the A1 and A2 halting heads. 99.166667% endpoint intervals. |
| C999006 | Split `test` and `test_unseen`, evaluation seed `999006`, `2,000` episodes per pool. | UNS Part B; 18 frozen checkpoints (P2-S-F1 M1 trained at L4 and B4, three writers, seeds 17/29/43). Descriptive 95% intervals. |
| C999007 | Confirmation, split `test`, evaluation seed `999007`, `2,000` episodes. | TSM; 18 newly trained arms (T0, T1, T2 × two writers × three seeds) and the released B4 reference. 98.75% endpoint intervals. |
| S4242 | Screening, split `test`, evaluation seed `4242`. | A5 diagnostics and stopping pilots. Stopping uses `2,000` episodes per checkpoint. No stopping CI was computed. UNS Part A audits its `test` and `test_unseen` screening evaluations, and NCR re-reads its saved predictions (descriptive, no new evaluation). |
| V4242 | Validation, split `val`, evaluation seed `4242`. | Earlier pilot thresholds frozen before S4242; P4-S-L and JS thresholds separately locked before C999002/C999003. Validation optima are selection results, not confirmation estimates. |

Rows whose set is marked “(reused)” cite C777 after P1-S or C888 after P2-S: an earlier study had
already opened that set, so a follow-up's interval there is not an independent confirmation.

For P1-S/F1/F2, let **D** = accuracy at `L=4` minus `L=1` on hops `{2,3}`;
**F** = the same difference on far hops `{6,7,8}`; **W** = `D−F` (walk specificity);
**H** = `L=4−L=1` on hop `1`; **Q** = evolving-query reader minus matched fixed-query
reader B5 at `L=4` on hops `{2,3}`. In P2-S/F1, **reach** is absolute accuracy averaged
over hops `{2,3}` at the listed evaluation depth. **Long** always means hops `{4,5,6,7,8}`,
and must not be substituted for the far set. In the r32 stopping table, D/F/H instead compare
the selected stopping rule with `L=1`; W remains D−F. P4-S-L uses these same H/D/F/W
hop definitions. JS uses hops `{1,2,3}` for its short-hop endpoints, not P1/P2 reach;
its overshoot statistic S is `Acc16−Acc4` on hops1–3.

Common task is `t2_chain_end`, primary layout `E=12`, chains `(8,4)`, entity dimension `64`,
and `rho=0.25`. The primary readers share their step matrices R/U/Q; M1 uses ReLU and
M1-lin uses identity. B5 caches one memory read, B3 has no memory, B6 increases rank, B7
increases memory dimension, and B8 has step-specific R/U/Q at fixed depth, so it is **not
parameter-matched** to the shared readers: +9,216 parameters at r=r_mem=32, about +69.2% in the
symbolic unfactorized joint model and +8.9% (rank 32) or +31.0% (rank 128) in the P1-L adapter (FR1).
A depth sweep of one checkpoint holds x/u/M fixed. A comparison of separately trained models does not.

Oracle codes are fixed across reader variants. `t1_frozen` and `t1_frozen_fact` share one
T1-trained frozen writer per training seed across reader variants. `joint` and `joint_fact`
train a separate writer together with each reader variant: their cross-variant contrasts are
comparisons of trained systems, not interventions holding the writer fixed. `joint_fact` and
`t1_frozen_fact` separate head/relation keys from tail-only values; this removes the direct
head-subtraction statistic identified in Proposition 10, without proving the absence of every
possible global shortcut.

## Scope of every row

These statements bound every row below; a row's own conditions add to them.

- **Vocabulary.** All entities come from the closed 1,024-name vocabulary (`V=1024`: seen pool
  [0,768), unseen pool [768,1024)). Every set used here (`train`, `val`, `test`) draws from the
  seen pool, with one exception: the UNS rows (UN1–UN4) evaluate the `test_unseen` split, 256 further
  names of the same vocabulary with fixed random features that no training episode used. Those rows are
  descriptive and support only the wording their preregistration allows. No row supports a statement
  about names outside the vocabulary, new entity codes, or open vocabularies.
- **Depth.** Effects are stated within the training depths and hops: hops 1–3 are trained; shared
  readers train at mixed depths {1,2,4} (B4 at L1), B8 and M1-L4 at fixed L4, and the JS readers at
  depths 4–8. Hops 4–8 and larger evaluation depths are reported as measured cells, including their
  failures (S7, B6, H1–H6, JS3), not as extrapolation claims.
- **Writers.** Oracle writing (orthogonal or random codes), learned writing (joint, factorized,
  T1-frozen), and the LLM's oracle and learned writers answer different questions and are separate
  claims; no row combines them.
- **Oracle design.** The symbolic oracle takes the question code, keys, values and decoder from one
  code table (`u=codes[start]`, `key=codes[head]`, `value=codes[tail]`, logits against `codes`), so
  every transition is identity-like. The LLM oracle (LP2 in the [LLM ledger](evidence_llm.md)) keys
  facts by independent random (entity,relation) pair codes and stores entity value codes from a
  different table, so each read after the first needs a learned value-to-key lookup. Symbolic
  results do not transfer to it.
- **Follow-up rows.** The rows of the follow-up section (IV, CV, TS, UN, PM1, NC) use frozen checkpoints
  (IVD, CVH, UNS, NCR) or newly trained arms (TSM) at r=r_mem=32 with three training seeds, and the
  "Use of the results" wording of their preregistrations. A result that fails its rule only narrows the
  paper's wording; no similar study follows. Comparisons of B4 and M1 there are same-rank; no row makes an
  equal-parameter claim except shared M1-L4 against B4.

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
| ISE | [is/endpoints.csv](../../reports/is/endpoints.csv), [accuracy.csv](../../reports/is/accuracy.csv), [decision.json](../../reports/is/decision.json), [best_L.csv](../../reports/is/best_L.csv), [audit.json](../../reports/is/audit.json) | `e144246` |
| ISU | [is/t3_update.csv](../../reports/is/t3_update.csv) | `e144246` |
| ISD | [is/design.json](../../reports/is/design.json), [training_manifest.csv](../../reports/is/training_manifest.csv) | `fb62cf1` |
| HLE | [p4s-l/endpoints.csv](../../reports/p4s-l/endpoints.csv), [endpoints_per_seed.csv](../../reports/p4s-l/endpoints_per_seed.csv), [accuracy.csv](../../reports/p4s-l/accuracy.csv), [qualification.csv](../../reports/p4s-l/qualification.csv), [latency.csv](../../reports/p4s-l/latency.csv), [analysis.json](../../reports/p4s-l/analysis.json) | `95736ec` |
| HLD | [p4s-l/design.json](../../reports/p4s-l/design.json), [label_free_thresholds.csv](../../reports/p4s-l/label_free_thresholds.csv) | `9366da4` |
| HLT | [p4s-l/theta.json](../../reports/p4s-l/theta.json), [theta.csv](../../reports/p4s-l/theta.csv), [head_manifest.json](../../reports/p4s-l/head_manifest.json), [theta_lock_audit.json](../../reports/p4s-l/theta_lock_audit.json) | `6e4545e` |
| JSE | [js/endpoints.csv](../../reports/js/endpoints.csv), [accuracy.csv](../../reports/js/accuracy.csv), [stopping.csv](../../reports/js/stopping.csv), [rho.csv](../../reports/js/rho.csv), [rho_per_seed.csv](../../reports/js/rho_per_seed.csv), [decision.json](../../reports/js/decision.json), [audit.json](../../reports/js/audit.json) | `96ff2e9` |
| JSD | [js/design.json](../../reports/js/design.json), [method.json](../../reports/js/method.json), [training_manifest.csv](../../reports/js/training_manifest.csv) | `852d293` |
| JST | [js/thresholds.csv](../../reports/js/thresholds.csv), [thresholds.json](../../reports/js/thresholds.json), [checkpoint_manifest.csv](../../reports/js/checkpoint_manifest.csv) | `0c99cc1` |
| FRP | [P1-L Amendments B4 and B6](../prereg/P1L.md), [plan E.2/§11.6](../../plan/recurrent_lora_research_plan_2026-10-26.md) | B4 `379115d`; B6 `26b96bd` |
| FRA | [core/accounting.py](../../src/looplora/core/accounting.py); [p1l-writer-fix/runs.csv](../../reports/p1l-writer-fix/runs.csv); P2C | `216febf`; `cd5a45c`; `bb101a6` |
| IVE | [ivd/results/interference.csv](../../reports/ivd/results/interference.csv), [cells_m1.csv](../../reports/ivd/results/cells_m1.csv), [cells_m1_full_vocabulary.csv](../../reports/ivd/results/cells_m1_full_vocabulary.csv), [accuracy_b4.csv](../../reports/ivd/results/accuracy_b4.csv), [summary.json](../../reports/ivd/results/summary.json) | `3fa622e` |
| IVD | [ivd/design.json](../../reports/ivd/design.json) (design hash `50746f54e0cb346190f176a5e258fdb9`) | lock `2a8d063`; code `d32463e` |
| CVE | [cvh/results/endpoints.csv](../../reports/cvh/results/endpoints.csv), [latency.csv](../../reports/cvh/results/latency.csv), [theta.csv](../../reports/cvh/results/theta.csv), [accuracy.csv](../../reports/cvh/results/accuracy.csv), [summary.json](../../reports/cvh/results/summary.json) | `96ee7ef` |
| CVD | [cvh/design.json](../../reports/cvh/design.json), [cvh/theta.json](../../reports/cvh/theta.json) | design `8a13c45`; theta lock `19c5613` |
| TSE | [tsm/results/endpoints.csv](../../reports/tsm/results/endpoints.csv), [descriptive_contrasts.csv](../../reports/tsm/results/descriptive_contrasts.csv), [accuracy.csv](../../reports/tsm/results/accuracy.csv), [training.csv](../../reports/tsm/results/training.csv), [gate_profiles.csv](../../reports/tsm/results/gate_profiles.csv), [summary.json](../../reports/tsm/results/summary.json) | `70bfdfc` |
| TSD | [tsm/design.json](../../reports/tsm/design.json) | `1bf689b` |
| UNA | [uns/part-a/endpoints.csv](../../reports/uns/part-a/endpoints.csv), [audit.json](../../reports/uns/part-a/audit.json), [reproduction.csv](../../reports/uns/part-a/reproduction.csv), [context.csv](../../reports/uns/part-a/context.csv), [accuracy.csv](../../reports/uns/part-a/accuracy.csv), [summary.json](../../reports/uns/part-a/summary.json) | `491e282` |
| UNB | [uns/part-b/endpoints.csv](../../reports/uns/part-b/endpoints.csv), [audit.json](../../reports/uns/part-b/audit.json), [accuracy.csv](../../reports/uns/part-b/accuracy.csv), [summary.json](../../reports/uns/part-b/summary.json) | `3fa622e` |
| UND | [uns/design.json](../../reports/uns/design.json) (Part B lock) | `a836928` |
| NCE | [ncr/accuracy.csv](../../reports/ncr/accuracy.csv), [categories.csv](../../reports/ncr/categories.csv), [summary.json](../../reports/ncr/summary.json) | `491e282` |

For P1/P2/IS/JS endpoint CSVs select `confidence=0.975`; both confidence levels occur
in the P2 files. P4-S-L instead exports `confidence=0.9916666666666667` throughout.
The follow-up files use IVD `0.975` (decision rows; the cell tables use `0.95`), CVH `0.9916666666666667`,
TSM `0.9875` and UNS and NCR `0.95`; their `role` and `gating` columns mark which rows decide.
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
| S8. “The rank follow-up found larger learned-code depth effects in the tested larger-rank training regimes.” | narrowed | C777 (reused): joint/M1 r64 D **+30.533333 pp [29.224896,31.900000]**, W **+20.927778 [19.052361,22.930556]**, Q **+24.016667 [22.575000,25.575208]**; seed Q **+22.575000/+25.075000/+24.400000 pp**. r128/M1-lin D **+40.591667 [39.133333,41.983542]**; hop8 at L8 **63.783333%** (absolute point, CI unavailable). | All joint r64/r128 reader rows pass the follow-up's unreplicated gate. **Rank and training steps are confounded:** F1 changes rank **and** training duration, 20,000 versus P1-S's 15,000 steps; do not attribute the contrast solely to rank or claim replication/new-seed robustness. | F1G `960a84f`: joint rows; F1C `8798483` for training duration. |
| S9. “Factorized learned writing yielded positive, walk-specific depth effects, including with a frozen T1 writer, but failed hop1 preservation.” | supported | C777 (reused), order joint_fact M1/M1-lin; frozen_fact M1/M1-lin. D pp: **+22.975000 [22.016667,23.975104] / +27.608333 [26.533333,28.708333] / +11.683333 [10.975000,12.433333] / +11.575000 [10.825000,12.350000]**. W pp: **+20.158333 [19.058333,21.191875] / +24.813889 [23.694375,25.964167] / +8.944444 [8.158194,9.722257] / +9.125000 [8.344306,9.897222]**. H pp: **−35.616667 [−37.217083,−33.933125] / −37.466667 [−39.100000,−35.850000] / −29.816667 [−31.367083,−28.199375] / −27.933333 [−29.483542,−26.416667]**. All G1 false, G2 true. | F2/r32, shared-step readers, mixed depths, no replication. Frozen rows are `t1_frozen_fact`. Positive reach/specificity do not constitute a full gate pass. | F2E `a898c87`: confidence .975, reach/specificity/hop1 and gate criteria. |
| S10. “Factorization improved the depth contrast without improving final L4 reachable-hop accuracy over the unfactorized joint system.” | supported | C777 (reused) joint_fact−joint, M1: L4 reach **−19.108333 pp [−20.533646,−17.599896]**; difference of D **+14.683333 [13.333333,16.025000]**. | r32, 15,000 steps, mixed depth, jointly trained writers per variant. The changed L1 baseline is part of the depth-effect difference; a larger D need not mean a better L4 system. | F2E `a898c87`: cross_family_contrasts.csv, joint_fact/joint M1. |
| S11. “Successor-swap diagnostics are consistent with path-sensitive behavior, but do not identify a unique learned algorithm.” | narrowed | S4242, CI unavailable. P1-S L4, `new_hop_le_L=true`, pooling original hops1–8: oracle_orth/M1-lin follows the new terminal **65.594446%** versus unswapped **0.172541%** (`15967/24342`, `42/24342`); joint/M1 **44.219867%/35.292909%** (`10764/24342`, `8591/24342`); joint/M1-lin **48.336209%/47.913072%** (`11766/24342`, `11663/24342`). F2 joint_fact/M1, **new_path_length=1** only: **47.985348%/4.426129%** (`1572/3276`, `145/3276`). | Screening counterfactuals, not confirmation. F2 and P1-S strata differ, so their rates do not estimate a controlled effect of factorization. Responses to local path changes can coexist with global terminal statistics. | P1S `98e70eb`: swap_successors.csv; F2A `a898c87`: swap_successors_new_path_length.csv, primary/L4 strata above. |
| S12. “Factorized T1 validation performance alone establishes multihop terminal retrieval.” | not supported | V4242, CI unavailable: factorized frozen-writer source T1 accuracies **91.800000/91.200000/92.400000%** (seeds17/29/43). With those writers frozen, F2 t1_frozen_fact/M1 C777 (reused) h1/h2/h3/h4 at L1: **85.100000/1.983333/1.033333/1.166667%**; at L4: **55.283333/18.833333/7.550000/4.333333%**. | T1 writer validation and T2 reader confirmation are different tasks. The matched frozen-writer profile supports some multihop use but still fails the full gate (S9); T1 loss/accuracy alone does not certify it. | F2A `a898c87`: writer_t1_scores.csv (`writer=joint_fact`, names f2-writer-s17/s29/s43, the frozen-writer sources) and accuracy_summary.csv (`t1_frozen_fact/M1`). |

## Current comparison policy and static accounting

| ID and paper sentence | Status | Numbers, interval, and set | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| FR1. “The current protocol compares multiple reads with a same-rank single read and reports added latency as a cost; shared M1-L4 and B4 have equal architectural parameter counts under matched dimensions.” | supported | Set/CI not applicable: deterministic accounting and protocol decision, not accuracy evidence. Scalar-gate shared count `r(d_in+d_out)+r²+4*r*r_mem+2`; four-step B8 adds `3*(r²+2*r*r_mem)`: **9,216** at r=r_mem=32 and **147,456** at r=r_mem=128. P1-L adapter 2048→1024, rank 32: shared **103,426**, derived B8 **112,642**, about **8.9%** extra; rank 128: shared **475,138**, derived B8 **622,594**, about **31.0%** extra. Symbolic unfactorized joint 192→64: **13,314/22,530**, about **69.2%** extra. Symbolic oracle registered **7,170/16,386**, trainable **4,096/13,312** (derived +128.5% and +225.0%); joint_fact totals are not recorded. | Equality requires identical rank, memory width, input/output dimensions, writer architecture, and gate type; separately trained weights/writers need not agree. P1-L count includes all registered adapter parameters, excludes the training-only head, and is not a whole-backbone denominator. B8 P1-L counts are architectural derivations, not benchmark measurements; the rank-128 counts are those of Amendment B6 (the shared 475,138 also equals the registered adapter count recorded by the read-fix runs, L4E), and the B8 count follows from the same formula. This policy establishes neither P1-L success nor efficiency over a larger-rank single read. | FRP `379115d` (B4), `26b96bd` (B6); FRA `216febf`, `cd5a45c`; P2C `bb101a6`, writer joint/oracle_random, reader B4/M1/B8, r32. |

The historical P2-S/P2-S-F1 comparisons below retain their original numerical findings and
practical-gate outcome. The current policy does not retroactively turn a failed gate into a pass.

## Historical larger single reads, sharing, and measured budgets

P2-S primary endpoint models and all P2-S-F1 models use `15,000` steps and the same
training seeds. The P2-S M1 reference is r32, trained at mixed depths and evaluated at L4;
B6/B7 are trained/evaluated at L1, B8 at fixed L4; B8 is not parameter-matched to the shared
readers (+9,216 parameters, about +69.2% in the joint model; FR1). The follow-up adds shared M1 trained
only at L4 and factorized B8/B6/B7, with unchanged mixed-depth references re-evaluated on
C888. Independently selected protocols are part of these comparisons. The selected numeric
M1 learning rate is nevertheless **0.003 for both fixed and mixed training** for every
writer; differing actual learning-rate values must not be invented as an observed confound
(P2FM; P1A plan; F2M). All F-E3 contrasts below use **mixed-depth M1**, not M1-L4.

| ID and paper sentence | Status | Numbers, interval, and set | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| B1. “A larger single read could not replace the shared four-read system with unfactorized learned writing.” | not supported | C888 M1−B6/r64 reach **+14.875000 pp [13.658333,16.183542]**, but −B6/r128 **−3.233333 [−4.675000,−1.833125]**; −B7/r_mem64 **−1.458333 [−2.858438,−0.125000]**, −B7/r_mem128 **−2.283333 [−3.591771,−0.891562]**. r64 contrast by seed **+33.125000/−20.675000/+32.175000 pp**. | joint, M1 r32/L4/mixed-depth; larger one-read models. Seed29 is negative in every E1/E2 contrast. Gate3 capability is false. | P2E `bb101a6`: E1/E2, joint, reach. |
| B2. “With random oracle codes, four reads outperformed both tested larger-rank single-read systems on reachable hops.” | supported | C888 M1 reach **34.775000%** versus B6/r64 **0.025000%** and r128 **0.000000%**; differences **+34.750000 pp [33.466667,36.075000]** and **+34.775000 [33.491667,36.083438]**. All tested-seed effects positive. | oracle_random; M1 r32 mixed-depth/L4 versus B6 L1. Only the tested B6 ranks, not arbitrary one-read functions. Absolute accuracies have no separate CI. | P2E `bb101a6`: E1, oracle_random, reach, corresponding accuracy.csv cells. |
| B3. “Step-specific four-read systems exceeded shared four-read systems even after matching training depth.” | supported | C888 (reused) B8−M1-L4 reach: joint **+13.216667 pp [11.983333,14.450000]**; joint_fact **+5.458333 [4.241667,6.750000]**; oracle_random **+3.200000 [1.958229,4.516771]**. Every tested-seed reach effect positive. | Both trained/evaluated fixed L4, r32. Matching depth does not match parameter count: P2C joint B8/M1 counts are **22,530/13,314** (B8 +9,216, about +69.2%; B8 is not parameter-matched; oracle_random counts are in FR1, joint_fact counts are not recorded). Joint writers are trained separately. This is a trained-system advantage, not a parameter-neutral isolation of sharing. | P2FE `ae7bda6`: F-E1/reach; P2C `bb101a6`: joint B8/M1 counts. |
| B4. “The selected fixed-depth training protocol improved shared M1's reachable-hop accuracy over the mixed-depth reference.” | supported | C888 (reused) M1-L4−M1-mixed: joint **+20.516667 pp [19.208333,21.783333]**; joint_fact **+3.483333 [2.249792,4.758333]**; oracle_random **+8.241667 [7.116667,9.400000]**. Joint B8−M1 gap shrinks from **33.733333** to **13.216667 pp**. | Shared r32, evaluation L4; separately trained/selected protocols, all 15,000 steps. Descriptive gap reduction does not identify a unique causal percentage attributable to depth alone. The B8−M1 gap compares a model with +9,216 parameters (about +69.2% in the joint model; B8 is not parameter-matched; FR1). | P2FE `ae7bda6`: F-E2/reach; P2E `bb101a6`: E3/joint/reach. |
| B5. “With factorized joint writing, four reads exceeded every tested larger single-read rank/state configuration on reachable hops.” | supported | C888 (reused) M1-mixed−B6/r64 **+24.016667 pp [23.000000,25.108333]**; −B6/r128 **+24.241667 [23.241563,25.333333]**; −B7/r_mem64 **+23.875000 [22.858333,24.975104]**; −B7/r_mem128 **+23.550000 [22.533333,24.658542]**. Every tested-seed contrast positive. M1 reach **24.483333%** versus **0.466667/0.241667/0.608333/0.933333%**, respectively (absolute points, CI unavailable). | joint_fact; M1 r32 mixed-depth/L4; B6 L1 at r=r_mem64/128, B7 L1 at r32/r_mem64/128. Capability at these budgets, not universal single-read impossibility or a measured efficiency frontier. | P2FE `ae7bda6`: F-E3/reach and matching accuracy.csv cells. |
| B6. “The unshared reader was better on every hop, or demonstrated superior depth extrapolation.” | not supported | C888 (reused) B8−M1-L4 **long** accuracy: joint **−5.533333 pp [−6.533625,−4.466583]**, joint_fact **−0.753333 [−1.076750,−0.433333]**, oracle_random **−0.536667 [−0.796708,−0.279958]**. Pooled hop1 gains are **+4.450000 [3.300000,5.750208] / +6.466667 [4.500000,8.316667] / +8.216667 [6.466458,10.017083]**, respectively; joint seed43 hop1 is **−0.850000 pp**. | Fixed training/evaluation L4, r32. Long averages are over untrained hop lengths, not larger evaluation L; B8 has no legal shared-weight depth scan. Pooled hop1 improvement is not every-seed improvement. B8 is not parameter-matched (+9,216; about +69.2% in the joint model; FR1). | P2FE `ae7bda6`: F-E1, subsets long/hop1, per_seed. |
| B7. “The symbolic joint-writer study established a general practical accuracy–latency advantage for repeated reads.” | not supported | C888 descriptive cost points, CI unavailable: joint M1 r32/L4 hops1–3 **48.016667%**, **0.847758 ms/batch64**; joint B6 r128/L1 **57.244444%**, **0.352139 ms/batch64**. Every eligible joint M1 L1/2/4/8 point is matched by an equal-or-faster one-step point for batch1 and batch64. `capability_passed=false`, `practical_passed=false`. | P2-S Gate3 fails. M1 has **13,314** trained parameters and **4,096** persistent FP32 bytes/episode versus B6/r128 **114,690** and **65,536**: the points trade different static budgets. Timings measure prepared-query batches, not end-to-end inference. | P2C `bb101a6`: cost.csv, joint/source wph M1 and source p2s B6; practical.csv; P2E decision.json. |
| B8. “Factorized learned writing gives a favorable accuracy–latency frontier.” | open | C888 supplies the positive **accuracy** contrasts in B5. **No P2-S-F1 latency benchmark or per-model cost CSV was committed; timing CI unavailable.** | Static architecture budgets and capability contrasts cannot supply missing factorized timings, writer/preparation cost, per-model comparative training cost, or end-to-end cost. Aggregate stage budget ledgers exist. The oracle M1/L4 point is unmatched in P2-S's latency test, but that does not pass the joint-writer/global gate; the factorized capability result does not retroactively pass it either. | P2FE `ae7bda6`: audit/decision and released inventory; P2C `bb101a6`: practical.csv, oracle_random M1 L4. |
| B9. “The follow-up selected a reader protocol for the next stage, without establishing its LLM performance.” | supported | C888 (reused): `reader_for_P1L=B8`, fixed training L4; selection uses F-E1 sharing and F-E2 training-depth outcomes (claims B3/B4). No LLM effect or CI is supplied by these symbolic files. | B8 fixed L4 with shared M1-L4 comparator; protocol choice, not a symbolic same-checkpoint depth scan or an end-to-end LLM result. B8 is not parameter-matched: +69.2% symbolic joint, +8.9% LLM rank 32, +31.0% LLM rank 128 (FR1). | P2FE `ae7bda6`: decision.json. |

P2-S timings use FP32 on GH200, `5` warmups and `20` repeats. They include reader,
output projection, and candidate decoding of prepared inputs, excluding context/writer
preparation and input projection. State bytes are per episode, not the whole batch.
The primary gate excludes the descriptive F1 `20,000`-step references. Parameter/state,
analytic FLOPs, measured prepared-batch latency, and total training cost are different
budgets; success against one must not be reported as success against all.

## Stopping and certificates

P4-S reuses the mixed-depth F1 checkpoints (r64/r128, `20,000` training steps).
P4-S-r32 reuses the mixed-depth P1-S/F2 checkpoints (r32, `15,000` steps). There is
no retraining. Rank and training steps are confounded across the two stopping studies as in S8
(r64/r128 at 20,000 steps, r32 at 15,000), so a difference between them is not a rank effect. All
are shared-step readers. Each family/rule threshold is selected on
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
| H1. “Validation-selected relative-change stopping improved the screening long-hop accuracy of the tested r128 random-oracle readers, while retaining a hop1 loss relative to L1.” | narrowed | S4242 M1: long L8 **45.746667%** → stopped **70.690000%**; hop1 L1 **99.983333%**, stopped **87.883333%**; mean updates **8.301167**. M1-lin: long L8 **44.846667%** → **64.410000%**; hop1 **100.000000%** → **96.616667%**; mean updates **7.147146**. V4242 selected thresholds **0.7/1**, respectively. | oracle_random/r128, from the 20,000-step F1 checkpoints (rank and steps confounded, cf. S8). No full recovery of hop1, no formal gate pass, no CI; no inference that other ranks/writers receive the same benefit. | P4 `9dd0e04`: headline.csv and thresholds.csv, oracle_random/r128/relative_change. |
| H2. “Fewer adaptive updates made the tested stopping implementation faster than fixed-depth evaluation.” | not supported | Prepared batch64, CUDA mean ms, fixed L16 versus relative-change: oracle_random/r128/M1 **2.912083 → 11.058517**; M1-lin **2.691596 → 10.293335**. All **48** adaptive family×rule×batch cells are slower than fixed L16 in both CUDA and wall means. | S4242 pilot implementation, batch1/64; active-query bookkeeping and candidate decoding included, preparation/backbone/writing excluded. This is not a claim about every query or a future optimized implementation. | P4 `9dd0e04`: latency.csv, eight families × three rules × two batch sizes; P4D `7888cf0`. |
| H3. “The sufficient factorized norm test certified none of the tested P4-S episode evaluations.” | supported | V4242+S4242: **0/96,000** bounds below one; minimum **4.571932**, maximum **37.074453**. | Bound is `abs(rho)*(norm(R)+norm(U)*norm(M)*norm(Q))`, with FP64 spectral component norms. It is an upper bound on the exact contraction expression, not its measurement. Repeated checkpoint evaluations are not independent held-out episodes. No r32 certificates were computed. Failure to certify is not proof of noncontraction/divergence. | P4 `9dd0e04`: contraction.csv, audit.json; P4D `7888cf0`. |
| H4. “The tested label-free stopping policies repaired rank32 factorized overshoot while preserving hop1.” | not supported | S4242: all **12** factorized family/rule choices fail hop1 preservation. Even the least-loss rule **descriptively chosen on screening** in each family gives H: joint_fact/M1 **−33.033333 pp**, joint_fact/M1-lin **−41.733333**, t1_frozen_fact/M1 **−29.683333**, t1_frozen_fact/M1-lin **−24.900000**. Their D/W are respectively **+21.350000/+17.905556**, **+21.458333/+15.102778**, **+10.233333/+7.327778**, **+9.516667/+6.988889 pp**. | r32. Rules in the same order: relative_change .3, output_stability 4, output_stability 4, read_norm 1. D/W positive in every tested seed still do not compensate for H failure. This does not exclude every possible halting method. | P4R `3e15f07`: endpoints.csv, endpoints_per_seed.csv, thresholds.csv; P4RD `ce77e74`. |
| H5. “The unfactorized r32 joint/M1 screening flag establishes a benefit from adaptive stopping.” | not supported | S4242: all three rules use exactly **16** mean updates and reproduce fixed-L16 endpoints: H **+5.150000 pp**, D **+8.666667**, F **+6.027778**, W **+2.638889**; minimum seed W **−0.125000 pp**. Read-norm costs **17** reads. | joint/M1/r32. Aggregate `screening_candidate=true` is a point-estimate flag, not confirmation or a formal G1 pass. There is no stopping benefit over fixed L16 here. | P4R `3e15f07`: endpoints.csv, endpoints_per_seed.csv, accuracy.csv. |
| H6. “Stopping at the true graph distance is an accuracy upper bound for a learned reader.” | not supported | S4242 oracle_random/r128/M1 overall accuracy: true-hop **23.564583%**, relative-change **75.302083%**. | True-hop stopping is a label-informed diagnostic reference, not an upper bound. Learned update count need not equal graph distance. | P4 `9dd0e04`: headline.csv, oracle_true_hop/relative_change. |

## Learned halting confirmation (P4-S-L)

C999002 evaluates 24 unchanged r32 mixed-depth readers with one newly trained halting head
per checkpoint. Six primary families are `{oracle_random,joint_fact,t1_frozen_fact}` ×
`{M1,M1-lin}`; unfactorized joint/M1 and joint/M1-lin are descriptive. Head training is
supervised: 51→64 ReLU→1 sigmoid, **3,393 parameters**, features z, state norm, relative
change, post-state read-norm ratio and step one-hot, with no answer/hop/correctness or
decoder-confidence input. Correctness labels supply training targets. Frozen readers/writers
are not retrained. Training seed 31337 supplies 20,000 episodes, all eight queries and sixteen
post-update states; BCE/Adam lr .001, 2,000 updates, batches of 256 whole episodes sampled
with replacement, independently derived head initialization/sampler seeds (HLT manifest).

HLD locked design/reference thresholds before head training. On V4242/2,000 episodes, one
θ per family was selected from .05,.10,…,.95 to maximize mean accuracy over all eight hops
and three fixed seeds, ties fewer **chosen** steps then smaller θ. HLT locked θ and head
seals before C999002; this objective did not require hop 1 preservation. The best label-free
rule was also chosen on validation (ties read_norm, relative_change, output_stability), using
18 unchanged C4 thresholds plus six missing oracle thresholds selected on validation before
head training. No best rule is selected on confirmation.

All 64 exported endpoint CIs are **99.166667%** (`1−.05/6`), 2,000 paired episode-bootstrap
draws, base 2718, with shared episode draws across fixed seeds 17/29/43. This is the registered
six-primary-family adjustment, not simultaneous coverage of all 64 claims. The intervals
condition on the fitted readers/heads and do not resample training or threshold selection.
Each family has 2,000 confirmation episodes and 48,000 query instances across the three seeds.
The analysis audit's 768,000 queries includes validation **and** test, not confirmation alone.

| ID and paper sentence | Status | Numbers, interval, and set | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| HL1. “None of six primary frozen-reader families met the combined learned-halting criterion.” | supported | C999002: all six fail H≥−1 pp, despite D≥3 pp with lower CI>0 and W lower CI>0. All 18 primary per-seed H points are below −1 pp. Full H/D/F/W/overall effects and adjusted CIs appear below. | Learned−L1; H hop 1, D hops 2–3, F hops 6–8, W=D−F. H preservation and D≥3 are point criteria, not lower-CI thresholds. Negative result for this post-hoc head/protocol, not impossibility of learned halting. | HLE `95736ec`: endpoints, endpoints_per_seed, qualification; primary=True. |
| HL2. “The learned policy improved overall accuracy over the validation-selected label-free reference in four primary families, without repairing preservation.” | supported | C999002: positive adjusted intervals for both oracle_random and joint_fact readers; both frozen-fact estimates are negative with intervals crossing zero. All six effects/CIs and reference rule/threshold below. | Hops 1–8; locked validation-selected rule, not best test rule. These small overall improvements do not overturn HL1. | HLE `95736ec`: learned_minus_best_label_free; HLD `9366da4`: best_label_free. |
| HL3. “All primary learned policies remained below the any-step oracle, while true-hop depth was not an accuracy upper bound.” | supported | C999002: all six any-step-minus-learned adjusted intervals are positive. Learned−true-hop is positive for oracle_random and joint_fact, negative for frozen_fact; all adjusted intervals exclude zero. Exact six-family values below. | Any-step oracle uses answer labels and succeeds if any post-update state 1…16 is correct; upper bound only for selecting among these frozen predictions. True-hop selects graph distance 1…8 and is a separate label-informed reference, not an upper bound. | HLE `95736ec`: any_step_oracle_minus_learned / learned_minus_true_hop. |
| HL4. “Earlier chosen states did not yield a measured latency improvement for this learned-halting implementation.” | supported | Full-confirmation chosen/executed/read means and timing table below. Primary chosen depths 5.306125–10.038146, executed 12.739750–15.994521; learned CUDA means 20.533381–25.252794 ms versus fixed 16 means 3.071751–3.308061 ms. CI unavailable. | First p≥θ state; otherwise execute 16 and select earliest max-p state. Executed+1 memory reads, hence 17 on fallback. Timings use a prepared 64-query batch/seed, five warmups/20 repeats; exclude data/backbone/writing/head training/end-to-end LM work. Family timings average three seed summaries. Full-confirmation counters differ from timed microbatch counters. | HLE `95736ec`: accuracy split=test/method=learned/hop=overall; latency method=learned/fixed_4/fixed_16. |
| HL5. “The descriptive unfactorized joint/M1 row satisfied the numerical inequalities but was not a qualifying primary result.” | supported | C999002: joint/M1 endpoints_satisfied=True, primary=False, qualifies=False; joint/M1-lin specificity CI crosses zero. All 16 descriptive endpoints appear in the same tables below. | Neither descriptive row is a seventh/eighth primary trial or a primary remedy. Joint/M1 has an overall learned-minus-label-free loss; joint/M1-lin's interval crosses zero. | HLE `95736ec`: qualification.csv and endpoints.csv, writer=joint. |

All following effects/CIs are pp, confidence 99.166667%. Family abbreviations retain
writer/reader, all r32; `P`=primary, `D`=descriptive. Together the next two tables contain
**all 64 endpoints**, including far and overall; each cell is estimate [lower,upper].

| Family | Scope | θ | H | D | F | W | Overall learned−L1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint/r32/M1 | D | 0.9 | 6.050000 [4.694306,7.394722] | 8.741667 [7.699896,9.822257] | 5.683333 [4.644444,6.777847] | 3.058333 [1.666667,4.500139] | 6.622917 [5.994436,7.241667] |
| joint/r32/M1-lin | D | 0.75 | 2.966667 [1.688819,4.077847] | 3.016667 [2.027743,3.944514] | 2.088889 [1.174051,3.087060] | 0.927778 [-0.260231,2.162060] | 2.427083 [1.831936,3.103507] |
| joint_fact/r32/M1 | P | 0.45 | -33.900000 [-35.777847,-31.872153] | 21.583333 [20.433333,22.755590] | 5.066667 [4.472222,5.709282] | 16.516667 [15.262801,17.665752] | 5.550000 [4.945807,6.131250] |
| joint_fact/r32/M1-lin | P | 0.4 | -49.150000 [-51.011181,-47.216667] | 24.733333 [23.463715,26.025104] | 9.294444 [8.396273,10.248241] | 15.438889 [14.025914,16.886111] | 7.389583 [6.680512,8.111181] |
| oracle_random/r32/M1 | P | 0.25 | -15.700000 [-18.227847,-13.688819] | 32.508333 [31.019306,33.938924] | 4.533333 [3.829606,5.144444] | 27.975000 [26.378657,29.575069] | 11.714583 [10.992960,12.455694] |
| oracle_random/r32/M1-lin | P | 0.3 | -20.850000 [-23.205903,-18.655486] | 37.500000 [35.841667,39.205590] | 18.661111 [17.364769,19.977917] | 18.838889 [16.947188,20.677847] | 20.002083 [18.859019,21.037578] |
| t1_frozen_fact/r32/M1 | P | 0.25 | -21.900000 [-23.627847,-20.216667] | 9.441667 [8.677743,10.283438] | 2.550000 [2.118495,2.974190] | 6.891667 [5.996273,7.774120] | 1.339583 [0.915929,1.729878] |
| t1_frozen_fact/r32/M1-lin | P | 0.35 | -26.166667 [-27.911181,-24.338819] | 10.216667 [9.436076,11.116667] | 2.666667 [2.235162,3.088889] | 7.550000 [6.661076,8.447361] | 1.181250 [0.755408,1.585469] |


| Family | Best label-free rule / threshold | Learned−best label-free | Any-step oracle−learned | Learned−true-hop |
| --- | --- | --- | --- | --- |
| joint/r32/M1 | read_norm / 0.3 | -0.516667 [-0.807648,-0.236788] | 10.000000 [9.545095,10.495148] | 0.981250 [0.631250,1.332674] |
| joint/r32/M1-lin | relative_change / 0.03 | -0.147917 [-0.436128,0.116693] | 6.720833 [6.173524,7.284731] | 0.125000 [-0.172257,0.400738] |
| joint_fact/r32/M1 | output_stability / 4 | 0.883333 [0.463160,1.303481] | 21.156250 [20.550686,21.827847] | 0.979167 [0.456250,1.493802] |
| joint_fact/r32/M1-lin | output_stability / 4 | 1.614583 [1.125000,2.145252] | 28.454167 [27.587474,29.335417] | 2.518750 [1.859679,3.178507] |
| oracle_random/r32/M1 | relative_change / 0.7 | 1.258333 [0.705486,1.851398] | 36.337500 [35.279826,37.350026] | 6.425000 [5.832622,7.066667] |
| oracle_random/r32/M1-lin | relative_change / 0.7 | 1.522917 [0.867300,2.234071] | 35.572917 [34.512474,36.584783] | 13.412500 [12.354062,14.421545] |
| t1_frozen_fact/r32/M1 | output_stability / 4 | -0.102083 [-0.390981,0.175712] | 10.250000 [9.848602,10.647257] | -2.491667 [-2.815981,-2.146519] |
| t1_frozen_fact/r32/M1-lin | read_norm / 1.0 | -0.097917 [-0.449314,0.256962] | 10.950000 [10.508281,11.401398] | -2.372917 [-2.723628,-2.021519] |

Primary H per-seed point effects in pp (17/29/43); no new per-seed intervals:

| Family | H seed 17 /29 /43 |
| --- | --- |
| joint_fact/r32/M1 | -33.850000 / -34.950000 / -32.900000 |
| joint_fact/r32/M1-lin | -52.050000 / -46.200000 / -49.200000 |
| oracle_random/r32/M1 | -12.350000 / -19.250000 / -15.500000 |
| oracle_random/r32/M1-lin | -17.800000 / -20.350000 / -24.400000 |
| t1_frozen_fact/r32/M1 | -22.300000 / -20.450000 / -22.950000 |
| t1_frozen_fact/r32/M1-lin | -26.350000 / -25.500000 / -26.650000 |

Confirmation counters and separate timing population (all absolute CIs unavailable):

| Family | Accuracy % | Chosen | Executed | Read calls | Learned CUDA ms | L4 CUDA ms | L16 CUDA ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint/r32/M1 | 43.245833 | 6.488792 | 15.999792 | 16.999792 | 20.497695 | 0.960222 | 3.219443 |
| joint/r32/M1-lin | 48.129167 | 4.899479 | 15.998833 | 16.998833 | 20.387726 | 0.912897 | 3.050142 |
| joint_fact/r32/M1 | 17.208333 | 8.608792 | 15.994521 | 16.994521 | 21.077982 | 0.977411 | 3.308061 |
| joint_fact/r32/M1-lin | 19.354167 | 10.038146 | 15.971396 | 16.971396 | 20.893956 | 0.931035 | 3.093715 |
| oracle_random/r32/M1 | 21.452083 | 5.306125 | 14.207396 | 15.207396 | 21.939381 | 0.979735 | 3.273001 |
| oracle_random/r32/M1-lin | 30.097917 | 8.191250 | 12.739750 | 13.739750 | 25.252794 | 0.922389 | 3.071751 |
| t1_frozen_fact/r32/M1 | 13.020833 | 6.189854 | 15.994521 | 16.994521 | 20.567235 | 0.973297 | 3.283272 |
| t1_frozen_fact/r32/M1-lin | 13.147917 | 6.877062 | 15.991417 | 16.991417 | 20.533381 | 0.921618 | 3.080122 |


## Jacobian regularization confirmation (JS)

C999003 uses M1/ReLU r=r_mem=32, E12 chains (8,4), train hops 1–3, seeds 17/29/43.
Twelve new RS/RS+J models train 15,000 steps, batch 64, lr .003; each batch draws one uniform
L∈{4,5,6,7,8}. At each seed RS/RS+J share initialization and training-data/depth RNG
streams, then optimize separately. RS+J adds the episode/query mean of
`max(radius(J(z_L))−.9,0)^2`, lambda 1. Twelve unchanged M1-L4/M1-mixed reference
checkpoints are newly evaluated on this same confirmation set. References have different
training-depth histories; only RS+J−RS isolates the tested regularization intervention.

JSD fixed the dense 32×32 CPU float32 eigenvalue method before scores, chosen by timing.
J(z)=rho diag(1[Rz+UMQz>0])(R+UMQ) uses the **next-update gate at reached z_L**.
ReLU's chosen zero derivative is zero; gradients flow through matrix factors and trainable
memory, with locally constant Boolean gates. “Exact eigenvalues” denotes full numerical
eigensolution, not interval-certified arithmetic. The preregistration is `f5bb486`, method/design
lock `852d293`, analysis source introduced `731a2a2`; no scores were used to choose the backend.

After all training, JST selected eight relative-change thresholds, one per writer/condition
across all three seeds, on V4242/2,000 episodes, from
[0,.001,.003,.01,.03,.1,.2,.3,.5,.7,1,1.5,2,4]. Selection maximizes total correct
over hops 1–8/seeds, ties fewer total updates then smaller threshold. Lock `0c99cc1`
precedes confirmation. C999003 has 2,000 episodes, one query/hop 1–8, depths 1–16 and
stopped cap 16; first qualifying post-update state, else z16. Endpoint L=17 is only a
stopped-policy sentinel, never a seventeenth update. All 24 traces precede one aggregation.

Endpoint intervals are 97.5%, 2,000 paired episode-bootstrap draws, base 2718, shared draws
across the three fixed training seeds. DeltaS jointly resamples all four terms. They exclude
new-training-seed and reselection uncertainty. Short hops 1–3 differ from earlier reach 2–3;
long hops 4–8 and stopped hops 1–8 are explicit below. Absolute rates, radius summaries,
and stopping counters have **CI unavailable**. JS measured no latency advantage.

| ID and paper sentence | Status | Numbers, interval, and set | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| JS1. “Regularization reduced overshoot for both writers but neither met the registered removal criterion.” | supported | C999003: DeltaS positive with positive lower CI and all three seed effects positive for both writers; both S(RS+J) points remain below −1 pp. Exact S/DeltaS estimates and 97.5% CIs below. | S=Acc16−Acc4 on hops 1–3. Removal requires DeltaS point>0 and lower CI>0, plus S(RS+J) point≥−1 pp; not a lower-CI floor on S. No gate or followup depends on JS. | JSE `96ff2e9`: JS-E1-S/JS-E1-DeltaS, decision.json; JSD `852d293`. |
| JS2. “The L4 short-hop effect was positive for oracle_random, while joint_fact's negative point had an interval crossing zero.” | supported | C999003: RS+J−RS, exact JS-E2 estimates/CIs and all seed effects below. | L4, hops 1–3; no claim that every capability endpoint improves. | JSE `96ff2e9`: JS-E2. |
| JS3. “Regularization reduced L16 long-hop accuracy in both tested writers.” | supported | C999003: JS-E3 upper CIs below zero, all seed effects negative; exact values below. | RS+J−RS, L16, hops 4–8. Does not imply universal harm outside this protocol. | JSE `96ff2e9`: JS-E3. |
| JS4. “Regularization reduced the accuracy of the validation-selected stopped policy in both writers.” | supported | C999003: JS-E4 upper CIs below zero, all seed effects negative; stopped rates, thresholds, mean reads and cap fractions below. | Each condition has its own validation-selected threshold; hops 1–8, cap 16. The contrast includes the deployed selection policy. No absolute-rate/counter CI or latency result. | JSE `96ff2e9`: JS-E4/stopping.csv; JST `0c99cc1`. |
| JS5. “RS-reference outcomes depended on writer, evaluation depth, and hop subset.” | supported | C999003: all 24 descriptive RS−M1-L4/M1-mixed contrasts and 97.5% CIs below. At L4 short hops, oracle RS is below both references; joint_fact versus mixed is +0.483333 [−0.488889,1.466667], inconclusive. All L16 short-hop contrasts are positive; joint_fact L16 long-hop contrasts are negative. | Both references unchanged but newly evaluated on 999003; historical training protocols differ. No stopped RS-reference CI is exported; stopped absolute rates below are descriptive. | JSE `96ff2e9`: endpoint=RS-reference, exact comparator/L/hops selectors below. |
| JS6. “Regularization lowered the pooled reached-state Jacobian-radius mean at all three measured depths for each writer.” | supported | C999003: all 24 pooled radius summaries below, including references; CI unavailable. L16 RS+J fractions below 1 are oracle .222375 and joint_fact .926979. | hop=0 pools hops 1–8/three seeds, 48,000 query states per row; only L4/8/16 measured. Pointwise radii do not certify fixed points, global contraction, transient products, terminal decoding, or correctness. | JSE `96ff2e9`: rho.csv, hop=0; rho_per_seed.csv retains seed summaries. |

JS primary endpoint estimates/CIs are pp; seed effects ordered 17/29/43. Exact selectors
are endpoint,writer,condition,comparator,L_a/L_b,hops in JSE endpoints.csv.

| Endpoint | Writer | A−B | L_a/L_b | Hops | Effect [97.5% CI], pp | Seed17 /29 /43, pp |
| --- | --- | --- | --- | --- | --- | --- |
| JS-E1-S | oracle_random | RS−RS | 16/4 | 1-3 | -11.266667 [-12.355694,-10.049931] | -11.900000 / -11.016667 / -10.883333 |
| JS-E1-S | oracle_random | RS+J−RS+J | 16/4 | 1-3 | -2.833333 [-3.800139,-1.805556] | -3.533333 / -2.400000 / -2.566667 |
| JS-E1-DeltaS | oracle_random | RS+J−RS | 16-4/16-4 | 1-3 | 8.433333 [7.483264,9.417569] | 8.366667 / 8.616667 / 8.316667 |
| JS-E2 | oracle_random | RS+J−RS | 4/4 | 1-3 | 0.788889 [0.261042,1.294514] | 1.200000 / 0.883333 / 0.283333 |
| JS-E3 | oracle_random | RS+J−RS | 16/16 | 4-8 | -5.546667 [-6.200000,-4.889917] | -5.670000 / -4.540000 / -6.430000 |
| JS-E4 | oracle_random | RS+J−RS | 17/17 | 1-8 | -1.039583 [-1.537552,-0.543750] | -0.693750 / -0.512500 / -1.912500 |
| JS-E1-S | joint_fact | RS−RS | 16/4 | 1-3 | -3.016667 [-3.788889,-2.272153] | -2.800000 / -1.950000 / -4.300000 |
| JS-E1-S | joint_fact | RS+J−RS+J | 16/4 | 1-3 | -1.772222 [-2.483542,-1.061111] | -1.983333 / -0.900000 / -2.433333 |
| JS-E1-DeltaS | joint_fact | RS+J−RS | 16-4/16-4 | 1-3 | 1.244444 [0.222014,2.200069] | 0.816667 / 1.050000 / 1.866667 |
| JS-E2 | joint_fact | RS+J−RS | 4/4 | 1-3 | -0.727778 [-1.566667,0.066875] | -0.233333 / -0.733333 / -1.216667 |
| JS-E3 | joint_fact | RS+J−RS | 16/16 | 4-8 | -1.443333 [-1.990083,-0.933292] | -0.630000 / -1.250000 / -2.450000 |
| JS-E4 | joint_fact | RS+J−RS | 17/17 | 1-8 | -0.677083 [-1.227109,-0.114557] | -0.162500 / -0.606250 / -1.262500 |

All 24 RS-reference contrasts (descriptive comparisons, pp):

| Writer | Comparator | L | Hops | RS−reference [97.5% CI] | Seed17 /29 /43 |
| --- | --- | --- | --- | --- | --- |
| oracle_random | M1-L4 | 4 | 1-3 | -14.694444 [-15.838958,-13.588611] | -16.500000 / -14.366667 / -13.216667 |
| oracle_random | M1-L4 | 4 | 4-8 | -0.880000 [-1.166667,-0.593292] | -0.870000 / -0.850000 / -0.920000 |
| oracle_random | M1-L4 | 4 | 1-8 | -6.060417 [-6.518776,-5.620755] | -6.731250 / -5.918750 / -5.531250 |
| oracle_random | M1-L4 | 16 | 1-3 | 7.816667 [6.794028,8.805625] | 6.333333 / 8.400000 / 8.716667 |
| oracle_random | M1-L4 | 16 | 4-8 | 7.493333 [6.563333,8.523333] | 7.820000 / 6.990000 / 7.670000 |
| oracle_random | M1-L4 | 16 | 1-8 | 7.614583 [6.856198,8.450000] | 7.262500 / 7.518750 / 8.062500 |
| oracle_random | M1-mixed | 4 | 1-3 | -8.738889 [-9.733403,-7.766667] | -9.233333 / -9.283333 / -7.700000 |
| oracle_random | M1-mixed | 4 | 4-8 | -2.013333 [-2.303333,-1.719917] | -1.900000 / -2.020000 / -2.120000 |
| oracle_random | M1-mixed | 4 | 1-8 | -4.535417 [-4.966745,-4.102083] | -4.650000 / -4.743750 / -4.212500 |
| oracle_random | M1-mixed | 16 | 1-3 | 5.300000 [4.366667,6.239028] | 4.766667 / 5.483333 / 5.650000 |
| oracle_random | M1-mixed | 16 | 4-8 | 5.080000 [4.209958,6.023375] | 6.180000 / 3.950000 / 5.110000 |
| oracle_random | M1-mixed | 16 | 1-8 | 5.162500 [4.393620,5.881380] | 5.650000 / 4.525000 / 5.312500 |
| joint_fact | M1-L4 | 4 | 1-3 | -2.522222 [-3.533403,-1.499931] | -2.500000 / -3.650000 / -1.416667 |
| joint_fact | M1-L4 | 4 | 4-8 | -1.153333 [-1.466708,-0.846625] | -1.630000 / -0.890000 / -0.940000 |
| joint_fact | M1-L4 | 4 | 1-8 | -1.666667 [-2.112526,-1.231250] | -1.956250 / -1.925000 / -1.118750 |
| joint_fact | M1-L4 | 16 | 1-3 | 5.066667 [3.966528,6.172292] | 3.850000 / 6.000000 / 5.350000 |
| joint_fact | M1-L4 | 16 | 4-8 | -2.620000 [-3.306667,-1.933292] | -2.680000 / -4.340000 / -0.840000 |
| joint_fact | M1-L4 | 16 | 1-8 | 0.262500 [-0.393776,0.920859] | -0.231250 / -0.462500 / 1.481250 |
| joint_fact | M1-mixed | 4 | 1-3 | 0.483333 [-0.488889,1.466667] | 0.200000 / -0.566667 / 1.816667 |
| joint_fact | M1-mixed | 4 | 4-8 | -1.773333 [-2.106708,-1.433292] | -1.840000 / -1.720000 / -1.760000 |
| joint_fact | M1-mixed | 4 | 1-8 | -0.927083 [-1.372969,-0.485365] | -1.075000 / -1.287500 / -0.418750 |
| joint_fact | M1-mixed | 16 | 1-3 | 4.911111 [3.855347,6.022222] | 4.900000 / 5.666667 / 4.166667 |
| joint_fact | M1-mixed | 16 | 4-8 | -1.033333 [-1.620167,-0.383292] | -2.400000 / -1.220000 / 0.520000 |
| joint_fact | M1-mixed | 16 | 1-8 | 1.195833 [0.579115,1.808359] | 0.337500 / 1.362500 / 1.887500 |

Stopped confirmation counters, pooled equal-size seeds (48,000 queries/row); relative-change
uses one memory read per update. Reaching cap 16 includes first qualification at 16 and fallback.
No counter/rate intervals; validation accuracies in thresholds.csv are not confirmation rates.

| Writer | Condition | Threshold | Accuracy % | Mean updates/reads | Fraction before 16 | Fraction at cap 16 |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_random | RS | 0.3 | 19.743750 | 12.380438 | 0.479583 | 0.520417 |
| oracle_random | RS+J | 0.01 | 18.704167 | 15.937500 | 0.015292 | 0.984708 |
| oracle_random | M1-L4 | 1 | 11.189583 | 12.802479 | 0.218979 | 0.781021 |
| oracle_random | M1-mixed | 0.7 | 20.343750 | 8.814583 | 0.718625 | 0.281375 |
| joint_fact | RS | 0.01 | 16.556250 | 15.348292 | 0.228958 | 0.771042 |
| joint_fact | RS+J | 0.01 | 15.879167 | 15.028479 | 0.316812 | 0.683188 |
| joint_fact | M1-L4 | 0.3 | 17.225000 | 7.505646 | 0.827667 | 0.172333 |
| joint_fact | M1-mixed | 0.3 | 15.966667 | 6.692271 | 0.858312 | 0.141688 |

Radius summaries pool query states rather than seed quantiles. These are descriptive
radii, not spectral norms. `n=48,000,n_seeds=3,hop=0`; each hop-specific row has n=6,000.
Percentiles are empirical spans, not CIs; no minimum or full observed range is exported.
All 24 rows follow; per-seed distributions remain in rho_per_seed.csv.

| Writer | Condition | L | Mean | p05 | p25 | Median | p75 | p95 | Maximum | Fraction≤.9 | Fraction<1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| joint_fact | M1-L4 | 4 | 0.989793 | 0.764256 | 0.888172 | 0.981061 | 1.081827 | 1.241228 | 1.912912 | 0.278500 | 0.550604 |
| joint_fact | M1-L4 | 8 | 0.937635 | 0.718047 | 0.838661 | 0.931563 | 1.031757 | 1.175421 | 1.912912 | 0.411271 | 0.678083 |
| joint_fact | M1-L4 | 16 | 0.913369 | 0.704723 | 0.818281 | 0.908558 | 1.001585 | 1.138885 | 1.651033 | 0.473938 | 0.746479 |
| joint_fact | M1-mixed | 4 | 0.958940 | 0.739268 | 0.856705 | 0.947863 | 1.048944 | 1.215554 | 1.899726 | 0.363833 | 0.638312 |
| joint_fact | M1-mixed | 8 | 0.926829 | 0.710374 | 0.829178 | 0.921789 | 1.017651 | 1.160152 | 1.741270 | 0.439979 | 0.708875 |
| joint_fact | M1-mixed | 16 | 0.907296 | 0.704082 | 0.817077 | 0.903580 | 0.990967 | 1.123866 | 1.552292 | 0.488875 | 0.770812 |
| joint_fact | RS | 4 | 0.914272 | 0.728813 | 0.832473 | 0.907331 | 0.987391 | 1.122517 | 1.586463 | 0.475042 | 0.782083 |
| joint_fact | RS | 8 | 0.875872 | 0.683402 | 0.791569 | 0.872439 | 0.955038 | 1.079344 | 1.467736 | 0.589083 | 0.850938 |
| joint_fact | RS | 16 | 0.861332 | 0.668890 | 0.777233 | 0.860239 | 0.943519 | 1.056833 | 1.441559 | 0.625146 | 0.877604 |
| joint_fact | RS+J | 4 | 0.876390 | 0.699490 | 0.799213 | 0.869755 | 0.945306 | 1.072570 | 1.609046 | 0.608979 | 0.866521 |
| joint_fact | RS+J | 8 | 0.845951 | 0.660253 | 0.765791 | 0.843005 | 0.921565 | 1.039391 | 1.605100 | 0.687312 | 0.907396 |
| joint_fact | RS+J | 16 | 0.834440 | 0.649651 | 0.754675 | 0.832990 | 0.912751 | 1.021459 | 1.406855 | 0.713333 | 0.926979 |
| oracle_random | M1-L4 | 4 | 2.157681 | 1.965627 | 2.115246 | 2.179317 | 2.225690 | 2.256769 | 2.901083 | 0.000000 | 0.000000 |
| oracle_random | M1-L4 | 8 | 2.222501 | 2.152510 | 2.216916 | 2.228158 | 2.236275 | 2.263444 | 2.901083 | 0.000000 | 0.000000 |
| oracle_random | M1-L4 | 16 | 2.230459 | 2.194981 | 2.221781 | 2.230045 | 2.237514 | 2.267407 | 2.671225 | 0.000000 | 0.000000 |
| oracle_random | M1-mixed | 4 | 1.612574 | 1.391506 | 1.528376 | 1.609965 | 1.696003 | 1.836332 | 2.442857 | 0.000042 | 0.000042 |
| oracle_random | M1-mixed | 8 | 1.664782 | 1.500866 | 1.597321 | 1.660107 | 1.724806 | 1.848288 | 2.385803 | 0.000000 | 0.000042 |
| oracle_random | M1-mixed | 16 | 1.673661 | 1.527784 | 1.607157 | 1.673439 | 1.728013 | 1.839008 | 2.352504 | 0.000042 | 0.000042 |
| oracle_random | RS | 4 | 1.224589 | 1.051677 | 1.163818 | 1.232037 | 1.289379 | 1.375712 | 1.736513 | 0.002667 | 0.019188 |
| oracle_random | RS | 8 | 1.280159 | 1.161424 | 1.242955 | 1.283830 | 1.321203 | 1.390361 | 1.709372 | 0.000479 | 0.001521 |
| oracle_random | RS | 16 | 1.307088 | 1.238773 | 1.278106 | 1.302462 | 1.329621 | 1.394285 | 1.643792 | 0.000312 | 0.000625 |
| oracle_random | RS+J | 4 | 1.010545 | 0.837629 | 0.944092 | 1.011461 | 1.077807 | 1.182706 | 1.606321 | 0.137333 | 0.452792 |
| oracle_random | RS+J | 8 | 1.039018 | 0.886026 | 0.984700 | 1.039177 | 1.097384 | 1.190965 | 1.692890 | 0.064833 | 0.313292 |
| oracle_random | RS+J | 16 | 1.052610 | 0.910700 | 1.007045 | 1.053060 | 1.105079 | 1.187647 | 1.482803 | 0.042854 | 0.222375 |


## Memory load, adaptation, and updates (study IS)

IS primary writers are `oracle_random` and `joint_fact`; unfactorized `joint` is
descriptive only. All models have `r=r_mem=32`, ReLU, and train hops `{1,2,3}`.
The fixed-checkpoint scan reuses E12-trained mixed-depth M1 at `L=1…8` for loads/layouts
`E6:(4,2)`, `E12:(8,4)`, `E18:(8,8,2)`, `E24:(8,8,8)`, and
`E36:(8,8,8,8,4)`. Thus load, graph layout, query distribution, and candidate count
change together; this is not an isolated crosstalk manipulation. E6 has hops1–4 and no
far-hop stratum. New E24/E36 training uses `15,000` steps, batch64, and the inherited
learning rate `0.003` for every primary writer/reader, without a load-specific search.
M1 samples training depths `{1,2,4}`; B8 is unshared fixed-L4 (+9,216 parameters, so not
parameter-matched to M1 or B4; FR1); B4 is L1.

The design and update-task validity changes were locked before inference/training.
C999 is opened after training. All IS intervals condition on seeds17/29/43 and use
`2,000` episode-bootstrap draws with purpose-derived streams from base seed2718.
IS1's DeltaD uses independent E12/E36 streams; it is not obtained by pairing episode
IDs across different layouts or subtracting endpoints of marginal CIs. The printed
`best_L` values are descriptive confirmation-set maxima, not selected deployment depths.

| ID and paper sentence | Status | Exact numbers, interval, and set | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| IS1. “For fixed E12-trained checkpoints, the reachable-hop depth benefit decreased from E12 to E36 under both registered load/layout families.” | narrowed | C999, D(E)=M1@4−same checkpoint@1 on hops2–3. Oracle_random: D12 **+34.750000 pp [33.458333,36.066771]**, D36 **+7.491667 [6.791563,8.200104]**, DeltaD **−27.258333 [−28.800000,−25.816250]**, seed effects **−28.725000/−25.900000/−27.150000 pp**. Joint_fact: D12 **+24.041667 [23.024896,25.083333]**, D36 **+4.641667 [4.091667,5.166667]**, DeltaD **−19.400000 [−20.541667,−18.241458]**, seed effects **−20.525000/−19.200000/−18.475000 pp**. | r32 shared M1, original mixed-depth checkpoints; load and layout vary together. Consistent with the registered operational depth-amplification prediction, not proof of Prop.4's causal mechanism, bound tightness, or universal monotonic accuracy loss. IVD identifies key interference for `oracle_random` only (IV1) and withdraws it for `joint_fact` (IV2). | ISE `e144246`: endpoints.csv, IS-E1-D/IS-E1-DeltaD, train_E12, E12/36, subset reach; ISD `fb62cf1`. |
| IS2. “At every tested training load, M1@4 and B8@4 exceeded B4@1 under the registered reachable-hop capability criterion; E36 was the largest tested qualifying load.” | supported | C999: all **12** primary writer×reader×E12/24/36 contrasts have point estimate≥**3 pp** and CI lower>0; all **36** tested-seed effects positive. At E36, M1/B8 minus B4: oracle **+4.483333 pp [3.949896,5.033438] / +5.941667 [5.225000,6.675104]**; joint_fact **+3.433333 [2.983333,3.866667] / +5.008333 [4.500000,5.500000]**. | Each system trained at its evaluation load, r32; M1 mixed-depth, B8 fixed-L4, B4 L1. Joint_fact writers are separately trained across variants; B8 has step-specific parameters (+9,216, so it is not parameter-matched to B4 or M1; FR1). Criterion is point≥3 and CI>0, not CI lower≥3. The failure boundary beyond E36 was not located; no hop1-preservation, efficiency, or original-gate conclusion follows. | ISE `e144246`: endpoints.csv IS-E2, subset reach; decision.json qualifying_E; ISD `fb62cf1`. |
| IS3. “Under the fixed inherited training protocol, training at E24/E36 reduced reachable-hop accuracy relative to evaluating E12-trained checkpoints at that same load.” | supported | C999 load-trained minus E12-trained, both @4. Oracle M1 E24/E36: **−2.750000 pp [−3.650104,−1.874896] / −3.175000 [−3.900104,−2.483125]**; oracle B8: **−4.900000 [−5.825104,−3.974896] / −6.283333 [−7.158437,−5.433229]**. Joint_fact M1: **−3.291667 [−4.150000,−2.383125] / −2.041667 [−2.691875,−1.383229]**; joint_fact B8: **−2.450000 [−3.458438,−1.475000] / −3.491667 [−4.208437,−2.733229]**. All eight intervals and all **24** seed effects are negative. | Same evaluation E within each comparison, r32. All training uses the inherited LR .003, 15,000 steps and batch64, without load-specific tuning or matched convergence; this is not a general claim that adaptation is harmful. B8 is not parameter-matched to M1 (+9,216 parameters; FR1). | ISE `e144246`: endpoints.csv IS-E3, reader M1-mixed/B8, train_E=E24/36, subset reach; ISD `fb62cf1`. |
| IS4. “On updated-path queries, the tested B8@4 systems selected current terminals more often than obsolete terminals, with partial rather than complete task success.” | narrowed | C999, absolute percentages and their **97.5% rate CIs**. Oracle affected current **41.749696% [40.235159,43.299799]**, untouched current **37.606178% [36.051074,39.211368]**, affected stale **0.510328% [0.331909,0.712153]**. Joint_fact affected current **28.618874% [27.451917,29.735415]**, untouched current **30.467610% [29.357424,31.647769]**, affected stale **6.034832% [5.486256,6.591167]**. Affected/untouched denominators are **12,345/11,655** predictions pooled over three fixed seeds, not independent episode counts. | Unchanged E12-trained B8, r32, L4; update task has **14 stored facts = 12 current edges + 2 obsolete facts**, current chains(8,4), queries hops1–4 pooled (`hop=0`). Stale target follows first occurrences of repeated keys, current target follows latest. Untouched stale rate is undefined, not zero. No paired CI for current−stale or affected−untouched was exported; strata do not establish a causal update effect, and the claim does not cover every reader. B8 is not parameter-matched to the shared readers (+9,216 parameters; FR1). | ISU `e144246`: t3_update.csv, writer oracle_random/joint_fact, reader B8, E14/base_E12/n_updates2, hop0/L4, updated_path True/False; ISD `fb62cf1`. |

## Follow-up studies of WP-S9 (IVD, CVH, TSM, UNS, NCR)

IVD, CVH, TSM, UNS and NCR were registered on master on 2026-10-27, before any lock, smoke or run, and
were run and analysed at wp/sym code `d32463e` (merge `d689235`; reports `docs/reports/{IVD,CVH,TSM,UNS,NCR}.md`).
Each row below uses the "Use of the results" wording of its preregistration and nothing stronger; a result that fails
its rule only narrows the paper's wording, and no similar study follows. All intervals are joint whole-episode
bootstrap intervals (2,000 resamples, base seed `2718`) conditional on the fixed training seeds `17/29/43`; they do
not quantify uncertainty over new training seeds. The levels differ and are stated in each row: IVD **97.5%** (two
writers), CVH **99.166667%** (1−0.05/6), TSM **98.75%** (1−0.05/4); UNS and NCR report **95%** descriptive
intervals with no multiplicity correction. IVD, CVH, TSM and UNS Part B evaluate new seeds (C999004–C999007); UNS
Part A and NCR re-read saved S4242 (and, for NCR, C777) predictions and evaluate nothing. All models are
`r=r_mem=32`. A row labelled **descriptive** has no gate or decision rule in its preregistration.

| ID and paper sentence | Status | Numbers, interval, and set | Conditions | Source file and commit |
| --- | --- | --- | --- | --- |
| IV1. “The IS shrinkage of the depth gain is consistent with key interference, identified by IVD for oracle_random.” | supported | C999004, frozen M1 checkpoints, hops2–3. IC=G(L24-low)−G(L24-high), G=L4−L1 of M1: **+13.975000 pp [12.641667,15.241771]** (97.5%), seed effects **+14.850000/+13.250000/+13.825000 pp**. Both registered conditions hold (IC≥5 pp, lower bound>0): pattern passes. Achieved overlap gap high−low **0.499999977** (≥0.2; label `supported`). G in the L24-low/L24-high cells **+25.541667/+11.566667 pp**; base cell **+33.091667**. | Frozen `oracle_random` r32 M1 (mixed depths) and B4, 12-fact episodes plus up to 24 extra facts (head outside the episode's entities, tail inside); 18 or 36 stored facts lie outside the readers' training distribution; target chains, queries, answers and the 14-entity candidate set are identical in every cell. High-overlap codes are constructed (cosine 0.5) and can be out of distribution. Wording is limited to the registered sentence: it does not make interference the sole cause of IS1, validate Prop.4's bound, or extend to other writers, readers or loads. | IVE `3fa622e`: interference.csv, writer oracle_random, load L24, hops2,3, role decision; summary.json decisions; IVD `2a8d063`. |
| IV2. “Key interference is the identified explanation of the IS shrinkage for joint_fact.” | not supported | C999004, hops2–3. IC at L24 **+3.208333 pp [2.175000,4.208333]** (97.5%), seed effects **+3.400000/+2.450000/+3.775000 pp**: the lower bound is above 0 but IC is below the 5 pp threshold, so the pattern does not pass. Overlap gap high−low **0.285009** (≥0.2; label `not_supported`; a gap below 0.2 would be `inconclusive`, with the same action). G in the L24-low/L24-high cells **+10.833333/+7.625000 pp**; base cell **+23.383333**. | Frozen `joint_fact` r32 M1/B4. Its overlap is measured and selected among 128 candidate heads, so the result is associational; the achieved gap is smaller than for the constructed oracle codes. Use of the result: the paper does not offer interference as the identified explanation of the IS shrinkage for `joint_fact` and states that load, graph layout and candidate distribution were changed together in IS. IS1's numbers are unchanged. | IVE `3fa622e`: interference.csv, writer joint_fact, load L24, hops2,3, role decision; summary.json decisions; IVD `2a8d063`. |
| IV3. (descriptive) Reported items of IVD: IC at the smaller load, IC by hop group, G per cell, decay-only control, full-vocabulary view. | descriptive | C999004. IC at L6, hops2–3, 97.5%: oracle_random **+1.566667 [0.516667,2.641667]**, joint_fact **+0.991667 [0.033333,1.941771]**. IC at L24 on hop1: **−13.683333 [−16.067500,−11.466458]** and **−7.333333 [−9.483542,−5.250000]**; on hops4–5 and 6–8 every L6 and L24 interval of both writers includes 0. G on hops2–3 (95%), base / L24-low / L24-nat / L24-high: oracle_random **+33.091667 / +25.541667 / +10.991667 / +11.566667**; joint_fact **+23.383333 / +10.833333 / +8.283333 / +7.625000**. Decay-only cells of joint_fact (L6, L24): G **+23.383333** = base. Full-vocabulary G, hops2–3, base / L24-low / L24-nat / L24-high: oracle_random **+19.600000 / +13.475000 / +5.550000 / +6.841667**, joint_fact **+5.466667 / +1.866667 / +1.541667 / +1.341667**. | No gate. G in every L24 cell, including L24-low, is below the base cell's G, and for oracle_random L24-high is not below L24-nat; IVD registers only the low-versus-high contrast. No full-vocabulary IC interval is exported. | IVE `3fa622e`: interference.csv (reported rows), cells_m1.csv, cells_m1_full_vocabulary.csv. |
| CV1. “A halting head with the correctness label (the P4-S-L head) and a hop-1-constrained validation threshold repairs overshoot for oracle_random/r32/M1 on these frozen readers.” | supported | C999005 (test seed `999005`, 2,000 episodes per reader), arm A1, θ=**0.10**. Reach gain over L1 on hops2–3 **+9.950000 pp [9.152743,10.844514]** (99.166667%); hop-1 Δ **+2.883333 pp** (point; reported interval [1.688819,4.150000]); mean executed updates **1.967708**. Registered pass: gain≥3 pp with interval above 0, hop-1 Δ≥−1 pp, executed≤8; all hold. | Only this family and arm. The P4-S-L head and its label are unchanged (A0: θ=0.25, hop-1 Δ **−14.800000**, executed **14.209104**, which still fails); only the threshold rule changes, chosen on V4242 under the hop-1 constraint (tolerance 1 pp) and locked before C999005. The pass is a registered pattern, not a match to fixed-depth training: A1's reach gain is **−23.275000 pp [−24.622257,−21.941562]** against fixed L4 and **−2.908333 [−4.122257,−1.608125]** against fixed L16 (CV4). The hop-1 criterion is a point rule. No gate depends on CVH. | CVE `96ee7ef`: endpoints.csv, family oracle_random/r32/M1, arm A1, gating; summary.json passing; CVD theta lock `19c5613`, design `8a13c45`. |
| CV2. “A continuation-value head (A2), or the hop-1-constrained correctness head (A1) for the factorized families, repairs overshoot on these frozen readers.” | not supported | C999005, 99.166667%. A2 `oracle_random`: reach gain **+0.491667 pp [0.336076,0.675000]**, hop-1 Δ **−0.283333**, executed **1.052458** (gain below 3 pp: fails). A1 and A2 of `joint_fact` and of `t1_frozen_fact`: θ **0.05**, reach gain **+0.000000 [0.000000,0.000000]**, hop-1 Δ **+0.000000**, executed **1.000000**: every query stops at step 1 and the accuracies equal L1's at the printed precision. No arm was infeasible (every validation threshold preserves hop 1, `failures` empty), so the confirmation set was opened for all three families. | The prereg's “no arm passes” sentence does not apply because A1 of `oracle_random` passes (CV1); the sentence for the other arms is therefore stated per family and arm. The A2 label margin is 0.25 nats, fixed before data; heads see only current-state features, so locally unpredictable later gains cannot be used. The two factorized families share a writer family and are not independent replications. | CVE `96ee7ef`: endpoints.csv, arms A1/A2 of the three families, gating; theta.csv; summary.json failures. |
| CV3. “A halting arm of CVH is faster than the fixed 16-update read.” | not supported | Wall ms per prepared batch of 64 against fixed L16 (CUDA ms in brackets). Passing arm `oracle_random` A1 **7.828609 vs 3.107745** (7.809237 vs 3.089188), mean executed in the timed batches 1.869792: the registered latency condition fails. `oracle_random` A0 **21.947406**, A2 **5.096679**. The condition holds numerically for four arms: `joint_fact` A1/A2 **2.402189/2.394764 vs 3.081098** and `t1_frozen_fact` A1/A2 **2.420249/2.414957 vs 3.113714**; none of the four passes the registered pattern (reach gain 0, step-1 stopping). No latency claim is made for any arm. No timing intervals. | Wall time is the primary clock, CUDA-event time secondary; P4-S-L protocol (prepared batch of 64, 5 warm-ups, 20 repeats), writer/backbone preparation excluded, so this is not end-to-end request latency. The “Use of the results” text allows a latency claim only for an arm that passes where the condition holds; the numeric flags in `latency_claims` mark the condition alone. | CVE `96ee7ef`: latency.csv (`claim` marks only the numeric condition); summary.json latency_claims. |
| CV4. (descriptive) Reported items of CVH: A0 reference, reach gain against L4 and L16, validation, reproduction of A0. | descriptive | C999005, 99.166667%. A0 (P4-S-L head at its locked θ): reach gain / hop-1 Δ / executed updates `oracle_random` **+32.875000 / −14.800000 / 14.209104**, `joint_fact` **+21.308333 / −34.833333 / 15.995458**, `t1_frozen_fact` **+9.133333 / −22.600000 / 15.992333**. Reach gain of A1 `oracle_random` against L4 and L16 in CV1. A0's recomputed validation accuracy equals the P4-S-L record for all three families (absolute deviation 0). | No gate. Per-hop accuracy tables, the validation threshold tables and the A2 per-step label rates are in `docs/reports/CVH.md`; the full θ-against-validation table is not committed. | CVE `96ee7ef`: endpoints.csv A0 rows, accuracy.csv, theta.csv; CVD `19c5613`: theta.json `a0_reproduction`. |
| TS1. “Neither stay training (T1) nor the sticky gate (T2) removed the overshoot under this protocol.” | supported | C999007 (test seed `999007`, 2,000 episodes), 98.75% (1−0.05/4). No arm passes for either writer (summary `arm_outcome` none). Stay (absolute L16−L4 difference on hops1–3 ≤1 pp, point): holds only `joint_fact` T1 (**−0.794444 pp [−1.427778,−0.141562]**); fails `oracle_random` T1/T2 **−6.661111/−6.400000** and `joint_fact` T2 **−4.355556**. Far (hops4–8, arm at L8 minus T0 at L8, ≥−1 pp): fails in all four, **−11.666667 / −11.780000** (`oracle_random` T1/T2) and **−5.733333 / −4.553333** (`joint_fact` T1/T2), upper interval ends below −1 pp. Gain (arm at L4 minus B4 at L1, hops2–3, ≥3 pp and CI>0): holds in all four, **+16.575000 [15.470781,17.583438] / +16.566667 [15.491667,17.637552]** and **+16.950000 [15.912448,17.895885] / +21.533333 [20.549896,22.575000]**. | One learning rate (0.003, the released recipe) for every arm, no search and no retry, so the result does not generalise from one learning rate. T1 changes loss and training depth (8) together; T2 adds a content-gated sticky update (two extra parameters). 18 runs all completed on the first attempt, no training failure, one code version. Fixed-depth training (M1-L4, B8; B3, B4) remains the remedy found; no follow-up is registered. | TSE `70bfdfc`: endpoints.csv, summary.json arm_outcome/writer_passes; TSD `1bf689b`. |
| TS2. (descriptive) Reported items of TSM: hop 1 at larger depth, arms against T0, T2 gate. | descriptive | C999007; the file has no level column and `arm_endpoints` uses the endpoints' **98.75%**. Hop 1 at L8/L16 minus the arm's own L1: `oracle_random` T1 **−6.466667/−22.750000**, T2 **−6.550000/−22.516667**; `joint_fact` T1 **−25.350000/−25.800000**, T2 **−40.033333/−44.000000** (all eight negative, intervals exclude 0). Arm minus T0 at L4 on hops2–3: **−24.150000, −24.158333, −10.783333, −6.200000** (`oracle_random` T1/T2, `joint_fact` T1/T2); at L16 on hops4–8: **+2.230000, +2.000000, −5.920000, −1.390000**. Mean T2 gate: `oracle_random` at least 0.9999 at every hop and step; `joint_fact` (mean of the three seeds per hop) from **0.599–0.601** at step 1 to **0.282–0.325** at step 16. | No gate. T0 here is the arm retrained under the same code version, not the P2-S-F1 checkpoint. The T2 parameter counts are 4,098 (`oracle_random`) and 13,316 (`joint_fact`), two above T1. | TSE `70bfdfc`: descriptive_contrasts.csv, gate_profiles.csv, training.csv. |
| UN1. “The committed unseen-pool M1 depth-contrast rows for joint, joint_fact and oracle_random are reproduced from the saved samples.” | supported | UNS Part A, S4242 (`test` and `test_unseen`, seed `4242`, 2,000 episodes per pool): 18 r32 runs (three writers × M1 and B4 × seeds 17/29/43) pass the audit, 0 failures. All **18 of 18** comparisons (three writers × L4−L1 on hop1, hops2–3, hops6–8 × confidence 0.95 and 0.975) reproduce, maximum absolute deviation **0.0** (tolerance 1e-9). | Part A is not blind to the committed numbers; its purpose is to audit them. Only these 18 rows are reproduced. The other committed `dataset=unseen` rows of the P1-S, F1 and F2 exports (vs_fixed, vs_loop, replication, M1−M1-lin, B4@1−M1@1, the r16 secondary rows and every F1 row) are transcribed in `context.csv` for context and not recomputed; they stay unaudited. | UNA `491e282`: audit.json, reproduction.csv, summary.json. |
| UN2. (descriptive) “In Part A (released mixed-depth M1) the depth gain is also present on unseen-pool entities, for joint, joint_fact and oracle_random.” | descriptive | UNS Part A, S4242, 95%, no multiplicity correction. Unseen-pool M1 depth gain (L4−L1, hops2–3): `joint` **+7.800000 pp [7.066667,8.508333]** (seen **+7.766667 [6.974583,8.533333]**; unseen−seen **+0.033333 [−1.025000,+1.083542]**; seeds 17/29/43 **+13.475000/+1.875000/+8.050000**); `joint_fact` **+19.958333 [19.108125,20.808333]** (seen **+23.900000**; unseen−seen **−3.941667 [−5.191875,−2.708125]**; seeds **+20.625000/+20.525000/+18.725000**); `oracle_random` **+31.591667 [30.433125,32.750000]** (seen **+32.733333**; unseen−seen **−1.141667 [−2.691875,+0.475000]**; seeds **+32.550000/+30.900000/+31.325000**). Rule: wording “the depth gain is also present on unseen-pool entities” if the unseen-pool gain is ≥3 pp with the interval above 0; all three meet it. | The unseen pool is 256 further names of the same 1,024-name vocabulary with fixed random features, never used in a training episode; the readers were not trained to be tested on new codes. The `joint` unseen gain is +1.875 pp in seed29. The seen-pool values are recomputed from `test` of the same runs, not the committed screening rows (which pool seed17). Part A and Part B are never pooled. No sentence that composition holds for unseen entities or beyond the closed vocabulary is justified. | UNA `491e282`: endpoints.csv, endpoint m1_depth_gain, hops2,3, pools seen/unseen/unseen_minus_seen. |
| UN3. (descriptive) “In Part B (M1 trained at L=4, P2-S-F1) the depth gain is also present on unseen-pool entities, for joint, joint_fact and oracle_random.” | descriptive | UNS Part B, C999006 (new episodes, seed `999006`), 95%. Unseen-pool M1 depth gain, hops2–3: `joint` **+47.008333 pp [45.983125,48.075208]** (seen **+54.083333**; unseen−seen **−7.075000 [−8.533542,−5.583125]**; seeds **+52.050000/+36.500000/+52.475000**); `joint_fact` **+21.025000 [20.083125,21.975208]** (seen **+25.550000**; unseen−seen **−4.525000 [−5.833750,−3.108333]**; seeds **+19.225000/+21.500000/+22.350000**); `oracle_random` **+41.266667 [39.966458,42.533542]** (seen **+42.700000**; unseen−seen **−1.433333 [−3.191875,+0.300417]**; seeds **+42.125000/+40.350000/+41.325000**). All three meet the ≥3 pp, interval>0 rule. | Fresh checkpoints sealed in the lock (18 frozen checkpoints; audit 0 failures). The unseen−seen interval excludes 0 for `joint_fact` in Part A and for `joint` and `joint_fact` in Part B, always with the smaller unseen-pool gain; no rule is attached. Part A and Part B differ in reader and episodes and are not like-for-like. Scope as UN2. | UNB `3fa622e`: endpoints.csv, endpoint m1_depth_gain, hops2,3; summary.json; UND `a836928`. |
| UN4. (descriptive) Reported items of UNS: M1@L4 minus B4@L1 and hop-1 gains on the unseen pool. | descriptive | Unseen pool, 95%. M1@L4−B4@L1 on hops2–3, Part A / Part B: `joint` **−1.116667 [−2.216875,−0.016667] / +13.833333 [12.699792,14.950000]**, `joint_fact` **+20.308333 / +23.125000**, `oracle_random` **+32.116667 / +40.883333**. M1 depth gain on hop1, Part A / Part B: `joint` **+7.183333 / +40.400000**, `joint_fact` **−28.283333 / +2.850000**, `oracle_random` **−15.300000 / +38.733333**. | No gate. Chance 1/14 and per-hop accuracies are in the report and `accuracy.csv`. The B4 contrast is a comparison of separately trained systems; B8 is not part of UNS. | UNA `491e282` and UNB `3fa622e`: endpoints.csv, endpoints m1_l4_minus_b4 and m1_depth_gain. |
| PM1. “In the groups where the hop-2 single read is below the chance line, its wrong answers are mostly the one-hop neighbour of the queried entity; this qualifies the reading of the depth gain D.” | descriptive | NCR, saved predictions of S4242 and C777 (split `test`, 2,000 episodes per run, seeds 17/29/43 pooled), M1 and B4 at L1, hop2. **16 groups** (writers `t1_frozen_fact`, `joint_fact`, `oracle_random`, `oracle_orth` × M1, B4 × the two sets) have hop-2 accuracy **0.000000–1.983333%** against chance **7.142857%**. In each, the first-hop share of errors is **58.666667–100.000000%** (95% lower ends ≥ **56.983333%**) against the uniform share **7.692308%**, and the first-hop rate is **58.666667–100.000000%** against **7.142857%**. First-hop share, S4242/C777: `t1_frozen_fact` M1 **87.159386/86.258558**, B4 **90.437051/89.739099**; `joint_fact` M1 **86.467252/85.079235**, B4 **92.853995/92.847762**; `oracle_random` M1 **69.621256/69.370381**, B4 **96.499858/96.659953**; `oracle_orth` M1 **58.966667/58.666667**, B4 **100.000000/100.000000**. Hop 3: the same four writers are below chance, shares **57.477180–100.000000%**. | A share above its uniform value says that the single read returns the neighbour more often than a uniform error would; it does not by itself identify the mechanism. L1 errors only: errors at L4 are not categorised and no fraction of D is estimated. The tables are used only to qualify the reading of D (that part of D measures escape from the neighbour error). Categories partition the 13 wrong candidates in the order first_hop, one_short (hop3 only), start, terminal_other, other; 95% percentile intervals, descriptive; only the existing r32 P1-S and F2 layouts (not F1, not P2-S-F1). `t1_frozen_fact` M1 C777 hop-2 accuracy **1.983333%** is S12. No threshold, test or decision is attached. | NCE `491e282`: categories.csv (category first_hop, hop 2, L1), accuracy.csv (below_chance); summary.json (48 groups, 72 directories and 288,000 rows verified, none missing or undefined). |
| NC2. (descriptive) Other NCR groups and categories: writers above the chance line, hop 3, `other` category. | descriptive | NCR, 95%. `joint` and `t1_frozen` are above the chance line at hop2 and hop3 in every reader and set, so the neighbour-confusion rate is not reported for them. First-hop share of errors, S4242/C777, hop2: `joint` M1 **10.044496/10.160699** (above uniform 7.692308), B4 **5.326452/5.373708** (below); `t1_frozen` M1 **22.999274/22.642307**, B4 **17.198269/16.868836**. In `joint` the largest category is terminal_other (M1 about 54–55%, B4 about 78–79% of errors). In the four below-chance writers the `other` category holds **0.0–13.2%** of hop-2 errors against a uniform share of 76.9%; under `oracle_orth` the B4 errors are all first_hop and the M1 errors are first_hop (about 59%) and start (about 41%). | No gate. Per-seed shares and error counts are in `categories.csv` and `accuracy.csv`. | NCE `491e282`: categories.csv, accuracy.csv. |

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
| 2: linear matrix polynomial | Shared linear iteration gives `sum_{j=0}^L T^j u`, `T=rho*(R+UMQ)`; memory composition degree can increase with L. At fixed snapshot/L a linear map may be precomputed. | M1-lin depth profiles S4/S6/S7/S8 are compatible with depth-dependent composition and failure. They do not verify a particular polynomial coefficient/gain or show that more degree guarantees correctness. B8 has different per-step matrices (+9,216 parameters; FR1). | supported algebraically; accuracy alone does not identify the polynomial or prove serial reads computationally unavoidable. No amortized precomputation benchmark was measured here. |
| 3: exact chain-terminal decoding | Orthonormal codes, exact oracle writes, nilpotent chains, aligned linear transition with gain greater than one, and matched decoder imply the correct visited node and stable terminal decoding after enough reads. The stable state is a weighted sum of visited codes, not the bare terminal code; the nilpotent transition has spectral radius 0 even with Euclidean norm>1. | S7 contradicts the blanket assertion that a **trained** oracle reader automatically has exact one-hop timing or terminal stability. Alignment/gain/decoder assumptions were not measured, so it does **not** contradict the proposition. H6/HL3 also show learned update count is not graph distance. JS1/JS6 measure trained ReLU reached states, not the exact orthogonal/aligned construction, and neither establish nor refute its assumptions. | supported as an existence construction; narrowed empirical claim to the measured hop/depth cells. Exact oracle storage alone is insufficient. |
| 4: interference amplification | Delta writes with nonorthogonal keys can erase earlier associations; an operator perturbation bound permits depth-amplified error. Growth is an upper-bound possibility, not a theorem forcing the sign of an accuracy contrast. | IS1/C999 gives negative E36−E12 changes in the fixed-checkpoint depth benefit for both primary writers, consistent with the registered operational prediction. IS2 still meets its capability criterion through the largest tested load, and IS3's load-specific training does not improve accuracy under the inherited protocol. S6/H1 remain compatible observations; S8 also changes training duration. | IS measures a load/layout association, not memory perturbation norms, edge/terminal residuals, causal crosstalk, or bound tightness. Layout, candidate count, and query distribution change with E. No universal monotonic law, failure beyond E36, or `sqrt(E/r)` formula for learned chain values follows. |
| 5: conditional single-read information limit | B4/B5 receive memory only through one `MQu`; identical full observations force identical predictions. The chain Bayes ceiling additionally needs exact aligned first-hop access and the stated side-information restrictions. | S2 and B2/B5 support limits of **tested** one-read/fixed-query systems. They do not measure indistinguishable-observation pairs or verify a universal Bayes ceiling for arbitrary trained Q/decoders. The proposition explicitly supplies a counterexample to an unrestricted one-read impossibility interpretation. | narrow the capability claim to the tested architectures/ranks/writers. Candidate labels, a general learned query, and nonorthogonal leakage matter. |
| 6: bias-free ReLU code transmission | A width-r bias-free through-map cannot transmit a positively spanning signed code set exactly; this is not a theorem of universal classification failure for the residual reader. | S1's ReLU success alongside linear hop1 failure is compatible with the theorem: the full reader has reinjection, Rz, and a learned decoder. No exported measurement isolates value-to-query identity transmission. | no claim that the symbolic results disprove the obstruction, or that identity must outperform ReLU for every trained reader. |
| 7: contraction and stopping error | A valid constant below one bounds distance to a fixed point; answer correctness additionally requires task/decoder information. A sufficient norm upper bound failing does not prove divergence. | H3 supplies **no successful certificate**; H1/H4/HL1 stopping policies are empirical. H2/HL4 contradict inferring runtime speedup from fewer updates. JS6 reports reached-state Jacobian radii, not global Lipschitz constants. A smooth map with radius <1 at an actual fixed point has local asymptotic stability; finite reached states, ReLU boundaries, and non-normal transient products need additional control. Neither relative change nor output stability supplies the missing certificate or a correctness proof. | use the negative certificate result and measured stopping tradeoffs; never substitute a pointwise spectral radius for the contraction constant in residual/(1−c), claim certified halting, or infer correctness from convergence. |
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
7. The original symbolic P2-S Gate3 joint practical failure remains a failure (B7). Factorized F-E3 is a capability
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
10. The [IS report](../reports/IS.md) agrees with the audited CSV/JSON at its stated
    rounding. Its “consistent with Prop.4” wording refers to the registered operational
    prediction, not a proof that the upper bound must be tight or that interference alone
    caused the loss. “Capability through E36” means the tested loads and point≥3 pp,
    CI lower>0 rule (IS2); no failure boundary was located. IS-E4's current-over-obsolete
    description is restricted to the displayed primary B8 systems, not every reader,
    and its absolute-rate CIs are not contrast intervals. The design-locked update-task
    validity changes and undefined untouched stale rates remain part of the result.

11. P4-S-L reports agree with raw adjusted endpoints. Its 20.39 ms timing minimum includes
    descriptive joint rows; primary-only minimum is 20.533381 ms (HL4). Head supervision
    includes all eight queries per training episode. Descriptive joint/M1 satisfies numerical
    inequalities but is intentionally not primary/qualifying (HL5). Any-step oracle and true-hop
    are different label-informed references; only the former bounds frozen state selection.
12. JS report rounding agrees with raw exports. Its 97.5% intervals differ from P4-S-L's
    99.166667% intervals. Reduced overshoot is not removal (JS1); joint_fact E2 crosses zero,
    and long/stopped effects are negative (JS2–JS4). Radius diagnostics have no intervals or
    fixed-point certification, and no stopped RS-reference CI is available (JS5/JS6).
13. FR1's 8.9% is specific to the P1-L rank-32 adapter denominator; at rank 128 the same B8
    step matrices add 147,456 parameters, about 31.0% of the registered 475,138. B8 adds 9,216
    parameters at r=r_mem=32, but the symbolic unfactorized joint fraction is about 69.2%. Registered and
    trainable counts differ for oracle/frozen components. Same-rank capability establishes
    neither historical P2 practical-gate passage nor larger-rank efficiency.
14. WP-T10 narrows four earlier sentences. (a) “Post-hoc halting does not repair hop preservation in the
    primary families” (HL1) describes the P4-S-L protocol; CVH adds one registered pass, the hop-1-constrained
    correctness head A1 for `oracle_random/r32/M1`, which stops after about two updates and sits far below fixed
    L4 on reachable hops (CV1, CV4), while the factorized families stop at step 1 (CV2). (b) “Consistent with key
    interference” for IS1 is now writer-specific: identified by IVD for `oracle_random` only (IV1, IV2). (c) The
    statement that the committed `dataset=unseen` rows are unaudited is narrowed to the 18 M1 depth-contrast rows
    that UN1 reproduces; every other unseen row stays unaudited. (d) PM1 is filled for the groups below the chance
    line (PM1, NC2). No earlier row's status changes.

## Pending measurements

A pending row is a measurement the paper may need that does not yet exist. Its status is **open**
and no value is imputed.

No symbolic measurement is pending after WP-T10 part 1. PM1, which WP-T9 left open for the neighbour-confusion rate,
is filled by the NCR analysis; its ID is kept in the follow-up section above. The P1-L confirmation, the memory-use check and P1L-C
are reported in the [LLM ledger](evidence_llm.md#p1-l-confirmation-and-p1l-c-registered-results) (WP-T10 part 2); DPFP (negative result) and the
not-run status of P1L-D are in the same ledger ([DPFP and P1L-D](evidence_llm.md#dpfp-and-p1l-d-registered-result), WP-T10 part 3b).

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
  latency require different evidence (Prop.7, H1–H5, HL4, JS6). A reached-state radius below 1
  is not a fixed-point, global-contraction, or answer certificate.
- “An r² state stores r² arbitrary facts losslessly,” “fixed col(B) makes depth useless,” or
  “few extra parameters make extra computation almost free.” Capacity, function composition,
  state size, and runtime are separate quantities (Props.1–4, B7, H2).
- “No single-read model can compose,” “joint training is necessary for every composable
  writer,” “factorization removes every shortcut,” or “the learned reader has been proved to
  use pure pointer chasing/terminal guessing.” Theorems and measured systems have narrower
  assumptions (Props.5/10, S5/S9/S11, B1/B5).
- “The method passes the general efficiency gate,” “the factorized follow-up measured an
  accuracy–latency advantage,” or “prepared-query timings establish end-to-end/training
  savings.” The original symbolic P2-S Gate3 fails and its missing measurements remain open
  (B7/B8). Current same-rank capability is not larger-rank efficiency.
- “B8 is parameter-matched to B4 or M1,” or an “equal-parameters” claim for any pair other than
  shared M1-L4 versus B4. B8's step-specific R/U/Q add `3*(r²+2*r*r_mem)` parameters: **9,216** at
  r=r_mem=32 (+8.9% of the P1-L adapter, +69.2% of the symbolic unfactorized joint model) and
  **147,456** at r=r_mem=128 (+31.0% of the P1-L adapter). Each percentage belongs to its own model
  and denominator and may not be quoted for another (FR1).
- “The LLM oracle condition is the symbolic oracle condition,” or “symbolic oracle results (S1, S2,
  S6, B2) carry over to P1-L.” The symbolic oracle takes the question code, keys, values and decoder
  from one code table, so its transitions are identity-like; the LLM oracle uses independent random
  pair-key and entity-value tables, so every read after the first needs a learned value-to-key
  lookup (LP2, LP6; Amendment B7).
- Any statement that multiple reads help, or fail, in the LLM at r=32. Amendment B7 fixes that no
  multi-read claim is made at r=32 whatever the gate outcomes: the r=32 readers had not learned to
  use the oracle memory in training logs or Pilot-2 validation (LP6), a failure of that oracle
  design rather than a reader comparison. An r=128 pass would not make the LLM oracle equivalent to
  the symbolic one.
- “The composition holds for unseen entities,” “beyond the closed 1,024-name vocabulary,” or “for
  new entity codes.” UNS does not justify them: its unseen pool is 256 further names of the same
  vocabulary with fixed random features, never used in a training episode, and the readers were not
  trained to be tested on new codes. Every other row here uses seen-pool episodes (`val`, `test`).
  The only unseen-pool wording UNS licenses is “the depth gain is also present on unseen-pool entities”,
  per (writer, reader) and labelled Part A (released mixed-depth M1) or Part B (M1 trained at L=4), for
  M1's depth gain on hops2–3 (UN2, UN3: gain ≥3 pp with the 95% interval above 0 in all six results);
  Part A and Part B are never pooled. It does not cover B4, hop1, M1@4−B4@1 or any other endpoint. The
  committed P1-S, F1 and F2 `dataset=unseen` rows are audited only for the 18 M1 depth contrasts that
  UN1 reproduces; every other unseen row (vs_fixed, vs_loop, replication, M1−M1-lin, B4@1−M1@1, r16
  and F1) stays unaudited and supports no statement. The unseen−seen difference excludes 0 in three of
  the six results, always with the smaller unseen-pool gain (UN2, UN3).
- “OA is an addressing control” or “oracle addressing.” OA replaces the memory read by the true
  value, so it is a value/output ceiling that bypasses addressing, attenuation and interference
  (LL5–LL10 in the [LLM ledger](evidence_llm.md)). No symbolic row uses it.
- “Every seed improves,” “episode CIs establish new-training-seed robustness,” “the reused
  confirmation sample is independent replication,” or “screening stopping flags are
  confirmed gate passes.” Report the actual set, seed effects, selection, and CI scope.
- “Low normalized addressing/value auxiliary loss guarantees a useful raw memory read or
  final answer.” The missing scale, erasure, interference, and decoding conditions are explicit
  in Prop.11 and are not measured by this symbolic evidence table.
- “IS proves the operator bound is tight,” “load alone caused the loss,” “the capability
  boundary is E36,” “load-specific training is always harmful,” or “low stale rates mean
  all update questions are answered correctly.” IS1–IS4 support only the stated layouts,
  inherited protocols, readers, and metrics. JS is now a completed regularization study,
  but it does not certify stable terminal states or a successful overshoot remedy. Latent-readout
  claims still need their own completed evidence.
- “Learned halting repaired the six primary families,” “descriptive joint/M1 passed the primary
  criterion,” “true-hop is an oracle upper bound,” or “chosen depth equals executed cost.”
  HL1–HL5 distinguish the negative primary result, reference semantics, and physical counters. The
  one later registered pass (CV1) is a different head threshold rule on one family, not a revision of HL1.
- “JS improved every endpoint” or “RS+J removed overshoot.” JS1–JS4 support reduced overshoot,
  failed removal, a writer-dependent L4 result, and lower long-hop/stopped accuracy.
- “Key interference is the identified explanation of the IS shrinkage for joint_fact, or for every
  writer,” “IVD showed that interference alone causes the shrinkage,” or “IVD tested the E36 layout.”
  IVD passes only for `oracle_random` (IV1: constructed codes at cosine 0.5, 12-fact episodes plus extra
  facts); for `joint_fact` it does not pass (IV2: IC +3.208333 pp, below the 5 pp threshold) and the paper
  states that load, graph layout and candidate distribution were changed together in IS. IVD's
  `joint_fact` overlap is selected, hence associational.
- “Halting heads repair overshoot,” “the continuation-value head works,” “a halting head repairs the
  factorized families,” or “CVH confirms learned halting.” Exactly one CVH arm passes: `oracle_random`
  with the P4-S-L correctness-label head and a hop-1-constrained threshold (CV1), whose reach gain is far
  below fixed L4 (CV1, CV4). A2 passes for no family, and the A1 and A2 arms of the factorized families stop
  at step 1 for every query (CV2). The label-free and P4-S-L (A0) failures stay reported.
- Any latency claim for a CVH arm. The passing arm is slower than fixed L16 (7.828609 vs 3.107745 ms,
  CV3); the four arms below fixed-L16 wall time fail the pass criterion and stop at step 1. All timings are
  prepared 64-query batches, not end-to-end request latency.
- “Stay training or the sticky gate removes the overshoot,” “TSM shows that no training recipe works,” or
  “T1/T2 is a remedy.” No arm passes for either writer (TS1); one learning rate (0.003) and one choice of loss,
  training depth and gate were tested. Fixed-depth training (M1-L4, B8) remains the remedy found.
- “Most single-read errors are neighbour confusions” for every writer, “the neighbour error is the
  mechanism,” or “X% of D is escape from the neighbour error.” PM1 reports the first-hop rate and share
  only for the groups whose hop-2 single read is below the chance line (`t1_frozen_fact`, `joint_fact`,
  `oracle_random`, `oracle_orth`; M1 and B4 at L1; shares 58.666667–100.000000% against a uniform 7.692308%);
  `joint` and `t1_frozen` read above chance at hop2 with shares of 5.3–23.0% (NC2). A share above uniform does
  not identify the mechanism, only L1 errors are categorised, and no fraction of D is estimated; the tables
  only qualify the reading of D.

## Audit boundary

All displayed empirical numbers were read from the committed CSV/JSON sources identified
above, using bounded streaming CSV/JSON extraction and checking file provenance with git.
Only fraction-to-percent conversion, equal-seed means already defined by the exports, and
the explicitly disclosed A5/JS count pooling and FR1 static parameter accounting were used; no bootstrap was rerun and no new CI
was invented. The decision/report Markdown was used to find claims and discrepancies,
not as the numerical source. This task loaded no models or tensor traces, ran no training,
evaluation, tests, or numerical experiments. The WP-T4 baseline audit submitted no compute
jobs; WP-T6's committed-data figure render is recorded separately in
[its report](../reports/WP-T6.md) and adds no scientific run or new interval. Earlier
theory checks, IS audits/tests, P4-S-L tests, and JS tests are historical evidence, not reruns here.
WP-T7 source artifacts and protocol locks are enumerated above; any committed-data figure
render is documented separately and supplies no new scientific endpoint or interval.
WP-T9 changed no committed number. It added arithmetic on counts already listed here (the rank-128
B8 count and the derived oracle percentages in FR1) and the pending row PM1, which has no value. It
also listed the `dataset` values of the committed P1-S/F1/F2 `reported_contrasts` exports to confirm
that non-gating `unseen` rows exist; those rows are neither audited nor used. The chance level in PM1
was read from the primary E12 confirmation rows of the P1A accuracy table.
WP-T10 part 1 added the follow-up section. Its numbers were read from the committed CSV and JSON files named in
IVE, CVE, TSE, UNA/UNB and NCE with the Python standard library on the login node: fractions were converted to
percent and rounded to six decimals, and no statistic, interval or bootstrap was recomputed. The study reports
(`docs/reports/{IVD,CVH,TSM,UNS,NCR}.md`) were used for the registered wording, the decision tables, the job
records and the manipulation checks; each of their tables is rebuilt from the same files by the study's own
`--check`, which WP-T10 did not rerun. Three items in the reports come from git-ignored run files and are used
in no row: the A2 per-step label rates, the run manifests (node-hours, checkpoint-unchanged flags) and the
non-committed θ-against-validation table of CVH. The mean T2 gate in TS2 and the A0 reproduction in CV4 were
checked against `gate_profiles.csv` and `reports/cvh/theta.json`. No model or checkpoint was loaded and no job ran.
