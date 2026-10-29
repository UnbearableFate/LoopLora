# SLOT-LOOP-v1 S1B: storage and independently trained loop depth (WP-L14)

**Status: registered 2026-10-29 22:45 JST by the orchestrator (decisions at the end).** Drafted by wk-loop on `wp/loop` (73250eb), against master `8f4bb1da18d146d5979a6ae0fe9ea5c5dac27e1f`. No S1B code, model execution, episode generation, calibration, dev or confirmation access accompanied this draft. The orchestrator registers this addendum on master before implementation. Formal execution additionally waits for S1A completion, its recorded verdict/ablation status, the S1B preflight/lock and an approved ready release. This document defines future behavior; the existing S0/S1A runner does not implement it yet.

Authority: [parent preregistration](SLOT-LOOP-v1.md), especially registration decision 2; [task WP-L14](../tasks/WP-L14_slot_s1b.md); [plan](../../plan/looplora_diagnosis_and_execution_plan_2026-10-29.md) §§5, 8, 11–13 and its execution record. The 2026-10-29 budget clarification supersedes historic aggregate and one-hour caps: every job remains at most 16 node·h, with measured walltime ×1.5. No old verdict or writer gate is reopened.

## 1. Question, branch and common parent

S1B tests the prerequisite of H2: does independently training four shared reads improve multi-hop answer accuracy over independently training one read of the **same backend**? It does not establish per-transition consistency training, relation composition, semantic halting or H3; those belong to S2. Equal updates and samples do not imply equal FLOPs or elapsed time.

The following branch is mechanical, determined only by S1A's registered outcome. No branch is selected from S1B scores.

| S1A outcome | S1B formal arms | query projection | consequence |
| --- | --- | --- | --- |
| ADVANCE | M1, M4, S1, S4 | shared, frozen `A` | full Memory × Depth comparison |
| NOT ADVANCED; §7.4 ablation absent or not flagged | M1, M4 | shared, frozen `A` | slot expansion stops; matrix loop claim only |
| NOT ADVANCED; §7.4 ablation flagged (registered dev gain ≥3 pp) | M1, M4, S1, S4 | **every arm** gets a trainable `A_query = clone(A)` | new four-arm confirmation of the query-copy architecture; S1A remains NOT ADVANCED |

M denotes ridge Matrix, S denotes SoftSlot, the numeral denotes the **training and primary evaluation depth**. HardSlot is not a training arm. An invalid/unstable S1A without a resolved orchestrator disposition is not a valid negative verdict and cannot select a branch. S1A confirmation data never enter this study's manifests or comparisons.

Every arm and every run starts from the identical, unadapted S-RE inference export, **not** an S1A winner, ablation export, pilot or a trained L4 checkpoint. Parent: `/work/xg24i002/x10041/LoopLora/runs/p1l/sq2/arms/S-RE/writer.pt`, blake2b-128 `3994a5631021de03b2039974ac934e56`; parent config hash `1bfb72b7d87bc0aff743c1e018e8680e`. The inherited inventory also pins `record.json` to `816570a0882ed6952b4f851b67d4ef9f`; S1B re-verifies actual files on compute. Parent optimizer and training-only value head are unused.

Fixed: pinned Qwen3-0.6B-Base, layer 14/o_proj, backbone bf16, adapter/state fp32, rank/value dimension 128, DPFP ν3 key dimension 768, ridge η=1e-4, parent's normalization and epsilon, ρ=0.25, α=1 and ReLU. Writer `A,K,V,a_lambda,a_beta`, `input_mean/input_std`, backbone and preprocessing stay frozen. Only `Q,R,U,B` train (180,224 parameters); in the flagged branch `A_query` also trains (+262,144; total trainable 442,368). Total inference parameters are 475,138 / 737,282 respectively, equal across the active main arms. Writer `A` always stays frozen. No restandardization, auxiliary losses, additional head, relation/bridge input, teacher-forced path or learned temperature is introduced.

## 2. Task/layout choice and evidence

**Choose the plan's E12, two-chain T2 chain-end, with lengths (8,4).** Retain the old T2 metadata `n_rel=4, head_sharing=0.3`; these knobs affect T1, while the plain T2 generator writes and queries **relation 0 exclusively**. Setting `n_rel=1` would change the digest without changing these graph semantics. No alternate E16 layout is run or selected.

The complete `TaskSpec` (including defaults, for a reproducible layout digest) is:

```json
{"E":12,"Q":4,"V":1024,"base":null,"chain_lengths":[8,4],"head_sharing":0.3,"hops":[1,2,3],"missing_rate":0.5,"n_distract":0,"n_rel":4,"n_updates":2,"name":"t2_chain_end","seen":[0,768],"unseen":[768,1024],"update_query_rate":0.5}
```

Generator `relations-v1`; layout blake2b-128 **`f556c1b91252b0b74d3ff674717be37c`**. Digest = `stable_hash(asdict(TaskSpec))`, JSON sorted keys, compact separators, blake2b digest size 16. The integer/default values and digest were derived on login from the source AST with stdlib only, without importing the generator. Re-derivation with the actual `TaskSpec` is a mandatory compute preflight check; mismatch stops execution.

Evidence and limits:

- S-RE's real `record.json` config is T1/E16/Q4, four relations, head sharing 0.3, hops [1]; its layout digest is `b7f93c1ca56e5dd1f79a9dccfe2f6054`. T2 changes fact count, repeated head/tail roles and question wording. The frozen writer transfers structurally; its accuracy on these chains is **unverified**, and no T1 score licenses a T2 performance claim.
- `relations.py:TaskSpec`, `_layout`, `_chain_queries` and `generate` already support E12/(8,4), relation-zero chains, uniformly sampled training hops 1–3 and shuffled fact presentation. `LLMExperiment.writer_signature` in `llm/train.py` binds vocabulary/backbone/insertion/rank/write policy/features, not E or task name. `_WriterFix.scan_final` and `WriterFixExperiment.prepare` in `writer_fix.py` operate over the actual fact positions, with one `fact_final` write per fact; there is no E16-shaped learned parameter. This is code evidence of structural compatibility, not an executed E12 check.
- Old P1-L used precisely E12/(8,4), `n_rel=4, head_sharing=0.3`, train hops 1–3 and evaluation hops 1–8 (`configs/p1l/main-r128/M1-L4-s17.json`). Its prereg allowed T1-trained learned writers to transfer, but **only oracle_random actually ran** (`docs/reports/P1L.md`; learned writer failed its gate). Its positive shared-L4 result therefore does not validate S-RE on T2. WP-C6 tested frozen-writer/reader-sharing compatibility and T2 loop machinery, not this parent's chain accuracy.
- S0 code currently enforces T1/E16 in `slot_experiment.load_parent`; `slot_data.task_spec`, `slot_train.fit`, `evaluate`, and export metadata also assume T1/L1 or one evaluation question. Load S-RE strictly under its original config first, then attach an explicit, separately validated S1B task/depth. Never weaken the SQ2 signature or silently mutate the S1A global task. S1B needs task-aware evaluation and trajectory capture; S0 `final_query/final_read` recompute only the first read and cannot stand in for L4 traces.

Training has 32 episodes × four questions, sampled on hops 1–3. Evaluation keeps **one query per hop per episode** using `sym/evaluate.stratify_queries`, with its purpose base seed as query seed. Calibration, pilot selection and scheduled checkpoint selection score hops 1–3 only (equal weight per hop). Confirmation and the post-selection descriptive dev depth sweep score hops 1–8, eight correlated questions per unchanged graph. Evaluation uses the **same TaskSpec/digest above**, then stratifies; it does not regenerate graphs with a different `hops` field or set Q=8 in the generator. `TaskSpec`'s hop ceiling allows 8 via the first chain, and stratification constructs the evaluation queries from each graph.

The chain-end prompt asks to follow relation 0 to its terminal; true hop is not rendered (`data/render.py:_question`), and `q_hop_given=-1`. True hop, fact indices, path entities and terminal labels are metrics/loss inputs only. There are no missing edges or UNKNOWN targets; UNKNOWN prediction frequency is descriptive, UNKNOWN recall is not applicable. Graphs and fact order are identical across arms/seeds at a given purpose and index. Support KV/raw support/hidden history and the alternate backend are unavailable to question forward; each question/depth forks the same read-only owner-indexed memory. Ordinary SoftSlot has capacity **12**, no recency variant or hidden Matrix fallback. Slot bytes including bool mask and int64 count: `4*12*(768+128)+12+8 = 43,028` per episode; Matrix is 393,216. These are logical state sizes, not deployment peak memory.

## 3. Data namespace, seals and manifest

Use the parent's derivations with stage label **S1B**:
`eval_base(purpose)=derive_seed(20261029,"slot-loop-v1",purpose,"S1B")`;
`train_base(run)=derive_seed(20261029,"slot-loop-v1","train","S1B",run)`.
Episode identity remains `derive_seed(base,"relations","relations-v1",layout_digest,split,index)`. Bootstrap uses the evaluation-style purpose `bootstrap`. The resolved integers below were independently computed with the source's stdlib blake2b formula; runtime must re-derive and refuse mismatches.

| purpose | split / indices | episodes | base seed |
| --- | --- | ---: | ---: |
| smoke / preflight / reload | train [0,64) | 64 | 6584302131768675481 |
| calibration | val [0,2048) | 2,048 | 254983947625000635 |
| pilot selection shard | val [0,1000) | 1,000 | 2437977920860386590 |
| dev | val [0,1000) | 1,000 | 5100482262494855894 |
| confirmation, sealed | test [0,2000) | 2,000 | 3460300042162207747 |
| optional unseen, sealed | test_unseen [0,2000) | 2,000 | 4377963709756993594 |
| train seed 17 | train [0,192000) | 192,000 | 693232309558345634 |
| train seed 29 | train [0,192000) | 192,000 | 373104883948690398 |
| train seed 43 | train [0,192000) | 192,000 | 6045459755310898920 |
| LR pilot train | train [0,6400) | 6,400 | 907865416675862555 |
| bootstrap, no episodes | numpy default_rng | 10,000 draws | 5799536610111230359 |

The unseen diagnostic is enabled/disabled in the release **before any S1B formal run** and never substitutes for confirmation. All active arms/seeds use the same evaluation graphs and query ordering. Reuse of a training stream among main arms and preregistered mechanism controls is intentional pairing, not disjointness failure. Each unique purpose/stream is registered once.

Extend the purpose-seed registry with stage-specific layout, count and query-hop metadata; preserve existing S0/S1A entries and old `CONFIRMATION_WINDOWS` literally. Check pairwise ID disjointness between all distinct S1B sets/streams, S0/S1A registered sets and complete registered train prefixes, and the inherited old windows/D128-3/S-RE streams. **Use each old entry's own original layout digest**, not S1B's digest to reconstruct old identities. This metadata-only check may derive sealed IDs; it must not generate sealed graphs, questions or scores. No old confirmation payloads are read. New purpose seeds and different layouts are not by themselves a completed disjointness check; the S1B lock records that check's actual result.

Sealed seeds occur only in the registry under `src/` (source-scan tests inherit the S1A rule); committed manifest metadata omit their base/query seeds, keeping counts, ordered ID digest and query design. Both registry acquisition and raw/stratified generation reject unopened sealed entries. Old S1A open tokens cannot open S1B. Analysis requires `--manifest` and verifies episode IDs, hop coverage/order, counts, source/release/config identities and checkpoint provenance, refusing partial or mismatched inputs.

All persistent outputs live below `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/`; committed metadata/reports below `reports/slot-loop-v1/s1b/`. S1B's own lock/inventory do not replace the S0/S1A lock. Records and every sample file identify arm, continuation seed, backend, projection mode, trained/evaluated depth, selected checkpoint blake2b-128, exact read settings, manifest/release digest and source commit. Keep per-episode/per-hop correctness, NLL and predictions; no flattened-question bootstrap.

Before confirmation: all active main runs finish; dev selects checkpoints; an independent reviewer re-runs the **frozen** dev analysis; write selection with all selected checkpoint digests and stop for the orchestrator. The orchestrator's S1B open record binds this addendum and the parent prereg digests, branch/contrast list, lock/inventory, the exact ready-release **file digest**, frozen analysis-script digest and the complete arm/seed → checkpoint map (6 or 12 entries). All training records must carry the same approved release digest. The opening-root paths are explicit and absolute. Confirmation evaluation permits only the selected mapping and release-listed parent/step-0 references, each with an unambiguous label; reference digests cannot occupy a main label. The chosen L4 checkpoint's descriptive sweep is authorized at L=1/2/4/8 under the same checkpoint mapping. Reference L1 parent rows are descriptive and must not replace independently trained L1 controls. Evaluation/analysis stdout carries counts/status/digests only; the verdict and metric artifacts are opened at the registered analysis stage. Freeze code/config/selection once confirmation opens; any revised science requires a new addendum and fresh confirmation.

## 4. Calibration, training, pilot and release

After the S1B preflight passes and its lock is committed, calibrate **once per active backend** on S1B calibration, hops 1–3, using the untrained common S-RE reader at **L1**. Copy branch, if active, uses its identical freshly copied `A_query`; nothing is optimized before calibration. Matrix g ∈ {0.5,1,2}; SoftSlot g ∈ {0.5,1,2}, τ ∈ {0.03,0.1,0.3,1}. Use mean full-vocabulary answer NLL; candidates within 1e-4 nats of the minimum tie, then gain nearest 1, then larger τ, as S0. Matrix-only has three candidates, full design 15; HardSlot is not needed. Report uncalibrated g=1/τ=1 descriptively. L1 and L4 of a backend share the selected g/τ for the entire study. No S0 values are imported and no trained checkpoint or sweep depth is retuned.

Main arms use the parent S1A optimizer contract: fresh AdamW (betas .9/.999, eps 1e-8, weight decay .01), clip global norm 1 with nonfinite refusal, linear warmup 20 updates then constant LR, answer CE only, all valid question-token reads, immutable support, deterministic runtime and identical batches/order within seeds 17/29/43. Freeze checks compare all frozen tensor/buffer bytes and fixed-smoke-batch K/V/Matrix before/after smoke, each checkpoint and run end; verify the optimizer's exact trainable set and finite nonzero reader gradients after backward. No backbone gradients/updates; the differentiable upper-model read path remains intact. Record every step's loss/pre-clip norm and clipped fraction.

**LR pilot:** each active arm/depth gets the same {3e-4,1e-3,3e-3} candidates ×200 steps from the common parent, the identical pilot training stream and one hop1–3 pilot-shard evaluation, plus its step-0 row. Choose per arm the lowest pilot NLL; within .01 nats tie, lower LR wins. Nonfinite candidates are excluded and recorded; no finite candidate, or every finite candidate worse than step 0, stops for the orchestrator. Pilot weights are discarded. This is equal tuning effort, not a common-LR optimization trajectory.

**Lock N=6,000 updates** for all active main arms and all seeds (192,000 episodes / 768,000 questions per run). Historic aggregate caps no longer require shrinking N to a one-hour job. Pilot timing changes requests/packing only, never N based on scores. If a complete run cannot fit the ≤16 node·h job cap, stop for an execution decision before formal runs; do not silently reduce N. Batch size, read positions and science precision cannot vary by backend/depth. Physical microbatching, if needed, must be common and preflight-verified to preserve the registered logical update; otherwise stop for an amendment.

Dev at step 0 (not selectable) and steps 600,1200,…,6000. Select lowest mean hop1–3 dev NLL; within 1e-6, earlier checkpoint wins. Every access is logged. No h4–8 score selects a checkpoint. Nonfinite loss/gradient/NLL, incomplete runs or a missing selectable export are failures, never zero-score data; if a selected checkpoint's dev NLL exceeds its step-0 row, mark unstable and stop before opening confirmation. All main runs/contrasts remain mandatory; no dropping a bad seed/backend.

Release after timing, calibration, 20-step smoke/freeze and real JSON export→candidate release→CLI reload checks: stage/branch, N, per-arm LR/read/depth, parent/task digest, all main science-config hashes (6 or 12), source digest/full 40-hex science commit, preflight/lock/inventory, both prereg file digests, frozen analysis-script digest, selected optional unseen flag, resource estimates and checkpoint-selection/bootstrap/contrast rules. Reload must reproduce ordinary-smoke IDs, predictions, correctness and NLL and restore complete buffers/backend/projection/depth metadata. Candidate `pending_reload` never trains. STOP at ready release for orchestrator `approval.json`; write formal records with that exact release digest. Optional mechanism runs require a separately recorded release extension, without editing the main ready-release file or confirmation mapping.

## 5. Confirmatory contrasts and continuation

Let `c[a,s,i,h]` be full-vocabulary 0/1 correctness of selected arm a, continuation seed s, episode i and hop h. For A−B, `d_i = mean_(s∈{17,29,43}, h∈{2,3})(c[A,s,i,h]−c[B,s,i,h])`; Δ is the mean over 2,000 confirmation episodes. Report each seed's corresponding Δ_s and a separate hop-1 point difference averaged across seeds. This is conditional on one S-RE writer, these seeds and this layout.

| branch | registered contrast family | two-sided interval per contrast |
| --- | --- | --- |
| matrix-only | C_M = M4−M1 | 95% |
| either four-arm branch | C_S = S4−S1; C_M = M4−M1; C_storage = S4−M4 | **98⅓%** (Bonferroni, family size 3) |

Use 10,000 paired **whole-episode** resamples with numpy `default_rng(5799536610111230359)`, one index draw shared by all arms/seeds/hops/contrasts. Seeds are averaged before resampling; all questions from an episode remain together. Percentile bounds use numpy linear interpolation: quantiles .025/.975 (one contrast), or **1/120 and 119/120** (three contrasts). Compute threshold comparisons exactly from correctness counts, without rounding. Supplemental per-seed intervals and the plan's 97.5% intervals may be shown as descriptive only, never substituted for the registered family.

A backend has a **confirmed loop gain** iff its independently trained L4−L1 contrast has Δ ≥3 pp, adjusted lower bound >0, Δ_s>0 for each seed, and hop-1 point drop ≤2 pp. C_storage supports a slot advantage at L4 iff Δ>0, its adjusted lower bound >0 and every seed's point difference >0. There is no claim that a slot advantage alone establishes a loop gain or interference-removal mechanism. The matrix contrast is retained even if the slot contrast fails. Report full intervals, counts, NLL and per-hop EM irrespective of pass/fail; S1−M1 and the Memory×Depth interaction are descriptive.

For each selected M4/S4 checkpoint, score L=1/2/4/8 on the identical inputs/memory, with all-token scope fixed. Save actual query/read/core trajectories at each loop of the final valid question token. This sweep is **descriptive** and never replaces M1/S1. Preregister its contribution to plan §8.3: a backend's confirmed loop gain opens S2 only if the same L4-trained checkpoints also have pooled hop2–3 EM@4 > EM@1, no negative per-seed sweep difference, and verified trace/forward agreement with immutable memory. No monotonicity at L8 or perfect edge-by-edge decoding is required; neither is claimed. A trace disagreement is an engineering failure. The sweep sign requirement is a descriptive continuation screen, not another inferential success claim.

Continue on SoftSlot only when its loop criterion and sweep screen pass. If Matrix passes both and slots do not establish an additional L4 advantage, S2 may proceed with Matrix as the primary backend. If only Matrix passes both even with a positive storage contrast, Matrix is the only loop-qualified backend. If both qualify and C_storage passes, SoftSlot may lead S2 with Matrix retained as comparator. A qualifying SoftSlot alone permits S2 on it, but a failed C_storage forbids a slot-superiority claim. No qualifying backend means no S2 advancement: shrink claims to reading/memory adaptation and report the negative. S2 still needs its own registration; S1B does not certify semantic transitions/NULL/terminal stability.

## 6. Conditional mechanism controls and diagnostics

Run these only **after the confirmation analysis finds a confirmed loop gain** for that backend; absent a gain, neither control trains. To avoid adaptive reuse, they use S1B train/dev only and cannot change its confirmation verdict or the continuation rule above. Their results are descriptive evidence to inform the separately registered S2 mechanism tests. Each triggered backend gets both controls, seeds 17/29/43, N=6000, main L4 LR/g/τ and identical data/checkpoint rules; no new LR search. Train them after a release extension records all configurations and measured costs. If a control cannot be executed as specified, report it unrun rather than replacing it after scores are seen.

1. **Fixed-query L4:** from the common parent, q uses z0 at every loop; compute/cache its read and iterate R four times, then train with answer CE. The parent projection mode/B and frozen set are unchanged. Also a no-update clamp of the selected shared checkpoint may be shown on dev, explicitly labeled separately from this independently trained control. The cached control performs one backend read per valid position, four recurrent updates; it must not be billed as four executed memory reads.
2. **Warm-start untied L4:** from the same S-RE parent, copy initial shared Q/R/U into all four steps; retain one common B and A_query if applicable. Prove full logits and state trajectories equal the shared L4 initialization bitwise before optimization. Then independently train the same N updates with fresh AdamW. Starting from an already trained L4 and training another N would violate this equal-update control and is excluded. Untying adds 147,456 parameters (three extra copies of Q/R/U): 622,594 inference parameters, or 884,738 in the copy branch. Old B8's random/unmatched initialization is not reused.

Diagnostics at the final valid question position use the **actual forward** q_s,m_s,z_s and written DPFP keys/raw values. Report key top-1, SoftSlot true-path weights/gap/entropy, path-target relative read error `||m−v_next||²/(||v_next||²+1e-6)`, cosine/norm ratio per step, and separate terminal drift summaries. For s≤h path-target is the s-th edge on the gold path; after completion it is N/A (there is no NULL slot). A state-target is reported only with an explicitly validated state decoder, separately from path-target; otherwise N/A. No forced assignment of the final answer's fact to every loop or early question token. No new value head is trained for this study. Per-step metrics are descriptive; H2's semantic consistency claim still awaits S2.

## 7. Cost accounting, estimates and retained artifacts

Record separately: loop_depth, number of valid question read positions P (excluding padding), **logical_reads=P×L** for ordinary iterative/untied arms, Python/backend calls and their batch-row counts, measured GPU read/kernel time, full training step time, support preparation and complete question-forward time. Different token positions/owners can be batched: one call is not one scalar read. Fixed-query has P executed logical reads (cached), 4P recurrent updates/readout uses; retain both counts. One wrapped base projection and one upper-model question forward remain independent of L in this architecture; instrument the actual counts. Record warmup/sample counts, median/P90 and synchronized elapsed timings; isolate instrumented microbenchmarks from uninstrumented end-to-end timing. No FLOP/time equivalence or S3 deployment claim follows from equal N.

Known measured anchors: real S-RE `record.json` reports **0.1990696014 s/step** and **6.8412884050 s/1000 one-question episodes** at L1; step-0 evaluation took 9.1088620980 s. WP-C6's real-backbone T2 batch-32/r32 shared-L4 benchmark measured **0.27355 s/step** (`reports/llm-b8/benchmark.json`, job 3715193). These are different task/rank/objective/backend settings, **not S1B timing measurements**. S1A's pending pilot supplies the first measured slot-read training anchor; S1B's own depth/task pilot decides every request. Do not invent a measured SoftSlot-L4 time or scale full step time by exactly four.

For planning only, S-RE's L1 anchor gives 6000 steps ≈1194.4 s; conservative query-count scaling gives a hop1–3 dev pass ≈3×9.109=27.33 s, and 11 passes ≈300.6 s. With an explicitly provisional 60 s load, a run is ≈1555 s (0.432 node·h actual forecast), request ×1.5 rounded up ≈40 min. Using WP-C6's L4 step anchor instead gives ≈2002 s (0.556 node·h), rounded request ≈60 min. These are planning scenarios, not runnable release values; the 60 s load is unmeasured here. Actual shared-support evaluation need not scale linearly with questions. Read latest S1A pilot **timings only** when available and replace scenarios with measured per-arm S1B measurements at release.

| job class | quantity (matrix-only / four-arm) | expected-time model, resolved before submission |
| --- | --- | --- |
| targeted tests / real-parent preflight | short shards / one preflight | nearest measured related test time, then its own measurement; S0 preflight 53 s is only an anchor |
| 20-step smoke and reload | 2 / 4 arms | load + Σ(20×measured step + smoke/reload evaluation) |
| L1 calibration | one per backend | load + 3 / 15 candidate hop1–3 evaluations |
| 200-step LR pilot | 6 / 12 candidates | load + Σ(200×arm step + pilot evaluation), plus step-0 evaluations |
| main arm/seed | 6 / 12 jobs | load + 6000×arm steady step + 11×hop1–3 dev evaluation + measured export/I/O overhead |
| selected dev sweep / confirmation / unseen | shard by arm/seed/depth | load + measured 1000/2000-episode eight-hop evaluations, preserving complete paired records |
| frozen analysis | one per analysis stage | measured related analysis time, then actual S1B time; S0 analysis was 6 s |
| conditional controls | 6 jobs per qualifying backend | same main-job model, with separately measured fixed/untied step times |

Measure steady steps 11–200 in the pilot, use the largest finite candidate's mean per arm and the largest observed load/evaluation/export overhead in its cost class. Include generation, tokenization, synchronization and serialization where actually incurred. Request `ceil_to_10min(1.5×expected_seconds)` (short jobs may round to minutes), never more than **16 h on one node**. Do not clamp an over-cap estimate into a false request: shard independent candidate/evaluation units or STOP if an indivisible training run exceeds the cap. Use debug-g only for ≤30 min; otherwise regular-g. At most six wk-loop jobs queued/running. All compute goes through `scripts/submit.py --commit <resolved full 40-hex hash>`; no HEAD literal, second interpreter for Python script payloads, relative outputs, sleep or polling. End a turn WAIT for the orchestrator's watcher, or STOP at gates. Record requested and actual node·h (including failures/retries) and cumulative totals. Historic 11.5 node·h is context, not a cap. Under the illustrative anchors, main training alone is roughly 3–6.7 actual node·h depending on branch/timing; optional controls add cost, never silently disappear to meet the old total.

Retain one selected model export, training log/every-step loss and norm arrays, all scheduled dev summaries, final frozen/support digests, selected per-episode/per-hop predictions, compact loop diagnostics, calibration/pilot tables, lock/release/selection/open/analysis source inventories and PBS ledger. Keep model exports separate from episode memory; no raw support, dual-state snapshot, training-only head or cache in an inference export. Transient caches go to job-local scratch. Checkpoints needed to reproduce selection remain until selection is audited; no deletion of user artifacts in this stage. Report metrics and resource measurements in separate tables with explicit units.

## 8. Preflight, source lock and review sequence

Implementation stays on wp/loop after registration. Keep S1A's registered path bitwise unchanged and its task registry, sealed sets, exports and decisions isolated. No further merge from master without the orchestrator's instruction. Core changes receive the task's **one** cross-review (rv-code and an Opus reviewer); compute tests check fixes, with no automatic further review round.

Required checks before **any S1B calibration/pilot/dev**:

| check | evidence required on compute |
| --- | --- |
| compatibility identity | unchanged S0/T1 goldens and S1A off-path behavior; Matrix-L4 uncalibrated g=1 logits and reader gradients bitwise equal the pre-S1B native tensor loop on fixed T2 batches, plus existing T1 replay/golden cases at L1/2/4; initialize A_query=A and verify equality too |
| writer/task transfer | strict parent digest/config/buffer load; actual T2 TaskSpec hash equals registered digest; E12 support positions yield exactly 12 writes and correct [B,12,768]/[B,12,128] captures; no standardization call |
| trajectory fidelity | captured q/m/z at each of L1/2/4/8 reproduce the actual recurrence/forward; disabled capture leaves outputs bitwise unchanged; no label use by non-oracle forward/read |
| owner/mask/isolation | two sentinel episodes, four/eight-question expansion, padding, row reorder and depth forks read only the correct owner; memory bytes unchanged throughout |
| frozen set and reachability | real-parent short smoke for every active arm; fixed support K/V/Matrix and all frozen tensors/buffers unchanged, exact optimizer set, finite nonzero trainable gradients/updates |
| backend features/capacity/bytes | actual DPFP code, no base-key substitution; empty/padded state correct, E=S=12 allowed, E>S raises; state byte sum includes masks/counts |
| paired data/seals | every purpose's original layout/seed and full registered train prefix checked; no S0/S1A/old-ID overlaps; ordered graph/query identities match across arms; unopened S1B generation refused; old protections unchanged |
| evaluation/statistics | genuine multi-question episode/hop pairing, expected integer contrast thresholds and Bonferroni quantiles, per-seed/hop1 screens, exact selection/ties, missing/unstable/provenance mismatch refusal |
| export/reload/release | complete JSON-shaped task/projection/backend/depth/buffers metadata and parent provenance survive real smoke→release-builder→CLI reload; swapped checkpoint labels, rebuilt release digest and changed analysis script refused |

The preflight uses **ordinary S1B smoke only** plus existing registered identity fixtures; no S1A confirmation and no S1B calibration/dev. Where legacy T2 support is unavailable, compare the existing generic native tensor `FixAdapter` loop with the new Matrix path on the same prepared smoke batch; do not fabricate an old learned-T2 result. Bitwise mismatch stops before scores, with max difference/cause; a tolerance requires registered clarification.

After passing, commit S1B preflight/inventory/manifest lock bound to sorted `src/` relative paths and file bytes by blake2b-128, excluding bytecode, following the existing lock contract. Every S1B science mode checks actual source and input digests against it. **Any src edit invalidates this lock and requires a new preflight**, including later mechanism code; if controls need a new source release, retain the immutable main lock/release and issue a separately named control lock after new preflight. Docs-only commits do not change the source digest. Do not import S1A's S0 source hash as proof for S1B.

Cycle: register addendum → implement and compute-test → one cross-review/fix validation → preflight/committed lock → calibration/smoke/pilot/reload → ready release/STOP for approval → common-start main runs → dev selection/independent frozen-analysis re-run/STOP → orchestrator open record → paired confirmation/sweep/frozen verdict → conditional dev-only controls → report/STOP. Jobs and ledger are future work; none is submitted by this draft.

## 9. Explicit deviations and resolutions for registration

1. **Branching/copy architecture:** decision 2 authorizes matrix-only after a null and delegates the flagged-ablation choice. This draft chooses all four arms with identically initialized trainable query copies in the flagged case. No use of ablation-trained weights; comparisons are conditional on this architecture and fresh S1B confirmation.
2. **No task-layout substitution:** E12/(8,4) follows §8.1, retaining the legacy n_rel/head-sharing metadata while documenting actual relation-zero semantics. T1/E16→T2/E12 is an explicit frozen-writer transfer, not evidence already established by oracle P1-L. Runtime compatibility and quality are not asserted in advance.
3. **Multiplicity and matrix continuation:** plan §5.4 named two primary contrasts with 97.5% intervals. Registering M4−M1 as a possible independently supported continuation requires a third contrast in four-arm branches, so this draft uses the stricter **98⅓%** intervals for all three. Matrix-only has one 95% contrast. Loop +3 pp and hop1 −2 pp thresholds are retained; unanimous seed direction, storage-positive rule and the numerical descriptive sweep screen are explicit conservative completions of §§5.4/8.3.
4. **Training schedule and pilot:** N=6000 is fixed, replacing the inherited one-hour timing-derived schedule under the lifted caps. Equal-effort LR selection applies to each backend×depth (not just backend), with its own pilot shard; 10 scheduled dev checkpoints and S1A's optimizer/ties are retained. This avoids selecting depth-specific updates and does not equalize compute.
5. **Calibration scope/count:** use S0's identical g/τ grid and tie rule but only the active Matrix/SoftSlot backend candidates (3 or 15), L1 parent, equal-weight hop1–3 S1B NLL. No HardSlot training, extra grid, recency arm or checkpoint recalibration.
6. **Mechanism controls:** §8.2 leaves the gain trigger, untied starting checkpoint, seed count and data access unspecified. This draft waits for confirmed loop gain, compares identical parent initialization with equal updates and uses dev-only descriptive controls after confirmation; no adaptive confirmation reuse or extra confirmatory conclusion. Their results inform S2 but cannot rescue a failed S1B.
7. **Later-stage estimate is conditional:** no S1B or slot-L4 step timing exists here. Explicit measured anchors, forecast arithmetic and per-job formulas are provided; S1A/S1B pilots must resolve them before release. Lifted aggregate caps do not authorize an unmeasured 16-hour request.

Registration must accept or amend these choices before code. This draft changes no source, config, parent prereg, existing lock, old report verdict or S1A behavior.

## Decisions at registration (orchestrator, 2026-10-29 22:45 JST)
1. **Deviations 1 to 7 are accepted as written.**
2. **Seeds and layout checked.** The orchestrator re-derived with stdlib blake2b:
   - the eleven base seeds of §3;
   - the layout digest `f556c1b91252b0b74d3ff674717be37c` of the §2 `TaskSpec`, whose 15 fields match `data/relations.py`.

   All match. The compute preflight still re-derives them with the real `TaskSpec`.
3. **Parent transfer, one sanity stop.**
   - The calibration output reports the untrained L1 parent's EM and NLL per hop. This is descriptive.
   - If the calibrated Matrix-L1 parent's hop-1 EM on S1B calibration is below 10 %, STOP before the pilot: that points to a broken transfer, prompt or position rule, not to a reading limit.
   - Otherwise the transfer quality is reported, not gated.
4. **Branch source.**
   - The branch comes from S1A's frozen decision record and its §7.4 ablation record, and is written into the S1B release.
   - S1B formal runs start only after S1A's decision record exists. An S1A that ends unstable or invalid needs the orchestrator's disposition first (§1).
5. **Process.**
   - Implement, then test on compute.
   - The S1B core code gets one cross-review (rv-code and an Opus subagent). Fixes are checked by tests only (user, 2026-10-29: "不要搞这么多次review,核心代码没问题就行").
   - Then the preflight and the lock; calibration, smoke, pilot, reload and release; and STOP for `approval.json`.
6. **Resources.** Each job ≤ 16 node·h (user, 2026-10-29). The §7 estimates are planning figures; the release uses measured times.
