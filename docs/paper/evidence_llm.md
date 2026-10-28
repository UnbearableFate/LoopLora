# LLM writer screens and registered P1-L claims (WP-T8)

Evidence cutoff: `master` at `f1dd6e2`, including the final WP-L4 release
(`91fac0c`, merge `39f7f0b`) and P1-L Amendment B5 (`6de902b`). This ledger
synthesizes committed validation screens and implementation/protocol records.
It contains the registered P1-L confirmation and P1L-C results (WP-T10 part 2: rows LP7–LP13 and PC1–PC4) and no DPFP or P1L-D
result in its original text; the DPFP result and the not-run status of P1L-D were added by part 3b (DP1–DP3, PD1). The corresponding
[symbolic ledger](evidence_symbolic.md) remains authoritative for symbolic claims.
WP-T9 (2026-10-27, on `dc659e0`) adds LP6 (pre-confirmation memory-use evidence and Amendments
B6/B7), scope statements, the OA rename (value/output ceiling) and further exclusions after two
external reviews. Sources newer than the cutoff above are Amendments B6 (`26b96bd`) and B7
(`2bf7506`), the run locks (PLK, merged `ca85a40`) and the capacity check (L6C). The ledger still
contains no P1-L confirmation result.
WP-T10 part 1 (2026-10-29, on `master` `abc2fa1`) adds the registered results of RFX (RX1–RX3) and the WP-L6 stop
(WL1, WL2), whose records were then on wp/llm and wp/oracle (now merged to master), and a section of pending
placeholders (DP1, PC1, PD1; P1-L confirmation and the B7 check stayed in LP4 and LP6; PC1 and LP4 are filled by part 2). It changes the status of no
earlier row and adds no P1-L confirmation result.
WP-T10 part 2 (2026-10-29, on `master` `5d9c121`) adds the P1-L confirmation with the Amendment-B7 memory-use check (LP7–LP13;
LP4 narrowed, LP6 and LP3 cross-referenced) and the P1L-C results (PC1–PC4), both merged to master, and repoints the RFX and
WP-L6 citations to their master paths. The per-step readout results, DPFP (DP1) and P1L-D (PD1) stayed open.
WP-T10 part 3a (2026-10-29, on `master` `88e8652`) enters the per-step readouts in LP5 as descriptive rows, from the corrected P1-L
record (`f4f9a98`, merged `5eca4d2`).
WP-T10 part 3b (2026-10-29, on `master` `711d1ac`) enters the DPFP negative result (DP1 not supported; DP2, DP3 descriptive) and marks
P1L-D as not run (PD1); the former placeholder section is now "DPFP and P1L-D: registered result". Nothing is pending in this ledger.

## Status, populations, and metrics

**Not run** (PD1) means the preregistered decision rule never registered the study.
**Supported** means supported under the row's conditions; **narrowed** requires
its stated restriction; **open** means the outcome is not available. Protocol
support establishes what was registered, not that its eventual gates passed.
All LL1–LL10 measurements use **training seed 17 and validation split `val`**,
evaluation seed 4242; **CIs are unavailable**. Repeated use of this validation set,
arm selection, and unequal budgets preclude treating the tables as independent
confirmation or uncertainty over training seeds. No new statistics were estimated.

| Label | Population | Metric/uncertainty |
| --- | --- | --- |
| V-L1 | 1000 T1 episodes, one answer per episode; E16 policy grid and selected E8/12/16/24 capacity runs | Full-vocabulary next-token EM; no CI. E16 is reused in the capacity scan. |
| V-L2 | 1000 T1 E16 episodes, one answer per episode; reused labels across D2/D3/D4 | Full-vocabulary EM; no CI. D1 uses 16000 fact examples or 1000 question examples from these episodes. |
| V-L3 | 1000 T1 E16 episodes, one answer per episode, shared across arms/checkpoints | Full-vocabulary EM; no CI. Pre-tail probes use 16000 fact examples. |
| V-L4 | 1000 T1 E16 episodes, one answer per episode; frozen WP-L3 baseline audit and seven new arms | Full-vocabulary EM and exported diagnostics; no CI. Four gradient-audit minibatches are separate training examples. |
| P1-L validation (planned protocol) | Split `val`, seed4242, 1000 episodes | Pilot-2 learning-rate selection, not confirmation. |
| V-L8 | P1-L confirmation: split `test`, episode indices `[100000,102000)`, 2,000 episodes, eval seed4242, memory-only primary setting, oracle_random (oracle seed17) at rank 32 and 128, training seeds17/29/43 | Full-vocabulary EM; 97.5% paired episode-bootstrap intervals (2,000 draws; k=2) over the three trained seeds, per-seed effects beside them. The interval is an index range, not an evaluation seed. |
| V-L10 | DPFP (WP-L5): split `val`, seed4242, 1,000 episodes per arm, T1 E=16 (E=8 for the one retrain), training seed17, arms D32-3, D128-1, D128-3 | Full-vocabulary EM and the WP-L4 diagnostics at the final checkpoint; one seed per arm; no CI (standard error about 1.5 pp); validation data only. |
| V-L9 | P1L-C: split `test`, episode indices `[102000,104000)`, 2,000 episodes (read position and replication); the first 256 hop-2 long-chain episodes per seed (bridge patching); the nine rank-128 checkpoints (M1-L4, B8, B4 × seeds17/29/43), nothing retrained | Full-vocabulary EM and answer rates; 95% paired episode-bootstrap intervals (2,000 draws), descriptive except the registered causal-bridge bar. |
| T-L1 | P1-L main-run training logs at step 17000, rank 32 and 128: mean of the last five logged batches (steps 16600–17000; batch size 64 episodes), hops1–3 mixed, trained depth | Training-batch accuracy and loss as logged; no CI; neither validation nor confirmation. Local files (`runs/` is git-ignored). |
| V-L5 | P1-L Pilot-2 at rank 32 (S/3 steps, training seed17, lr in {.001,.003}): split `val`, eval seed4242, 1000 episodes per (reader, lr, hop) cell, memory-only protocol, five readers (B3-L4 has no memory) | Full-vocabulary EM and episode-candidate accuracy against the 1/14 chance level; no CI; learning-rate selection data, not confirmation. Local files. |
| V-L6 | RFX, frozen WP-L4/WP-L3 checkpoints (R/r128, R/r32, W5/r32; training seed17): split `val`, seed4242, episode indices [200000,204000), one hop-1 query per episode (4,000 queries); fitting set 4,096 `train` episodes (16,384 queries) | Full-vocabulary EM, normalized read error and entity-restricted EM of cells that replace the final-position read; one seed; no CI; validation data only. |
| V-L7 | WP-L6 pilots, rank 32, M1-L4 with oracle code designs (seed17, 5,666 steps): split `val`, seed4242, 1,000 episodes, L=4, memory-only | Hop-1 full-vocabulary EM of the final checkpoint against a 30% threshold; one seed, one learning rate; no CI; pilot, not confirmation. |

The backbone is frozen `Qwen/Qwen3-0.6B-Base` at revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Screens use one layer14 adapter,
normally at attention `o_proj` input 2048/output 1024, fp32 adapter/state,
rho=.25, alpha=1, rank32 unless stated, ReLU M1 at L1, and memory-only access.
T1 is a direct single-hop fact task with four relations and head sharing 0.3;
it has **no T2 chain layout**. Training uses four queries per episode, but
reported validation n is 1000, not 4000. The stage name “writer” does not freeze
R/U/Q: the screen optimizer can train all adapter parameters requiring gradients.

Full-vocabulary EM, episode-candidate accuracy, source-fact key top1, and auxiliary
head accuracy are different quantities. Probe name classification is 1024-way over
the seen 768-name pool, not full-vocabulary LM prediction or unseen-name testing.
`%` is a rate; `pp` is a difference. Tables round source fractions/continuous
values to six decimals where useful; the raw files retain full precision. A
report rounded to 100.0% need not have exact 100% underlying accuracy.

## Source keys and provenance

Artifact commits below identify the introduction of the relevant rows or their
final update, not merely the subsequent merge or training source. Row selectors
are unique in the named file. Scientific execution occurred in the original
compute jobs; this synthesis reads only committed tables, records, and source.

| Key | Source | Artifact/source commit and selector convention |
| --- | --- | --- |
| L1E | [pilot runs](../../reports/p1l_pilot1_runs.csv), [summary](../../reports/p1l_pilot1_summary.json), [report](../reports/P1L-pilot1.md) | `aef2e28`; runs `(E,write_policy,lr)`, summary `capacity` and `diagnosis_required` |
| L1D | [locked plan](../../configs/p1l/pilot1/locked-plan.json), [policy selection](../../reports/p1l_pilot1_policy_selection.json) | `c98a233` / `add488e`; same commits are the policy/capacity run sources |
| L2E | [runs](../../reports/p1l-diagnosis/runs.csv), [probes](../../reports/p1l-diagnosis/probes.csv), [D1](../../reports/p1l-diagnosis/D1.json), [summary](../../reports/p1l-diagnosis/summary.json), [report](../reports/P1L-diagnosis.md) | `a87b17e`; runs `(diagnostic,input_site,lr,step)`; probes `(site,position,target,rank)` with blank rank=full |
| L2S | [diagnostic source](../../src/looplora/llm/diagnosis.py), [probe source](../../src/looplora/llm/diagnosis_probes.py) | Initial diagnostic implementation `9ffabc4`; D1 source `bcab714`; D2/D3 run source `99daecb`, D4 `3c1cffb` |
| L3E | [runs](../../reports/p1l-writer-fix/runs.csv), [probes](../../reports/p1l-writer-fix/probes.csv), [screen summary](../../reports/p1l-writer-fix/screen-summary.json), [selection](../../reports/p1l-writer-fix/selection.json) | Initial screen/probe rows `a4afc80`; `(arm,extension=False,step)` and probe `(site,position,target,rank)` |
| L3F | [final runs](../../reports/p1l-writer-fix/runs.csv), [final summary](../../reports/p1l-writer-fix/summary.json), [report](../reports/P1L-writer-fix.md) | Final A1 rows/report `cd5a45c`; `(arm,extension=True,step)` |
| L3D | [initial release](../../reports/p1l-writer-fix/release.json), [rerun release](../../reports/p1l-writer-fix/rerun-release.json), [interventions](../../src/looplora/llm/writer_fix.py) | Initial release `022109a`, science `3fbc504`; A1 amendment `5b0d847`, selection `a4afc80`, schedule `9243a95`, rerun science `0713cfb` |
| L4E | [runs](../../reports/p1l-read-fix/runs.csv), [selection](../../reports/p1l-read-fix/selection.json), [summary](../../reports/p1l-read-fix/summary.json), [report](../reports/P1L-read-fix.md) | `91fac0c`; runs `(rank,arm,E,step)`, selection `choice` and `context` |
| L4A | [baseline audit](../../reports/p1l-read-fix/audit.csv), [age strata](../../reports/p1l-read-fix/audit-age.csv), [audit summary](../../reports/p1l-read-fix/audit-summary.json) | `26a8020`; audit `arm=OA/W5`; age strata `(arm,age)` |
| L4D | [rank32 release](../../reports/p1l-read-fix/release.json), [rank128 release](../../reports/p1l-read-fix/release-r128.json), [read-fix source](../../src/looplora/llm/read_fix.py) | Rank32 release `f56cb68`, final schedule `26a8020`; rank128 release `2fd95f9`; implementation `ffea4b8`; run sources `0b03d60` / `be07f7f` |
| C6 | [B8 report](../reports/WP-C6.md), [benchmark](../../reports/llm-b8/benchmark.json) | `23c5711`; tested implementation `214518c` |
| C7 | [oracle report](../reports/WP-C7.md), [benchmark](../../reports/llm-oracle/benchmark.json), [oracle source](../../src/looplora/llm/oracle.py), [configuration guard](../../src/looplora/llm/config.py) | Report/benchmark `b0bf9a3`; implementation/tests `b54ec3a` |
| PL | [P1-L preregistration](../prereg/P1L.md) | v2 final/readouts `f6dcb4e`; B1 `ae1ae22`, B2 `7bfaf16`, B3 `df21654`, B4 `379115d`, B5 `6de902b`, B6 `26b96bd`, B7 `2bf7506`; timestamp corrections `0dbea09`/`6b7f292` |
| P1LE | P1-L confirmation: `reports/p1l/endpoints.json`, `reports/p1l/r128/endpoints.json`, the `memory-use.csv`, `accuracy.csv` and `inference.csv` of both ranks, `reports/p1l/rank-comparison.json`, `reports/p1l/jobs.json`, `reports/p1l/readout-pooled.csv`, `reports/p1l/per-step-readout.csv`, `reports/p1l/record-numbers.txt` (from `scripts/p1l_record_numbers.py`), [report](../reports/P1L.md), [decision record](../../reports/decisions/P1L.md) | Results `395e309`, merged to master as `16e4fca`; corrected descriptive record `f4f9a98`, merged `5eca4d2`; run at `P1L_SOURCE` `419276578c16f8d191a7ccb4d2ca67ea07fdec00`, authorization `reports/p1l/confirmation-authorization.json`. Selectors: `contrasts.<name>`, `memory_use.contrasts.<name>`, `G1_L`, `G2_L`, `memory_use_passed`; CSV rows by (rank, reader or contrast, seed) |
| DPFE | DPFP: `reports/p1l-dpfp/` (`arms/`, `arms-e8/`, `refs/`, `e8-lock.json`, `release.json`, `release-e8.json`, `audit-e16.json`, `audit-e8.json`), [report](../reports/DPFP.md), [preregistration](../prereg/DPFP.md) | Branch wp/llm, merged to master as `711d1ac`; E=16 runs at `efadc36` (jobs 3721087–3721089), E=8 retrain at `72ed469` (job 3721535), final records and report `2b75fcf`, audit report `9a20135`. Selectors: arm records (`actual_steps`, validation EM, diagnostics), `e8-lock.json` `choice`, audit `runs.<arm>` |
| P1LC | P1L-C: `reports/p1lc/results.json`, `read-position.csv`, `replication.csv`, `bridge-shift.csv`, `bridge-rates.csv`, [report](../reports/P1L-C.md), [preregistration](../prereg/P1L-C.md) | `2d9bb1d`, merged as `28e122f`; source `P1LC_SOURCE` `395e30914149898006d63cd289b20dac15d732fe`. Selectors: `(reader, hops, seed=pooled)`, `(reader, patch, seed)`, `(contrast, confidence, seed)` |
| PLK | [r32 run lock](../../reports/p1l/lr-S-lock.json), [r128 run lock](../../reports/p1l/r128/lr-S-lock.json) | r32 `d644b9b`; r128 `1d1bbfd`; merged with `ca85a40`, after this ledger's original cutoff; fields `S`, `lrs`, `selection` |
| L6T | Main-run training logs `runs/p1l/main/main/*/*/train_log.jsonl` and `runs/p1l/main/r128/main/*/*/train_log.jsonl`; Pilot-2 `runs/p1l/main/pilot/*/*/validation/metrics.json` | **Not committed** (`runs/` is git-ignored). The committed record is the summary in PL B7 (`2bf7506`); WP-T9 re-read the raw files read-only |
| L6C | [capacity check](../../reports/p1l-c1/capacity.json), [script](../../scripts/analyze_oracle_capacity.py) | `2bf7506`; rows `(r,seed,keys,map)`, means under `mean` |
| RFXS | `reports/p1l-rfx/summary.json`, `R-r128.json`, `R-r32.json`, `W5-r32.json` and `docs/reports/RFX.md`; [preregistration](../prereg/RFX.md) | Merged to master (`d4de948`, from wp/llm): summary and records introduced by `379fccd`, report by `faacd37`, both unchanged since; run at `efadc36`, job 3721047.opbs. Selectors: `verdicts`, `rescue`, `lsq_arms`; per-checkpoint `cells`, `refit`, `conditioning`, `write_order`, `checks` |
| WL6D | `reports/wp-l6/decision.json`, `sizing.json` and `docs/reports/WP-L6.md`; [preregistration](../prereg/WP-L6.md) | Merged to master (`ddfcbc2`, from wp/oracle): introduced by `98d048c`, unchanged since; run at `7175e09`, jobs 3720811 and 3720812. Selectors: `status`, `selected`, `arms.<arm>.hop1_L4_full_em`, `threshold` |

## Writer-screen claim ledger

Each numerical row uses the indicated V-L population and **no CI**. Detailed
tables below preserve the selected endpoints and intervention definitions.

| Claim | Status | Evidence | Conditions and non-implications | Source/selector |
| --- | --- | --- | --- | --- |
| LL1. “The initial screen found no admissible learned T1 writer at the registered budget.” | supported | V-L1: all nine endpoints≤0.300000% EM; selected fact_final/lr.0003 gives E8/12/16/24 EM **0/0/.3/0%**. `E_star=null`, diagnosis required. | r32,1500steps,B32,seed17; policy ties favor fact_final then lower lr. The failure is under tested thresholds/budgets, not an asymptotic capacity bound or impossibility. CI unavailable. | L1E `aef2e28`, nine `(E,write_policy,lr)` cells; L1D selection |
| LL2. “Information relevant to writing and querying is recoverable at the adapter's input positions.” | narrowed | V-L2 rank32 probes at o14: tail-token tail **97.618750%**; question-final start **99.400000%**, relation **100.000000%**; fact-final head/tail **80.862500/84.150000%**. V-L3 pre-tail head/relation **95.612500/99.987500%**. | Separate target-specific probes, held-out examples but seen name identities; not one shared A jointly transmitting all information, nor proof that answer-only optimization can learn alignment. CI unavailable. | L2E `a87b17e`, probes `(o14,position,target,32)`; L3E `a4afc80`, `(o14,pre_tail,head/relation,32)` |
| LL3. “The frozen LM output path is usable when an answer-identity code is supplied.” | supported | V-L2 D2 lr.003/1500: **98.700000% EM**, **99.300000% candidate**, NLL**.101494**; B0 EM**0%**. | D2 supplies answer identity to a learned1024×32 table and trains that table/B; it bypasses the reader and rho, alpha1. Not learned retrieval, natural memory access, or the OA/fact-list oracle. CI unavailable. | L2E `a87b17e`, `(D2,o_proj,.003,1500/B0)`; L2S |
| LL4. “The tested longer unassisted runs and residual-input alternative did not rescue retrieval; key discrimination remained near chance.” | narrowed | V-L2 final6400step D3/D4 EM **0/.1/.1/0%**; source-fact top1 **7.1/7.5/6/6%**, versus6.25% chance, all mean target margins negative. | Two learning rates at each input site, seed17; D4 moves A input only, output remains o_proj14. Near-chance is descriptive without a significance test; does not prove no input information, sufficient budget, or a uniquely identified cause. CI unavailable. | L2E `a87b17e`, D3/D4 input/lr/6400 cells |
| LL5. “The writer-fix screen improved some diagnostics without selecting a feasible writer.” | supported | V-L3 at4800steps: OA **22.1% EM**, W0–W5 **0/0/.5/.6/.3/7.8%**. W2 key top1 **78.9%** despite.5% EM; W4 **.3% EM**, W5 **7.8%**. | W5=standardization+address+value auxiliaries, **not W4 factorization**; W4 has extra parameters and contextual pre-tail/tail features. OA, the value/output ceiling, is a privileged true-value bypass and ineligible. CI unavailable. | L3E `a4afc80`, `(arm,False,4800)`; L3D definitions |
| LL6. “The final longer OA/W5 runs still selected no learned writer.” | supported | V-L3 final32900: W5 **14.4% EM**, candidate**39.3%**, key top1**82.4%**; OA **70.3% EM**, candidate**95.7%**. W5<.80 selection and.30 partial-learning threshold; OA ineligible. | Fresh reruns from the same initialization/data stream, not resumed37700step runs. OA/W5 budgets exceed W0–W4; final checkpoints, not best-validation selection. OA (value/output ceiling) bypasses the whole memory read, not only address ranking. CI unavailable. | L3F `cd5a45c`, `(W5/OA,True,32900)`, summary selected_arm=null |
| LL7. “The later baseline audit finds a mismatch between clean values and deployed reads in W5.” | narrowed | V-L4 audit of L3 W5: read error**26.635210**, mean norms q/value/read **19.011824/30.749941/159.350082**; clean/read-head accuracy**85.4/10.5%**, Gram RMS**.330676**, mean max abs cosine**.854223**. | Actual numerical audit supports read distortion under these weights; separate scale/Gram/decoder effects are not uniquely isolated. Clean-head transfer is not a bound on information in the read. OA coefficients describe its bypassed store. CI unavailable. | L4A `26a8020`, `arm=W5/OA`, tables below |
| LL8. “Rank32 read supervision and gain/decorrelation variants did not reach the writer gate.” | supported | V-L4 finals: R/Rrho/RK EM**12.0/12.2/12.0%** at unequal timing-derived budgets; diagnostic value/output-ceiling arm OArho**74.7%**. Matched6580step OA/OArho EM**33.7/43.2%**. | R supervises raw read with stopped target/denominator; Rrho changes rho to.5, RK adds key decorrelation. Matched-step gain comparison is descriptive; other final contrasts do not isolate a factor at equal steps. CI unavailable. | L4E `91fac0c`, rank32 final/6580 cells; L3F `cd5a45c`, OA True/6580 |
| LL9. “The final rank128 arms still selected no learned writer.” | supported | V-L4 rank128 final W5/R/OA EM**29.9/33.5/84.0%**; R clean/read-head**93.3/19.7%**. Eligible best R<.80 gate and.50 E8 trigger: no E8 retraining, no selected writer. | Single seed, unequal budgets, unchanged lr.003 without rank128 LR search. OA (value/output ceiling) is ineligible. B5 cancels learned P1-L/P3, **not** oracle-r32/r128 P1-L. No universal learned-writing impossibility. CI unavailable. | L4E `91fac0c`, rank128 finals, selection.choice/action=stop; PL B5 `6de902b` |
| LL10. “The baseline models depend on their deployed memory channel, but zero/shuffle interventions do not localize the failure.” | narrowed | V-L4 audit: W5 original/zero/shuffle EM **14.4/.1/.2%**, NLL**4.851514/8.525247/9.720540**; OA **70.3/0/0%**, NLL**1.074996/13.702414/20.339018**. | Questions/labels held fixed. OA intervention must alter the read it actually receives, not merely an unused matrix. These tests establish sensitivity, not correct addressing, pure causal composition, or the fraction of error caused by interference. CI unavailable. | L4A `26a8020`, `arm`, original/zero/shuffle fields |

## LL1–LL4: initial feasibility and input/output diagnostics

LL1 endpoints, L1E `aef2e28`; all 1500 steps, n=1000, V-L1, CI unavailable.
Training CE is not the later validation NLL. Selection requires 95% EM at
E16/24 or E12, or flagged 80% at E8; every criterion failed.

| E | Write policy | lr | EM% | Candidate% | Final training CE |
| --- | --- | --- | --- | --- | --- |
| 16 | all_tokens | .0003 | .300000 | 3.600000 | 7.435145 |
| 16 | all_tokens | .001 | .000000 | 3.500000 | 7.363760 |
| 16 | all_tokens | .003 | .000000 | 3.300000 | 7.285064 |
| 16 | fact_final | .0003 | .300000 | 3.900000 | 7.464932 |
| 16 | fact_final | .001 | .000000 | 3.200000 | 7.367683 |
| 16 | fact_final | .003 | .100000 | 3.000000 | 7.301948 |
| 8 | fact_final | .0003 | .000000 | 9.000000 | 7.403627 |
| 12 | fact_final | .0003 | .000000 | 5.700000 | 7.416298 |
| 24 | fact_final | .0003 | .000000 | 2.600000 | 7.510422 |

LL2 selected probe rows, L2E `a87b17e`, except pre_tail rows L3E `a4afc80`.
All site=o14, separate full/rank32 fits; CI unavailable. Full/rank32 values
need not be ordered under finite fitting. Probes use4096 train episodes,
1200steps; 1024-way names use seen IDs0…767, relations4-way. The full set of96
D1 probes remains in the linked CSV. Fact-final is the final fact token,
**not** necessarily its tail-name token; pre_tail is immediately before the tail.

| Position | Target | Validation n | Full / rank32 accuracy% |
| --- | --- | --- | --- |
| fact_final | head | 16000 | 59.781250 / 80.862500 |
| fact_final | tail | 16000 | 56.768750 / 84.150000 |
| fact_final | relation | 16000 | 99.975000 / 99.987500 |
| tail_token | tail | 16000 | 89.412500 / 97.618750 |
| question_final | start | 1000 | 99.300000 / 99.400000 |
| question_final | relation | 1000 | 100.000000 / 100.000000 |
| question_start | relation | 1000 | 24.500000 / 25.500000 |
| pre_tail | head | 16000 | 83.975000 / 95.612500 |
| pre_tail | relation | 16000 | 99.993750 / 99.987500 |

The question-start relation control precedes the relation text and is near25%
chance; it does not imply absence of relation information at question-final.

LL3/LL4 validation rows, L2E `a87b17e`; V-L2, n1000, CI unavailable.
The three identical D2 B0 records are displayed once, not independent baselines.
“Key1” ranks the true source fact among16 normalized keys (chance6.25%),
not the episode candidate answers; negative margin is a mean, not every query.

| Diagnostic/input | lr | Step | EM% | Candidate% | Validation NLL | Key1% | Mean target margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D2/o_proj | .0003 | 1500 | 20.000000 | 33.600000 | 6.058221 | — | — |
| D2/o_proj | .001 | 1500 | 85.300000 | 92.900000 | .983372 | — | — |
| D2/o_proj | .003 | 1500 | 98.700000 | 99.300000 | .101494 | — | — |
| D2 frozen B0 | repeated | B0 | .000000 | .000000 | 10.684949 | — | — |
| D3/o_proj | .0003 | 1500 | .300000 | 3.900000 | 7.500602 | 6.100000 | −.039144 |
| D3/o_proj | .0003 | 3200 | .100000 | 3.300000 | 7.385636 | 6.000000 | −.031534 |
| D3/o_proj | .0003 | 6400 | .000000 | 4.000000 | 7.244281 | 7.100000 | −.045950 |
| D3/o_proj | .001 | 1500 | .000000 | 3.200000 | 7.366396 | 8.700000 | −.263238 |
| D3/o_proj | .001 | 3200 | .000000 | 2.900000 | 7.306273 | 5.700000 | −.131238 |
| D3/o_proj | .001 | 6400 | .100000 | 4.600000 | 7.208796 | 7.500000 | −.143058 |
| D4/residual | .0003 | 1500 | .000000 | 4.300000 | 7.283866 | 5.700000 | −.068699 |
| D4/residual | .0003 | 3200 | .000000 | 3.200000 | 7.233919 | 7.100000 | −.123545 |
| D4/residual | .0003 | 6400 | .100000 | 4.200000 | 7.141348 | 6.000000 | −.348351 |
| D4/residual | .001 | 1500 | .200000 | 3.800000 | 7.266557 | 6.800000 | −.027340 |
| D4/residual | .001 | 3200 | .000000 | 3.300000 | 7.246611 | 6.400000 | −.018117 |
| D4/residual | .001 | 6400 | .000000 | 4.000000 | 7.212559 | 6.000000 | −.104355 |

Each6400step run receives204800 training episodes, versus48000 in Pilot-1.
Neither the longer budget nor D4's alternative A input establishes an asymptotic
optimum. D1/D2 success plus D3/D4 failure does not uniquely identify raw-read
distortion; those measurements arrive in LL7–LL9.

## LL5–LL6: writer-fix arms and final reruns

All arms use lr.003, T1 E16, B32/Q4, seed17, V-L3. Auxiliary supervision is
additional to answer loss. Arm semantics come from L3D `3fbc504`:

| Arm | Intervention |
| --- | --- |
| W0 | Fact-final writer, common lr.003 control |
| W1 | W0 plus pooled train-only feature standardization |
| W2 | W0 plus weight1 normalized key/query address CE, temperature.1 |
| W3 | W0 plus weight1 clean-value tail-name CE through a training-only affine1024-name decoder |
| W4 | Separate A_k on pre-tail and A_v on contextual tail token; reader shares A_k |
| W5 | W1+W2+W3; **does not include W4** |
| OA | Value/output ceiling: supply the true fact's clean learned value instead of Mq at every valid question token; retain u/R/U/ReLU/B and downstream LM |

W3/W5 add33792 training-only head parameters, discarded for inference. W4
has168962 registered adapter parameters versus103426 for the other r32 arms.
Its contextual value can retain head/relation information. OA bypasses lookup,
attenuation, and interference, so it is a value/output ceiling and not an addressing-only
causal contrast.

LL5 initial4800step rows, L3E `a4afc80`, `extension=False`; n1000, no CI:

| Arm | EM% | Candidate% | Key1% | Validation NLL |
| --- | --- | --- | --- | --- |
| W0 | .000000 | 3.600000 | 6.900000 | 7.232457 |
| W1 | .000000 | 3.800000 | 6.600000 | 7.238392 |
| W2 | .500000 | 8.800000 | 78.900000 | 6.872767 |
| W3 | .600000 | 5.500000 | 6.100000 | 6.944125 |
| W4 | .300000 | 8.300000 | 26.400000 | 6.792845 |
| W5 | 7.800000 | 31.100000 | 75.900000 | 5.536867 |
| OA | 22.100000 | 74.400000 | 6.500000 (bypassed) | 3.296924 |

Amendment A1 was locked before reading validation EM, after training-loss trends.
Its top-two ranking included OA and selected OA/W5, **not three arms**.
The reruns start fresh with the same initialization and absolute data stream;
they are not a continuation from4800. LL6 final32900step rows, L3F `cd5a45c`,
`extension=True`, n1000, no CI:

| Arm | EM% | Candidate% | Key1% | Full prediction is an episode tail% | Validation NLL |
| --- | --- | --- | --- | --- | --- |
| W5 | 14.400000 | 39.300000 | 82.400000 | 23.400000 | 4.851514 |
| OA | 70.300000 | 95.700000 | 5.800000 (bypassed) | 70.900000 | 1.074996 |

Final W5 mean target margin=.176787 and signed off-diagonal key-pair cosine
mean=.099428 do not supply a Gram RMS or raw-read certificate. OA's best
saved EM71.0% at26320 is not its registered final70.3%. Candidate-minus-full
accuracy is not the exact fraction of predictions outside the candidate set;
episode-tail membership is another different statistic. L3 itself exports no
raw-read error/SIR/norm/head-transfer audit; those numbers require L4A.

## LL7–LL10: deployed-read diagnostics and the final writer decision

The baseline audit reuses the final L3 OA/W5 checkpoints without optimization.
V-L4, n1000, CI unavailable; original predictions reproduce exactly with zero
repeat-logit drift. The norms and coefficients are means, not equalities among
means. Full source selectors: L4A `26a8020`, `arm=OA/W5`.

| Arm | q norm | Target-value norm | Deployed-read norm | u norm | Ru norm | Uread norm | gM norm | BgM norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OA | 37.686028 | 27.765163 | 27.765163 | 42.343483 | 189.097000 | 124.632355 | 21.916075 | 245.235657 |
| W5 | 19.011824 | 30.749941 | 159.350082 | 42.333004 | 233.929718 | 211.721207 | 45.564556 | 515.621704 |

| Arm | b_t | a_t | d_t | Value SIR | Parallel interference | Key Gram RMS | Mean max abs cosine | Clean/read-head accuracy% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OA | 1.020543 | .771157 | .074485 | .045549 | 12.209503 | .354888 | .808862 | unavailable |
| W5 | 15.085691 | 12.751815 | 2.641140 | .537923 | .108563 | .330676 | .854223 | 85.400000 / 10.500000 |

OA's coefficients/SIR refer to its **bypassed stored matrix**; its deployed
read is the true value and has zero read error by construction. W5's deployed
normalized squared read error is 26.635210. The clean-value decoder applied to
the actual read is a transfer diagnostic, not an optimal read decoder or an
information-theoretic bound. The decomposition and final-run tables below
retain the corresponding source definitions and selectors.

LL10 interventions (L4A, V-L4, no CI):

| Arm | Original EM% / NLL | Zero EM% / NLL | Shuffle EM% / NLL |
| --- | --- | --- | --- |
| OA | 70.300000 / 1.074996 | .000000 / 13.702414 | .000000 / 20.339018 |
| W5 | 14.400000 / 4.851514 | .100000 / 8.525247 | .200000 / 9.720540 |

For these scalar-gate writes, the audited decomposition uses
`b_i=k_i^T q`, `(I+beta*triu(G,1))a=b`, and
`d_i=beta*lambda^(E−1−i)*a_i`, with `G_ij=k_i^T k_j`. Reconstruction is
`sum_i d_i v_i`; target signal is `d_t v_t`, interference is reconstruction
minus signal. Value SIR is squared signal norm divided by squared interference
norm (denominator floor 1e−12); parallel interference is their inner product
divided by squared signal norm (same floor). They are averaged per query,
not ratios of mean energies and not cosines. Coefficients can be signed and
are not normalized attention weights. OA matrix diagnostics must not be
combined with its deployed true-value norm as if they described one read.
Here `gM=rho*(phi(Ru+Uread)−phi(Ru))` and `BgM=alpha*B*gM` are low-rank
state/output-feature effects, not LM logits.

Maximum relative Gram reconstruction errors are 3.739737e−7 for OA and
1.407663e−7 for W5; independently reconstructed coefficients and exactly
repeated predictions are consistency checks, not independent replications
(L4A `26a8020`, audit-summary `runs.OA/W5`). Four training minibatches per arm,
128 episodes and zero optimizer updates, all have would-clip flags. W5
answer/address gradient cosines are .241538/.089387/−.048157/.243380;
answer/value cosines .480550/.348947/.395876/.381898. These snapshots do not
establish persistent gradient conflict throughout training. Age-conditioned
tables exist only for these two baselines (`arm,age=0…15`); mean target_age in
later runs is not an age-conditioned treatment effect, and no monotone age law
is inferred.

R adds weight 1 raw-read regression
`||Mq−stopgrad(v_t)||²/(||stopgrad(v_t)||²+1e−6)` to W5; the entire target
and denominator are detached. Rrho uses rho=0.5, RK adds weight 1 squared key
cosine over distinct addresses, and OArho changes the diagnostic OA gain
to rho=0.5. The reported key Gram RMS instead uses all off-diagonal ordered
pairs, so it is not identical to RK's masked penalty. OA is a separately
trained arm, not W5's weights with an oracle switch (L4D).

LL8/LL9 final rows, L4E `91fac0c`, selector `(rank,arm,E=16,step)` as shown;
V-L4,n1000,seed17,CI unavailable. Budgets are quality-blind, timing-sized and
unequal. Rank128 inherits lr.003 without a new LR search. Head columns are
the training-only1024-name auxiliary classifier, not the LM or candidate metric.

| r | Arm | Final step | EM% | Candidate% | NLL | Raw-read error | Key1% | Clean/read-head accuracy% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | R | 23590 | 12.000000 | 36.100000 | 5.045488 | .665751 | 83.200000 | 84.500000 / 8.800000 |
| 32 | Rrho | 24220 | 12.200000 | 37.900000 | 4.996533 | .663259 | 84.700000 | 84.200000 / 8.800000 |
| 32 | RK | 21980 | 12.000000 | 36.400000 | 4.993353 | .668646 | 82.300000 | 84.900000 / 7.700000 |
| 32 | OArho | 27040 | 74.700000 | 97.300000 | .922419 | .000000 | 6.400000 (bypassed) | unavailable |
| 128 | W5 | 24635 | 29.900000 | 52.900000 | 3.648279 | .894427 | 88.300000 | 94.200000 / 20.900000 |
| 128 | R | 24645 | 33.500000 | 54.100000 | 3.531441 | .630860 | 88.500000 | 93.300000 / 19.700000 |
| 128 | OA | 31175 | 84.000000 | 98.400000 | .578033 | .000000 | 5.700000 (bypassed) | unavailable |

The predeclared matched-step diagnostic, excluded from selection, is
OA/r32/L3 at6580 (L3F `cd5a45c`, extension=True) versus OArho/r32/L4 at6580
(L4E): EM33.700000/43.200000%, candidate84.600000/89.000000%, and
NLL2.614228/2.241182. This is not a matched comparison of their final32900
and27040step checkpoints, and it supplies no CI or multi-seed gain guarantee.

LL8/LL9 final norm diagnostics, same L4E selectors/population/no-CI scope:

| Arm,r,step | q norm | Target-value norm | Deployed-read norm | u norm | Ru norm | Uread norm | gM norm | BgM norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R,32,23590 | 16.531506 | 31.152311 | 18.564251 | 30.789068 | 126.755051 | 88.528252 | 21.285852 | 253.150391 |
| Rrho,32,24220 | 18.075056 | 31.822556 | 18.285238 | 32.921593 | 88.487816 | 62.204620 | 28.218746 | 323.163452 |
| RK,32,21980 | 18.363707 | 30.991169 | 18.250257 | 31.384363 | 130.225769 | 86.218689 | 20.932180 | 247.473465 |
| W5,128,24635 | 139.118088 | 92.720253 | 90.429550 | 58.096920 | 529.309265 | 395.791412 | 76.530273 | 973.912598 |
| R,128,24645 | 131.647263 | 91.404327 | 51.956760 | 48.648678 | 272.743195 | 290.085663 | 60.250389 | 821.119690 |
| OArho,32,27040 | 43.141518 | 20.230858 | 20.230858 | 43.529457 | 140.267059 | 74.484764 | 24.301903 | 261.734192 |
| OA,128,31175 | 67.937157 | 102.398796 | 102.398796 | 70.041466 | 862.187866 | 930.155029 | 123.279877 | 1752.950684 |

| Arm,r,step | b_t | a_t | d_t | Value SIR | Parallel interference | Key Gram RMS | Mean max abs key cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R,32,23590 | 13.743399 | 13.503869 | .284258 | .484481 | .321554 | .331674 | .871360 |
| Rrho,32,24220 | 15.049532 | 14.818033 | .275042 | .500772 | .324728 | .329948 | .870321 |
| RK,32,21980 | 14.451152 | 14.234972 | .288957 | .517406 | .277825 | .302003 | .805485 |
| W5,128,24635 | 101.892394 | 101.557836 | .487682 | .501636 | .219292 | .288205 | .764257 |
| R,128,24645 | 93.794394 | 93.609114 | .287829 | .549290 | .282737 | .287691 | .767005 |
| OArho,32,27040 | −2.475875 | −2.261756 | −.207163 | .048357 | 6.267669 | .324940 | .777356 |
| OA,128,31175 | −.078120 | −.045492 | −.008577 | .067042 | .152772 | .271303 | .688179 |

Final R/r128 max-cosine mean is .767005, **not** the historical W5/r32 baseline's
.854223. The seven final artifacts reproduce their saved predictions with the
training head/optimizer removed (`summary.export_checks`, n1000 each); this
is export consistency, not seven new validation samples. L4 registered adapter
counts are 103426 at r32 and 475138 at r128; learned auxiliary heads add 33792
and 132096 training-only parameters, respectively, while OA/OArho have none.

Selection uses the final checkpoints: earlier R/r32 EM 12.9% at 18872 or
W5/r128 EM 30.4% at 19708 do not replace the table. The eligible counts are
120/122/120/299/335 out of 1000; none reaches 800, and best 335 also falls below
the 500-count E8 retry trigger. L4E `selection.choice` records stop with no
arm/rank/E, and no E8 retraining. This decision excludes OA/OArho and applies
to learned writing; PL B5 separately retains both random-oracle P1-L ranks.

## Registered P1-L protocol and pending outcomes

| Claim | Status | Registered evidence/status | Conditions and non-implications | Source |
| --- | --- | --- | --- | --- |
| LP1. “P1-L tests a single adapter in a pinned frozen LM with memory-only full-vocabulary scoring.” | supported | Qwen3-0.6B-Base revision above; bf16/SDPA; layer14 o_proj2048→1024; fp32 adapter/state, rho.25, alpha1. T2 E12/chains(8,4), four relations, head sharing.3; train hops1–3, evaluate1–8. Single-token name EM is full-vocabulary greedy answer-position accuracy; candidate score secondary. | Protocol statement only; all gates memory-only; B0–B2 both-protocol reports are descriptive. Actual fact-list oracle sends no support text through the LM. Not natural-language benchmark generalization. No result/CI implied. | PL v2 `f6dcb4e`, amendments; C7; static [scoring](../../src/looplora/llm/evaluate.py) |
| LP2. “B8 and the fact-only random oracle have implementation and acceptance-smoke support at the documented rank32 configuration.” | supported | C6 supports fixedL4, four R/U/Q sets, reload/frozen-writer compatibility. C7 fixed random unit pair-address keys, symbolic entity-value codes, lambda=beta=1, ordered fact writes, fixed oracle_seed17; code tables fp32 buffers. Question A trains; unused learned K/V/gates freeze. | Implementation support only; C6/C7 tests and train-only smokes are not quality evidence. At this cutoff `config.py` explicitly restricts oracle_random to r32; the later r128 protocol does not prove all required r128 execution changes are merged here. Random codes are not exact orthogonal/lossless storage. B8 is not parameter-matched to B4 (+9,216 adapter parameters, +8.9% at rank 32; FR1). | C6 `23c5711`; C7 `b0bf9a3`/`b54ec3a`, configuration oracle guard |
| LP3. “The final registered P1-L has two same-rank oracle conditions and 97.5% gate intervals.” | supported | B5: oracle_random r32/r128 only, k2; no learned writer/P3. G1-L compares B8@4−B4@1 on hops2/3: point≥3pp, CI lower>0, every-seed positive; hop1 point≥−1pp; same-checkpoint typed replication CI lower>0. G2-L requires positive CIs versus B5-L4 and B3-L4. Amendment B6: an equal-parameter claim only from M1-L4@4−B4@1 meeting the G1-L criterion. Amendment B7: a per-rank memory-use check (hop-1 EM of B4@1, B8@4, M1-L4@4 each above B3-L4@4, 97.5% CI>0) precedes interpretation. | Registered design/decision only; 2000 paired episode draws over fixed seeds17/29/43; not simultaneous coverage of every comparison. Primary B8 is same-rank, not parameter-matched: 103426→112642 (+8.9%) at rank32 and 475138→622594 (+31.0%) at rank128 (FR1; Amendment B6); shared M1-L4/B4 are parameter-matched. The outcomes are in LP7–LP13. | PL B3 `df21654`, B4 `379115d`, B5 `6de902b`, B6 `26b96bd`, B7 `2bf7506`; L4E selection |
| LP4. “At rank128, with a clean oracle memory, one shared module iterated four times composes stored associations inside the LM and exceeds a single read of equal parameter count on hops2–3; the gated multi-read reader B8 does not pass G1-L; no claim about multiple reads is made at rank32.” | narrowed | Confirmation (V-L8, 97.5% intervals): rank128 passes the memory-use check (LP7); M1-L4@4−B4@1 **+21.550000 [20.383229, 22.758333]** pp, per seed 24.450/15.100/25.100, with **475,138** parameters each (LP11); B8@4−B4@1 **+0.983333 [0.266563, 1.716667]**, below the 3 pp bar, so G1-L fails, while G2-L passes (LP9, LP10); rank32 fails the check, and its gates are reported without interpretation (LP7, LP8). The readouts are descriptive (LP5) and not evidence for it. | Narrowed to the `oracle_random` memory, the 768 seen names and the trained hops (Amendment B7 rule 3): the result does not establish equivalence with the symbolic oracle, composition for unseen names or entity codes, a learned writer or end-to-end use, and it is not a claim about B8 or about unshared reads in general. The equal-parameter claim is only M1-L4 against B4 at rank128 (Amendment B6); B8 adds +31.0% parameters. Latency is a measured cost (LP13). P1L-C gives no causal-bridge claim (PC1), and the descriptive readouts (LP5) are not evidence for this claim. Neither symbolic B9 reader selection, C6/C7 smokes, D2 answer codes, nor OA validation accuracy supports the claim. | LP7–LP13; P1LE; [PL](../prereg/P1L.md) B5, B6, B7; [decision record](../../reports/decisions/P1L.md) |
| LP5. (descriptive) “Per-step readouts of the rank-128 readers: decoding the state z_s (s=1…4) through the trained output path and the frozen upper layers (logit lens) recovers earlier-hop entities from M1-L4's early states, best at a step s ≥ e rather than at s = e, and recovers names from B8 only at s=4; the name-gradient projections (contrastive, raw logit) and the random direction are at or near chance for both readers; at rank32 every method is at chance.” | descriptive | Registered readout (PL v2/readouts): M1-L4 and B8, z_s at the answer position, s1…4; top-1 rate of the hop-e entity (entity step e) by reader step s, with name-gradient directions at the layer14 residual (512 validation questions, 1024 names) against a random direction. V-L8 (confirmation split, 2,000 episodes per seed, three seeds pooled; chance about 0.1%, mean rank about 512 of 1,024 names). **Rank 128, logit lens, M1-L4** (top-1 %, s=1…4; pooled): hop-1 questions, e=1 **38.62/64.28/61.13/47.93** (mean rank 59.5/22.6/26.3/42.5; largest single-seed cell **73.25** at s=2); hop-2 questions, e=1 **37.57/54.70/37.37/22.47**, e=2 **1.10/10.97/30.15/35.35**; hop-3 questions, e=1 **39.20/55.62/37.32/20.13**, e=2 **1.05/10.75/24.67/22.72**, e=3 **0.97/1.42/7.47/18.48**. The s=4 cells for e equal to the question hop (**47.93, 35.35, 18.48**) are within 0.1 pp of the deployed EM (47.85, 35.35, 18.47; LP12), and the s=2 and s=3 cells for hop-1 questions (**64.28, 61.13**) are close to the EM of the same checkpoint at depth L=2 and L=3 (**64.15, 61.28**). **B8, logit lens:** names are decoded only at s=4: e=1 **19.08/19.33/18.38** (hops 1–3), e=2 **6.57/6.90**, e=3 **6.05**; at s=1…3 every cell over hops 1–8 is at most **0.23**. **Contrastive, raw logit, random direction** (both readers, every cell): maximum pooled cell **0.27/0.28** (contrastive, M1-L4/B8), **0.27/0.08** (raw logit), **0.10/0.22** (random); mean ranks 476–509 (contrastive, raw logit), 500–526 (random). **Rank 32, all methods:** maximum pooled cell **0.27/0.20** (logit lens, M1-L4/B8), **0.22/0.22**, **0.17/0.25**, **0.02/0.08** (contrastive, raw logit, random); logit-lens mean ranks 382–431. | No gate. Decoding only: the readouts are not an ordered entity trajectory, not path following and not a causal claim (P1L-C gives no causal bridge, PC1), and they are not evidence for LP4. The corrected P1-L record reads the logit-lens rates as tracking the depth-s accuracy of the same reader, i.e. what the reader would answer if it stopped at step s; this is that record's interpretation of the matches above, not a measured stop. The name-gradient probe is J-lens-inspired, not the published averaged residual-transport J-lens and not JS's transition Jacobian; its chance result limits what these directions decode and does not show that names are absent from z_s. Upper-layer forwards incur diagnostic work distinct from deployed loop cost. The rank32 readers are at the memory-free level (LP7), so their chance readouts add nothing. B8 is not parameter-matched (FR1). The earlier committed record stated top-1 at most 0.3% at every step for all three decoders, which the committed CSV contradicted for the logit lens; the corrected record (`f4f9a98`) resolves this, and the numbers here are read from it and its CSV. | P1LE: `reports/p1l/readout-pooled.csv` `(rank, method, reader, hop, entity_step, reader_step)` (`top1_rate`, `top1_min`, `top1_max`, `mean_rank`), `reports/p1l/record-numbers.txt` (output of `scripts/p1l_record_numbers.py`), `per-step-readout.csv`; [decision record](../../reports/decisions/P1L.md) §3; PL v2/readouts `f6dcb4e`; [primary-source mapping](../reports/lit_check.md#two-jacobians-and-the-projects-proposed-readouts) |
| LP6. “Before any confirmation score, the rank-32 oracle readers were at or near the memory-free level in training and Pilot-2 validation, whereas the rank-128 readers were well above it; an in-sample capacity check of the oracle's independent random key and value tables gives hop-2 addressing of 48% at rank 32 and 99.85% at rank 128.” | narrowed | **Training/validation evidence, not a result of the registered test.** Training logs at step 17000 (T-L1; detail table below): r32 memory readers (B4, B5-L4, B8, M1-L4) batch accuracy **0.391–1.641%**, loss **7.0707–7.1774**; memory-free B3-L4 **0.312–0.391%**, **7.1453–7.1784**; context-retained B1/B2 **99.375–100.000%** (seeds17/29). r128: B4 **10.000–10.781%**, B5-L4 **10.312–13.516%**, B8 **11.250–12.812%**, M1-L4 **28.125–41.250%**; B3-L4 **0.312–0.625%**. Pilot-2 validation at r32 (V-L5; S/3 steps, training seed17, memory-only; five readers including the no-memory B3-L4): 30 reader×lr×hop cells, n1000 each, full-vocabulary EM **0.0–0.3%**, candidate accuracy **7.2–9.1%** against chance **7.142857%**; the selecting EM of the locked Pilot-2 (PLK, pooled hops1–3, n3000) is **0.033–0.133%** at r32 and **3.067–8.133%** at r128 for the memory readers (B4, B5-L4, B8, M1-L4; both learning rates), and **0.067%** for B3-L4 at both ranks. Capacity check (L6C; 768 seen names, 3 seeds, 2000 replayed T2 episodes per cell, exact hop-1 key): for the oracle's independent pair codes the reader-form map reaches fit cosine **0.395774/0.751933** and hop-2 addressing **48.316667/99.850000%** at r32/r128 (linear least squares **0.199764/0.407183**, **29.683333/99.316667%**); rotated entity codes give **99.200000/100.000000%**. | Training batches are not validation, and the logs are local uncommitted files re-read read-only by WP-T9. Amendment B7 calls the r32 readers “at chance”; some r32 seeds exceed the memory-free B3-L4 (by up to 1.250 pp, B5-L4 seed29 **1.641%** versus **0.391%**), so this ledger says “at or near the memory-free level”. B1/B2 seed43 (r32) and B4 seed43 (r128) had not run (Amendments B5, B7). The capacity check is an in-sample fit of an idealized one-step map with the exact hop-1 key, an upper bound on one mechanism. Amendment B7 reads it as the random-key table limiting hops ≥2 at r32 but not at r128, and as not explaining the r32 hop-1 failure, which it attributes to the other two random maps (question state→key, value code→name); that attribution is an inference and was not tested. No reader comparison and no multi-read claim at r32 follows (Amendment B7); an r128 pass would not make the LLM oracle equivalent to the symbolic one. At confirmation the check gave rank32 fail and rank128 pass (LP7). | L6T (local logs); PLK `d644b9b`/`1d1bbfd` (S17000, Pilot-2 selection); L6C `2bf7506`; PL B7 `2bf7506` |

### Amendments, selection, and measurement boundaries

The final design must be read with amendment precedence; earlier text is retained
as a historical record, not an additional active condition.
The registered paired bootstrap resamples whole episodes, conditional on the
fixed training seeds; it does not resample training seeds or give uncertainty
for the single-seed writer screens.

| Record | Effect at cutoff | Commit |
| --- | --- | --- |
| v2 | B8 primary fixedL4; M1-L4 shared comparator; random fact-list oracle always included; conditional learned writer | `f6dcb4e` |
| B1 | Historical total-budget/common-step rule; subsequently superseded by B2 | `ae1ae22`, time correction `6b7f292` |
| B2 | Total cap relaxed; per-job sizing restores target S17000, Pilot-2 floor(S/3)=5666 | `7bfaf16` |
| B3 | Add r128 oracle with its own Pilot-2; S17000 if timing rule permits, otherwise flagged lower S; B0–B2 not repeated | `df21654` |
| B4 | Remove r32-multiread versus r128-single-read contrasts; keep B8-r128 versus B8-r32 descriptive; report parameters/inference time | `379115d` |
| B5 | No learned writer selected; oracle r32/r128 are k2, every gate97.5%; learned condition/P3 not run; unfinished training/confirmation scheduled after maintenance | `6de902b` |
| B6 | Equal-parameter claim only from M1-L4@4−B4@1 (hops2/3, per rank) meeting the G1-L criterion (≥3 pp, 97.5% CI>0, every seed positive); otherwise B8 is reported with its parameter difference and no equal-parameter LLM claim is made. B8 adds 9,216 (+8.9%) at r32 and 147,456 (+31.0%) at r128 | `26b96bd` |
| B7 | Per-rank memory-use check (hop-1 full EM of B4@1, B8@4, M1-L4@4 each above B3-L4@4, 97.5% CI>0; otherwise the rank's G1-L, G2-L and B6 outcomes are reported but not interpreted). No multi-read claim at r32 whatever the gates show. At r128 a pass shows a learned value→key lookup and composition inside the LM, not equivalence with the symbolic oracle, and no unseen-name claim. Registered prediction: r32 fails, r128 passes. Follow-up studies are registered separately | `2bf7506` |

Per rank, B8@4, M1-L4@4, B4@1, B5-L4@4, and B3-L4@4 train with seeds17/29/43.
B3 is writer-independent but still rank-specific. B0 is frozen/untrained;
B1/B2 train with retained context and are reported in both protocols. AdamW,
B64, warmup20, weight decay.01, clip1, constant LR; Pilot-2 uses training seed 17
and validation generator seed 4242 to select lr in {.001,.003} by validation EM
on hops1–3 at trained depth, ties lower LR.
At this ledger's original cutoff the registered S target/contingency was **not a
verified final run lock**. The later committed locks (PLK, merged with `ca85a40`) record
S17000 at both ranks and the Pilot-2 selections (r32: lr .003 for M1-L4, .001 for the other
readers; r128: .001 for B3-L4, .003 for the others), and the main-run logs reach step 17000
(LP6). Confirmation is held until all required runs/baselines finish and the writer decision
fixes k; this paragraph describes the protocol and invents no outcome (the confirmation results are LP7–LP13).

M1-L4 comparisons are secondary; its L1…8 sweep holds that checkpoint's x/u/M
fixed. B8 has no shared-weight depth scan. Typed replication uses the same
checkpoints, E12 and three same-head distractors. Swap follow rates, per-hop
tables, B8−M1-L4, parameter counts and measured inference time are descriptive.
The current design does not test efficiency against a larger-rank single read.
[FR1](evidence_symbolic.md) retains the policy provenance and scoped accounting:
at the P1-L r32 projection, shared103426 versus derived B8 112642 registered
adapter parameters (+8.9%); at r128, shared475138 versus derived B8 622594 (+31.0%);
the symbolic unfactorized joint model grows by +69.2%. Each percentage belongs to its own
model and is not a trainable-only or whole-backbone figure. The comparison policy was amended before P1-L
confirmation; it does not retroactively change the historical P2-S gate.

LP5's registered readouts remain distinct from a proposed output-alignment
intervention. The updated [plan T-04/T-05 LLM record](../../plan/recurrent_lora_research_plan_2026-10-26.md)
(`6de902b`) does not pursue the latter after the writer-screen decision. For
the readout, a name-logit gradient g_j at the injection residual induces local
reader sensitivity `alpha B^T g_j`; this mapping is a proposed diagnostic,
not a measured alignment result or a transition-stability certificate.

### LP6: pre-confirmation memory-use evidence (training and validation only)

No confirmation or test episode was read. Training logs, T-L1, source L6T (local, uncommitted);
mean of the last five logged batches at step 17000 (batch accuracy as logged; loss is the training
cross-entropy). The seeds present are listed; B1/B2 seed43 (r32) and B4 seed43 (r128) had not run
(Amendments B5, B7). Rounded to 3 decimals (accuracy) and 4 decimals (loss):

| Rank | Reader | Seeds present | Batch accuracy %, seeds in order | Loss, seeds in order |
| --- | --- | --- | --- | --- |
| 32 | B3-L4 | 17/29/43 | 0.312 / 0.391 / 0.312 | 7.1784 / 7.1734 / 7.1453 |
| 32 | B4 | 17/29/43 | 0.938 / 1.250 / 0.391 | 7.1774 / 7.0881 / 7.1267 |
| 32 | B5-L4 | 17/29/43 | 0.391 / 1.641 / 0.469 | 7.1753 / 7.1164 / 7.1120 |
| 32 | B8 | 17/29/43 | 0.547 / 0.391 / 0.391 | 7.1689 / 7.1499 / 7.1406 |
| 32 | M1-L4 | 17/29/43 | 1.328 / 0.625 / 1.328 | 7.1351 / 7.1232 / 7.0707 |
| 32 | B1 | 17/29 | 100.000 / 99.375 | 0.0022 / 0.0587 |
| 32 | B2 | 17/29 | 99.844 / 99.609 | 0.0043 / 0.0277 |
| 128 | B3-L4 | 17/29/43 | 0.625 / 0.391 / 0.312 | 7.1875 / 7.1734 / 7.1480 |
| 128 | B4 | 17/29 | 10.781 / 10.000 | 5.1571 / 5.0929 |
| 128 | B5-L4 | 17/29/43 | 13.516 / 10.312 / 10.781 | 4.9873 / 5.2431 / 5.2407 |
| 128 | B8 | 17/29/43 | 12.031 / 12.812 / 11.250 | 4.9258 / 4.8669 / 4.9786 |
| 128 | M1-L4 | 17/29/43 | 38.906 / 28.125 / 41.250 | 3.4587 / 4.2808 / 3.2897 |

Pilot-2 validation at r32, V-L5, source L6T (S/3 steps, training seed17); five readers under the
memory-only protocol (B3-L4 has no memory), two learning rates, hops1–3, n1000 per cell; chance is
the mean inverse episode-candidate count, **7.142857%**:

| Reader | (lr, hop) cells | Full-vocabulary EM %, min–max | Candidate accuracy %, min–max |
| --- | --- | --- | --- |
| B3-L4 | 6 | 0.0–0.2 | 7.8–8.9 |
| B4 | 6 | 0.0–0.2 | 7.6–8.6 |
| B5-L4 | 6 | 0.0–0.2 | 7.3–8.5 |
| B8 | 6 | 0.0–0.2 | 7.2–8.4 |
| M1-L4 | 6 | 0.0–0.3 | 8.3–9.1 |
| all five readers | 30 | 0.0–0.3 | 7.2–9.1 |

Capacity check, L6C `2bf7506`: means over 3 code-table seeds, 768 seen names, 2000 replayed T2 episodes
(chains 8 and 4, E12, delta rule with lambda=beta=1, shuffled write order) per cell. The hop-1
query is the exact key, so the figures are an upper bound. “Reader-form map” is
`q=d+Q relu(U m+c)` with r hidden units fitted in sample (1500 Adam steps); “linear” is least
squares. Hop-2 addressing is top-1 among the 12 episode keys; Amendment B7 quotes the r128 pair-code
value as 99.8% (exact **99.85%**). Rotated keys are `normalize(Pi·global_codes[e])`.

| r | Keys | Map | Fit cosine | Hop-1 decode | Hop-2 addressing | Hop-2 decode (episode nodes) | Hop-2 decode (all 768) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | independent pair codes (P1-L oracle) | linear least squares | 0.199764 | 99.200000% | 29.683333% | 23.800000% | 16.183333% |
| 32 | independent pair codes (P1-L oracle) | reader-form map, r hidden units | 0.395774 | 99.266667% | 48.316667% | 39.800000% | 31.066667% |
| 32 | rotated entity codes | linear least squares | 1.000000 | 99.100000% | 99.200000% | 94.933333% | 92.750000% |
| 32 | rotated entity codes | reader-form map, r hidden units | 0.999985 | 99.200000% | 99.200000% | 95.033333% | 93.116667% |
| 128 | independent pair codes (P1-L oracle) | linear least squares | 0.407183 | 100.000000% | 99.316667% | 98.633333% | 98.633333% |
| 128 | independent pair codes (P1-L oracle) | reader-form map, r hidden units | 0.751933 | 100.000000% | 99.850000% | 99.850000% | 99.850000% |
| 128 | rotated entity codes | linear least squares | 1.000000 | 100.000000% | 100.000000% | 100.000000% | 100.000000% |
| 128 | rotated entity codes | reader-form map, r hidden units | 0.999814 | 100.000000% | 100.000000% | 100.000000% | 100.000000% |

These tables describe the training and validation record the registered memory-use check (B7)
responds to. The registered prediction (r32 fails the check, r128 passes) was a prediction, not an
outcome; the check itself is confirmation-set evidence, and LP7 reports that it was borne out. WL1 adds a rank-32 pilot with rotated and
shared oracle codes: neither reached the registered hop-1 threshold (validation EM 0.1% and 11.8% against 30%), so P1L-S does
not run; WL1 is pilot evidence and does not change the Amendment B7 rule or this prediction.

## P1-L confirmation and P1L-C: registered results

The P1-L confirmation ran after the post-maintenance release (authorization `reports/p1l/confirmation-authorization.json`,
k=2, 97.5%), with both ranks at source `P1L_SOURCE` `419276578c16f8d191a7ccb4d2ca67ea07fdec00`. Only the `oracle_random`
condition ran: WP-L4 selected no learned writer, so there is no learned-writer or P3 result (Amendment B5). Population V-L8;
every effect is a seed-pooled paired episode-bootstrap difference in percentage points over the three training seeds, with the
per-seed values listed in the order 17/29/43; the intervals condition on those seeds. The decision record is
[reports/decisions/P1L.md](../../reports/decisions/P1L.md). P1L-C (rows PC1–PC4) is an evaluation-only study of the nine
rank-128 checkpoints on a second range (population V-L9) with 95% intervals. Rows labelled **descriptive** have no gate in
their preregistration, or are reported without interpretation under Amendment B7 rule 1.

| ID and paper sentence | Status | Evidence | Conditions and non-implications | Source/selector |
| --- | --- | --- | --- | --- |
| LP7. “At rank128 B4@1, B8@4 and M1-L4@4 each exceed the memory-free B3-L4@4 on hop-1 full-vocabulary EM (the Amendment-B7 memory-use check passes, in every seed); at rank32 the check fails: the readers did not learn to use the oracle memory.” | supported | V-L8, hop 1, L as trained, memory-only. **Rank 128:** EM B4@1 **19.633333%**, B8@4 **19.033333%**, M1-L4@4 **47.850000%**, B3-L4@4 **0.000000%**; differences over B3-L4@4 **+19.633333 [18.416667, 20.933333]**, **+19.033333 [17.866667, 20.366667]**, **+47.850000 [46.016458, 49.666667]** pp; per seed B4 20.300/18.750/19.850, B8 18.000/19.950/19.150, M1-L4 51.400/39.650/52.500; check `memory_use_passed` true. **Rank 32:** EM **0.066667%**, **0.066667%**, **0.150000%** against B3-L4@4 **0.033333%**; differences **+0.033333 [0.000000, 0.100000]** (seeds 0.050/0.050/0.000), **+0.033333 [−0.050000, 0.133333]** (−0.050/0.050/0.100), **+0.116667 [0.016667, 0.233333]** (0.000/0.100/0.250); `memory_use_passed` false (the B4 lower bound is not above 0 and the B8 lower bound is below 0). | Rule (Amendment B7 rule 1, 97.5% intervals): each reader's hop-1 EM must exceed B3-L4@4 with the interval above 0. At rank32 M1-L4@4 alone is above B3-L4@4, and every hop-1 EM there is at most 0.15%. By rule 2 no claim about multiple reads in the LM is made at rank32 whatever the gates show, and by rule 1 the rank32 gate outcomes are not interpreted (LP8). Rule 2 attributes the rank32 failure to the oracle design; that attribution is an inference that no measurement here tests (LP6), and the paper states only the rule 1 sentence. The check is a hop-1 floor, not a proof that composition uses the memory (PC1). The LP6 prediction (rank32 fails, rank128 passes) was registered before any confirmation score and is borne out; it was not a result. | P1LE: `reports/p1l/endpoints.json` and `reports/p1l/r128/endpoints.json` `memory_use`, `memory_use_passed`; `memory-use.csv` rows `(rank, contrast, seed)`; [PL](../prereg/P1L.md) B7 `2bf7506` |
| LP8. (descriptive) Rank-32 gate outcomes, reported and not interpreted. | descriptive | V-L8, rank 32, hops2–3 unless stated. G1-L reach B8@4−B4@1 **−0.008333 [−0.108333, 0.100000]** pp (0.025/−0.075/0.025), hop 1 **0.000000 [−0.100000, 0.100000]**, typed **+0.058333 [−0.016667, 0.150000]**; G2-L against B5-L4@4 **−0.050000 [−0.166667, 0.066667]** and B3-L4@4 **+0.050000 [−0.033333, 0.150000]**; `G1_L` false, `G2_L` false. Equal-parameter contrast M1-L4@4−B4@1 **+0.058333 [−0.050000, 0.166667]** (0.025/−0.050/0.200). Hops2–3 EM: B8 **0.100000%**, B4 **0.108333%**, M1-L4 **0.166667%**, B5-L4 **0.150000%**, B3-L4 **0.050000%**. Context-retained baselines (B1, B2; hops1–3 seed means): **99.65–99.93%**. | Amendment B7 rule 1: these outcomes are reported but not interpreted as evidence for or against multiple reads, and rule 2 forbids a multi-read claim at rank32. The labels `fail` are gate arithmetic at the memory-free level (LP7), not “multiple reads fail at rank32”. B1/B2 retain the context and are descriptive baselines, not repeated at rank128. | P1LE: `reports/p1l/endpoints.json` `contrasts`, `G1_L`, `G2_L`; `accuracy.csv` (seed means over hops); [PL](../prereg/P1L.md) B7 |
| LP9. “At rank128 the four-unshared-reads reader B8 passes G1-L (B8@4−B4@1 on hops2–3 of at least 3 pp, interval above 0, positive in every seed, hop 1 at least −1 pp, typed replication interval above 0).” | not supported | V-L8, rank 128, hops2–3. B8@4 **6.341667%** against B4@1 **5.358333%**: **+0.983333 [0.266563, 1.716667]** pp (seeds 0.925/0.050/1.975); the point estimate is below 3 pp, the interval is above 0 and every seed is positive. Hop 1 **−0.600000 [−1.983542, 0.766667]** (−2.300/1.200/−0.700), not below −1 pp. Typed replication **+1.550000 [0.766667, 2.366667]** (1.575/0.600/2.475), interval above 0. `G1_L` false: the 3 pp criterion is the one that fails. | The decision-use text of the preregistration: a condition that fails G1-L is a negative LLM result for that condition. It concerns this reader (four unshared reads), this oracle memory and 17,000 steps with the locked Pilot-2 choices, not unshared reads in general. B8 adds 147,456 adapter parameters (**+31.0%**, 622,594 against 475,138), so B8−B4 is a same-rank, not an equal-parameter, contrast (Amendment B6). Intervals at 97.5% (k=2), conditioning on three seeds. P3 is not run (it required a passing learned condition). | P1LE: `reports/p1l/r128/endpoints.json` `contrasts.G1_reach`, `G1_hop1`, `G1_typed`, `G1_L`; [PL](../prereg/P1L.md) Decision use, B5, B6 |
| LP10. “At rank128 B8@4 is above the fixed-query B5-L4@4 and the memory-free B3-L4@4 on hops2–3 (G2-L passes).” | supported | V-L8, rank 128, hops2–3. B8@4 against B5-L4@4 (**5.508333%**): **+0.833333 [0.058125, 1.550104]** pp (0.875/0.100/1.525); against B3-L4@4 (**0.000000%**): **+6.341667 [5.758333, 6.933333]** (6.525/5.675/6.825). `G2_L` true. | A G2-L pass without a G1-L pass is not a multi-read claim for B8 (LP9). The interval against B5-L4@4 has a lower bound of 0.058 pp, and the large difference against B3-L4@4 reflects that the memory-free reader is at 0.000%. Same rank, not parameter-matched (+31.0%). | P1LE: `reports/p1l/r128/endpoints.json` `contrasts.G2_fixed_query`, `G2_no_memory`, `G2_L` |
| LP11. “At rank128, with a clean oracle memory, one shared module iterated four times (M1-L4@4) exceeds a single read of equal parameter count (B4@1) on hops2–3, in every seed.” | supported | V-L8, rank 128, hops2–3. M1-L4@4 **26.908333%** against B4@1 **5.358333%**: **+21.550000 [20.383229, 22.758333]** pp (seeds 24.450/15.100/25.100); point ≥3 pp, interval above 0, every seed positive, so the Amendment-B6 criterion is met. Both have **475,138** adapter parameters (442,368 trainable). Secondary: against B5-L4@4 **+21.400000 [20.216563, 22.550104]** (24.400/15.150/24.650) and against B3-L4@4 **+26.908333 [25.758125, 27.975000]** (30.050/20.725/29.950). | The equal-parameter claim is made only from this contrast and only at rank128 (Amendment B6); B8 is not parameter-matched. Amendment B7 rule 3: the result shows that the reader learns the value-to-key lookup over the 768 seen names and composes reads inside the LM; it does not establish equivalence with the symbolic oracle condition, and no claim is made about unseen names, new entity codes, a learned writer or end-to-end use. One oracle seed (17), one layout, `oracle_random` only; three training seeds with per-seed effects 15.1–25.1 pp. Absolute accuracy is moderate (hop-by-hop in LP12). The contrast is secondary and not a G1-L gate; the typed replication interval is computed only for the gate contrast. The extra reads have a measured latency cost (LP13), and no efficiency claim is made. | P1LE: `reports/p1l/r128/endpoints.json` `contrasts.M1_vs_B4`, `M1_vs_B5`, `M1_vs_B3`; `inference.csv`; [PL](../prereg/P1L.md) B6, B7 |
| LP12. (descriptive) Depth effect, walk specificity, comparison of B8 with M1-L4 and accuracy by hop at rank128. | descriptive | V-L8, rank 128. M1-L4 depth effect (L4−L1 of the same checkpoint): hops2–3 **+25.925000 [24.816042, 27.050104]** (29.550/18.525/29.700), hops6–8 **+3.677778 [3.205556, 4.183403]** (3.833/3.833/3.367); walk specificity (hops2–3 minus hops6–8 depth effect) **+22.247222 [21.155521, 23.372326]** (25.717/14.692/26.333). B8@4−M1-L4@4 on hops2–3 **−20.566667 [−21.783437, −19.358229]** (−23.525/−15.050/−23.125). Seed-mean accuracy by hop (primary, %): M1-L4@4 hops1–4 **47.85/35.35/18.47/7.25**, hops5–8 **3.90/4.50/4.72/4.97**; M1-L4@1 hops1–3 **37.83/1.07/0.90**; B4@1 **19.63/5.35/5.37**; B8@4 **19.03/6.65/6.03**; B5-L4@4 **17.27/5.38/5.63**; B3-L4@4 **0.00/0.00/0.00**. Secondary: B8(rank128)@4−B8(rank32)@4 on hops2–3 **+6.241667 [5.641667, 6.841667]** (6.375/5.625/6.725). | Secondary contrasts without a gate; the L1…8 scan holds the L4-trained checkpoint fixed. The depth effect beyond hop 3 is small in absolute terms. The rank comparison is descriptive and between a floor-level reader (rank32, LP7) and a rank128 reader; it is not read as a rank effect. B8 has no depth scan. | P1LE: `reports/p1l/r128/endpoints.json` `contrasts.M1_depth_reach`, `M1_depth_far`, `M1_walk_specificity`, `B8_vs_M1`; `accuracy.csv`; `rank-comparison.json` |
| LP13. (descriptive) Adapter parameters, measured inference time and compute cost of the P1-L confirmation. | descriptive | Adapter parameters (trainable): rank128 B3-L4 **409,600**, B4 = B5-L4 = M1-L4 **475,138** (442,368), B8 **622,594** (589,824; +147,456, +31.0%); rank32 B3-L4 **99,328**, B4 = B5-L4 = M1-L4 **103,426** (101,376), B8 **112,642** (110,592; +9,216, +8.9%). Mean time per question over the three seeds (ms; memory-only, a batch of 16 episodes = 128 questions, 20 repeats after 5 warm-ups), rank128: B3-L4 **0.370**, B4 **0.392**, B5-L4 **0.459**, M1-L4 **0.556**, B8 **0.617** (M1-L4 about 1.42 and B8 about 1.57 times B4); rank32: B4 **0.438**, B5-L4 **0.436**, M1-L4 **0.565**, B8 **0.554**. Cost: the P1-L stage ledger holds 101 jobs and **41.39** node·h in total (the 13 confirmation evaluation groups **2.11** node·h); the P1L-C step **0.14** node·h (106 jobs, 41.53 node·h in the ledger afterwards). | Latency is a cost and no efficiency claim is made (Amendment B4): the multi-read readers are slower than B4 at rank128. Timings of the two ranks come from different jobs and are not compared with each other. They are per-question reader timings and not whole-pipeline latencies. B8 is not parameter-matched; M1-L4 = B4 in parameters. | P1LE: `inference.csv` (both ranks), `jobs.json`, `docs/reports/P1L.md`; [PL](../prereg/P1L.md) B4 |
| PC1. “At rank128 the reader's answer depends on the bridge entity carried by its state after the first read (a causal bridge claim, P1L-C analysis B).” | not supported | V-L9, 256 hop-2 episodes per seed on the long chain, L=4, z_1 patched at the final question position, shift = rate(answer = t_B under the B-patch) − the larger of the C-patch and random-patch rates. Pooled shift M1-L4 **+0.390625 [−0.130208, 0.781250]** pp (seeds 0.391/0.781/0.000); B8 **0.000000 [−0.520833, 0.000000]** (0/0/0). The registered bar (≥10 pp, 95% interval above 0, positive in every seed) is met by neither reader. | The decision rule: otherwise the paper makes no causal claim, and per-step readability is reported as decoding only (LP5). The study does not show where the bridge is carried. 95% intervals; one oracle seed and layout. The C-patch control is not norm-matched (the random patch is). | P1LC: `reports/p1lc/bridge-shift.csv` `(reader, patch=final, seed=pooled)`, `results.json` `bridge`; [prereg](../prereg/P1L-C.md) Decision rule |
| PC2. (descriptive) Memory at the final question position only (analysis A, read position). | descriptive | V-L9, hops1–3 pooled, full-vocabulary EM, deployed reads against the memory kept only at the final question position, 95% intervals: M1-L4@4 **33.644444 → 3.738889%**, **−29.905556 [−30.755694, −29.094444]** pp (seeds −34.200/−21.400/−34.117); B8@4 **10.477778 → 0.988889%**, **−9.488889 [−10.077778, −8.911111]** (−10.017/−9.783/−8.667); B4@1 **10.394444 → 1.261111%**, **−9.133333 [−9.711111, −8.588611]** (−9.267/−8.967/−9.167). M1-L4@4 by hop: **−43.400000 [−44.950000, −41.982917]**, **−29.950000 [−31.400000, −28.550000]**, **−16.366667 [−17.550000, −15.216667]**. Negative in every seed and hop, B4@1 hop 1 **−17.466667 [−18.683750, −16.282917]** included. | Gates nothing. The preregistration reads a drop as a distribution shift, not as evidence of leakage: these readers were trained with reads at every question position. The report infers that the deployed readers also use earlier question positions; that is an inference from the drop. It does not identify where the bridge is carried. No multiplicity correction. | P1LC: `reports/p1lc/read-position.csv`; `docs/reports/P1L-C.md` |
| PC3. (descriptive) Rates of the bridge-patching analysis and the all-position patch. | descriptive | V-L9, 256 episodes per seed, pooled. M1-L4: deployed answer = t_B in world A **7.03%** and world B **37.24%** (t_A **33.59%** and **2.60%**), world C **6.90%**; final-position B/C/random patches t_B **7.55/7.03/7.03%**, t_A **33.07/33.20/32.94%**. All-position patch shift **+0.911458 [−0.651042, 2.343750]** (0.781/0.781/1.172). B8: world A t_B **9.38%**, world B **4.69%**, world C **7.29%**; final-position patches **9.51/9.51/9.38%**; all-position shift **−0.260417 [−0.781250, 0.000000]** (0/−0.391/−0.391). Mean state norms z_1^A **40.396** (M1-L4) and **39.358** (B8); ‖z_1^B−z_1^A‖ **6.921** and **4.367**. | Secondary, no gate. World B changes the memory (the first edge's tail), and the M1-L4 answer then follows t_B in 37.24% of episodes, whereas replacing z_1 alone, at the final or at every question position, does not move it. B8 does not follow world B. | P1LC: `reports/p1lc/bridge-rates.csv`, `bridge-shift.csv`, `results.json` `bridge`, `patch_episodes` |
| PC4. (descriptive) Replication of the P1-L contrasts on the P1L-C range. | descriptive | V-L9, hops2–3, deployed reads. M1-L4@4−B4@1 **+20.808333 [19.724792, 21.850000]** pp (97.5%: [19.574792, 21.941875]), seeds 24.425/15.250/22.750; B8@4−B4@1 **+0.925000 [0.283333, 1.533333]** (97.5%: [0.191667, 1.608333]), seeds 0.800/1.600/0.375. | P1-L's claim rules (G1-L, Amendment B6) are not changed by this range, and no gate is evaluated on it; the B8 point estimate is below the 3 pp G1-L bar. Same checkpoints, oracle and layout as LP9–LP11; not an independent replication of the training. | P1LC: `reports/p1lc/replication.csv`, `results.json` `replication` |

## RFX and WP-L6: registered follow-up results

Both studies were registered on master on 2026-10-27 ([RFX](../prereg/RFX.md), [WP-L6](../prereg/WP-L6.md)) before any run,
and report single-seed validation results with no confidence intervals. Their records are on master (merged `d4de948` and `ddfcbc2`; paths unchanged since the commits that introduced them): RFX
`reports/p1l-rfx/*.json` (`379fccd`) and `docs/reports/RFX.md` (`faacd37`; run at `efadc36`, job 3721047.opbs); WP-L6
`reports/wp-l6/decision.json` and `sizing.json` and `docs/reports/WP-L6.md` (`98d048c`; run at `7175e09`, jobs 3720811 and 3720812). Neither is a P1-L confirmation result. Rows labelled
**descriptive** have no gate in their preregistration.

| ID and paper sentence | Status | Evidence | Conditions and non-implications | Source/selector |
| --- | --- | --- | --- | --- |
| RX1. “At the frozen WP-L4 checkpoints R/r128 and R/r32 only an exact (ridge) store read together with the true key rescues the read (registered verdict `store+query`); at W5/r32 the factors are not separated (verdict `none`); no LSQ arm was triggered.” | supported | V-L6, training seed17, one hop-1 query per episode. Rescue rule: normalized read error ≤0.5 of the baseline and EM gain ≥10 pp over the baseline EM_b (the better of the as-is and gain-fitted deployed read). Baseline error e_b **0.614510 / 0.655206 / 0.713594** and EM_b **33.800000 / 11.625000 / 14.875000%** for R/r128, R/r32, W5/r32. Rescue flags are true only for the clean control, `ridge/true/asis` and `ridge/true/fit` at R/r128 and R/r32, and for no cell at W5/r32. `ridge/true/fit`: error **3.22e-10 / 4.42e-08 / 4.44e-08**, EM **84.200000 / 49.050000 / 17.675000%**; clean control EM **84.400000 / 48.800000 / 17.625000%**. By the first rule that applies the verdicts are R/r128 `store+query`, R/r32 `store+query`, W5/r32 `none`; storage, normalization, query and decoder do not apply. `lsq_arms` is empty. | Single seed, one evaluation set of 4,000 validation queries (EM standard error 0.5–0.8 pp); no confidence intervals. The replaced read is a distribution shift for the LM path after the read, and the true-key cells use the label and are not deployable. The `store+query` verdict means that only the exact store together with the true key rescues; the exact store with the deployed or the unit query does not (EM **38.800000 / 38.750000%** against EM_b 33.8% at R/r128; **13.475000 / 13.650000%** against 11.6% at R/r32), and because `ridge/true` almost equals the clean control the rule does not separate store from query. The W5/r32 verdict `none` does not separate the factors: its own clean control reaches only 17.6%, so no cell can rescue. Under the LSQ preregistration no LSQ arm runs and no LSQ release exists; DPFP is the only remaining writer screen. The rule was changed at registration (orchestrator decisions 1–5 of the preregistration). | RFXS wp/llm `379fccd`: summary.json `verdicts`, `rescue`, `lsq_arms`; R-r128.json, R-r32.json, W5-r32.json `verdict`, `cells`; [prereg](../prereg/RFX.md) decision rule |
| RX2. (descriptive) Reported cells of the RFX factorial: storage, query, decoder, conditioning. | descriptive | V-L6, R/r128 / R/r32 / W5/r32; error, full-vocabulary EM %. Deployed read (`delta/raw/asis`): **0.614522 / 0.658633 / 25.4377**, **33.800000 / 11.625000 / 14.875000**; gain-fitted (`delta/raw/fit`): **0.614510 / 0.655206 / 0.713594**, **33.775000 / 11.575000 / 8.525000**. Unit query on the deployed store (`delta/unit/fit`): **0.609681 / 0.650788 / 0.709338**, **33.950000 / 11.525000 / 8.625000**. True query on the deployed store (`delta/true/fit`): **0.497853 / 0.584000 / 0.566368**, **39.875000 / 13.950000 / 9.375000**. Exact store, raw query (`ridge/raw/fit`): **0.576519 / 0.601491 / 0.690632**, **38.800000 / 13.475000 / 9.100000**. Zero read EM **21.900000 / 7.500000 / 6.800000**. Clean-value head accuracy on the clean value **93.05 / 84.90 / 84.575%** and on the deployed read **18.95 / 7.125 / 11.10%**; refit linear decoder on the deployed reads **18.425 / 6.775 / 8.125%**, below the entity-restricted baseline at every checkpoint. Key Gram log10 condition number, median over 4,000 episodes (E=16 keys): **2.386 / 3.549 / 3.547**. | No gate. D3 is compared with the entity-restricted EM, not the full-vocabulary EM. The tables also hold error and EM by write age, by four random write orders (agreement with the natural order **97.8% / 95.3% / 77.2%** for the first permutation) and the normalization and gain effects (unit minus raw error about −0.005, EM at most +0.2 pp); they are descriptive. The four instrumentation checks pass at every checkpoint and the entity-EM guard passes (`full_hit_entity_miss` 0 in every cell). The W5/r32 as-is error 25.4 is a scale artifact that the fitted gain removes. | RFXS `379fccd`: R-r128.json, R-r32.json, W5-r32.json `cells`, `refit`, `conditioning`, `write_order`, `checks`; `docs/reports/RFX.md` `faacd37` tables |
| RX3. (descriptive) “By the preregistered reading guide, at R/r128 and R/r32 write interference in the deployed delta store dominates and the query path is not the bottleneck.” | descriptive | Guide: `delta/true/fit` near the OA level means the query path is the bottleneck; near the checkpoint's own EM means write interference dominates. `delta/true/fit` EM **39.875000%** at R/r128 against its own EM 33.8% and the OA level 84.0%; **13.950000%** at R/r32 against its own 11.6% and the clean control 48.8%. The unit true key on the learned delta state still reads a cross-talked value (error **0.497853 / 0.584000**, against **3.22e-10 / 4.42e-08** for the exact store). The ridge store needed η=1 (a regularized, no longer interpolating store) with the learned queries and η=1e-6 with the true key. | The guide is descriptive and not a rule; one seed and one evaluation set. It does not isolate a unique mechanism or prove interference, it does not cover W5/r32 (`none`), and OA remains the value/output ceiling, not an addressing control. It is consistent with, and not additional to, LL7–LL10 (read distortion without a unique cause). | RFXS `379fccd`: `cells` delta/true/fit and ridge/true/fit, `parameters.eta`; [prereg](../prereg/RFX.md) “Reading guide”. |
| WL1. “At rank 32 the LM path did not learn the two oracle code designs of WP-L6 within the pilot budget (one seed, one learning rate, 5,666 steps); P1L-S is not run.” | supported | V-L7, validation (`val`, eval seed4242, 1,000 episodes), memory-only, L=4, final checkpoint after step 5,666: hop-1 full-vocabulary EM O-rot **0.100000%**, O-shared **11.800000%**, both below the **30%** pass threshold. Registered decision: `stop`, `selected` null. Both pilots completed all 5,666 steps. | Registered conditions: M1-L4, r=r_mem=32, lr 3e-3 (P1-L's r32 selection, not retuned), seed17, constant lr, no early stopping, one run per arm. The result is not a proof of impossibility. O-shared above O-rot is descriptive and is not used for selection; no arm is compared with B4 or B8, and no statement about multiple reads follows. P1L-S (the rank-32 study with the selected code) does not run, and the P1-L r32 and r128 `oracle_random` conditions of LP3 are unchanged. | WL6D wp/oracle `98d048c`: `reports/wp-l6/decision.json` (`status`, `arms.*.hop1_L4_full_em`, `threshold`); [prereg](../prereg/WP-L6.md) |
| WL2. (descriptive) Cost and step times of the WP-L6 pilots. | descriptive | Jobs 3720811 (O-rot) and 3720812 (O-shared), 19.3 and 22.8 min, **0.321** and **0.380** node·h; step time **0.194 / 0.231 s**, peak memory about 7.2 GB each; about **0.97** node·h for the whole step 2 including five test jobs and the smoke. `sizing.json` recommends separate jobs (a packed job would need 01:05:00). | No gate. The 32-cell (L, hop) tables and code-drift diagnostics of the registered reports are not cited here. At `7175e09` one of 140 regression tests (a time-dependent P1-L launcher guard, not on the WP-L6 path) failed; it passes at `4dcc6b3`, and no assertion was changed. | WL6D `98d048c`: `docs/reports/WP-L6.md` “Step 2 on compute”, `sizing.json` |

## DPFP and P1L-D: registered result

DPFP ([preregistration](../prereg/DPFP.md), registered 2026-10-27, Amendment A1 before any run) is the last registered writer screen:
learned writers with DPFP key and query features in the WP-L4 protocol (T1, E=16, training seed17, layer14, L=1, batch 32/Q 4, lr .003),
arms D32-3 (r32, ν=3), D128-1 and D128-3 (r128, ν=1 and 3), one run per arm, validation episodes only (population V-L10). LSQ ran no arm
(RX1), so this study ends the learned-writer line. No test or confirmation data were read. Rows labelled **descriptive** have no gate
in the preregistration.

| ID and paper sentence | Status | Evidence | Conditions and non-implications | Source/selector |
| --- | --- | --- | --- | --- |
| DP1. “A learned writer with DPFP key and query features passes the WP-L4 writer gate (validation EM ≥0.80).” | not supported | V-L10, final checkpoint, 1,000 validation episodes, eval seed4242. E=16: D32-3 **39.0%** (390/1000), D128-1 **58.0%** (580), D128-3 **60.8%** (608); the best arm is **19.2 pp** below the gate. WP-L4 references, not rerun: R at r32 **12.0%** (120), R at r128 **33.5%** (335). Registered rule: no arm reached 0.80 and the best reached ≥0.50, so D128-3 was retrained once at E=8 (flagged; lock `e8-lock.json`): **73.0%** (730/1000) at 26,065 steps (checkpoints **56.9/64.1/70.7/72.4/73.0%**), below 0.80 and not accepted. **Registered decision: stop. No writer is selected.** | Registered consequence: the paper reports the negative result and no further writer screen follows from this study; LSQ ran no arm (RX1), so no learned writer reaches the WP-L4 gate in this project. One seed, one learning rate, fixed budgets of 22,370–26,065 steps; a validation EM from 1,000 episodes has a standard error of about 1.5 pp. The curves flatten at E=16 but the E=8 arm was still rising (+0.6 pp over the last interval) and no longer run was made, so this is not a proof that learned writing is impossible. Amendment A1: DPFP changes both the key/query feature geometry and the query normalization (unit norm), so the gains over the WP-L4 references (**+27.0 pp** at r32, **+24.5/+27.3 pp** at r128) are not credited to decorrelation alone; the unit-normalized query effect on the frozen WP-L4 checkpoints was measured separately by RFX (RX2: error about −0.005, EM at most +0.2 pp, evaluation only), and whether a unit query changes training is not tested because no LSQ arm ran. The E=8 retrain halves the facts per episode, so its **+12.2 pp** gain over E=16 mixes memory load with the DPFP features and is not an equivalent comparison. Inference parameters are 103,426 (r32) and 475,138 (r128), equal to B4's at the same rank. | DPFE: `reports/p1l-dpfp/arms/*/record.json`, `arms-e8/D128-3`, `refs/R-r32`, `refs/R-r128`, `e8-lock.json`, `release.json`, `release-e8.json`; `docs/reports/DPFP.md` decision table; [prereg](../prereg/DPFP.md) Decision rule, A1 |
| DP2. (descriptive) Read diagnostics of the DPFP arms. | descriptive | V-L10, final checkpoint, one seed. Normalized read error **0.391/0.431/0.363** (D32-3, D128-1, D128-3; E=8 retrain **0.269**) against **0.666** (R at r32) and **0.631** (R at r128). Read-value head accuracy **30.2/46.0/51.0%** (E=8 **62.9%**) against **8.8%** and **19.7%**; clean-value head accuracy **83.3/92.8/90.6%** (E=8 **89.1%**) against **84.5%** and **93.3%**, so the clean/read gap narrows (R at r128 93.3 to 19.7; D128-3 90.6 to 51.0) without closing. Value SIR **5.47/6.88/8.00** (E=8 **28.53**) against **0.48/0.55**. Key Gram RMS in the original space **0.374/0.366/0.321** (E=8 **0.313**) and mean maximum absolute cosine **0.907/0.870/0.829** (E=8 **0.710**) against **0.332/0.288** and **0.871/0.767** for R. | No gate. DPFP lowers the read error and raises the read-value head accuracy but not enough here; key overlap of the learned projections in the original space is not lower for the E=16 arms, so what changes is the addressing geometry and the read, not a decorrelation of the learned keys. The phi-space overlap values (Gram RMS 0.20/0.18/0.19) are not comparable with them, since phi features are nonnegative and have a positive cosine floor. Each head is a decoder fitted on its own checkpoint (one seed); a lower read error does not by itself give a correct answer. | DPFE: `docs/reports/DPFP.md` diagnostics table (printed by `scripts/analyze_dpfp.py reports/p1l-dpfp`) |
| DP3. (descriptive) Zero- and shuffled-memory audit of the final DPFP checkpoints. | descriptive | V-L10, the same 1,000 validation episodes, original / zero memory / another episode's memory (EM %, NLL): D32-3 **39.0 (3.501) / 0.0 (15.087) / 0.0 (12.746)**; D128-1 **58.0 (2.277) / 0.0 (12.736) / 0.2 (13.635)**; D128-3 **60.8 (2.129) / 0.0 (17.440) / 0.2 (13.336)**; D128-3 at E=8 **73.0 (1.316) / 0.1 (11.612) / 0.2 (14.502)**. The original condition reproduces each arm's record (correct count and NLL). | Evaluation only, no training. The answers of every DPFP arm depend on the episode's own memory: with zero or another episode's memory the EM is at most 0.2% (2 of 1,000). This shows the writer's memory is used and is episode specific; it does not show that the read is accurate (normalized read error 0.27–0.43, DP2). | DPFE: `reports/p1l-dpfp/audit-e16.json`, `audit-e8.json` (`runs.<arm>.original`, `zero`, `shuffle`) |
| PD1. “A learned writer selected by DPFP passes G1-L and G2-L in a P1-L learned condition at its rank (P1L-D).” | not run | None. P1L-D is registered only if DPFP selects a writer, and DP1's registered decision is stop (no writer selected), so P1L-D was not registered and not run. There is no P1-L learned condition and no P3. | Preregistration decision rule: a selected writer would trigger a new registration with a 95% interval for one condition, the oracle conditions keeping k=2, and P3 reconsidered only if it passed; none of this happened. The oracle conditions of P1-L are unchanged (LP7–LP11). | [DPFP prereg](../prereg/DPFP.md) “Decision rule”; `docs/reports/DPFP.md` (“Not done”); plan F3 |

## Three distinct privileged controls

| Control | What is supplied/bypassed | What success can establish |
| --- | --- | --- |
| D2 answer-code intervention (LL3) | Answer identity indexes a trained code table; reader and memory bypassed | Output-path feasibility with supplied answer identity, not retrieval |
| OA/OArho, the value/output ceiling (LL5–LL10) | True clean learned fact value replaces Mq at each valid question token | The ceiling for the learned value encoding and output path under privileged read replacement; not an addressing control, since addressing, attenuation and interference are all bypassed |
| P1-L oracle_random (LP1–LP4, LP6–LP13) | Fact-list writes fixed random pair-address/entity codes into M; question uses normal LM encoding/reader/output | The registered gates test the LM reader condition (LP7–LP11); code writes are privileged, and random storage can still interfere |

The fact-list oracle addresses `(entity,relation)` by
`entity*n_rel+relation`; values are the symbolic `OracleWriter.global_codes`
at the same oracle seed. Each valid fact writes in list order with invalid
slots skipped. Cross-reader checks preserve M/write counts, while same-checkpoint
depth checks preserve x/u/M/counts; separately trained question-side A need not
produce identical u (C7 `b0bf9a3`/`b54ec3a`).

## Scope statements

These statements bound every row; a row's own conditions add to them.

- **Vocabulary.** Names are single-token entries of the closed 1,024-name vocabulary (seen pool IDs
  0…767, unseen pool 768…1023). Probes classify names over the seen pool, and every screen and
  registered P1-L episode (splits `val`, `test`) uses seen names. Nothing here supports a claim about
  unseen names or open vocabularies; Amendment B7 repeats this for r128.
- **Depth.** Training covers hops1–3. P1-L evaluates hops1–8, B8 only at its trained fixed L4 (no depth
  scan) and M1-L4 with a secondary L1…8 scan. Claims stay within the trained hops and depths unless a
  row reports a measured longer-hop cell.
- **Writers.** Learned writers (LL1–LL10) and the oracle writer (LP1–LP6) are separate claims: no
  learned writer was selected, P1-L is an oracle-reader study, and D2 and OA are privileged controls,
  not writers.
- **Oracle design.** The LLM oracle is not the symbolic oracle (LP6; the
  [symbolic scope statement](evidence_symbolic.md#scope-of-every-row)).
- **DPFP rows.** DP1–DP3 are single-seed validation results (training seed17, one learning rate, validation seed4242, 1,000
  episodes) without confidence intervals; they test the WP-L4 writer gate only, and P1L-D (PD1) was not run. DPFP does not
  isolate decorrelation from the query normalization (Amendment A1).
- **Confirmation rows.** LP7–LP13 and PC1–PC4 concern the `oracle_random` memory only (oracle seed17, one layout, no
  learned writer), the 768 seen names and the trained hops. At rank128 they show that the shared M1-L4 reader learns the
  value-to-key lookup and composes reads on hops2–3 (LP11); they do not establish equivalence with the symbolic oracle (Amendment B7
  rule 3), and no claim is made at rank32 (LP7, LP8). B8 is not parameter-matched to B4 (+31.0% at rank128) in any of them;
  the equal-parameter claim is M1-L4 against B4 only. The per-step readouts (LP5) are descriptive decoding only.
- **Follow-up rows.** RX1–RX3 and WL1–WL2 are single-seed validation results (training seed17, validation seed4242)
  without confidence intervals. RFX replaces the read at the final question position of frozen checkpoints and
  trains nothing; WL1 is a pilot with a registered stop, not a P1-L confirmation. Neither changes the P1-L
  preregistration, and B8 is not parameter-matched to B4 in either.

## Must-not-say list

- “Learned writing is impossible in LLMs,” “rank32 is fundamentally insufficient,”
  or “the available budget cannot be the cause.” These screens test particular
  optimizers, sites, ranks and finite budgets under one training seed.
- “Input probes prove one shared A can jointly recover all targets,” “the names
  were unseen,” or “probe classification is full-vocabulary LM EM.”
- “D2 proves learned codes are accessible,” “OA is a fair addressing-only control,”
  “OA is the P1-L random oracle,” or “oracle_random is an exact lossless store.”
- “W5 is the factorized-position writer.” W4 is that intervention; W5 combines
  standardization and two auxiliary losses, and has a different parameter budget.
- “Key top1, candidate accuracy, tail membership, or auxiliary-head accuracy is
  full-vocabulary EM,” or “the candidate/full gap counts outside-candidate errors.”
- “The clean/read head gap proves all information was erased,” “key correlation
  uniquely caused failure,” or “small normalized address/read loss guarantees a
  useful decoded answer.” Prop.11 distinguishes raw gain, interference and decoder
  compatibility; these observations do not uniquely identify every cause.
- “Every repair received the same training budget,” “OA/W5 resumed from4800,” or
  “the best saved validation checkpoint is the registered final checkpoint.”
- “Validation screens have97.5% CIs,” “all P1-L work stopped,” “P1-L passed,” or
  “there is already a learned end-to-end multihop result.” No writer was selected and no learned condition ran. The
  registered 97.5% intervals belong to the oracle-r32/r128 confirmation (LP7–LP13), not to LL1–LL10; at confirmation the
  gated reader B8 failed G1-L at rank128 (LP9) and rank32 failed the memory-use check (LP7).
- “B8 is parameter-matched to B4,” “same-rank success proves larger-rank efficiency,”
  or “short smoke training-step timing is inference latency.” B8 adds 9,216 parameters (+8.9%) at
  rank32 and 147,456 (+31.0%) at rank128 over B4 = M1-L4; only M1-L4 versus B4 is an
  equal-parameter comparison (Amendment B6). The symbolic unfactorized joint model grows by +69.2%
  (FR1); each percentage belongs to its own model.
- “The LLM oracle condition is the symbolic oracle condition,” or “symbolic oracle results carry
  over to P1-L.” The symbolic oracle takes question code, keys, values and decoder from one code
  table; the LLM oracle keys facts by independent random (entity,relation) pair codes and stores
  values from a different table, so every read after the first needs a learned value-to-key lookup
  (LP2, LP6).
- Any statement that multiple reads help, or fail, in the LLM at r=32, such as “the r32 readers
  learned to walk hops” or “r32 failed to compose.” Amendment B7: no multi-read claim at r32 whatever
  the gates show. Before confirmation the r32 readers were at or near the memory-free level, which
  is a failure of this oracle design, not a reader comparison (LP6). The confirmation check confirms that the rank32 readers
  did not learn to use the oracle memory (LP7); its rank32 gate outcomes (LP8) are not evidence for or against multiple reads.
  The rank128 pass does not make the LLM oracle equivalent to the symbolic one.
- “Composition holds for unseen names,” “outside the 1,024-name vocabulary,” or “for new entity
  codes.” Probes, screens and P1-L episodes use seen names only; the unseen pool is not tested in any
  row here.
- “OA/OArho is an addressing control” or “oracle addressing.” OA is the value/output ceiling: it
  replaces the read by the true value, bypassing addressing, attenuation and interference, so its EM
  bounds the learned value encoding and output path, not addressing (LL5–LL10).
- “The r128 oracle implementation and final hyperparameter locks are established
  by a protocol amendment,” “episode index100000 is the evaluation seed,” or
  “planned per-step readability already demonstrates causal path following.”
- “RFX identified storage, query or normalization as the failure of the deployed read,” “write
  interference is established as the cause,” or “an exact store fixes the deployed read.” The RFX verdict
  is `store+query` at R/r128 and R/r32: only the exact store together with the true key (a label, not
  deployable) rescues, the exact store with the deployed or unit query does not, and the rule does not separate
  store from query (RX1). The reading “write interference dominates” is a descriptive, single-seed use of
  the preregistered guide (RX3). Nothing is identified for W5/r32 (`none`), and no LSQ arm was run.
- “WP-L6 shows that the LM cannot learn oracle codes,” “O-shared is the selected design,” “P1L-S ran or was
  decided,” or any statement about multiple reads at r32 from WL1. The pilot stopped on one seed, one
  learning rate and 5,666 steps (O-rot 0.1%, O-shared 11.8% hop-1 EM against 30%); it is not a proof of
  impossibility, and O-shared above O-rot is descriptive and not a selection.
- “Multiple reads beat one read in the language model,” “B8 composes at rank128,” or “the shared and the unshared reader
  agree.” At rank128 only the shared M1-L4 beats B4@1 (LP11, +21.550000 pp); B8 fails G1-L (LP9, +0.983333 pp against the
  3 pp bar) and its G2-L pass is small against B5-L4@4 (LP10, +0.833333 pp). M1-L4 against B8 is secondary and ungated (LP12).
- “The language-model result establishes composition for the symbolic oracle, for unseen names or entity codes, for a learned
  writer, or end-to-end,” or an equal-parameter claim for B8, B5-L4 or B3-L4. The rank128 statement is bounded by Amendment B7
  rule 3 to the `oracle_random` memory over the 768 seen names; the equal-parameter claim is M1-L4@4 against B4@1 only
  (LP11); B8 adds 147,456 parameters (+31.0%).
- “The first-step state carries the bridge entity,” “the bridge is causally carried by z_1,” or “the final-position read carries
  the accuracy.” P1L-C gives no causal-bridge claim for either reader (PC1); final-position-only reads collapse every reader,
  a distribution shift that does not locate the bridge (PC2); the answer follows world B when the memory changes, not when z_1
  alone is replaced (PC3).
- “The extra reads cost nothing,” “M1-L4 or B8 is faster than a single read,” or an efficiency claim over a larger-rank single
  read. M1-L4 takes about 1.42 and B8 about 1.57 times B4's measured time per question at rank128 (LP13); latency is a cost.
- “The intermediate states are the successive entities of the chain,” “the reader follows the path step by step,” “the readouts
  show the bridge is carried in z_s,” or “the readouts support the composition claim (LP4).” The logit-lens readouts (LP5) are
  decoding only: M1-L4's early states decode earlier-hop entities, best at a step s ≥ e rather than at s = e, and the rates
  track the reader's own depth-s accuracy; B8 decodes names only at s=4; the name-gradient and random directions are at or
  near chance for both readers, and every method is at chance at rank32. P1L-C gives no causal bridge (PC1).
- “DPFP selects a learned writer,” “a learned writer reaches the WP-L4 gate,” “P1L-D ran or passed,” “a learned P1-L condition or
  P3 is planned,” or “learned writing in the LM is impossible.” DPFP reached 39.0%, 58.0% and 60.8% at E=16 and 73.0% for the E=8
  retrain against a gate of 80%: the registered decision is stop, no writer, P1L-D not run (DP1, PD1); this is a one-seed,
  one-learning-rate result with fixed budgets and not a proof of impossibility.
- “DPFP's gain is due to decorrelation of the keys,” or “DPFP lowered key overlap.” DPFP also changed the query normalization
  (Amendment A1); the original-space key overlap of the E=16 arms is not lower than R's (DP2); the E=8 gain over E=16 mixes load
  with the features (DP1).
- “The zero- and shuffled-memory audit shows the DPFP read is accurate.” It shows the answers depend on the episode's own
  memory (DP3); the read error stays 0.27–0.43 (DP2).

The screen diagnostics motivate a bounded read-distortion interpretation, not
a theorem about optimization or a universal ceiling. The symbolic B9 record
supports only historical reader selection; the LLM-specific claims above require
their own LL/LP sources and statuses.

## Audit boundary

Numbers were checked against the committed scalar CSV/JSON artifacts named above,
using bounded file inspection and Git provenance. Percentage conversion and
rounding preserve the exported metrics; no confidence interval or scientific
estimate was recomputed. Reports guided claim selection, while raw tables and
protocol locks supply the numerical authority. No compute job, experiment,
model/checkpoint/sample load, plotting, or test suite ran for this synthesis.
WP-T9 added LP6 from local, uncommitted run files and committed files: the last five logged
batches of each main-run `train_log.jsonl` and the Pilot-2 `validation/metrics.json` files were
read with the Python standard library on the login node (no torch, no checkpoint or sample load),
and the committed `capacity.json` and run locks were read as scalars. Only means of logged values,
minima/maxima and arithmetic on registered counts were computed; no confirmation or test data were
read.
The original studies' runtime audits are historical evidence.
WP-T10 part 1 added RX1–RX3, WL1–WL2 and the placeholders. The RFX and WP-L6 numbers were read with `git show` from the
committed JSON files and reports at wp/llm `379fccd`/`faacd37` and wp/oracle `98d048c` (those files are not on this
branch); fractions were converted to percent and no statistic was recomputed. No checkpoint or sample was loaded
and no job ran. The plan's result line for RFX cites `379fccd`; the report is at `faacd37`.
WP-T10 part 3b added DP1–DP3 and PD1 from the committed `reports/p1l-dpfp/` records and `docs/reports/DPFP.md` (all numbers of the
report's tables were reproduced with `scripts/analyze_dpfp.py reports/p1l-dpfp` on the login node, stdlib only; no checkpoint was
loaded and no job ran; the audit values were read from `audit-e16.json` and `audit-e8.json`). The ledger records no cost deviations,
so the report's test and smoke overrun (about 0.78 against the registered 0.5 node·h) is not entered here.
WP-T10 part 2 added LP7–LP13 and PC1–PC4 and repointed the RFX and WP-L6 citations to master, where those files are unchanged from
the cited commits (checked with `git diff`). The P1-L and P1L-C numbers were read on the login node with the Python standard
library from the committed `reports/p1l/**` and `reports/p1lc/**` JSON and CSV files and converted to percent at six decimals
(per-hop accuracy and timing values are arithmetic means of the committed per-seed rows); no statistic was recomputed, and no
checkpoint was loaded and no job ran. The decision record `reports/decisions/P1L.md` was used for the registered interpretation; its
summary of the per-step readouts and its B0 value did not match the committed CSVs at that time; the corrected record (`f4f9a98`,
merged `5eca4d2`) fixes both, B0 is still not cited, and WP-T10 part 3a reads the readout numbers from `readout-pooled.csv`
(arithmetic maxima and ranges over its cells; nothing recomputed from raw runs) and checks them against `record-numbers.txt`.
