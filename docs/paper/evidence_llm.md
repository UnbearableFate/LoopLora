# LLM writer screens and registered P1-L claims (WP-T8)

Evidence cutoff: `master` at `f1dd6e2`, including the final WP-L4 release
(`91fac0c`, merge `39f7f0b`) and P1-L Amendment B5 (`6de902b`). This ledger
synthesizes committed validation screens and implementation/protocol records.
It contains **no P1-L confirmation result**. The corresponding
[symbolic ledger](evidence_symbolic.md) remains authoritative for symbolic claims.
WP-T9 (2026-10-27, on `dc659e0`) adds LP6 (pre-confirmation memory-use evidence and Amendments
B6/B7), scope statements, the OA rename (value/output ceiling) and further exclusions after two
external reviews. Sources newer than the cutoff above are Amendments B6 (`26b96bd`) and B7
(`2bf7506`), the run locks (PLK, merged `ca85a40`) and the capacity check (L6C). The ledger still
contains no P1-L confirmation result.

## Status, populations, and metrics

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
| P1-L confirmation (pending) | Split `test`, episode indices `[100000,102000)`, 2000 episodes | Registered 97.5% paired episode-bootstrap intervals; no effects or intervals supplied here. The interval is an index range, not an evaluation seed. |
| T-L1 | P1-L main-run training logs at step 17000, rank 32 and 128: mean of the last five logged batches (steps 16600–17000; batch size 64 episodes), hops1–3 mixed, trained depth | Training-batch accuracy and loss as logged; no CI; neither validation nor confirmation. Local files (`runs/` is git-ignored). |
| V-L5 | P1-L Pilot-2 at rank 32 (S/3 steps, training seed17, lr in {.001,.003}): split `val`, eval seed4242, 1000 episodes per (reader, lr, hop) cell, memory-only protocol, five readers (B3-L4 has no memory) | Full-vocabulary EM and episode-candidate accuracy against the 1/14 chance level; no CI; learning-rate selection data, not confirmation. Local files. |

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
| PLK | [r32 run lock](../../reports/p1l/lr-S-lock.json), [r128 run lock](../../reports/p1l/r128/lr-S-lock.json) | r32 `d644b9b`; r128 `1d1bbfd`; merged with `ca85a40`, after this ledger's original cutoff; fields `S`, `lrs`, `selection` |
| L6T | Main-run training logs `runs/p1l/main/main/*/*/train_log.jsonl` and `runs/p1l/main/r128/main/*/*/train_log.jsonl`; Pilot-2 `runs/p1l/main/pilot/*/*/validation/metrics.json` | **Not committed** (`runs/` is git-ignored). The committed record is the summary in PL B7 (`2bf7506`); WP-T9 re-read the raw files read-only |
| L6C | [capacity check](../../reports/p1l-c1/capacity.json), [script](../../scripts/analyze_oracle_capacity.py) | `2bf7506`; rows `(r,seed,keys,map)`, means under `mean` |

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
| LP3. “The final registered P1-L has two same-rank oracle conditions and 97.5% gate intervals.” | supported | B5: oracle_random r32/r128 only, k2; no learned writer/P3. G1-L compares B8@4−B4@1 on hops2/3: point≥3pp, CI lower>0, every-seed positive; hop1 point≥−1pp; same-checkpoint typed replication CI lower>0. G2-L requires positive CIs versus B5-L4 and B3-L4. Amendment B6: an equal-parameter claim only from M1-L4@4−B4@1 meeting the G1-L criterion. Amendment B7: a per-rank memory-use check (hop-1 EM of B4@1, B8@4, M1-L4@4 each above B3-L4@4, 97.5% CI>0) precedes interpretation. | Registered design/decision only; 2000 paired episode draws over fixed seeds17/29/43; not simultaneous coverage of every comparison. Primary B8 is same-rank, not parameter-matched: 103426→112642 (+8.9%) at rank32 and 475138→622594 (+31.0%) at rank128 (FR1; Amendment B6); shared M1-L4/B4 are parameter-matched. No gate outcome supplied. | PL B3 `df21654`, B4 `379115d`, B5 `6de902b`, B6 `26b96bd`, B7 `2bf7506`; L4E selection |
| LP4. “Oracle P1-L establishes a multi-read capability gain inside the LM.” | open | **Pending:** separate r32 and r128 confirmation effects/CIs/gate outcomes, comparative latency, M1 depth scans/swaps, and readout results. No numerical placeholder is filled. | Neither symbolic B9 reader selection, C6/C7 smokes, D2 answer codes, nor OA validation accuracy establishes the registered claim. Same-rank controls are not enlarged-rank efficiency controls. Each rank's outcome is read through the memory-use check (LP6); no multi-read claim at rank32. | PL design only; no final P1-L confirmation artifact supplied at cutoff |
| LP5. “The protocol registers descriptive per-step output and name-gradient readouts.” | supported | M1-L4/B8 z_s,s1…4: trained output path/frozen upper layers; name-logit-gradient directions at layer14 residual, averaged over512 validation questions for1024 names. Report hop-s entity rank/top1 by hop against other steps and random directions. | Registered analysis only; results remain open. Not a gate or observed ordered trajectory; no causal pointer-chasing conclusion from readability alone. Upper-layer forwards incur diagnostic work distinct from deployed loop cost. This proposed probe differs from the published averaged residual-transport J-lens and from JS transition Jacobians. B8 is not parameter-matched (FR1). | PL v2/readouts `f6dcb4e`; [primary-source mapping](../reports/lit_check.md#two-jacobians-and-the-projects-proposed-readouts) |
| LP6. “Before any confirmation score, the rank-32 oracle readers were at or near the memory-free level in training and Pilot-2 validation, whereas the rank-128 readers were well above it; an in-sample capacity check of the oracle's independent random key and value tables gives hop-2 addressing of 48% at rank 32 and 99.85% at rank 128.” | narrowed | **Training/validation evidence, not a result of the registered test.** Training logs at step 17000 (T-L1; detail table below): r32 memory readers (B4, B5-L4, B8, M1-L4) batch accuracy **0.391–1.641%**, loss **7.0707–7.1774**; memory-free B3-L4 **0.312–0.391%**, **7.1453–7.1784**; context-retained B1/B2 **99.375–100.000%** (seeds17/29). r128: B4 **10.000–10.781%**, B5-L4 **10.312–13.516%**, B8 **11.250–12.812%**, M1-L4 **28.125–41.250%**; B3-L4 **0.312–0.625%**. Pilot-2 validation at r32 (V-L5; S/3 steps, training seed17, memory-only; five readers including the no-memory B3-L4): 30 reader×lr×hop cells, n1000 each, full-vocabulary EM **0.0–0.3%**, candidate accuracy **7.2–9.1%** against chance **7.142857%**; the selecting EM of the locked Pilot-2 (PLK, pooled hops1–3, n3000) is **0.033–0.133%** at r32 and **3.067–8.133%** at r128 for the memory readers (B4, B5-L4, B8, M1-L4; both learning rates), and **0.067%** for B3-L4 at both ranks. Capacity check (L6C; 768 seen names, 3 seeds, 2000 replayed T2 episodes per cell, exact hop-1 key): for the oracle's independent pair codes the reader-form map reaches fit cosine **0.395774/0.751933** and hop-2 addressing **48.316667/99.850000%** at r32/r128 (linear least squares **0.199764/0.407183**, **29.683333/99.316667%**); rotated entity codes give **99.200000/100.000000%**. | Training batches are not validation, and the logs are local uncommitted files re-read read-only by WP-T9. Amendment B7 calls the r32 readers “at chance”; some r32 seeds exceed the memory-free B3-L4 (by up to 1.250 pp, B5-L4 seed29 **1.641%** versus **0.391%**), so this ledger says “at or near the memory-free level”. B1/B2 seed43 (r32) and B4 seed43 (r128) had not run (Amendments B5, B7). The capacity check is an in-sample fit of an idealized one-step map with the exact hop-1 key, an upper bound on one mechanism. Amendment B7 reads it as the random-key table limiting hops ≥2 at r32 but not at r128, and as not explaining the r32 hop-1 failure, which it attributes to the other two random maps (question state→key, value code→name); that attribution is an inference and was not tested. No reader comparison and no multi-read claim at r32 follows (Amendment B7); an r128 pass would not make the LLM oracle equivalent to the symbolic one. | L6T (local logs); PLK `d644b9b`/`1d1bbfd` (S17000, Pilot-2 selection); L6C `2bf7506`; PL B7 `2bf7506` |

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
fixes k; no validation or confirmation outcomes are invented.

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
responds to. The registered prediction (r32 fails the check, r128 passes) is a prediction, not an
outcome; the check itself is confirmation-set evidence and is pending.

## Three distinct privileged controls

| Control | What is supplied/bypassed | What success can establish |
| --- | --- | --- |
| D2 answer-code intervention (LL3) | Answer identity indexes a trained code table; reader and memory bypassed | Output-path feasibility with supplied answer identity, not retrieval |
| OA/OArho, the value/output ceiling (LL5–LL10) | True clean learned fact value replaces Mq at each valid question token | The ceiling for the learned value encoding and output path under privileged read replacement; not an addressing control, since addressing, attenuation and interference are all bypassed |
| P1-L oracle_random (LP1–LP4, LP6) | Fact-list writes fixed random pair-address/entity codes into M; question uses normal LM encoding/reader/output | A future gate can test the registered LM reader condition; code writes are privileged, and random storage can still interfere |

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
  “there is already a learned end-to-end multihop result.” No writer was selected;
  oracle-r32/r128 confirmation remains pending, and their registered CIs are not
  retrospective uncertainty for LL1–LL10.
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
  is a failure of this oracle design, not a reader comparison (LP6). An r128 pass would not make the
  LLM oracle equivalent to the symbolic one.
- “Composition holds for unseen names,” “outside the 1,024-name vocabulary,” or “for new entity
  codes.” Probes, screens and P1-L episodes use seen names only; the unseen pool is not tested in any
  row here.
- “OA/OArho is an addressing control” or “oracle addressing.” OA is the value/output ceiling: it
  replaces the read by the true value, bypassing addressing, attenuation and interference, so its EM
  bounds the learned value encoding and output path, not addressing (LL5–LL10).
- “The r128 oracle implementation and final hyperparameter locks are established
  by a protocol amendment,” “episode index100000 is the evaluation seed,” or
  “planned per-step readability already demonstrates causal path following.”

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
The original studies' runtime audits are historical evidence. P1-L confirmation,
comparative inference measurements, and intermediate-state outcomes remain open.
