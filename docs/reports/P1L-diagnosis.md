# P1-L writer diagnosis — completed 2026-10-27

**The output path works; the learned writer still fails. No tested writer is
feasible for T2.** All scientific runs used seed 17, E=16 T1, four relations,
head sharing 0.3, r=32/layer 14/L=1, batch 32/Q=4, memory-only;
1000 val episodes, seed 4242.
No test/confirm, other seeds, reader/joint or T2 runs occurred.

**D1 / link (a):** 96 frozen-feature probes used 4096 train episodes (all 16
facts/four questions) and Pilot-1's val questions. Full linear and learned
rank-32 classifiers trained 1200 steps, batch 512, AdamW lr 0.01→0.001,
with train-only standardization and no validation selection.
Names had 1024 outputs; train/val retain the existing 768-name seen pool.
Question identity/relation are available at the current input; fact-final
identity generalizes less well. Separate probes do not prove a shared A can
encode all targets jointly; overfitting prevents absence claims.

**D2 / link (b):** supplied `z=c(answer)` at the final question token, training
only the 1024×32 code table and B, S=1500, α=1. The reader (ρ=0.25) was
bypassed. EM 98.7% at lr 0.003 establishes output-path feasibility;
frozen B0 EM 0%, NLL 10.685. This does not establish writer-code learnability.

**D3 / link (c), D4:** 6400 steps (204,800 episodes/run; 4.27× Pilot-1) fit
one-hour jobs; neither time guard fired. Loss was logged every 50 steps.
Lower loss brought no EM rescue. D2 success/D3 failure triggered D4: only A's
input moved to the pre-layer-14 residual; output stayed at o_proj14.
The tested option preserves default behavior/checkpoints and rejects cross-site
writer loads. Residual input also failed. Larger budgets remain untested.

**Link (d):** final source-fact key/query top1 was 6.0–7.5% (chance 6.25%), with
negative mean true-key margins. Learned matching/retrieval is the leading
failure diagnosis; these measurements do not uniquely separate shared-A,
K/Q, value-code or memory-training failures.

**Recommendation:** retain **o_proj input, fact_final, S=1500, batch=32,
lr=3e-4** for further writer diagnosis only. Neither extension improved EM
beyond the existing 0.5 pp tie band. Investigate task-conditioned matching
before another P1-L pilot. [Decision](../../reports/p1l-diagnosis/recommendation.json).

**Validation:** compute tests passed 2/18/3/68 cases across four jobs (overlap);
`pytest -q tests/test_llm_{residual,train,evaluate,run}.py` passed 68 in 3713800
on mg0746. [Exact commands](../../reports/p1l-diagnosis/submissions.json).
Final audit 3714453 on mg0846 reproduced all 96 probes and 18 validation
tables, including key statistics and paired labels; it checked scope, clean
source, determinism, checkpoint steps and D4 completion.
All runtime used `scripts/submit.py`. One invalid test-path submission was
corrected/rerun; its 7 s is included. **Total 2.06833 node·h**, below 4; all jobs
one node ≤1 h. No WP-L2 checks remain.

Artifacts: [all runs/timings](../../reports/p1l-diagnosis/runs.csv),
[all probes](../../reports/p1l-diagnosis/probes.csv),
[loss curves](../../reports/p1l-diagnosis/loss_curves.png),
[loss CSV](../../reports/p1l-diagnosis/loss_curves.csv),
[audited summary](../../reports/p1l-diagnosis/summary.json).
Raw checkpoints/samples: `/work/xg24i002/x10041/LoopLora/runs/p1l/diagnosis-20261026/`.

Probe accuracies below are **full / rank-32 %**, rounded; resN means the
pre-layer residual (zero-based). Write positions have 16,000 held-out facts;
question positions have 1000 held-out queries. Relation at the start-name
position is a causal negative control: its text occurs later.

| Site | Position | Tail % | Head/start % | Relation % |
| --- | --- | ---: | ---: | ---: |
| o14 | fact_final | 56.8 / 84.2 | 59.8 / 80.9 | 100.0 / 100.0 |
| o14 | tail token | 89.4 / 97.6 | 75.6 / 89.8 | 100.0 / 100.0 |
| o14 | question final | — | 99.3 / 99.4 | 100.0 / 100.0 |
| o14 | question start | — | 99.4 / 99.4 | 24.5 / 25.5 |
| o14 | question relation | — | 98.8 / 99.4 | 100.0 / 100.0 |
| res14 | fact_final | 81.4 / 86.3 | 64.2 / 69.0 | 100.0 / 100.0 |
| res14 | tail token | 99.6 / 100.0 | 48.2 / 62.4 | 100.0 / 100.0 |
| res14 | question final | — | 99.0 / 99.3 | 100.0 / 100.0 |
| res14 | question start | — | 99.4 / 99.4 | 23.9 / 25.4 |
| res14 | question relation | — | 98.4 / 99.4 | 100.0 / 100.0 |
| res7 | fact_final | 96.2 / 97.3 | 82.9 / 76.5 | 100.0 / 100.0 |
| res7 | tail token | 100.0 / 100.0 | 37.0 / 48.2 | 100.0 / 100.0 |
| res7 | question final | — | 99.2 / 99.4 | 100.0 / 100.0 |
| res7 | question start | — | 99.4 / 99.4 | 23.7 / 24.1 |
| res7 | question relation | — | 98.6 / 99.3 | 100.0 / 100.0 |
| res21 | fact_final | 91.2 / 99.9 | 86.2 / 91.4 | 100.0 / 100.0 |
| res21 | tail token | 99.8 / 100.0 | 55.5 / 86.6 | 100.0 / 100.0 |
| res21 | question final | — | 99.4 / 99.4 | 100.0 / 100.0 |
| res21 | question start | — | 99.4 / 99.4 | 22.8 / 25.8 |
| res21 | question relation | — | 98.6 / 99.3 | 100.0 / 100.0 |

All subsequent accuracies use 1000 validation answers. Loss is mean full-vocabulary
validation NLL. D2 runtime includes training and final validation; B0 is a
separate untrained reference evaluation repeated identically for each lr.

| D2 | Steps | EM % | Candidate % | NLL | Runtime s |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 frozen | 0 | 0.0 | 0.0 | 10.685 | 2.8–6.3 |
| Oracle lr=3e-4 | 1500 | 20.0 | 33.6 | 6.058 | 235.5 |
| Oracle lr=1e-3 | 1500 | 85.3 | 92.9 | 0.983 | 235.4 |
| Oracle lr=3e-3 | 1500 | 98.7 | 99.3 | 0.101 | 164.7 |

Triples are measurements at **1500 / 3200 / 6400 steps**. Key top1 measures
cosine ranking of the true source fact's normalized Ku against Q u_question;
this diagnostic does not replace the actual delta-memory computation.
Runtime includes the three validations. Two independent runs shared each GPU.

| Diagnostic / input | lr | EM % triple | NLL triple | Final candidate % | Final key top1 % | Runtime s |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| D3 / o_proj | 3e-4 | 0.3 / 0.1 / 0.0 | 7.501 / 7.386 / 7.244 | 4.0 | 7.1 | 3365.7 |
| D3 / o_proj | 1e-3 | 0.0 / 0.0 / 0.1 | 7.366 / 7.306 / 7.209 | 4.6 | 7.5 | 3366.0 |
| D4 / residual | 3e-4 | 0.0 / 0.0 / 0.1 | 7.284 / 7.234 / 7.141 | 4.2 | 6.0 | 3304.4 |
| D4 / residual | 1e-3 | 0.2 / 0.0 / 0.0 | 7.267 / 7.247 / 7.213 | 4.0 | 6.0 | 3304.9 |

PBS allocated costs below count each job once, including shared-worker jobs.

| Job | Work / result | Host | Allocated s | Node·h |
| --- | --- | --- | ---: | ---: |
| 3713089 | Probe tests: 2 passed | mg0086 | 20 | 0.00556 |
| 3713110 | D1: 96 probes complete | mg0943 | 123 | 0.03417 |
| 3713111 | Oracle/train tests: 18 passed | mg0841 | 26 | 0.00722 |
| 3713137 | Diagnosis tests: 3 passed | mg0852 | 31 | 0.00861 |
| 3713139 | D2:three oracle lrs complete | mg0853 | 439 | 0.12194 |
| 3713165 | D3:two writer lrs complete | mg1048 | 3395 | 0.94306 |
| 3713783 | Invalid test path; replaced | mg0745 | 7 | 0.00194 |
| 3713800 | Residual/train/eval/run tests: 68 passed | mg0746 | 31 | 0.00861 |
| 3713853 | D4:two residual lrs complete | mg0833 | 3334 | 0.92611 |
| 3714411 | Initial artifact audit passed | mg0367 | 20 | 0.00556 |
| 3714453 | Final audit: paired labels/key statistics passed | mg0846 | 20 | 0.00556 |
