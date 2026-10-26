# P1-L Pilot-1 — completed 2026-10-26

**No admissible E*_L at this budget.** Validation full-vocabulary EM was below
0.95 at every tested E; E=8 also missed the 0.80 fallback. The preregistered
decision is **writer diagnosis before any P1-L T2 run**. No further tuning ran.

Throughput selected **batch 32 per run, two independent processes/GPU**,
93.53 training episodes/s. All six cells fit; selected summed per-worker peak
allocation was 5.77 GiB. E=24/all-token writer timing used five warmups and 25
measured steps. Worst-worker p95=0.717092 s;
`floor100((1800−90)/(1.5×p95))` fixed **S=1500** before policy results: **48,000
training episodes / 192,000 answer queries per run**.
[Locked P1-L batch/budget](../../configs/p1l/pilot1/locked-plan.json).

All nine scientific runs used seed 17, writer/M1, r=32, layer 14, L=1,
memory-only; T1 had four relations and 0.3 head sharing. Evaluation was 1000 val
episodes, seed 4242, one single-hop question per episode. The T2 chain layout
(8,4,2,2) does not apply to direct T1 facts. Policy and capacity sources were
`c98a233` and `add488e`, with identical numerical implementation.
Train-plus-eval totals were **582–998 s**, below 30 minutes.

At E=16 the best EM was 3/1000; all six cells were within the five-answer tie
band. The rule selected **fact_final, lr=3e-4**, locked before capacity
configurations/runs and verified by the final audit. E=16 was reused for capacity.

Loss fell initially, then plateaued for seven runs. In the final 20%, the E=8
fitted loss fell 1.02%, while E=24 rose 1.41%; the other fitted changes were
within ±1%. These are descriptive trends, not proof of optimization completion.
[Curves](../../reports/p1l_pilot1_loss_curves.png) and
[raw loss CSV](../../reports/p1l_pilot1_loss_curves.csv).

**Validation:** 78 tests passed in `3712351.opbs` (`mg0705`):
`tests/test_llm_pilot1.py`, `tests/test_llm_train.py`, `tests/test_llm_evaluate.py`;
four analysis regressions passed in `3712586.opbs` (`mg0622`):
`tests/test_llm_pilot1_analysis.py`. None skipped. Final audit `3712906.opbs`
(`mg0841`) reproduced all nine sample tables and checked 1000 unique val
episodes/run, clean source, determinism, budgets and selection chronology.
All runtime used compute via `scripts/submit.py`.

All seven one-node jobs exited 0; **total 1.08528 node·h**, including tests,
throughput and audits; scientific writer jobs cost 1.02472 node·h. Artifacts:
[run CSV](../../reports/p1l_pilot1_runs.csv),
[audited summary](../../reports/p1l_pilot1_summary.json),
[policy lock](../../reports/p1l_pilot1_policy_selection.json),
[raw throughput](../../reports/p1l_pilot1_throughput.json),
[job costs](../../reports/p1l_pilot1_job_costs.csv).
Checkpoints/samples remain under
`/work/xg24i002/x10041/LoopLora/runs/p1l/pilot1-20261026/`.
All WP-L1 checks completed; writer diagnosis is a separate task.

| Training batch | 1 process: episodes/s | 2 processes: episodes/s |
| --- | ---: | ---: |
| 8 | 16.26 | 30.47 |
| 16 | 32.07 | 55.72 |
| 32 | 60.72 | **93.53** |

Percentages below use 1000 validation answers per row. Loss is final logged CE;
times are measured training / evaluation seconds, with concurrent workers.

| E | Write policy | lr | EM % | Candidate % | Loss | Tail | Train / eval s |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 16 | **fact_final** | **3e-4** | **0.3** | 3.9 | 7.465 | plateau | 803.7 / 13.9 |
| 16 | fact_final | 1e-3 | 0.0 | 3.2 | 7.368 | plateau | 773.6 / 13.5 |
| 16 | fact_final | 3e-3 | 0.1 | 3.0 | 7.302 | plateau | 783.8 / 13.6 |
| 16 | all_tokens | 3e-4 | 0.3 | 3.6 | 7.435 | plateau | 804.0 / 13.5 |
| 16 | all_tokens | 1e-3 | 0.0 | 3.5 | 7.364 | plateau | 774.9 / 12.7 |
| 16 | all_tokens | 3e-3 | 0.0 | 3.3 | 7.285 | plateau | 785.7 / 12.7 |
| 8 | fact_final | 3e-4 | 0.0 | 9.0 | 7.404 | falling | 571.5 / 10.4 |
| 12 | fact_final | 3e-4 | 0.0 | 5.7 | 7.416 | plateau | 606.8 / 8.9 |
| 24 | fact_final | 3e-4 | 0.0 | 2.6 | 7.510 | rising | 981.0 / 16.8 |

| PBS job | Work | Host | Allocated s | Node·h |
| --- | --- | --- | ---: | ---: |
| 3712351 | Tests + throughput | mg0705 | 193 | 0.05361 |
| 3712393 | Policy lr=3e-4 | mg0876 | 840 | 0.23333 |
| 3712394 | Policy lr=1e-3 | mg0904 | 810 | 0.22500 |
| 3712395 | Policy lr=3e-3 | mg0919 | 820 | 0.22778 |
| 3712586 | Tests + selection lock | mg0622 | 13 | 0.00361 |
| 3712628 | E=8/12/24 capacity | mg0562 | 1219 | 0.33861 |
| 3712906 | Final audit + curves | mg0841 | 12 | 0.00333 |
