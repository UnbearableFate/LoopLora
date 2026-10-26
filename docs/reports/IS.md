# IS — memory load × read depth (2026-10-27)

**Both primary writers are consistent with Prop. 4's depth-amplification
prediction. Multi-read capability holds through E36, the largest trained load;
its failure boundary was not located.** Scope is r=r_mem=32 and the registered
layouts. [Figure](../../reports/is/accuracy_vs_L.png) / [PDF](../../reports/is/accuracy_vs_L.pdf).

The [design lock](../../reports/is/design.json), commit `fb62cf1`, preceded
inference/training: **54 E12 checkpoints**, five loads E6/12/18/24/36, and
**36 new runs**, seeds17/29/43, 15000 steps, batch64. Oracle_random and
joint_fact are primary; joint is descriptive. Inherited LRs were fixed without
a new search, as preregistered. The 10% E36 smoke read only timing/memory.
All new training finished before seed **999001** confirmation: 2000 episodes
per load; all **360 sample sets** preceded aggregation.

[All 50 contrasts](../../reports/is/tables.md) retain 97.5% CIs and per-seed
values; [CSV](../../reports/is/endpoints.csv) also retains absolute per-seed
accuracies. Reach means hops2/3; effects below are percentage points.

- **IS-E1:** oracle D(12)=34.750, D(36)=7.492;
  **ΔD=−27.258 [−28.800,−25.816]**, per seed −28.725/−25.900/−27.150.
  Joint_fact D(12)=24.042, D(36)=4.642;
  **ΔD=−19.400 [−20.542,−18.241]**, per seed −20.525/−19.200/−18.475.
  ΔD uses independent load bootstraps. [Best L](../../reports/is/best_L.csv)
  is descriptive; far-hop D(36) is only +0.800/+0.722 pp respectively.
- **IS-E2:** every E12/24/36 comparison meets estimate≥3 pp and CI>0.
  At E36, oracle M1/B8 margins over B4 are
  **+4.483 [3.950,5.033] / +5.942 [5.225,6.675]**;
  joint_fact margins are **+3.433 [2.983,3.867] / +5.008 [4.500,5.500]**.
- **IS-E3:** load-specific training is worse than evaluating E12 checkpoints
  at that load: **all eight CIs are negative**, effects −2.042 to −6.283 pp.
  The fixed inherited LR/step protocol limits interpretation of retraining.
- **IS-E4:** B8@4 current-answer accuracy on affected/untouched paths is
  **41.75%/37.61% oracle**, **28.62%/30.47% joint_fact**. Affected-path stale
  rates are **0.51% / 6.03%**. [Full update table](../../reports/is/t3_update.csv)
  has counts, CIs and per-seed values across supported L; untouched stale rate
  is undefined. Current terminals are selected more often than obsolete ones.

Generator-valid Q3 adjustments were locked before inference: **14 stored
facts = 12 current edges (8,4) + two obsolete facts**; hops1–4 preserve default
mixed updated/untouched queries. B8 supports L4 only. E6 has hops1–4 and no
far-hop summary. Load and chain layout vary together; CIs condition on three
fixed training seeds (2000 episode-bootstrap draws, seed2718). No gate changes.

Validation: `pytest -q -o addopts= tests/test_is.py` — **23 passed**, PBS
**3715626**, `mg0031`; includes batch64 forward/gradient/optimizer equivalence,
independent-load intervals and release barriers. [IS-only dense writes](../../reports/is/runtime.json)
remove per-fact synchronization; forwards and three-step losses matched exactly.
Shared production code and earlier runs stayed unchanged. Final PBS **3715657**
exited 0; [audit](../../reports/is/audit.json) passed **13,296,000 rows**, seals,
paired labels, immutable depth snapshots and checkpoint checks.

[Cost](../../reports/is/budget.json): **0.819/2 node-hours**, all **11 jobs
exited 0**, one node each, ≤23-minute requests via `scripts/submit.py`.
[Artifacts](../../reports/is/README.md) contain CSV summaries and the figure;
raw runs/checkpoints remain in `runs/is/`. No open checks.
