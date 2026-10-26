# P4-S-L — learned halting on frozen readers

**0/6 primary families qualify.** Every family fails hop-1 preservation (required
Δ ≥ −1 pp), despite passing reach gain (≥3 pp, CI >0) and walk specificity
(reach minus far gain, CI >0). All 18 primary seed results also fail preservation.
Differences below are learned minus L1, in percentage points; specificity brackets
are 99.166667% CIs from 2000 paired episode-bootstrap draws across three fixed
seeds. These intervals condition on those seeds.

| Primary r32 family | θ | Hop-1 Δ | Reach Δ | Specificity [CI] |
| --- | ---: | ---: | ---: | ---: |
| joint_fact / M1 | .45 | −33.90 | +21.58 | +16.52 [15.26, 17.67] |
| joint_fact / M1-lin | .40 | −49.15 | +24.73 | +15.44 [14.03, 16.89] |
| oracle_random / M1 | .25 | −15.70 | +32.51 | +27.98 [26.38, 29.58] |
| oracle_random / M1-lin | .30 | −20.85 | +37.50 | +18.84 [16.95, 20.68] |
| t1_frozen_fact / M1 | .25 | −21.90 | +9.44 | +6.89 [6.00, 7.77] |
| t1_frozen_fact / M1-lin | .35 | −26.17 | +10.22 | +7.55 [6.66, 8.45] |

The two joint families remain descriptive: M1 (θ=.90) satisfies all inequalities
(hop-1 +6.05, reach +8.74, specificity +3.06 [1.67, 4.50]); M1-lin (θ=.75)
fails specificity (+0.93 [−0.26, 2.16]). Four primary joint_fact/oracle families
gain 0.88–1.61 pp overall over their validation-selected best locked label-free
reference; frozen-fact differences are about −0.10 pp with CIs crossing zero.
Primary gaps below the label-informed any-step oracle remain 10.25–36.34 pp.
[Full tables](../../reports/p4s-l/tables.md) include every family, per-seed effects,
counts, L1/L4/L16, all three label-free rules, and the separate true-hop reference.

Learned inference takes **20.39–25.25 CUDA ms per prepared batch of 64**, versus
3.05–3.31 ms for L16 (6.26–8.22× slower). Timing uses five warmups and 20 repeats;
wall/CUDA mean, median and p95 are exported. Primary mean chosen depths are
5.31–10.04, but executed updates are 12.74–15.99: fallback scans all 16 states,
and the post-state read feature adds one memory read. There is no measured speedup.

All 24 frozen readers received a 3393-parameter 51→64 ReLU→1 head, trained on
20,000 train episodes with all eight queries and 16 states, 2000 Adam updates,
lr .001 and 256 whole episodes per batch. Reader hashes stayed unchanged.
Design lock **9366da4** preceded every head; eight validation-only θ choices
(seed 4242, 2000 episodes; full 152-row grid) were committed in **6e4545e** before
fresh confirmation (test seed 999002, 2000 episodes).

**Approved prereg clarification:** no oracle_random/r32 label-free thresholds
existed in C3/C4. Both missing families used C3's unchanged validation grid/tie
rule before any head training, recorded in the design lock. Existing C4 thresholds
were reused unchanged. There were no other scientific departures.

All **159 compute tests passed** (3715743); formal training (3715784) and
confirmation/timing/analysis (3715868) exited 0. Audits cover 768,000 queries,
72 latency cells, lock chronology and 24 exported head seals. Five PBS jobs used
**0.144444/1 node-hours**, including one corrected test-fixture failure; no
runtime checks ran on login or were skipped. [Artifact index](../../reports/p4s-l/README.md)
and [job ledger](../../reports/p4s-l/jobs.md) retain reproducibility details.

The registered negative interpretation applies: neither label-free nor this
post-hoc learned head repairs overshoot in these six frozen-reader families;
fixed-depth training (B8, M1-L4; P2-S-F1) remains the remedy found so far. This
does not establish that learned halting is impossible. No follow-up is registered.
