# WP-T6 — IS integration and latent-reasoning literature

Merged `master` at `14af341` into `wp/theory`. Added IS1–IS4 to
[evidence_symbolic.md](../paper/evidence_symbolic.md), checked against committed IS
CSV/JSON at `e144246` and design lock `fb62cf1`: C999, exact intervals, selectors,
conditions, and the revised Prop.4 measurement ledger. No discrepancy at the IS
report's rounding was found. Load/layout covariance, the point≥3 pp/CI>0 rule,
the unlocated failure boundary, inherited training protocol, absolute update-rate
intervals, and undefined untouched stale rates remain explicit.

Updated [draft.md](../paper/draft.md) with IS results/claim tags and the completed
memory-load figure. Extended [paper_figures.py](../../scripts/paper_figures.py);
the [PDF/PNG](../paper/figures/README.md) and
[provenance](../paper/figures/provenance.json) include fixed-checkpoint load curves
and existing within-load CIs. Only the end-to-end figure remains pending; the earlier
F2 single-checkpoint-export limitation remains labeled.

Added all nine requested works to the [primary-source literature audit](lit_check.md)
and manuscript. Recorded identities, versions, inspected sections, overlaps and claim
boundaries. All nine bodies were accessible; Ouro's HTML returned 404 but its PDF
opened, and ReLIT's image-only algorithm/external code were not audited. Distinguished
J-lens transport/name-gradient proxies from reader Jacobians, STARS's K=1 JVP proxy
from spectral radius, and local fixed-point stability from global contraction or
answer correctness. JS and P1-L probes remain planned evidence, not results.

Validation: `scripts/submit.py --name paper-wpt6 --nodes 1 --walltime 00:10:00
--worktree . --wait -- scripts/paper_figures.py --revision HEAD` resolved scientific
inputs to **23a5165**. One compute job **3715853.opbs**, host **mg0032**, exit **0**;
recorded start/end 14 seconds apart. Six figure pairs rendered; visually checked the
new load figure. Independent numerical/literature reviews, local links/anchors,
claim IDs, provenance/output checks, and `git diff --cached --check` passed.
No new experiment, model evaluation, statistical resampling, or pytest run; no
plotting or application workload ran on the login node. No task deviation.
