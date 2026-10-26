# WP-T5 — symbolic paper draft

Merged `master` at `5e5afff` into `wp/theory`. Delivered
[draft.md](../paper/draft.md), approximately 5,000 main-text words plus proof sketches
and a protocol appendix. Empirical sentences carry evidence-ledger claim IDs; Gate 3
failure, writer dependence, and stopping failures remain in the main text. The LLM
section, IS memory-load figure, and symbolic/end-to-end comparison are marked TODO.

Added [paper_figures.py](../../scripts/paper_figures.py) and five
[PDF/PNG figure pairs](../paper/figures/README.md): depth profiles, hop×L heatmaps,
paired controls, larger-single-read accuracy–latency points, and stopping mean/tail
latency. All scientific inputs come from immutable committed CSV/JSON at `5e5afff`;
[provenance.json](../paper/figures/provenance.json) records revisions, selectors,
aggregation, exact plotted points, and CI/status scope. No evaluation, training,
timing measurement, or bootstrap was rerun.

Validation command: `scripts/submit.py --name paper-wpt5 --nodes 1 --walltime 00:10:00
--worktree . --wait -- scripts/paper_figures.py --revision 5e5afff`.
One Miyabi-G compute job, **3715578.opbs**, host **mg0037**, exit **0**, rendered all
five pairs, with recorded job start/end 11 seconds apart. The PBS wrapper confirmed compute
allocation before execution. Static source/selector and independent evidence/theory
reviews passed after corrections; visually inspected all PNGs. Local links, claim
IDs, main-text length, output formats, provenance, and staged whitespace checked.

Deviation/limitation: F2 commits only three-seed-mean hop×L cells, so a literal
single-checkpoint F2 curve cannot be recovered within the committed-data-only rule.
The F2 panels explicitly use the three-seed mean; the P1-S panels use the individual
seed17 checkpoint. No missing checkpoint curve was inferred or reevaluated. Other
unavailable study results remain placeholders as requested. No plotting, model execution,
or statistical analysis ran on the login node; only file/metadata inspection, editing,
and submission did.
