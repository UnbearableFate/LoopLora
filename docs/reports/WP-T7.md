# WP-T7 — overshoot evidence and same-rank framing

Merged `master` at **5d2efd3** into `wp/theory`. Updated
[evidence_symbolic.md](../paper/evidence_symbolic.md) with HL1–HL5, JS1–JS6 and
FR1, checked against committed P4-S-L (`95736ec`) and JS (`96ff2e9`) CSV/JSON.
The ledger includes all 64 learned-halting endpoints, all 36 JS contrasts
(24 RS-reference contrasts), reached-state radius summaries, execution/timing
scope, source locks, fixed-seed uncertainty, and the distinct confirmation sets
999002/999003 and CI levels 99.166667%/97.5%.

Updated [draft.md](../paper/draft.md): none of six primary learned-halting
families preserves hop1; JS reduces overshoot but fails removal, lowers long-hop
and stopped accuracy, and supplies no fixed-point/contraction certificate.
Props.3/7's measurement ledger and the must-not-say list retain these limits.
The 2026-10-27 policy makes same-rank capability primary and reports extra latency
as a cost. Historical P2-S/F1 larger-rank findings and the original practical-gate
failure remain unchanged. Scope clarification: B8's +8.9% means registered P1-L
adapter parameters (103,426→112,642), not symbolic or whole-backbone overhead.

Extended [paper_figures.py](../../scripts/paper_figures.py) with the JS depth
profiles, registered overshoot/long-hop intervals, and immutable source selectors.
The [JS PDF/PNG](../paper/figures/README.md#js_overshoot) and
[provenance](../paper/figures/provenance.json) identify seeds, subsets, source
commits, claim statuses, and the explicitly zoomed long-hop scale. The previous
six figure pairs are byte-identical after rendering.

Validation: `scripts/submit.py --name paper-wpt7 --nodes 1 --walltime 00:10:00
--worktree . --wait -- scripts/paper_figures.py --revision HEAD` resolved inputs to
**ca25892**. Exactly one compute job **3716227.opbs**, host **mg0025**, exit **0**;
recorded start/end 17 seconds apart. Seven figure pairs rendered; new JS figure
visually checked. Independent endpoint/theory/framing reviews, static syntax,
claim-ID/table/link checks, artifact/provenance checks, and staged diff checks passed.
No new experiment, model evaluation, resampling, or pytest run; plotting ran only
on compute. No task deviation. The earlier F2 per-checkpoint export limitation and
pending end-to-end/latent-readout evidence remain explicit.
