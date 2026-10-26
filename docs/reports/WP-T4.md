# WP-T4 — symbolic claims and evidence

Merged `master` at `30f86e2` into `wp/theory` (fast-forward; required ancestor present).
Delivered [evidence_symbolic.md](../paper/evidence_symbolic.md): 27 empirical claim rows
with status, conditions, set, intervals or explicit CI unavailability, and committed source
provenance; a ledger for Props.1–11; interpretation discrepancies; and statements to exclude
from the paper. Covered P1-S/F1/F2, P2-S/F1, and both P4-S stopping pilots.

Validation: bounded Python-stdlib CSV/JSON extraction checked the quoted numbers against
committed artifacts; independent read-only subagent audits reviewed each stage and the full
draft. Verified local Markdown links, cited artifact revisions, and table structure.
`git diff --cached --check` passed. No new scientific discrepancy at the prior reports' stated
rounding was found; the evidence document records the material scope corrections, including
candidate versus full-vocabulary accuracy, confirmation reuse, fixed-seed CI scope,
factorization guarantees, training-duration/depth conditions, and failed efficiency/stopping
claims. Review also corrected artifact filenames/selectors and matched the T1/T2 frozen-writer
comparison before completion.

No experiments, pytest, training, evaluation, bootstrap reruns, model loads, or compute jobs
were run (job ID: not applicable), following WP-T4's explicit no-new-runs instruction.
Only existing CSV/JSON exports were inspected on the login/control node. No task deviation;
this report is added under the repository's standard worker completion convention.

Open evidence gaps remain explicit: no factorized accuracy–latency benchmark, no confirmed
rank32 stopping repair, no successful P4-S sufficient contraction certificate, and no
symbolic-to-LLM performance inference. No additional action is required to use this synthesis.
