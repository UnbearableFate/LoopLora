# P1-L budget reconciliation needed before Pilot-2

RESOLVED by master Amendments B1 then B2. B2 drops the total cap and fixes
S17000/Pilot5666. This file preserves the pre-score question and evidence.

Design commit ea44bb6; preflight3715826 completed successfully on mg0024.
All7 actual train paths fit memory (oracle readers~7.2GB; B1/B2~17.2GB).
The longest path is B1 context-retained, approximately.28s per step; other
paths are approximately.13-.20s per step after warmup. A compute-only sizing
job is producing exact steady-state extrapolations in reports/p1l/preflight-sizing.json.

At the preregistered largest S fitting2h/1.5, the21 main trainings plus14
Pilot-2 trainings alone are projected above16node-hours, before smoke, startup,
confirmation or J-lens-inspired name-gradient probe. The two constraints therefore require a decision.

Requested amendment (before any Pilot-2 score): choose the largest common S
that satisfies BOTH the per-job rule and the total16node-hour limit, with a
declared reserve for confirmation/readout/startup and a possible second writer.
Alternatively the orchestrator may explicitly amend the total cap. I retain
the cap and will not silently lower the preregistered S rule or spend beyond it.
No Pilot-2, main or confirmation run has started; timing-only preparation
and readout implementation continue independently.
