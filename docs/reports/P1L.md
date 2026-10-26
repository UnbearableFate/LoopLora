# WP-P1L — in progress

Oracle-random design/all22 run configs were committed before training(ea44bb6).
The orchestrator's J-lens-inspired name-gradient probe A1 amendment was committed before Pilot-2(b5d923c).
Merged master Amendments B1/B2: B2 supersedes the total cap, fixes S17000 and
Pilot-2 5666steps; per-job<=8nodes x2h and maintenance deadline remain binding.
Single GPU process, sequential backbone reuse, prefetch2. Smoke timing audit
3715950 passed; S/packing locked1dc8a76. Pilot-2 jobs3715959–3715965 passed,
5666steps per lr candidate. Selection3716286 onmg0019 fixed M1-L4 lr.003,
all other trained readers lr.001; lr/S/config lockd644b9b precedes main.
R32 main jobs3716288–3716306 submitted (19jobs/22runs), one node each,
maximum request1h55min, B4/B3 pairs sequentially packed.
Confirmation remains held for all main runs and the WP-L4 decision.
Amendment B3 adds oracle r128 (15runs, own Pilot-2, B0–B2 reused); confirmation
is now held for ALL conditions, with confidence1-.05/k fixed beforehand.
B4 removes cross-rank single-read contrasts and adds descriptive adapter counts/
inference timing; wrapper3716084 passed onmg0025. R128 smokes completed,
sizing3716162 locked S17000(e955765), and Pilot-2 jobs3716168–3716172 are
running. R128 pilot watcher remains armed; see P1L-state.md for continuation.
No test or confirmation data have been accessed.

Raw root: /work/xg24i002/x10041/LoopLora/runs/p1l/main/.
Preflight3715826 passed all7 paths onmg0024; peak7.2GB oracle/17.2GB context.
Smoke3715854/3715855 passed (900steps, timing only; B2 accepts this ratio).
Readout/oracle tests3715857 passed33checks onmg0022. J-lens-inspired name-gradient probe trial3715862 passed;
full512-question validation-only gradient cache3715870 completed onmg0865 in932s.
R128 support/exact reload tests3716000 passed34; short preflight3716009 passed,
S17000 feasible. Its1700-step smokes3716020/3716021 passed. Gates/
authorization tests3715889 passed4checks. A validation-only swap-wrapper test
3715928 caught grouping by rewired hop rather than source hop; fixed and passed
3715942 with source episode/query/hop retained and raw counterfactual samples.
Initial
sizing3715842 failed on sparse preflight timing logs; fixed/rechecked3715847.
Validation-only full-model evaluation3716191 caught readout batch-bound
shadowing; regression3716212 passed. Timing acceptance3716213 exposed a2.29e-5
FP32 difference from separately recomputed residuals; actual deployed residual
capture replaces recomputation(2823982), with5 exact replay/capture checks
passed3716225 onmg0023. Full-model recheck3716226 passed onmg0022 with bit-exact
deployed residual capture. A baseline timing submission
3716221 used the central HEAD and failed before runtime; explicit-source
replacement3716223 passed both protocols on validation only.
Reproduction: reports/p1l/README.md. Final ledger/results/limitations pending.
