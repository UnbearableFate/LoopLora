# WP-P1L — in progress

Oracle-random design/all22 run configs were committed before training(ea44bb6).
The orchestrator's J-lens A1 amendment was committed before Pilot-2(b5d923c).
Merged master Amendments B1/B2: B2 supersedes the total cap, fixes S17000 and
Pilot-2 5666steps; per-job<=8nodes x2h and maintenance deadline remain binding.
Single GPU process, sequential backbone reuse, prefetch2. Lrs and main remain
pending. Confirmation remains held for all main runs and the WP-L4 decision.
No test or confirmation data have been accessed.

Raw root: /work/xg24i002/x10041/LoopLora/runs/p1l/main/.
Preflight3715826 passed all7 paths onmg0024; peak7.2GB oracle/17.2GB context.
Smoke3715854/3715855 running (900steps, timing only; B2 accepts this ratio).
Readout/oracle tests3715857 passed33checks onmg0022. J-lens trial3715862 passed;
full512-question validation-only gradient cache3715870 is running. Initial
sizing3715842 failed on sparse preflight timing logs; fixed/rechecked3715847.
Reproduction: reports/p1l/README.md. Final ledger/results/limitations pending.
