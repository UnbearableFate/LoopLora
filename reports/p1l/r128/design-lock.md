# P1-L r128 design lock — Amendment B3

Locked before any r128 training or timing preflight. Source masterdf21654,
docs/prereg/P1L.md B3; the r32 Pilot-2 remains unchanged and running.
All15 configs in configs/p1l/main-r128 enumerate B8/M1-L4/B4/B5-L4/B3-L4 x
seeds17/29/43. Rank and memory dimension128; all other reader/task settings
match r32 (E12, chains8/4, four relations, oracle seed17, memory-only,
layer14/o_proj, batch64, prefetch2). Oracle values are the symbolic OracleWriter
global random codes at dimension128; pair keys have the same namespace/seed
rule. Both remain fixed fp32 buffers. B0/B1/B2 are reused from r32, not retrained.

First a short training-only timing preflight of all5 variants. S17000 if the
slowest measured run x1.5 fits2hours; otherwise largest feasible S rounded
down to500, flagged for cross-rank step mismatch. Own Pilot-2 on val, seed17,
lr.001/.003, floor(S/3) steps; select full-vocabulary EM on hops1/2/3 at trained
depth, ties lower lr. Commit its lrs/S before15 main runs. Pack sequential runs
only when their measured summed time x1.5 fits2hours. Record every node-hour.

Same confirmation/typed replication, stepwise exact logit replay and frozen
512-val-question J-lens-inspired name-gradient probe as r32. The direction
cache is rank-independent (frozen backbone and same questions) and is shared.
G1/G2 are separate per rank. Secondary contrasts on hops2/3: B8r32@4-B4r128@1,
M1-L4r32@4-B4r128@1; B8r128@4-B8r32@4 descriptively, with actual steps stated.

Confirmation is held until ALL conditions' main runs have finished, including
any learned condition selected by WP-L4. Before its first evaluation, fix
k=number of gated conditions (two oracle ranks plus any learned condition)
and confidence=1-.05/k. No test or confirmation data have been accessed.
