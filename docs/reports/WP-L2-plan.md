# WP-L2 diagnostic execution plan (before results)

Merged master `d274869` into wp/llm. Scientific scope: seed17, E16 T1,
n_rel4/head_sharing .3, writer M1/r32/layer14/L1, memory-only, val1000 seed4242.
No reader/joint/T2 or test/confirm. All runtime via submit.py on Miyabi-G.
Total allocated budget <=4 node-hours, each job one node <=1h.

D1: 4096 train episodes (all16 facts and four questions each); 1000 val
episodes (all facts, the same one stratified query as Pilot-1). Frozen features,
no adapter. Four sites: o_proj14 input and pre-layer residual inputs7/14/21
(zero-based). Write positions fact_final and tail name. Question positions:
final Answer colon, start-name token and final relation token. Fit separate
full linear / learned rank32 linear softmax classifiers, 1200 AdamW steps,
batch512, lr .01 cosine to .001, train-only standardization, no validation tuning.
Nominal name output1024; both train and ordinary val use the existing seen768
names. Relation output4. Positive-control test checks known recoverable signal.

D2: learned answer code table1024x32 and zero-initialized B1024x32, direct
z=c(answer) at the final read token only; no writer or memory. Alpha1; rho.25
is recorded but bypassed with the reader. S1500, batch32, Q4, lr3e-4/1e-3/3e-3.
Same train episode stream and val queries as Pilot-1. Frozen B0 val NLL/EM
reference. No output-path tuning beyond the task's grid.

D3: fact_final lr3e-4/1e-3, batch32. Two independent processes/GPU. Target
6400 steps (half3200), based on Pilot-1's slowest E16 training time804s/1500
steps (~3431s plus <=60s for three validations and startup). One-hour PBS
walltime, internal elapsed reserve guards the final evaluation/checkpoint.
Log every50 steps; validate at1500,3200,6400. If time guard is reached, preserve
the actual final budget and explicitly report deviation rather than claim6400.

D4 is conditional on D1 or D2/D3 as the task states. Only then add/test the
residual-input option and run the same two D3 configurations. Output remains
at o_proj14. No additional scientific factors are authorized.
