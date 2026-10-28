# WP-P5L — second-scale confirmation of the P1-L oracle result (sn-p1l; LoopLora-wt/p1l, wp/p1l)

Read first: `docs/tasks/_common_rules.md`, `plan/recurrent_lora_research_plan_2026-10-26.md` §11.1 (P5 "第二规模确认")
and §11.7 (Gate 4), `docs/prereg/P1L.md` (fixed design, endpoints, Amendments B5–B7), `reports/decisions/P1L.md`.

## Why
P5 of the plan is a confirmation at a second model scale. In this project the result that stands is the r = 128
`oracle_random` result of P1-L on Qwen3-0.6B-Base:
- B7 passes;
- M1-L4@4 − B4@1 = +21.55 pp at equal parameters;
- G1-L fails for B8, G2-L passes.

The learned-writer line has stopped (DPFP, `docs/reports/DPFP.md`), so P3 does not run. P5L asks whether that
r = 128 oracle result holds on a larger backbone of the same family. A failure narrows LP4 to 0.6B.

## Design to register (`docs/prereg/P5L.md`; orchestrator registers it on master before any run)
Same as P1-L except:
- Backbone `Qwen/Qwen3-1.7B-Base` @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. It is cached in
  `/work/xg24i002/x10041/hf_home` (28 layers, hidden 2048, 16 heads × 128). The module is at layer 14 `o_proj`, the
  same relative depth; record its in/out sizes.
- Writer `oracle_random` only. Rank r = r_m = 128 only.
- Readers: B3-L4, B4, M1-L4, B8, B5-L4, with seeds 17/29/43; B0 is evaluated (no training). B1 and B2 are not run
  (P1-L reported them but did not use them for any gate); say so.
- S = 17,000 steps for every reader, as P1-L, so that the two scales are comparable.
- lr carried from the P1-L r = 128 lock (`reports/p1l/r128/lr-S-lock.json`): 0.003 for B8, M1-L4, B4 and B5-L4; 0.001 for
  B3-L4. There is no lr pilot, and the prereg states this as a carried choice.
- Confirmation set: split `test` [102000, 104000), disjoint from P1-L's [100000, 102000), with P1-L's eval seed
  and 2000 episodes. Evaluate it only after every training run has finished.
- Endpoints and rules: P1-L's, at 97.5% intervals.
  - B7 rule 1 memory-use check (hop 1).
  - The Amendment-B6 equal-parameter rule for M1-L4@4 − B4@1 on hops {2, 3}: ≥ 3 pp, CI > 0, positive in every seed.
  - G1-L (B8@4 − B4@1) and G2-L.
  - Per-hop tables, parameter counts and latency as P1-L.
- Decision use:
  - B7 fails: "the readers did not learn to use the oracle memory at 1.7B", with no interpretation.
  - B7 passes and B6 passes: the equal-parameter result holds at a second scale (oracle condition, seen names,
    trained hops; B7 rule 3 limits apply).
  - B7 passes and B6 fails: LP4 is narrowed to 0.6B.
- Budget: tests plus smoke ≤ 0.5 node·h. Give the main-training estimate from the smoke before main training. Each
  job ≤ 1 node × 2 h with `--scale-up docs/prereg/P5L.md`.

## Step 1 (now)
1. **Code.** Make the backbone id and revision a config option, with 0.6B as the unchanged default. Prove the off
   path is bitwise unchanged with a test. Derive the module and adapter shapes from the loaded model config.
2. **Fit.** If 17,000 steps of the largest reader cannot fit one 2 h job at 1.7B, check whether the pipeline's
   interrupted-run/resume path is tested and gives the same result as an uninterrupted run. Do not invent a new
   resume; say what exists, then STOP for a decision.
3. **Tests.** Run the tests on compute, including a tiny-model test of the new option.
4. **Smoke.** A 10% smoke at 1.7B for time and memory: all five readers, seed 17, r = 128, validation only, no
   scores read. ≤ 0.5 node·h together with the tests.
5. **Prereg.** Draft `docs/prereg/P5L.md`. End with `STOP:` and give the prereg path, the smoke timings and the
   projected node·h of main training plus confirmation.

Report: `docs/reports/P5L-state.md` (jobs, node·h, verification). Commit on wp/p1l; no merge, no push.
