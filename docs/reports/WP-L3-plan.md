# WP-L3 execution plan (before results)

Task: docs/tasks/WP-L3_p1l_writer_fix.md. Branch wp/llm; merged master8001347.
Read WP-L2 report, theory10.6, P1S decision7, symbolic FactorizedWriter.
No codegraph tool is available in this session; inspect the bounded source tree.

All scientific arms: OA/W0/W1/W2/W3/W4/W5, one run each, seed17, T1 E16,
n_rel4/headsharing.3, rank32/layer14/L1, o_proj input/output, batch32/Q4,
lr.003/S4800; same writer initialization and training episode stream as WP-L2.
Val1000 episodes/seed4242, same one stratified single-hop query; evaluations
at1200/2400/4800; losses every50 steps. No other seeds/splits, T2, reader or
joint experiments. Max4 allocated node-hours; each job1node<=1h.

Implementation choices fixed before scores:
- W0/OA/W1/W2/W3/W5 write at fact_final as Pilot-1. W4 writes once per fact
  at the tail-name token, pairing the immediately preceding token's input as
  key features with the tail token's input as value features. Its A_k is also
  the unchanged reader input map (as symbolic FactorizedWriter); A_v is an
  independent copy of the same initial A. Contextual values can retain head
  information; this is the task's position-based analog, not hard factorization.
- W1/W5 use one pooled per-feature mean/std over fact_final write positions
  and every valid question read position from the first256 training episodes.
  Sample std clamped at1e-4; buffers FP32, saved, never updated by validation.
- W2/W5: CE over16 normalized keys with normalized Qz0 at the final question
  token, temperature.1 and weight1. Target fact determined by query head/relation.
- W3/W5: CE1024 with a training-only affine linear head on every raw V u_j,
  target the fact tail. Weight1; head excluded from inference writer export.
- OA: replace each valid question token's Mq with the queried fact's raw
  written v_j; keep u+rho*ReLU(Ru+Uv_j), rho.25/alpha1. No oracle name code.
- Pre-tail head/relation probes: same frozen o14 features and full/rank32
  method as WP-L2,4096 train episodes/1000 val; fit1200steps, no score tuning.

WP-C5 reached master before any scientific submission and was merged at7f68a3c.
Every arm uses core.scan_final, including W4's packed paired inputs. Its published
fixed-backbone timing (batch32/two processes:286ms/step) sizes the jobs; no extra
accuracy run is used for system tuning. No later backend mixing.

Decision: W0<W1<W4<W3<W2<W5; first finalEM>=.80 wins. If none>=.80 but at
least one>=.30, extend the best final-EM non-oracle arm once (simplicity breaks
exact ties), from the same initialization/data stream, to the longest
measured <=1h budget. If none>=.30, stop P1-L, interpret OA to locate the link.
No other screen/tuning run. Save immutable selection before any extension.

WP-C5 estimate: paired4800step arms ~1400s/job; three paired jobs request40min
each, last single-arm job30min. Tests/probes/audit <=15min requested; conditional
extension <=60min: total requested<=225min and actual expected<3nodeh (cap4).
Extension starts from scratch with the same seed/stream and runs until3450s
elapsed training/evaluation time, preserving final evaluation/export reserve.
