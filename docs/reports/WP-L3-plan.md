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

## Amendment A1 received during the running screen

Master3548d6e replaces the decision branches below.80 above. The original screen
and its source3fbc504 remain unchanged. Keep the existing wake-up armed. After
the screen finishes, merge master and implement the amended rerun on a common
source with WP-C5 scan_final and WP-B4 train.prefetch_workers:2. The custom
writer_fix_run.fit must actually consume prefetched batches; merely setting the
configuration field is insufficient. Validate exact behavior on compute.

If no writer reaches.80 at4800, rerun the two highest-final-EM arms (ties: lower
final main loss), plus OA if absent. Freeze selection before submission. Use the
same initialization/data/seed/batch32/lr.003 and the longest measured schedule
that fits one<=2h job, with five evenly spaced validation checkpoints. Submit
with --scale-up docs/tasks/WP-L3_p1l_writer_fix.md. The total4nodeh cap remains;
include all tests/probes/screen/audits when sizing this job. The original <=1h
single-arm extension and stop-below.30 rules are superseded. Select the simplest
rerun writer reaching.80; otherwise stop and report>=.30 as partial learning.
No validation scores were read by this worker when recording this amendment.

## A1 release, after the completed screen

Screen audited on compute3714886, immutable selection committeda4afc80. Ranked
all seven arms (OA may be among the two under A1's explicit wording): OA221/1000,
W578/1000. Thus rerun OA/W5 together; no third arm is added. OA is diagnostic.
Merged master3548d6e after screen completion. Frozen new numerical source0713cfb
passed68 tests plus exact100-step real-model workers0/2 parameter/loss equality.
W5 prefetched steady mean/p95=.199155/.209157s, OA mean=.166809s. Common32900
steps is the largest multiple100 fitting (7000-100)/slow-p95, reserving100s inside
a7000s fit guard for prefetch/evaluation/close, plus200s PBS setup/export reserve.
One7200s job, two training processes, prefetch_workers2 each; five evaluations
6580/13160/19740/26320/32900. Gate at the final fixed checkpoint as in the original
screen. Fresh same seed/initialization/episode stream; no checkpoint continuation.
Budget spent6114s; full rerun leaves1086s below4nodeh for final audits.
