# WP-L4 active state — Amendment A1

Branch wp/llm only. No Goal. Never import torch/runpytest or validators on login;
all runtime via scripts/submit.py on Miyabi-G compute. Final exact line:
WP-L4 done: docs/reports/P1L-read-fix.md

WP-L3 COMPLETE and announced already (do NOT redo): report P1L-writer-fix.md
commitcd5a45c; W5finalEM.144,OA.703; no writer selected; finalaudit3715599passed,
3.52806nodeh separately. Old A1 wake-upbba58d0f...acknowledged. WP-L4 gate OPEN.

User added WP-L4 Amendment A1 BEFORE ANY scientific submission. Merged master
throughad966aa (contains70d29d4). A1 supersedes old8225step/1h/4nodeh instructions:
now each arm uses largest smoke-sized schedule with1.5x margin in one<=2h job,
--scale-up docs/tasks/WP-L4_p1l_read_fix.md; TOTAL WP-L4<=8nodeh. No jobs canceled.
Decision unchanged: R<Rrho<RK; simplest finalEM>=.80 wins. Else best>=.50 gets
exactly ONE E8 retrain, own smoke-sized A1 budget; otherwise stop. No more screens.

Root /work/xg24i002/x10041/LoopLora/runs/p1l/read-fix-20261027/.
Reports /work/xg24i002/x10041/LoopLora-wt/llm/reports/p1l-read-fix/.
Source frozen for ALL four scientific arms0b03d606f3c752cddfcfd716946dbbc3e36bf025.
Tensor math/data/init unchanged from tested smoke b1c64c1; only scheduler/guard,
extra predeclared OAρ matched check and E8 nested-lock parsing changed.

Frozen release.json (do not overwrite with E8): per-arm steps/primary checkpoints
R23590:[4718,9436,14154,18872,23590]; Rrho24220:[4844,9688,14532,19376,24220];
RK21980:[4396,8792,13188,17584,21980]; OArho27040:[5408,10816,16224,21632,27040].
OAρ additionally evaluates6580 as matched diagnostic vs A1OA.337 (not selection).
Budget includes6full evals forOAρ,5others. Maximal multiple5 with
1.5*(S*own smoke mean step + N_eval*own max100episode-eval time*10)+90<=7200.
Always same seed17/T1E16/n_rel4/headsharing.3/r32/layer14/o_proj/L1/B32Q4/lr.003,
scan_final/prefetch2/val1000seed4242. No test/T2/reader/joint/otherseed.

Completed preparation (all costs count in jobs.json; initial failed tests included):
3715417initialtests2failed/55 (BF16audit export and bad quantized-shuffle assertion);
3715437 corrected55passed; smoke3715461R/Rrho and3715462RK/OArho allpassed;
3715504smoke sample/export/normalization/timing auditpassed. Initial smoke was
10%of8225, reused for expanded A1 from timing ONLY (not quality). Historical
release-before-A1.json retained. Current release is the A1 authority.
3715621mg0837 no-training audit finalA1OA/W5passed (no third A1arm exists),
relativeGrammax<4e-7,repeatedlogitdrift0;128trainingepisodes gradients,optimizer0.
3715659mg0844 A1tests57passed; independently reconstructed all audit coefficients
and EXACTLY matched all original A1predictions/labels; current release frozen.
Audit.csv/age.csv/summary.json now committed; raw vectors NPZ persist under root.

After BOTH scientific jobs complete, verify all four records/status/schedules.
No quality-driven new schedules. Run compute audit scripts/analyze_p1l_read_fix.py
--root ROOT --out REPORTS --release ABS/release.json --e16-only via submit.py
--commit current wp/llm HASH (HEAD resolves master in submit.py; use explicithash).
It audits samples/exports/provenance and locks selection.json. If actionretry_e8,
ONLY that arm may run E8: use scientific CLI --smoke --e8-decision ABS/selection.json
--steps <declared smoke count> --arms <arm> with same prior L3decision/report args;
size own largest A1schedule from timing, before reading E8quality. Write separate
E8release with E:8,amendment:A1,steps_by_arm:{arm:S},science_commit:0b03d60;
then one fresh retrain --e8-decision LOCK --schedule E8RELEASE --out ROOT/e8,
<=1node2h --scale-up task. Do not overwrite E16release. No other E8arm/extension.
Final compute audit omits --e16-only, adds --verify-export and --e8-release ifused.
Reload all exports, predictions identical; head absent at inference. Then collect
ALL PBS costs (including conditional jobs/final audit), <=8nodeh, and write/commit
<=1page+tables docs/reports/P1L-read-fix.md with baseline audit, every quality/loss
curve, matched OAρ comparison, selection and limitations. Do not redo L3work.

SCIENTIFIC SUBMISSIONS (03:53 JST), both source0b03d60 and1node02h via --scale-up:
3715707.opbs R/Rrho: log /work/xg24i002/x10041/LoopLora/logs/pbs/r-l4-R-Rrho-20261027-035343-1708819.log
3715708.opbs RK/OArho: log /work/xg24i002/x10041/LoopLora/logs/pbs/r-l4-RK-OArho-20261027-035344-1708832.log
Specs same stems under logs/pbs/specs/ with.json. Each CLI consumes the frozen
A1release.json and checks code source equality. Verify ALL4 complete after wake;
watching3715708 does not prove3715707 finished. Spent0.178333nodeh before science;
full two jobs4nodeh +conditionalE8job2h still leaves>1.8nodeh below8 for its
smoke/tests/audits. See pending entries in jobs.json. Do not cancel/change jobs
or mix numerical sources. No scientific submission existed before A1 steering.

AMENDMENT A2 ACTIVE (master2844d5e, merged through df21654): user adds rank128
W5,R,OA in parallel; same fixed recipe, own smoke-sized<=2h schedules. DO NOT
READ ANY WP-L4 VALIDATION SCORE until all three new scientific arms are submitted.
No WP-L4 quality score has been read by this worker. Existing r32 jobs/source
remain unchanged. Decision now waits for ALL7 runs: r32 R<Rrho<RK first, then
r128 W5<R; both oracle arms diagnostic/ineligible. E8 applies to best learned
arm across BOTH ranks. Total WP-L4 cap REMOVED; record nodeh perjob, each<=2h.
Prepare separate rank128 release/root directories and re-arm the wake-up after
submission. Older4-only decisions/8nodeh instructions above are superseded.

A2 IMPLEMENTATION SOURCE be07f7fb34c1cdd14b64f65b7f687364fcd6e36f (mergedmaster
df21654/2844d5e). Tests3716019 onmg0851 PASSED69, zero failures/skips,182.079s
pytest; PBS191s. XMLreports/p1l-read-fix/tests-r128.xml. No runtime on login.
Smokes2400steps (10% of provisional24000), eval100 five480step intervals:
3716039.opbs W5/R, log /work/xg24i002/x10041/LoopLora/logs/pbs/s-l4-128-W5-R-20261027-044344-1999311.log
3716040.opbs OA, log /work/xg24i002/x10041/LoopLora/logs/pbs/s-l4-128-OA-20261027-044344-1999323.log
Bothsourcebe07f7f,1node20min, unchangedrank32jobs. AfterBOTHsmokes pass, compute
scripts/prepare_p1l_read_fix_a2.py --root ROOT/smoke-r128 --out ABSREPORTS
verifies allsamples/exports/lossfiniteness without exposing quality; freezes
smoke-r128-timing.json and release-r128.json, largestperarm2hschedule. All
scientificrank128 submissions mustusebe07f7f (sameassmoke), --rank128,
--schedule ABS/release-r128.json --out ROOT/arms-r128, pairsW5/R andsoloOA,
samepriorL3decision/report arguments,2h and --scale-up task. Only AFTERALL3
scientificarms are submitted may ANY WP-L4 quality score be read. Recordthis
boundary explicitly and replace the old rank32 wake-up withoneonthelast128job.

Finalanalyzer isrank-aware, requires --rank128-release ABS/release-r128.json,
waitsforALL7; immutable selection includesrank. ConditionalE8bestacrossboth
ranks usesrankselected, commonA2implementation/source,smokegate supports
--E8-decision lock throughCLI --e8-decision, --rank and --arms. Separate
e8-release.json; scienceROOT/e8/ARM. No furtherwritertrainingotherwise.

CURRENT WAKE-UP (A2 smoke boundary): old3715708 observer46f49560 canceled;
PBS3715707/08 themselves are untouched. New message-mode observer on smoke
3716039: .job-wakeup/debda6d3-3982-4fc0-96d7-068d88d57eba, worker2025081,
5minpoll, thread01a1779d-c123-75a1-9458-ae849ee0fcde, noGoal. Onwake inspect/ack,
verifyBOTHsmokes, run timing/artifact gate ONCOMPUTE, then submit all3rank128
scientificarms BEFORE reading ANY WP-L4 validation quality. Noqualityreadyet.
Usecurrentexplicitc041efef8f3964dd18faf207971647c88e9a4db0 (or laterwp/llm)
for gate/analyzer code; use FROZENbe07f7f for rank128 science matching smoke.
Re-arm on finalnewscientificjob aftersubmission; keep both r32jobs unchanged.
