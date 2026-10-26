# WP-L4 CURRENT — Amendment A2, all science submitted

LATEST 06:22JST: ALL4 scientific jobs FINISH/exit0, ALL7 records complete.
3715707mg0856=4759s,3715708mg0859=4483s,3716159mg0852=4920s,
3716160mg0853=4612s. Frozen schedules/sources matched in records. Still no
scientificquality read by worker; seven-run COMPUTE audit is next.
Newaudit3716541.opbs (sourceb5ff4ee8bf30c976f64b9b3668908a067de6b81b),
log /work/xg24i002/x10041/LoopLora/logs/pbs/a-l4-selection-20261027-061953-2563190.log,
1node5min,regular-g. It FIRST calls analyzer --e16-only tolockall7; THEN
automatically calls fullanalyzer --verify-export IFF action!='retry_e8'. Thus
ifselect/stop and exit0, finalsummary/7exports alreadyverified in SAMEjob;
do not rerun audit unnecessarily. Ifretry_e8, onlye16-summary/selection exist,
proceed singleconditional E8 below. Everycallincludesbothreleaseflags.
At06:21PBS said wouldexceedgroupxg24i002njobsglimit, estimatedstart06:59:49;
leavequeued, do not cancelothers/touchscientificruns. Currentcost20563s=
5.711944nodeh before audit, jobs.json updated withonlyauditpending.
Untracked docs/reports/P1L-read-fix.md is a DRAFT withbaselinetables/protocol,
notcomplete; filldecision/results/validation/costs and removeDraft beforefinal
commit. Currentrawsamples/exports/records retainedunderROOT, nofinalselection
untilcompute audit. Previous3716160wakeab5a16c0 inspected/acknowledged.

CURRENT ARMED WAKE-UP: job3716160.opbs, event
.job-wakeup/ab5a16c0-aff3-4f80-b5b2-603b100b891a, worker2095488, message-mode,
5minpoll; thread01a1779d-c123-75a1-9458-ae849ee0fcde. NoGoal. Previous smoke
eventdebda6d3 acknowledged. Scientificreceipt/state commita5e5aa6; current
analyzercode can be run from this orlaterexplicitwp/llmhash. Firstinspect/ack
new event, then checkALL4jobs/ALL7records before selectionaudit; no newtraining
except the singleconditionalE8 prescribed below. Allscienceimmutablesources.

All three r128scientific arms SUBMITTED04:56:45JST, before ANY WP-L4 quality
score was read; none read yet even after that boundary. Both newjobs started
04:57:31JST, in parallel with unchangedrank32jobs. NoPBSjob was canceled.
WP-L3 already COMPLETE/reportcd5a45c/done-line emitted; do not redo it.

Current scientific jobs (all1node02h via--scale-up task):
- 3715707.opbs r32 R/Rrho, source0b03d606f3c752cddfcfd716946dbbc3e36bf025.
- 3715708.opbs r32 RK/OArho, same0b03d60, unchangedrelease.json.
- 3716159.opbs r128 W5/R, sourcebe07f7fb34c1cdd14b64f65b7f687364fcd6e36f.
- 3716160.opbs r128 OA, samebe07f7f, separate release-r128.json.
Newlogs /work/xg24i002/x10041/LoopLora/logs/pbs/r-l4-128-W5-R-20261027-045645-2066285.log
and r-l4-128-OA-20261027-045645-2066310.log. Check ALL4 terminal and ALL7 records
complete after wake. Do not change runningjobs or their frozen sources.

R128tests69passed3716019mg0851; smokes3716039mg0855/3716040mg0897 exit0,
2400steps/100evalepisodes. Compute gate3716145mg0842 passed: allsamples/configs,
finite diagnostics/objectives/gradients, sharednormalization, 128x128 memory,
head-free reload identical100predictions. NO quality exposed. Gatecode7e776bd;
release committed2fd95f9. Allsmoke and scientificr128MLsourcebe07f7f.
Maximal5-step-resolution budgets,1.5*(S*ownmeanstep+5*ownmaxeval100x10)+90<=7200:
W5 S24635 checkpoints4927/9854/14781/19708/24635;
R S24645 checkpoints4929/9858/14787/19716/24645;
OA S31175 checkpoints6235/12470/18705/24940/31175.
Smoke was10% of provisional24000; actualfractions9.74%W5/R,7.70%OA after
maximal sizing; recorded explicitly, no quality-guided settings. Fixedlr.003
at r128 withoutsearch is flagged. OA is learned-value oracleaddressing,
not WP-C7 externaloracle_random. TOTALWP-L4capremoved; recordALLactualjobcosts.
Completedcostpre-current4sciencejobs0.496944nodeh; jobs.json updated/pending.

After ALL7E16 complete: onCOMPUTE via submit.py --commit CURRENTexplicitwp/llm
hash (HEAD resolves master), scripts/analyze_p1l_read_fix.py --root ROOT
--out REPORTS --release REPORTS/release.json --rank128-release
REPORTS/release-r128.json --e16-only. Auditsallconfig/source/samples/metrics,
fixedbudgets, exportheadcounts and allcurves; locksimmutable selection.json.
Candidate order r32R<Rrho<RK then r128W5<R. SimplestfinalEM>=.80 selected.
Bothoracles ineligible. Ifnone>=.80 and bestlearned>=.50, exactlyONE E8retrain
at bestarm'srank (tiescandidateorder), same seed17/lr/objective, ownsmoke-sized
2hrelease. Usebe07f7f scientificimplementation for E8 at EITHERrank. E8CLI:
--rank RANK --arms ARM --smoke --steps2400 --e8-decision REPORTS/selection.json
--out ROOT/smoke-e8 pluspriorL3arguments. Thencompute
scripts/prepare_p1l_read_fix_a2.py --root ROOT/smoke-e8 --out REPORTS --rank RANK
--arms ARM --steps2400 --e8-decision REPORTS/selection.json writes e8-release.json,
ONLYtiming/artifacts exposed. ScientificE8 uses --schedule REPORTS/e8-release.json
--e8-decision REPORTS/selection.json --rank RANK --arms ARM --out ROOT/e8,
samepriorL3arguments,be07source,--scale-up task,<=2h. No further writertraining.
Finalanalyzer omit--e16-only add--verify-export (+--e8-release ifused); all7(+E8)
exports must reproduce1000predictions head-free. UpdateALLPBSactualcosts. Write
<=1page+tables docs/reports/P1L-read-fix.md, commitonwp/llm, exactdone-line.

ROOT=/work/xg24i002/x10041/LoopLora/runs/p1l/read-fix-20261027
REPORTS=/work/xg24i002/x10041/LoopLora-wt/llm/reports/p1l-read-fix
PriorL3decisionREPORTS/../p1l-writer-fix/summary.json; priorreport
/work/xg24i002/x10041/LoopLora-wt/llm/docs/reports/P1L-writer-fix.md.
Never importtorch/runpytest/validators onlogin. NoGoal. Olderhistory below
is superseded where A2 differs; no totalnodeh cap,7arms/bothranks forselection.

# Historical A1 and A2 preparation

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
