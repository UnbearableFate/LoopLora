# WP-L4 complete — 2026-10-27

Branch wp/llm only. Report docs/reports/P1L-read-fix.md; final audit passed.
No Goal used. All task-owned PBS jobs ended; all wake-ups acknowledged/canceled.
No further WP-L4 training or compute is required. Do not redo WP-L3/WP-L4.

Decision: STOP P1-L, no selected writer. Final learned EM r32R0.120,
Rrho0.122,RK0.120,r128W50.299,r128R0.335. Best0.335 is below0.50 E8 trigger
and0.80 writer threshold. No E8 run occurred. Oracle r32OArho0.747/r128OA0.840
are diagnostic/ineligible. Immutable reports/p1l-read-fix/selection.json.

All7 scientific records complete; all4 sciencejobs exit0, within1node2h:
3715707mg0856 R/Rrho=4759s;3715708mg0859 RK/OArho=4483s (bothr32);
3716159mg0852 W5/R=4920s;3716160mg0853 OA=4612s (bothr128).
R32source0b03d606f3c752cddfcfd716946dbbc3e36bf025/release.json;
r128sourcebe07f7fb34c1cdd14b64f65b7f687364fcd6e36f/release-r128.json.
Allr128arms submitted04:56:45JST before ANY WP-L4 validation score was read;
A2 merged throughdf21654/master2844d5e, rank32 jobs untouched, nojobs canceled.
SameT1E16/seed17/nrel4/headsharing.3/layer14o_proj/L1/B32Q4/lr.003,
scan_final/prefetch2,val1000seed4242. Rank128LR unchanged/no search flagged.
No scientificT2/reader/joint/test/confirmation/otherseeds.

Final audit3716541mg0859, sourceb5ff4ee,79s, exit0: every samplemetric/pairedlabel,
config/source/maximal2hbudget verified, all7 head-free exports reloaded with
identical1000 predictions each. Full summary.json ande16-summary.json, curves,
CSVs, selection lock and all7 export-check samples/metrics retained under
reports/p1l-read-fix. No audit rerun is needed. Compute finished06:47JST before
user's08:00 normal-operation cutoff. Wake8b05200b-6566-43b7-8b35-67220d527791
inspected andACKNOWLEDGED. Prior auditwait wasprojectnjobsglimit, resolved.

Tests corrected55/57A1/69A2 passed; initial2failuretestjob counted. All baseline
and smoke gates passed; detailed evidence/commands/provenance in README.md,
WP-L4-plan.md and git history. No-training L3A1 audit onlyOA/W5 (thirdarmnever
existed): zerooptimizersteps, samepriorpredictions, repeatdrift0, maxGram<4e-7,
allcoefficientvectors independently reconstructed, actualmemoryzero/shuffle.
Costs ALL16 jobs20642s=5.7338888889nodeh; jobs.json complete,pendingempty.
A2 removed studytotalcap; everyjob1node<=2h. No torch/pytest/validators onlogin.

Persistentroot /work/xg24i002/x10041/LoopLora/runs/p1l/read-fix-20261027 contains
all scientific records/checkpoints/writerexports/percheckpoint samples, original
andrank128 smokes, no-training auditdensevectors. Preserve these artifacts.
Report contains everycheckpoint, matchedOArho6580vspriorOA.337, all costs,
objectives/extra supervision, diagnosticinterpretation, single-seedlimitations.
Report and evidence are committed on wp/llm by the completion commit containing
this file. No remaining WP-L4 work. Final EXACTline:
WP-L4 done: docs/reports/P1L-read-fix.md.
