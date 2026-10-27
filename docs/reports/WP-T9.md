# WP-T9 — claim contract and paper changes from the external reviews

Worktree `wp/paper` on **dc659e0**; text only (no code, preregistration or figure changed). No result
is written for P1-L, DPFP or any follow-up.

**[draft.md](../paper/draft.md).** *Appendix C* lists 15 principal claims with comparator (same rank;
"equal parameters" only shared M1-L4 versus B4), writer, split, gate and evidence rows, with reading
rules and a note separating ledger claim IDs from reader names and P1-L amendments. A new §2 subsection
and an overshoot paragraph add related work (two-hop composition, test-time regression, slot/lookup
memories, retrieval in weights, looped overshoot) with 32 references the reviews opened. Scope
statements (closed 1,024-name vocabulary, trained hops/depths, oracle and learned writers separate)
and the five exclusions are in the abstract, §1, §8.4, §9 and Appendix C. B8's overhead (+8.9% r32,
+31.0% r128, +69.2% symbolic) is stated wherever B8 appears; OA is the value/output ceiling. Per-seed
effects sit beside pooled ones (S8, B1, B3, IS1–IS3, H5, HL1, JS); C777/C888 reuse and the S8/H1
rank-steps confound are marked. §6 has the neighbour-confusion placeholder (WP-S9, PM1); §8.4 states
Amendments B6/B7, the run locks and the pre-confirmation evidence; §8.5 adds the memory-use-check TODO.

**Ledgers.** [evidence_symbolic.md](../paper/evidence_symbolic.md): scope section, "(reused)" labels,
B8-overhead tags (FR1 gains the r128 counts), confound flags, row PM1 (open, no value), exclusions.
[evidence_llm.md](../paper/evidence_llm.md): row LP6 with detail tables (training logs, Pilot-2
validation, capacity check), labelled training/validation evidence and not a result of the registered
test; B6/B7 in the amendments table; run-lock sentence reconciled; scope section; OA rename; exclusions.
No status changed; IDs unique.

**Left out.** Per-step-readout citations and rv-fable §4 reframing beyond the five related-work lines
named; works the reviews only saw in search results, and RecLoRA (unopened); per-seed values that exist
only in committed CSVs (S9, B2, B3, B5); a joint_fact B8 parameter total (not recorded, so no percentage);
theory.md, plan and old reports that still say "oracle addressing"; figures and provenance (pinned at
ca25892, statuses unchanged); a chance level for the neighbour-confusion rate itself (it depends on
WP-S9's definition; only the 1/14 accuracy chance is stated).

**Validation (login node, static).** Every `<!-- ID -->` comment resolves to a ledger row; table columns,
links and anchors and `git diff --check` checked; every number added to the draft
occurs in the ledgers or P1L.md B6/B7, exactly or as a rounding. LP6 was re-read with stdlib JSON from the
local training logs, Pilot-2 `metrics.json`, `capacity.json` and the run locks; S8 per-seed Q from
`reports/p1s-f1/gates.json`. No compute job, pytest, model load or figure render.

**Open questions.** (a) The reviews and plan assume no unseen-pool results exist, but the committed
P1-S/F1/F2 `reported_contrasts` exports hold non-gating `dataset=unseen` rows that no ledger audits; the
paper makes no unseen claim. (b) The LP6 logs are local (`runs/` is git-ignored); a small committed summary
would make LP6 reproducible. (c) Amendment B7 says the r32 readers are "at chance", yet some seeds exceed
B3-L4 by up to 1.25 pp (B5-L4 seed29, 1.641% versus 0.391%); the paper says "at or near the memory-free
level". (d) Master's c4d27fa (B7 label time) touches only prereg/plan files; no WP-T9 text cites that time.
No task deviation.
