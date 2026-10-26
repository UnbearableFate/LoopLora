# Decision record — P2-S (symbolic budget matching), 2026-10-27

Sources: `docs/prereg/P2S.md`, `docs/reports/P2S.md`, `reports/p2s/` (merge 0f84084). Confirmation set: split
`test`, eval seed 888001, 2000 episodes, seeds 17/29/43; 97.5% pooled paired episode-bootstrap CIs. Cost 1.35 node·h.

## 1. Hypothesis
With learned (`joint`) or oracle (`oracle_random`) codes, a single read with more rank (B6) or more state (B7)
cannot reach M1 (r = 32, L = 4) on hops {2, 3}, and weight sharing across reads is not what limits M1 (B8).

## 2. Results (M1 − comparator on hops {2, 3}, pp)
| Endpoint | Writer | Comparator | Estimate [97.5% CI] | Per seed |
| --- | --- | --- | --- | --- |
| E1 rank | joint | B6 r = 64 | +14.9 [13.7, 16.2] | +33.1 / −20.7 / +32.2 |
| E1 rank | joint | B6 r = 128 | −3.2 [−4.7, −1.8] | +7.2 / −21.6 / +4.7 |
| E1 rank | oracle_random | B6 r = 64 / 128 | +34.8 / +34.8 (CI > 0) | consistent |
| E2 state | joint | B7 r_mem = 64 / 128 | −1.5 [−2.9, −0.1] / −2.3 [−3.6, −0.9] | +9.1 / −20.4 / +6.9 (64) |
| E3 sharing | joint | B8 (unshared, L = 4) | −33.7 [−34.9, −32.4] | −26.4 / −52.2 / −22.6 |
| E3 sharing | oracle_random | B8 | −11.4 [−12.8, −10.1] | consistent |

Accuracy by hop 1–4 (seed mean): joint M1@4 58/45/41/41%; joint B8@4 **90/86/68/43%**; joint B6 r128@1
79/47/46/45% (hop 1 plus the terminal-guess level); oracle_random M1@4 50/44/25/7%, B8@4 62/54/38/2%, B6@1
100/0/0/0%. Every joint M1 point is matched by a single-step point of equal or lower latency; oracle M1@4 is not.

## 3. Decision under the registered rule
**Gate 3 (symbolic) fails**: E1 (r = 128) and E2 are ≤ 0 for the joint writer. The depth claim stays restricted to
P1-S's matched-parameter and matched-state comparisons; no efficiency claim. With oracle codes a single read
cannot compose at any tested rank (+34.8 pp for M1), so the capability limit of single reads holds where the codes
leave no shortcut; with unfactorized learned codes, a larger single read exploits the terminal shortcut
(Prop. 10) better than M1 at r = 32 does, and seed 29 is a failed M1 run in every contrast.

## 4. What changes
- The strongest multi-read result so far is B8: step-specific reads of one memory snapshot, trained at L = 4,
  compose hops 1–3 without the hop-1 overshoot. Weight sharing across reads is not required and, at L = 4, costs
  accuracy. Open: is the gap due to sharing or to M1's mixed-depth training (L ∈ {1, 2, 4})?
- The capability comparison must be repeated with shortcut-free learned codes (`joint_fact`), where single reads
  answer only hop 1 (F2: B4 hops ≥ 2 at 1%).

## 5. Next question
P2-S-F1 (`docs/prereg/P2S-F1.md`): at matched training depth, does sharing cost accuracy (M1 trained at L = 4 vs
B8), and with shortcut-free codes do larger single reads still fail to match M1? Its answer fixes the reader
(shared M1 or unshared B8) and training-depth distribution carried into P1-L.

## 6. P2-S-F1 outcome (2026-10-27 02:20 JST, merge f7d2aed; `docs/reports/P2S-F1.md`, 1.10 node·h)
Same confirmation set (seed 888001, reused as registered), 97.5% CIs, hops {2, 3} unless stated.
- Sharing at matched training depth (B8@4 − M1-L4@4): joint +13.2 [12.0, 14.5], joint_fact +5.5 [4.2, 6.8],
  oracle_random +3.2 [2.0, 4.5]; B8 is also better on hop 1 but worse on hops 4–8 for every writer (joint −5.5):
  step-specific reads fit the trained depths better, shared reads extrapolate further.
- Training depth (M1-L4@4 − M1-mixed@4): joint +20.5, joint_fact +3.5, oracle_random +8.2 (all CI > 0); this
  includes separately tuned lrs. About 60% of the joint B8 − M1 gap was training depth (33.7 → 13.2 pp).
- Shortcut-free capability (joint_fact M1-mixed@4 − larger single read): +23.6 to +24.2 pp against B6 r = 64, 128
  and B7 r_mem = 64, 128, every seed positive; the single reads answer 0.2–0.9% of hops 2–3.

Decisions (registered rules): **P1-L carries B8 trained at fixed L = 4**, with M1 trained at L = 4 as the shared
variant; the capability claim (one read cannot replace several, at up to 16× the state or 8.6× the parameters)
holds for oracle codes and for shortcut-free learned codes, not for unfactorized learned codes. Because B8 has
no depth scan, P1-L endpoints must compare separately trained models (B8@4 vs B4, B5 and B3 trained at L = 4);
this is fixed in the P1-L prereg before Pilot-2.
