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
