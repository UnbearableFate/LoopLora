# SLOT-LOOP-v1 S1A decision: ADVANCE

The frozen analysis supports H1 at the registered level. On the 2,000 confirmation episodes, mean EM over the three continuation seeds is 82.266667% for Matrix-L1 and 86.000000% for SoftSlot-L1.

| registered test | observed | passes |
| --- | --- | --- |
| mean SoftSlot − Matrix ≥ +3.0 pp | +3.733333 pp; exact integer check 224 × 100 ≥ 9 × 2000 | yes |
| paired 95% whole-episode interval lower bound > 0 | [2.700000, 4.816667] pp | yes |
| each seed difference > 0 | 17: +3.5 pp; 29: +4.2 pp; 43: +3.5 pp | yes |

The percentile interval uses the registered 10,000 paired resamples and numpy seed 9056755664197379988, with the same episode draws across arms and seeds. One question per episode. Confirmation ordered episode-ID digest: `3cd75bc09be400623e523da95d11f10b`.

Both arms use the same S-RE parent, 6,000 continuation steps, seeds 17/29/43, T1/E16, L1, full-vocabulary EM and all valid question tokens. Writer/shared A/backbone/buffers stayed frozen; Q/R/U/B trained. Equal pilot effort selected Matrix LR 0.001 and SoftSlot LR 0.003; g=1 for both, SoftSlot tau=0.1. Checkpoints were selected solely by dev NLL with the registered tie rule. All six runs were stable; the dev mean EM difference +3.5 pp did not meet the below-3 pp trigger, so no interface ablation ran. No common-fixed-LR comparison ran.

Unseen names are descriptive and do not enter the verdict: Matrix 10.450000%, SoftSlot 13.433333%, difference +2.983333 pp, interval [2.083333, 3.883333]. The positive relative difference coexists with low absolute EM. Its ordered episode-ID digest is `10df09f596dbce1e23567f67ed6be852`.

Scope is one fixed parent writer, three continuation data-order seeds and this registered read interface/task. The result establishes neither interference removal as the causal mechanism nor a loop, multi-hop, capacity or real-text claim. ADVANCE makes the full Matrix/SoftSlot × L1/L4 S1B eligible after the orchestrator registers its addendum and new data namespace. No S1B data were accessed or jobs started.

Opening was authorized at commit `7a113ae7dd03ad957a28388297610980b2e1c382` after independent rv-code job 3729280 reproduced the dev selection byte for byte. Confirmation/unseen jobs 3729313/3729314 used exactly six selected exports plus three locked references. Frozen analysis job 3729369 completed with exit 0 on mg0018 in 7 s. Its source index binds all 21 inputs; every input and output digest was verified before archiving. Exact [frozen verdict](s1a-summary/verdict.md), [summary](s1a-summary/summary.json), [source index](s1a-summary/source_index.json), [artifact inventory](s1a-summary/artifacts.json), and [machine decision](decision_s1a.json) are committed beside this record.

| binding | value |
| --- | --- |
| science commit | `1523b19c7949673ddcd1d89cc1e5fe518736805c` |
| prereg blake2b-128 | `56e8c611c01e0ef9b7ae32db97b0e334` |
| release blake2b-128 | `523d11d57a08a2491893ebe7cedc86a7` |
| open record blake2b-128 | `5a3b037e5aff8f66445df0091398fde0` |
| analysis script blake2b-128 | `843d2ff4d1a03224ccd16e68720ba344` |
| source index blake2b-128 | `5a461a6f91a78ecbb3661bf8f6b7e302` |
| summary blake2b-128 | `4b53592eb8501f5c2d8ed6a65b2df367` |
| verdict blake2b-128 | `ae98f1eddaadee4f3b230820097c31e9` |

Persistent originals are under `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/{s1a-evaluation,s1a-summary}`. No source, frozen analysis, release or preregistration change was needed. Final reporting and archive checks used bounded metadata operations on login; all experiment and statistical computation ran on PBS.
