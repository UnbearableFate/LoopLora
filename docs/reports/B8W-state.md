# WP-L10 (B8W) state — continuations done and verified, authorization committed, no test episode generated (2026-10-29)

Author: sn-p1l (Claude Sonnet 5.5), branch `wp/p1l`. Prereg (registered, master 3c9c396): `docs/prereg/B8W.md`, blake2b-128
`7414b6e0118d8d53a913467d6e7848b2`. No test episode was generated or read; the equality gate and the smoke use the validation split.

## Built (code commit d0b2792, test fix 0a5b2c9)
`src/looplora/llm/b8w.py` (conversion, gate comparison with tolerance 0, decision rule), `continue_training` and the optional
`train.data_start_step` in `llm/train.py` and `llm/config.py` (absent = original stream; resolved P1-L config hashes unchanged,
pinned by test), window `b8w` = 108000, `scripts/run_b8w.py` (`smoke`, `main`), `scripts/make_b8w_lock.py`, ledger study `b8w`,
`tests/test_b8w.py`; `tests/test_p1ls_pipeline.py` now asserts each registered window separately.

## Jobs (`reports/b8w/jobs.json`, `--study b8w`)
| Job | Name | Exit | node·h | Note |
| --- | --- | ---: | ---: | --- |
| 3723982 | `t-b8w-tests` | 1 | 0.088 | 1 failure in my new test (a dict of tensors compared with `==`: test defect); 376 passed |
| 3723983 | `sm-b8w-s17` | 0 | 0.025 | validation-only smoke, seed 17 |
| 3724025 | `t-b8w-tests2` | 0 | 0.089 | same list on 0a5b2c9: 377 tests, no F, E or skip |
Spent 0.20 node·h. Test list: test_b8w, test_p1ls_pipeline, test_p5l, test_llm_prefetch, test_llm_train, test_llm_b8, test_llm_oracle.

## Smoke (`runs/b8w/smoke-100`, seed 17, 100 steps per arm, source d0b2792)
Gate: validation, 64 episodes, depth 4; logits (bf16) `torch.equal`, argmax equal, max |difference| 0.0, control (M1-L4 twice)
bitwise equal, fp32 adapter read equal. Start hash c90d23ff…, converted 1f0c56ae…. Both arms complete, 100 logged steps, lr 0.003,
`data_start_step` 17000, warmup restarted. Step-1 loss identical in both arms; they diverge from step 2.
| Arm | steady step | 100 steps | peak memory |
| --- | ---: | ---: | ---: |
| shared (M1, 442,368 trainable) | 0.1871 s | 31.7 s | 7.69 GB |
| untied (B8, 589,824 trainable) | 0.1939 s | 32.1 s | 7.69 GB |

## Projection
Continuation 6 × 2,000 steps ≈ 0.64 node·h (+ about 0.05 start-up, gate and saves; one job per seed, about 14 min each, request 30 min),
evaluation about 0.12, finalizer 0.02: **about 1.05 node·h in total** including the 0.20 spent (budget 1.0-2.0); a rerun of one continuation job adds about 0.25.

## Continuations (lock approved at 70bb12c; jobs pinned to 70bb12c7a46833f5e436582ae6434d5a47485135, regular-g, 1 node, 30 min)
| Job | Name | Exit | Walltime used | node·h |
| --- | --- | ---: | --- | ---: |
| 3724077 / 3724078 / 3724079 | `tr-b8w-s17/29/43` | 0 / 0 / 0 | 14:13 / 14:07 / 13:56 | 0.237 / 0.235 / 0.232 |
| 3724117 | `t-b8w-pipeline` (`tests/test_b8w_pipeline.py` + `tests/test_b8w.py`: 59 passed, no F, E or skip) | 0 | 0:44 | 0.012 |
Equality gate at full scale, every seed: passed, logits (bf16) `torch.equal`, argmax equal, control equal, max |difference| 0.0
(start hashes c90d23ff / 59e293ee / 37a7baf0; converted 1f0c56ae / 6f62abbe / 9c623e91). Steady step times shared / untied: seed 17
0.2000 / 0.2009 s, seed 29 0.1978 / 0.1982 s, seed 43 0.1915 / 0.1993 s; peak memory 7.69 GB. Total so far (ledger, 7 jobs): 0.92 node·h.
`scripts/make_b8w_manifest.py --source 70bb12c…` verified all six continuations (records complete, 21 logged rows ending at step 2000,
lr 0.003, `data_start_step` 17000, start-checkpoint hashes (shared = locked M1-L4; untied = the gate's converted checkpoint), P1-L stream
`data_seed`, first-step losses identical in both arms, configs identical apart from name and variant, one source, not dirty, oracle writer
hash equal to P1-L's, exit-0 receipts) and wrote `reports/b8w/continuation-manifest.json` and `reports/b8w/confirmation-authorization.json`.
Training loss (logged batch loss, training data only; not a decision input): step 2000 shared / untied: seed 17 2.989 / 3.169,
seed 29 3.781 / 3.595, seed 43 2.800 / 2.656; mean of the last five logged steps (1600-2000): 2.789 / 2.736, 3.423 / 3.301, 2.968 / 2.872.

## Next (after the orchestrator checks the authorization)
Authorization check job (`evaluate_b8w.py --check-only`), then 15 evaluations in one or two jobs, `finalize_b8w.py`, decision record and report.
