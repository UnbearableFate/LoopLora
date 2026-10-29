# WP-L10 (B8W) state — step 2 up to the checkpoint lock (2026-10-29)

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

## Next (after the orchestrator checks `reports/b8w/checkpoint-lock.json`)
Three continuation jobs (`scripts/run_b8w.py main --seed S`), verification, authorization, `scripts/evaluate_b8w.py` and `scripts/finalize_b8w.py` (not written yet).
