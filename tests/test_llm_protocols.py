"""Integration invariants against the pinned, full 28-layer Qwen3 backbone."""
from __future__ import annotations

import pytest
import torch

from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed
from looplora.llm.inject import attach_adapters, detach_adapters, load_model
from looplora.llm.protocols import (
    answer_scores, encode_support, greedy_answer, incremental_answer_scores,
)


@pytest.fixture(scope="module")
def model():
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(dtype=torch.float32)
    assert len(model.model.layers) == 28
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def adapted(model):
    cfg = AdapterConfig(variant="iterq", d_in=2048, d_out=1024, r=4)
    bank = attach_adapters(model, cfg, seed=107)
    yield model, bank
    model.zero_grad(set_to_none=True)
    detach_adapters(model)


def ids(rows):
    return torch.tensor(rows, dtype=torch.long, device="cuda")


def mask(rows):
    return torch.tensor(rows, dtype=torch.bool, device="cuda")


def nonzero_branch(bank):
    generator = torch.Generator(device="cuda").manual_seed(derive_seed(107, "llm-test-nonzero-B"))
    with torch.no_grad():
        for module in bank.modules.values():
            module.adapter.B.normal_(0, 0.08, generator=generator)


@pytest.mark.parametrize("protocol", ["memory_only", "context_retained"])
def test_zero_branch_matches_unmodified_model(adapted, protocol):
    model, bank = adapted
    support, question, answer = ids([[100, 200, 300], [101, 201, 301]]), ids([[400, 500], [401, 501]]), ids([[600, 700], [601, 701]])
    with torch.no_grad():
        snap = encode_support(model, support, torch.ones_like(support), protocol=protocol)
        observed = answer_scores(model, snap, question, torch.ones_like(question), answer, loop_steps=4)
        first = greedy_answer(model, snap, question, torch.ones_like(question), loop_steps=4)
        detach_adapters(model)
        try:
            prefix = support if protocol == "context_retained" else support[:, :0]
            inputs = torch.cat((prefix, question, answer[:, :-1]), -1)
            expected = model(input_ids=inputs, use_cache=False).logits[:, -answer.shape[1]:]
        finally:
            for layer, module in bank.modules.items():
                model.model.layers[layer].self_attn.o_proj = module
            model.looplora_adapters = bank
    torch.testing.assert_close(observed.logits, expected, atol=3e-5, rtol=2e-5)
    torch.testing.assert_close(first[:, 0], expected[:, 0].argmax(-1))


@pytest.mark.parametrize("protocol", ["memory_only", "context_retained"])
@pytest.mark.parametrize("loop_steps", [0, 1, 4])
def test_teacher_forcing_matches_cached_decode(adapted, protocol, loop_steps):
    model, bank = adapted
    nonzero_branch(bank)
    support = ids([[0, 100, 200, 300], [101, 201, 0, 0]])
    support_mask = mask([[0, 1, 1, 1], [1, 1, 0, 0]])
    question = ids([[0, 400, 500], [401, 501, 0]])
    question_mask = mask([[0, 1, 1], [1, 1, 0]])
    answer = ids([[0, 600, 700], [601, 701, 0]])
    answer_mask = mask([[0, 1, 1], [1, 1, 0]])
    with torch.no_grad():
        snapshot = encode_support(model, support, support_mask, protocol=protocol)
        full = answer_scores(model, snapshot, question, question_mask, answer, answer_mask, loop_steps=loop_steps)
        incremental = incremental_answer_scores(model, snapshot, question, question_mask, answer, answer_mask, loop_steps=loop_steps)
        generated = greedy_answer(model, snapshot, question, question_mask, loop_steps=loop_steps, answer_budget=3)
        generated_scores = answer_scores(model, snapshot, question, question_mask, generated, loop_steps=loop_steps)
    torch.testing.assert_close(full.logits, incremental.logits, atol=4e-5, rtol=3e-5)
    torch.testing.assert_close(full.loss, incremental.loss, atol=3e-6, rtol=2e-6)
    assert full.token_loss[~answer_mask].eq(0).all()
    torch.testing.assert_close(generated, generated_scores.logits.argmax(-1))


@pytest.mark.parametrize("protocol", ["memory_only", "context_retained"])
def test_padding_and_packed_resets_match_separate_episodes(adapted, protocol):
    model, bank = adapted
    nonzero_branch(bank)
    # Two rows carry the same episode with different padding; row 0 also packs
    # a preceding episode that must be erased from both M and backbone attention.
    packed = ids([[910, 920, 0, 100, 200, 300, 0], [0, 0, 100, 200, 300, 0, 0]])
    valid = mask([[1, 1, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0]])
    reset = mask([[1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]])
    question = ids([[0, 400, 500], [400, 500, 0]])
    qmask = mask([[0, 1, 1], [1, 1, 0]])
    answer = ids([[600, 700], [600, 700]])
    with torch.no_grad():
        joint = encode_support(model, packed, valid, reset=reset, protocol=protocol)
        separate = encode_support(model, ids([[100, 200, 300]]), mask([[1, 1, 1]]), protocol=protocol)
        joint_scores = answer_scores(model, joint, question, qmask, answer, loop_steps=4)
        separate_scores = answer_scores(model, separate, ids([[400, 500]]), mask([[1, 1]]), ids([[600, 700]]), loop_steps=4)
    for layer in joint.states:
        torch.testing.assert_close(joint.states[layer], separate.states[layer].expand(2, -1, -1), atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(joint_scores.logits, separate_scores.logits.expand(2, -1, -1), atol=5e-5, rtol=3e-5)


@pytest.mark.parametrize("protocol", ["memory_only", "context_retained"])
def test_snapshot_forks_question_order_and_loops(adapted, protocol):
    model, bank = adapted
    nonzero_branch(bank)
    with torch.no_grad():
        snapshot = encode_support(model, ids([[100, 200, 300]]), mask([[1, 1, 1]]), protocol=protocol)
        original = {layer: state.clone() for layer, state in snapshot.states.items()}
        original_kv = None if snapshot.past_key_values is None else tuple((k.clone(), v.clone()) for k, v in snapshot.past_key_values)
        q, qm, a = ids([[400, 500]]), mask([[1, 1]]), ids([[600, 700]])
        before = answer_scores(model, snapshot, q, qm, a, loop_steps=4)
        answer_scores(model, snapshot, ids([[710, 720]]), qm, a, loop_steps=1)
        greedy_answer(model, snapshot, q, qm, answer_budget=3, loop_steps=0)
        after = answer_scores(model, snapshot, q, qm, a, loop_steps=4)
    torch.testing.assert_close(before.logits, after.logits, atol=0, rtol=0)
    for layer, state in original.items():
        torch.testing.assert_close(state, snapshot.states[layer], atol=0, rtol=0)
        torch.testing.assert_close(state, bank.states()[layer], atol=0, rtol=0)
    if original_kv is not None:
        for (k0, v0), (k1, v1) in zip(original_kv, snapshot.past_key_values):
            torch.testing.assert_close(k0, k1, atol=0, rtol=0)
            torch.testing.assert_close(v0, v1, atol=0, rtol=0)


def test_gradients_cross_frozen_upper_layers(adapted):
    model, bank = adapted
    nonzero_branch(bank)
    snapshot = encode_support(model, ids([[100, 200, 300]]), mask([[1, 1, 1]]))
    scores = answer_scores(model, snapshot, ids([[400, 500]]), mask([[1, 1]]), ids([[600]]), loop_steps=4)
    scores.loss.backward()
    adapter_params = set(bank.modules[14].adapter.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters() if p not in adapter_params)
    for name in ("A", "B", "R", "U", "Q", "K", "V", "a_lambda", "a_beta"):
        param = getattr(bank.modules[14].adapter, name)
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0, name


def test_multiple_layers_detach_restores_projections(model):
    originals = {i: model.model.layers[i].self_attn.o_proj for i in (0, 14, 27)}
    bank = attach_adapters(model, AdapterConfig(variant="iterq", d_in=1, d_out=1, r=4), layers=(0, 14, 27), seed=19)
    try:
        assert set(bank.modules) == {0, 14, 27}
        assert all(next(m.adapter.parameters()).dtype == torch.float32 for m in bank.modules.values())
        with torch.no_grad():
            snap = encode_support(model, ids([[100, 200]]), mask([[1, 1]]))
            assert set(snap.states) == {0, 14, 27}
            assert all(v.dtype == torch.float32 and v.shape == (1, 4, 4) for v in snap.states.values())
            scores = answer_scores(model, snap, ids([[300]]), mask([[1]]), ids([[400]]))
            assert torch.isfinite(scores.loss)
    finally:
        detach_adapters(model)
    assert all(model.model.layers[i].self_attn.o_proj is value for i, value in originals.items())


def test_context_exception_restores_wrappers(adapted):
    _, bank = adapted
    with pytest.raises(RuntimeError, match="deliberate"):
        with bank.use(valid=mask([[1]])):
            raise RuntimeError("deliberate")
    assert all(module.context is None for module in bank.modules.values())


@pytest.mark.parametrize("policy", ["all_tokens", "fact_final"])
def test_projection_and_write_policies_are_independent_of_loop_count(adapted, policy):
    model, bank = adapted
    nonzero_branch(bank)
    module = bank.modules[14]
    support = ids([[0, 100, 200, 300, 0], [101, 201, 301, 0, 0]])
    valid = mask([[0, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    final = mask([[0, 0, 1, 1, 1], [0, 1, 1, 0, 1]])
    permitted = valid if policy == "all_tokens" else valid & final
    captured = []
    hook = module.base.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
    try:
        snapshots = []
        with torch.no_grad():
            for loops in (0, 1, 4, 8):
                captured.clear()
                snapshot = encode_support(model, support, valid, policy, fact_final=final, loop_steps=loops)
                # One batched projection covers every token exactly once, even
                # when the low-rank reader performs eight recurrent updates.
                assert len(captured) == 1 and captured[0].shape[:2] == support.shape
                u = module.adapter.u(captured[0].float())
                expected = bank.zero_states(support.shape[0])[14]
                for step in range(support.shape[1]):
                    expected = module.adapter.write(expected, u[:, step], permitted[:, step])
                torch.testing.assert_close(snapshot.states[14], expected, atol=3e-7, rtol=3e-6)
                snapshots.append(snapshot.states[14])
            for state in snapshots[1:]:
                torch.testing.assert_close(state, snapshots[0], atol=0, rtol=0)
    finally:
        hook.remove()


def test_bfloat16_backbone_keeps_adapter_and_state_fp32():
    model = load_model(dtype=torch.bfloat16)
    bank = attach_adapters(model, AdapterConfig(d_in=2048, d_out=1024, r=4), seed=23)
    try:
        nonzero_branch(bank)
        with torch.no_grad():
            snapshot = encode_support(model, ids([[100, 200, 300]]), mask([[1, 1, 1]]))
        scores = answer_scores(model, snapshot, ids([[400, 500]]), mask([[1, 1]]), ids([[600]]), loop_steps=4)
        scores.loss.backward()
        assert model.dtype == torch.bfloat16
        assert all(p.dtype == torch.float32 for p in bank.modules[14].adapter.parameters())
        assert snapshot.states[14].dtype == torch.float32
        assert scores.logits.dtype == torch.bfloat16 and torch.isfinite(scores.loss)
        assert bank.modules[14].adapter.B.grad.abs().sum() > 0
    finally:
        detach_adapters(model)
        del model
