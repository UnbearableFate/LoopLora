"""Full-model evidence for truncating memory-only support at the writer."""
import pytest
import torch

from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed
from looplora.llm.inject import attach_adapters, detach_adapters, load_model
from looplora.llm.protocols import answer_scores, encode_support


@pytest.fixture(scope="module")
def backbone():
    model = load_model(dtype=torch.float32)
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.mark.parametrize("policy", ["all_tokens", "fact_final"])
def test_truncated_support_matches_state_logits_and_gradients(backbone, policy):
    model = backbone
    bank = attach_adapters(model, AdapterConfig(d_in=2048, d_out=1024, r=8), seed=71)
    generator = torch.Generator(device="cuda").manual_seed(derive_seed(71, "support-test-B"))
    with torch.no_grad():
        bank.modules[14].adapter.B.normal_(0, .03, generator=generator)
    support = torch.tensor([[0, 100, 200, 300, 400], [101, 201, 301, 0, 0]], device="cuda")
    valid = support.ne(0)
    final = torch.tensor([[0, 0, 1, 0, 1], [0, 1, 1, 0, 1]], device="cuda", dtype=torch.bool)
    reset = torch.tensor([[0, 0, 0, 1, 0], [1, 0, 0, 0, 0]], device="cuda", dtype=torch.bool)
    question = torch.tensor([[501, 502], [503, 504]], device="cuda")
    answer = torch.tensor([[601], [602]], device="cuda")
    upper_calls = []
    hook = model.model.layers[15].register_forward_pre_hook(lambda *args: upper_calls.append(1))
    results = []
    try:
        for truncate in (False, True):
            model.zero_grad(set_to_none=True)
            upper_calls.clear()
            snapshot = encode_support(model, support, valid, policy, fact_final=final,
                                      reset=reset, loop_steps=4, truncate=truncate)
            assert len(upper_calls) == (0 if truncate else 1)
            assert snapshot.past_key_values is None
            scores = answer_scores(model, snapshot, question, torch.ones_like(question), answer, loop_steps=4)
            scores.loss.backward()
            gradients = {name: p.grad.clone() for name, p in bank.modules[14].adapter.named_parameters()
                         if p.grad is not None}
            results.append((snapshot.states[14].detach().clone(), scores.logits.detach().clone(), gradients))
        for position in (0, 1):
            torch.testing.assert_close(results[0][position], results[1][position], atol=0, rtol=0)
        assert results[0][2].keys() == results[1][2].keys()
        for name in results[0][2]:
            # Online evaluates scalar gate sigmoids per token; write_only's
            # scan evaluates them once and sums their gradients first. The
            # resulting FP32 reduction order can differ by a few ulps.
            torch.testing.assert_close(results[0][2][name], results[1][2][name], atol=1e-12, rtol=1e-6)
        for name in ("A", "K", "V", "a_lambda", "a_beta"):
            assert results[1][2][name].abs().sum() > 0
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
        detach_adapters(model)


def test_truncation_rejects_retained_kv_and_multiple_modules(backbone):
    model = backbone
    config = AdapterConfig(d_in=2048, d_out=1024, r=4)
    tokens = torch.tensor([[100, 200]], device="cuda")
    attach_adapters(model, config)
    try:
        with pytest.raises(ValueError, match="memory_only"):
            encode_support(model, tokens, torch.ones_like(tokens), protocol="context_retained", truncate=True)
    finally:
        detach_adapters(model)
    attach_adapters(model, config, layers=(14, 20))
    try:
        with pytest.raises(ValueError, match="exactly one"):
            encode_support(model, tokens, torch.ones_like(tokens), truncate=True)
    finally:
        detach_adapters(model)
