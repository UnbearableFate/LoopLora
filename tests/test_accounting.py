import pytest
import torch

from looplora.core.accounting import macs_per_token, param_count, state_bytes
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed


@pytest.mark.parametrize("variant", ["lora", "parallel", "loop", "iterq", "fixedq", "unshared", "double_ba"])
@pytest.mark.parametrize("gate", ["scalar", "input"])
@pytest.mark.parametrize("r_mem", [3, 5])
def test_parameter_count_matches_all_trainable_tensors(variant, gate, r_mem):
    torch.manual_seed(derive_seed(17, "accounting", variant, gate, r_mem))
    cfg = AdapterConfig(d_in=7, d_out=7, r=3, r_mem=r_mem, variant=variant, gate=gate, unshared_steps=4)
    adapter = make_adapter(cfg)
    assert param_count(cfg) == sum(p.numel() for p in adapter.parameters() if p.requires_grad)


@pytest.mark.parametrize("form, extra_gate", [("skip", 0), ("gated", 1)])
def test_double_ba_reuses_bottleneck_parameters(form, extra_gate):
    cfg = AdapterConfig(d_in=7, d_out=7, r=3, variant="double_ba", double_ba_form=form)
    adapter = make_adapter(cfg)
    assert param_count(cfg) == 42 + extra_gate
    assert param_count(cfg) == sum(p.numel() for p in adapter.parameters())
    assert macs_per_token(cfg, 4) == 84


@pytest.mark.parametrize("r_mem", [3, 5])
@pytest.mark.parametrize("loop_steps", [0, 1, 4])
@pytest.mark.parametrize("writes", [False, True, 2])
@pytest.mark.parametrize("gate", ["scalar", "input"])
def test_iterative_macs_follow_notation(r_mem, loop_steps, writes, gate):
    cfg = AdapterConfig(d_in=7, d_out=11, r=3, r_mem=r_mem, gate=gate)
    bottleneck = 3 * (7 + 11)
    reader = loop_steps * (3**2 + 2 * 3 * r_mem + r_mem**2)
    writer = 2 * 3 * r_mem + 2 * r_mem**2 + (6 if gate == "input" else 0)
    assert macs_per_token(cfg, loop_steps, writes) == bottleneck + reader + int(writes) * writer


@pytest.mark.parametrize("loop_steps", [0, 1, 4])
def test_fixed_query_caches_query_memory_and_readout(loop_steps):
    cfg = AdapterConfig(d_in=7, d_out=11, r=3, r_mem=5, variant="fixedq")
    expected = 54 + 9 * loop_steps + (55 if loop_steps else 0)
    assert macs_per_token(cfg, loop_steps, writes=False) == expected
    assert macs_per_token(cfg, loop_steps, writes=True) == expected + 80


def test_unshared_only_duplicates_reader_parameters_and_uses_configured_steps():
    cfg = AdapterConfig(d_in=7, d_out=11, r=3, r_mem=5, variant="unshared", unshared_steps=4, gate="input")
    assert param_count(cfg) == 54 + 4 * (9 + 30) + 30 + 2 + 6
    assert macs_per_token(cfg, 4) == 54 + 4 * (9 + 30 + 25) + 30 + 50 + 6
    with pytest.raises(ValueError, match="unshared"):
        macs_per_token(cfg, 3)


@pytest.mark.parametrize("variant", ["lora", "parallel", "loop"])
def test_no_memory_variants_have_no_writer_cost(variant):
    cfg = AdapterConfig(d_in=7, d_out=11, r=3, variant=variant)
    expected = 54 + (4 * 9 if variant == "loop" else 0)
    assert macs_per_token(cfg, 4, writes=10) == expected


@pytest.mark.parametrize("variant", ["lora", "parallel", "loop", "iterq", "fixedq", "unshared", "double_ba"])
@pytest.mark.parametrize("dtype, itemsize", [(torch.float64, 8), (torch.float32, 4), (torch.bfloat16, 2)])
def test_state_storage_matches_persistent_state(variant, dtype, itemsize):
    cfg = AdapterConfig(d_in=7, d_out=7, r=3, r_mem=5, variant=variant)
    expected = 4 * 2 * 25 * itemsize if variant in {"iterq", "fixedq", "unshared"} else 0
    assert state_bytes(cfg, batch=2, n_modules=4, dtype=dtype) == expected
    assert state_bytes(cfg, batch=0, n_modules=4, dtype=dtype) == 0


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_counts_are_rejected(bad):
    cfg = AdapterConfig(d_in=7, d_out=11, r=3)
    with pytest.raises(ValueError):
        macs_per_token(cfg, bad)
    with pytest.raises(ValueError):
        state_bytes(cfg, bad, 1, torch.float32)
    with pytest.raises(ValueError):
        state_bytes(cfg, 1, bad, torch.float32)


@pytest.mark.parametrize("bad", [-1, 1.5])
def test_invalid_write_multiplicity_is_rejected(bad):
    cfg = AdapterConfig(d_in=7, d_out=11, r=3)
    with pytest.raises(ValueError):
        macs_per_token(cfg, 1, writes=bad)
