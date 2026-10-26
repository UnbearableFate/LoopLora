import pytest
import torch

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed


@pytest.mark.parametrize("variant", ["iterq", "fixedq", "unshared"])
@pytest.mark.parametrize("fast", [False, True])
def test_future_tokens_cannot_change_past_outputs_or_states(variant, fast):
    torch.manual_seed(derive_seed(711, "causality", variant, fast))
    adapter = make_adapter(AdapterConfig(
        d_in=4, d_out=5, r=3, r_mem=2, variant=variant,
        gate="input", unshared_steps=2,
    )).double()
    with torch.no_grad():
        adapter.B.normal_()
        adapter.w_lambda.normal_(std=0.2)
        adapter.w_beta.normal_(std=0.2)
    x = torch.randn(2, 8, 4, dtype=torch.float64)
    initial = torch.randn(2, 2, 2, dtype=torch.float64)
    valid = torch.rand(2, 8) > 0.2
    write = torch.rand(2, 8) > 0.3
    reset = torch.rand(2, 8) > 0.8
    process = adapter.process_sequence_fast if fast else adapter.process_sequence
    original, _, _ = process(x, initial, 2, valid=valid, write=write, reset=reset)
    for boundary in (1, 3, 6):
        changed = x.clone()
        changed[:, boundary + 1:] = torch.randn_like(changed[:, boundary + 1:]) * 7
        # Future controls must also be unable to affect the prefix.
        changed_valid, changed_write, changed_reset = valid.clone(), write.clone(), reset.clone()
        changed_valid[:, boundary + 1:] = ~changed_valid[:, boundary + 1:]
        changed_write[:, boundary + 1:] = ~changed_write[:, boundary + 1:]
        changed_reset[:, boundary + 1:] = ~changed_reset[:, boundary + 1:]
        result, _, _ = process(changed, initial, 2, valid=changed_valid, write=changed_write, reset=changed_reset)
        torch.testing.assert_close(result[:, :boundary + 1], original[:, :boundary + 1], rtol=0, atol=0)
        # Inspect full-sequence snapshots to catch future dependence in the scan.
        snapshots, _ = adapter.scan_prewrite(initial, adapter.u(x), valid=valid, write=write, reset=reset)
        other_snapshots, _ = adapter.scan_prewrite(
            initial, adapter.u(changed), valid=changed_valid,
            write=changed_write, reset=changed_reset,
        )
        torch.testing.assert_close(
            snapshots[:, :boundary + 1], other_snapshots[:, :boundary + 1], rtol=0, atol=0,
        )


@pytest.mark.parametrize("fast", [False, True])
def test_current_write_affects_only_later_reads(fast):
    torch.manual_seed(derive_seed(711, "prewrite", fast))
    adapter = make_adapter(AdapterConfig(d_in=3, d_out=3, r=2, variant="iterq")).double()
    with torch.no_grad():
        adapter.B.normal_()
    x = torch.randn(1, 3, 3, dtype=torch.float64)
    initial = torch.randn(1, 2, 2, dtype=torch.float64)
    process = adapter.process_sequence_fast if fast else adapter.process_sequence
    write_on = torch.ones(1, 3, dtype=torch.bool)
    write_off = write_on.clone()
    write_off[:, 1] = False
    on, final_on, _ = process(x, initial, 2, write=write_on)
    off, final_off, _ = process(x, initial, 2, write=write_off)
    torch.testing.assert_close(on[:, :2], off[:, :2], rtol=0, atol=0)
    assert not torch.allclose(final_on, final_off)
