"""SLOT-LOOP-v1: the memory states and read backends (docs/prereg/SLOT-LOOP-v1.md, plan 11.2 and 12).

Compute only (imports torch). Pure tensor tests of `core/memory_backend.py` and the scoped diagnostic read of `llm/slot_read.py`:
the matrix read is the original operation bitwise; slot reads against hand-computed cases; empty rows, padding slots, ties;
owner-indexed reads with sentinels; capacity, byte accounting and read-only isolation.
"""
import math

import pytest
import torch

from looplora.core.memory_backend import (CapacityError, HardSlotBackend, MatrixBackend, MatrixState, MemoryRead,
                                          SlotState, SoftSlotBackend, build_slots, persistent_bytes, reset_rows,
                                          select_rows, state_digest)
from looplora.llm.slot_read import ReadSpec, ScopedRead

DEVICES = ['cpu', 'cuda']


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def slots(keys, values, capacity=None, valid=None):
    return build_slots(keys, values, capacity or keys.shape[1], valid)


# ------------------------------------------------------------------------------------------------- the matrix backend

@pytest.mark.parametrize('device', DEVICES)
def test_matrix_read_is_the_original_read_bitwise(device):
    """The old path: the state repeated per question (repeat_interleave), rows selected at a token, matmul with the query."""
    torch.manual_seed(0)
    B, Q, dv, dk = 5, 4, 8, 12
    matrix = torch.randn(B, dv, dk, device=device)
    query = torch.randn(B * Q, dk, device=device)
    indices = torch.tensor([0, 3, 4, 7, 8, 15, 19], device=device)
    old = torch.matmul(matrix.repeat_interleave(Q, dim=0).index_select(0, indices), query.index_select(0, indices).unsqueeze(-1)).squeeze(-1)
    read = MemoryRead.bind(MatrixBackend(), MatrixState(matrix)).expand(Q).select(indices)
    new = read.read(query.index_select(0, indices))
    assert torch.equal(old, new)
    scaled = MemoryRead.bind(MatrixBackend(2.0), MatrixState(matrix)).expand(Q).select(indices).read(query.index_select(0, indices))
    assert torch.equal(scaled, old * 2.0)


# ---------------------------------------------------------------------------------------------------- slot reads, by hand

@pytest.mark.parametrize('device', DEVICES)
def test_soft_read_against_a_hand_computed_case(device):
    keys = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0.6, 0.8, 0.]]], device=device)
    values = torch.tensor([[[1., 10.], [2., 20.], [3., 30.]]], device=device)
    query = torch.tensor([[0.6, 0.8, 0.]], device=device)
    tau, gain = 0.5, 1.5
    scores = torch.tensor([0.6, 0.8, 1.0], dtype=torch.float64)
    weights = torch.softmax(scores / tau, 0)
    expected = gain * (weights[:, None] * values[0].double().cpu()).sum(0)
    owner = torch.zeros(1, dtype=torch.long, device=device)
    got = SoftSlotBackend(tau, gain).read(slots(keys, values), query, owner)
    assert torch.allclose(got.double().cpu()[0], expected, atol=1e-6)
    assert torch.allclose(SoftSlotBackend(tau).weights(slots(keys, values), query, owner).double().cpu()[0], weights, atol=1e-6)
    # Hard picks the largest k.q, here the third slot, with the gain
    assert torch.equal(HardSlotBackend(gain).read(slots(keys, values), query, owner), gain * values[:, 2])


@pytest.mark.parametrize('device', DEVICES)
def test_a_small_temperature_approaches_the_hard_read(device):
    torch.manual_seed(1)
    keys, values = unit(torch.rand(64, 6, 9, device=device)), torch.randn(64, 6, 5, device=device)
    query = unit(torch.rand(64, 9, device=device))
    owner = torch.arange(64, device=device)
    state = slots(keys, values)
    top = torch.einsum('bsk,bk->bs', keys, query).topk(2, dim=-1).values
    clear = (top[:, 0] - top[:, 1]) > 0.02  # rows with a clear winner
    assert int(clear.sum()) > 32
    soft, hard = SoftSlotBackend(1e-4).read(state, query, owner), HardSlotBackend().read(state, query, owner)
    assert torch.allclose(soft[clear], hard[clear], atol=1e-4)
    assert torch.allclose(SoftSlotBackend(1e9).read(state, query, owner), values.mean(1), atol=1e-4)  # uniform in the limit


@pytest.mark.parametrize('device', DEVICES)
def test_empty_rows_read_zero_without_nan_and_padding_slots_are_inert(device):
    torch.manual_seed(2)
    B, S, dk, dv = 3, 6, 7, 4
    keys, values = unit(torch.rand(B, S, dk, device=device)), torch.randn(B, S, dv, device=device)
    valid = torch.zeros(B, S, dtype=torch.bool, device=device)
    valid[0, :4] = True
    valid[1, :6] = True  # row 2 stays empty
    state = SlotState(keys, values, valid, valid.sum(1))
    query = unit(torch.rand(B, dk, device=device)).requires_grad_()
    owner = torch.arange(B, device=device)
    for backend in (SoftSlotBackend(0.3, 1.0), SoftSlotBackend(0.3, 1.0, decay=0.9), HardSlotBackend()):
        out = backend.read(state, query, owner)
        assert bool(torch.isfinite(out).all()) and bool((out[2] == 0).all()) and bool((out[0] != 0).any())
        if isinstance(backend, SoftSlotBackend):  # the hard read selects, it has no gradient to the query
            out.sum().backward()
            assert bool(torch.isfinite(query.grad).all()) and bool((query.grad[2] == 0).all()) and bool((query.grad[:2] != 0).any())
            query.grad = None
        # slots beyond the valid prefix cannot matter, whatever they hold
        junk = SlotState(torch.where(valid[..., None], keys, torch.full_like(keys, 7.)),
                         torch.where(valid[..., None], values, torch.full_like(values, -9.)), valid, state.count)
        assert torch.equal(out, backend.read(junk, query.detach(), owner))


@pytest.mark.parametrize('device', DEVICES)
def test_hard_ties_take_the_lowest_slot(device):
    keys = torch.zeros(1, 4, 3, device=device)
    keys[0, :, 0] = torch.tensor([0.5, 1.0, 1.0, 0.2])  # slots 1 and 2 tie for the largest score
    values = torch.arange(4., device=device).reshape(1, 4, 1).expand(1, 4, 2).contiguous() + 10
    query, owner = torch.tensor([[1., 0., 0.]], device=device), torch.zeros(1, dtype=torch.long, device=device)
    chosen, nonempty = HardSlotBackend().index(slots(keys, values), query, owner)
    assert chosen.tolist() == [1] and nonempty.tolist() == [True]
    assert HardSlotBackend().read(slots(keys, values), query, owner).tolist() == [[11., 11.]]


@pytest.mark.parametrize('device', DEVICES)
def test_the_recency_weight_is_log_lambda_to_the_age(device):
    torch.manual_seed(3)
    keys, values = unit(torch.rand(2, 5, 6, device=device)), torch.randn(2, 5, 3, device=device)
    query, owner = unit(torch.rand(2, 6, device=device)), torch.arange(2, device=device)
    lam, tau = 0.7, 0.4
    state = slots(keys, values)
    ages = torch.arange(4, -1, -1, device=device, dtype=torch.float32)
    logits = torch.einsum('bsk,bk->bs', keys, query) / tau + ages * math.log(lam)
    expected = torch.einsum('bs,bsv->bv', torch.softmax(logits, -1), values)
    assert torch.allclose(SoftSlotBackend(tau, 1.0, decay=lam).read(state, query, owner), expected, atol=1e-6)
    # lam = 1 adds nothing
    assert torch.allclose(SoftSlotBackend(tau, 1.0, decay=1.0).read(state, query, owner), SoftSlotBackend(tau).read(state, query, owner))


# -------------------------------------------------------------------------------------------------------- owner and mask

@pytest.mark.parametrize('device', DEVICES)
def test_owner_indexed_reads_see_only_their_episode_under_reorder_expansion_and_padding(device):
    """Two episodes with sentinel memories; every flattened question reads exactly its owner's memory."""
    dk, dv, S = 5, 3, 4
    keys = torch.zeros(2, S, dk, device=device)
    keys[0, :, 0] = torch.tensor([1., .5, .2, .1])  # episode 0 prefers slot 0 for a query on axis 0
    keys[1, :, 1] = torch.tensor([.1, .2, 1., .5])  # episode 1 prefers slot 2 for a query on axis 1
    values = torch.stack([torch.full((S, dv), 1000., device=device) + torch.arange(S, device=device)[:, None],
                          torch.full((S, dv), 2000., device=device) + torch.arange(S, device=device)[:, None]])
    state = slots(keys, values)
    query = torch.zeros(7, dk, device=device)
    owner = torch.tensor([0, 1, 1, 0, 0, 1, 0], device=device)
    query[owner == 0, 0], query[owner == 1, 1] = 1., 1.
    got = HardSlotBackend().read(state, query, owner)
    assert got[owner == 0].tolist() == [[1000.] * dv] * 4 and got[owner == 1].tolist() == [[2002.] * dv] * 3
    # reordering the episodes of the state and renaming the owners changes nothing
    swapped = select_rows(state, torch.tensor([1, 0], device=device))
    assert torch.equal(HardSlotBackend().read(swapped, query, 1 - owner), got)
    # changing the other episode's memory leaves every row of this owner untouched (bitwise)
    other = SlotState(keys.clone(), values.clone(), state.valid, state.count)
    other.values[1] += 77.
    for backend in (SoftSlotBackend(0.5), HardSlotBackend()):
        assert torch.equal(backend.read(state, query, owner)[owner == 0], backend.read(other, query, owner)[owner == 0])
    # the matrix backend: question expansion only repeats owners, padding rows (owner of a padded question) read their own row
    matrix = MatrixState(torch.stack([torch.eye(dv, dk, device=device) * 1., torch.eye(dv, dk, device=device) * 5.]))
    q = torch.ones(7, dk, device=device)
    out = MemoryRead.bind(MatrixBackend(), matrix, torch.tensor([0, 1], device=device)).expand(4).select(torch.tensor([0, 5, 7, 2], device=device)).read(q[:4])
    assert out[:, 0].tolist() == [1., 5., 5., 1.]


def test_memory_read_rejects_a_bad_owner_and_a_wrong_dtype():
    state = MatrixState(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match='outside'):
        MemoryRead.bind(MatrixBackend(), state, torch.tensor([0, 2]))
    with pytest.raises(ValueError, match='int64'):
        MemoryRead.bind(MatrixBackend(), state, torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(TypeError):
        MemoryRead.bind(SoftSlotBackend(1.0), state)
    with pytest.raises(ValueError, match='fp32'):
        MemoryRead.bind(MatrixBackend(), MatrixState(torch.zeros(2, 3, 4, dtype=torch.float64))).check_fp32()
    with pytest.raises(ValueError):
        MemoryRead.bind(MatrixBackend(), state).expand(0)
    read = MemoryRead.bind(MatrixBackend(), state)
    assert read.expand(3).owner.tolist() == [0, 0, 0, 1, 1, 1] and read.expand(3).select(torch.tensor([1, 4])).owner.tolist() == [0, 1]


# ---------------------------------------------------------------------------------------------------- capacity and bytes

@pytest.mark.parametrize('device', DEVICES)
def test_capacity_boundary_is_exact_and_nothing_is_truncated(device):
    keys, values = torch.randn(2, 16, 768, device=device), torch.randn(2, 16, 128, device=device)
    full = build_slots(keys, values, 16)
    assert full.count.tolist() == [16, 16] and bool(full.valid.all()) and torch.equal(full.keys, keys)
    with pytest.raises(CapacityError):
        build_slots(keys, values, 15)  # E > S
    padded = build_slots(keys, values, 20)
    assert padded.capacity == 20 and padded.count.tolist() == [16, 16] and not bool(padded.valid[:, 16:].any())
    assert bool((padded.keys[:, 16:] == 0).all())
    # a row with fewer facts than the array width is allowed while it fits; a hole in the write order is not
    valid = torch.ones(2, 16, dtype=torch.bool, device=device)
    valid[1, 12:] = False
    assert build_slots(keys, values, 12 + 4, valid).count.tolist() == [16, 12]
    with pytest.raises(CapacityError):
        build_slots(keys, values, 15, valid)
    hole = valid.clone()
    hole[1, 3] = False
    with pytest.raises(ValueError, match='prefix'):
        build_slots(keys, values, 16, hole)
    with pytest.raises(ValueError):
        build_slots(keys, values, 0)


def test_byte_accounting_counts_masks_and_counts_too():
    one = build_slots(torch.zeros(1, 16, 768), torch.zeros(1, 16, 128), 16)
    assert persistent_bytes(one) == 57_344 + 16 + 8 == 57_368
    assert persistent_bytes(build_slots(torch.zeros(3, 16, 768), torch.zeros(3, 16, 128), 16)) == 3 * 57_368
    assert persistent_bytes(MatrixState(torch.zeros(1, 128, 768))) == 4 * 128 * 768 == 393_216
    assert MatrixBackend().persistent_bytes(MatrixState(torch.zeros(2, 128, 768))) == 2 * 393_216
    assert persistent_bytes(build_slots(torch.zeros(1, 16, 768), torch.zeros(1, 16, 128), 32)) == 4 * 32 * (768 + 128) + 32 + 8
    with pytest.raises(TypeError):
        SoftSlotBackend(1.0).persistent_bytes(MatrixState(torch.zeros(1, 2, 3)))


# ---------------------------------------------------------------------------------------------------------- read-only

@pytest.mark.parametrize('device', DEVICES)
def test_reads_leave_the_state_untouched_and_online_write_refuses(device):
    torch.manual_seed(4)
    state = build_slots(unit(torch.rand(3, 16, 20, device=device)), torch.randn(3, 16, 8, device=device), 16)
    matrix = MatrixState(torch.randn(3, 8, 20, device=device))
    before = (state_digest(state), state_digest(matrix))
    for backend, memory in ((SoftSlotBackend(0.1), state), (SoftSlotBackend(1.0, 2.0, 0.99), state), (HardSlotBackend(), state),
                            (MatrixBackend(1.5), matrix)):
        for depth in (1, 2, 4):  # repeated reads, as at several loop depths and for several questions
            read = MemoryRead.bind(backend, memory).expand(4)
            for _ in range(depth):
                read.read(unit(torch.rand(12, 20, device=device)))
            read.clone().read(unit(torch.rand(12, 20, device=device)))
        with pytest.raises(NotImplementedError, match='online_write'):
            backend.online_write(memory, torch.zeros(1, 20), torch.zeros(1, 8))
    assert (state_digest(state), state_digest(matrix)) == before
    cleared = reset_rows(state, torch.tensor([1], device=device))
    assert cleared.count.tolist()[1] == 0 and not bool(cleared.valid[1].any()) and state_digest(state) == before[0]
    assert state_digest(MatrixState(torch.zeros(1, 2, 2))) != state_digest(MatrixState(torch.ones(1, 2, 2)))


# ------------------------------------------------------------------------------------------------------ scoped diagnostic

@pytest.mark.parametrize('device', DEVICES)
def test_a_scoped_read_replaces_only_the_final_token_and_the_oracle_uses_the_value(device):
    torch.manual_seed(5)
    B, S, dk, dv = 3, 5, 6, 4
    keys, values = unit(torch.rand(B, S, dk, device=device)), torch.randn(B, S, dv, device=device)
    matrix = MatrixState(torch.randn(B, dv, dk, device=device))
    native = MemoryRead.bind(MatrixBackend(), matrix)
    primary = MemoryRead.bind(HardSlotBackend(), build_slots(keys, values, S))
    query, rows = unit(torch.rand(B, dk, device=device)), torch.arange(B, device=device)
    final = torch.tensor([True, False, True], device=device)
    scoped = ScopedRead(primary.backend, primary.state, primary.owner, native=native).select(rows, final)
    expected = torch.where(final[:, None], primary.read(query), native.read(query))
    assert torch.equal(scoped.read(query), expected)
    oracle = torch.randn(B, dv, device=device)
    clean = ScopedRead(primary.backend, primary.state, primary.owner, native=native, oracle=oracle).select(rows, final)
    assert torch.equal(clean.read(query), torch.where(final[:, None], oracle, native.read(query)))
    with pytest.raises(RuntimeError, match='final-token flag'):
        ScopedRead(primary.backend, primary.state, primary.owner, native=native).select(rows).read(query)
    with pytest.raises(ValueError, match='one question per episode'):
        ScopedRead(primary.backend, primary.state, primary.owner, native=native, oracle=oracle).expand(4)
    assert scoped.expand(2).owner.tolist() == [0, 0, 1, 1, 2, 2]


def test_read_specs_are_validated_and_round_trip():
    for spec in (ReadSpec('matrix', 2.0), ReadSpec('soft', 0.5, tau=0.1), ReadSpec('soft', tau=1.0, recency=True), ReadSpec('hard'),
                 ReadSpec('soft', tau=0.3, scope='final_only'), ReadSpec('clean_final', scope='final_only')):
        assert ReadSpec.from_dict(spec.as_dict()) == spec
    for bad in (dict(kind='soft'), dict(kind='matrix', tau=1.0), dict(kind='hard', recency=True), dict(kind='clean_final'),
                dict(kind='clean_final', scope='final_only', gain=2.0), dict(kind='other'), dict(kind='matrix', scope='x'),
                dict(kind='matrix', gain=0.0)):
        with pytest.raises(ValueError):
            ReadSpec(**bad)
