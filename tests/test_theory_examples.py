"""Small deterministic witnesses for docs/theory.md; run on a compute node."""

import pytest
import torch

from looplora.core import AdapterConfig, make_adapter
from looplora.core.memory import delta_update
from looplora.infra.seeds import derive_seed


DTYPE = torch.float64


def _reader(r, variant="iterq", activation="identity"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(17, "theory-examples", r, variant, activation))
        model = make_adapter(AdapterConfig(
            d_in=r, d_out=r, r=r, variant=variant, activation=activation,
            rho=.25, lambda_init=.5, beta_init=.5,
        )).double()
    with torch.no_grad():
        for name in ("A", "B", "K", "V", "Q"):
            getattr(model, name).copy_(torch.eye(r, dtype=DTYPE))
        model.U.copy_(8 * torch.eye(r, dtype=DTYPE))
        model.R.zero_()
    return model


def _store(r, edges):
    codes = torch.eye(r, dtype=DTYPE)
    memory = torch.zeros(1, r, r, dtype=DTYPE)
    for head, tail in edges:
        memory = delta_update(memory, codes[head:head + 1], codes[tail:tail + 1],
                              None, lam=1., beta=1.)
    return memory


@pytest.mark.parametrize("h", [1, 2, 4, 8])
@pytest.mark.parametrize("reverse_writes", [False, True])
def test_orthogonal_chain_terminal_and_default_gain(h, reverse_writes):
    edges = [(j, j + 1) for j in range(h)]
    memory = _store(h + 1, edges[::-1] if reverse_writes else edges)
    expected_memory = torch.zeros_like(memory)
    for head, tail in edges:
        expected_memory[0, tail, head] = 1
    torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)
    model = _reader(h + 1)
    u = torch.eye(h + 1, dtype=DTYPE)[:1]
    for depth in range(h + 3):
        actual = model.branch(model.read(u, memory, depth))
        expected = torch.zeros_like(u)
        for j in range(min(depth, h) + 1):
            expected[0, j] = 2 ** j
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.argmax(-1).item() == min(depth, h)
    with torch.no_grad():
        model.U.copy_(torch.eye(h + 1, dtype=DTYPE))
    for depth in range(1, h + 3):
        assert model.read(u, memory, depth).argmax(-1).item() == 0


@pytest.mark.parametrize("activation", ["identity", "relu"])
def test_same_first_hop_hides_terminal_from_b4_and_fixed_query(activation):
    first = _store(6, [(0, 1), (1, 2), (3, 4), (4, 5)])
    second = _store(6, [(0, 1), (1, 5), (3, 4), (4, 2)])
    u = torch.eye(6, dtype=DTYPE)[:1]
    torch.testing.assert_close(first @ u.unsqueeze(-1), second @ u.unsqueeze(-1))
    # Both episodes share the same six candidate codes, but terminals are 2/5.
    for variant in ("iterq", "fixedq"):
        model = _reader(6, variant, activation)
        with torch.no_grad():
            model.R.copy_(.5 * torch.eye(6, dtype=DTYPE))
        for depth in ([1] if variant == "iterq" else [1, 2, 4, 8]):
            torch.testing.assert_close(model.branch(model.read(u, first, depth)),
                                       model.branch(model.read(u, second, depth)),
                                       rtol=0, atol=0)
        if variant == "iterq":
            assert model.read(u, first, 2).argmax(-1).item() == 2
            assert model.read(u, second, 2).argmax(-1).item() == 5


def test_sequence_memory_path_jacobian_can_exceed_module_rank():
    model = _reader(1, activation="relu")
    with torch.no_grad():
        model.U.fill_(1)
    x = torch.tensor([1., 2., 3.], dtype=DTYPE, requires_grad=True)

    def sequence(value):
        output, _, _ = model.process_sequence(value.reshape(1, 3, 1), loop_steps=1)
        return output.flatten()

    jacobian = torch.autograd.functional.jacobian(sequence, x)
    history = torch.tril(jacobian, diagonal=-1)
    expected = .25 * .5 * torch.tensor([[0., 0., 0.], [2., 0., 0.],
                                        [3. / 4, 3., 0.]], dtype=DTYPE)
    torch.testing.assert_close(history, expected, rtol=0, atol=0)
    assert torch.linalg.matrix_rank(history).item() == 2 > model.cfg.r
    assert torch.linalg.matrix_rank(jacobian).item() == 3


def test_noncommuting_power_identity_and_polynomial_perturbation_bound():
    model = _reader(2)
    with torch.no_grad():
        model.R.copy_(torch.tensor([[.2, -.1], [.3, .1]], dtype=DTYPE))
        model.U.copy_(torch.tensor([[1., .2], [0., .8]], dtype=DTYPE))
        model.Q.copy_(torch.tensor([[.7, 0.], [-.2, 1.]], dtype=DTYPE))
    memory = torch.tensor([[.1, .8], [0., -.2]], dtype=DTYPE)
    reference = torch.tensor([[.3, 0.], [.4, -.1]], dtype=DTYPE)
    X = model.cfg.rho * (model.R + model.U @ memory @ model.Q)
    Y = model.cfg.rho * (model.R + model.U @ reference @ model.Q)
    assert torch.linalg.matrix_norm(X @ Y - Y @ X).item() > 1e-3
    norm = lambda value: torch.linalg.matrix_norm(value, ord=2)
    u = torch.tensor([[.7, -.4]], dtype=DTYPE)
    d = abs(model.cfg.rho) * norm(model.U) * norm(memory - reference) * norm(model.Q)
    polynomial_x = torch.eye(2, dtype=DTYPE)
    polynomial_y = polynomial_x.clone()
    series_bound = torch.zeros((), dtype=DTYPE)
    for depth in range(1, 7):
        telescoped = sum(torch.linalg.matrix_power(X, depth - 1 - k) @ (X - Y)
                         @ torch.linalg.matrix_power(Y, k) for k in range(depth))
        power_x, power_y = (torch.linalg.matrix_power(Z, depth) for Z in (X, Y))
        torch.testing.assert_close(power_x - power_y, telescoped, rtol=1e-12, atol=1e-14)
        scalar_sum = sum(norm(X) ** (depth - 1 - k) * norm(Y) ** k
                         for k in range(depth))
        assert norm(power_x - power_y) <= norm(X - Y) * scalar_sum + 1e-14
        polynomial_x, polynomial_y = polynomial_x + power_x, polynomial_y + power_y
        actual = model.read(u, memory[None], depth)
        baseline = model.read(u, reference[None], depth)
        torch.testing.assert_close(actual, u @ polynomial_x.T, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(baseline, u @ polynomial_y.T, rtol=1e-12, atol=1e-14)
        series_bound = series_bound + scalar_sum
        assert torch.linalg.vector_norm(actual - baseline) <= d * u.norm() * series_bound + 1e-14


def test_relu_cannot_be_moved_through_input_projection():
    A = torch.tensor([[1., 1.], [0., 1.]], dtype=DTYPE)
    B = torch.eye(2, dtype=DTYPE)
    z = torch.tensor([1., -1.], dtype=DTYPE)
    before = A @ torch.relu(B @ z)
    after = torch.relu(A @ B @ z)
    torch.testing.assert_close(before, torch.tensor([1., 0.], dtype=DTYPE))
    torch.testing.assert_close(after, torch.tensor([0., 0.], dtype=DTYPE))
    assert not torch.equal(before, after)


def test_signed_codes_need_more_than_a_hidden_bias_for_identity():
    codes = torch.cat((torch.eye(2, dtype=DTYPE), -torch.eye(2, dtype=DTYPE)))
    hidden_only = torch.relu(codes + 2)
    assert bool((hidden_only >= 0).all())
    assert not torch.equal(hidden_only, codes)
    torch.testing.assert_close(hidden_only - 2, codes, rtol=0, atol=0)
    U = torch.cat((torch.eye(2, dtype=DTYPE), -torch.eye(2, dtype=DTYPE)))
    Q = U.T
    torch.testing.assert_close(torch.relu(codes @ U.T) @ Q.T, codes, rtol=0, atol=0)


def test_exact_elimination_preserves_state_dependent_router():
    A = torch.tensor([[.2, -.1, .4], [.3, .2, -.1]], dtype=DTYPE)
    B = torch.tensor([[.2, -.1], [.1, .3], [-.2, .1]], dtype=DTYPE)
    x = torch.tensor([.4, -.2, .7], dtype=DTYPE)
    w = torch.tensor([.1, -.3, .2], dtype=DTYPE)
    h, z = x.clone(), torch.zeros(2, dtype=DTYPE)
    for _ in range(8):
        gate_full = torch.sigmoid(w @ h + .1)
        gate_low = torch.sigmoid(w @ x + (w @ B) @ z + .1)
        torch.testing.assert_close(gate_full, gate_low, rtol=1e-12, atol=1e-14)
        h = h + gate_full * B @ torch.relu(A @ h)
        z = z + gate_low * torch.relu(A @ x + (A @ B) @ z)
        torch.testing.assert_close(h, x + B @ z, rtol=1e-12, atol=1e-14)


def test_double_ba_zero_first_derivative_but_not_a_two_sided_hessian():
    def original(a, b):
        return b * a * torch.relu(b * a)  # x=1

    a = torch.tensor(1., dtype=DTYPE, requires_grad=True)
    b = torch.tensor(0., dtype=DTYPE, requires_grad=True)
    gradient = torch.autograd.grad(original(a, b), (a, b))
    assert all(value.item() == 0 for value in gradient)
    # Homogeneity gives distinct directional quadratic coefficients at zero.
    for size in (1e-2, 1e-4):
        for sign, coefficient in ((1, 1.), (-1, 0.)):
            point = torch.tensor(sign * size, dtype=DTYPE, requires_grad=True)
            value = original(a, point)
            assert (value / size ** 2).item() == pytest.approx(coefficient)
            slope = torch.autograd.grad(value, point, create_graph=True)[0]
            curvature = torch.autograd.grad(slope, point)[0]
            assert curvature.item() == 2 * coefficient
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(17, "theory-examples", "double-ba-skip"))
        skip = make_adapter(AdapterConfig(d_in=1, d_out=1, r=1,
                                         variant="double_ba", double_ba_form="skip")).double()
    with torch.no_grad():
        skip.A.fill_(1)
    output, _, _ = skip.process_sequence(torch.ones(1, 1, 1, dtype=DTYPE))
    assert output.item() == 0
    assert torch.autograd.grad(output.sum(), skip.B)[0].item() == 1


# WP-T2: constructed witnesses, not an interpretation of any trained checkpoint.
def _wpt2_model(variant="M1", activation="identity", beta=1e-5):
    from looplora.sym.config import resolve_config
    from looplora.sym.model import SymbolicModel

    config = resolve_config({
        "device": "cpu", "writer": "joint", "variant": variant,
        "activation": activation, "seed": derive_seed(17, "WP-T2", "parameters"),
        "backbone_seed": 31415, "model": {"r": 32, "r_mem": 32,
        "d_e": 64, "beta_init": beta, "lambda_init": 1 - 1e-9},
        "task": {"name": "t2_chain_end", "E": 12, "Q": 8,
                 "chain_lengths": [8, 4], "hops": list(range(1, 9))}})
    model = SymbolicModel(config).double()
    p, anchor, out, baseline = 15, 15, slice(16, 31), 31
    backbone, reader, writer = model.backbone, model.adapter, model.writer
    constraints = torch.cat((backbone.relations, backbone.hops))
    target = torch.zeros(len(constraints), dtype=DTYPE)
    target[0] = 1
    # Exact finite-table hop erasure; a relation channel supplies the offset.
    g = torch.linalg.lstsq(constraints, target).solution
    torch.testing.assert_close(constraints @ g, target, atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        for name in ("A", "K", "V", "Q", "U", "R", "B"):
            getattr(reader, name).zero_()
        writer.A[:p, :p] = -torch.eye(p, dtype=DTYPE)
        writer.A[:p, 128:128 + p] = torch.eye(p, dtype=DTYPE)
        writer.A[anchor, 64:128] = g
        writer.K[0, anchor] = reader.Q[0, anchor] = 1
        writer.V[:p, :p] = torch.eye(p, dtype=DTYPE)
        reader.U[out, :p] = torch.eye(p, dtype=DTYPE)
        actual_beta = writer.a_beta.sigmoid().item()
        scale = model.config["model"]["alpha"] * model.config["model"]["rho"] * actual_beta
        reader.B[:p, out] = torch.eye(p, dtype=DTYPE) / scale
        if activation == "relu":
            shift = 2 * backbone.entities[:, :p].abs().max().item() + 1
            reader.R[out, anchor] = shift
            reader.R[baseline, anchor] = shift
            reader.B[:p, baseline] = -1 / scale
    assert reader.A is writer.A and reader.K is writer.K and reader.V is writer.V
    return model


@pytest.mark.parametrize("variant", ["B4", "B5", "M1"])
@pytest.mark.parametrize("activation", ["identity", "relu"])
@pytest.mark.parametrize("ideal_geometry", [False, True])
@torch.no_grad()
def test_wpt2_actual_shared_writer_ema_and_depth_flat_shortcut(
        variant, activation, ideal_geometry):
    from looplora.data.relations import TaskSpec, generate

    model = _wpt2_model(variant, activation)
    arrays = generate(TaskSpec(**model.config["task"]), "val", 0, 1,
                      derive_seed(17, "WP-T2", "shared-validation"))
    facts = torch.as_tensor(arrays.facts, dtype=torch.long)
    entities = facts[0, :, (0, 2)].unique()
    if ideal_geometry:
        # Fourteen orthogonal episode entities fit in p=15; no random margin claim.
        model.backbone.entities.zero_()
        model.backbone.entities[entities, :len(entities)] = torch.eye(len(entities), dtype=DTYPE)
    batch = model.prepare(arrays)
    values = model.writer(model.backbone.facts(facts), torch.ones(1, 12, dtype=torch.bool))[1]
    beta = model.writer.a_beta.sigmoid().item()
    lam = model.writer.a_lambda.sigmoid().item()
    a, n = lam * (1 - beta), facts.shape[1]
    weights = beta * a ** torch.arange(n - 1, -1, -1, dtype=DTYPE)
    expected = (values * weights[None, :, None]).sum(1)
    torch.testing.assert_close(batch.memory[:, :, 0], expected, atol=1e-13, rtol=1e-11)
    torch.testing.assert_close(batch.memory[:, :, 1:], torch.zeros_like(batch.memory[:, :, 1:]),
                               atol=1e-13, rtol=0)
    ideal_sum = values.sum(1)[:, :15]
    actual_sum = batch.memory[:, :15, 0] / beta
    bound = values.norm(dim=-1).max() * (1 - a) * n * (n - 1) / 2
    assert (actual_sum - ideal_sum).norm() <= bound + 1e-11
    heads, tails = facts[0, :, 0], facts[0, :, 2]
    terminals = tails[~torch.isin(tails, heads)]
    roots = heads[~torch.isin(heads, tails)]
    endpoint_sum = (model.backbone.entities[terminals, :15].sum(0)
                    - model.backbone.entities[roots, :15].sum(0))
    torch.testing.assert_close(ideal_sum[0], endpoint_sum, atol=1e-12, rtol=1e-12)
    # Independently vary both query entity and stated hop, including unknown hop.
    starts = entities[:, None].expand(-1, 5)
    hops = torch.tensor([-1, 0, 1, 4, 8]).expand_as(starts)
    u = model.writer.u(model.backbone.queries(starts, torch.zeros_like(starts), hops))
    q = u @ model.adapter.Q.T
    torch.testing.assert_close(q, torch.nn.functional.one_hot(
        torch.zeros_like(starts), 32).to(DTYPE), atol=1e-12, rtol=1e-12)
    expected_output = torch.zeros_like(batch.x[..., :64])
    expected_output[..., :15] = actual_sum[:, None]
    varied_memory = batch.memory[:, None].expand(len(entities), 5, -1, -1)
    varied_output = model.adapter.branch(model.adapter.read(
        u, varied_memory, 1 if variant == "B4" else 4))
    torch.testing.assert_close(varied_output, expected_output[:, :1].expand_as(varied_output),
                               atol=2e-9, rtol=1e-9)
    for depth in ([1] if variant == "B4" else [1, 2, 4, 8]):
        result = model.read(batch, depth)
        output = model.adapter.branch(result["z"])
        torch.testing.assert_close(output, expected_output, atol=2e-9, rtol=1e-9)
        if ideal_geometry:
            scores = result["logits"][0, 0]
            nonterminals = entities[~torch.isin(entities, terminals)]
            assert scores[terminals].min() > scores[nonterminals].max()
    assert model.read(batch, 0)["logits"].abs().max() == 0


@torch.no_grad()
def test_wpt2_large_beta_constant_key_does_not_telescope():
    model = _wpt2_model(beta=.8)
    model.backbone.entities.zero_()
    model.backbone.entities[:3, :3] = torch.eye(3, dtype=DTYPE)
    # Latest write is the first edge: its internal tail outranks the terminal.
    facts = torch.tensor([[[1, 0, 2], [0, 0, 1]]])
    memory, values = model.writer(model.backbone.facts(facts), torch.ones(1, 2, dtype=torch.bool))
    beta, lam = (getattr(model.writer, name).sigmoid().item()
                 for name in ("a_beta", "a_lambda"))
    expected = beta * (lam * (1 - beta) * values[:, 0] + values[:, 1])
    torch.testing.assert_close(memory[:, :, 0], expected, atol=1e-13, rtol=1e-12)
    assert memory[0, 1, 0] > memory[0, 2, 0]
    assert (memory[:, :, 0] / beta - values.sum(1)).norm() > .5


def _wpt2_chain_graph(lengths, typed=False):
    edges, depths, terminals, roots, offset = [], [], [], [], 0
    for length in lengths:
        roots.append(offset)
        terminals.append(offset + length)
        depths.extend(range(length + 1))
        edges.extend((offset + d, offset + d + 1) for d in range(length))
        offset += length + 1
    # Typed distractor tails are candidates but absent from relation-zero S.
    return edges, torch.tensor(depths + ([-1] * 3 if typed else [])), terminals, roots


@pytest.mark.parametrize("lengths,typed", [((8, 4), False), ((8, 1), True),
                                           ((8, 8, 2), False), ((4, 4), False)])
def test_wpt2_factorized_exact_store_filtrations_and_swap(lengths, typed):
    edges, depths, terminals, roots = _wpt2_chain_graph(lengths, typed)
    size = len(depths)
    S = _store(size, edges)[0]
    ones = torch.ones(size, dtype=DTYPE)
    b1 = S @ ones
    torch.testing.assert_close(b1, (depths >= 1).to(DTYPE), atol=0, rtol=0)
    assert int(b1.sum()) == sum(lengths)  # tie-breaking accuracy 1/E_r, chance 1/N.
    accumulated = torch.zeros(size, dtype=DTYPE)
    for step in range(1, max(lengths) + 1):
        filtration = torch.linalg.matrix_power(S, step) @ ones
        torch.testing.assert_close(filtration, (depths >= step).to(DTYPE), atol=0, rtol=0)
        accumulated = b1 + S @ accumulated
        expected = torch.clamp(depths, min=0, max=step).to(DTYPE)
        torch.testing.assert_close(accumulated, expected, atol=0, rtol=0)
        top = set(torch.where(accumulated == accumulated.max())[0].tolist())
        assert top == set(torch.where(depths >= step)[0].tolist())
        if lengths == (8, 4) and step <= 4:
            assert top != set(terminals)
    last_support = set(torch.where((torch.linalg.matrix_power(S, max(lengths)) @ ones) > 0)[0].tolist())
    assert (last_support == set(terminals)) == (len(set(lengths)) == 1)
    # Swap an edge at depth 1 in chain A with an edge at depth 0 in chain B.
    second_root = roots[1]
    swapped = list(edges)
    i, j = swapped.index((1, 2)), swapped.index((second_root, second_root + 1))
    swapped[i], swapped[j] = (1, second_root + 1), (second_root, 2)
    T = _store(size, swapped)[0]
    torch.testing.assert_close(S @ ones, T @ ones, atol=0, rtol=0)
    torch.testing.assert_close(S @ ones - S.T @ ones,
                               T @ ones - T.T @ ones, atol=0, rtol=0)
    assert set(torch.where((S @ ones > 0) & (S.T @ ones == 0))[0].tolist()) == set(terminals)
    assert set(torch.where((T @ ones > 0) & (T.T @ ones == 0))[0].tolist()) == set(terminals)
    assert not torch.equal(S @ S @ ones, T @ T @ ones)


def test_wpt2_swap_walk_length_law_by_exhaustive_cuts():
    for hop in range(1, 9):
        for other_length in (1, 2, 4, 8):
            for budget in (1, 4, 8):
                counts = sum(p + q <= budget for p in range(1, hop + 1)
                             for q in range(other_length))
                formula = sum(min(other_length, max(0, budget - p + 1))
                              for p in range(1, hop + 1))
                assert counts == formula


@torch.no_grad()
def test_wpt2_descriptive_frozen_gaussian_endpoint_margins():
    import json

    from looplora.data.relations import TaskSpec, generate
    from looplora.sym.backbone import SymbolicBackbone

    # Validation only. This ideal uniform endpoint statistic is constructed;
    # it is neither delta accumulation at trained gates nor a learned checkpoint.
    spec = TaskSpec(name="t2_chain_end", E=12, Q=1, chain_lengths=(8, 4), hops=(1,))
    arrays = generate(spec, "val", 0, 256, derive_seed(17, "WP-T2", "gaussian-validation"))
    table = SymbolicBackbone(backbone_seed=31415).double().entities
    descriptions = []
    for p in (7, 15, 31, 63, 64):
        terminal_scores, interior_scores, margins, top, all_top = [], [], [], [], []
        pair_differences = []
        for row in arrays.facts:
            heads, tails = torch.as_tensor(row[:, 0]).long(), torch.as_tensor(row[:, 2]).long()
            terminals = tails[~torch.isin(tails, heads)]
            roots = heads[~torch.isin(heads, tails)]
            candidates = torch.cat((heads, tails)).unique()
            interior = candidates[~torch.isin(candidates, torch.cat((terminals, roots)))]
            nonterminals = candidates[~torch.isin(candidates, terminals)]
            projected = table[:, :p]
            endpoint = projected[terminals].sum(0) - projected[roots].sum(0)
            scores = projected[candidates] @ endpoint
            direct = (projected[tails] - projected[heads]).sum(0)
            torch.testing.assert_close(direct, endpoint, atol=1e-12, rtol=1e-12)
            t, v = projected[terminals] @ endpoint, projected[interior] @ endpoint
            terminal_scores.extend(t.tolist())
            interior_scores.extend(v.tolist())
            pair_differences.extend((t[:, None] - v).flatten().tolist())
            margin = (t.min() - (projected[nonterminals] @ endpoint).max()).item()
            margins.append(margin)
            top.append(int(candidates[scores.argmax()]) in terminals.tolist())
            all_top.append(margin > 0)
        def moments(values):
            value = torch.tensor(values, dtype=DTYPE)
            return {"mean": value.mean().item(), "variance": value.var(unbiased=False).item()}
        descriptions.append({"p": p, "terminal": moments(terminal_scores),
            "interior": moments(interior_scores), "terminal_minus_interior": moments(pair_differences),
            "theory_terminal_mean": p / 64, "theory_terminal_variance": 5 * p / 64**2,
            "theory_interior_variance": 4 * p / 64**2,
            "theory_pair_variance": 9 * p / 64**2,
            "top_one_terminal_fraction": sum(top) / len(top),
            "all_terminals_first_fraction": sum(all_top) / len(all_top),
            "minimum_terminal_nonterminal_margin": min(margins)})
    print("WP-T2 constructed Gaussian measurements " + json.dumps({
        "label": "constructed descriptive validation; no learned mechanism claim",
        "backbone_seed": 31415, "episodes": 256, "layout": [8, 4],
        "projection_ranks": descriptions}, sort_keys=True))


def test_wpt2_tail_sum_alone_does_not_determine_terminal_set():
    ones = torch.ones(6, dtype=DTYPE)
    S = _store(6, [(0, 1), (1, 2), (3, 4), (4, 5)])[0]
    T = _store(6, [(0, 2), (2, 1), (3, 4), (4, 5)])[0]
    torch.testing.assert_close(S @ ones, T @ ones, atol=0, rtol=0)
    first = set(torch.where((S @ ones > 0) & (S.T @ ones == 0))[0].tolist())
    second = set(torch.where((T @ ones > 0) & (T.T @ ones == 0))[0].tolist())
    assert first == {2, 5} and second == {1, 5}


# WP-T3: exact delta coefficients and constructed retrieval failure witnesses.
def _wpt3_memory(keys, values, lam, beta):
    state = torch.zeros(1, keys.shape[1], keys.shape[1], dtype=DTYPE)
    for key, value in zip(keys, values):
        state = delta_update(state, key[None], value[None], None, lam=lam, beta=beta)
    return state


def _wpt3_coefficients(keys, query, lam, beta):
    gram, b = keys @ keys.T, keys @ query
    a = torch.zeros_like(b)
    for i in range(len(keys) - 1, -1, -1):
        a[i] = b[i] - beta * (gram[i, i + 1:] @ a[i + 1:])
    decay = lam ** torch.arange(len(keys) - 1, -1, -1, dtype=DTYPE)
    return beta * decay * a


@pytest.mark.parametrize("beta", [.1, .7])
@torch.no_grad()
def test_wpt3_exact_nonorthogonal_products_gram_and_correlated_value_sir(beta):
    generator = torch.Generator().manual_seed(derive_seed(17, "WP-T3", "general-coefficients"))
    keys = torch.nn.functional.normalize(torch.randn(16, 32, generator=generator, dtype=DTYPE), dim=-1)
    common = torch.randn(32, generator=generator, dtype=DTYPE)
    values = common + .3 * torch.randn(16, 32, generator=generator, dtype=DTYPE)
    query = torch.nn.functional.normalize(torch.randn(32, generator=generator, dtype=DTYPE), dim=0)
    lam, eye = .99, torch.eye(32, dtype=DTYPE)
    memory = _wpt3_memory(keys, values, lam, beta)[0]
    coeff = _wpt3_coefficients(keys, query, lam, beta)
    product_coeff = []
    for i in range(len(keys)):
        product = eye.clone()
        for j in range(i + 1, len(keys)):
            product = product @ (eye - beta * torch.outer(keys[j], keys[j]))
        product_coeff.append(beta * lam ** (15 - i) * (keys[i] @ product @ query))
    torch.testing.assert_close(coeff, torch.stack(product_coeff), atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(memory @ query, coeff @ values, atol=1e-14, rtol=1e-12)
    signal, noise = coeff[0] * values[0], coeff[1:] @ values[1:]
    value_gram = values @ values.T
    gram_noise_sq = coeff[1:] @ value_gram[1:, 1:] @ coeff[1:]
    torch.testing.assert_close(noise.square().sum(), gram_noise_sq, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(signal @ noise, coeff[0] * (value_gram[0, 1:] @ coeff[1:]),
                               atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(signal.square().sum() / noise.square().sum(),
                               coeff[0].square() * value_gram[0, 0] / gram_noise_sq,
                               atol=1e-14, rtol=1e-12)


@torch.no_grad()
def test_wpt3_small_beta_leading_and_second_order_rigorous_bounds():
    generator = torch.Generator().manual_seed(derive_seed(17, "WP-T3", "small-beta"))
    keys = torch.nn.functional.normalize(torch.randn(16, 32, generator=generator, dtype=DTYPE), dim=-1)
    values = torch.randn(16, 32, generator=generator, dtype=DTYPE)
    query = torch.randn(32, generator=generator, dtype=DTYPE)
    lam, beta = .99, 1e-4
    gram, b, norms = keys @ keys.T, keys @ query, values.norm(dim=-1)
    decay = lam ** torch.arange(15, -1, -1, dtype=DTYPE)
    actual = _wpt3_memory(keys, values, lam, beta)[0] @ query
    leading = (beta * decay * b) @ values
    first_correction = torch.zeros_like(actual)
    bound2 = torch.zeros((), dtype=DTYPE)
    bound3 = torch.zeros((), dtype=DTYPE)
    for i in range(16):
        bound2 += beta**2 * query.norm() * decay[i] * norms[i] * gram[i, i + 1:].abs().sum()
        for j in range(i + 1, 16):
            first_correction -= beta**2 * decay[i] * gram[i, j] * b[j] * values[i]
            bound3 += (beta**3 * query.norm() * decay[i] * norms[i]
                       * gram[i, j].abs() * gram[j, j + 1:].abs().sum())
    assert (actual - leading).norm() <= bound2 + 1e-16
    assert (actual - leading - first_correction).norm() <= bound3 + 1e-16


def _wpt3_ranked_but_interfering():
    keys = torch.eye(32, dtype=DTYPE)[:16].clone()
    keys[1:, 0] = .72
    keys[1:, 1:] *= (1 - .72**2)**.5
    values = torch.eye(32, dtype=DTYPE)[:16]
    return keys, values, _wpt3_memory(keys, values, .99, .1)


@torch.no_grad()
def test_wpt3_low_auxiliary_losses_do_not_certify_read_and_q_scale(capsys):
    import json

    keys, values, memory = _wpt3_ranked_but_interfering()
    query = torch.eye(32, dtype=DTYPE)[0]
    cosine = keys @ query
    addressing_ce = torch.nn.functional.cross_entropy((cosine / .1)[None], torch.tensor([0]))
    torch.testing.assert_close(addressing_ce, torch.log1p(15 * torch.exp(torch.tensor(-2.8, dtype=DTYPE))))
    assert addressing_ce.item() == pytest.approx(.648, abs=.002)
    assert cosine.argmax().item() == 0
    head = torch.zeros(1024, 32, dtype=DTYPE)
    head[:16, :16] = 12 * torch.eye(16, dtype=DTYPE)
    value_ce = torch.nn.functional.cross_entropy(values @ head.T, torch.arange(16))
    assert value_ce < .01
    read = memory[0] @ query
    h = .99 * (1 - .1 * .72**2)
    assert read[0].item() == pytest.approx(.1 * h**15, rel=1e-12, abs=1e-14)
    assert read.argmax().item() == 15  # newest distractor, despite top-ranked target key.
    assert (head @ read).argmax().item() == 15
    reader = _reader(32, activation="relu")
    reader.U.copy_(torch.eye(32, dtype=DTYPE))
    records, differences = [], []
    for scale in (1., 1e-3):
        reader.Q.copy_(scale * torch.eye(32, dtype=DTYPE))
        q = reader.Q @ query
        qhat = torch.nn.functional.normalize(q, dim=0)
        loss = torch.nn.functional.cross_entropy(((keys @ qhat) / .1)[None], torch.tensor([0]))
        torch.testing.assert_close(loss, addressing_ce, atol=1e-14, rtol=1e-12)
        scaled_read = memory[0] @ q
        torch.testing.assert_close(scaled_read, scale * read, atol=1e-15, rtol=1e-12)
        with_memory = reader.read(query[None], memory, 1)
        without_memory = reader.read(query[None], torch.zeros_like(memory), 1)
        difference = (with_memory - without_memory)[0]
        differences.append(difference)
        torch.testing.assert_close(difference, .25 * scale * read, atol=1e-15, rtol=1e-12)
        records.append({"query_scale": scale, "normalized_addressing_ce": loss.item(),
            "query_norm": q.norm().item(), "read_norm": scaled_read.norm().item(),
            "target_component": scaled_read[0].item(), "interference_norm": scaled_read[1:].norm().item(),
            "memory_specific_z_norm": difference.norm().item(), "u_norm": query.norm().item()})
    torch.testing.assert_close(differences[1], 1e-3 * differences[0], atol=1e-15, rtol=1e-12)
    # Disabled capture makes the constructed numbers available in the compute log.
    with capsys.disabled():
        print("WP-T3 constructed measurements " + json.dumps({
            "label": "constructed E16/r32; no real checkpoint", "value_head_ce": value_ce.item(),
            "read_head_ce": torch.nn.functional.cross_entropy((head @ read)[None], torch.tensor([0])).item(),
            "target_key_top1": True, "actual_read_top1": read.argmax().item(), "query_scales": records},
            sort_keys=True))


def test_wpt3_zero_B_blocks_first_main_gradient_then_nonzero_B_enables_U():
    reader = _reader(2)
    x = torch.tensor([[1., 2.]], dtype=DTYPE)
    memory = torch.tensor([[[0., 1.], [1., 0.]]], dtype=DTYPE)
    target = torch.tensor([[1., -1.]], dtype=DTYPE)
    with torch.no_grad():
        reader.B.zero_()
        reader.U.copy_(torch.eye(2, dtype=DTYPE))
    output = reader.branch(reader.read(reader.u(x), memory, 1))
    (.5 * (output - target).square().sum()).backward()
    assert reader.B.grad.norm() > 0
    for name in ("A", "U", "Q", "R"):
        gradient = getattr(reader, name).grad
        assert gradient is not None and gradient.abs().max() == 0
    reader.zero_grad(set_to_none=True)
    with torch.no_grad():
        reader.B.copy_(.1 * torch.eye(2, dtype=DTYPE))
    output = reader.branch(reader.read(reader.u(x), memory, 1))
    (.5 * (output - target).square().sum()).backward()
    assert reader.U.grad.norm() > 0


@torch.no_grad()
def test_wpt3_finite_large_beta_can_reverse_target_coefficient_at_same_infonce():
    a, c = .8, .2
    delta = .1 * torch.log(15 / torch.expm1(torch.tensor(.65, dtype=DTYPE)))
    b = a - delta
    keys = torch.zeros(16, 32, dtype=DTYPE)
    keys[0, :2] = torch.tensor([.8, .6], dtype=DTYPE)
    for j in range(1, 16):
        keys[j, 0], keys[j, 1], keys[j, j + 1] = b, c, (1 - b**2 - c**2).sqrt()
    values = torch.eye(32, dtype=DTYPE)[:16]
    query = torch.eye(32, dtype=DTYPE)[0]
    loss = torch.nn.functional.cross_entropy(((keys @ query) / .1)[None], torch.tensor([0]))
    assert loss.item() == pytest.approx(.65, rel=1e-12)
    assert (keys @ query).argmax().item() == 0
    memory = _wpt3_memory(keys, values, .99, .99)[0]
    g, D = b**2 + c**2, a * b + .6 * c
    expected = .99 * .99**15 * (a - D * b / g * (1 - (1 - .99 * g)**15))
    torch.testing.assert_close(memory[0] @ query, expected, atol=1e-14, rtol=1e-12)
    assert expected < 0 and (memory @ query)[0] < 0
