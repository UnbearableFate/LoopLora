"""LSR: instrumentation, rules, validation selection, windows and the test-range guard (docs/prereg/LSR.md)."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_lsr
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm import lsr, p1lc
from looplora.llm import lsr_rules as rules
from looplora.llm.batching import repeat_snapshot
from looplora.llm.config import CONFIRMATION_WINDOWS, resolve_config, split_window
from looplora.llm.inject import configure_numerics, load_tokenizer
from looplora.llm.train import LLMExperiment, make_task_spec
from looplora.sym.evaluate import stratify_queries

LEFTOVERS = {"process_sequence", "_masked_branch", "_read"}


# ------------------------------------------------------------------------------------------- windows

def test_the_lsr_window_is_named_registered_and_disjoint_and_leaves_the_others_unchanged():
    # each registered entry individually: other studies add windows to the same dict
    assert CONFIRMATION_WINDOWS["p1l"] == 100000 and CONFIRMATION_WINDOWS["p1l-s"] == 110000
    assert CONFIRMATION_WINDOWS["p5l"] == 104000 and CONFIRMATION_WINDOWS["lsr"] == 106000
    assert split_window({"confirmation": "lsr"}, "confirm") == ("test", 106000)
    assert split_window({}, "confirm") == ("test", 100000)
    for split in ("val", "test"):
        assert split_window({"confirmation": "lsr"}, split) == (split, 0)
    assert resolve_config(dict(eval=dict(splits=["confirm"], allow_confirmation=True, confirmation="lsr")))
    ranges = sorted((start, start + 2000) for start in CONFIRMATION_WINDOWS.values()) + [(102000, 104000)]
    ranges.sort()
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))
    assert rules.VAL_RANGE["split"] == "val" and rules.VAL_RANGE["start"] >= 1000   # P1-L / P5L validation was [0, 1000)
    assert rules.SMOKE_RANGE["split"] == "val" and rules.LATENCY_RANGE["split"] == "val"


def test_the_test_range_needs_a_matching_committed_authorization(tmp_path):
    (tmp_path / "docs/prereg").mkdir(parents=True)
    (tmp_path / "reports/lsr").mkdir(parents=True)
    (tmp_path / "docs/prereg/LSR.md").write_text("prereg")
    (tmp_path / "reports/lsr/thresholds.json").write_text("{}")
    digest = evaluate_lsr.digest
    prereg = dict(path=evaluate_lsr.PREREG, hash=digest(tmp_path / evaluate_lsr.PREREG))
    thresholds = dict(path=evaluate_lsr.THRESHOLDS, hash=digest(tmp_path / evaluate_lsr.THRESHOLDS))
    authorization = dict(study="lsr", status="allowed", orchestrator_decision="released", confirmation_window="lsr",
                         gated_comparisons=6, confidence=1 - 0.05 / 6, prereg=prereg, thresholds=thresholds)
    locked = dict(status="locked", prereg=prereg)
    evaluate_lsr.check_test_authorization(authorization, locked, tmp_path)
    for change in (dict(status="held"), dict(study="p5l"), dict(orchestrator_decision=""), dict(confirmation_window="p5l"),
                   dict(gated_comparisons=2), dict(confidence=.975), dict(prereg=dict(prereg, hash="0")),
                   dict(thresholds=dict(thresholds, hash="0"))):
        with pytest.raises(ValueError):
            evaluate_lsr.check_test_authorization({**authorization, **change}, locked, tmp_path)
    for broken in (dict(locked, status="draft"), dict(locked, prereg=dict(prereg, hash="0"))):
        with pytest.raises(ValueError):
            evaluate_lsr.check_test_authorization(authorization, broken, tmp_path)


# ------------------------------------------------------------------------------------------- rules

def trace_of(pred, change=None, read_ratio=None, answer=None):
    pred = np.asarray(pred, dtype=np.int32)
    shape = pred.shape[:-1]
    change = np.full((*shape, 8), 9., dtype=np.float32) if change is None else np.asarray(change, dtype=np.float32)
    ratio = np.full((*shape, 8), 9., dtype=np.float32) if read_ratio is None else np.asarray(read_ratio, dtype=np.float32)
    ratio[..., 7] = np.nan
    return dict(pred=pred, change=change, read_ratio=ratio,
                answer=np.zeros(shape, dtype=np.int32) if answer is None else np.asarray(answer, dtype=np.int32))


def test_r1_and_r3_stop_at_the_first_qualifying_step_and_are_forced_at_eight():
    change = np.array([[.9, .25, .05, .01, .3, .3, .3, .3],     # first <= .1 at s = 3
                       [.9, .9, .9, .9, .9, .9, .9, .9],        # never: forced at 8
                       [.05, .9, .9, .9, .9, .9, .9, .9]])      # s = 1 is allowed
    trace = trace_of(np.zeros((3, 9)), change=change, read_ratio=change)
    assert rules.stop_depths("R1", trace, .1).tolist() == [3, 8, 1]
    assert rules.stop_depths("R3", trace, .1).tolist() == [3, 8, 1]
    assert rules.stop_depths("R1", trace, .25).tolist() == [2, 8, 1]      # "<=" as in P4-S
    assert rules.stop_depths("R3", trace, 9.).tolist() == [1, 1, 1]       # NaN at s = 8 never fires, but s = 8 is forced
    depths = np.array([3, 8, 1])
    assert rules.reads_of("R1", depths).tolist() == [3, 8, 1] and rules.reads_of("R3", depths).tolist() == [4, 9, 2]
    assert rules.reads_of("R2", depths).tolist() == [3, 8, 1] and rules.reads_of("fixed", depths).tolist() == [3, 8, 1]


def test_r2_needs_three_equal_lens_predictions_starting_at_p0():
    pred = np.array([[5, 5, 5, 1, 1, 1, 2, 3, 4],      # p0 = p1 = p2: stops at 2
                     [7, 1, 1, 1, 9, 9, 9, 9, 9],      # p1 = p2 = p3: stops at 3
                     [1, 2, 3, 4, 5, 6, 7, 8, 9],      # never: 8
                     [1, 1, 2, 2, 3, 3, 4, 4, 5]])     # never equal three in a row: 8
    assert rules.stop_depths("R2", trace_of(pred)).tolist() == [2, 3, 8, 8]
    with pytest.raises(ValueError):
        rules.stop_depths("R4", trace_of(pred))


def test_correct_at_reads_the_prediction_of_the_stop_depth():
    pred = np.arange(9)[None, None, :] + 10 * np.arange(2)[None, :, None]       # [1, 2 hops, 9]
    trace = trace_of(pred, answer=[[3, 17]])
    assert rules.correct_at(trace, np.array([[3, 7]])).tolist() == [[True, True]]
    assert rules.correct_at(trace, np.array([[4, 7]])).tolist() == [[False, True]]
    assert rules.grid_k(2 ** -3.5) == -7 and rules.GRID_K == tuple(range(-20, 5)) and len(rules.GRID) == 25
    assert rules.GRID[0] == 2 ** -10 and rules.GRID[-1] == 4.


# ------------------------------------------------------------------------------------- selection

def random_trace(seed, episodes=60):
    """Episodes whose accuracy depends on hop and depth, and whose stop signals are noisy functions of the depth."""
    rng = np.random.default_rng(seed)
    pred = rng.integers(0, 3, size=(episodes, 8, 9)).astype(np.int32)
    answer = rng.integers(0, 3, size=(episodes, 8)).astype(np.int32)
    change = np.exp(rng.normal(-1 - 0.3 * np.arange(8), 1, size=(episodes, 8, 8))).astype(np.float32)
    ratio = np.exp(rng.normal(-0.3 * np.arange(8), 1, size=(episodes, 8, 8))).astype(np.float32)
    return trace_of(pred, change, ratio, answer)


def brute_force(traces, rule, tau):
    seeds = sorted(traces)
    depths = {s: rules.stop_depths(rule, traces[s], tau) for s in seeds}
    ok = {s: rules.correct_at(traces[s], depths[s]) for s in seeds}
    base = {s: rules.correct_at(traces[s], rules.fixed_depths(traces[s], 4)) for s in seeds}
    pooled = sum(int(ok[s][:, 0].sum()) - int(base[s][:, 0].sum()) for s in seeds) / sum(len(ok[s]) for s in seeds)
    each = [(int(ok[s][:, 0].sum()) - int(base[s][:, 0].sum())) / len(ok[s]) for s in seeds]
    feasible = pooled >= -.08 - 1e-12 and min(each) >= -.10 - 1e-12
    return feasible, sum(int(ok[s][:, 3:].sum()) for s in seeds), sum(int(rules.reads_of(rule, depths[s]).sum()) for s in seeds)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_selection_matches_an_independent_enumeration(seed):
    traces = {s: random_trace(seed * 100 + s) for s in rules.SEEDS}
    result = rules.select_thresholds(traces)
    for rule in ("R1", "R3"):
        table = [(k, *brute_force(traces, rule, tau)) for k, tau in zip(rules.GRID_K, rules.GRID)]
        feasible = [row for row in table if row[1]]
        entry = result["rules"][rule]
        assert [row["feasible"] for row in entry["grid"]] == [row[1] for row in table]
        assert [row["objective"] for row in entry["grid"]] == [row[2] for row in table]
        if feasible:
            expected = min(feasible, key=lambda row: (-row[2], row[3], row[0]))[0]
            assert entry["selected"]["k"] == expected and entry["feasible"]
            assert entry["selected"]["tau"] == 2.0 ** (expected / 2)
        else:
            assert entry["selected"] is None and not entry["feasible"]
    assert result["rules"]["R2"]["selected"] is None and result["rules"]["R2"]["feasible"]
    assert len(result["fixed"]["grid"]) == 8 and len(result["oracle_best_fixed_L_by_hop"]) == 8
    assert all(1 <= depth <= 8 for depth in result["oracle_best_fixed_L_by_hop"])


def test_selection_breaks_ties_by_reads_then_by_k_and_reports_infeasible_rules():
    # every question has p^s = answer for every s: all thresholds tie on the objective; fewer reads win (stop at s = 1: tau >= .5), then the smaller k
    trace = trace_of(np.zeros((20, 8, 9)), change=np.full((20, 8, 8), .5), read_ratio=np.full((20, 8, 8), .5))
    traces = {s: trace for s in rules.SEEDS}
    result = rules.select_thresholds(traces)
    assert result["rules"]["R1"]["selected"]["k"] == -2 and result["rules"]["R3"]["selected"]["k"] == -2   # tau = .5 stops at s = 1
    assert result["fixed"]["selected"]["L"] == 1
    # hop 1 is correct only at L = 4: every stop at another depth loses more than 10 pp, so only stopping at 4 is feasible
    pred = np.ones((20, 8, 9), dtype=np.int32)
    pred[:, 0, 4] = 0
    change = np.full((20, 8, 8), 9.)
    change[..., 3] = .01                                   # stops at 4 when tau >= .01
    always = trace_of(pred, change=change, read_ratio=change)
    result = rules.select_thresholds({s: always for s in rules.SEEDS})
    rows = {row["k"]: row for row in result["rules"]["R1"]["grid"]}
    assert not rows[-20]["feasible"] and rows[-6]["feasible"] and result["rules"]["R1"]["selected"]["k"] <= -6
    # a rule that never leaves the depths where hop 1 is wrong has no feasible value
    broken = trace_of(pred, change=np.full((20, 8, 8), 9.), read_ratio=np.full((20, 8, 8), 9.))
    result = rules.select_thresholds({s: broken for s in rules.SEEDS})
    assert result["rules"]["R1"]["selected"] is None and not result["rules"]["R1"]["feasible"]
    assert result["rules"]["R3"]["selected"] is None


def test_the_pooled_margin_is_8_pp_and_the_per_seed_margin_10_pp():
    # 100 episodes; fixed L = 4 is right on hop 1 for all; a stop at s = 1 is right on `right` of them
    def seed_trace(right):
        pred = np.zeros((100, 8, 9), dtype=np.int32)
        pred[right:, 0, 1] = 1
        change = np.full((100, 8, 8), 9.)
        change[..., 0] = .01
        return trace_of(pred, change=change, read_ratio=change)

    def last_row(right):                                           # tau = 4 stops every question at s = 1
        return rules.select_thresholds({s: seed_trace(n) for s, n in zip(rules.SEEDS, right)})["rules"]["R1"]["grid"][-1]
    assert last_row((92, 92, 92))["feasible"] is True              # pooled -8 pp is allowed
    assert last_row((91, 91, 91))["feasible"] is False             # pooled -9 pp
    row = last_row((99, 99, 87))                                   # pooled -5 pp but one seed at -13 pp
    assert row["hop1_delta"][43] == -13 and row["feasible"] is False
    assert last_row((90, 99, 99))["feasible"] is True              # a seed at -10 pp is allowed


# ------------------------------------------------------------------ adapter level (instrumentation, CPU)

def adapter_case(rows=4, length=7, r=8):
    generator = torch.Generator().manual_seed(derive_seed(17, "lsr-adapter"))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "lsr-adapter-init"))
        adapter = make_adapter(AdapterConfig(d_in=6, d_out=5, r=r, variant="iterq"))
    with torch.no_grad():
        adapter.B.normal_(0, .5, generator=generator)
    x = torch.randn(rows, length, 6, generator=generator)
    memory = torch.randn(rows, r, r, generator=generator)
    valid = torch.arange(length)[None, :] < torch.tensor([7, 5, 2, 6])[:rows, None]
    return adapter, x, memory, valid


def run(adapter, x, memory, valid, steps):
    return adapter.process_sequence(x, memory, steps, valid, torch.zeros_like(valid))[0]


def test_trace_leaves_the_deployed_output_unchanged_and_matches_the_definitions():
    adapter, x, memory, valid = adapter_case()
    last = p1lc.last_positions(valid)
    plain = run(adapter, x, memory, valid, 8)
    trace = {}
    with lsr.instrument(adapter, last, trace=trace):
        traced = run(adapter, x, memory, valid, 8)
    assert torch.equal(traced, plain) and not LEFTOVERS & set(vars(adapter))
    assert trace["seen"].eq(1).all() and trace["change"].shape == (4, 8)
    u = adapter.u(x)
    for row in range(4):
        position = int(last[row])
        z0 = u[row, position][None]
        states = p1lc.read_states(adapter, z0, memory[row][None], 8)
        zs = [z0] + states
        change = [float((zs[s] - zs[s - 1]).norm() / zs[s - 1].norm().clamp_min(1e-12)) for s in range(1, 9)]
        reads = [float(adapter._query(zs[s], memory[row][None], adapter.Q, adapter.U).norm()) for s in range(8)]
        torch.testing.assert_close(trace["change"][row], torch.tensor(change), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(trace["read_ratio"][row, :7], torch.tensor(reads[1:]) / reads[0], rtol=1e-4, atol=1e-6)
        assert torch.isnan(trace["read_ratio"][row, 7])


def test_trace_rejects_other_depths_and_wrong_shapes_and_restores_the_adapter():
    adapter, x, memory, valid = adapter_case()
    last = p1lc.last_positions(valid)
    with pytest.raises(ValueError, match="L = 8"):
        with lsr.instrument(adapter, last, trace={}):
            run(adapter, x, memory, valid, 4)
    assert not LEFTOVERS & set(vars(adapter))
    with pytest.raises(ValueError, match="different number"):
        with lsr.instrument(adapter, last[:3], trace={}):
            run(adapter, x, memory, valid, 8)
    with pytest.raises(ValueError, match="separate"):
        with lsr.instrument(adapter, last, trace={}, depth_rows=torch.ones(4, dtype=torch.long)):
            pass
    with pytest.raises(ValueError, match="one entry"):
        with lsr.instrument(adapter, last, depth_rows=torch.ones(3, dtype=torch.long)):
            pass
    with pytest.raises(RuntimeError, match="boom"):
        with lsr.instrument(adapter, last, trace={}):
            raise RuntimeError("boom")
    assert not LEFTOVERS & set(vars(adapter))


def test_graded_depths_read_each_row_at_its_own_depth_at_every_position():
    adapter, x, memory, valid = adapter_case()
    last = p1lc.last_positions(valid)
    depth = torch.tensor([1, 4, 8, 4])
    with lsr.instrument(adapter, last, depth_rows=depth):
        graded = run(adapter, x, memory, valid, 1)
    for row, steps in enumerate(depth.tolist()):
        expected = run(adapter, x, memory, valid, steps)[row]
        torch.testing.assert_close(graded[row], expected, rtol=1e-6, atol=1e-6)
    assert not LEFTOVERS & set(vars(adapter))
    with lsr.instrument(adapter, last, depth_rows=torch.full((4,), 4)):
        assert torch.equal(run(adapter, x, memory, valid, 1), run(adapter, x, memory, valid, 4))


# -------------------------------------------------------------------- tiny Qwen with an adapter (GPU)

def tiny_config():
    return resolve_config(dict(writer="oracle_random", stage="reader", variant="M1",
        model=dict(r=32, layer=0), task=dict(E=6, Q=2, chain_lengths=[4, 2], n_rel=4, head_sharing=.3),
        train=dict(batch_size=2, depths=[4]),
        eval=dict(episodes=6, batch_size=3, hops=[1, 2, 3, 4], counterfactuals=False)))


@pytest.fixture(scope="module")
def model():
    configure_numerics()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "lsr-tiny-qwen"))
        value = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield value
    assert not hasattr(value, "looplora_adapters")


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def data():
    cfg = tiny_config()
    spec = make_task_spec(cfg, evaluation=True)
    return stratify_queries(generate(spec, "val", 0, 6, 4242), spec, cfg["eval"]["hops"], 4242)


@pytest.fixture
def experiment(model, tokenizer):
    value = LLMExperiment(tiny_config(), model=model, tokenizer=tokenizer)
    with torch.no_grad():
        value.adapter.B.normal_(0, .3, generator=torch.Generator(device="cuda").manual_seed(derive_seed(17, "lsr-B")))
    yield value
    value.close()


def test_scan_gives_the_fixed_depth_predictions_and_the_audit_holds(experiment, data):
    prepared = experiment.prepare(data)
    plain = [experiment.score(prepared, depth)[0].argmax(-1) for depth in range(9)]
    scan = lsr.scan_batch(experiment, prepared, audit=True)
    assert scan["pred"].shape == (6, 4, 9) and scan["change"].shape == scan["read_ratio"].shape == (6, 4, 8)
    for depth in range(9):
        assert np.array_equal(scan["pred"][..., depth], plain[depth].cpu().numpy())
    assert np.isfinite(scan["change"]).all() and np.isnan(scan["read_ratio"][..., 7]).all()
    assert np.isfinite(scan["read_ratio"][..., :7]).all()
    assert not LEFTOVERS & set(vars(experiment.adapter))
    again = lsr.scan_batch(experiment, prepared)
    assert all(np.array_equal(scan[key], again[key], equal_nan=True) for key in scan)


def test_scan_trace_matches_the_p1_l_replica_at_each_final_token(experiment, data):
    prepared = experiment.prepare(data)
    scan = lsr.scan_batch(experiment, prepared)
    _, inputs = experiment.score(prepared, 4, capture_inputs=True)
    B, Q, T, _ = inputs.shape
    u = experiment.adapter.u(inputs.float()).reshape(B * Q, T, -1)
    memory = repeat_snapshot(prepared.snapshot, Q).states[experiment.layer]
    last = p1lc.last_positions(prepared.question_mask)
    adapter = experiment.adapter
    for row in range(B * Q):
        z0 = u[row, int(last[row])][None]
        zs = [z0] + p1lc.read_states(adapter, z0, memory[row][None], 8)
        change = torch.stack([(zs[s] - zs[s - 1]).norm() / zs[s - 1].norm().clamp_min(1e-12) for s in range(1, 9)])
        reads = torch.stack([adapter._query(zs[s], memory[row][None], adapter.Q, adapter.U).norm() for s in range(8)])
        torch.testing.assert_close(torch.from_numpy(scan["change"].reshape(B * Q, 8)[row]).to(change.device), change,
                                   rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(torch.from_numpy(scan["read_ratio"].reshape(B * Q, 8)[row, :7]).to(reads.device),
                                   reads[1:] / reads[0], rtol=1e-4, atol=1e-6)


def test_the_stopped_pass_equals_the_fixed_depth_prediction_at_the_stop_depth(experiment, data):
    prepared = experiment.prepare(data)
    scan = lsr.scan_batch(experiment, prepared)
    B, Q = scan["pred"].shape[:2]
    constant = lsr.stopped_logits(experiment, prepared, torch.full((B * Q,), 4, device=experiment.device))
    assert torch.equal(constant, experiment.score(prepared, 4)[0])
    depth = (np.arange(B * Q) % 8 + 1).reshape(B, Q)
    stopped = lsr.stopped_logits(experiment, prepared, torch.as_tensor(depth.reshape(-1), device=experiment.device))
    assembled = np.take_along_axis(scan["pred"], depth[..., None], axis=-1)[..., 0]
    assert np.array_equal(stopped.argmax(-1).cpu().numpy(), assembled)
    assert not LEFTOVERS & set(vars(experiment.adapter))
