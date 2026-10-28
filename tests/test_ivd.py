"""IVD (F8) construction: off-path extras, constructed/selected overlap, decay-only control.

Run only inside a Miyabi-G compute allocation, including the CPU cases.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from looplora.data.relations import generate
from looplora.sym import ivd
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.evaluate import stratify_queries
from looplora.sym.followup_stats import aligned_terms
from looplora.sym.model import SymbolicModel, make_task_spec

SEED = 999004


def _config(writer, **updates):
    base = {
        "name": "ivd-tiny", "device": "cpu", "seed": 17, "writer": writer, "variant": "M1",
        "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 8},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 96, "n_rel": 4,
                 "seen": [0, 72], "unseen": [72, 96], "chain_lengths": [4, 4], "hops": [1, 2, 3, 4]},
        "train": {"steps": 1, "batch_size": 2, "warmup_steps": 0, "depths": [1, 4], "log_every": 1},
        "eval": {"episodes": 6, "batch_size": 3, "splits": ["test"], "depths": [1, 4],
                 "hops": [1, 2, 3, 4], "counterfactuals": False},
    }
    return resolve_config(recursive_merge(base, updates))


def _episodes(config, count=6, seed=4242):
    spec = make_task_spec(config, evaluation=True)
    arrays = stratify_queries(generate(spec, "test", 0, count, seed), spec, config["eval"]["hops"], seed)
    return spec, arrays


def _setup(writer, *, count=6, n_extra=4, candidates=24):
    config = _config(writer)
    model = SymbolicModel(config)
    model.eval()
    spec, arrays = _episodes(config, count)
    entities = ivd.episode_entities(arrays)
    heads, tails = ivd.draw_extras(arrays, entities, seed=SEED, n_extra=n_extra, pool=spec.seen,
                                   candidates=candidates)
    is_extra = ivd.slot_layout(arrays.episode_seed, arrays.facts.shape[1], n_extra, SEED)
    return model, spec, arrays, entities, heads, tails, is_extra


@pytest.mark.parametrize("writer", ["oracle_random", "joint_fact"])
def test_zero_load_reproduces_the_episode_and_its_memory(writer):
    model, _, arrays, _, heads, tails, is_extra = _setup(writer)
    extended, flags = ivd.assemble(arrays, heads, tails, is_extra, 0)
    assert np.array_equal(extended.facts, arrays.facts)
    assert not flags.any()
    assert torch.equal(model.prepare(arrays).memory, model.prepare(extended).memory)


def test_relevant_facts_keep_their_order_and_smaller_loads_are_subsequences():
    _, _, arrays, _, heads, tails, is_extra = _setup("joint_fact", n_extra=6)
    small, small_flags = ivd.assemble(arrays, heads, tails, is_extra, 3)
    large, large_flags = ivd.assemble(arrays, heads, tails, is_extra, 6)
    assert large.facts.shape[1] == arrays.facts.shape[1] + 6
    assert small.facts.shape[1] == arrays.facts.shape[1] + 3
    for b in range(len(arrays.episode_seed)):
        assert np.array_equal(large.facts[b][~large_flags[b]], arrays.facts[b])
        assert np.array_equal(small.facts[b][~small_flags[b]], arrays.facts[b])
        rank = np.cumsum(large_flags[b]) - 1
        keep = ~large_flags[b] | (rank < 3)
        assert np.array_equal(large.facts[b][keep], small.facts[b])
        extras = large.facts[b][large_flags[b]]
        assert np.array_equal(extras[:, 0], heads[b, :6])
        assert np.array_equal(extras[:, 2], tails[b, :6])
        assert np.all(extras[:, 1] == 0)


def test_extras_are_off_path_with_tails_inside_the_episode_and_are_deterministic():
    _, spec, arrays, entities, heads, tails, _ = _setup("joint_fact", n_extra=6)
    for b in range(len(arrays.episode_seed)):
        assert not np.isin(heads[b], entities[b]).any()
        assert np.isin(tails[b], entities[b]).all()
        assert len(np.unique(heads[b])) == heads.shape[1]
        assert heads[b].min() >= spec.seen[0] and heads[b].max() < spec.seen[1]
        assert not np.isin(arrays.q_start[b], heads[b]).any()
    again = ivd.draw_extras(arrays, entities, seed=SEED, n_extra=6, pool=spec.seen, candidates=24)
    assert np.array_equal(again[0], heads) and np.array_equal(again[1], tails)
    other = ivd.draw_extras(arrays, entities, seed=SEED + 1, n_extra=6, pool=spec.seen, candidates=24)
    assert not np.array_equal(other[0], heads)


def test_prepare_cell_keeps_the_original_candidate_set():
    model, _, arrays, _, heads, tails, is_extra = _setup("joint_fact", n_extra=4)
    extended, _ = ivd.assemble(arrays, heads, tails, is_extra, 4)
    base = model.prepare(arrays)
    assert (model.prepare(extended).candidate_mask != base.candidate_mask).any()
    cell = ivd.prepare_cell(model, extended, candidate_mask=base.candidate_mask)
    assert torch.equal(cell.candidate_mask, base.candidate_mask)
    assert torch.equal(cell.answer, base.answer)
    assert cell.candidate_mask.gather(1, cell.answer).all()


def test_constructed_oracle_codes_have_the_requested_overlap():
    model, _, arrays, entities, heads, _, _ = _setup("oracle_random", n_extra=6)
    facts = torch.as_tensor(arrays.facts, dtype=torch.long)
    ents = torch.as_tensor(entities, dtype=torch.long)
    extra_heads = torch.as_tensor(heads[:, :6], dtype=torch.long)
    codes_global = model.writer.global_codes
    rows = torch.arange(len(facts))[:, None]
    for kind in ivd.OVERLAPS:
        codes = ivd.constructed_codes(codes_global, facts, ents, extra_heads, kind,
                                      episode_seed=arrays.episode_seed, seed=7)
        assert torch.equal(codes[rows, ents], codes_global[ents])
        extra = codes[rows, extra_heads]
        torch.testing.assert_close(extra.norm(dim=-1), torch.ones(extra.shape[:2]), atol=1e-5, rtol=0)
        if kind == "nat":
            assert torch.equal(codes, codes_global.unsqueeze(0).expand_as(codes))
        elif kind == "low":
            cosines = torch.einsum("bnr,ber->bne", extra, codes[rows, ents])
            assert cosines.abs().max() < 1e-5
        else:
            target = codes[rows, facts[:, torch.arange(6) % facts.shape[1], 0]]
            torch.testing.assert_close((extra * target).sum(-1), torch.full((len(facts), 6), 0.5),
                                       atol=1e-5, rtol=0)
    again = ivd.constructed_codes(codes_global, facts, ents, extra_heads, "high",
                                  episode_seed=arrays.episode_seed, seed=7)
    assert torch.equal(again, ivd.constructed_codes(codes_global, facts, ents, extra_heads, "high",
                                                    episode_seed=arrays.episode_seed, seed=7))


def test_oracle_low_overlap_extras_leave_reads_of_relevant_keys_unchanged_and_high_overlap_does_not():
    model, _, arrays, entities, heads, tails, is_extra = _setup("oracle_random", n_extra=6)
    base = model.prepare(arrays)
    extended, _ = ivd.assemble(arrays, heads, tails, is_extra, 6)
    facts = torch.as_tensor(arrays.facts, dtype=torch.long)
    ents = torch.as_tensor(entities, dtype=torch.long)
    rows = torch.arange(len(facts))[:, None]
    reads = {}
    for kind in ("low", "high"):
        codes = ivd.constructed_codes(model.writer.global_codes, facts, ents,
                                      torch.as_tensor(heads[:, :6], dtype=torch.long), kind,
                                      episode_seed=arrays.episode_seed, seed=7)
        cell = ivd.prepare_cell(model, extended, candidate_mask=base.candidate_mask, codes=codes)
        reads[kind] = torch.einsum("bij,bkj->bki", cell.memory, codes[rows, facts[:, :, 0]])
    expected = torch.einsum("bij,bkj->bki", base.memory, model.writer.global_codes[facts[:, :, 0]])
    torch.testing.assert_close(reads["low"], expected, atol=1e-4, rtol=0)
    assert (reads["high"] - expected).abs().max() > 1e-2


def test_constructed_codes_change_the_write_keys_but_not_the_decode_table():
    model, _, arrays, entities, heads, tails, is_extra = _setup("oracle_random", n_extra=6)
    base = model.prepare(arrays)
    extended, _ = ivd.assemble(arrays, heads, tails, is_extra, 6)
    facts = torch.as_tensor(arrays.facts, dtype=torch.long)
    ents = torch.as_tensor(entities, dtype=torch.long)
    extra_heads = torch.as_tensor(heads[:, :6], dtype=torch.long)
    codes = ivd.constructed_codes(model.writer.global_codes, facts, ents, extra_heads, "high",
                                  episode_seed=arrays.episode_seed, seed=7)
    own = ivd.prepare_cell(model, extended, candidate_mask=base.candidate_mask, codes=codes)
    shared = ivd.prepare_cell(model, extended, candidate_mask=base.candidate_mask, codes=codes,
                              decode_codes=base.codes)
    assert not torch.equal(own.codes, base.codes)  # the extras' heads carry constructed codes in the cell's own table
    assert torch.equal(shared.codes, base.codes)
    for name in ("memory", "u", "true_value", "candidate_mask", "retrieval_valid"):  # nothing else changes
        assert torch.equal(getattr(shared, name), getattr(own, name)), name
    for depth in (1, 4):  # the episode's entities keep their codes, so the candidate-restricted decode is unchanged
        restricted = [model.read(cell, depth)["logits"].masked_fill(~base.candidate_mask[:, None, :], -torch.inf)
                      for cell in (own, shared)]
        torch.testing.assert_close(restricted[0], restricted[1], atol=1e-6, rtol=0)


def test_learned_overlap_selection_orders_candidates_by_measured_overlap():
    model, _, arrays, _, heads, _, _ = _setup("joint_fact", n_extra=4, candidates=24)
    facts = torch.as_tensor(arrays.facts, dtype=torch.long)
    pool = torch.as_tensor(heads, dtype=torch.long)
    count, n = len(facts), 8
    shuffled = torch.as_tensor(ivd.shuffle_order(arrays.episode_seed, n, 5), dtype=torch.long)
    identity = torch.arange(n).expand(count, -1)
    relevant = ivd.write_keys(model, facts)
    chosen = {kind: ivd.select_by_overlap(model, facts, pool, kind, n, shuffled) for kind in ivd.OVERLAPS}
    for kind, (selected, rho) in chosen.items():
        zeros = torch.zeros_like(selected)
        recomputed = ivd.overlap(ivd.write_keys(model, torch.stack((selected, zeros, zeros), dim=-1)), relevant)
        torch.testing.assert_close(rho, recomputed, atol=1e-6, rtol=0)
        for b in range(count):
            assert np.isin(selected[b].numpy(), heads[b]).all()
    assert (chosen["low"][1].amax(dim=1) <= chosen["high"][1].amin(dim=1)).all()
    assert chosen["low"][1].mean() < chosen["nat"][1].mean() < chosen["high"][1].mean()
    for kind in ("low", "high"):
        plain, plain_rho = ivd.select_by_overlap(model, facts, pool, kind, n, identity)
        assert torch.equal(plain.sort(dim=1).values, chosen[kind][0].sort(dim=1).values)
        steps = plain_rho[:, 1:] - plain_rho[:, :-1]
        assert (steps >= 0).all() if kind == "low" else (steps <= 0).all()
    nat, _ = ivd.select_by_overlap(model, facts, pool, "nat", n, identity)
    assert torch.equal(nat, pool[:, :n])
    assert torch.equal(torch.sort(shuffled, dim=1).values, identity)


def test_decay_only_control_matches_the_writer_and_pure_decay():
    model, _, arrays, _, heads, tails, is_extra = _setup("joint_fact", n_extra=3)
    layout = np.zeros_like(is_extra)
    layout[:, arrays.facts.shape[1]:] = True
    extended, flags = ivd.assemble(arrays, heads, tails, layout, 3)
    assert flags[:, -3:].all() and not flags[:, :-3].any()
    torch.testing.assert_close(ivd.decay_only_memory(model, extended, np.zeros_like(flags)),
                               model.prepare(extended).memory, atol=1e-6, rtol=1e-5)
    decay = model.writer.a_lambda.sigmoid().item() ** 3
    torch.testing.assert_close(ivd.decay_only_memory(model, extended, flags),
                               decay * model.prepare(arrays).memory, atol=1e-6, rtol=1e-5)


def test_cell_names_cover_both_writers_and_reject_others():
    assert ivd.cell_names("oracle_random", {"L2": 2}) == ["base", "L2-low", "L2-nat", "L2-high"]
    assert ivd.cell_names("joint_fact", {"L2": 2}) == ["base", "L2-low", "L2-nat", "L2-high", "L2-decay"]
    with pytest.raises(ValueError):
        ivd.cell_names("joint")


@pytest.mark.parametrize("writer", ["oracle_random", "joint_fact"])
def test_evaluate_cells_pairs_every_cell_with_the_baseline(writer):
    config = _config(writer)
    model = SymbolicModel(config)
    spec, arrays = _episodes(config, 6)
    loads = {"L2": 2, "L4": 4}
    names = ivd.cell_names(writer, loads)
    kwargs = dict(depths=[1, 4], seed=SEED, pool=spec.seen, loads=loads, candidates=24)
    samples = ivd.evaluate_cells(model, arrays, batch_size=4, **kwargs)
    assert set(samples["condition"].tolist()) == set(names)
    terms = [("test", name, depth) for name in names for depth in (1, 4)]
    aligned = aligned_terms({17: samples}, terms)
    assert len(aligned["episode_id"]) == 6 * 4
    # The baseline cell is the unmodified evaluation path.
    base = model.prepare(arrays)
    index = {int(seed): i for i, seed in enumerate(arrays.episode_seed)}
    for depth in (1, 4):
        logits = model.read(base, depth)["logits"]
        expected = logits.masked_fill(~base.candidate_mask[:, None, :], -torch.inf).argmax(-1).numpy()
        selected = np.flatnonzero((samples["condition"] == "base") & (samples["L"] == depth))
        assert len(selected) == expected.size
        for row in selected:
            assert samples["prediction"][row] == expected[index[int(samples["episode_id"][row])],
                                                           int(samples["query_id"][row])]
    # Batch size does not change any cell.
    other = ivd.evaluate_cells(model, arrays, batch_size=2, **kwargs)

    def ordered(values):
        order = np.lexsort((values["query_id"], values["episode_id"], values["L"], values["condition"]))
        return {name: array[order] for name, array in values.items()}

    first, second = ordered(samples), ordered(other)
    for name in ("condition", "episode_id", "query_id", "hop", "L", "answer", "candidate_size", "prediction"):
        assert np.array_equal(first[name], second[name]), name
    for name in ("read_residual", "overlap_mean", "overlap_max"):
        np.testing.assert_allclose(first[name], second[name], rtol=1e-4, atol=1e-5, equal_nan=True)
    # Overlap ordering of the achieved extras, and NaN for the baseline.
    summary = {row["condition"]: row for row in ivd.overlap_summary(samples)}
    assert summary["base"]["overlap_mean"] is None
    for load in loads:
        assert summary[f"{load}-low"]["overlap_mean"] < summary[f"{load}-high"]["overlap_mean"]
        if writer == "joint_fact":  # a decay-only cell writes nothing, so it reports no overlap
            assert summary[f"{load}-decay"]["overlap_mean"] is None


def _synthetic_cells(index, *, high_gain, low_gain=50, writer="oracle_random", episodes=100):
    """Samples with exact depth gains: the leading episodes are correct (nested between L1 and L4).

    Gains are counts of episodes out of ``episodes``, so a gain of g is g / episodes in accuracy. Every
    cell is correct on 10 episodes at L1; at L4 it gains ``low_gain`` (low), ``high_gain`` (high) or 25
    (every other cell) episodes, plus ``index`` episodes in all cells (a shift that cancels in contrasts).
    """
    columns = {name: [] for name in ("split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct",
                                     "full_correct", "read_residual", "overlap_mean", "overlap_max")}
    for condition in ivd.cell_names(writer):
        kind = condition.partition("-")[2]
        gain = {"low": low_gain, "high": high_gain}.get(kind, 25)
        for depth in (1, 4):
            correct_count = 10 if depth == 1 else 10 + gain + index
            for episode in range(episodes):
                for hop in (1, 2, 3):
                    row = dict(split="test", condition=condition, episode_id=5000 + episode, query_id=hop - 1,
                               hop=hop, L=depth, answer=1, correct=episode < correct_count,
                               full_correct=episode < correct_count and episode % 2 == 0,
                               read_residual=0.1 * (1 + (kind == "high")),
                               overlap_mean=float("nan") if condition == "base" else (0.5 if kind == "high" else 0.05),
                               overlap_max=0.0)
                    for name, value in row.items():
                        columns[name].append(value)
    arrays = {name: np.asarray(values) for name, values in columns.items()}
    arrays["split"], arrays["condition"] = arrays["split"].astype(str), arrays["condition"].astype(str)
    return arrays


def _seeds(**options):
    return {seed: _synthetic_cells(index, **options) for index, seed in enumerate((17, 29, 43))}


def test_interference_contrast_recovers_the_difference_of_depth_gains_and_the_registered_pattern():
    result = ivd.interference_contrast(_seeds(high_gain=25))
    assert result["difference"] == pytest.approx(0.25, abs=1e-12)  # (50 - 25) / 100; the index shift cancels
    assert result["passes"] and result["ci_low"] > 0 and result["confidence"] == 0.975
    clear = ivd.interference_contrast(_seeds(high_gain=42))
    assert clear["difference"] == pytest.approx(0.08, abs=1e-12) and clear["ci_low"] > 0 and clear["passes"]
    small = ivd.interference_contrast(_seeds(high_gain=47))
    assert small["difference"] == pytest.approx(0.03, abs=1e-12) and not small["passes"]  # below 5 pp
    flat = ivd.interference_contrast(_seeds(high_gain=50))
    assert flat["difference"] == pytest.approx(0.0, abs=1e-12) and not flat["passes"]
    reversed_ = ivd.interference_contrast(_seeds(high_gain=60))
    assert reversed_["difference"] == pytest.approx(-0.10, abs=1e-12) and not reversed_["passes"]
    assert ivd.interference_contrast(_seeds(high_gain=25), load="L6")["difference"] == pytest.approx(0.25, abs=1e-12)


def test_interference_contrast_uses_only_the_requested_hops_and_the_paired_cells():
    seeds = _seeds(high_gain=25)
    for samples in seeds.values():  # make hop 1 contradict the effect: it must not enter the hops 2-3 contrast
        rows = (samples["hop"] == 1) & (samples["condition"] == "L24-high") & (samples["L"] == 4)
        samples["correct"][rows] = True
    assert ivd.interference_contrast(seeds)["difference"] == pytest.approx(0.25, abs=1e-12)
    assert ivd.interference_contrast(seeds, hops=(1, 2, 3))["difference"] < 0.25
    broken = {seed: dict(samples) for seed, samples in seeds.items()}
    keep = ~((broken[17]["condition"] == "L24-high") & (broken[17]["L"] == 1) & (broken[17]["episode_id"] == 5000))
    broken[17] = {name: values[keep] for name, values in broken[17].items()}
    with pytest.raises(ValueError):  # unpaired queries are never intersected or filled
        ivd.interference_contrast(broken)


def test_depth_gain_and_cell_table_report_the_pooled_gain_accuracy_and_overlap():
    gain = ivd.depth_gain(_seeds(high_gain=25), "L24-low")
    assert gain["difference"] == pytest.approx(0.5 + np.mean([0, 1, 2]) / 100, abs=1e-12)
    assert gain["accuracy_l1"] == pytest.approx(0.10) and gain["confidence"] == 0.95
    assert gain["accuracy_l4"] - gain["accuracy_l1"] == pytest.approx(gain["difference"], abs=1e-12)
    rows = ivd.cell_table(_seeds(high_gain=25, writer="joint_fact"), "joint_fact")
    assert [row["condition"] for row in rows] == ivd.cell_names("joint_fact")
    by_name = {row["condition"]: row for row in rows}
    assert by_name["base"]["overlap_mean"] is None
    assert by_name["L24-high"]["overlap_mean"] == pytest.approx(0.5)
    assert by_name["L24-low"]["overlap_mean"] == pytest.approx(0.05)
    assert by_name["L24-high"]["read_residual_l1"] == pytest.approx(0.2)
    assert by_name["L24-low"]["gain"] > by_name["L24-high"]["gain"]
    assert by_name["L24-low"]["gain_seed_29"] == pytest.approx(0.51)
    assert all(row["gain_ci_low"] <= row["gain"] <= row["gain_ci_high"] for row in rows)


def test_accuracy_table_matches_the_exact_cell_accuracies_and_pools_the_requested_hops():
    seeds = _seeds(high_gain=25)
    rows = ivd.accuracy_table(seeds, "oracle_random", depth=1)
    assert [row["condition"] for row in rows] == ivd.cell_names("oracle_random")
    assert all(row["accuracy"] == pytest.approx(0.10) and row["ci_low"] <= row["accuracy"] <= row["ci_high"]
               for row in rows)
    four = {row["condition"]: row for row in ivd.accuracy_table(seeds, "oracle_random", depth=4, hops=(1,))}
    assert four["L24-low"]["accuracy"] == pytest.approx(0.60 + np.mean([0, 1, 2]) / 100, abs=1e-12)
    assert four["L24-low"]["hops"] == "1"
    with pytest.raises(ValueError):
        ivd.accuracy_table(seeds, "oracle_random", depth=2)


def test_the_full_vocabulary_field_is_analysed_like_the_candidate_restricted_one():
    seeds = _seeds(high_gain=25)  # full_correct holds on the even episodes among the correct ones
    gain = ivd.depth_gain(seeds, "L24-low", field="full_correct")
    assert gain["accuracy_l1"] == pytest.approx(0.05)  # even episodes below 10
    assert gain["difference"] == pytest.approx(np.mean([30 - 5, 31 - 5, 31 - 5]) / 100, abs=1e-12)
    rows = ivd.cell_table(seeds, "oracle_random", field="full_correct")
    assert {row["field"] for row in rows} == {"full_correct"}
    assert {row["field"] for row in ivd.cell_table(seeds, "oracle_random")} == {"correct"}
    table = ivd.accuracy_table(seeds, "oracle_random", depth=1, field="full_correct")
    assert all(row["accuracy"] == pytest.approx(0.05) and row["field"] == "full_correct" for row in table)
    default = ivd.depth_gain(seeds, "L24-low")  # the registered statistic still uses the restricted field
    assert default["accuracy_l1"] == pytest.approx(0.10)
