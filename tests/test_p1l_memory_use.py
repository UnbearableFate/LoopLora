"""Amendment B7: the hop-1 memory-use check is computed beside, never inside, the registered gates."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from finalize_p1l_main import (MEMORY_USE_CONTRASTS, endpoints, memory_use_check, memory_use_lines,
                               memory_use_rows)
from run_p1l_main import LABELS, SEEDS

EPISODES = 40
READERS = {"B4": 1, "B8": 4, "M1-L4": 4}


def sample(depths, correct):
    """Confirmation-format samples of 40 episodes x 8 hops; correct(hop, episode) gives the outcome."""
    rows = [("confirm", "memory_only/original", episode, hop - 1, hop, depth, 7, bool(correct(hop, episode)))
            for depth in depths for episode in range(EPISODES) for hop in range(1, 9)]
    names = ("split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct")
    return {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(names)}


def uses_memory(hop, episode):
    return episode % 5 != 0       # 80% correct


def no_memory(hop, episode):
    return episode % 10 == 0      # 10% correct, only on episodes the readers get wrong


def never(hop, episode):
    return False


def cohorts(**rules):
    """Rank-128 cohorts: readers use the memory and B3-L4 does not, unless a rule says otherwise."""
    default = {"B4": uses_memory, "B8": uses_memory, "M1-L4": uses_memory, "B5-L4": never, "B3-L4": no_memory}
    default.update(rules)
    result = {}
    for label in LABELS[:5]:
        depths = range(1, 9) if label == "M1-L4" else [1] if label == "B4" else [4]
        result[label] = {seed: sample(depths, default[label]) for seed in SEEDS}
    return result


def test_memory_use_passes_when_every_reader_beats_no_memory_at_hop_one():
    memory = memory_use_check(cohorts(), .975)
    assert memory["passed"] is True and memory["rule"] == "B7.1" and memory["hop"] == 1
    assert list(memory["contrasts"]) == [name for name, _, _ in MEMORY_USE_CONTRASTS]
    assert [row["comparison"] for row in memory["contrasts"].values()] == [
        "B4@1 - B3-L4@4", "B8@4 - B3-L4@4", "M1-L4@4 - B3-L4@4"]
    for row in memory["contrasts"].values():
        assert row["hops"] == [1] and row["confidence"] == .975 and row["resamples"] == 2000
        assert row["n_episodes"] == EPISODES and [item["seed"] for item in row["per_seed"]] == list(SEEDS)
        assert row["difference"] == pytest.approx(.7) and row["ci_low"] > 0
        assert all(item["difference"] == pytest.approx(.7) for item in row["per_seed"])


@pytest.mark.parametrize("reader", sorted(READERS))
def test_memory_use_fails_when_any_one_reader_is_not_above_the_no_memory_loop(reader):
    memory = memory_use_check(cohorts(**{reader: no_memory}), .975)
    assert memory["passed"] is False
    flags = {name: row["ci_low"] > 0 for name, row in memory["contrasts"].items()}
    assert [name for name, ok in flags.items() if not ok] == [f"{reader.split('-')[0]}_vs_B3_hop1"]


def test_memory_use_reads_hop_one_and_nothing_else():
    assert memory_use_check(cohorts(**{label: lambda hop, episode: hop == 1 and uses_memory(hop, episode)
                                       for label in READERS}), .975)["passed"] is True
    assert memory_use_check(cohorts(**{label: lambda hop, episode: hop > 1 and uses_memory(hop, episode)
                                       for label in READERS}), .975)["passed"] is False


def test_memory_use_is_beside_the_registered_gates_and_changes_nothing_in_them():
    primary = cohorts(**{"B8": uses_memory})
    typed = copy.deepcopy(primary)
    frozen = copy.deepcopy(primary)
    before = endpoints(primary, typed, .975)
    memory = memory_use_check(primary, .975)
    memory_use_rows(128, memory), memory_use_lines(128, memory)
    assert endpoints(primary, typed, .975) == before
    assert set(before) == {"writer", "confidence", "G1_L", "G2_L", "passes", "contrasts"}
    assert set(before["contrasts"]).isdisjoint(name for name, _, _ in MEMORY_USE_CONTRASTS)
    for label in primary:
        for seed in SEEDS:
            assert all(np.array_equal(primary[label][seed][key], frozen[label][seed][key])
                       for key in frozen[label][seed])


def test_memory_use_table_and_report_lines_carry_pooled_and_per_seed_values():
    memory = memory_use_check(cohorts(), .975)
    rows = memory_use_rows(128, memory)
    assert len(rows) == 3 * (1 + len(SEEDS))
    pooled = [row for row in rows if row["seed"] == "pooled"]
    assert [row["contrast"] for row in pooled] == [name for name, _, _ in MEMORY_USE_CONTRASTS]
    assert all(row["ci_above_zero"] and row["confidence"] == .975 and row["rank"] == 128 for row in pooled)
    assert sorted({row["seed"] for row in rows if row["seed"] != "pooled"}) == list(SEEDS)
    text = "\n".join(memory_use_lines(128, memory))
    assert "Memory-use check: **pass**" in text and "Rule 3" in text and "Rule 2" not in text
    assert "did not learn" not in text and "B8_vs_B3_hop1" in text and "17:70.00" in text


def test_report_lines_state_the_rank_specific_interpretation_rules():
    failed = memory_use_check(cohorts(**{label: no_memory for label in READERS}), .975)
    text = "\n".join(memory_use_lines(32, failed))
    assert "Memory-use check: **fail**" in text
    assert "did not learn to use the oracle memory" in text and "not interpreted" in text
    assert "Rule 2: no claim about multiple reads in the LM at r = 32" in text
    passed = memory_use_check(cohorts(), .975)
    text = "\n".join(memory_use_lines(32, passed))
    assert "Rule 2" in text and "Rule 3" not in text
    text = "\n".join(memory_use_lines(128, failed))
    assert "did not learn" in text and "Rule 2" not in text and "Rule 3" not in text
