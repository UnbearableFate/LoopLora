"""P1L-C analysis: read-position pairing, the bridge shift and its decision rule, audit and report."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import finalize_p1l_causal as fin
from evaluate_p1l_causal import PATCH_READERS, READERS, reader_depth, require_memory_use
from looplora.llm.p1lc import PATCH_CONDITIONS
from run_p1l_main import SEEDS, digest

NAMES = ("split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct")


def read_samples(depth, deployed, final, episodes=40):
    """Flat read-position samples; deployed(hop, episode) and final(hop, episode) give correctness."""
    rows = [("p1lc", condition, 1000 + episode, hop - 1, hop, depth, 7, bool(rule(hop, episode)))
            for condition, rule in (("memory_only/deployed", deployed), ("memory_only/final_only", final))
            for episode in range(episodes) for hop in range(1, 9)]
    result = {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(NAMES)}
    result["prediction"] = np.where(result["correct"], result["answer"], result["answer"] + 1)
    return result


def even(hop, episode):
    return episode % 2 == 0


def final_position(hop, episode):
    """Equal to deployed at hop 1, correct on half of deployed's episodes at hops 2 and up."""
    return episode % 2 == 0 if hop == 1 else episode % 4 == 0


def cohort(depth, deployed=even, final=final_position, episodes=40):
    return {seed: read_samples(depth, deployed, final, episodes) for seed in SEEDS}


def only(condition, samples):
    keep = samples["condition"] == condition
    return {key: value[keep] for key, value in samples.items()}


def test_read_position_reports_final_only_minus_deployed_by_hop_per_seed_and_pooled():
    result = fin.read_position("M1-L4", cohort(4))
    assert list(result) == ["hop1", "hop2", "hop3", "hops1-3"]
    expected = {"hop1": 0., "hop2": -.25, "hop3": -.25, "hops1-3": -.5 / 3}
    for name, row in result.items():
        assert row["difference"] == pytest.approx(expected[name])
        assert row["ci_low"] <= row["difference"] <= row["ci_high"]
        assert row["confidence"] == .95 and row["resamples"] == 2000 and row["depth_a"] == row["depth_b"] == 4
        assert [item["seed"] for item in row["per_seed"]] == list(SEEDS)
        assert row["comparison"] == "M1-L4@4: final-position memory - deployed reads"
    assert result["hop2"]["accuracy_a"] == pytest.approx(.25) and result["hop2"]["accuracy_b"] == pytest.approx(.5)
    assert result["hop2"]["ci_high"] < 0
    assert result["hop1"]["ci_low"] == result["hop1"]["ci_high"] == 0


def test_read_position_uses_each_readers_trained_depth_and_needs_both_conditions():
    assert fin.read_position("B4", cohort(1))["hop2"]["depth_a"] == 1
    with pytest.raises(ValueError):
        fin.read_position("B4", cohort(4))          # B4 has no L = 4 samples
    with pytest.raises(ValueError):
        fin.read_position("M1-L4", {seed: only("memory_only/deployed", s) for seed, s in cohort(4).items()})


def test_replication_pairs_deployed_reads_across_readers_at_their_trained_depths():
    by_reader = {"M1-L4": cohort(4, deployed=even), "B8": cohort(4, deployed=lambda hop, episode: episode % 4 != 3),
                 "B4": cohort(1, deployed=lambda hop, episode: episode % 5 == 0)}
    narrow, wide = fin.replication(by_reader, .95), fin.replication(by_reader, .975)
    assert list(narrow) == ["M1-L4_vs_B4", "B8_vs_B4"]
    assert narrow["M1-L4_vs_B4"]["difference"] == pytest.approx(.3)
    assert narrow["B8_vs_B4"]["difference"] == pytest.approx(.55)
    assert narrow["B8_vs_B4"]["comparison"] == "B8@4 - B4@1" and narrow["B8_vs_B4"]["hops"] == [2, 3]
    for name in narrow:
        assert wide[name]["ci_low"] <= narrow[name]["ci_low"] and wide[name]["ci_high"] >= narrow[name]["ci_high"]
        assert [item["seed"] for item in narrow[name]["per_seed"]] == list(SEEDS)


def bridge(hit, n=256):
    """Bridge-patch arrays; hit(condition, episode) -> 'B', 'A' or None (answers t_B, t_A or neither)."""
    token = dict(B=5, A=7, none=99)
    prediction = np.asarray([[token[hit(name, e) or "none"] for e in range(n)] for name in PATCH_CONDITIONS],
                            dtype=np.int64)
    return dict(episode_id=np.arange(n) + 1000, episode_index=np.arange(n) + 102000, start=np.arange(n),
                target_a=np.full(n, 70), target_b=np.full(n, 50), bridge_b=np.full(n, 40),
                unrelated=np.full(n, 30), token_t_A=np.full(n, 7), token_t_B=np.full(n, 5),
                conditions=np.asarray(PATCH_CONDITIONS), prediction=prediction,
                norm_a=np.full(n, 3.), norm_b=np.full(n, 2.), norm_c=np.full(n, 4.))


def planted(condition, e, *, b=lambda e: e % 5 < 3, c=lambda e: e % 20 == 0, r=lambda e: e % 20 == 1):
    if condition.startswith("world_"):
        return {"world_A": "A", "world_B": "B", "world_C": None}[condition]
    rule = {"B": b, "C": c, "R": r}[condition[0]]
    return "B" if rule(e) else None


def rate(rule, n=256):
    return float(np.mean([rule(e) for e in range(n)]))


def seeds_with(rules=None):
    """The planted patch behavior for every seed; rules maps a seed to keyword overrides of planted()."""
    rules = rules or {}
    return {seed: bridge(lambda name, e, seed=seed: planted(name, e, **rules.get(seed, {}))) for seed in SEEDS}


def test_planted_bridge_effect_is_a_claim_with_registered_statistics():
    summary = fin.bridge_summary("M1-L4", seeds_with())
    expected = rate(lambda e: e % 5 < 3) - max(rate(lambda e: e % 20 == 0), rate(lambda e: e % 20 == 1))
    final = summary["final"]
    assert final["shift"] == pytest.approx(expected) and final["ci_low"] > 0 and final["claim"] is True
    assert all(row["shift"] == pytest.approx(expected) for row in final["per_seed"])
    assert [row["seed"] for row in final["per_seed"]] == list(SEEDS)
    assert summary["confidence"] == .95 and summary["resamples"] == 2000 and summary["n_episodes"] == 256
    assert summary["threshold"] == .10 and "claim" not in summary["all"]
    pooled = {row["condition"]: row for row in summary["rates"] if row["seed"] == "pooled"}
    assert pooled["world_A"]["rate_t_A"] == 1 and pooled["world_B"]["rate_t_B"] == 1
    assert pooled["B_final"]["rate_t_B"] == pytest.approx(rate(lambda e: e % 5 < 3))
    assert len(summary["rates"]) == (len(SEEDS) + 1) * len(PATCH_CONDITIONS)


def test_a_positive_pooled_shift_is_not_a_claim_unless_every_seed_is_positive():
    never = lambda e: False          # noqa: E731
    summary = fin.bridge_summary("B8", seeds_with({29: dict(b=never)}))
    final = summary["final"]
    assert final["shift"] >= .10 and final["ci_low"] > 0
    assert [row["shift"] > 0 for row in final["per_seed"]] == [True, False, True]
    assert final["claim"] is False


def test_the_shift_subtracts_the_larger_of_the_two_controls():
    summary = fin.bridge_summary("M1-L4", seeds_with({seed: dict(r=lambda e: e % 10 < 3) for seed in SEEDS}))
    for row in summary["final"]["per_seed"]:
        assert row["rate_R"] > row["rate_C"]
        assert row["shift"] == pytest.approx(row["rate_B"] - row["rate_R"])
    summary = fin.bridge_summary("M1-L4", seeds_with({seed: dict(c=lambda e: e % 10 < 3) for seed in SEEDS}))
    for row in summary["final"]["per_seed"]:
        assert row["rate_C"] > row["rate_R"] and row["shift"] == pytest.approx(row["rate_B"] - row["rate_C"])


def test_a_claim_needs_ten_points_even_when_the_interval_is_above_zero():
    never = lambda e: False          # noqa: E731
    below = fin.bridge_summary("M1-L4", seeds_with({seed: dict(b=lambda e: e % 16 == 0, c=never, r=never)
                                                    for seed in SEEDS}))["final"]
    assert below["shift"] == pytest.approx(16 / 256) and below["ci_low"] > 0 and below["claim"] is False
    above = fin.bridge_summary("M1-L4", seeds_with({seed: dict(b=lambda e: e % 8 == 0, c=never, r=never)
                                                    for seed in SEEDS}))["final"]
    assert above["shift"] == pytest.approx(32 / 256) and above["ci_low"] > 0 and above["claim"] is True


def test_a_claim_needs_the_interval_above_zero_not_just_a_large_point_estimate():
    patches = {seed: bridge(lambda name, e: "B" if name == "B_final" and e == 0 else None, n=4) for seed in SEEDS}
    final = fin.bridge_summary("M1-L4", patches)["final"]
    assert final["shift"] == pytest.approx(.25) and all(row["shift"] > 0 for row in final["per_seed"])
    assert final["ci_low"] == 0 and final["claim"] is False


def test_bridge_summary_rejects_mixed_worlds_or_conditions():
    patches = seeds_with()
    patches[43]["token_t_B"] = patches[43]["token_t_B"] + 1
    with pytest.raises(ValueError, match="differ between seeds"):
        fin.bridge_summary("M1-L4", patches)
    patches = seeds_with()
    patches[17]["conditions"] = patches[17]["conditions"][::-1]
    with pytest.raises(ValueError, match="registered list"):
        fin.bridge_summary("M1-L4", patches)


def synthetic_runs(episodes=12, patch_episodes=6):
    """Read-position samples and bridge patches for every reader and seed."""
    read = {reader: cohort(reader_depth(reader), episodes=episodes) for reader in READERS}
    bridges = {reader: {seed: bridge(lambda name, e: planted(name, e), n=patch_episodes) for seed in SEEDS}
               for reader in PATCH_READERS}
    return read, bridges


def test_analysis_tables_and_report_cover_every_registered_quantity():
    result = fin.analyze(*synthetic_runs())
    assert set(result) == {"read_position", "replication", "bridge"}
    assert set(result["replication"]) == {"95", "97.5"} and list(result["bridge"]) == list(PATCH_READERS)
    read_rows, replication_rows, rate_rows, shift_rows = fin.tables(result)
    assert len(read_rows) == 3 * 4 * (1 + len(SEEDS)) and len(replication_rows) == 2 * 2 * (1 + len(SEEDS))
    assert len(rate_rows) == 2 * (len(SEEDS) + 1) * len(PATCH_CONDITIONS)
    assert len(shift_rows) == 2 * 2 * (1 + len(SEEDS))
    text = "\n".join(fin.report_lines(result))
    assert "not evidence of leakage" in text and "Causal bridge claim" in text
    assert "| M1-L4 | final position |" in text and "| B8 | all position |" in text
    assert "B8@4 - B4@1" in text and "not norm-matched" in text
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("rank,passed,authorization", [(32, True, "same"), (128, False, "same"), (128, None, "same"),
                                                        (128, "yes", "same"), (128, True, "other")])
def test_registered_run_requires_the_r128_memory_use_check_to_have_passed(rank, passed, authorization):
    held = dict(status="allowed", gated_conditions=2)
    endpoints = dict(rank=rank, authorization=held if authorization == "same" else dict(held, status="held"))
    if passed is not None:
        endpoints["memory_use_passed"] = passed
    with pytest.raises(ValueError, match="memory-use"):
        require_memory_use(endpoints, held)
    require_memory_use(dict(rank=128, memory_use_passed=True, authorization=held), held)


class Tree:
    """A synthetic on-disk P1L-C evaluation with the registered layout and provenance fields."""

    def __init__(self, tmp_path, monkeypatch, episodes=12, patch_episodes=6):
        self.window = dict(split="test", start=102000, count=episodes, seed=4242)
        monkeypatch.setattr(fin, "REGISTERED_RANGE", self.window)
        monkeypatch.setattr(fin, "PATCH_EPISODES", patch_episodes)
        self.root, self.train = tmp_path / "causal", tmp_path / "train"
        self.authorization = dict(status="allowed", gated_conditions=2)
        self.names = [f"{reader}-s{seed}" for reader in READERS for seed in SEEDS]
        self.paths = {}
        self.manifest = dict(rank=128, runs={name: str(self.train / name) for name in self.names})
        for key, value in (("manifest", self.manifest), ("authorization", self.authorization),
                           ("endpoints", dict(rank=128, memory_use_passed=True, authorization=self.authorization))):
            self.paths[key] = tmp_path / f"{key}.json"
            self.paths[key].write_text(json.dumps(value))
        read, bridges = synthetic_runs(episodes, patch_episodes)
        for name in self.names:
            reader, seed = name.rsplit("-s", 1)
            (self.train / name).mkdir(parents=True)
            (self.train / name / "checkpoint.pt").write_bytes(name.encode())
            out = self.root / name
            out.mkdir(parents=True)
            np.savez_compressed(out / "read-position.npz", **read[reader][int(seed)])
            record = dict(status="complete", name=name, smoke=False, label=reader, depth=reader_depth(reader),
                          window=self.window, checkpoint_hash=digest(self.train / name / "checkpoint.pt"),
                          manifest_hash=digest(self.paths["manifest"]),
                          authorization_hash=digest(self.paths["authorization"]),
                          endpoints_hash=digest(self.paths["endpoints"]),
                          read_position_hash=digest(out / "read-position.npz"), spec_hash="spec", episodes_hash="episodes",
                          read_position_memory_hashes=["m0", "m1"], seconds=1.5, git_commit="c" * 40,
                          source_hash="source", job_id="1.pbs")
            if reader in PATCH_READERS:
                np.savez_compressed(out / "bridge-patch.npz", **bridges[reader][int(seed)])
                record.update(bridge_patch_hash=digest(out / "bridge-patch.npz"), patch_episodes=patch_episodes,
                              worlds_hash="worlds", bridge_memory_hashes=[dict(batch_start=0, A="a", B="b", C="c")])
            self.write(name, record)

    def write(self, name, record):
        (self.root / name / "record.json").write_text(json.dumps(record))

    def record(self, name):
        return json.loads((self.root / name / "record.json").read_text())


def test_audit_loads_every_run_and_rejects_changed_inputs_or_mixed_episodes(tmp_path, monkeypatch):
    tree = Tree(tmp_path, monkeypatch)
    read, patches, artifacts = fin.audit(tree.root, tree.manifest, tree.paths)
    assert set(read) == set(READERS) and set(patches) == set(PATCH_READERS) and len(artifacts) == 9
    assert all(set(by_seed) == set(SEEDS) for by_seed in read.values())
    for key, value in (("episodes_hash", "other"), ("smoke", True), ("window", dict(tree.window, start=100000)),
                       ("checkpoint_hash", "0" * 32), ("endpoints_hash", "0" * 32)):
        record = tree.record("B8-s29")
        tree.write("B8-s29", dict(record, **{key: value}))
        with pytest.raises(ValueError, match="B8-s29|differs across"):
            fin.audit(tree.root, tree.manifest, tree.paths)
        tree.write("B8-s29", record)
    fin.audit(tree.root, tree.manifest, tree.paths)
    (tree.train / "M1-L4-s17" / "checkpoint.pt").write_bytes(b"changed after evaluation")
    with pytest.raises(ValueError, match="M1-L4-s17"):
        fin.audit(tree.root, tree.manifest, tree.paths)


@pytest.mark.parametrize("key,other", [("source_hash", "another tree"), ("git_commit", "d" * 40)])
def test_audit_requires_one_source_for_all_nine_runs(tmp_path, monkeypatch, key, other):
    tree = Tree(tmp_path, monkeypatch)
    for name in (name for name in tree.names if name.endswith("-s29")):    # one per-seed job rerun from another source
        tree.write(name, dict(tree.record(name), **{key: other}))
    with pytest.raises(ValueError, match=f"{key} differs across"):
        fin.audit(tree.root, tree.manifest, tree.paths)
    for name in tree.names:                                                # one (different) source for all nine is fine
        tree.write(name, dict(tree.record(name), **{key: other}))
    assert len(fin.audit(tree.root, tree.manifest, tree.paths)[2]) == 9


def test_finalize_writes_results_tables_and_manifest_only_after_the_gate_and_audit(tmp_path, monkeypatch):
    tree = Tree(tmp_path, monkeypatch)
    monkeypatch.setattr(fin, "check_training", lambda manifest, authorization, check_all: None)
    args = argparse.Namespace(root=tree.root, manifest=tree.paths["manifest"], authorization=tree.paths["authorization"],
                              endpoints=tree.paths["endpoints"], out=tmp_path / "reports")
    fin.finalize(args)
    result = json.loads((args.out / "results.json").read_text())
    assert set(result) >= {"read_position", "replication", "bridge", "registered_range", "patch_episodes"}
    assert result["registered_range"] == tree.window and result["patch_episodes"] == 6
    for stem in ("read-position", "replication", "bridge-rates", "bridge-shift"):
        assert (args.out / f"{stem}.csv").read_text().startswith(("reader", "contrast"))
    manifest = json.loads((args.out / "run-manifest.json").read_text())
    assert manifest["status"] == "audited" and len(manifest["runs"]) == 9 and manifest["seconds"] == 13.5
    assert "Causal bridge claim" in (args.out / "results.md").read_text()
    tree.paths["endpoints"].write_text(json.dumps(dict(rank=128, memory_use_passed=False,
                                                       authorization=tree.authorization)))
    with pytest.raises(ValueError, match="memory-use"):
        fin.finalize(args)
