"""Synthetic release, integrity, cache and reporting regressions; no real runs."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from looplora.infra.seeds import stable_hash

_SPEC = importlib.util.spec_from_file_location(
    "analyze_p1s_main", Path(__file__).parents[1] / "scripts" / "analyze_p1s_main.py")
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _release(tmp_path):
    """Complete metadata with intentionally unreadable sample/checkpoint files."""
    root = tmp_path / "main"
    specs = [("writer", "joint", "B4", seed, 32) for seed in analysis.SEEDS]
    specs += [("primary", writer, variant, seed, 32) for writer in analysis.WRITERS
              for variant in analysis.VARIANTS for seed in analysis.SEEDS]
    specs += [("replication", writer, variant, seed, 32)
              for writer in ("oracle_random", "t1_frozen", "joint")
              for variant in analysis.READERS for seed in analysis.SEEDS]
    specs += [("secondary", writer, variant, 17, 16)
              for writer in ("oracle_random", "oracle_orth", "joint")
              for variant in ("M1", "M1-lin", "B5", "B5-lin")]
    runs = []
    for i, (group, writer, variant, seed, rank) in enumerate(specs):
        name = f"run-{i}"
        folder = root / "runs" / name
        folder.mkdir(parents=True)
        config = {"train": {"steps": 20000, "lr": .003}, "eval": dict(seed=4242, episodes=2000,
                  splits=["test", "test_unseen"], counterfactuals=True, depths=list(analysis.DEPTHS), hops=list(analysis.HOPS))}
        run = dict(name=name, group=group, writer=writer, variant=variant, seed=seed,
                   r=rank, config=config, run_dir=str(folder), dependencies=[])
        runs.append(run)
        record = dict(status="complete", training_complete=True, config=config,
                      config_hash=stable_hash(config), planned_config_hash=stable_hash(config),
                      confirmation_samples_written=group != "writer")
        _write(folder / "record.json", record)
        _write(folder / "config.json", {"config": config, "config_hash": stable_hash(config)})
        (folder / "checkpoint.pt").write_text("not a tensor archive")
        (folder / "train_log.jsonl").write_text("not a training log")
        if group == "writer":
            (folder / "writer.pt").write_text("not a tensor archive")
        else:
            evaluation_config = copy.deepcopy(config)
            evaluation_config["eval"].update(seed=777001, splits=["test"], counterfactuals=False)
            metadata = dict(aggregation_deferred=True, evaluation_seed=777001, config=evaluation_config,
                            config_hash=stable_hash(evaluation_config), checkpoint_config=config,
                            checkpoint_config_hash=stable_hash(config))
            _write(folder / "confirmation" / "evaluation.json", metadata)
            (folder / "confirmation" / "samples.npz").write_text("not a numpy archive")
    plan = dict(root=str(root), steps=20000, runs=runs)
    _write(root / "plan.json", plan)
    barrier = dict(status="passed", plan_hash=stable_hash(plan), created_at="2026-10-26T23:00:00+09:00",
                   runs=[run["name"] for run in runs], confirmation_samples=[str(Path(run["run_dir"]) /
                   "confirmation" / "samples.npz") for run in runs if run["group"] != "writer"])
    _write(root / "barrier.json", barrier)
    return root / "plan.json", plan, barrier


def test_release_audits_all_metadata_without_opening_sample_or_tensor_archives(tmp_path, monkeypatch):
    plan_path, plan, barrier = _release(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("release verification opened a numpy archive")

    monkeypatch.setattr(analysis.np, "load", forbidden)
    assert analysis.verify_release(plan_path) == (plan, barrier)


@pytest.mark.parametrize("failure", ["status", "last_run", "missing_sample", "plan_hash", "missing_member", "wrong_matrix"])
def test_incomplete_global_release_blocks_every_confirmation_read(tmp_path, monkeypatch, failure):
    plan_path, plan, barrier = _release(tmp_path)
    if failure == "status":
        barrier["status"] = "waiting"
    elif failure == "last_run":
        path = Path(plan["runs"][-1]["run_dir"]) / "record.json"
        record = json.loads(path.read_text())
        record["status"] = "running"
        _write(path, record)
    elif failure == "missing_sample":
        Path(barrier["confirmation_samples"][-1]).rename(plan_path.parent / "not-samples.npz")
    elif failure == "plan_hash":
        barrier["plan_hash"] = "different"
    elif failure == "missing_member":
        barrier["confirmation_samples"].pop()
    else:
        plan["runs"][-1]["variant"] = "B3"
        _write(plan_path, plan)
        barrier["plan_hash"] = stable_hash(plan)
    _write(plan_path.parent / "barrier.json", barrier)
    monkeypatch.setattr(analysis.np, "load", lambda *args, **kwargs: pytest.fail("pre-release numpy access"))
    monkeypatch.setattr(analysis, "collect", lambda *args, **kwargs: pytest.fail("collection before complete release"))
    with pytest.raises(ValueError):
        analysis.analyze(plan_path, tmp_path / "out", tmp_path / "figures")
    assert not (tmp_path / "out" / "analysis_started.json").exists()


def _samples(*, counterfactuals=False, successors=False, unseen=False):
    chunks = []
    groups = [("test", "original")]
    if unseen:
        groups.append(("test_unseen", "original"))
    if counterfactuals:
        groups.extend(("test", condition) for condition in analysis.CF)
    if successors:
        groups.append(("test", analysis.A5))
    for split, condition in groups:
        for depth in ((1, 4, 8) if condition == analysis.A5 else analysis.DEPTHS):
            episode = np.repeat([100, 200], 8)
            query = np.tile(np.arange(8), 2)
            original = query + 10
            answer = original + (1 if condition in ("replace_key_fact", analysis.A5) else 0)
            prediction = answer.copy()
            # For A5, one unswapped query already predicts the new terminal.
            old_prediction = original.copy()
            old_prediction[0] += 1
            if condition == "original":
                prediction = old_prediction.copy()
            n = len(episode)
            chunks.append(dict(split=np.full(n, split), condition=np.full(n, condition),
                               episode_id=episode, query_id=query, hop=query + 1,
                               L=np.full(n, depth), answer=answer, prediction=prediction,
                               correct=prediction == answer, full_prediction=prediction.copy(),
                               full_correct=prediction == answer, candidate_size=np.full(n, 24),
                               original_answer=original, original_prediction=old_prediction,
                               memory_episode_id=300 - episode if condition == "swap_memory" else episode.copy(),
                               counterfactual_hop_true=query + 2))
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    config = {"eval": dict(episodes=2, batch_size=1, splits=["test", "test_unseen"] if unseen else ["test"],
                           counterfactuals=counterfactuals, swap_successors=successors,
                           depths=list(analysis.DEPTHS), hops=list(analysis.HOPS))}
    return arrays, config


def test_audit_reproduces_all_cells_and_accepts_partial_a5_sweep():
    from looplora.sym.evaluate import aggregate_samples
    arrays, config = _samples(counterfactuals=True, successors=True, unseen=True)
    table, audit = analysis.audit_samples(arrays, config)
    assert table == aggregate_samples(arrays)
    assert audit["rows"] == 2 * 8 * (5 * 9 + 3)
    assert len(table) == 8 * (5 * 9 + 3)
    assert len(audit["data_fingerprints"]) == 6
    assert {row["L"] for row in table if row["condition"] == analysis.A5} == {1, 4, 8}


@pytest.mark.parametrize("failure", ["coverage", "correct", "depth_labels", "original_prediction", "self_donor", "candidate_change"])
def test_audit_rejects_concrete_sample_integrity_failures(failure):
    arrays, config = _samples(counterfactuals=True)
    if failure == "coverage":
        arrays = {key: value[:-1] for key, value in arrays.items()}
    elif failure == "correct":
        arrays["correct"][0] = ~arrays["correct"][0]
    elif failure == "depth_labels":
        index = np.flatnonzero((arrays["condition"] == "original") & (arrays["L"] == 4))[0]
        arrays["answer"][index] += 3
        arrays["correct"][index] = arrays["prediction"][index] == arrays["answer"][index]
        arrays["full_correct"][index] = arrays["full_prediction"][index] == arrays["answer"][index]
        arrays["original_answer"][index] = arrays["answer"][index]
    else:
        index = np.flatnonzero(arrays["condition"] == "swap_memory")[0]
        if failure == "original_prediction":
            arrays["original_prediction"][index] += 1
        elif failure == "self_donor":
            arrays["memory_episode_id"][index] = arrays["episode_id"][index]
        else:
            arrays["candidate_size"][index] += 1
    with pytest.raises(ValueError):
        analysis.audit_samples(arrays, config)


def test_successor_diagnostic_uses_new_hop_and_matching_unswapped_prediction():
    arrays, _ = _samples(counterfactuals=True, successors=True)
    rows = analysis.successor_follow_rates(arrays)
    first = next(row for row in rows if row["L"] == 1 and row["original_hop"] == 1 and row["new_hop_le_L"] == "all")
    assert first["follow_rate"] == 1
    assert first["unswapped_follow_rate"] == .5
    assert first["paired_follow_difference"] == .5
    assert first["gating"] is False
    assert not any(row["L"] == 1 and row["new_hop_le_L"] == "true" for row in rows)
    assert any(row["L"] == 4 and row["original_hop"] == 3 and row["new_hop_le_L"] == "true" for row in rows)
    assert any(row["L"] == 4 and row["original_hop"] == 4 and row["new_hop_le_L"] == "false" for row in rows)


def _snapshots(config):
    groups = [(split, "original") for split in config["eval"]["splits"]]
    if config["eval"]["counterfactuals"]:
        groups.extend(("test", condition) for condition in analysis.CF)
    if config["eval"]["swap_successors"]:
        groups.append(("test", analysis.A5))
    rows = []
    for split, condition in groups:
        count = 16 if condition in (*analysis.CF[:2], analysis.A5) else 2
        for begin in range(count):
            for depth in ((1, 4, 8) if condition == analysis.A5 else analysis.DEPTHS):
                rows.append(dict(split=split, condition=condition, batch_start=begin, L=depth,
                                 **{field: str(i) * 32 for i, field in enumerate(analysis.FIELDS)}))
    return {"gate0_passed": True, "memory_hashes": rows}


def test_snapshots_audit_query_fork_batch_counts_and_partial_a5_depths():
    _, config = _samples(counterfactuals=True, successors=True, unseen=True)
    metadata = _snapshots(config)
    assert len(analysis.audit_snapshots(metadata, config)) == 54
    metadata["memory_hashes"][-1]["memory"] = "x" * 32
    with pytest.raises(ValueError, match="depth-dependent"):
        analysis.audit_snapshots(metadata, config)


def test_compact_cache_bounds_memory_preserves_ids_and_keeps_only_analysis_depths(tmp_path):
    arrays, _ = _samples(counterfactuals=True, unseen=True)
    arrays["episode_id"] += 2**50
    cache = analysis.CompactCache(tmp_path / "cache", limit=1)
    cache.save("one", "confirmation", arrays, "test")
    cache.save("two", "unseen", arrays, "test_unseen")
    first = cache.get("one", "confirmation")
    assert set(first) == {"split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct"}
    assert set(first["L"]) == {1, 4, 8}
    assert set(first["condition"]) == {"original"}
    assert min(first["episode_id"]) == 2**50 + 100
    assert first["correct"].dtype == bool
    assert len(first["correct"]) == 2 * 8 * 3
    cache.get("two", "unseen")
    assert list(cache.memory) == [("two", "unseen")]
    assert np.array_equal(cache.get("one", "confirmation")["correct"], first["correct"])


def test_paired_confidences_share_bootstrap_seed_and_original_query_scope(monkeypatch):
    import looplora.sym.stats as stats
    calls = []

    def fake(a, b, da, db, **kwargs):
        calls.append((a, b, da, db, kwargs))
        return kwargs

    class Cache:
        def get(self, name, dataset):
            return name, dataset

    monkeypatch.setattr(stats, "pooled_compare_runs", fake)
    runs = [{"name": f"seed-{seed}", "seed": seed} for seed in analysis.SEEDS]
    pair = analysis.paired(Cache(), runs, runs, "unseen", 4, 1, (2, 3))
    assert pair["report"]["confidence"] == .95 and pair["gate"]["confidence"] == .975
    assert [call[-1]["seed"] for call in calls] == [2718, 2718]
    assert all(call[-1]["split"] == "test_unseen" and call[-1]["condition"] == "original" for call in calls)
    assert all(call[-1]["resamples"] == 2000 and call[-1]["hops"] == (2, 3) for call in calls)


def test_statistics_wires_eight_gate_rows_and_never_gates_screening_or_a5(tmp_path, monkeypatch):
    import looplora.sym.gates as gates
    _, plan, _ = _release(tmp_path)
    captured = {}

    def fake_pair(cache, a, b, dataset, da, db, hops, *, specificity=False):
        return {"a": [run["name"] for run in a], "b": [run["name"] for run in b],
                "seeds": [run["seed"] for run in a], "dataset": dataset, "specificity": specificity}

    def gate_table(endpoints, *, evaluation_seed):
        captured.update(endpoints=endpoints, evaluation_seed=evaluation_seed)
        return list(endpoints)

    monkeypatch.setattr(analysis, "paired", fake_pair)
    monkeypatch.setattr(gates, "evaluate_gate_table", gate_table)
    rows, reported = analysis.statistics(plan, None)
    assert len(rows) == 8
    assert captured["evaluation_seed"] == 777001
    assert all(set(endpoint) == set(gates.ENDPOINTS) for endpoint in captured["endpoints"].values())
    for writer in ("oracle_random", "oracle_orth"):
        for reader in analysis.READERS:
            endpoint = captured["endpoints"][writer, reader]
            assert endpoint["replication_reach"]["a"] == captured["endpoints"]["oracle_random", reader]["replication_reach"]["a"]
            assert endpoint["reach"]["dataset"] == "confirmation"
            assert endpoint["reach"]["seeds"] == [17, 29, 43]
            assert endpoint["specificity"]["specificity"] is True
    screening = [row for row in reported if row["dataset"] == "screening"]
    assert all(row["statistics"]["seeds"] == [17] for row in screening)
    assert all(row["gating"] is False and analysis.A5 not in row["endpoint"] for row in reported)
    assert {row["dataset"] for row in reported} == {"confirmation", "screening", "unseen"}
    assert any(row["endpoint"] == "B4_L1_minus_M1_L1_all_hops" for row in reported)


def test_figures_cover_requested_readers_baselines_load_contrasts_and_training(tmp_path):
    _, plan, _ = _release(tmp_path)
    tables, logs, gates, reported = [], {}, [], []
    pair = {"report": {"difference": .1, "ci_low": .05, "ci_high": .15},
            "gate": {"difference": .1, "ci_low": .04, "ci_high": .16}}
    for run in plan["runs"]:
        logs[run["name"]] = [{"step": 1, "loss": 2.}, {"step": 20000, "loss": 1.}]
        if run["group"] != "writer":
            tables.extend({**run, "dataset": "confirmation", "hop": hop, "L": depth,
                           "accuracy": .1 + .01 * min(depth, hop)}
                          for hop in analysis.HOPS for depth in analysis.DEPTHS)
    for writer in analysis.WRITERS:
        for reader in analysis.READERS:
            gates.append(dict(writer=writer, reader=reader,
                              fixed_query_comparator="B5" if reader == "M1" else "B5-lin",
                              statistics={"vs_fixed": pair, "vs_loop": pair}))
        reported.append(dict(group="primary", writer=writer, dataset="confirmation",
                             endpoint="M1_minus_M1lin_L4_reach", statistics=pair))
    paths = analysis.figures(tmp_path / "figures", tables, gates, reported, logs, plan)
    assert len(paths) == 29
    assert all(Path(path).is_file() and Path(path).stat().st_size > 1000 for path in paths)
    names = {Path(path).stem for path in paths}
    assert {f"accuracy-depth-{writer}-{reader}" for writer in analysis.WRITERS for reader in analysis.READERS} <= names
    assert {f"hop-depth-{writer}" for writer in analysis.WRITERS} <= names
    assert {f"paired-differences-{writer}" for writer in analysis.WRITERS} <= names
    assert {f"load-depth-oracle_random-{reader}" for reader in analysis.READERS} <= names
