"""UNS (F11): audit of the committed unseen-pool evaluations (NumPy only, synthetic samples).

Run only inside a Miyabi-G compute allocation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from looplora.infra.seeds import stable_hash
from looplora.sym import uns
from looplora.sym.stats import pooled_compare_runs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_uns  # noqa: E402

N = 20
SEEDS = (17, 29, 43)


@pytest.fixture(autouse=True)
def small_registered_episode_count(monkeypatch):
    monkeypatch.setattr(uns, "EPISODES", N)


def _counts(reader, index):
    """Number of leading episodes answered correctly in a (split, depth, hop) cell of one synthetic run."""
    def count(split, depth, hop):
        unseen = split == "test_unseen"
        if hop == 1:
            return 19
        if hop in (2, 3):
            if reader == "B4":
                return (5 if unseen else 6) if depth == 1 else 0
            return {1: 2, 4: (6 if unseen else 12) + index}.get(depth, 0)
        if hop in (6, 7, 8):
            if reader == "B4":
                return 0
            return {1: 1, 4: 2 if unseen else 3}.get(depth, 0)
        return 0
    return count


def _samples(count, *, extra_conditions=False):
    columns = {name: [] for name in uns.KEEP}
    conditions = ("original", "swap_memory") if extra_conditions else ("original",)
    for condition in conditions:
        for split in ("test", "test_unseen"):
            low, high = uns.ENTITY_RANGE[split]
            base = 1000 if split == "test" else 500000
            for depth in uns.DEPTHS:
                for episode in range(N):
                    for hop in uns.HOPS:
                        answer = low + (episode * 7 + hop) % (high - low)
                        ok = episode < count(split, depth, hop)
                        row = dict(split=split, condition=condition, episode_id=base + 3 * episode, query_id=hop - 1,
                                   hop=hop, L=depth, answer=answer, correct=ok,
                                   prediction=answer if ok else low + (answer - low + 1) % (high - low),
                                   candidate_size=14)
                        for name, value in row.items():
                            columns[name].append(value)
    arrays = {name: np.asarray(values) for name, values in columns.items()}
    arrays["split"], arrays["condition"] = arrays["split"].astype(str), arrays["condition"].astype(str)
    return arrays


def _select(rows, **match):
    found = [row for row in rows if all(row[key] == value for key, value in match.items())]
    assert len(found) == 1, (match, len(found))
    return found[0]


def _expected(split, cell_a, cell_b, hops):
    total = []
    for index, _ in enumerate(SEEDS):
        a, b = _counts(cell_a[0], index), _counts(cell_b[0], index)
        total.append(np.mean([(a(split, cell_a[1], hop) - b(split, cell_b[1], hop)) / N for hop in hops]))
    return float(np.mean(total))


def test_run_directory_resolves_both_layouts_and_rejects_missing_files(tmp_path):
    p1s, f2 = tmp_path / "p1s", tmp_path / "f2"
    for directory in (p1s / "runs" / "primary-joint-M1-r32-s17" / "screening",
                      f2 / "evaluation" / "f2-main-joint_fact-B4-s29" / "screening"):
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text("{}")
        (directory / "samples.npz").write_bytes(b"")
    assert uns.run_directory("joint", "M1", 17, p1s_root=p1s, f2_root=f2) == \
        p1s / "runs" / "primary-joint-M1-r32-s17" / "screening"
    assert uns.run_directory("joint_fact", "B4", 29, p1s_root=p1s, f2_root=f2) == \
        f2 / "evaluation" / "f2-main-joint_fact-B4-s29" / "screening"
    with pytest.raises(FileNotFoundError):
        uns.run_directory("joint", "B4", 17, p1s_root=p1s, f2_root=f2)
    with pytest.raises(ValueError, match="screening"):
        uns.run_directory("joint", "M1", 17, p1s_root=p1s, f2_root=f2, dataset="test")
    confirmation = p1s / "runs" / "primary-joint-M1-r32-s17" / "confirmation"
    confirmation.mkdir()
    (confirmation / "samples.npz").write_bytes(b"")  # a confirmation directory needs no metrics.json
    assert uns.run_directory("joint", "M1", 17, p1s_root=p1s, f2_root=f2, dataset="confirmation") == confirmation
    (p1s / "runs" / "primary-joint-M1-r32-s17" / "screening" / "samples.npz").unlink()
    with pytest.raises(FileNotFoundError):
        uns.run_directory("joint", "M1", 17, p1s_root=p1s, f2_root=f2)


def test_audit_samples_accepts_the_registered_protocol():
    report = uns.audit_samples(uns.restrict(_samples(_counts("M1", 0))))
    assert report["failures"] == []
    assert report["pools"]["seen"]["rows"] == len(uns.DEPTHS) * len(uns.HOPS) * N
    assert report["pools"]["unseen"]["candidate_sizes"] == [14]


def _corrupt(samples, name, mutate):
    copy = {key: value.copy() for key, value in samples.items()}
    mutate(copy[name], copy["split"])
    return copy


def _subset(samples, keep):
    return {key: value[keep] for key, value in samples.items()}


def _answer_outside(value, split):
    value[np.flatnonzero(split == "test")[0]] = 900


def _share_episode_ids(value, split):
    value[split == "test_unseen"] = value[np.flatnonzero(split == "test")[0]]


def _flip_first(value, split):
    value[0] = not value[0]


def _constant_candidates(value, split):
    value[:] = 13


@pytest.mark.parametrize("label, build, message", [
    ("answer outside the pool", lambda s: _corrupt(s, "answer", _answer_outside), "answers outside the entity pool"),
    ("prediction outside the pool", lambda s: _corrupt(s, "prediction", _answer_outside),
     "predictions outside the entity pool"),
    ("shared episodes", lambda s: _corrupt(s, "episode_id", _share_episode_ids), "share episode ids"),
    ("missing depth", lambda s: _subset(s, s["L"] != 16), "depths"),
    ("missing hop", lambda s: _subset(s, s["hop"] != 8), "hops"),
    ("candidate size", lambda s: _corrupt(s, "candidate_size", _constant_candidates), "candidate sizes"),
    ("correct flag", lambda s: _corrupt(s, "correct", _flip_first), "saved correctness differs"),
    ("missing pool", lambda s: _subset(s, s["split"] != "test_unseen"), "no original-condition rows"),
])
def test_audit_samples_reports_each_protocol_violation(label, build, message):
    samples = uns.restrict(_samples(_counts("M1", 0)))
    failures = uns.audit_samples(build(samples))["failures"]
    assert any(message in failure for failure in failures), (label, failures)


def test_audit_samples_flags_a_wrong_episode_count(monkeypatch):
    samples = uns.restrict(_samples(_counts("M1", 0)))
    monkeypatch.setattr(uns, "EPISODES", N + 1)
    assert any("episodes instead of" in failure for failure in uns.audit_samples(samples)["failures"])


def _write_run(directory, samples, *, writer="joint", reader="M1", seed=17, **overrides):
    directory.mkdir(parents=True)
    config = dict(writer=writer, variant=reader, seed=seed, model=dict(r=32),
                  task=dict(name="t2_chain_end", E=12, chain_lengths=[8, 4], V=1024, seen=[0, 768],
                            unseen=[768, 1024]),
                  eval=dict(splits=["test", "test_unseen"], episodes=N, seed=4242, hops=list(uns.HOPS),
                            depths=list(uns.DEPTHS)))
    for key, value in overrides.items():
        section, name = key.split("__")
        config[section][name] = value
    np.savez_compressed(directory / "samples.npz", **samples)
    metadata = dict(config=config, config_hash=stable_hash(config), evaluation_seed=4242,
                    samples_hash=uns.file_hash(directory / "samples.npz"))
    (directory / "metrics.json").write_text(json.dumps(metadata))
    return directory


def test_audit_run_passes_a_registered_evaluation_and_drops_other_conditions(tmp_path):
    full = _samples(_counts("M1", 0), extra_conditions=True)
    directory = _write_run(tmp_path / "run", full)
    report, restricted = uns.audit_run(directory, writer="joint", reader="M1", seed=17)
    assert report["failures"] == []
    assert set(restricted["condition"].tolist()) == {"original"}
    assert len(restricted["correct"]) == len(full["correct"]) // 2
    assert report["samples_hash"] == uns.file_hash(directory / "samples.npz")


@pytest.mark.parametrize("overrides, message", [
    ({"eval__seed": 777001}, "evaluation seed"),
    ({"eval__splits": ["test"]}, "evaluation splits"),
    ({"eval__episodes": 2000}, "episode count"),
    ({"task__unseen": [700, 1024]}, "entity pools"),
    ({"task__E": 24}, "task differs"),
    ({"model__r": 64}, "run identity"),
    ({"seed": 29}, "run identity"),
])
def test_audit_run_flags_protocol_deviations(tmp_path, overrides, message):
    overrides = dict(overrides)
    top = {key: overrides.pop(key) for key in list(overrides) if "__" not in key}
    directory = _write_run(tmp_path / "run", _samples(_counts("M1", 0)), **overrides)
    metadata = json.loads((directory / "metrics.json").read_text())
    metadata["config"].update(top)
    metadata["config_hash"] = stable_hash(metadata["config"])
    (directory / "metrics.json").write_text(json.dumps(metadata))
    failures = uns.audit_run(directory, writer="joint", reader="M1", seed=17)[0]["failures"]
    assert any(message in failure for failure in failures), failures


def test_audit_run_detects_a_changed_sample_file_and_a_changed_config_hash(tmp_path):
    directory = _write_run(tmp_path / "run", _samples(_counts("M1", 0)))
    metadata = json.loads((directory / "metrics.json").read_text())
    metadata["config_hash"] = "0" * 32
    (directory / "metrics.json").write_text(json.dumps(metadata))
    assert any("config hash" in failure
               for failure in uns.audit_run(directory, writer="joint", reader="M1", seed=17)[0]["failures"])
    changed = _samples(_counts("M1", 1))
    np.savez_compressed(directory / "samples.npz", **changed)
    assert any("sample file hash" in failure
               for failure in uns.audit_run(directory, writer="joint", reader="M1", seed=17)[0]["failures"])


def test_endpoint_rows_reproduce_hand_computed_accuracy_differences():
    runs = {(reader, seed): uns.restrict(_samples(_counts(reader, index)))
            for reader in uns.READERS for index, seed in enumerate(SEEDS)}
    rows = uns.endpoint_rows(runs, "joint", resamples=2000)
    assert len(rows) == len(uns.ENDPOINTS) * len(uns.GROUPS) * 3
    for endpoint, (cell_a, cell_b) in uns.ENDPOINTS.items():
        for group, hops in uns.GROUPS.items():
            seen = _select(rows, endpoint=endpoint, group=group, pool="seen")
            unseen = _select(rows, endpoint=endpoint, group=group, pool="unseen")
            gap = _select(rows, endpoint=endpoint, group=group, pool="unseen_minus_seen")
            assert seen["difference"] == pytest.approx(_expected("test", cell_a, cell_b, hops))
            assert unseen["difference"] == pytest.approx(_expected("test_unseen", cell_a, cell_b, hops))
            assert gap["difference"] == pytest.approx(unseen["difference"] - seen["difference"])
            assert gap["ci_low"] <= gap["ci_high"] and seen["ci_low"] <= seen["ci_high"]
            for seed in SEEDS:
                assert gap[f"seed_{seed}"] == pytest.approx(unseen[f"seed_{seed}"] - seen[f"seed_{seed}"])
    reach = _select(rows, endpoint="m1_depth_gain", group="reach", pool="seen")
    assert reach["difference"] == pytest.approx(((10 + 0) + (10 + 1) + (10 + 2)) / 3 / N)
    assert reach["confidence"] == 0.95 and reach["n_episodes"] == N
    again = uns.endpoint_rows(runs, "joint", resamples=2000)
    assert again == rows


def _shift_ids(value, split):
    value += 1


def test_endpoint_rows_reject_seeds_that_evaluate_different_episodes():
    runs = {(reader, seed): uns.restrict(_samples(_counts(reader, index)))
            for reader in uns.READERS for index, seed in enumerate(SEEDS)}
    runs["M1", 29] = _corrupt(runs["M1", 29], "episode_id", _shift_ids)
    with pytest.raises(ValueError, match="disagree"):
        uns.endpoint_rows(runs, "joint", resamples=2000)


def test_accuracy_rows_give_every_hop_cell_and_the_chance_level():
    runs = {(reader, seed): uns.restrict(_samples(_counts(reader, index)))
            for reader in uns.READERS for index, seed in enumerate(SEEDS)}
    rows = uns.accuracy_rows(runs, "joint_fact", resamples=2000)
    assert len(rows) == len(uns.CELLS) * len(uns.HOPS) * 3
    seen = _select(rows, reader="M1", L=4, hop=2, pool="seen")
    unseen = _select(rows, reader="M1", L=4, hop=2, pool="unseen")
    gap = _select(rows, reader="M1", L=4, hop=2, pool="unseen_minus_seen")
    assert seen["accuracy"] == pytest.approx(np.mean([(12 + i) / N for i in range(3)]))
    assert unseen["accuracy"] == pytest.approx(np.mean([(6 + i) / N for i in range(3)]))
    assert gap["accuracy"] == pytest.approx(unseen["accuracy"] - seen["accuracy"])
    assert seen["chance"] == pytest.approx(1 / 14) and gap["chance"] == ""
    assert _select(rows, reader="B4", L=1, hop=1, pool="seen")["accuracy"] == pytest.approx(19 / N)
    for row in rows:
        assert row["ci_low"] <= row["ci_high"]


def _committed_from(samples_by_seed, *, writer="joint", group="primary", r="32", shift=0.0, drop=None):
    rows = []
    for endpoint, hops in uns.REPRODUCED.items():
        for confidence in (0.95, 0.975):
            result = pooled_compare_runs(samples_by_seed, samples_by_seed, 4, 1, hops=hops, split="test_unseen",
                                         condition="original", resamples=2000, seed=uns.BOOTSTRAP_SEED,
                                         confidence=confidence)
            rows.append(dict(study="P1-S", group=group, writer=writer, r=r, reader="M1", dataset="unseen",
                             endpoint=endpoint, confidence=confidence, difference=result["difference"] + shift,
                             ci_low=result["ci_low"], ci_high=result["ci_high"], training_seeds="17,29,43"))
    return [row for row in rows if (row["endpoint"], row["confidence"]) != drop]


def test_reproduce_reported_matches_the_committed_pooled_statistics_and_flags_changes():
    samples = {seed: uns.restrict(_samples(_counts("M1", index))) for index, seed in enumerate(SEEDS)}
    checks = uns.reproduce_reported(samples, _committed_from(samples), writer="joint")
    assert len(checks) == len(uns.REPRODUCED) * 2
    assert all(check["reproduced"] and check["max_abs_deviation"] == 0.0 for check in checks)
    off = uns.reproduce_reported(samples, _committed_from(samples, shift=1e-6), writer="joint")
    assert not any(check["reproduced"] for check in off)
    with pytest.raises(ValueError, match="0 committed rows"):
        uns.reproduce_reported(samples, _committed_from(samples, drop=("reach", 0.95)), writer="joint")
    with pytest.raises(ValueError, match="2 committed rows"):
        uns.reproduce_reported(samples, _committed_from(samples) * 2, writer="joint")
    with pytest.raises(ValueError, match="0 committed rows"):
        uns.reproduce_reported(samples, _committed_from(samples, r="16"), writer="joint")


def test_load_committed_and_context_table_transcribe_without_recomputation(tmp_path):
    p1s = tmp_path / "p1s.csv"
    p1s.write_text("\n".join([
        "group,writer,reader,dataset,endpoint,confidence,difference,ci_low,ci_high,training_seeds,per_seed",
        'primary,joint,M1,unseen,reach,0.95,0.25,0.2,0.3,"17,29,43","[]"',
        'primary,joint,M1,confirmation,reach,0.95,0.5,0.4,0.6,"17,29,43","[]"',
        'primary,joint,M1,unseen,reach,0.975,0.25,0.1,0.4,"17,29,43","[]"',
        'secondary,joint,M1,unseen,far,0.95,0.01,-0.1,0.1,17,"[]"',
        'primary,joint,M1,unseen,replication_far,0.95,0.0,0.0,0.0,"17,29,43","[]"',
        'primary,joint,M1,screening,reach,0.95,0.3,0.2,0.4,17,"[]"',
    ]) + "\n")
    f2 = tmp_path / "f2.csv"
    f2.write_text("\n".join([
        "writer,reader,dataset,endpoint,confidence,difference,ci_low,ci_high,per_seed",
        # the F1 and F2 exports have no training_seeds column: the seeds come from the JSON per_seed entries
        'joint_fact,M1,unseen,hop1,0.95,-0.01,-0.02,0.0,"[{""seed"": 17}, {""seed"": 29}, {""seed"": 43}]"',
        'joint_fact,M1,screening,hop1,0.95,-0.02,-0.03,0.0,"[{""seed"": 17}]"',
        'joint_fact,M1,confirmation,hop1,0.95,-0.03,-0.04,0.0,"[]"',
    ]) + "\n")
    rows = uns.load_committed(p1s, "P1-S", {"primary": "32", "secondary": "16"}) + uns.load_committed(f2, "P1-S-F2")
    assert [row["r"] for row in rows] == ["32", "32", "32", "16", "32", "32", "32", "32", "32"]
    assert [row["training_seeds"] for row in rows] == ["17,29,43", "17,29,43", "17,29,43", "17", "17,29,43", "17",
                                                       "17,29,43", "17", ""]
    assert rows[0]["difference"] == 0.25 and rows[-1]["group"] == "primary" and rows[-1]["study"] == "P1-S-F2"
    table = uns.context_table(rows)
    assert len(table) == 3  # replication_far and the 0.975 row are not part of the context table
    reach = _select(table, study="P1-S", endpoint="reach")
    assert reach["unseen_difference"] == 0.25 and reach["confirmation_difference"] == 0.5
    # the committed screening row pools seed 17 only, the unseen row three seeds: both are listed
    assert reach["screening_difference"] == 0.3
    assert reach["screening_training_seeds"] == "17" and reach["unseen_training_seeds"] == "17,29,43"
    far = _select(table, study="P1-S", endpoint="far")
    assert far["r"] == "16" and far["unseen_ci_low"] == -0.1 and far["unseen_training_seeds"] == "17"
    hop1 = _select(table, study="P1-S-F2", endpoint="hop1")
    assert hop1["unseen_training_seeds"] == "17,29,43" and hop1["screening_training_seeds"] == "17"
    assert hop1["confirmation_training_seeds"] == ""


def _write_committed(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "per_seed": "[]"} for row in rows)


def _layout(tmp_path):
    """Synthetic P1-S/F2 run trees and committed exports for the three audited writers."""
    p1s, f2, reports = tmp_path / "p1s", tmp_path / "f2", tmp_path / "reports"
    committed = {}
    for writer in (*uns.WRITERS, uns.CONTROL):
        for reader in uns.READERS:
            for index, seed in enumerate(SEEDS):
                samples = _samples(_counts(reader, index))
                base = (f2 / "evaluation" / f"f2-main-{writer}-{reader}-s{seed}" if writer in uns.FACTORIZED
                        else p1s / "runs" / f"primary-{writer}-{reader}-r32-s{seed}")
                _write_run(base / "screening", samples, writer=writer, reader=reader, seed=seed)
        by_seed = {seed: uns.restrict(_samples(_counts("M1", index))) for index, seed in enumerate(SEEDS)}
        committed[writer] = _committed_from(by_seed, writer=writer)
    columns = ["group", "writer", "reader", "dataset", "endpoint", "confidence", "difference", "ci_low", "ci_high",
               "training_seeds", "per_seed"]
    _write_committed(reports / "p1s" / "reported_contrasts.csv", committed["joint"] + committed[uns.CONTROL], columns)
    _write_committed(reports / "p1s-f1" / "reported_contrasts.csv", committed["joint"][:2],
                     ["writer", "r", "reader", "dataset", "endpoint", "confidence", "difference", "ci_low", "ci_high",
                      "per_seed"])
    _write_committed(reports / "p1s-f2" / "reported_contrasts.csv", committed["joint_fact"],
                     ["writer", "reader", "dataset", "endpoint", "confidence", "difference", "ci_low", "ci_high",
                      "per_seed"])
    return p1s, f2, reports


def _run_script(tmp_path, monkeypatch, p1s, f2, reports):
    monkeypatch.setattr(audit_uns, "require_compute", lambda: None)
    out = tmp_path / "out"
    code = audit_uns.main(["--p1s-root", str(p1s), "--f2-root", str(f2), "--reports", str(reports),
                           "--out", str(out), "--resamples", "2000"])
    return code, out


def test_audit_script_writes_every_table_for_a_consistent_synthetic_layout(tmp_path, monkeypatch):
    p1s, f2, reports = _layout(tmp_path)
    code, out = _run_script(tmp_path, monkeypatch, p1s, f2, reports)
    assert code == 0
    assert json.loads((out / "audit.json").read_text())["status"] == "passed"
    with (out / "reproduction.csv").open(newline="") as stream:
        reproduction = list(csv.DictReader(stream))
    assert len(reproduction) == 3 * len(uns.REPRODUCED) * 2 and all(row["reproduced"] == "True" for row in reproduction)
    for name, count in (("endpoints.csv", 3 * len(uns.ENDPOINTS) * len(uns.GROUPS) * 3),
                        ("accuracy.csv", 3 * len(uns.CELLS) * len(uns.HOPS) * 3)):
        with (out / name).open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == count
    with (out / "context.csv").open(newline="") as stream:
        context = list(csv.DictReader(stream))
    assert {row["study"] for row in context} == {"P1-S", "P1-S-F1", "P1-S-F2"}
    summary = (out / "summary.md").read_text()
    assert "joint_fact" in summary and "oracle_random" in summary
    assert json.loads((out / "summary.json").read_text())["audit_failures"] == 0


def test_audit_script_stops_before_statistics_when_a_run_violates_the_protocol(tmp_path, monkeypatch):
    p1s, f2, reports = _layout(tmp_path)
    metrics = f2 / "evaluation" / "f2-main-joint_fact-B4-s29" / "screening" / "metrics.json"
    metadata = json.loads(metrics.read_text())
    metadata["config"]["seed"] = 17
    metadata["config_hash"] = stable_hash(metadata["config"])
    metrics.write_text(json.dumps(metadata))
    code, out = _run_script(tmp_path, monkeypatch, p1s, f2, reports)
    assert code == 1
    audit = json.loads((out / "audit.json").read_text())
    assert audit["status"] == "failed" and any("joint_fact/B4/s29" in message for message in audit["failures"])
    assert not (out / "endpoints.csv").exists() and (out / "context.csv").exists()


def test_audit_script_reports_a_missing_run_as_a_finding(tmp_path, monkeypatch):
    p1s, f2, reports = _layout(tmp_path)
    (p1s / "runs" / "primary-joint-B4-r32-s43" / "screening" / "samples.npz").unlink()
    code, out = _run_script(tmp_path, monkeypatch, p1s, f2, reports)
    assert code == 1
    assert any("joint/B4/s43" in message for message in json.loads((out / "audit.json").read_text())["failures"])
