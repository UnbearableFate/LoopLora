"""CVH runner and analysis scripts on tiny CPU readers (lock, smoke, train, confirm, audit, analysis).

Run only inside a Miyabi-G compute allocation, including the CPU cases. The CUDA latency
measurement is the one part that cannot run here (the stages take ``--skip-latency``).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from looplora.core.learned_halting import HaltingHead
from looplora.sym import cvh
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.model import SymbolicModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_cvh  # noqa: E402
import run_cvh  # noqa: E402
from s9_common import digest_checkpoint  # noqa: E402

HOPS = [1, 2, 3, 4]
EPISODES = 6
WRITERS = {"oracle_random/r32/M1": "oracle_random", "joint_fact/r32/M1": "joint_fact",
           "t1_frozen_fact/r32/M1": "t1_frozen_fact"}


def _config(writer, seed):
    base = {
        "name": "cvh-script-tiny", "device": "cpu", "seed": seed, "writer": writer, "variant": "M1",
        "model": {"d_e": 8, "r": 32, "r_mem": 32, "max_hop": 8},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 96, "n_rel": 4,
                 "seen": [0, 72], "unseen": [72, 96], "chain_lengths": [4, 4], "hops": HOPS},
        "train": {"steps": 1, "batch_size": 2, "warmup_steps": 0, "depths": [1, 4], "log_every": 1},
        "eval": {"episodes": EPISODES, "batch_size": 3, "splits": ["val"], "depths": [1, 4], "hops": HOPS,
                 "counterfactuals": False},
    }
    return resolve_config(recursive_merge(base, {}))


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _commit_file(repo, relative, source):
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(source).read_bytes())
    _git(repo, "add", relative)
    _git(repo, "commit", "-q", "-m", f"lock {relative}")
    return target, _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """Nine tiny M1 readers with P4-S-L style heads, a fake committed reports/p4s-l, and tiny registered sizes."""
    monkeypatch.setattr(run_cvh, "require_compute", lambda: None)
    monkeypatch.setattr(run_cvh, "HOPS", HOPS)
    monkeypatch.setattr(run_cvh, "TRAINING", dict(split="train", seed=31337, episodes=EPISODES, steps=3, lr=0.001,
                                                  batch_episodes=2, label="cvh", margin=cvh.MARGIN))
    monkeypatch.setattr(run_cvh, "VALIDATION", dict(split="val", seed=4242, episodes=EPISODES))
    monkeypatch.setattr(run_cvh, "CONFIRMATION", dict(split="test", seed=999005, episodes=EPISODES))
    monkeypatch.setattr(run_cvh, "SMOKE", dict(training_episodes=EPISODES, steps=3, validation_episodes=EPISODES,
                                               training_seed=17))
    repo = tmp_path / "repo"
    (repo / "reports" / "p4s-l" / "heads").mkdir(parents=True)
    _git(repo, "init", "-q")
    runs, head_runs, traces = [], [], {}
    for family, writer in WRITERS.items():
        for seed in cvh.SEEDS:
            name = f"{writer}-M1-r32-s{seed}"
            config = _config(writer, seed)
            torch.manual_seed(seed)
            model = SymbolicModel(config)
            path = tmp_path / "readers" / name / "checkpoint.pt"
            path.parent.mkdir(parents=True)
            torch.save({"config": config, "model": model.state_dict(), "summary": {}}, path)
            reader_hash = digest_checkpoint(path)
            head = HaltingHead(seed=seed + 1).eval()
            exported = repo / "reports" / "p4s-l" / "heads" / f"{name}.pt"
            torch.save(dict(head={k: v.detach().cpu() for k, v in head.state_dict().items()},
                            reader_checkpoint=str(path), reader_blake2b=reader_hash), exported)
            runs.append(dict(name=name, writer=writer, r=32, variant="M1", seed=seed, family=family,
                             checkpoint=str(path), checkpoint_blake2b=reader_hash))
            head_runs.append(dict(name=name, head_blake2b=digest_checkpoint(exported)))
            frozen = run_cvh.load_frozen(runs[-1] | dict(path=str(path)), "cpu")
            a0, _ = run_cvh.load_head(exported, "cpu", reader_blake2b=reader_hash)
            trace = cvh.collect_traces(frozen, HOPS, "val", 4242, EPISODES, heads={"A0": a0})
            traces.setdefault(family, {})[seed] = dict(predictions=trace["predictions"], answer=trace["answer"],
                                                       hop=trace["hop"], p_A0=trace["probabilities"]["A0"])
    # the 'recorded' P4-S-L validation accuracy of each family at its locked theta, from the same traces
    selected = []
    for family, by_seed in traces.items():
        row = cvh.threshold_table(by_seed, "p_A0", thetas=[0.5])[0]
        selected.append(dict(family=family, theta=0.5, validation_accuracy=row["validation_accuracy"]))
    reports = repo / "reports" / "p4s-l"
    extra = dict(name="joint-M1-r32-s17", writer="joint", r=32, variant="M1", seed=17, family="joint/r32/M1",
                 checkpoint="/nonexistent/other.pt", checkpoint_blake2b="0" * 32)  # outside the CVH families
    (reports / "design.json").write_text(json.dumps(dict(runs=[*runs, extra])))
    (reports / "head_manifest.json").write_text(json.dumps(dict(status="passed", all_readers_unchanged=True,
                                                                  runs=head_runs)))
    (reports / "theta.json").write_text(json.dumps(dict(selected=selected)))
    return tmp_path, repo


def _locked_design(tmp_path, repo):
    design = tmp_path / "design.json"
    assert run_cvh.main(["lock", "--repo", str(repo), "--out", str(design)]) == 0
    return _commit_file(repo, "reports/cvh/design.json", design)


def test_lock_selects_the_nine_registered_readers_and_seals_them(tiny):
    tmp_path, repo = tiny
    design = run_cvh.build_design(repo)
    assert len(design["readers"]) == 9 and {r["family"] for r in design["readers"]} == set(cvh.FAMILIES)
    assert all(r["variant"] == "M1" and r["r"] == 32 for r in design["readers"])
    assert set(design["a0_theta"]) == set(cvh.FAMILIES)
    assert design["decision"]["confidence"] == pytest.approx(1 - 0.05 / 6) and design["confirmation"]["seed"] == 999005
    assert len(run_cvh.verify_design(design)) == 9
    with pytest.raises(ValueError):  # a drifted setting
        run_cvh.verify_design({**design, "hop1_tolerance": 0.02})
    with pytest.raises(ValueError):  # a missing reader
        run_cvh.verify_design({**design, "readers": design["readers"][1:]})
    head = Path(design["readers"][0]["head_path"])
    head.write_bytes(head.read_bytes() + b"x")
    with pytest.raises(ValueError, match="A0 head changed"):
        run_cvh.verify_design(design)
    with pytest.raises(ValueError, match="exported P4-S-L head differs"):
        run_cvh.build_design(repo)


def test_a_head_trained_on_another_reader_is_refused(tiny):
    _, repo = tiny
    design = run_cvh.build_design(repo)
    first, second = design["readers"][:2]
    with pytest.raises(ValueError, match="trained on another reader"):
        run_cvh.load_head(first["head_path"], "cpu", reader_blake2b=second["blake2b"])


def test_pipeline_runs_end_to_end_and_the_analysis_audits_every_seal(tiny, monkeypatch):
    tmp_path, repo = tiny
    original = cvh.select_theta  # tiny random readers need not keep hop 1: select without the constraint
    monkeypatch.setattr(run_cvh.cvh, "select_theta", lambda table, tolerance=None: original(table, tolerance=None))
    locked, design_commit = _locked_design(tmp_path, repo)

    smoke = tmp_path / "smoke"
    assert run_cvh.main(["smoke", "--design", str(locked), "--root", str(smoke), "--device", "cpu"]) == 0
    report = json.loads((smoke / "smoke.json").read_text())
    assert report["status"] == "passed" and len(report["timings"]) == 3 and len(report["endpoints"]) == 9
    assert report["theta"]["smoke"] is True and report["suggested_train_walltime_seconds"] > 0
    assert all(0 <= rate <= 1 for record in report["label_rates"].values() for rate in record["label_rate"])

    root = tmp_path / "run"
    assert run_cvh.main(["train", "--design", str(locked), "--design-commit", design_commit, "--root", str(root),
                         "--device", "cpu"]) == 0
    theta = json.loads((root / "theta.json").read_text())
    assert theta["confirmation_allowed"] and theta["smoke"] is False and len(theta["selected"]) == 9
    assert all(row["reproduced"] for row in theta["a0_reproduction"])  # the P4-S-L validation numbers are rebuilt
    assert {row["arm"] for row in theta["selected"]} == set(cvh.ARMS) and not theta["failures"]
    assert len(list((root / "heads").glob("*.pt"))) == 9 and len(list((root / "traces").glob("*/val.npz"))) == 9
    theta_path, theta_commit = _commit_file(repo, "reports/cvh/theta.json", root / "theta.json")

    confirm = ["confirm", "--design", str(locked), "--design-commit", design_commit, "--theta", str(theta_path),
               "--theta-commit", theta_commit, "--root", str(root), "--device", "cpu", "--skip-latency"]
    theta_path.write_text(theta_path.read_text() + " ")
    with pytest.raises(ValueError, match="differs from its Git lock"):
        run_cvh.main(confirm)
    _git(repo, "checkout", "-q", "--", "reports/cvh/theta.json")
    assert not (root / "manifest.json").exists()
    assert run_cvh.main(confirm) == 0
    with pytest.raises(ValueError, match="already started"):
        run_cvh.main(confirm)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["checkpoints_unchanged"] is True
    assert manifest["latency_measured"] is False and len(list((root / "traces").glob("*/test.npz"))) == 9

    out = tmp_path / "analysis"
    assert analyze_cvh.main(["--root", str(root), "--out", str(out)]) == 0
    with (out / "endpoints.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9 and {row["arm"] for row in rows} == set(cvh.ARMS)
    assert all(row["passes"] in {"True", "False"} and float(row["confidence"]) == pytest.approx(1 - 0.05 / 6)
               for row in rows)
    assert {row["gating"] for row in rows} == {"True", "False"}  # A0 is the reference arm
    summary = json.loads((out / "summary.json").read_text())
    assert summary["failures"] == [] and summary["latency_claims"] == []  # no latency was measured, so no claim
    assert "Registered pattern" in (out / "summary.md").read_text()
    with (out / "accuracy.csv").open(newline="") as stream:
        accuracy = list(csv.DictReader(stream))
    assert {row["method"] for row in accuracy} == {"L1", "L4", "L16", "A0", "A1", "A2"}

    # tampering is refused before any statistic is computed
    victim = next((root / "traces").glob("*/test.npz"))
    original = victim.read_bytes()
    victim.write_bytes(original[:-4] + b"zzzz")
    with pytest.raises(ValueError, match="traces missing or changed"):
        analyze_cvh.main(["--root", str(root), "--out", str(tmp_path / "again")])
    victim.write_bytes(original)
    head = Path(json.loads((root / "train_manifest.json").read_text())["readers"][0]["a2_head_path"])
    head.write_bytes(head.read_bytes() + b"x")
    with pytest.raises(ValueError, match="A2 head changed"):
        analyze_cvh.main(["--root", str(root), "--out", str(tmp_path / "again")])


def test_no_feasible_theta_is_a_registered_failure_and_blocks_the_confirmation(tiny, monkeypatch):
    tmp_path, repo = tiny
    monkeypatch.setattr(run_cvh.cvh, "select_theta", lambda table, tolerance=None: None)
    locked, design_commit = _locked_design(tmp_path, repo)
    root = tmp_path / "run"
    assert run_cvh.main(["train", "--design", str(locked), "--design-commit", design_commit, "--root", str(root),
                         "--device", "cpu"]) == 0
    theta = json.loads((root / "theta.json").read_text())
    assert theta["confirmation_allowed"] is False and len(theta["failures"]) == 6
    infeasible = [row["arm"] for row in theta["selected"] if not row["feasible"]]
    assert sorted(infeasible) == ["A1"] * 3 + ["A2"] * 3  # the reference arm A0 keeps its locked theta
    theta_path, theta_commit = _commit_file(repo, "reports/cvh/theta.json", root / "theta.json")
    with pytest.raises(ValueError, match="registered failure"):
        run_cvh.main(["confirm", "--design", str(locked), "--design-commit", design_commit, "--theta",
                      str(theta_path), "--theta-commit", theta_commit, "--root", str(root), "--device", "cpu",
                      "--skip-latency"])
    assert not (root / "manifest.json").exists() and not list((root / "traces").glob("*/test.npz"))


def test_a_family_without_a_feasible_new_arm_is_never_traced_on_the_confirmation_set(tiny, monkeypatch):
    tmp_path, repo = tiny
    original, calls = cvh.select_theta, []

    def select(table, tolerance=None):
        calls.append(1)  # the first family in sorted order (joint_fact) is asked first: both of its new arms fail
        return None if len(calls) <= 2 else original(table, tolerance=None)

    monkeypatch.setattr(run_cvh.cvh, "select_theta", select)
    locked, design_commit = _locked_design(tmp_path, repo)
    root = tmp_path / "run"
    assert run_cvh.main(["train", "--design", str(locked), "--design-commit", design_commit, "--root", str(root),
                         "--device", "cpu"]) == 0
    theta = json.loads((root / "theta.json").read_text())
    assert theta["confirmation_allowed"] is True
    assert [row["family"] for row in theta["failures"]] == ["joint_fact/r32/M1"] * 2
    assert run_cvh.active_families(theta) == ["oracle_random/r32/M1", "t1_frozen_fact/r32/M1"]
    theta_path, theta_commit = _commit_file(repo, "reports/cvh/theta.json", root / "theta.json")
    assert run_cvh.main(["confirm", "--design", str(locked), "--design-commit", design_commit, "--theta",
                         str(theta_path), "--theta-commit", theta_commit, "--root", str(root), "--device", "cpu",
                         "--skip-latency"]) == 0
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["active_families"] == ["oracle_random/r32/M1", "t1_frozen_fact/r32/M1"]
    assert len(list((root / "traces").glob("*/test.npz"))) == 6 and not list((root / "traces").glob("joint_fact-*/test.npz"))
    out = tmp_path / "analysis"
    assert analyze_cvh.main(["--root", str(root), "--out", str(out)]) == 0
    with (out / "endpoints.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    skipped = [row for row in rows if row["family"] == "joint_fact/r32/M1"]
    assert len(rows) == 8 and sorted(row["arm"] for row in skipped) == ["A1", "A2"]
    assert all(row["feasible"] == "False" and not row.get("passes") for row in skipped)
    with (out / "accuracy.csv").open(newline="") as stream:
        assert {row["family"] for row in csv.DictReader(stream)} == {"oracle_random/r32/M1", "t1_frozen_fact/r32/M1"}
    assert len(json.loads((out / "summary.json").read_text())["failures"]) == 2
