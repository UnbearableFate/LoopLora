"""TSM runner and analysis scripts on tiny CPU runs (lock, smoke, train, evaluate, audit, analysis).

Run only inside a Miyabi-G compute allocation, including the CPU cases. The training children are
run in-process here; their process management is exercised by the smoke job.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from looplora.sym import tsm, uns
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.model import SymbolicModel
from looplora.sym.train import train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_tsm  # noqa: E402
import run_tsm  # noqa: E402

HOPS = [1, 2, 3, 4]
EPISODES = 6


def _config(writer, variant, seed):
    base = {
        "name": "tsm-script-tiny", "device": "cpu", "seed": seed, "writer": writer, "variant": variant,
        "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 16},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 96, "n_rel": 4, "seen": [0, 72], "unseen": [72, 96],
                 "chain_lengths": [4, 4], "hops": [1, 2, 3]},
        "train": {"steps": 3, "batch_size": 4, "lr": 0.01, "warmup_steps": 0, "depths": [1] if variant == "B4" else [1, 2, 4],
                  "log_every": 1},
        "eval": {"episodes": EPISODES, "batch_size": 3, "splits": ["test"], "depths": [1, 4], "hops": HOPS,
                 "counterfactuals": False},
    }
    return resolve_config(recursive_merge(base, {}))


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


class Done:
    """A finished child: the in-process stand-in for ``start_child``."""

    def poll(self):
        return 0

    def stop(self):
        pass


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """Tiny released M1 and B4 checkpoints in the released layout, with the registered constants shrunk to match."""
    monkeypatch.setattr(run_tsm, "require_compute", lambda: None)
    monkeypatch.setattr(run_tsm, "RANK", 16)
    monkeypatch.setattr(run_tsm, "TRAINING", dict(steps=3, lr=0.01, batch_size=4, released_depths=[1, 2, 4],
                                                  t0_depth=2, stay_depth=4, hops=[1, 2, 3]))
    monkeypatch.setattr(run_tsm, "EVALUATION", dict(split="test", seed=999007, episodes=EPISODES, hops=HOPS,
                                                    depths=list(tsm.DEPTHS), batch_size=3))
    monkeypatch.setattr(run_tsm, "SMOKE", dict(steps=2, split="val", seed=4242, episodes=EPISODES, training_seed=17))
    monkeypatch.setattr(run_tsm, "POLL_SECONDS", 0.0)

    def start_child(name, design_path, root, *, device, steps):
        run_tsm.train_one(json.loads(Path(design_path).read_text()), root, name, device=device, steps=steps)
        return Done()

    monkeypatch.setattr(run_tsm, "start_child", start_child)
    p1s, f2 = tmp_path / "p1s", tmp_path / "f2"
    for writer in run_tsm.WRITERS:
        for variant in ("M1", "B4"):
            for seed in run_tsm.SEEDS:
                config = _config(writer, variant, seed)
                torch.manual_seed(seed)
                path = uns.run_root(writer, variant, seed, p1s_root=p1s, f2_root=f2) / "checkpoint.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"config": config, "model": SymbolicModel(config).state_dict(), "summary": {}}, path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return tmp_path, p1s, f2, repo


def _locked_design(tmp_path, p1s, f2, repo):
    design = tmp_path / "design.json"
    assert run_tsm.main(["lock", "--p1s-root", str(p1s), "--f2-root", str(f2), "--out", str(design)]) == 0
    locked = repo / "reports" / "tsm" / "design.json"
    locked.parent.mkdir(parents=True)
    locked.write_bytes(design.read_bytes())
    _git(repo, "add", "reports/tsm/design.json")
    _git(repo, "commit", "-q", "-m", "lock")
    return locked, _git(repo, "rev-parse", "HEAD")


def test_lock_seals_the_released_configurations_and_b4_references(tiny):
    tmp_path, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    assert len(design["templates"]) == 6 and len(design["references"]) == 6
    assert design["decision"]["confidence"] == pytest.approx(0.9875) and design["evaluation"]["seed"] == 999007
    run_tsm.verify_design(design)
    with pytest.raises(ValueError):  # a drifted registered setting
        run_tsm.verify_design({**design, "gate_init": [0.0, 3.0]})
    with pytest.raises(ValueError):  # a template whose released training settings differ
        bad = json.loads(json.dumps(design))
        bad["templates"][0]["config"]["train"]["lr"] = 0.5
        run_tsm.verify_design(bad)
    path = Path(design["references"][0]["path"])
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="B4 reference changed"):
        run_tsm.verify_design(design)


def test_arm_configurations_differ_from_the_released_one_only_in_name_depths_and_evaluation(tiny):
    _, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    template = next(t for t in design["templates"] if t["writer"] == "joint_fact" and t["seed"] == 29)["config"]
    configs = {arm: run_tsm.make_config(design, "joint_fact", 29, arm, device="cpu") for arm in tsm.ARMS}
    assert template["train"]["depths"] == [1, 2, 4]  # the released mixed-depth recipe
    for arm, depth in (("T0", 2), ("T1", 4), ("T2", 4)):  # the registered training depths of the tiny fixture
        assert configs[arm]["train"]["depths"] == [depth]
        assert {k: v for k, v in configs[arm]["train"].items() if k != "depths"} == {
            k: v for k, v in template["train"].items() if k != "depths"}
    for arm, config in configs.items():
        assert config["name"] == f"tsm-joint_fact-{arm}-s29" and config["seed"] == 29
        assert config["model"] == template["model"] and config["task"] == template["task"]
        assert config["eval"]["seed"] == 999007 and config["eval"]["splits"] == ["test"]
        assert config["eval"]["hops"] == HOPS and config["eval"]["counterfactuals"] is False
    assert run_tsm.make_config(design, "joint_fact", 29, "T0", device="cpu", steps=2)["train"]["steps"] == 2


def test_t0_is_the_unchanged_training_function(tiny, tmp_path_factory):
    tmp_path, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    root = tmp_path / "root"
    record = run_tsm.train_one(design, root, "tsm-oracle_random-T0-s17", device="cpu")
    assert record["status"] == "complete" and record["arm"] == "T0" and record["steps"] == 3
    reference, _ = train(run_tsm.make_config(design, "oracle_random", 17, "T0", device="cpu"),
                         tmp_path_factory.mktemp("reference"))
    model, _, label = tsm.load_tsm(record["checkpoint"], device="cpu")
    assert label == "T0"
    for name, value in reference.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    with pytest.raises(FileExistsError):  # a run directory is never reused
        run_tsm.train_one(design, root, "tsm-oracle_random-T0-s17", device="cpu")


def test_schedule_limits_concurrency_and_stops_everything_on_the_first_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tsm, "POLL_SECONDS", 0.0)
    active, peak, stopped = set(), [0], []

    class Stub:
        def __init__(self, name):
            self.name, self.polls = name, 0

        def poll(self):
            self.polls += 1
            return None if self.polls < 2 else (1 if self.name == "c" else 0)

        def stop(self):
            active.discard(self.name)
            stopped.append(self.name)

    def start(name, design_path, root, *, device, steps):
        active.add(name)
        peak[0] = max(peak[0], len(active))
        return Stub(name)

    monkeypatch.setattr(run_tsm, "start_child", start)
    run_tsm.schedule(["a", "b"], 2, tmp_path / "d.json", tmp_path, device="cpu")
    assert peak[0] == 2 and sorted(stopped) == ["a", "b"] and not active
    active.clear()
    peak[0] = 0
    with pytest.raises(ValueError, match="run c exited 1"):
        run_tsm.schedule(["a", "b", "c", "d"], 2, tmp_path / "d.json", tmp_path, device="cpu")
    assert peak[0] <= 2 and not active  # every started child was stopped


def test_pipeline_runs_end_to_end_and_the_analysis_audits_every_seal(tiny, monkeypatch):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)

    smoke = tmp_path / "smoke"
    assert run_tsm.main(["smoke", "--design", str(locked), "--root", str(smoke), "--procs", "2",
                         "--device", "cpu"]) == 0
    report = json.loads((smoke / "smoke.json").read_text())
    assert report["status"] == "passed" and len(report["train_step_seconds"]) == 6
    assert len(report["endpoints"]) == 4 and len(report["evaluation_rows"]) == 8 and len(report["gate_profiles"]) == 2
    assert report["suggested_walltime_seconds"] > 0 and report["extrapolated_total_run_seconds"] > 0
    assert all(row["steps"] == 2 for row in (json.loads((smoke / "runs" / n / "record.json").read_text())
                                              for n in report["train_step_seconds"]))

    root = tmp_path / "run"
    for writer, label in (("oracle_random", "a"), ("joint_fact", "b")):
        assert run_tsm.main(["train", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                             "--procs", "3", "--label", label, "--writers", writer, "--device", "cpu"]) == 0
    manifests = [json.loads((root / f"train_manifest-{label}.json").read_text()) for label in ("a", "b")]
    assert all(m["status"] == "complete" and len(m["names"]) == 9 for m in manifests)
    assert len(list((root / "runs").glob("*/checkpoint.pt"))) == 18
    with pytest.raises(FileExistsError):  # a registered run is never retrained into the same directory
        run_tsm.main(["train", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                      "--procs", "1", "--label", "again", "--writers", "joint_fact", "--arms", "T0",
                      "--seeds", "17", "--device", "cpu"])
    locked.write_text(locked.read_text() + " ")
    with pytest.raises(ValueError, match="differs from its Git lock"):
        run_tsm.main(["evaluate", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                      "--device", "cpu"])
    _git(repo, "checkout", "-q", "--", "reports/tsm/design.json")
    assert run_tsm.main(["evaluate", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                         "--device", "cpu"]) == 0
    with pytest.raises(ValueError, match="already started"):
        run_tsm.main(["evaluate", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                      "--device", "cpu"])
    manifest = json.loads((root / "evaluation" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["evaluations"]) == 24
    assert len(list((root / "evaluation" / "samples").glob("*.npz"))) == 24
    assert len(json.loads((root / "evaluation" / "gate_profiles.json").read_text())) == 6

    out = tmp_path / "analysis"
    assert analyze_tsm.main(["--root", str(root), "--out", str(out)]) == 0
    with (out / "endpoints.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and {(r["writer"], r["arm"]) for r in rows} == {
        (w, a) for w in run_tsm.WRITERS for a in ("T1", "T2")}
    assert all(float(r["confidence"]) == pytest.approx(0.9875) and r["passes"] in {"True", "False"} for r in rows)
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["arms_pass"]) == {"T1", "T2"}
    for arm, passed in summary["arms_pass"].items():  # an arm passes only if it passes for both writers
        assert passed == all(summary["writer_passes"][f"{w}/{arm}"] for w in run_tsm.WRITERS)
    assert "Registered conditions" in (out / "summary.md").read_text()
    with (out / "training.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 18

    victim = next((root / "evaluation" / "samples").glob("*.npz"))
    original = victim.read_bytes()
    victim.write_bytes(original[:-4] + b"zzzz")
    with pytest.raises(ValueError, match="samples missing or changed"):
        analyze_tsm.main(["--root", str(root), "--out", str(tmp_path / "again")])
    victim.write_bytes(original)
    checkpoint = next((root / "runs").glob("*/checkpoint.pt"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checkpoint changed"):
        analyze_tsm.main(["--root", str(root), "--out", str(tmp_path / "again")])
