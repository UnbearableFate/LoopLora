"""IVD runner and analysis scripts on tiny CPU checkpoints (lock, smoke, run, audit, analysis).

Run only inside a Miyabi-G compute allocation, including the CPU cases.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from looplora.sym import ivd, uns
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.evaluate import evaluate
from looplora.sym.model import SymbolicModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_ivd  # noqa: E402
import run_ivd  # noqa: E402

HOPS = [1, 2, 3, 4]
EPISODES = 6


def _config(writer, variant, seed):
    base = {
        "name": "ivd-script-tiny", "device": "cpu", "seed": seed, "writer": writer, "variant": variant,
        "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 8},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 96, "n_rel": 4,
                 "seen": [0, 72], "unseen": [72, 96], "chain_lengths": [4, 4], "hops": HOPS},
        "train": {"steps": 1, "batch_size": 2, "warmup_steps": 0, "depths": [1, 4], "log_every": 1},
        "eval": {"episodes": EPISODES, "batch_size": 3, "splits": ["test"], "depths": [1, 4], "hops": HOPS,
                 "counterfactuals": False},
    }
    return resolve_config(recursive_merge(base, {}))


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """Twelve tiny sealed checkpoints in the released layout, and the registered constants shrunk to match."""
    monkeypatch.setattr(run_ivd, "require_compute", lambda: None)
    monkeypatch.setattr(run_ivd, "HOPS", HOPS)
    monkeypatch.setattr(run_ivd, "RANK", 16)
    monkeypatch.setattr(run_ivd, "BATCH_SIZE", 3)
    monkeypatch.setattr(run_ivd, "CONFIRMATION", dict(split="test", seed=999004, episodes=EPISODES))
    monkeypatch.setattr(run_ivd, "SMOKE", dict(split="val", seed=4242, episodes=EPISODES, training_seed=17))
    monkeypatch.setattr(ivd, "HOP_GROUPS", {"hop1": (1,), "reach": (2, 3), "far": (4,)})
    p1s, f2 = tmp_path / "p1s", tmp_path / "f2"
    for writer in ivd.WRITERS:
        for variant in run_ivd.READERS:
            for seed in uns.SEEDS:
                config = _config(writer, variant, seed)
                torch.manual_seed(seed)
                path = uns.run_root(writer, variant, seed, p1s_root=p1s, f2_root=f2) / "checkpoint.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"config": config, "model": SymbolicModel(config).state_dict(), "summary": {}}, path)
    return tmp_path, p1s, f2


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _lock_design(tmp_path, p1s, f2):
    """Run the lock stage and commit its output in a scratch repository (the Git lock)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    design = tmp_path / "design.json"
    assert run_ivd.main(["lock", "--p1s-root", str(p1s), "--f2-root", str(f2), "--out", str(design)]) == 0
    (repo / "reports" / "ivd").mkdir(parents=True)
    locked = repo / "reports" / "ivd" / "design.json"
    locked.write_bytes(design.read_bytes())
    _git(repo, "add", "reports/ivd/design.json")
    _git(repo, "commit", "-q", "-m", "lock")
    return locked, _git(repo, "rev-parse", "HEAD")


def test_lock_lists_the_twelve_checkpoints_and_the_registered_settings(tiny):
    tmp_path, p1s, f2 = tiny
    design = run_ivd.build_design(p1s, f2)
    assert len(design["checkpoints"]) == 12 and design["confirmation"]["seed"] == 999004
    assert {entry["writer"] for entry in design["checkpoints"]} == {"oracle_random", "joint_fact"}
    assert design["decision"]["confidence"] == pytest.approx(0.975) and design["decision"]["load"] == "L24"
    assert len(run_ivd.verify_design(design)) == 12
    with pytest.raises(ValueError):  # a drifted setting
        run_ivd.verify_design({**design, "high_cosine": 0.4})
    with pytest.raises(ValueError):  # a missing checkpoint
        run_ivd.verify_design({**design, "checkpoints": design["checkpoints"][1:]})
    path = Path(design["checkpoints"][0]["path"])
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checkpoint changed"):
        run_ivd.verify_design(design)
    with pytest.raises(ValueError, match="new absolute path"):  # the lock never overwrites
        existing = tmp_path / "exists.json"
        existing.write_text("{}")
        run_ivd.main(["lock", "--p1s-root", str(p1s), "--f2-root", str(f2), "--out", str(existing)])


def test_smoke_report_fails_when_the_overlap_arms_are_not_ordered():
    design = run_ivd.settings()
    design["smoke"] = dict(split="val", seed=4242, episodes=2, training_seed=17)
    design["batch_size"] = 2
    design["hops"] = [1]

    def samples(order):
        names = ivd.cell_names("oracle_random")
        overlap = {"base": float("nan"), **{f"{load}-{kind}": value for load in ivd.LOADS
                                            for kind, value in zip(ivd.OVERLAPS, order)}}
        rows = [(name, episode) for name in names for episode in (0, 1)]
        return dict(condition=np.array([r[0] for r in rows]), L=np.ones(len(rows), dtype=int),
                    query_id=np.zeros(len(rows), dtype=int), overlap_mean=np.array([overlap[r[0]] for r in rows]),
                    overlap_max=np.array([overlap[r[0]] for r in rows]))

    def timing(rows):
        return dict(name="oracle_random-M1-r32-s17", writer="oracle_random", variant="M1", episodes=2, rows=rows,
                    cells=len(ivd.cell_names("oracle_random")), seconds=3.0)

    good = run_ivd.smoke_report(design, [(samples((0.0, 0.4, 0.5)), timing(2 * 1 * 2 * 7))])
    assert good["status"] == "passed" and good["failures"] == []
    # three seconds for one batch scale to 2000 / 2 = 1000 batches for each of the three training seeds,
    # and the suggested walltime adds the registered 1.5 margin, rounded up to whole minutes
    assert good["main_seconds_estimate"] == pytest.approx(3.0 * 1000 * 3)
    assert good["suggested_walltime_seconds"] == 13500
    bad = run_ivd.smoke_report(design, [(samples((0.5, 0.4, 0.0)), timing(2 * 1 * 2 * 7))])
    assert bad["status"] == "failed" and any("not ordered" in message for message in bad["failures"])
    short = run_ivd.smoke_report(design, [(samples((0.0, 0.4, 0.5)), timing(5))])
    assert short["status"] == "failed" and any("unexpected number" in message for message in short["failures"])


def test_stages_run_end_to_end_and_the_analysis_audits_every_seal(tiny, monkeypatch):
    tmp_path, p1s, f2 = tiny
    locked, commit = _lock_design(tmp_path, p1s, f2)
    smoke_dir, run_dir, analysis = tmp_path / "smoke", tmp_path / "run", tmp_path / "analysis"
    assert run_ivd.main(["smoke", "--design", str(locked), "--out", str(smoke_dir), "--device", "cpu"]) == 0
    report = json.loads((smoke_dir / "smoke.json").read_text())
    assert report["status"] == "passed" and len(report["timings"]) == 4
    assert set(report["overlaps"]) == {"oracle_random", "joint_fact"} and report["suggested_walltime_seconds"] > 0
    with pytest.raises(ValueError, match="absolute path"):  # no result may land in the job's working directory
        run_ivd.main(["smoke", "--design", str(locked), "--out", "relative-smoke", "--device", "cpu"])
    assert not Path("relative-smoke").exists()
    # the main run needs the committed lock, so a modified workspace design is refused
    locked.write_text(locked.read_text() + " ")
    with pytest.raises(ValueError, match="differs from its Git lock"):
        run_ivd.main(["run", "--design", str(locked), "--design-commit", commit, "--smoke-dir", str(smoke_dir),
                      "--out", str(run_dir), "--device", "cpu"])
    _git(locked.parents[2], "checkout", "-q", "--", "reports/ivd/design.json")
    assert not run_dir.exists()
    assert run_ivd.main(["run", "--design", str(locked), "--design-commit", commit, "--smoke-dir", str(smoke_dir),
                         "--out", str(run_dir), "--device", "cpu"]) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["checkpoints_unchanged"] is True
    assert len(manifest["runs"]) == 12 and len(list((run_dir / "samples").glob("*.npz"))) == 12
    assert manifest["node_hours"] == pytest.approx(manifest["total_seconds"] / 3600.0)

    assert analyze_ivd.main(["--run", str(run_dir), "--out", str(analysis)]) == 0
    with (analysis / "interference.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 3 * 2  # writers x hop groups x loads
    decision = [row for row in rows if row["role"] == "decision"]
    assert sorted(row["writer"] for row in decision) == ["joint_fact", "oracle_random"]
    assert all(row["load"] == "L24" and row["hop_group"] == "reach" and float(row["confidence"]) == 0.975
               for row in decision)
    summary = json.loads((analysis / "summary.json").read_text())
    assert set(summary["decisions"]) == {"oracle_random", "joint_fact"}
    assert all(entry["label"] in {"supported", "not_supported", "inconclusive"}
               for entry in summary["decisions"].values())
    assert "Registered decision" in (analysis / "summary.md").read_text()
    # the oracle writer's constructed overlap contrast is large, so a null result there is never 'inconclusive'
    assert summary["overlap_gaps"]["oracle_random"]["L24"] > analyze_ivd.MIN_OVERLAP_GAP

    # tampering is refused before any statistic is computed
    victim = next((run_dir / "samples").glob("*.npz"))
    original = victim.read_bytes()
    victim.write_bytes(original[:-4] + b"zzzz")
    with pytest.raises(ValueError, match="samples missing or changed"):
        analyze_ivd.main(["--run", str(run_dir), "--out", str(tmp_path / "again")])
    victim.write_bytes(original)
    entry = json.loads(locked.read_text())["checkpoints"][0]
    checkpoint = Path(entry["path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checkpoint changed"):
        analyze_ivd.main(["--run", str(run_dir), "--out", str(tmp_path / "again")])


def test_evaluated_checkpoint_matches_the_standard_evaluator_on_the_baseline_cell(tiny):
    """The baseline cell of an IVD evaluation is the ordinary candidate-restricted read of the unchanged episodes."""
    _, p1s, f2 = tiny
    design = run_ivd.build_design(p1s, f2)
    entry = next(e for e in design["checkpoints"] if e["writer"] == "joint_fact" and e["variant"] == "M1"
                 and e["seed"] == 17)
    samples, timing = run_ivd.evaluate_checkpoint(entry, design, design["confirmation"], device="cpu")
    assert timing["cells"] == len(ivd.cell_names("joint_fact")) and timing["queries"] == EPISODES * len(HOPS)
    from s9_torch import load_frozen
    model = load_frozen(entry, "cpu")
    config = {**model.config, "eval": {**model.config["eval"], "seed": 999004, "depths": [1, 4], "hops": HOPS,
                                       "splits": ["test"], "counterfactuals": False, "episodes": EPISODES}}
    out = Path(entry["path"]).parent / "reference"
    evaluate(model, config, out, samples_only=True)
    reference = np.load(out / "samples.npz")
    order = np.lexsort((reference["query_id"], reference["episode_id"], reference["L"]))
    base = (samples["condition"] == "base")
    mine = np.lexsort((samples["query_id"][base], samples["episode_id"][base], samples["L"][base]))
    assert np.array_equal(samples["correct"][base][mine], reference["correct"][order])
    assert np.array_equal(samples["episode_id"][base][mine], reference["episode_id"][order])
