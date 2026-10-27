"""UNS Part B runner and audit on tiny CPU checkpoints (lock, run, audit).

Run only inside a Miyabi-G compute allocation, including the CPU cases.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from looplora.sym import uns
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.model import SymbolicModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_uns  # noqa: E402
import run_uns  # noqa: E402

EPISODES = 6
SEEDS = uns.SEEDS


def _config(writer, variant, seed):
    """The registered task and rank with a tiny model, trained at the depths of the extension's readers."""
    base = {
        "name": "uns-script-tiny", "device": "cpu", "seed": seed, "writer": writer, "variant": variant,
        "model": {"d_e": 8, "r": 32, "r_mem": 32, "max_hop": 16},
        "task": {"name": "t2_chain_end", "E": 12, "Q": 4, "V": 1024, "n_rel": 4, "seen": [0, 768],
                 "unseen": [768, 1024], "chain_lengths": [8, 4], "hops": [1, 2, 3]},
        "train": {"steps": 1, "batch_size": 2, "warmup_steps": 0,
                  "depths": run_uns.TRAINED_DEPTHS[variant], "log_every": 1},
        "eval": {"episodes": EPISODES, "batch_size": 3, "splits": ["val"], "depths": [4], "hops": [1, 2, 3],
                 "counterfactuals": False},
    }
    return resolve_config(recursive_merge(base, {}))


def _digest(path):
    return hashlib.blake2b(Path(path).read_bytes(), digest_size=16).hexdigest()


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """Eighteen tiny sealed checkpoints in the released layouts, a P2-S-F1 manifest, and shrunk constants."""
    for module in (run_uns, audit_uns):
        monkeypatch.setattr(module, "require_compute", lambda: None)
    monkeypatch.setattr(uns, "EPISODES", EPISODES)
    monkeypatch.setattr(uns, "EXTENSION", dict(seed=999006, episodes=EPISODES, splits=["test", "test_unseen"],
                                               hops=list(uns.HOPS), depths=list(uns.DEPTHS), batch_size=3))
    p1s, f2, repo = tmp_path / "p1s", tmp_path / "f2", tmp_path / "repo"
    rows = []
    for writer in run_uns.WRITERS:
        for seed in SEEDS:
            for variant in run_uns.READERS:
                config = _config(writer, variant, seed)
                torch.manual_seed(seed)
                record = {"config": config, "model": SymbolicModel(config).state_dict(), "summary": {}}
                if variant == "B4":
                    path = uns.run_root(writer, variant, seed, p1s_root=p1s, f2_root=f2) / "checkpoint.pt"
                else:
                    name = f"main-{writer}-M1-r32-m32-s{seed}-lr0.003"
                    path = tmp_path / "p2sf1" / "main" / "shard-00" / name / "checkpoint.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(record, path)
                if variant == "M1":
                    rows.append(dict(name=path.parent.name, source="p2s-f1", writer=writer, variant="M1", r=32,
                                     r_mem=32, seed=seed, steps=15000, descriptive_only=False, checkpoint=str(path),
                                     checkpoint_hash=_digest(path)))
                    rows.append(dict(rows[-1], name="wph-decoy", source="wph", checkpoint_hash="0" * 32))
    manifest = repo / run_uns.P2SF1
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _git(repo, "init", "-q")
    return tmp_path, p1s, f2, repo


def _lock_design(tmp_path, p1s, f2, repo):
    """Run the lock stage and commit its output in the scratch repository (the Git lock)."""
    design = tmp_path / "design.json"
    assert run_uns.main(["lock", "--p1s-root", str(p1s), "--f2-root", str(f2), "--repo", str(repo),
                         "--out", str(design)]) == 0
    locked = repo / "reports" / "uns" / "design.json"
    locked.parent.mkdir(parents=True)
    locked.write_bytes(design.read_bytes())
    _git(repo, "add", "reports/uns/design.json")
    _git(repo, "commit", "-q", "-m", "lock")
    return locked, _git(repo, "rev-parse", "HEAD")


def test_lock_seals_the_eighteen_checkpoints_and_the_registered_settings(tiny):
    tmp_path, p1s, f2, repo = tiny
    design = run_uns.build_design(p1s, f2, repo)
    entries = design["checkpoints"]
    assert len(entries) == 18 and {entry["writer"] for entry in entries} == {"joint", "joint_fact", "oracle_random"}
    assert design["extension"]["seed"] == 999006 and design["part"] == "B"
    assert {entry["name"] for entry in entries} == {uns.extension_name(w, r, s) for w in run_uns.WRITERS
                                                    for r in run_uns.READERS for s in SEEDS}
    m1 = next(entry for entry in entries if entry["name"] == "joint-M1L4-r32-s17")
    assert m1["trained_depths"] == [4] and "p2sf1" in m1["path"]  # taken from the P2-S-F1 rows, not the decoy
    assert len(run_uns.verify_design(design)) == 18
    with pytest.raises(ValueError):  # a drifted setting
        run_uns.verify_design({**design, "rank": 64})
    with pytest.raises(ValueError):  # a missing checkpoint
        run_uns.verify_design({**design, "checkpoints": entries[1:]})
    path = Path(entries[0]["path"])
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checkpoint changed"):
        run_uns.verify_design(design)


def test_lock_rejects_a_checkpoint_that_differs_from_the_p2s_f1_manifest_or_the_registered_depths(tiny):
    tmp_path, p1s, f2, repo = tiny
    manifest = repo / run_uns.P2SF1
    text = manifest.read_text()
    first = next(row for row in csv.DictReader(text.splitlines()) if row["source"] == "p2s-f1")
    manifest.write_text(text.replace(first["checkpoint_hash"], "f" * 32))
    with pytest.raises(ValueError, match="differs from the P2-S-F1 manifest"):
        run_uns.build_design(p1s, f2, repo)
    manifest.write_text(text)
    path = uns.run_root("joint", "B4", 17, p1s_root=p1s, f2_root=f2) / "checkpoint.pt"
    record = torch.load(path, map_location="cpu", weights_only=True)
    record["config"]["train"]["depths"] = [4]
    torch.save(record, path)
    with pytest.raises(ValueError, match="unexpected checkpoint"):
        run_uns.build_design(p1s, f2, repo)


def test_run_then_audit_writes_the_descriptive_tables_and_detects_tampering(tiny):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _lock_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "extension"
    assert run_uns.main(["run", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                         "--device", "cpu"]) == 0
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["partial"] is False and manifest["checkpoints_unchanged"] is True
    assert len(manifest["runs"]) == 18 and manifest["node_hours"] > 0
    run = manifest["runs"][0]
    assert (Path(run["directory"]) / "samples.npz").is_file() and (Path(run["directory"]) / "metrics.json").is_file()
    out = tmp_path / "out"
    assert audit_uns.main(["--extension-root", str(root), "--out", str(out), "--resamples", "2000"]) == 0
    assert json.loads((out / "audit.json").read_text())["status"] == "passed"
    with (out / "endpoints.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3 * len(uns.ENDPOINTS) * len(uns.GROUPS) * 3
    with (out / "accuracy.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3 * len(uns.CELLS) * len(uns.HOPS) * 3
    assert not (out / "reproduction.csv").exists() and not (out / "context.csv").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["part"] == "B" and summary["audit_failures"] == 0 and summary["design_lock_commit"] == commit
    assert "Part B" in (out / "summary.md").read_text()
    samples = Path(run["directory"]) / "samples.npz"
    samples.write_bytes(samples.read_bytes() + b"x")
    with pytest.raises(ValueError, match="samples changed after the run"):
        audit_uns.main(["--extension-root", str(root), "--out", str(tmp_path / "out2"), "--resamples", "2000"])


def test_a_subset_run_is_partial_and_cannot_be_audited(tiny):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _lock_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "timing"
    names = "joint-M1L4-r32-s17,joint-B4-r32-s17"
    assert run_uns.main(["run", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                         "--names", names, "--device", "cpu"]) == 0
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "partial" and manifest["partial"] is True and len(manifest["runs"]) == 2
    with pytest.raises(ValueError, match="not a completed full run"):
        audit_uns.main(["--extension-root", str(root), "--out", str(tmp_path / "out"), "--resamples", "2000"])
    with pytest.raises(ValueError, match="unregistered name"):
        run_uns.main(["run", "--design", str(locked), "--design-commit", commit, "--root", str(tmp_path / "other"),
                      "--names", "joint-M1-r32-s17", "--device", "cpu"])


def test_run_requires_the_committed_design_and_a_new_root_without_confirmation_in_its_path(tiny):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _lock_design(tmp_path, p1s, f2, repo)
    options = ["--design-commit", commit, "--device", "cpu"]
    locked.write_text(locked.read_text() + " ")  # edited after the lock
    with pytest.raises(ValueError, match="differs from its Git lock"):
        run_uns.main(["run", "--design", str(locked), "--root", str(tmp_path / "a"), *options])
    _git(repo, "checkout", "--", "reports/uns/design.json")
    with pytest.raises(ValueError, match="confirmation"):
        run_uns.main(["run", "--design", str(locked), "--root", str(tmp_path / "confirmation" / "a"), *options])
    root = tmp_path / "b"
    assert run_uns.main(["run", "--design", str(locked), "--root", str(root), "--names", "joint-B4-r32-s17",
                         *options]) == 0
    with pytest.raises(FileExistsError):
        run_uns.main(["run", "--design", str(locked), "--root", str(root), *options])
