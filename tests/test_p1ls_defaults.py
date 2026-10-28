"""Every P1-L default is unchanged by the P1L-S pipeline work (WP-L6 step 1c; docs/reports/WP-L6.md).

The baseline is wp/p1l at 02fc561: src/, scripts/ and configs/ taken whole from `git archive`, so the baseline is a
consistent tree (overlaying single files would not be). tests/assets/p1ls_defaults_probe.py runs in a fresh process on
the baseline and on this checkout, on identical inputs, and every part of its record must be equal: resolved config
hashes (training and both confirmation templates), run_p1l_main's labels and configs, the launcher's command lines,
receipts and manifests, analyze size/select, check_training on two accepted and 21 edited inputs (same exception
text), endpoints and the memory-use text, finalize's output files on a synthetic r = 32 matrix, and evaluate()'s
episode windows. Run only through the compute test workflow.

When wp/p1l has been merged into master after its compute tests, merge master into wp/oracle and set BASELINE to
that merge commit: the proof then covers everything that reached master. The commit must be reachable: set
LOOPLORA_REPO to a repository that has it if neither this checkout nor the central one does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "assets"))
import p1l_matrix  # noqa: E402

BASELINE = "02fc561862ce0f09c2460589c79e7f6d389af46c"
PROBE = ROOT / "tests" / "assets" / "p1ls_defaults_probe.py"


def repository() -> Path:
    for candidate in (os.environ.get("LOOPLORA_REPO"), ROOT, "/work/xg24i002/x10041/LoopLora"):
        if candidate and subprocess.run(["git", "-C", str(candidate), "cat-file", "-e", f"{BASELINE}^{{commit}}"],
                                        capture_output=True).returncode == 0:
            return Path(candidate)
    pytest.fail(f"baseline commit {BASELINE} is not reachable; set LOOPLORA_REPO to a repository that has it")


@pytest.fixture(scope="module")
def baseline(tmp_path_factory) -> Path:
    tree = tmp_path_factory.mktemp("baseline")
    archive = subprocess.run(["git", "-C", str(repository()), "archive", BASELINE, "src", "scripts", "configs"],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-xf", "-", "-C", str(tree)], input=archive, check=True)
    return tree


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory) -> Path:
    """The synthetic P1-L matrix, built once: both trees read the same files."""
    root = tmp_path_factory.mktemp("matrix")
    paths = p1l_matrix.build_p1l(root)
    keys = ("manifest32", "manifest128", "authorization", "evaluations")
    (root / "fixtures.json").write_text(json.dumps({key: str(paths[key]) for key in keys}))
    return root / "fixtures.json"


def probe(tree: Path, fixtures: Path, out: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "PYTHONHASHSEED": "0"}
    command = [sys.executable, str(PROBE), "--tree", str(tree), "--fixtures", str(fixtures),
               "--artifacts", str(out / "finalize"), "--out", str(out / "probe.json")]
    out.mkdir(parents=True)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=3000)
    assert result.returncode == 0, result.stderr[-6000:]
    return json.loads((out / "probe.json").read_text())


def test_p1l_defaults_are_bitwise_identical_to_the_baseline(baseline, fixtures, tmp_path):
    # The baseline is really the baseline, and this checkout really has the P1L-S pipeline.
    assert not (baseline / "scripts" / "make_p1ls_configs.py").exists()
    assert "split_window" not in (baseline / "src/looplora/llm/config.py").read_text()
    assert "STUDIES" not in (baseline / "scripts" / "run_p1l_main.py").read_text()
    assert (ROOT / "scripts" / "make_p1ls_configs.py").exists()
    assert "split_window" in (ROOT / "src/looplora/llm/config.py").read_text()
    before = probe(baseline, fixtures, tmp_path / "baseline")
    after = probe(ROOT, fixtures, tmp_path / "current")
    assert before.pop("source") == str((baseline / "src").resolve())
    assert after.pop("source") == str((ROOT / "src").resolve())
    assert before.keys() == after.keys()
    for part in before:
        assert before[part] == after[part], part
    # The comparison is not vacuous.
    assert len(before["configs"]) == len(list((ROOT / "configs/p1l").rglob("*.json"))) > 0
    cases = before["check_training"]
    assert len(cases) == 23 and cases["accepted"] == "ok" and cases["record_B0_steps_unchecked"] == "ok"
    assert len({text for text in cases.values() if text != "ok"}) >= 10
    assert {name: len(value["commands"]) for name, value in before["launcher"].items() if name != "walltime"} == {
        "pilot-rNone": 7, "pilot-r32": 7, "main-r32": 19, "pilot-r128": 5, "main-r128": 15}
    assert before["evaluate"]["confirm"]["calls"] == [["test", 100000, 2, 4242]]
    assert before["evaluate"]["confirm"]["episode_ranges"] == {"confirm": dict(split="test", start=100000, count=2)}
    assert json.loads(before["finalize"]["endpoints.json"])["writer"] == "oracle_random"
    assert all(before["finalize"].values())
