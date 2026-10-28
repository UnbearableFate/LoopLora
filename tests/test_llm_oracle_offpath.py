"""oracle_random is bitwise unchanged by WP-L6 (baseline commit dc659e0, the master commit WP-L6 branched from).

Run only through the compute test workflow (docs/reports/WP-L6.md).

1. The writer state hash recorded before the change (reports/llm-oracle/benchmark.json, WP-C7) is reproduced on CPU.
2. The five files WP-L6 changed (llm/config.py, evaluate.py, oracle.py, run.py, train.py) are taken back to the
   baseline in a copy of the source tree. tests/assets/oracle_offpath_probe.py runs in a fresh process on that copy and
   on the current tree: code tables and written memory at r = 32 and 128, the resolved hash of every committed P1-L
   config, five tiny-model training runs (M1, B8, B4, B5, B3: losses, summaries, adapter and checkpoint hashes, writer
   signature, memory and logits, parameter counts), run_one records and compare_suite. Every recorded value must be
   identical. evaluate.py is in the set because it imports split_window from the changed config.py (a baseline
   config.py cannot serve the current evaluate.py).

The baseline commit must be reachable: set LOOPLORA_REPO to a repository that has it if neither this checkout nor the
central one does. If a later change legitimately edits one of the five files, retire part 2 and keep part 1.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from looplora.infra.seeds import stable_hash
from looplora.llm.oracle import OracleRandomWriter
from looplora.sym.evaluate import tensor_hash

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "dc659e0"
CHANGED = ("config.py", "evaluate.py", "oracle.py", "run.py", "train.py")
PROBE = ROOT / "tests" / "assets" / "oracle_offpath_probe.py"
RECORDED_CODE_HASH = "55179f8658c8c3a00a8aa58ed68deacd"


def test_oracle_random_code_tables_match_the_hash_recorded_before_wp_l6():
    recorded = json.loads((ROOT / "reports/llm-oracle/benchmark.json").read_text())
    assert {cell["code_hash"] for cell in recorded["cells"]} == {RECORDED_CODE_HASH}
    writer = OracleRandomWriter(V=1024, n_rel=4, seed=17, r_mem=32)
    assert stable_hash({name: tensor_hash(value) for name, value in writer.state_dict().items()}) == RECORDED_CODE_HASH


def repository() -> Path:
    for candidate in (os.environ.get("LOOPLORA_REPO"), ROOT, "/work/xg24i002/x10041/LoopLora"):
        if candidate and subprocess.run(["git", "-C", str(candidate), "cat-file", "-e", f"{BASELINE}^{{commit}}"],
                                        capture_output=True).returncode == 0:
            return Path(candidate)
    pytest.fail(f"baseline commit {BASELINE} is not reachable; set LOOPLORA_REPO to a repository that has it")


def baseline_source(tmp_path: Path) -> Path:
    source = tmp_path / "baseline" / "src"
    shutil.copytree(ROOT / "src", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in CHANGED:
        blob = subprocess.run(["git", "-C", str(repository()), "show", f"{BASELINE}:src/looplora/llm/{name}"],
                              check=True, capture_output=True).stdout
        (source / "looplora" / "llm" / name).write_bytes(blob)
    return source


def probe(source: Path, out: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(source), "LL_SOURCE_ROOT": str(ROOT), "PYTHONHASHSEED": "0"}
    result = subprocess.run([sys.executable, str(PROBE), "--configs", str(ROOT / "configs/p1l"), "--out", str(out)],
                            env=env, capture_output=True, text=True, timeout=1500)
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(out.read_text())


def test_oracle_random_paths_are_bitwise_identical_to_the_baseline(tmp_path):
    source = baseline_source(tmp_path)
    # The copy really is the baseline for these files, and the current tree really is not.
    assert "oracle_rot" not in (source / "looplora/llm/config.py").read_text()
    assert "oracle_rot" in (ROOT / "src/looplora/llm/config.py").read_text()
    baseline = probe(source, tmp_path / "baseline.json")
    current = probe(ROOT / "src", tmp_path / "current.json")
    assert baseline.pop("source") == str(source.resolve())
    assert current.pop("source") == str((ROOT / "src").resolve())
    assert baseline.keys() == current.keys()
    for part in baseline:
        assert baseline[part] == current[part], part
    assert len(baseline["configs"]) == len(list((ROOT / "configs/p1l").rglob("*.json"))) > 0
    assert set(baseline["train"]) == {"M1", "B8", "B4", "B5", "B3"}
    assert baseline["writer"]["32"]["code_hash"] == RECORDED_CODE_HASH
