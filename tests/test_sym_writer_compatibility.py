"""Exact CPU regression against legacy writers at baseline commit 01fcfc5.

The checked-in fingerprints were captured before adding the factorized writer.
Run only in a Miyabi compute allocation, using the recorded PyTorch version.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.sym.config import resolve_config
from looplora.sym.model import make_task_spec
from looplora.sym.train import train


_BASELINE_COMMIT = "01fcfc5"
_WRITERS = ("oracle_random", "oracle_orth", "t1_frozen", "joint")
_VARIANTS = ("M1", "M1-lin", "B5", "B5-lin", "B3", "B4")
_DEPTHS = (0, 1, 2, 4)
_GOLDEN = Path(__file__).parent / "assets" / "sym_writer_legacy_v1.json"


def _fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "blake2b": hashlib.blake2b(value.numpy().tobytes(), digest_size=16).hexdigest(),
    }


def _legacy_config(writer: str, variant: str) -> dict[str, Any]:
    return resolve_config({
        "name": "legacy-writer-regression", "device": "cpu", "writer": writer,
        "variant": variant,
        "seed": derive_seed(17, "legacy_writer_regression", writer, variant),
        "backbone_seed": derive_seed(17, "legacy_writer_regression", "backbone"),
        "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 4},
        "task": {"name": "t2_chain_end", "E": 4, "Q": 3, "V": 64,
                 "n_rel": 4, "seen": [0, 48], "unseen": [48, 64],
                 "chain_lengths": [4], "hops": [1, 2, 3]},
        "train": {"steps": 2, "batch_size": 2, "warmup_steps": 0,
                  "log_every": 1, "depths": [1, 2, 4]},
        "pretrain": {"steps": 2, "batch_size": 2, "warmup_steps": 0,
                     "task": {"head_sharing": .3}},
        "eval": {"episodes": 2, "batch_size": 2, "depths": list(_DEPTHS),
                 "splits": ["test"], "hops": [1, 2, 3], "counterfactuals": False},
        "runtime": {"threads": 1},
    })


def _capture_row(writer: str, variant: str, out: Path) -> dict[str, Any]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _legacy_config(writer, variant)
        model, _ = train(config, out)
        model.eval()
        # All rows use the same held-out arrays, independently of training seeds.
        arrays = generate(make_task_spec(config), "test", 0, 2,
                          derive_seed(17, "legacy_writer_regression", "heldout"))
        with torch.no_grad():
            batch = model.prepare(arrays)
            prepared = {name: _fingerprint(getattr(batch, name))
                        for name in ("memory", "u", "true_value", "codes")}
            logits = {str(depth): _fingerprint(model.read(batch, depth)["logits"])
                      for depth in _DEPTHS}
        return {
            "seed": config["seed"],
            "state_dict": {name: _fingerprint(value)
                           for name, value in sorted(model.state_dict().items())},
            "prepared": prepared,
            "logits": logits,
        }
    finally:
        torch.set_num_threads(previous_threads)


def _capture_legacy() -> dict[str, Any]:
    """Capture the baseline on compute; no test-time fixture regeneration."""
    with tempfile.TemporaryDirectory(prefix="sym-writer-legacy-") as directory:
        root = Path(directory)
        rows = {f"{writer}/{variant}": _capture_row(writer, variant, root / writer / variant)
                for writer in _WRITERS for variant in _VARIANTS}
    return {"baseline_commit": _BASELINE_COMMIT, "torch_version": str(torch.__version__),
            "device": "cpu", "threads": 1, "rows": rows}


@pytest.fixture(scope="module")
def legacy_golden():
    fixture = json.loads(_GOLDEN.read_text())
    assert fixture["baseline_commit"] == _BASELINE_COMMIT
    assert fixture["torch_version"] == str(torch.__version__), (
        "Exact legacy fingerprints require the PyTorch version used for baseline capture")
    assert fixture["device"] == "cpu" and fixture["threads"] == 1
    assert set(fixture["rows"]) == {f"{writer}/{variant}"
                                   for writer in _WRITERS for variant in _VARIANTS}
    return fixture


@pytest.mark.parametrize("writer", _WRITERS)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_legacy_writers_remain_bit_identical(writer, variant, tmp_path, legacy_golden):
    actual = _capture_row(writer, variant, tmp_path / "train")
    assert actual == legacy_golden["rows"][f"{writer}/{variant}"]
