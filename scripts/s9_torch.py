"""Torch-side helpers of the WP-S9 scripts (import only inside a Miyabi-G allocation, or in CPU tests there)."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch

from looplora.sym.train import load_checkpoint
from s9_common import require


def state_hash(model: torch.nn.Module) -> str:
    """blake2b over every named tensor (name, dtype, shape, bytes) of a model."""
    digest = hashlib.blake2b(digest_size=16)
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_frozen(entry: Mapping[str, Any], device: str) -> Any:
    """Load a sealed checkpoint on ``device`` in eval mode with gradients off; its identity must match the entry."""
    model = load_checkpoint(entry["path"], device=device)
    config = model.checkpoint_config
    identity = (config["writer"], config["variant"], config["seed"], config["model"]["r"])
    require(identity == (entry["writer"], entry["variant"], entry["seed"], entry["r"]),
            f"checkpoint identity {identity} differs from its manifest entry {entry['name']}")
    model.requires_grad_(False)
    return model.eval()


def synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
