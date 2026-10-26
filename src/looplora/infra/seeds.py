"""Deterministic seed derivation and content hashing (blake2b; SHA-256 is not used in this project)."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any, digest_size: int = 16) -> str:
    """Hex blake2b digest of a JSON-serialisable object with sorted keys."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def derive_seed(base_seed: int, *labels: Any) -> int:
    """63-bit seed determined by a base seed and a purpose label path, e.g. (17, "train", "episode", 5)."""
    digest = hashlib.blake2b(json.dumps([base_seed, *labels], default=str).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)
