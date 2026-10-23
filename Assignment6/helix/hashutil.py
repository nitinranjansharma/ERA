"""SHA-256 helpers, hash-chain links, and seed-derived RNGs.

All random draws use (seed, purpose, *parts) so the stream is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable


def canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_dumps(obj))


def sha256_ints(values: Iterable[int]) -> str:
    packed = b"".join(int(v).to_bytes(4, "little", signed=True) for v in values)
    return sha256_bytes(packed)


def derived_seed(seed: int, purpose: str, *parts: Any) -> int:
    material = canonical_dumps({"seed": int(seed), "purpose": purpose, "parts": list(parts)})
    return int(sha256_text(material)[:16], 16)


def derived_rng(seed: int, purpose: str, *parts: Any) -> random.Random:
    return random.Random(derived_seed(seed, purpose, *parts))


def hash_chain_link(prev_hash: str, payload: dict) -> tuple[str, str]:
    """Return (event_hash, prev_hash) for an append-only log."""
    event_hash = sha256_obj({"prev": prev_hash, "payload": payload})
    return event_hash, prev_hash


GENESIS = "0" * 64
