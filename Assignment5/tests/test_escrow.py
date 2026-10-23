"""Escrow is a tested property, not a README sentence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ledger import EPOCH_CAP, ROOT

MAN = ROOT / "data" / "escrow_manifest.json"
S13 = ROOT / "data" / "loader_s1_s3.json"
S5 = ROOT / "data" / "loader_s5.json"


def _load():
    if not MAN.exists():
        pytest.skip("run python scripts/build_escrow.py first")
    return json.loads(MAN.read_text()), json.loads(S13.read_text()), json.loads(S5.read_text())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def test_shard_ids_unique():
    man, _, _ = _load()
    ids = [s["shard_id"] for s in man["shards"]]
    assert ids and len(ids) == len(set(ids))


def test_hashes_match_files_on_disk():
    man, _, _ = _load()
    for s in man["shards"]:
        path = ROOT / s["path"]
        assert path.exists(), path
        assert sha256_file(path) == s["sha256"], s["shard_id"]


def test_s1_s3_cannot_name_escrow_shards():
    man, s13, _ = _load()
    escrow_ids = {s["shard_id"] for s in man["shards"]}
    allowed = set(s13.get("allowed_shard_ids", []))
    assert allowed.isdisjoint(escrow_ids)
    forbidden = set(s13["forbidden_shard_ids"])
    assert escrow_ids <= forbidden
    for prefix in s13["forbidden_shard_prefixes"]:
        assert prefix.startswith("escrow") or prefix.startswith("holdout")


def test_s1_s3_forbidden_prefixes_cover_every_escrow_path():
    man, s13, _ = _load()
    for s in man["shards"]:
        assert any(s["path"].startswith(p) or s["shard_id"].startswith("escrow")
                   for p in s13["forbidden_shard_prefixes"])


def test_s5_can_read_escrow_and_cannot_read_sft_holdout():
    man, _, s5 = _load()
    escrow_ids = {s["shard_id"] for s in man["shards"]}
    assert escrow_ids <= set(s5["allowed_shard_ids"])
    assert "holdout/sft/" in s5["forbidden_shard_prefixes"]


def test_scheduled_usage_fits_unique_supply():
    man, _, _ = _load()
    # Indic and reasoning reserves are sized to the 4-epoch cap.
    for lane in ("indic", "reason"):
        demand = man["scheduled_s4_b"][lane]
        unique = man["unique_b"][lane]
        assert demand / unique <= man["epoch_cap"] + 1e-9, (lane, demand, unique)
    # Agentic is the named synthesis programme: unique 0.36B cannot cover 7.2B at 4 epochs.
    ag_ep = man["scheduled_s4_b"]["agentic"] / man["unique_b"]["agentic"]
    assert ag_ep > man["epoch_cap"]
    need = man["scheduled_s4_b"]["agentic"] / man["epoch_cap"] - man["unique_b"]["agentic"]
    assert need > 1.0  # >1B must be synthesized; AGENTIC.md sizes this in samples.


def test_loader_files_are_not_mere_existence_checks():
    man, s13, s5 = _load()
    assert man["shards"]
    assert s13["forbidden_shard_ids"]
    assert s5["allowed_shard_ids"]
    # A test that only checked Path.exists() would pass an empty file. These do not.
    assert all(s["sha256"] for s in man["shards"])
