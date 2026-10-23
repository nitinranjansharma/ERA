"""Build audit fixtures for the anneal escrow.

These files are NOT 15B of Indic text. They are shard IDs, canonical hashes, and
tiny payloads so that tests/test_escrow.py can prove:

  1. shard IDs are unique
  2. hashes match the files on disk
  3. S1–S3 loader configs cannot name an escrow shard
  4. S4 scheduled usage ≤ unique × epoch cap

Run:  python scripts/build_escrow.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import EPOCH_CAP, ROOT, SUPPLY, ipf_stages, stage_tokens, indic_tiers, ANN_SHARE, ANN_B

SHARD_DIR = ROOT / "data" / "escrow_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

POOLS = [
    dict(id="escrow.indic.tier-a.v1", unique_b=SUPPLY["indic_A_escrow"],
         lane="indic", note="verified Indic quality partition, disjoint from main-run remainder"),
    dict(id="escrow.reason.aon-long.v1", unique_b=SUPPLY["reason_escrow"],
         lane="reason", note="long verified traces, disjoint from PT-main and SFT holdout"),
    dict(id="escrow.agentic.swe-gym.v1", unique_b=0.150,
         lane="agentic", note="SWE-Gym long trajectories"),
    dict(id="escrow.agentic.swe-smith.v1", unique_b=0.120,
         lane="agentic", note="SWE-smith long trajectories"),
    dict(id="escrow.agentic.openhands.v1", unique_b=0.090,
         lane="agentic", note="OpenHands rollouts"),
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    shards = []
    for pool in POOLS:
        path = SHARD_DIR / f"{pool['id']}.jsonl"
        payload = {
            "shard_id": pool["id"],
            "unique_tokens_b": pool["unique_b"],
            "lane": pool["lane"],
            "note": pool["note"],
            "fixture": True,
            "warning": "Audit fixture. Not production tokens.",
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        shards.append(dict(
            shard_id=pool["id"],
            path=str(path.relative_to(ROOT)),
            sha256=sha256_file(path),
            unique_tokens_b=pool["unique_b"],
            lane=pool["lane"],
            allowed_stages=["S5 anneal"],
            forbidden_stages=["S1 seed", "S2 general", "S3 reason/code", "S4 long-ctx"],
        ))

    M = ipf_stages()
    st = stage_tokens(M)
    indic_per, _ = indic_tiers(st)
    scheduled = {
        "indic": indic_per["S5 anneal"]["A"],
        "reason": st["S5 anneal"]["reason"],
        "agentic": st["S5 anneal"]["agentic"],
    }
    unique = {
        "indic": SUPPLY["indic_A_escrow"],
        "reason": SUPPLY["reason_escrow"],
        "agentic": SUPPLY["agentic_traj"],
    }

    manifest = dict(
        version=1,
        epoch_cap=EPOCH_CAP,
        shards=shards,
        scheduled_s4_b=scheduled,
        unique_b=unique,
        note="S1–S3 physically cannot name these shard IDs; the loader config is the lock.",
    )
    man_path = ROOT / "data" / "escrow_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")

    s1_s3 = dict(
        stages=["S1 seed", "S2 general", "S3 reason/code", "S4 long-ctx"],
        allowed_shard_prefixes=["web/", "code/", "stem/", "indic/sangraha/unverified",
                                "indic/sangraha/verified/", "indic/sangraha/synthetic",
                                "indic/indiccorpv2/", "indic/bpcc/", "reason/aon-pt/",
                                "reason/openthoughts", "reason/openmath", "reason/openr1",
                                "reason/numina", "agentic/fc/", "longctx/"],
        forbidden_shard_prefixes=["escrow/", "holdout/sft/"],
        forbidden_shard_ids=[s["shard_id"] for s in shards],
    )
    s5 = dict(
        stages=["S5 anneal"],
        allowed_shard_ids=[s["shard_id"] for s in shards],
        allowed_shard_prefixes=["escrow/", "web/", "code/", "stem/", "indic/", "reason/", "longctx/"],
        forbidden_shard_prefixes=["holdout/sft/"],
    )
    (ROOT / "data" / "loader_s1_s3.json").write_text(json.dumps(s1_s3, indent=2) + "\n")
    (ROOT / "data" / "loader_s5.json").write_text(json.dumps(s5, indent=2) + "\n")
    print(f"wrote {man_path}")
    print(f"wrote {len(shards)} shard fixtures under {SHARD_DIR}")
    for lane, d in scheduled.items():
        ep = d / unique[lane]
        print(f"  {lane}: scheduled {d:.2f}B / unique {unique[lane]:.2f}B = {ep:.2f} epochs")


if __name__ == "__main__":
    main()
