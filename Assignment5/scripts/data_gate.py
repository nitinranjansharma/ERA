"""Data-gate table: required unique vs documented vs cleaned.

Counting unit: cleaned AND documented unique tokens. Raw tokens, repeated
training tokens, and generated tokens do not count. Session 4 provenance
stamps are not in this repository, so eligible = 0 until they are loaded.

Run:  python scripts/data_gate.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import (
    EPOCH_CAP, ROOT, SUPPLY, build, INV_PATH,
)

STARVED = [
    ("indic A escrow (verified)", "indic_A_escrow", ["sangraha-verified-escrow"]),
    ("indic A main (verified)", "indic_A_main", ["sangraha-verified-indic"]),
    ("indic B unverified", "indic_B", ["sangraha-unverified", "indiccorpv2"]),
    ("indic C parallel", "indic_C", ["bpcc"]),
    ("indic D synthetic native", "indic_D", ["sangraha-synthetic-native"]),
    ("agentic real trajectories", "agentic_traj", ["swe-gym", "swe-smith", "openhands-rollouts"]),
    ("agentic short FC", "agentic_fc", ["glaive-fc-v2", "toolbench", "toolace", "nexus-raven", "xlam-apigen", "hermes-fc"]),
    ("reason PT main", "reason_pt_main", ["aon-v4-pt", "openthoughts2", "openmathreasoning", "openr1-math", "numinamath"]),
    ("reason anneal escrow", "reason_escrow", ["aon-v4-escrow"]),
    ("reason SFT holdout", "reason_sft", ["aon-v4-sft"]),
]


def main():
    inv = json.loads(INV_PATH.read_text())
    by_id = {d["id"]: d for d in inv["datasets"]}
    b = build()
    print("=" * 100)
    print("DATA GATE  counting unit = cleaned AND documented unique tokens")
    print("Course threshold is not published in the two original files; this table does not invent one.")
    print("=" * 100)
    print(f"{'lane':<28}{'required unique':>16}{'raw stated':>12}{'documented':>12}"
          f"{'cleaned':>10}{'eligible':>10}{'shortfall':>12}{'%':>8}  evidence")
    rows = []
    for name, key, ids in STARVED:
        required = SUPPLY[key]  # unique we need to HOLD, not the demand
        raw = sum(float(by_id[i]["unique_tokens_b"]) for i in ids if i in by_id)
        documented = sum(
            float(by_id[i]["unique_tokens_b"])
            for i in ids if i in by_id and by_id[i].get("license") and by_id[i]["label"] in
            ("stated", "derived", "hypothesis")
        )
        cleaned = 0.0  # Session 4 stamps not in this repo
        eligible = 0.0
        shortfall = max(0.0, required - eligible)
        pct = 0.0 if required else 100.0
        evidence = "inventory.json (stated/derived); cleaned=0 (no Session-4 shards here)"
        print(f"{name:<28}{required:16.2f}{raw:12.2f}{documented:12.2f}"
              f"{cleaned:10.1f}{eligible:10.1f}{shortfall:12.2f}{pct:7.0f}%  {evidence}")
        rows.append(dict(lane=name, required=required, raw=raw, documented=documented,
                         cleaned=cleaned, eligible=eligible, shortfall=shortfall))
    print("\nGate verdict: NOT MET on the cleaned-token counting unit.")
    print("Documented-from-widget coverage is high; that is not the same as cleaned unique native data.")
    (ROOT / "data" / "data_gate.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
