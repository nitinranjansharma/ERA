#!/usr/bin/env python3
"""Read-only auditor over submission_artifacts/.

    python audit.py --step 7
    python audit.py --interval 8 12
    python audit.py --shard web__train_000
    python audit.py --branches
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "submission_artifacts"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def deliveries(branch: str = "run-a"):
    return [
        r
        for r in load_jsonl(OUT / "ledgers" / f"consumption_{branch}.jsonl")
        if r.get("kind") == "delivery"
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int)
    p.add_argument("--interval", nargs=2, type=int, metavar=("START", "END"))
    p.add_argument("--shard", type=str)
    p.add_argument("--branches", action="store_true")
    args = p.parse_args()
    if args.step is not None:
        rows = [r for r in deliveries() if r["global_step"] == args.step]
        print(json.dumps(rows, indent=2))
        return 0
    if args.interval:
        lo, hi = args.interval
        rows = [r for r in deliveries() if lo <= r["global_step"] < hi]
        shards = sorted({sid for r in rows for sid in r["shard_ids"]})
        print(json.dumps({"start": lo, "end": hi, "n": len(rows), "shards": shards}, indent=2))
        return 0
    if args.shard:
        rows = [r for r in deliveries() if args.shard in r["shard_ids"]]
        steps = sorted({r["global_step"] for r in rows})
        print(json.dumps({"shard": args.shard, "steps": steps, "n": len(rows)}, indent=2))
        return 0
    if args.branches:
        fork = json.loads((OUT / "ledgers" / "fork.json").read_text(encoding="utf-8"))
        print(json.dumps(fork, indent=2))
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
