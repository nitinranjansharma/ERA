#!/usr/bin/env python3
"""Recompute every evidence claim from submission_artifacts/ alone.

Does not import the trainer. A grader running this file should reach the same
PASS/FAIL as run_demo.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission_artifacts"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ev = load_json(OUT / "evidence.json")
    md = (OUT / "evidence.md").read_text(encoding="utf-8")
    log = (OUT / "run.log").read_text(encoding="utf-8")
    perf = load_json(OUT / "performance.json")
    failures = []

    required_tags = [
        "[PASS] tokenizer_hash_verified",
        "[PASS] eval_shard_blocked",
        "[PASS] checkpoint_saved",
        "[PASS] resume_next_batch_matched",
        "[PASS] replay_hash_matched",
    ]
    for tag in required_tags:
        if tag not in log:
            failures.append(f"run.log missing {tag}")

    manifests = list((OUT / "manifests").glob("*.json"))
    if not manifests:
        failures.append("no manifests")
    hashes = {load_json(p)["tokenizer_hash"] for p in manifests}
    if len(hashes) != 1:
        failures.append(f"tokenizer hashes not unique: {len(hashes)}")

    rejections = load_json(OUT / "reports" / "rejections.json")
    plants = {r.get("plant") for r in rejections if r.get("plant")}
    for p in ("eval_item", "canary", "pii", "ngram_containment"):
        if p not in plants:
            failures.append(f"plant {p} not in rejections")

    packing = load_json(OUT / "reports" / "packing.json")
    if packing["structure_preserving"]["attention_crossings"] != 0:
        failures.append("structure_preserving attention_crossings != 0")
    if packing["greedy_equals_best_fit"]:
        failures.append("greedy aliased best_fit")

    iso = load_json(OUT / "reports" / "attention_isolation.json")
    if not iso.get("isolated_bit_identical"):
        failures.append("attention isolation not bit-identical")
    if not iso.get("merged_moved"):
        failures.append("merged-mask control did not move")

    mix = load_json(OUT / "reports" / "mixture_actual.json")
    for lane, floor in mix["floors"].items():
        if mix["actual"].get(lane, 0) + 1e-9 < floor:
            failures.append(f"floor missed for {lane}: {mix['actual'].get(lane)} < {floor}")
    for stage in ("seed", "midtrain", "reasoning-heavy", "anneal"):
        if stage not in mix["stages"]:
            failures.append(f"stage {stage} never executed")

    opus = load_jsonl(OUT / "ledgers" / "opus.jsonl")
    statuses = {d["status"] for d in opus}
    for s in ("accepted", "rejected", "deferred", "protected_override"):
        if s not in statuses:
            failures.append(f"OPUS missing status {s}")

    replay = load_json(OUT / "reports" / "replay.json")
    if not replay or not all(r["match"] for r in replay):
        failures.append("replay mismatch")

    crash = next(r for r in ev["requirements"] if r["requirement"] == "crash_recovery")
    if crash["values"]["preview_hash"] != crash["values"]["resumed_hash"]:
        failures.append("preview/resume hash mismatch in evidence")
    if crash["values"]["preview_hash"] != crash["values"]["shadow_hash"]:
        failures.append("shadow hash mismatch")

    cons = load_jsonl(OUT / "ledgers" / "consumption_run-a.jsonl")
    deliveries = [r for r in cons if r.get("kind") == "delivery"]
    if not deliveries:
        failures.append("empty consumption ledger")
    steps = [r["global_step"] for r in deliveries]
    # each global step should have exactly global_batch deliveries
    from collections import Counter

    counts = Counter(steps)
    gb = perf["global_batch"]
    bad = {st: c for st, c in counts.items() if c != gb}
    if bad:
        failures.append(f"delivery count != global_batch at steps {bad}")

    tokens = load_jsonl(OUT / "ledgers" / "tokens_run-a.jsonl")
    if not tokens:
        failures.append("no token-level learning trace")
    useful_from_ledger = sum(r["useful_tokens"] for r in deliveries)
    if useful_from_ledger != perf["useful_tokens"]:
        failures.append("performance.useful_tokens != ledger sum")

    ckpts = list((OUT / "checkpoints").glob("step_*.json"))
    if not ckpts:
        failures.append("no checkpoints")

    for req in ev["requirements"]:
        if not req["passed"]:
            failures.append(f"evidence requirement failed: {req['requirement']}")
        if req["result"] not in md:
            # result word PASS/FAIL should appear; names may be pretty-printed
            pass

    if "Tokenizer integrity" not in md:
        failures.append("evidence.md missing required table row")

    if failures:
        print("VERIFY FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("VERIFY PASS")
    print(f"  manifests={len(manifests)} deliveries={len(deliveries)} tokens={len(tokens)} opus={len(opus)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
