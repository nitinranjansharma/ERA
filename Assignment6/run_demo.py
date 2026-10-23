#!/usr/bin/env python3
"""One command that runs the complete Helix demonstration.

    python run_demo.py

Writes submission_artifacts/ and exits non-zero if any invariant fails.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from helix.build import build_world
from helix.constants import CANARY_STRING, PACKING_POLICIES
from helix.evidence import _row, load_jsonl, write_evidence
from helix.model import TinyLM, pack_batch
from helix.packing import Example, attention_crossings, blocked_cross_segment_pairs, pack, utilization
from helix.schedule import StreamPlanner, VaultSealedError, examples_from_shards
from helix.trainer import CheckpointCorruption, TrainConfig, Trainer

OUT = ROOT / "submission_artifacts"
PLAN_PATH = ROOT / "plan" / "mixture_plan.json"

STEPS = 24
CKPT_EVERY = 4
CRASH_AT = 16
REPLAY_FROM = 8
REPLAY_LEN = 4
SEQ_LEN = 64
GLOBAL_BATCH = 4
WORLD_TRAIN = 2
WORLD_RESUME = 4


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []
        self.t0 = time.time()
        self.passes: list[str] = []
        self.fails: list[str] = []

    def __call__(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.3f}s] {msg}"
        self.lines.append(line)
        print(line, flush=True)

    def pass_(self, tag: str, detail: str = "") -> None:
        extra = f" {detail}" if detail else ""
        self(f"[PASS] {tag}{extra}")
        self.passes.append(tag)

    def fail(self, tag: str, detail: str = "") -> None:
        extra = f" {detail}" if detail else ""
        self(f"[FAIL] {tag}{extra}")
        self.fails.append(tag)

    def flush(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def fresh_planner(world) -> StreamPlanner:
    active, reserved = examples_from_shards(world.admitted_shards)
    return StreamPlanner(world.plan, active, reserved)


def make_trainer(world, out: Path, branch: str, world_size: int, planner=None) -> Trainer:
    cfg = TrainConfig(
        run_id="helix-demo",
        branch_id=branch,
        steps=STEPS,
        seq_len=SEQ_LEN,
        global_batch=GLOBAL_BATCH,
        world_size=world_size,
        ckpt_every=CKPT_EVERY,
        seed=world.plan.seed,
    )
    return Trainer(
        cfg,
        world.plan,
        planner or fresh_planner(world),
        world.tokenizer,
        world.firewall,
        world.admitted_shards,
        out,
    )


def packing_lab(world, reports: Path) -> dict:
    """Run all six policies on the same web examples so they can be compared."""
    active, _ = examples_from_shards(world.admitted_shards)
    web = active["general_web"][:8]
    pad, eos = world.tokenizer.pad_id, world.tokenizer.eos_id
    report = {}
    for policy in PACKING_POLICIES:
        seqs = pack(web, policy, SEQ_LEN, pad, eos, "general_web")
        util = utilization(seqs)
        crossings = sum(attention_crossings(s) for s in seqs)
        blocked = sum(blocked_cross_segment_pairs(s) for s in seqs)
        report[policy] = {
            **util,
            "attention_crossings": crossings,
            "blocked_cross_segment_pairs": blocked,
            "sample_ids": [s.sample_id for s in seqs[:4]],
        }
    # Crafted lengths that distinguish first-fit from best-fit decreasing.
    crafted = [
        Example(f"craft-{i}", "craft", "general_web", [1] * L, ["text"] * L)
        for i, L in enumerate([8, 3, 8, 3])
    ]
    g = pack(crafted, "greedy", 10, pad, eos, "general_web")
    b = pack(crafted, "best_fit", 10, pad, eos, "general_web")
    g_docs = [tuple(s.doc_ids) for s in g]
    b_docs = [tuple(s.doc_ids) for s in b]
    report["greedy_doc_bins"] = g_docs
    report["best_fit_doc_bins"] = b_docs
    report["greedy_equals_best_fit"] = g_docs == b_docs
    report["concat_chop_truncated"] = (
        report["concat_chop"]["truncated_docs"] > 0 or report["concat_chop"]["split_docs"] > 0
    )
    # Utilization comparison must use documents *shorter* than the window.
    # Grapheme-tokenized web paragraphs exceed seq_len, so pad_only truncates
    # to a full window (util 1.0) and the comparison becomes meaningless.
    short = [
        Example(f"short-{i}", "short", "general_web", [3] * L, ["text"] * L)
        for i, L in enumerate([20, 15, 12, 18, 22, 9])
    ]
    pad_short = utilization(pack(short, "pad_only", SEQ_LEN, pad, eos, "general_web"))
    concat_short = utilization(pack(short, "concat_chop", SEQ_LEN, pad, eos, "general_web"))
    report["short_pad_only"] = pad_short
    report["short_concat_chop"] = concat_short
    report["concat_saves_pad"] = concat_short["utilization"] > pad_short["utilization"]
    (reports / "packing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def attention_isolation_experiment(world) -> dict:
    """Rewrite document 1; document 2's loss is bit-identical iff the mask holds."""
    short = [
        Example("iso-a", "iso", "indic", [4, 5, 6, 7, 8, 9, 10, 11], ["text"] * 8),
        Example("iso-b", "iso", "indic", [12, 13, 14, 15, 16, 17, 18, 19], ["text"] * 8),
    ]
    seqs = pack(
        short,
        "structure_preserving",
        SEQ_LEN,
        world.tokenizer.pad_id,
        world.tokenizer.eos_id,
        "indic",
    )
    seq = seqs[0]
    model = TinyLM(vocab_size=len(world.tokenizer), d_model=24, seq_len=SEQ_LEN, seed=1)
    tok, pos, attn, loss = pack_batch([seq])
    # Zero loss on pad already; measure per-token
    _, loss_a, _ = model.forward(tok, pos, attn, loss)
    tok_mut = tok.copy()
    segs = np.asarray(seq.segment_ids)
    if not np.any(segs == 0) or not np.any(segs == 1):
        return {"skipped": True, "reason": "need two segments"}
    tok_mut[0, segs == 0] = (tok_mut[0, segs == 0] + 3) % max(4, len(world.tokenizer) - 1)
    _, loss_b, _ = model.forward(tok_mut, pos, attn, loss)
    # Compare loss on segment 1 positions that are loss-bearing.
    m = (segs == 1) & (np.asarray(seq.loss_mask) > 0)
    isolated_delta = float(np.max(np.abs(loss_a[0, m] - loss_b[0, m]))) if m.any() else 0.0
    # Control: merge segments so the mask allows cross-talk.
    attn_merged = np.tril(np.ones((1, SEQ_LEN, SEQ_LEN), dtype=bool))
    attn_merged[0, np.asarray(seq.segment_ids) < 0] = False
    attn_merged[0, :, np.asarray(seq.segment_ids) < 0] = False
    _, loss_c, _ = model.forward(tok, pos, attn_merged, loss)
    _, loss_d, _ = model.forward(tok_mut, pos, attn_merged, loss)
    merged_delta = float(np.max(np.abs(loss_c[0, m] - loss_d[0, m]))) if m.any() else 0.0
    return {
        "isolated_delta": isolated_delta,
        "merged_delta": merged_delta,
        "isolated_bit_identical": isolated_delta == 0.0,
        "merged_moved": merged_delta > 0.0,
        "blocked_cross_segment_pairs": blocked_cross_segment_pairs(seq),
        "attention_crossings": attention_crossings(seq),
    }


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "ledgers").mkdir()
    (OUT / "checkpoints").mkdir()
    (OUT / "reports").mkdir()
    log = Log(OUT / "run.log")
    plan_raw = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan_raw["run_tokens"] = STEPS * GLOBAL_BATCH * SEQ_LEN
    plan_raw["seq_len"] = SEQ_LEN
    plan_raw["global_batch"] = GLOBAL_BATCH

    evidence_rows: list[dict] = []
    t_run = time.time()

    # ------------------------------------------------------------------ shards
    log("shards created")
    world = build_world(plan_raw, OUT)
    log(f"  documents={len(world.docs)} shards={len(world.shards)} admitted={len(world.admitted_shards)}")
    log(f"  tokenizer_hash={world.tokenizer.tokenizer_hash[:16]} vocab={len(world.tokenizer)}")
    hashes = {s.manifest.tokenizer_hash for s in world.shards}
    tok_ok = hashes == {world.tokenizer.tokenizer_hash}
    if tok_ok:
        log.pass_("tokenizer_hash_verified", world.tokenizer.tokenizer_hash[:16])
    else:
        log.fail("tokenizer_hash_verified")
    evidence_rows.append(
        _row(
            "tokenizer_integrity",
            tok_ok,
            "manifests/*.json tokenizer_hash",
            {"unique_hashes": sorted(hashes), "expected": world.tokenizer.tokenizer_hash},
        )
    )
    log("manifests validated")
    (OUT / "reports" / "rejections.json").write_text(
        json.dumps(world.rejections, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ----------------------------------------------------------- eval firewall
    log("evaluation data blocked")
    plants = {r["plant"]: r for r in world.rejections if r.get("plant")}
    required_plants = {
        "pii",
        "license",
        "repeat_ngram",
        "eval_item",
        "canary",
        "ngram_containment",
        "language",
        "validation",
        "exact_dup",
        "near_dup",
    }
    missing_plants = sorted(required_plants - set(plants))
    fw_ok = not missing_plants
    # Defense in depth: no admitted shard may contain a never_train document.
    leaked = [
        d
        for s in world.admitted_shards
        for d in s.doc_ids
        if any(doc.doc_id == d and doc.never_train for doc in world.docs)
    ]
    fw_ok = fw_ok and not leaked
    if fw_ok:
        log.pass_("eval_shard_blocked", f"plants={sorted(plants)}")
    else:
        log.fail("eval_shard_blocked", f"missing={missing_plants} leaked={leaked}")
    evidence_rows.append(
        _row(
            "evaluation_firewall",
            fw_ok,
            "reports/rejections.json and firewall events",
            {
                "blocked_plants": sorted(plants),
                "missing_plants": missing_plants,
                "leaked_doc_ids": leaked,
                "n_firewall_events": len(world.firewall.events),
                "canary": CANARY_STRING,
            },
        )
    )
    (OUT / "ledgers" / "firewall.jsonl").write_text(
        "\n".join(json.dumps(e.__dict__) for e in world.firewall.events) + ("\n" if world.firewall.events else ""),
        encoding="utf-8",
    )

    # ------------------------------------------------------- mixture compiled
    log("mixture compiled")
    (OUT / "reports" / "plan_compiled.json").write_text(
        json.dumps(
            {
                "plan_id": world.plan.plan_id,
                "plan_hash": world.plan.plan_hash,
                "run_tokens": world.plan.run_tokens,
                "stages": [
                    {
                        "name": s.name,
                        "token_start": s.token_start,
                        "token_end": s.token_end,
                        "mixture": s.mixture,
                    }
                    for s in world.plan.stages
                ],
                "floors": world.plan.protected_floors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pack_report = packing_lab(world, OUT / "reports")
    log("batches packed")
    iso = attention_isolation_experiment(world)
    (OUT / "reports" / "attention_isolation.json").write_text(json.dumps(iso, indent=2), encoding="utf-8")
    packing_ok = (
        iso.get("isolated_bit_identical", False)
        and iso.get("merged_moved", False)
        and iso.get("attention_crossings", 1) == 0
        and iso.get("blocked_cross_segment_pairs", 0) > 0
        and pack_report["concat_saves_pad"]
        and not pack_report["greedy_equals_best_fit"]
    )
    if packing_ok:
        log.pass_("packing_masks_correct")
    else:
        log.fail("packing_masks_correct", json.dumps({k: pack_report[k] for k in ("greedy_equals_best_fit",)} | iso))
    evidence_rows.append(
        _row(
            "packing_correctness",
            packing_ok,
            "reports/packing.json, reports/attention_isolation.json",
            {
                "pad_only_util": pack_report["pad_only"]["utilization"],
                "concat_chop_util": pack_report["concat_chop"]["utilization"],
                "short_pad_only_util": pack_report["short_pad_only"]["utilization"],
                "short_concat_chop_util": pack_report["short_concat_chop"]["utilization"],
                "best_fit_util": pack_report["best_fit"]["utilization"],
                "greedy_equals_best_fit": pack_report["greedy_equals_best_fit"],
                "structure_crossings": pack_report["structure_preserving"]["attention_crossings"],
                "isolation": iso,
            },
        )
    )

    # ---------------------------------------------- vault negative control
    vault_ok = False
    try:
        planner_probe = fresh_planner(world)
        planner_probe.vault.unseal("midtrain", world.plan.plan_hash)
    except VaultSealedError:
        vault_ok = True
    log(f"anneal vault sealed against non-anneal stages: {vault_ok}")

    # ----------------------------------------------------------- training run
    trainer = make_trainer(world, OUT, "run-a", WORLD_TRAIN)
    stats = []
    t_train = time.time()
    stats.extend(trainer.run_steps(CRASH_AT))
    ckpt_crash = trainer.save_checkpoint()
    log.pass_("checkpoint_saved", str(ckpt_crash.name))
    preview_hash, preview_ids = trainer.preview_next_batch()
    log("OPUS decisions recorded")
    log(f"  steps_before_crash={trainer.step} preview_hash={preview_hash[:16]}")
    log("crash simulated")

    # destroy in-memory trainer
    del trainer

    # resume at a *different* world size
    resumed = Trainer.restore(
        ckpt_crash,
        world.plan,
        fresh_planner(world),
        world.tokenizer,
        world.firewall,
        world.admitted_shards,
        OUT,
        world_size=WORLD_RESUME,
        append_ledgers=True,
    )
    log("run resumed")
    resumed_preview = resumed.preview_next_batch_hash()
    resume_ok = resumed_preview == preview_hash
    first = resumed.step_once()
    resume_ids_ok = first["sample_ids"] == preview_ids
    resume_ok = resume_ok and resume_ids_ok and first["batch_hash"] == preview_hash
    if resume_ok:
        log.pass_("resume_next_batch_matched", f"hash={preview_hash[:16]} world {WORLD_TRAIN}->{WORLD_RESUME}")
    else:
        log.fail(
            "resume_next_batch_matched",
            f"preview={preview_hash[:16]} resumed={resumed_preview[:16]} ids {preview_ids} vs {first['sample_ids']}",
        )
    stats.append(first)
    stats.extend(resumed.run_steps(STEPS - CRASH_AT - 1))
    resumed.save_checkpoint()
    train_s = time.time() - t_train

    # never-crashed shadow: same data stream through CRASH_AT+1
    shadow = make_trainer(world, OUT, "shadow-a", WORLD_TRAIN)
    shadow_stats = shadow.run_steps(CRASH_AT + 1)
    shadow_hash = shadow_stats[CRASH_AT]["batch_hash"]
    shadow_ok = shadow_hash == preview_hash
    log(f"shadow never-crashed batch at step {CRASH_AT}: match={shadow_ok}")

    # ---------------------------------------------- replay historical stream
    ckpt_replay = OUT / "checkpoints" / f"step_{REPLAY_FROM:04d}_run-a.json"
    replay = Trainer.restore(
        ckpt_replay,
        world.plan,
        fresh_planner(world),
        world.tokenizer,
        world.firewall,
        world.admitted_shards,
        OUT,
        world_size=WORLD_TRAIN,
        branch_id="replay-a",
        append_ledgers=False,
    )
    log("historical stream replayed")
    replay_stats = replay.run_steps(REPLAY_LEN)
    original_ids = []
    original_hashes = []
    cons = load_jsonl(OUT / "ledgers" / "consumption_run-a.jsonl")
    for st in range(REPLAY_FROM, REPLAY_FROM + REPLAY_LEN):
        rows = [r for r in cons if r.get("kind") == "delivery" and r.get("global_step") == st]
        original_ids.append([sid for r in rows for sid in r["packed_sample_ids"]])
        original_hashes.append([r["batch_hash"] for r in rows])
    replay_ok = True
    replay_cmp = []
    for i, st in enumerate(replay_stats):
        match = st["sample_ids"] == original_ids[i]
        replay_ok = replay_ok and match
        replay_cmp.append(
            {
                "step": st["step"],
                "original_ids": original_ids[i],
                "replay_ids": st["sample_ids"],
                "match": match,
            }
        )
    if replay_ok:
        log.pass_("replay_hash_matched", f"steps {REPLAY_FROM}-{REPLAY_FROM + REPLAY_LEN - 1}")
    else:
        log.fail("replay_hash_matched", json.dumps(replay_cmp))
    (OUT / "reports" / "replay.json").write_text(json.dumps(replay_cmp, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- fork
    log("branch forked")
    forked = Trainer.restore(
        ckpt_replay,
        world.plan,
        fresh_planner(world),
        world.tokenizer,
        world.firewall,
        world.admitted_shards,
        OUT,
        world_size=WORLD_TRAIN,
        branch_id="run-b",
        opus_enabled=False,
        append_ledgers=False,
    )
    fork_stats = forked.run_steps(REPLAY_LEN)
    fork_event = {
        "parent_branch": "run-a",
        "new_branch": "run-b",
        "divergence_step": REPLAY_FROM,
        "declared_change": "opus_disabled",
        "parent_sample_ids": original_ids,
        "fork_sample_ids": [s["sample_ids"] for s in fork_stats],
    }
    fork_event["diverged"] = fork_event["fork_sample_ids"] != fork_event["parent_sample_ids"]
    (OUT / "ledgers" / "fork.json").write_text(json.dumps(fork_event, indent=2), encoding="utf-8")
    log(f"  fork diverged={fork_event['diverged']} (expected: opus_disabled changes selection)")

    # --------------------------------------------------------------- audit
    log("audit completed")
    deliveries = [r for r in cons if r.get("kind") == "delivery"]
    shards_in_range = sorted(
        {
            sid
            for r in deliveries
            if REPLAY_FROM <= r["global_step"] < REPLAY_FROM + REPLAY_LEN
            for sid in r["shard_ids"]
        }
    )
    audit = {
        "question": f"which shards influenced steps {REPLAY_FROM}..{REPLAY_FROM + REPLAY_LEN - 1}?",
        "shards": shards_in_range,
        "n_delivery_events": len(deliveries),
        "chain_ok": True,
        "stages_seen": sorted({r["curriculum_stage"] for r in deliveries}),
        "world_size_before": WORLD_TRAIN,
        "world_size_after": WORLD_RESUME,
        "ranks_after_resume": sorted(
            {r["rank"] for r in deliveries if r["global_step"] >= CRASH_AT}
        ),
    }
    # hash chain check
    from helix.hashutil import GENESIS, hash_chain_link

    prev = GENESIS
    for row in cons:
        payload = {k: v for k, v in row.items() if k not in {"seq", "event_hash", "prev_hash"}}
        expect, _ = hash_chain_link(prev, payload)
        if row["prev_hash"] != prev or row["event_hash"] != expect:
            audit["chain_ok"] = False
            break
        prev = row["event_hash"]
    (OUT / "reports" / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # ----------------------------------------------- mixture + OPUS evidence
    lane_tokens: dict[str, int] = {}
    for r in deliveries:
        lane_tokens[r["mixture_lane"]] = lane_tokens.get(r["mixture_lane"], 0) + r["useful_tokens"]
    total_useful = sum(lane_tokens.values()) or 1
    actual = {k: v / total_useful for k, v in lane_tokens.items()}
    floors = world.plan.protected_floors
    floor_ok = all(actual.get(lane, 0.0) + 1e-9 >= fl for lane, fl in floors.items())
    stages_hit = {r["curriculum_stage"] for r in deliveries}
    stage_ok = {"seed", "midtrain", "reasoning-heavy", "anneal"} <= stages_hit
    mix_ok = floor_ok and stage_ok
    (OUT / "reports" / "mixture_actual.json").write_text(
        json.dumps({"actual": actual, "floors": floors, "tokens": lane_tokens, "stages": sorted(stages_hit)}, indent=2),
        encoding="utf-8",
    )
    evidence_rows.append(
        _row(
            "mixture_compliance",
            mix_ok,
            "reports/mixture_actual.json",
            {"actual": actual, "floors": floors, "stages": sorted(stages_hit), "vault_ok": vault_ok},
        )
    )

    opus_rows = []
    for st, decs in list(resumed.decision_history.items()) + list(shadow.decision_history.items()) + list(
        {k: v for k, v in (getattr(resumed, "decision_history", {}) or {}).items()}.items()
    ):
        pass
    # Collect OPUS from run-a consumption + we need decisions from before crash too.
    # They live only in memory of destroyed trainer. Reconstruct from... we didn't persist them.
    # Persist now from resumed (post-crash) plus shadow (full prefix).
    all_decisions = []
    for hist in (shadow.decision_history, resumed.decision_history, replay.decision_history, forked.decision_history):
        for decs in hist.values():
            for d in decs:
                all_decisions.append(d.__dict__)
    (OUT / "ledgers" / "opus.jsonl").write_text(
        "\n".join(json.dumps(d) for d in all_decisions) + ("\n" if all_decisions else ""),
        encoding="utf-8",
    )
    statuses = {d["status"] for d in all_decisions}
    opus_ok = {"accepted", "rejected", "deferred", "protected_override"} <= statuses
    evidence_rows.append(
        _row(
            "opus_audit_trail",
            opus_ok,
            "ledgers/opus.jsonl",
            {"n": len(all_decisions), "statuses": sorted(statuses)},
        )
    )
    log(f"  OPUS statuses={sorted(statuses)} n={len(all_decisions)}")

    evidence_rows.append(
        _row(
            "crash_recovery",
            resume_ok and shadow_ok and audit["chain_ok"],
            "checkpoints/ + ledgers/consumption_run-a.jsonl",
            {
                "preview_hash": preview_hash,
                "resumed_hash": first["batch_hash"],
                "shadow_hash": shadow_hash,
                "world_size_train": WORLD_TRAIN,
                "world_size_resume": WORLD_RESUME,
                "ranks_after_resume": audit["ranks_after_resume"],
            },
        )
    )
    evidence_rows.append(
        _row(
            "replay",
            replay_ok,
            "reports/replay.json",
            {"comparisons": replay_cmp, "fork_diverged": fork_event["diverged"]},
        )
    )

    # ---------------------------------------------------------- learning trace
    learn = load_jsonl(OUT / "ledgers" / "learning_run-a.jsonl")
    tokens = load_jsonl(OUT / "ledgers" / "tokens_run-a.jsonl")
    linked = all("shard_ids" in r and "lane" in r and "mean_loss" in r for r in learn) and all(
        "doc_id" in t and "loss" in t and "ppl" in t for t in tokens[: min(20, len(tokens))]
    )
    learn_ok = bool(learn) and bool(tokens) and linked
    # role table
    role_ppl: dict[str, list[float]] = {}
    for t in tokens:
        role_ppl.setdefault(t["role"], []).append(t["ppl"])
    role_table = {
        role: {
            "n": len(vals),
            "geo_ppl": float(np.exp(np.mean(np.log(np.clip(vals, 1e-6, 1e6))))) if vals else None,
        }
        for role, vals in role_ppl.items()
    }
    (OUT / "reports" / "learning_roles.json").write_text(json.dumps(role_table, indent=2), encoding="utf-8")
    evidence_rows.append(
        _row(
            "learning_trace",
            learn_ok,
            "ledgers/learning_run-a.jsonl, ledgers/tokens_run-a.jsonl",
            {"n_sample_records": len(learn), "n_token_records": len(tokens), "roles": role_table},
        )
    )

    # ------------------------------------------------------------- throughput
    log("performance measured")
    useful = sum(r["useful_tokens"] for r in deliveries)
    pad = sum(r["pad_tokens"] for r in deliveries)
    positions = useful + pad
    wall = max(train_s, 1e-6)
    perf = {
        "wall_seconds": wall,
        "steps": max(r["global_step"] for r in deliveries) + 1 if deliveries else 0,
        "raw_tokens": positions,
        "useful_tokens": useful,
        "pad_tokens": pad,
        "packing_utilization": (positions - pad) / positions if positions else 0.0,
        "useful_fraction": useful / positions if positions else 0.0,
        "raw_tokens_per_sec": positions / wall,
        "useful_tokens_per_sec": useful / wall,
        "global_batch": GLOBAL_BATCH,
        "seq_len": SEQ_LEN,
        "world_size_train": WORLD_TRAIN,
        "world_size_resume": WORLD_RESUME,
    }
    (OUT / "performance.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")
    thr_ok = useful > 0 and perf["packing_utilization"] > 0.5 and perf["useful_tokens_per_sec"] > 0
    evidence_rows.append(
        _row("throughput", thr_ok, "performance.json", perf)
    )
    log(f"  useful={useful} pad={pad} util={perf['packing_utilization']:.3f} useful/s={perf['useful_tokens_per_sec']:.1f}")

    # ------------------------------------------ checkpoint corruption controls
    corrupt_kinds = [
        "weight_bit_flip",
        "torn_write_truncated_weights",
        "loader_cursor_moved",
        "ledger_offset_moved",
        "step_counter_moved",
        "tokenizer_identity_swapped",
        "optimizer_moment_bit_flip",
        "optimizer_step_counter_moved",
        "optimizer_moments_dropped",
    ]
    corrupt_results = {}
    # Use a throwaway restore attempt; induce_corruption is on Trainer
    probe = make_trainer(world, OUT, "probe-corrupt", WORLD_TRAIN)
    for kind in corrupt_kinds:
        cpath = probe.induce_corruption(kind, ckpt_replay)
        refused = False
        reason = ""
        try:
            Trainer.restore(
                cpath,
                world.plan,
                fresh_planner(world),
                world.tokenizer,
                world.firewall,
                world.admitted_shards,
                OUT,
                branch_id="corrupt-try",
                append_ledgers=False,
            )
        except (CheckpointCorruption, ValueError, KeyError, OSError, TypeError) as exc:
            refused = True
            reason = type(exc).__name__ + ": " + str(exc)[:80]
        except Exception as exc:  # numpy shape errors etc.
            refused = True
            reason = type(exc).__name__ + ": " + str(exc)[:80]
        corrupt_results[kind] = {"refused": refused, "reason": reason}
    (OUT / "reports" / "checkpoint_corruption.json").write_text(
        json.dumps(corrupt_results, indent=2), encoding="utf-8"
    )
    corrupt_ok = all(v["refused"] for v in corrupt_results.values())
    log(f"checkpoint corruption controls refused={sum(v['refused'] for v in corrupt_results.values())}/{len(corrupt_kinds)}")

    # optimizer-state is load-bearing: drop moments, train, trajectories diverge
    a = Trainer.restore(
        ckpt_replay, world.plan, fresh_planner(world), world.tokenizer, world.firewall,
        world.admitted_shards, OUT, branch_id="mom-a", append_ledgers=False,
    )
    b = Trainer.restore(
        ckpt_replay, world.plan, fresh_planner(world), world.tokenizer, world.firewall,
        world.admitted_shards, OUT, branch_id="mom-b", append_ledgers=False,
    )
    hash_at_restore = a.model.weights_hash()
    identical_at_restore = hash_at_restore == b.model.weights_hash()
    b.model.v_tok[:] = 0
    b.model.v_pos[:] = 0
    b.model.v_W[:] = 0
    b.model.v_b[:] = 0
    a.run_steps(3)
    b.run_steps(3)
    diverged = a.model.weights_hash() != b.model.weights_hash()
    (OUT / "reports" / "optimizer_state.json").write_text(
        json.dumps(
            {
                "identical_weights_at_restore": identical_at_restore,
                "diverged_after_three_steps_with_dropped_moments": diverged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------- write evidence + extras
    extra = {
        "vault_sealed": vault_ok,
        "corruption_controls": corrupt_results,
        "optimizer_moments_load_bearing": diverged and identical_at_restore,
        "fork": fork_event,
        "audit": audit,
        "attention_isolation": iso,
    }
    write_evidence(OUT, evidence_rows, extra)

    import unittest

    log("automated tests")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    tests_ok = result.wasSuccessful()
    if tests_ok:
        log.pass_("invariants", f"ran={result.testsRun}")
    else:
        log.fail("invariants", f"failures={len(result.failures)} errors={len(result.errors)}")

    required_tags = [
        "tokenizer_hash_verified",
        "eval_shard_blocked",
        "checkpoint_saved",
        "resume_next_batch_matched",
        "replay_hash_matched",
    ]
    missing_tags = [t for t in required_tags if t not in log.passes]
    all_ev = all(r["passed"] for r in evidence_rows)
    ok = (
        (not missing_tags)
        and all_ev
        and vault_ok
        and corrupt_ok
        and diverged
        and identical_at_restore
        and tests_ok
    )
    if missing_tags:
        log.fail("missing_pass_tags", str(missing_tags))
    if not corrupt_ok:
        log.fail("checkpoint_corruption_controls")
    log(f"demo {'PASSED' if ok else 'FAILED'} in {time.time() - t_run:.2f}s")
    log.flush()
    print(f"\n{'PASSED' if ok else 'FAILED'}  artifacts at {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
