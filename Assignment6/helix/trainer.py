"""Training loop with checkpoints bound to ledger offsets.

Resume continues the same run from the latest checkpoint and ledger offset.
Replay restores an older checkpoint and feeds the same historical stream.
Fork restores a checkpoint and starts a new data branch with a declared change.
Audit reconstructs which shards trained a range of checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from helix.firewall import EvalFirewall
from helix.hashutil import sha256_obj
from helix.ledger import ConsumptionLedger, LearningLedger
from helix.model import TinyLM, pack_batch
from helix.opus import Opus
from helix.packing import PackedSequence, pack
from helix.schedule import MixturePlan, StreamPlanner
from helix.shards import Shard
from helix.tokenizer import FrozenTokenizer


class CheckpointCorruption(RuntimeError):
    pass


@dataclass
class TrainConfig:
    run_id: str
    branch_id: str
    steps: int
    seq_len: int
    global_batch: int
    world_size: int
    ckpt_every: int
    d_model: int = 32
    lr: float = 0.08
    seed: int = 20251023
    opus_enabled: bool = True


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        plan: MixturePlan,
        planner: StreamPlanner,
        tokenizer: FrozenTokenizer,
        firewall: EvalFirewall,
        shards: list[Shard],
        out_dir: Path,
        model: TinyLM | None = None,
        resume_ledgers: bool = False,
    ) -> None:
        self.cfg = cfg
        self.plan = plan
        self.planner = planner
        self.tokenizer = tokenizer
        self.firewall = firewall
        self.shards = {s.manifest.shard_id: s for s in shards}
        self.out_dir = out_dir
        self.step = 0
        self.model = model or TinyLM(
            vocab_size=len(tokenizer),
            d_model=cfg.d_model,
            seq_len=cfg.seq_len,
            lr=cfg.lr,
            seed=cfg.seed,
        )
        self.opus = Opus(plan)
        ledgers = out_dir / "ledgers"
        ledgers.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.consumption = ConsumptionLedger(
            ledgers / f"consumption_{cfg.branch_id}.jsonl", resume=resume_ledgers
        )
        self.learning = LearningLedger(
            ledgers / f"learning_{cfg.branch_id}.jsonl",
            ledgers / f"tokens_{cfg.branch_id}.jsonl",
            resume=resume_ledgers,
        )
        self.intent_log: list[dict] = []
        self.packed_history: dict[int, list[PackedSequence]] = {}
        self.decision_history: dict[int, list] = {}
        self.last_checkpoint_path: Path | None = None
        self.preview_cache: dict[int, str] = {}

    def _plan_candidates(self, step: int) -> tuple[list[PackedSequence], list[str], list[str], str]:
        tokens_so_far = self.planner.scheduled_tokens
        stage = self.plan.stage_at(tokens_so_far)
        self.planner._maybe_open_vault(stage)
        n_cand = self.cfg.global_batch * self.plan.opus_overprovision
        candidates: list[PackedSequence] = []
        lanes: list[str] = []
        docs: list[str] = []
        for slot in range(n_cand):
            lane = self.planner.pick_lane(stage, step, slot)
            examples = self.planner.take_for_sequence(lane, self.cfg.seq_len)
            policy = self.plan.packing_policy[lane]
            packed = pack(
                examples,
                policy,
                self.cfg.seq_len,
                self.tokenizer.pad_id,
                self.tokenizer.eos_id,
                lane,
            )
            if not packed:
                continue
            seq = packed[0]
            # Firewall: a never-train document in a loss-bearing sequence is fatal.
            ev = self.firewall.check_packed(seq.doc_ids, seq.useful_tokens > 0)
            if ev is not None:
                raise RuntimeError(f"eval firewall tripped on packed batch: {ev}")
            candidates.append(seq)
            lanes.append(lane)
            docs.extend(seq.doc_ids)
        return candidates, lanes, docs, stage.name

    def preview_next_batch(self) -> tuple[str, list[str]]:
        """Plan the next step without mutating durable state.

        Snapshot planner + OPUS, plan, restore. Preview and delivery share
        `_select_step`, so they cannot silently diverge.
        """
        from collections import Counter

        snap = self.planner.state_dict()
        opus_counts = Counter(self.opus.counts)
        opus_total = self.opus.total
        deferred = list(self.opus.deferred)
        opus_log = list(self.opus.log)
        sequences, _decisions, _stage = self._select_step(self.step, consume_state=False)
        batch_hash = sha256_obj([s.content_hash() for s in sequences])
        ids = [s.sample_id for s in sequences]
        self.planner.load_state_dict(snap)
        self.opus.counts = Counter(opus_counts)
        self.opus.total = opus_total
        self.opus.deferred = deferred
        self.opus.log = opus_log
        return batch_hash, ids

    def preview_next_batch_hash(self) -> str:
        return self.preview_next_batch()[0]

    def _select_step(
        self, step: int, consume_state: bool
    ) -> tuple[list[PackedSequence], list, str]:
        candidates, lanes, docs, stage_name = self._plan_candidates(step)
        stage = self.plan.stage_at(self.planner.scheduled_tokens)
        if consume_state:
            self.consumption.record_intent(
                self.cfg.run_id, self.cfg.branch_id, step, lanes, docs, stage_name
            )
            self.intent_log.append(
                {"step": step, "lanes": lanes, "docs": docs, "stage": stage_name}
            )
        floor_gaps = self.planner.floor_gaps()
        if self.cfg.opus_enabled:
            selected, decisions = self.opus.select(
                candidates, stage, step, floor_gaps, k=self.cfg.global_batch
            )
        else:
            selected = candidates[: self.cfg.global_batch]
            decisions = []
        if len(selected) < self.cfg.global_batch:
            # Fill from remaining candidates if OPUS returned short.
            have = {s.sample_id for s in selected}
            for seq in candidates:
                if seq.sample_id not in have:
                    selected.append(seq)
                if len(selected) >= self.cfg.global_batch:
                    break
        selected = selected[: self.cfg.global_batch]
        return selected, decisions, stage_name

    def step_once(self) -> dict:
        step = self.step
        sequences, decisions, stage_name = self._select_step(step, consume_state=True)
        tok, pos, attn, loss = pack_batch(sequences)
        mean, tok_loss, grad_norm = self.model.train_step(tok, pos, attn, loss)
        opus_ids = [d.candidate_id for d in decisions if d.status in {"accepted", "protected_override"}]
        while len(opus_ids) < len(sequences):
            opus_ids.append(None)
        opus_scores = []
        by_sample = {d.sample_id: d for d in decisions}
        for seq in sequences:
            d = by_sample.get(seq.sample_id)
            opus_scores.append(d.score if d else None)
            self.opus.observe(seq)
            self.planner.note_delivery(seq.lane, seq.useful_tokens, positions=len(seq.token_ids))

        ckpt_id = None
        events = self.consumption.record_delivery(
            self.cfg.run_id,
            self.cfg.branch_id,
            step,
            sequences,
            self.cfg.world_size,
            stage_name,
            self.tokenizer.tokenizer_hash,
            opus_ids,
            checkpoint_id=ckpt_id,
        )
        self.learning.record_step(
            step,
            self.cfg.branch_id,
            sequences,
            tok_loss.tolist(),
            grad_norm,
            mean,
            opus_scores,
            stage_name,
        )
        self.packed_history[step] = sequences
        self.decision_history[step] = decisions
        batch_hash = sha256_obj([s.content_hash() for s in sequences])
        self.preview_cache[step] = batch_hash
        self.step += 1
        if self.step % self.cfg.ckpt_every == 0:
            self.save_checkpoint()
        return {
            "step": step,
            "loss": mean,
            "grad_norm": grad_norm,
            "stage": stage_name,
            "batch_hash": batch_hash,
            "sample_ids": [s.sample_id for s in sequences],
            "useful": sum(s.useful_tokens for s in sequences),
            "pad": sum(s.pad_count for s in sequences),
            "lanes": [s.lane for s in sequences],
            "n_events": len(events),
            "n_opus": len(decisions),
        }

    def run_steps(self, n: int) -> list[dict]:
        stats = []
        for _ in range(n):
            if self.planner.scheduled_tokens >= self.plan.run_tokens:
                break
            stats.append(self.step_once())
        return stats

    def checkpoint_blob(self) -> dict:
        planner_state = self.planner.state_dict()
        meta = {
            "run_id": self.cfg.run_id,
            "branch_id": self.cfg.branch_id,
            "step": self.step,
            "ledger_offset": self.consumption.log.offset,
            "tokenizer_hash": self.tokenizer.tokenizer_hash,
            "plan_hash": self.plan.plan_hash,
            "world_size": self.cfg.world_size,
            "model_weight_hash": self.model.weights_hash(),
            "optimizer_hash": self.model.optimizer_hash(),
            "planner_state": planner_state,
            "opus_total": self.opus.total,
            "opus_counts": [[int(k), int(v)] for k, v in sorted(self.opus.counts.items())],
            "opus_proxy": self.opus.proxy_version,
            "cfg": {
                "steps": self.cfg.steps,
                "seq_len": self.cfg.seq_len,
                "global_batch": self.cfg.global_batch,
                "seed": self.cfg.seed,
                "opus_enabled": self.cfg.opus_enabled,
                "lr": self.cfg.lr,
                "d_model": self.cfg.d_model,
            },
        }
        meta["checkpoint_hash"] = sha256_obj(
            {
                "step": meta["step"],
                "branch": meta["branch_id"],
                "ledger_offset": meta["ledger_offset"],
                "tokenizer_hash": meta["tokenizer_hash"],
                "plan_hash": meta["plan_hash"],
                "model_weight_hash": meta["model_weight_hash"],
                "optimizer_hash": meta["optimizer_hash"],
                "planner_state": planner_state,
            }
        )
        return meta

    def save_checkpoint(self) -> Path:
        meta = self.checkpoint_blob()
        ckpt_dir = self.out_dir / "checkpoints"
        stem = f"step_{self.step:04d}_{self.cfg.branch_id}"
        meta_path = ckpt_dir / f"{stem}.json"
        npz_path = ckpt_dir / f"{stem}.npz"
        state = self.model.state_dict()
        np.savez_compressed(npz_path, **{k: v for k, v in state.items() if isinstance(v, np.ndarray)})
        extra = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
        meta["model_extra"] = extra
        meta["npz"] = str(npz_path.name)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.last_checkpoint_path = meta_path
        return meta_path

    @staticmethod
    def _verify_checkpoint(meta: dict, weights_hash: str, opt_hash: str) -> None:
        if meta["model_weight_hash"] != weights_hash:
            raise CheckpointCorruption("weight_hash_mismatch")
        if meta["optimizer_hash"] != opt_hash:
            raise CheckpointCorruption("optimizer_hash_mismatch")
        expect = sha256_obj(
            {
                "step": meta["step"],
                "branch": meta["branch_id"],
                "ledger_offset": meta["ledger_offset"],
                "tokenizer_hash": meta["tokenizer_hash"],
                "plan_hash": meta["plan_hash"],
                "model_weight_hash": meta["model_weight_hash"],
                "optimizer_hash": meta["optimizer_hash"],
                "planner_state": meta["planner_state"],
            }
        )
        if expect != meta["checkpoint_hash"]:
            raise CheckpointCorruption("checkpoint_hash_mismatch")

    @classmethod
    def restore(
        cls,
        ckpt_path: Path,
        plan: MixturePlan,
        planner: StreamPlanner,
        tokenizer: FrozenTokenizer,
        firewall: EvalFirewall,
        shards: list[Shard],
        out_dir: Path,
        world_size: int | None = None,
        branch_id: str | None = None,
        opus_enabled: bool | None = None,
        append_ledgers: bool = True,
    ) -> "Trainer":
        meta = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if meta["tokenizer_hash"] != tokenizer.tokenizer_hash:
            raise CheckpointCorruption("tokenizer_identity_swapped")
        if meta["plan_hash"] != plan.plan_hash:
            raise CheckpointCorruption("plan_hash_mismatch")
        cfg_d = meta["cfg"]
        cfg = TrainConfig(
            run_id=meta["run_id"],
            branch_id=branch_id or meta["branch_id"],
            steps=cfg_d["steps"],
            seq_len=cfg_d["seq_len"],
            global_batch=cfg_d["global_batch"],
            world_size=world_size if world_size is not None else meta["world_size"],
            ckpt_every=4,
            d_model=cfg_d["d_model"],
            lr=cfg_d["lr"],
            seed=cfg_d["seed"],
            opus_enabled=cfg_d["opus_enabled"] if opus_enabled is None else opus_enabled,
        )
        model = TinyLM(
            vocab_size=len(tokenizer),
            d_model=cfg.d_model,
            seq_len=cfg.seq_len,
            lr=cfg.lr,
            seed=cfg.seed,
        )
        npz = np.load(ckpt_path.parent / meta["npz"])
        loaded = {k: npz[k] for k in npz.files}
        loaded.update(meta["model_extra"])
        model.load_state_dict(loaded)
        cls._verify_checkpoint(meta, model.weights_hash(), model.optimizer_hash())
        planner.load_state_dict(meta["planner_state"])
        same_branch = (branch_id is None) or (branch_id == meta["branch_id"])
        trainer = cls(
            cfg,
            plan,
            planner,
            tokenizer,
            firewall,
            shards,
            out_dir,
            model=model,
            resume_ledgers=bool(append_ledgers and same_branch),
        )
        trainer.step = int(meta["step"])
        trainer.opus.total = int(meta.get("opus_total", 0))
        trainer.opus.counts.clear()
        for k, v in meta.get("opus_counts", []):
            trainer.opus.counts[int(k)] = int(v)
        trainer.last_checkpoint_path = ckpt_path
        return trainer

    def induce_corruption(self, kind: str, ckpt_path: Path) -> Path:
        """Write a corrupted copy of a checkpoint; restore() must refuse it."""
        meta = json.loads(ckpt_path.read_text(encoding="utf-8"))
        dest = ckpt_path.with_name(ckpt_path.stem + f"_corrupt_{kind}.json")
        npz_src = ckpt_path.parent / meta["npz"]
        npz_dest = dest.with_suffix(".npz")
        data = dict(np.load(npz_src))
        extra = dict(meta["model_extra"])
        if kind == "weight_bit_flip":
            data["tok_emb"] = data["tok_emb"].copy()
            data["tok_emb"][0, 0] += 1.0
        elif kind == "torn_write_truncated_weights":
            data["W"] = data["W"][: max(1, data["W"].shape[0] // 2)]
        elif kind == "loader_cursor_moved":
            meta["planner_state"] = dict(meta["planner_state"])
            cursors = dict(meta["planner_state"]["cursors"])
            lane = next(iter(cursors))
            cursors[lane] = cursors[lane] + 7
            meta["planner_state"]["cursors"] = cursors
        elif kind == "ledger_offset_moved":
            meta["ledger_offset"] = int(meta["ledger_offset"]) + 3
        elif kind == "step_counter_moved":
            meta["step"] = int(meta["step"]) + 1
        elif kind == "tokenizer_identity_swapped":
            meta["tokenizer_hash"] = "deadbeef" * 8
        elif kind == "optimizer_moment_bit_flip":
            data["v_tok"] = data["v_tok"].copy()
            data["v_tok"][0, 0] += 1.0
        elif kind == "optimizer_step_counter_moved":
            extra["opt_step"] = int(extra["opt_step"]) + 9
        elif kind == "optimizer_moments_dropped":
            data["v_tok"] = np.zeros_like(data["v_tok"])
            data["v_pos"] = np.zeros_like(data["v_pos"])
            data["v_W"] = np.zeros_like(data["v_W"])
            data["v_b"] = np.zeros_like(data["v_b"])
        else:
            raise ValueError(kind)
        np.savez_compressed(npz_dest, **data)
        meta["npz"] = npz_dest.name
        meta["model_extra"] = extra
        dest.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return dest
