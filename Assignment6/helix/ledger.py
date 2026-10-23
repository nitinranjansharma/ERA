"""Hash-chained consumption ledger and two-way learning ledger.

experiment = model checkpoint + optimizer state + data stream + code/config

The consumption ledger is the run's memory: every optimizer step records the
packed sample ids, shard ids, token spans, loss-mask hash, mixture lane,
curriculum stage and OPUS decision. Resume binds to a ledger offset. Replay
re-feeds the historical stream. Fork records the exact point of divergence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from helix.hashutil import GENESIS, hash_chain_link, sha256_obj
from helix.packing import PackedSequence, SpanRef


@dataclass
class ConsumptionEvent:
    seq: int
    event_hash: str
    prev_hash: str
    run_id: str
    branch_id: str
    global_step: int
    checkpoint_id: str | None
    rank: int
    microbatch_id: int
    packed_sample_ids: list[str]
    shard_ids: list[str]
    token_spans: list[dict]
    loss_mask_hash: str
    attention_policy: str
    position_policy: str
    mixture_lane: str
    curriculum_stage: str
    tokenizer_version: str
    dataloader_version: str
    opus_decision_id: str | None
    batch_hash: str
    useful_tokens: int
    pad_tokens: int
    kind: str = "delivery"


@dataclass
class IntentEvent:
    seq: int
    event_hash: str
    prev_hash: str
    run_id: str
    branch_id: str
    global_step: int
    planned_lanes: list[str]
    planned_doc_ids: list[str]
    stage: str
    kind: str = "intent"


@dataclass
class TokenObs:
    global_step: int
    branch_id: str
    sample_id: str
    token_id: int
    position: int
    doc_id: str
    shard_id: str
    lane: str
    role: str
    loss: float
    ppl: float
    special: bool
    boundary: bool
    opus_decision_id: str | None
    stage: str


@dataclass
class LearningRecord:
    global_step: int
    branch_id: str
    sample_id: str
    shard_ids: list[str]
    lane: str
    stage: str
    mean_loss: float
    mean_ppl: float
    grad_norm: float
    useful_tokens: int
    opus_score: float | None
    classification: str


class HashLog:
    def __init__(self, path: Path, resume: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prev = GENESIS
        self.seq = 0
        self.rows: list[dict] = []
        if resume and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows.append(row)
            if self.rows:
                self.seq = self.rows[-1]["seq"] + 1
                self.prev = self.rows[-1]["event_hash"]
        else:
            path.write_text("", encoding="utf-8")

    def append(self, payload: dict) -> tuple[int, str, str]:
        event_hash, prev = hash_chain_link(self.prev, payload)
        row = {"seq": self.seq, "event_hash": event_hash, "prev_hash": prev, **payload}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows.append(row)
        self.seq += 1
        self.prev = event_hash
        return row["seq"] - 0, event_hash, prev  # seq of this row

    @property
    def offset(self) -> int:
        return self.seq

    def verify_chain(self) -> bool:
        prev = GENESIS
        for row in self.rows:
            payload = {k: v for k, v in row.items() if k not in {"seq", "event_hash", "prev_hash"}}
            expect, _ = hash_chain_link(prev, payload)
            if row["prev_hash"] != prev or row["event_hash"] != expect:
                return False
            prev = row["event_hash"]
        return True


class ConsumptionLedger:
    def __init__(self, path: Path, resume: bool = False) -> None:
        self.log = HashLog(path, resume=resume)

    def record_delivery(
        self,
        run_id: str,
        branch_id: str,
        step: int,
        sequences: list[PackedSequence],
        world_size: int,
        stage: str,
        tokenizer_hash: str,
        opus_ids: list[str | None],
        checkpoint_id: str | None = None,
    ) -> list[ConsumptionEvent]:
        events = []
        for i, seq in enumerate(sequences):
            rank = i % max(1, world_size)
            payload = {
                "kind": "delivery",
                "run_id": run_id,
                "branch_id": branch_id,
                "global_step": step,
                "checkpoint_id": checkpoint_id,
                "rank": rank,
                "microbatch_id": i,
                "packed_sample_ids": [seq.sample_id],
                "shard_ids": seq.shard_ids,
                "token_spans": [
                    {
                        "shard_id": s.shard_id,
                        "doc_id": s.doc_id,
                        "doc_start": s.doc_start,
                        "doc_end": s.doc_end,
                        "packed_start": s.packed_start,
                        "packed_end": s.packed_end,
                    }
                    for s in seq.spans
                ],
                "loss_mask_hash": seq.loss_mask_hash(),
                "attention_policy": seq.policy,
                "position_policy": "reset" if seq.policy == "structure_preserving" else "contiguous",
                "mixture_lane": seq.lane,
                "curriculum_stage": stage,
                "tokenizer_version": tokenizer_hash,
                "dataloader_version": "helix-v1",
                "opus_decision_id": opus_ids[i] if i < len(opus_ids) else None,
                "batch_hash": seq.content_hash(),
                "useful_tokens": seq.useful_tokens,
                "pad_tokens": seq.pad_count,
            }
            seqno, eh, ph = self.log.append(payload)
            events.append(
                ConsumptionEvent(seq=seqno, event_hash=eh, prev_hash=ph, **payload)
            )
        return events

    def record_intent(
        self,
        run_id: str,
        branch_id: str,
        step: int,
        lanes: list[str],
        doc_ids: list[str],
        stage: str,
    ) -> None:
        self.log.append(
            {
                "kind": "intent",
                "run_id": run_id,
                "branch_id": branch_id,
                "global_step": step,
                "planned_lanes": lanes,
                "planned_doc_ids": doc_ids,
                "curriculum_stage": stage,
            }
        )

    def deliveries(self) -> list[dict]:
        return [r for r in self.log.rows if r.get("kind") == "delivery"]

    def batch_hash_at(self, step: int, branch_id: str) -> str:
        rows = [
            r
            for r in self.deliveries()
            if r["global_step"] == step and r["branch_id"] == branch_id
        ]
        return sha256_obj([r["batch_hash"] for r in rows])

    def sample_ids_at(self, step: int, branch_id: str) -> list[str]:
        ids = []
        for r in self.deliveries():
            if r["global_step"] == step and r["branch_id"] == branch_id:
                ids.extend(r["packed_sample_ids"])
        return ids


class LearningLedger:
    def __init__(self, path: Path, token_path: Path, resume: bool = False) -> None:
        self.log = HashLog(path, resume=resume)
        self.token_path = token_path
        if not resume:
            self.token_path.write_text("", encoding="utf-8")
            self.token_count = 0
        else:
            self.token_count = sum(
                1 for line in token_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ) if token_path.exists() else 0

    def record_step(
        self,
        step: int,
        branch_id: str,
        sequences: list[PackedSequence],
        tok_loss: list[list[float]],
        grad_norm: float,
        mean_loss: float,
        opus_scores: list[float | None],
        stage: str,
    ) -> None:
        import math

        for i, seq in enumerate(sequences):
            losses = [float(tok_loss[i][t]) for t, keep in enumerate(seq.loss_mask) if keep]
            mean = sum(losses) / max(1, len(losses))
            ppl = math.exp(min(mean, 20.0))
            prev = None
            classification = "neutral"
            if mean < 4.5:
                classification = "useful"
            if mean > 8.0:
                classification = "hard"
            payload = {
                "kind": "learn",
                "global_step": step,
                "branch_id": branch_id,
                "sample_id": seq.sample_id,
                "shard_ids": seq.shard_ids,
                "lane": seq.lane,
                "stage": stage,
                "mean_loss": mean,
                "mean_ppl": ppl,
                "grad_norm": grad_norm,
                "useful_tokens": seq.useful_tokens,
                "opus_score": opus_scores[i] if i < len(opus_scores) else None,
                "classification": classification,
            }
            self.log.append(payload)
            for t, keep in enumerate(seq.loss_mask):
                if not keep:
                    continue
                span = _span_at(seq.spans, t)
                rec = {
                    "global_step": step,
                    "branch_id": branch_id,
                    "sample_id": seq.sample_id,
                    "token_id": seq.token_ids[t],
                    "position": t,
                    "doc_id": span.doc_id if span else "",
                    "shard_id": span.shard_id if span else "",
                    "lane": seq.lane,
                    "role": seq.role_ids[t],
                    "loss": float(tok_loss[i][t]),
                    "ppl": float(math.exp(min(float(tok_loss[i][t]), 20.0))),
                    "special": seq.role_ids[t] not in {"text", "pad"},
                    "boundary": seq.role_ids[t] == "boundary",
                    "stage": stage,
                }
                with self.token_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.token_count += 1


def _span_at(spans: list[SpanRef], pos: int) -> SpanRef | None:
    for s in spans:
        if s.packed_start <= pos < s.packed_end:
            return s
    return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
