"""OPUS top-k selector. Decisions: accept, reject, defer, protected_override."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from helix.hashutil import sha256_obj
from helix.packing import PackedSequence
from helix.schedule import MixturePlan, Stage


@dataclass
class OpusDecision:
    candidate_id: str
    sample_id: str
    lane: str
    stage: str
    score: float
    status: str
    reason: str
    protected_floor_override: bool
    effective_tokens: int
    shard_ids: list[str]
    proxy_version: str


class Opus:
    def __init__(self, plan: MixturePlan) -> None:
        self.plan = plan
        self.counts: Counter[int] = Counter()
        self.total = 0
        self.proxy_version = "unigram-surprise-v1"
        self.log: list[OpusDecision] = []
        self.deferred: list[PackedSequence] = []

    def observe(self, seq: PackedSequence) -> None:
        for tid, keep in zip(seq.token_ids, seq.loss_mask):
            if keep:
                self.counts[int(tid)] += 1
                self.total += 1

    def score(self, seq: PackedSequence) -> float:
        denom = max(1, self.total)
        surprises = []
        for tid, keep in zip(seq.token_ids, seq.loss_mask):
            if not keep:
                continue
            p = (self.counts.get(int(tid), 0) + 1) / (denom + 2)
            surprises.append(-_log(p))
        if not surprises:
            return 0.0
        mean_s = sum(surprises) / len(surprises)
        scarcity = 0.15 if seq.lane in self.plan.protected_floors else 0.0
        return mean_s + scarcity

    def select(
        self,
        candidates: list[PackedSequence],
        stage: Stage,
        step: int,
        floor_gaps: dict[str, dict],
        k: int | None = None,
    ) -> tuple[list[PackedSequence], list[OpusDecision]]:
        k = k or self.plan.opus_accept_k
        scored = [(self.score(seq), i, seq) for i, seq in enumerate(candidates)]
        # Deterministic order: score desc, then sample_id.
        scored.sort(key=lambda t: (-t[0], t[2].sample_id, t[1]))

        accepted: list[PackedSequence] = []
        decisions: list[OpusDecision] = []
        taken: set[int] = set()
        taken_ids: set[str] = set()

        def _take(seq: PackedSequence, i: int, score: float, status: str, reason: str, override: bool) -> bool:
            if i in taken or seq.sample_id in taken_ids or len(accepted) >= k:
                return False
            accepted.append(seq)
            taken.add(i)
            taken_ids.add(seq.sample_id)
            d = self._decision(seq, score, status, reason, override, stage)
            decisions.append(d)
            self.log.append(d)
            return True

        # Protected-floor rescue happens *before* top-k so a scarce lane cannot
        # be starved by a high-scoring web candidate.
        starving = [lane for lane, g in floor_gaps.items() if not g["ok"]]
        for lane in starving:
            for score, i, seq in scored:
                if seq.lane == lane and _take(seq, i, score, "protected_override", "floor_deficit", True):
                    break

        for score, i, seq in scored:
            if len(accepted) >= k:
                break
            _take(seq, i, score, "accepted", "top_k", False)

        leftovers = [(s, i, seq) for s, i, seq in scored if i not in taken and seq.sample_id not in taken_ids]
        split = max(1, len(leftovers) // 2) if leftovers else 0
        seen_left: set[str] = set()
        for n, (score, i, seq) in enumerate(leftovers):
            if seq.sample_id in seen_left:
                continue
            seen_left.add(seq.sample_id)
            if n < split:
                status, reason = "deferred", "mid_score"
                self.deferred.append(seq)
            else:
                status, reason = "rejected", "low_proxy_utility"
            d = self._decision(seq, score, status, reason, False, stage)
            decisions.append(d)
            self.log.append(d)
        return accepted[:k], decisions

    def _decision(
        self,
        seq: PackedSequence,
        score: float,
        status: str,
        reason: str,
        override: bool,
        stage: Stage,
    ) -> OpusDecision:
        return OpusDecision(
            candidate_id=sha256_obj({"sample": seq.sample_id, "status": status, "score": round(score, 6)})[:16],
            sample_id=seq.sample_id,
            lane=seq.lane,
            stage=stage.name,
            score=float(score),
            status=status,
            reason=reason,
            protected_floor_override=override,
            effective_tokens=seq.useful_tokens,
            shard_ids=list(seq.shard_ids),
            proxy_version=self.proxy_version,
        )


def _log(p: float) -> float:
    import math

    return math.log(max(p, 1e-12))
