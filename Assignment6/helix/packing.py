"""Six packing policies plus loss / attention / position masks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helix.constants import (
    PACKING_POLICIES,
    ROLE_BOUNDARY,
    ROLE_PAD,
    SUPERVISED_ROLES,
)
from helix.hashutil import sha256_ints, sha256_obj


@dataclass
class Example:
    doc_id: str
    shard_id: str
    lane: str
    token_ids: list[int]
    roles: list[str]


@dataclass
class SpanRef:
    shard_id: str
    doc_id: str
    doc_start: int
    doc_end: int
    packed_start: int
    packed_end: int


@dataclass
class PackedSequence:
    sample_id: str
    token_ids: list[int]
    loss_mask: list[int]
    position_ids: list[int]
    segment_ids: list[int]
    role_ids: list[str]
    attn_mask: list[list[int]]
    spans: list[SpanRef]
    lane: str
    policy: str
    pad_count: int
    useful_tokens: int
    truncated_docs: int
    split_docs: int
    shard_ids: list[str]
    doc_ids: list[str]

    def content_hash(self) -> str:
        return sha256_obj(
            {
                "tok": self.token_ids,
                "loss": self.loss_mask,
                "pos": self.position_ids,
                "seg": self.segment_ids,
                "attn_rows": [sum(row) for row in self.attn_mask],
                "policy": self.policy,
                "lane": self.lane,
            }
        )

    def loss_mask_hash(self) -> str:
        return sha256_ints(self.loss_mask)

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "token_ids": np.asarray(self.token_ids, dtype=np.int32),
            "loss_mask": np.asarray(self.loss_mask, dtype=np.float64),
            "position_ids": np.asarray(self.position_ids, dtype=np.int32),
            "segment_ids": np.asarray(self.segment_ids, dtype=np.int32),
            "attn_mask": np.asarray(self.attn_mask, dtype=np.bool_),
        }


def _finish(
    tokens: list[int],
    roles: list[str],
    segments: list[int],
    positions: list[int],
    spans: list[SpanRef],
    lane: str,
    policy: str,
    seq_len: int,
    pad_id: int,
    truncated: int,
    split: int,
    isolate: bool,
) -> PackedSequence:
    assert len(tokens) <= seq_len
    pad = seq_len - len(tokens)
    tokens = tokens + [pad_id] * pad
    roles = roles + [ROLE_PAD] * pad
    segments = segments + [-1] * pad
    positions = positions + [0] * pad
    loss = [
        1 if (r in SUPERVISED_ROLES and t != pad_id and i > 0) else 0
        for i, (r, t) in enumerate(zip(roles, tokens))
    ]
    # First token of a new segment has no previous context inside the segment
    # when isolated; keep the loss off there too.
    if isolate:
        seen = set()
        for i, seg in enumerate(segments):
            if seg < 0:
                continue
            if seg not in seen:
                loss[i] = 0
                seen.add(seg)
    attn = _attention_mask(segments, seq_len, isolate=isolate)
    useful = int(sum(loss))
    shard_ids = sorted({s.shard_id for s in spans})
    doc_ids = []
    for s in spans:
        if s.doc_id not in doc_ids:
            doc_ids.append(s.doc_id)
    sample_id = sha256_obj({"tok": tokens, "loss": loss, "seg": segments, "pos": positions})[:16]
    return PackedSequence(
        sample_id=sample_id,
        token_ids=tokens,
        loss_mask=loss,
        position_ids=positions,
        segment_ids=segments,
        role_ids=roles,
        attn_mask=attn,
        spans=spans,
        lane=lane,
        policy=policy,
        pad_count=pad,
        useful_tokens=useful,
        truncated_docs=truncated,
        split_docs=split,
        shard_ids=shard_ids,
        doc_ids=doc_ids,
    )


def _attention_mask(segments: list[int], seq_len: int, isolate: bool) -> list[list[int]]:
    mask = [[0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        if segments[i] < 0:
            continue
        for j in range(i + 1):  # causal, includes self
            if segments[j] < 0:
                continue
            if isolate and segments[i] != segments[j]:
                continue
            mask[i][j] = 1
    return mask


def _as_pieces(examples: list[Example], eos_id: int, keep_eos: bool) -> list[dict]:
    pieces = []
    for ex in examples:
        ids = list(ex.token_ids)
        roles = list(ex.roles)
        if keep_eos:
            ids = ids + [eos_id]
            roles = roles + [ROLE_BOUNDARY]
        pieces.append({"ex": ex, "ids": ids, "roles": roles})
    return pieces


def pack_pad_only(examples: list[Example], seq_len: int, pad_id: int, eos_id: int, lane: str) -> list[PackedSequence]:
    out = []
    for ex in examples:
        ids = list(ex.token_ids)
        roles = list(ex.roles)
        truncated = 0
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            roles = roles[:seq_len]
            truncated = 1
        spans = [SpanRef(ex.shard_id, ex.doc_id, 0, len(ids), 0, len(ids))]
        out.append(
            _finish(
                ids, roles, [0] * len(ids), list(range(len(ids))),
                spans, lane, "pad_only", seq_len, pad_id, truncated, 0, isolate=True,
            )
        )
    return out


def pack_concat_chop(examples: list[Example], seq_len: int, pad_id: int, eos_id: int, lane: str) -> list[PackedSequence]:
    stream_ids: list[int] = []
    stream_roles: list[str] = []
    stream_meta: list[tuple[str, str, int]] = []  # shard, doc, local
    truncated = 0
    split_docs: set[str] = set()
    for ex in examples:
        for i, (tid, role) in enumerate(zip(ex.token_ids, ex.roles)):
            stream_ids.append(tid)
            stream_roles.append(role)
            stream_meta.append((ex.shard_id, ex.doc_id, i))
        stream_ids.append(eos_id)
        stream_roles.append(ROLE_BOUNDARY)
        stream_meta.append((ex.shard_id, ex.doc_id, -1))
    out = []
    for start in range(0, len(stream_ids), seq_len):
        end = min(start + seq_len, len(stream_ids))
        ids = stream_ids[start:end]
        roles = stream_roles[start:end]
        meta = stream_meta[start:end]
        # A document that continues across a window boundary is split; a window
        # that ends mid-document also truncated the remainder into the next window.
        docs_here = [m[1] for m in meta if m[2] >= 0]
        if start > 0 and docs_here:
            split_docs.add(docs_here[0])
        if end < len(stream_ids) and docs_here:
            split_docs.add(docs_here[-1])
            truncated += 1
        segments = list(range(len(ids)))  # concat_chop: one stream, causal only
        # Use a single segment so attention may cross EOS (pretraining).
        segments = [0] * len(ids)
        positions = list(range(len(ids)))
        spans: list[SpanRef] = []
        i = 0
        while i < len(meta):
            shard, doc, local = meta[i]
            j = i
            while j < len(meta) and meta[j][1] == doc:
                j += 1
            locals_ = [meta[k][2] for k in range(i, j) if meta[k][2] >= 0]
            if locals_:
                spans.append(SpanRef(shard, doc, locals_[0], locals_[-1] + 1, i, j))
            i = j
        out.append(
            _finish(
                ids, roles, segments, positions, spans, lane, "concat_chop",
                seq_len, pad_id, truncated, len(split_docs), isolate=False,
            )
        )
        truncated = 0
    return out


def _place_into_bins(
    pieces: list[dict],
    seq_len: int,
    policy: str,
) -> list[list[dict]]:
    bins: list[list[dict]] = []
    remain: list[int] = []
    ordered = pieces
    if policy == "best_fit":
        ordered = sorted(pieces, key=lambda p: len(p["ids"]), reverse=True)
    for piece in ordered:
        L = len(piece["ids"])
        if L > seq_len:
            piece = {
                "ex": piece["ex"],
                "ids": piece["ids"][:seq_len],
                "roles": piece["roles"][:seq_len],
                "truncated": True,
            }
            L = seq_len
        else:
            piece = dict(piece)
            piece["truncated"] = False
        choice = -1
        if policy == "greedy":
            for i, r in enumerate(remain):
                if L <= r:
                    choice = i
                    break
        elif policy == "best_fit":
            best = seq_len + 1
            for i, r in enumerate(remain):
                if L <= r < best:
                    best = r
                    choice = i
        else:
            raise ValueError(policy)
        if choice < 0:
            bins.append([piece])
            remain.append(seq_len - L)
        else:
            bins[choice].append(piece)
            remain[choice] -= L
    return bins


def _bins_to_sequences(
    bins: list[list[dict]],
    seq_len: int,
    pad_id: int,
    lane: str,
    policy: str,
    isolate: bool,
    reset_positions: bool,
) -> list[PackedSequence]:
    out = []
    for bin_pieces in bins:
        ids: list[int] = []
        roles: list[str] = []
        segs: list[int] = []
        pos: list[int] = []
        spans: list[SpanRef] = []
        truncated = 0
        for seg_i, piece in enumerate(bin_pieces):
            start = len(ids)
            ids.extend(piece["ids"])
            roles.extend(piece["roles"])
            segs.extend([seg_i] * len(piece["ids"]))
            if reset_positions:
                pos.extend(range(len(piece["ids"])))
            else:
                pos.extend(range(start, start + len(piece["ids"])))
            ex: Example = piece["ex"]
            spans.append(SpanRef(ex.shard_id, ex.doc_id, 0, len(piece["ids"]), start, start + len(piece["ids"])))
            if piece.get("truncated"):
                truncated += 1
        out.append(
            _finish(
                ids, roles, segs, pos, spans, lane, policy, seq_len, pad_id,
                truncated, 0, isolate=isolate,
            )
        )
    return out


def pack_greedy(examples: list[Example], seq_len: int, pad_id: int, eos_id: int, lane: str) -> list[PackedSequence]:
    pieces = _as_pieces(examples, eos_id, keep_eos=True)
    bins = _place_into_bins(pieces, seq_len, "greedy")
    return _bins_to_sequences(bins, seq_len, pad_id, lane, "greedy", isolate=False, reset_positions=False)


def pack_best_fit(examples: list[Example], seq_len: int, pad_id: int, eos_id: int, lane: str) -> list[PackedSequence]:
    pieces = _as_pieces(examples, eos_id, keep_eos=True)
    bins = _place_into_bins(pieces, seq_len, "best_fit")
    return _bins_to_sequences(bins, seq_len, pad_id, lane, "best_fit", isolate=False, reset_positions=False)


def pack_structure_preserving(
    examples: list[Example], seq_len: int, pad_id: int, eos_id: int, lane: str
) -> list[PackedSequence]:
    # Do not split structured examples. Truncate only if a single example
    # exceeds the window (recorded, not silent). Isolate attention.
    pieces = _as_pieces(examples, eos_id, keep_eos=True)
    bins = _place_into_bins(pieces, seq_len, "greedy")
    return _bins_to_sequences(
        bins, seq_len, pad_id, lane, "structure_preserving", isolate=True, reset_positions=True
    )


def pack_long_context(
    examples: list[Example], seq_len: int, pad_id: int, eos_id: int, lane: str
) -> list[PackedSequence]:
    """One long document per window. Short leftovers are not mixed in."""
    threshold = int(seq_len * 0.55)
    long_ex = [ex for ex in examples if len(ex.token_ids) >= threshold]
    if not long_ex:
        long_ex = examples[:1]
    return pack_pad_only(long_ex, seq_len, pad_id, eos_id, lane)


POLICY_FNS = {
    "pad_only": pack_pad_only,
    "concat_chop": pack_concat_chop,
    "greedy": pack_greedy,
    "best_fit": pack_best_fit,
    "structure_preserving": pack_structure_preserving,
    "long_context": pack_long_context,
}


def pack(
    examples: list[Example],
    policy: str,
    seq_len: int,
    pad_id: int,
    eos_id: int,
    lane: str,
) -> list[PackedSequence]:
    if policy not in POLICY_FNS:
        raise ValueError(f"unknown packing policy {policy}; known={PACKING_POLICIES}")
    if not examples:
        return []
    return POLICY_FNS[policy](examples, seq_len, pad_id, eos_id, lane)


def attention_crossings(seq: PackedSequence) -> int:
    """Count attention-allowed pairs that cross a document segment."""
    n = 0
    segs = seq.segment_ids
    for i, row in enumerate(seq.attn_mask):
        for j, allowed in enumerate(row):
            if allowed and segs[i] >= 0 and segs[j] >= 0 and segs[i] != segs[j]:
                n += 1
    return n


def blocked_cross_segment_pairs(seq: PackedSequence) -> int:
    n = 0
    segs = seq.segment_ids
    t = len(segs)
    for i in range(t):
        for j in range(i + 1):
            if segs[i] >= 0 and segs[j] >= 0 and segs[i] != segs[j]:
                if seq.attn_mask[i][j] == 0:
                    n += 1
    return n


def utilization(sequences: list[PackedSequence]) -> dict:
    if not sequences:
        return {"sequences": 0, "positions": 0, "useful": 0, "pad": 0, "utilization": 0.0}
    positions = sum(len(s.token_ids) for s in sequences)
    useful = sum(s.useful_tokens for s in sequences)
    pad = sum(s.pad_count for s in sequences)
    nonpad = positions - pad
    return {
        "sequences": len(sequences),
        "positions": positions,
        "useful": useful,
        "pad": pad,
        "nonpad": nonpad,
        "utilization": nonpad / positions if positions else 0.0,
        "useful_fraction": useful / positions if positions else 0.0,
        "truncated_docs": sum(s.truncated_docs for s in sequences),
        "split_docs": sum(s.split_docs for s in sequences),
    }
