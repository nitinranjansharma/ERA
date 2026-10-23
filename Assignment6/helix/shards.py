"""Immutable tokenized shards and manifests.

A shard is a sealed object. Modify it and it becomes a new shard with a new
hash and a new parent. The training loop consumes shards whose contents are
tied to one exact tokenizer version.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from helix.clean import CleanRecord
from helix.corpus import Document
from helix.firewall import EvalFirewall
from helix.hashutil import sha256_bytes, sha256_obj, sha256_text
from helix.tokenizer import FrozenTokenizer

HARD_FIELDS = (
    "shard_id",
    "tokenizer_hash",
    "content_hash",
    "cleaning_pipeline_hash",
    "license",
    "capability_lane",
    "dedup_status",
    "pii_status",
    "language_status",
    "contamination_status",
    "eval_overlap_status",
    "never_train",
)


@dataclass
class ShardManifest:
    shard_id: str
    source_ids: list[str]
    document_ids: list[str]
    tokenizer_hash: str
    token_count: int
    document_count: int
    language: str
    script: str
    capability_lane: str
    license: str
    provenance_tier: str
    cleaning_pipeline_hash: str
    dedup_status: str
    pii_status: str
    language_status: str
    contamination_status: str
    eval_overlap_status: str
    content_hash: str
    parent_shard_ids: list[str]
    never_train: bool
    reserved_for_anneal: bool
    admitted: bool
    admission_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shard:
    manifest: ShardManifest
    token_ids: list[list[int]]
    roles: list[list[str]]
    doc_ids: list[str]


def _worst(statuses: list[str], ok: str = "ok") -> str:
    bad = [s for s in statuses if s != ok and s != "unique" and s != "clean"]
    return bad[0] if bad else (statuses[0] if statuses else ok)


def seal_shards(
    docs: list[Document],
    clean_records: list[CleanRecord],
    tokenizer: FrozenTokenizer,
    firewall: EvalFirewall,
    docs_per_shard: int = 3,
) -> tuple[list[Shard], list[dict]]:
    """Group admitted documents by lane into sealed shards. Blocked documents
    still produce a rejected-shard record so the exclusion is auditable.
    """
    by_id = {d.doc_id: d for d in docs}
    rec_by_id = {r.doc_id: r for r in clean_records}

    shards: list[Shard] = []
    rejections: list[dict] = []
    buckets: dict[tuple[str, bool, bool], list[str]] = {}

    for rec in clean_records:
        doc = by_id[rec.doc_id]
        fw = firewall.check_document(doc)
        if fw is not None:
            rec.admitted = False
            rec.reasons.append(f"firewall:{fw.gate}")

        if not rec.admitted:
            rejections.append(
                {
                    "doc_id": rec.doc_id,
                    "lane": doc.lane,
                    "reasons": rec.reasons,
                    "plant": doc.plant,
                    "never_train": doc.never_train,
                }
            )
            continue

        key = (doc.lane, doc.reserved_for_anneal, doc.never_train)
        buckets.setdefault(key, []).append(doc.doc_id)

    shard_counter = 0
    for (lane, reserved, never_train), doc_ids in sorted(buckets.items()):
        for i in range(0, len(doc_ids), docs_per_shard):
            group = doc_ids[i : i + docs_per_shard]
            token_ids = [rec_by_id[d].token_ids for d in group]
            roles = [rec_by_id[d].roles for d in group]
            token_bytes = b"".join(
                np.asarray(seq, dtype=np.int32).tobytes() for seq in token_ids
            )
            content_hash = sha256_bytes(token_bytes)
            docs_g = [by_id[d] for d in group]
            recs_g = [rec_by_id[d] for d in group]
            shard_id = f"{lane[:4]}_{'anneal' if reserved else 'train'}_{shard_counter:03d}"
            shard_counter += 1
            manifest = ShardManifest(
                shard_id=shard_id,
                source_ids=sorted({d.source_id for d in docs_g}),
                document_ids=list(group),
                tokenizer_hash=tokenizer.tokenizer_hash,
                token_count=sum(len(s) for s in token_ids),
                document_count=len(group),
                language=docs_g[0].language,
                script=docs_g[0].script,
                capability_lane=lane,
                license=sorted({d.license for d in docs_g})[0],
                provenance_tier=min(d.provenance_tier for d in docs_g),
                cleaning_pipeline_hash=recs_g[0].cleaning_pipeline_hash,
                dedup_status=_worst([r.dedup_status for r in recs_g], "unique"),
                pii_status=_worst([r.pii_status for r in recs_g], "clean"),
                language_status=_worst([r.language_status for r in recs_g], "ok"),
                contamination_status="clean",
                eval_overlap_status="none",
                content_hash=content_hash,
                parent_shard_ids=[],
                never_train=never_train,
                reserved_for_anneal=reserved,
                admitted=True,
                admission_reasons=[],
            )
            # Hard-field presence is part of admission.
            missing = [f for f in HARD_FIELDS if getattr(manifest, f) in (None, "")]
            if missing:
                manifest.admitted = False
                manifest.admission_reasons.append("missing:" + ",".join(missing))
            if manifest.tokenizer_hash != tokenizer.tokenizer_hash:
                manifest.admitted = False
                manifest.admission_reasons.append("tokenizer_mismatch")
            shards.append(Shard(manifest=manifest, token_ids=token_ids, roles=roles, doc_ids=group))

    return shards, rejections


def write_shards(shards: list[Shard], out_dir: Path) -> None:
    man_dir = out_dir / "manifests"
    shard_dir = out_dir / "shards"
    man_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        (man_dir / f"{shard.manifest.shard_id}.json").write_text(
            json.dumps(shard.manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        np.savez_compressed(
            shard_dir / f"{shard.manifest.shard_id}.npz",
            **{doc_id: np.asarray(ids, dtype=np.int32) for doc_id, ids in zip(shard.doc_ids, shard.token_ids)},
        )


def verify_shard_integrity(shard: Shard, tokenizer: FrozenTokenizer) -> list[str]:
    errors = []
    m = shard.manifest
    token_bytes = b"".join(np.asarray(seq, dtype=np.int32).tobytes() for seq in shard.token_ids)
    if sha256_bytes(token_bytes) != m.content_hash:
        errors.append("content_hash_mismatch")
    if m.tokenizer_hash != tokenizer.tokenizer_hash:
        errors.append("tokenizer_hash_mismatch")
    if m.token_count != sum(len(s) for s in shard.token_ids):
        errors.append("token_count_mismatch")
    for field in HARD_FIELDS:
        if getattr(m, field) in (None, ""):
            errors.append(f"missing_{field}")
    return errors
