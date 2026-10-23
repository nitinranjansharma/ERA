"""Build tokenizer, clean docs, seal shards, compile the mixture plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from helix.clean import CleanRecord, clean_documents
from helix.corpus import Document, all_plain_text, build_corpus
from helix.firewall import EvalFirewall, build_firewall
from helix.schedule import MixturePlan, compile_plan, examples_from_shards
from helix.shards import Shard, seal_shards, verify_shard_integrity, write_shards
from helix.tokenizer import FrozenTokenizer


@dataclass
class BuiltWorld:
    docs: list[Document]
    tokenizer: FrozenTokenizer
    clean: list[CleanRecord]
    firewall: EvalFirewall
    shards: list[Shard]
    rejections: list[dict]
    plan: MixturePlan
    admitted_shards: list[Shard]


def build_world(plan_raw: dict, out_dir: Path | None = None) -> BuiltWorld:
    docs = build_corpus()
    tokenizer = FrozenTokenizer.from_texts(all_plain_text(docs))
    firewall = build_firewall(docs, tokenizer)
    clean = clean_documents(docs, tokenizer)
    shards, rejections = seal_shards(docs, clean, tokenizer, firewall)
    for shard in shards:
        errs = verify_shard_integrity(shard, tokenizer)
        if errs:
            raise RuntimeError(f"shard {shard.manifest.shard_id} failed integrity: {errs}")
    if out_dir is not None:
        write_shards(shards, out_dir)
    plan = compile_plan(plan_raw)
    admitted = [s for s in shards if s.manifest.admitted and not s.manifest.never_train]
    return BuiltWorld(
        docs=docs,
        tokenizer=tokenizer,
        clean=clean,
        firewall=firewall,
        shards=shards,
        rejections=rejections,
        plan=plan,
        admitted_shards=admitted,
    )
