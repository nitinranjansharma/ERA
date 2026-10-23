"""Eval/val firewall. Test shards are registered and blocked from training loss."""

from __future__ import annotations

from dataclasses import dataclass, field

from helix.constants import CANARY_STRING, EVAL_ITEM, NGRAM_CONTAINMENT_N
from helix.corpus import Document
from helix.hashutil import sha256_text
from helix.tokenizer import FrozenTokenizer, graphemes, nfc


@dataclass
class RegistryEntry:
    doc_id: str
    permission: str  # train | validate | test
    content_hash: str
    fingerprint: str
    ngrams: set[tuple[str, ...]]
    canaries: list[str]
    never_train: bool
    benchmark_id: str | None = None


@dataclass
class FirewallEvent:
    doc_id: str
    action: str
    gate: str
    detail: str


@dataclass
class EvalFirewall:
    entries: dict[str, RegistryEntry] = field(default_factory=dict)
    hash_index: dict[str, str] = field(default_factory=dict)
    events: list[FirewallEvent] = field(default_factory=list)

    def register(self, doc: Document, tokenizer: FrozenTokenizer, permission: str) -> RegistryEntry:
        text = nfc(doc.text)
        grams = _ngrams(text, NGRAM_CONTAINMENT_N)
        canaries = [CANARY_STRING] if (permission != "train" or CANARY_STRING in text) else []
        if CANARY_STRING in text:
            canaries = [CANARY_STRING]
        entry = RegistryEntry(
            doc_id=doc.doc_id,
            permission=permission,
            content_hash=sha256_text(text),
            fingerprint=sha256_text(text + "|fp|" + permission),
            ngrams=grams,
            canaries=canaries,
            never_train=permission != "train" or doc.never_train,
            benchmark_id="gsm8k-plant" if doc.plant == "eval_item" else None,
        )
        self.entries[doc.doc_id] = entry
        self.hash_index[entry.content_hash] = doc.doc_id
        return entry

    def check_document(self, doc: Document) -> FirewallEvent | None:
        """Return a blocking event if this document must not enter training."""
        text = nfc(doc.text)
        content_hash = sha256_text(text)

        entry = self.entries.get(doc.doc_id)
        if entry and entry.never_train:
            ev = FirewallEvent(doc.doc_id, "blocked", "never_train", entry.permission)
            self.events.append(ev)
            return ev

        if content_hash in self.hash_index:
            owner = self.hash_index[content_hash]
            owned = self.entries[owner]
            if owned.never_train:
                ev = FirewallEvent(doc.doc_id, "blocked", "content_hash", f"matches:{owner}")
                self.events.append(ev)
                return ev

        if CANARY_STRING in text:
            ev = FirewallEvent(doc.doc_id, "blocked", "canary", CANARY_STRING)
            self.events.append(ev)
            return ev

        doc_grams = _ngrams(text, NGRAM_CONTAINMENT_N)
        for other in self.entries.values():
            if not other.never_train:
                continue
            if other.ngrams and doc_grams & other.ngrams:
                overlap = next(iter(doc_grams & other.ngrams))
                ev = FirewallEvent(
                    doc.doc_id,
                    "blocked",
                    "ngram_containment",
                    f"overlap_with:{other.doc_id}:{' '.join(overlap[:4])}...",
                )
                self.events.append(ev)
                return ev

        return None

    def check_packed(self, doc_ids: list[str], loss_mask_nonzero: bool) -> FirewallEvent | None:
        for doc_id in doc_ids:
            entry = self.entries.get(doc_id)
            if entry and entry.never_train and loss_mask_nonzero:
                ev = FirewallEvent(doc_id, "blocked", "loss_bearing_eval", entry.permission)
                self.events.append(ev)
                return ev
        return None


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = graphemes(nfc(text))
    # Use whitespace-ish word tokens for containment of quoted sentences.
    words = "".join(tokens).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_firewall(docs: list[Document], tokenizer: FrozenTokenizer) -> EvalFirewall:
    fw = EvalFirewall()
    for doc in docs:
        if doc.plant in {"eval_item", "canary"} or (doc.never_train and doc.plant != "validation"):
            fw.register(doc, tokenizer, "test")
        elif doc.plant == "validation" or (doc.held_out and doc.never_train):
            fw.register(doc, tokenizer, "validate")
    # Always register the canonical eval item text so containment works even
    # if the plant document itself is also registered.
    probe = Document(
        doc_id="benchmark:gsm8k-plant",
        lane="math_science",
        segments=[("text", EVAL_ITEM)],
        language="en",
        script="Latn",
        license="CC-BY-4.0",
        source_id="benchmark",
        never_train=True,
        held_out=True,
        plant="eval_item",
    )
    fw.register(probe, tokenizer, "test")
    return fw
