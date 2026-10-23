"""Document admission: PII, license, script, repeated n-grams, exact + Jaccard dedup."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from helix.constants import (
    JACCARD_DEDUP_THRESHOLD,
    LICENSE_ALLOWLIST,
    PII_EMAIL,
    REPEAT_NGRAM_MAX_N,
    REPEAT_NGRAM_MIN_N,
    REPEAT_RUN_THRESHOLD,
    SHINGLE_N,
)
from helix.corpus import Document
from helix.hashutil import sha256_text
from helix.tokenizer import FrozenTokenizer, graphemes, nfc

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-]{8,}\d")


@dataclass
class CleanRecord:
    doc_id: str
    admitted: bool
    reasons: list[str] = field(default_factory=list)
    content_hash: str = ""
    nfc_text: str = ""
    token_ids: list[int] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    cleaning_pipeline_hash: str = ""
    dedup_status: str = "unique"
    pii_status: str = "clean"
    language_status: str = "ok"
    license_status: str = "ok"
    ngram_repeat_status: str = "ok"


PIPELINE_ID = "helix-clean-v1:nfc+pii+license+script+repeat+exact+jaccard"


def _pii_hits(text: str) -> list[str]:
    hits = []
    if EMAIL_RE.search(text) or PII_EMAIL in text:
        hits.append("email")
    if PHONE_RE.search(text):
        hits.append("phone")
    return hits


def _script_ok(doc: Document, text: str) -> bool:
    if doc.lane == "indic" or doc.script == "Deva":
        return any("\u0900" <= ch <= "\u097F" for ch in text)
    if doc.language == "en":
        latin = sum(ch.isascii() and (ch.isalpha() or ch.isspace()) for ch in text)
        return latin >= 8
    return True


def _repeat_run(token_ids: list[int]) -> bool:
    n = len(token_ids)
    for width in range(REPEAT_NGRAM_MIN_N, REPEAT_NGRAM_MAX_N + 1):
        if n < width * REPEAT_RUN_THRESHOLD:
            continue
        run = 1
        prev = tuple(token_ids[:width])
        i = width
        while i + width <= n:
            cur = tuple(token_ids[i : i + width])
            if cur == prev:
                run += 1
                if run >= REPEAT_RUN_THRESHOLD:
                    return True
                i += width
            else:
                prev = cur
                run = 1
                i += width
    return False


def _shingles(token_ids: list[int], n: int = SHINGLE_N) -> set[tuple[int, ...]]:
    if len(token_ids) < n:
        return {tuple(token_ids)} if token_ids else set()
    return {tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def encode_document(doc: Document, tokenizer: FrozenTokenizer) -> tuple[list[int], list[str]]:
    """Tokenize each role segment, wrapping structured roles with specials."""
    from helix.constants import ROLE_TEXT, ROLE_WRAPPERS

    ids: list[int] = []
    roles: list[str] = []
    for role, body in doc.segments:
        wrapper = ROLE_WRAPPERS.get(role)
        if wrapper:
            ids.append(tokenizer.encode_token(wrapper[0]))
            roles.append(role)
        for g in graphemes(body):
            ids.append(tokenizer.token_to_id.get(g, tokenizer.unk_id))
            roles.append(role)
        if wrapper:
            ids.append(tokenizer.encode_token(wrapper[1]))
            roles.append(role)
        if role == ROLE_TEXT:
            # web/code/math/indic/reasoning: already encoded above without wrappers
            pass
    return ids, roles


def clean_documents(docs: list[Document], tokenizer: FrozenTokenizer) -> list[CleanRecord]:
    records: list[CleanRecord] = []
    seen_hashes: dict[str, str] = {}
    admitted_shingles: list[tuple[str, set]] = []
    pipeline_hash = sha256_text(PIPELINE_ID + tokenizer.tokenizer_hash)

    for doc in docs:
        text = nfc(doc.text)
        rec = CleanRecord(
            doc_id=doc.doc_id,
            admitted=True,
            nfc_text=text,
            content_hash=sha256_text(text),
            cleaning_pipeline_hash=pipeline_hash,
        )
        rec.token_ids, rec.roles = encode_document(doc, tokenizer)

        if doc.license not in LICENSE_ALLOWLIST:
            rec.admitted = False
            rec.license_status = "blocked"
            rec.reasons.append(f"license:{doc.license}")

        pii = _pii_hits(text)
        if pii:
            rec.admitted = False
            rec.pii_status = "blocked:" + ",".join(pii)
            rec.reasons.append("pii")

        if not _script_ok(doc, text):
            rec.admitted = False
            rec.language_status = "blocked"
            rec.reasons.append("language_script")

        if _repeat_run(rec.token_ids):
            rec.admitted = False
            rec.ngram_repeat_status = "blocked"
            rec.reasons.append("repeat_ngram")

        if rec.content_hash in seen_hashes:
            rec.admitted = False
            rec.dedup_status = f"exact_dup_of:{seen_hashes[rec.content_hash]}"
            rec.reasons.append("exact_dup")

        shingles = _shingles(rec.token_ids)
        if rec.admitted:
            for other_id, other_sh in admitted_shingles:
                score = jaccard(shingles, other_sh)
                if score >= JACCARD_DEDUP_THRESHOLD:
                    rec.admitted = False
                    rec.dedup_status = f"near_dup_of:{other_id}:{score:.3f}"
                    rec.reasons.append("near_dup")
                    break

        if rec.admitted:
            seen_hashes[rec.content_hash] = doc.doc_id
            admitted_shingles.append((doc.doc_id, shingles))

        records.append(rec)
    return records
