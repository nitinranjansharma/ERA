"""NFC grapheme tokenizer. Same text -> same ids. Combining marks stay on the base char."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from helix.constants import PAD, SPECIAL_TOKENS, UNK
from helix.hashutil import sha256_obj


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def graphemes(text: str) -> list[str]:
    """NFC string → grapheme clusters (base character + combining marks)."""
    text = nfc(text)
    clusters: list[str] = []
    for ch in text:
        if clusters and unicodedata.combining(ch):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


@dataclass
class FrozenTokenizer:
    vocab: list[str]
    token_to_id: dict[str, int] = field(init=False)
    tokenizer_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        if self.vocab[0] != PAD:
            raise ValueError("token id 0 must be <pad> so padding is unambiguous")
        self.tokenizer_hash = sha256_obj(
            {
                "algo": "nfc-grapheme-v1",
                "vocab": self.vocab,
                "specials": list(SPECIAL_TOKENS),
            }
        )

    @classmethod
    def from_texts(cls, texts: list[str]) -> "FrozenTokenizer":
        vocab = list(SPECIAL_TOKENS)
        seen = set(vocab)
        for text in texts:
            for g in graphemes(text):
                if g not in seen:
                    seen.add(g)
                    vocab.append(g)
        return cls(vocab=vocab)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for g in graphemes(text):
            ids.append(self.token_to_id.get(g, self.unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for i in ids:
            if 0 <= i < len(self.vocab):
                tok = self.vocab[i]
                if tok not in SPECIAL_TOKENS:
                    pieces.append(tok)
        return "".join(pieces)

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)
