"""Dataset preparation, tokenization, splits, and packed causal batches.

Token-counting convention
-------------------------
A packed example of sequence length ``T`` is formed from ``T + 1`` consecutive
token ids. Inputs are ``ids[0:T]`` and targets are ``ids[1:T+1]``. That yields
``T`` **training target tokens**. Padding is not used in the packed path, so
``padding_tokens = 0``. Tokens left over after packing are ``discarded_tokens``
and are not counted toward the budget. Validation tokens are stored on a
separate counter and never added to training target tokens.

The 50M assignment budget is 50 million training target-token positions,
counting repeats when the corpus is iterated more than once.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from helpers.config import ExperimentConfig, SPECIAL_TOKENS, ensure_dir
from helpers.metrics import TokenAccounting
from helpers.model import shift_input_target

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WIKITEXT2_FILES = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}
SYNTHETIC_VOCAB = (
    "the of and to in a is that for it as was with on be by this are from or at "
    "an which have not were but had his they you one all we can her she there "
    "their said will each about how up out many then them some these so would "
    "other into has more two like him see time could no make than first been "
    "its who now people my over year also just because good work well even new "
    "want give most us after back use our very any day same old think around "
    "another come only still should own under such through where much before "
    "those never between few since those while both during without however "
    "model token layer residual reversible midpoint euler transformer attention "
    "gradient memory batch sequence language training loss optimizer adam cosine"
).split()


class DataError(ValueError):
    """Invalid text, tokenizer state, or empty packed dataset."""


@dataclass
class Tokenizer:
    """Whitespace/punctuation tokenizer with a frozen vocabulary."""

    stoi: dict[str, int]
    itos: list[str]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        if text is None:
            raise DataError("text is None")
        pieces = tokenize_text(text)
        ids = [self.stoi.get(tok, self.unk_id) for tok in pieces]
        if add_special and ids:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        words = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                words.append("<unk>")
            else:
                token = self.itos[idx]
                if token not in SPECIAL_TOKENS:
                    words.append(token)
        return " ".join(words)

    def to_dict(self) -> dict[str, object]:
        return {"itos": self.itos, "stoi": self.stoi}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Tokenizer":
        itos = list(payload["itos"])
        stoi = {token: int(idx) for token, idx in dict(payload["stoi"]).items()}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls.from_dict(json.loads(Path(path).read_text()))


def tokenize_text(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return TOKEN_RE.findall(text.lower())


def train_tokenizer(texts: Sequence[str], vocab_size: int) -> Tokenizer:
    if vocab_size < len(SPECIAL_TOKENS):
        raise DataError(f"vocab_size must be >= {len(SPECIAL_TOKENS)}")
    counts: dict[str, int] = {}
    for text in texts:
        for token in tokenize_text(text):
            if token in SPECIAL_TOKENS:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    itos = list(SPECIAL_TOKENS)
    for token, _count in ranked:
        if len(itos) >= vocab_size:
            break
        itos.append(token)
    # If the corpus is tiny, fill remaining ids with placeholder tokens so
    # embedding tables match the configured vocab_size.
    filler = 0
    while len(itos) < vocab_size:
        name = f"<extra_{filler}>"
        if name not in itos:
            itos.append(name)
        filler += 1
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Tokenizer(stoi=stoi, itos=itos)


def synthetic_corpus(n_sentences: int = 256, seed: int = 0, words_per: int = 12) -> list[str]:
    """Deterministic local corpus used by tests and CPU smoke runs."""
    rng = torch.Generator().manual_seed(seed)
    vocab = SYNTHETIC_VOCAB
    sentences = []
    for _ in range(n_sentences):
        idxs = torch.randint(0, len(vocab), (words_per,), generator=rng).tolist()
        sentences.append(" ".join(vocab[i] for i in idxs) + ".")
    return sentences


def split_texts(texts: Sequence[str], val_fraction: float = 0.1) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise DataError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if not texts:
        return [], []
    n_val = max(1, int(round(len(texts) * val_fraction)))
    n_val = min(n_val, len(texts) - 1) if len(texts) > 1 else 0
    train = list(texts[: len(texts) - n_val])
    valid = list(texts[len(texts) - n_val :])
    return train, valid


def encode_corpus(texts: Sequence[str], tokenizer: Tokenizer) -> list[int]:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special=True))
    return ids


def pack_token_ids(
    token_ids: Sequence[int],
    sequence_length: int,
) -> tuple[torch.Tensor, int]:
    """Pack non-overlapping causal examples. Returns (N, T+1) tensor and discarded count."""
    if sequence_length < 2:
        raise DataError(f"sequence_length must be >= 2, got {sequence_length}")
    block = sequence_length + 1
    n_tokens = len(token_ids)
    n_blocks = n_tokens // block
    used = n_blocks * block
    discarded = n_tokens - used
    if n_blocks == 0:
        return torch.zeros(0, block, dtype=torch.long), discarded
    data = torch.tensor(token_ids[:used], dtype=torch.long).view(n_blocks, block)
    return data, discarded


class PackedCausalDataset(Dataset):
    """Each item is a (T+1) token row; collate shifts to input/target of length T."""

    def __init__(self, packed: torch.Tensor) -> None:
        if packed.ndim != 2:
            raise DataError(f"packed tensor must be (N, T+1), got {tuple(packed.shape)}")
        self.packed = packed.contiguous()

    def __len__(self) -> int:
        return int(self.packed.size(0))

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.packed[index]


def collate_causal(batch: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    rows = torch.stack(list(batch), dim=0)
    inputs, targets = shift_input_target(rows)
    return {
        "input_ids": inputs.long(),
        "labels": targets.long(),
        "n_input_tokens": torch.tensor(inputs.numel(), dtype=torch.long),
        "n_target_tokens": torch.tensor(targets.numel(), dtype=torch.long),
        "n_padding_tokens": torch.tensor(0, dtype=torch.long),
    }


@dataclass
class PreparedData:
    tokenizer: Tokenizer
    train_loader: DataLoader
    val_loader: DataLoader
    train_target_tokens_per_epoch: int
    val_target_tokens: int
    discarded_train_tokens: int
    discarded_val_tokens: int
    accounting: TokenAccounting
    source: str

    def tokenizer_info(self) -> dict[str, object]:
        algorithm = "regex_wordpiece_frequency"
        if getattr(self.tokenizer, "_json_path", None) is not None:
            algorithm = "byte_level_bpe"
        return {
            "vocab_size": self.tokenizer.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS) if algorithm != "byte_level_bpe" else ["<|endoftext|>"],
            "algorithm": algorithm,
            "source": self.source,
        }


def _download_text(url: str, dest: Path, timeout: int = 60) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_text(encoding="utf-8", errors="replace")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
    dest.write_bytes(payload)
    return dest.read_text(encoding="utf-8", errors="replace")


def load_wikitext2(cache_dir: str | Path) -> dict[str, list[str]]:
    cache = ensure_dir(cache_dir)
    splits: dict[str, list[str]] = {}
    for name, url in WIKITEXT2_FILES.items():
        text = _download_text(url, cache / f"{name}.txt")
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        splits[name] = paragraphs
    return splits


def prepare_data(
    config: ExperimentConfig,
    *,
    download: bool = False,
    texts: Sequence[str] | None = None,
) -> PreparedData:
    """Build tokenizer, packed loaders, and token accounting for one run."""
    source = config.dataset_name
    if texts is not None:
        source = "provided"
        train_texts, val_texts = split_texts(list(texts))
    elif config.dataset_name == "sangraha":
        from helpers.sangraha import build_sangraha_loaders

        tokenizer, train_loader, val_loader, meta, discarded, _train_ids, _val_ids, ctx = build_sangraha_loaders(
            config, download=download
        )
        train_target = int(len(train_loader) * config.micro_batch_size * ctx)
        val_target = int(len(val_loader) * val_loader.batch_size * ctx) if len(val_loader) else 0
        accounting = TokenAccounting(discarded_tokens=discarded)
        return PreparedData(
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            train_target_tokens_per_epoch=train_target,
            val_target_tokens=val_target,
            discarded_train_tokens=discarded,
            discarded_val_tokens=0,
            accounting=accounting,
            source="sangraha",
        )
    elif config.dataset_name == "synthetic" or not download:
        train_texts = synthetic_corpus(n_sentences=256, seed=config.seed)
        val_texts = synthetic_corpus(n_sentences=32, seed=config.seed + 1)
        source = "synthetic"
    else:
        try:
            splits = load_wikitext2(config.dataset_cache_dir)
            train_texts = splits["train"]
            val_texts = splits["valid"]
            source = "wikitext-2"
        except Exception as exc:  # noqa: BLE001 — fallback is required for Colab/offline
            train_texts = synthetic_corpus(n_sentences=256, seed=config.seed)
            val_texts = synthetic_corpus(n_sentences=32, seed=config.seed + 1)
            source = f"synthetic(fallback after {type(exc).__name__})"

    if not train_texts:
        raise DataError("training split is empty")
    tokenizer = train_tokenizer(train_texts, config.vocab_size)
    train_ids = encode_corpus(train_texts, tokenizer)
    val_ids = encode_corpus(val_texts, tokenizer)
    seq = int(config.sequence_length)
    train_packed, discarded_train = pack_token_ids(train_ids, seq)
    val_packed, discarded_val = pack_token_ids(val_ids, seq)
    if len(train_packed) == 0:
        raise DataError(
            "no complete training examples could be formed; provide more text "
            f"or reduce sequence_length (have {len(train_ids)} tokens, need "
            f"{seq + 1} per example)"
        )
    train_ds = PackedCausalDataset(train_packed)
    val_ds = PackedCausalDataset(val_packed)
    if len(train_ds) < config.micro_batch_size:
        raise DataError(
            f"packed training set has {len(train_ds)} examples but micro_batch_size="
            f"{config.micro_batch_size}"
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_causal,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, min(config.micro_batch_size, len(val_ds) or 1)),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_causal,
    )
    train_target = int(train_packed.size(0) * seq)
    val_target = int(val_packed.size(0) * seq) if val_packed.numel() else 0
    accounting = TokenAccounting(discarded_tokens=discarded_train)
    accounting.validation_tokens = 0  # populated only during evaluation
    return PreparedData(
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_target_tokens_per_epoch=train_target,
        val_target_tokens=val_target,
        discarded_train_tokens=discarded_train,
        discarded_val_tokens=discarded_val,
        accounting=accounting,
        source=source,
    )


def infinite_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        if len(loader) == 0:
            raise DataError("cannot iterate an empty data loader")
        yield from loader
