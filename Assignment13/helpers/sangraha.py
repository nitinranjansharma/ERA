"""Optional Indic corpus path matching the proven S13 Colab notebook.

Downloads a mixture of AI4Bharat Sangraha (Hindi, Telugu, Tamil, Marathi,
Bengali, English), trains an 8192-piece byte-level BPE tokenizer, and writes
``train.bin`` / ``val.bin`` as uint16 arrays loaded with ``np.fromfile``
(RAM, not memmap). Tests never import this module.

Requires optional packages: ``datasets`` and ``tokenizers``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from helpers.config import ExperimentConfig, ensure_dir
from helpers.data import DataError, collate_causal

SANGRAHA_MIX = {
    "hin": 0.30,
    "tel": 0.25,
    "tam": 0.15,
    "mar": 0.10,
    "ben": 0.10,
    "eng": 0.10,
}
EOT = "<|endoftext|>"
DEFAULT_VAL_TOKENS = 1_000_000


class DataUnavailable(DataError):
    """Sangraha extras are not installed or the cache is missing."""


class BpeTokenizerAdapter:
    """Duck-types the word tokenizer's ``save`` / ``vocab_size`` surface."""

    def __init__(self, hf_tokenizer: Any, vocab_size: int, json_path: Path) -> None:
        self._tok = hf_tokenizer
        self._vocab_size = int(vocab_size)
        self._json_path = Path(json_path)
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._json_path.exists():
            path.write_bytes(self._json_path.read_bytes())
        else:
            self._tok.save(str(path))
        return path

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = list(self._tok.encode(text).ids)
        if add_special:
            eot = self._tok.token_to_id(EOT)
            if eot is not None and (not ids or ids[-1] != eot):
                ids.append(eot)
        return ids


class BinaryWindowDataset(Dataset):
    """Random next-token windows from a uint16 ``.bin`` file held in RAM."""

    def __init__(self, data: np.ndarray, context: int, n_windows: int, seed: int) -> None:
        if data.dtype != np.uint16:
            data = data.astype(np.uint16, copy=False)
        if len(data) < context + 2:
            raise DataError(
                f"token file is too short ({len(data)} ids) for context {context}"
            )
        self.data = data
        self.context = int(context)
        self.max_start = int(len(data) - context - 1)
        rng = np.random.default_rng(seed)
        n_windows = max(1, int(n_windows))
        self.starts = rng.integers(0, self.max_start, size=n_windows, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        start = int(self.starts[int(index)])
        row = self.data[start : start + self.context + 1].astype(np.int64, copy=True)
        return torch.from_numpy(row)


def _require_extras() -> tuple[Any, Any]:
    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
    except ImportError as exc:
        raise DataUnavailable(
            "dataset='sangraha' needs optional packages: "
            "pip install 'datasets>=2.18' 'tokenizers>=0.15'"
        ) from exc
    return load_dataset, {
        "Tokenizer": Tokenizer,
        "models": models,
        "pre_tokenizers": pre_tokenizers,
        "decoders": decoders,
        "trainers": trainers,
        "processors": processors,
    }


def _download_language(lang: str, want_bytes: int, dest: Path, load_dataset: Any) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= want_bytes * 0.98:
        print(f"  {lang}: cached")
        return dest
    t0, n = time.time(), 0
    with dest.open("w", encoding="utf-8") as handle:
        for rec in load_dataset("ai4bharat/sangraha", "verified", split=lang, streaming=True):
            text = (rec.get("text") or "").strip()
            if len(text) < 200:
                continue
            handle.write(text + "\n" + EOT + "\n")
            n += len(text.encode()) + 16
            if n >= want_bytes:
                break
    print(f"  {lang}: {n / 1e6:5.0f} MB ({time.time() - t0:.0f}s)")
    return dest


def _train_bpe(files: dict[str, Path], vocab_size: int, tok_mod: dict[str, Any], dest: Path) -> Any:
    Tokenizer = tok_mod["Tokenizer"]
    models = tok_mod["models"]
    pre_tokenizers = tok_mod["pre_tokenizers"]
    decoders = tok_mod["decoders"]
    trainers = tok_mod["trainers"]
    processors = tok_mod["processors"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    def sample():
        for path in files.values():
            used = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip() != EOT:
                        yield line
                        used += len(line)
                    if used > 10_000_000:
                        break

    tok.train_from_iterator(
        sample(),
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOT],
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(dest))
    return tok


def build_sangraha_cache(
    cache_dir: str | Path,
    token_budget: int,
    vocab_size: int,
    val_tokens: int = DEFAULT_VAL_TOKENS,
    download: bool = True,
) -> dict[str, Any]:
    """Download + tokenize, or reuse ``meta.json`` if the cache is complete."""
    cache = ensure_dir(cache_dir)
    meta_path = cache / "meta.json"
    train_bin = cache / "train.bin"
    val_bin = cache / "val.bin"
    tok_path = cache / f"bpe_{vocab_size}.json"
    if (
        meta_path.exists()
        and train_bin.exists()
        and val_bin.exists()
        and tok_path.exists()
    ):
        meta = json.loads(meta_path.read_text())
        if int(meta.get("train_tokens", 0)) >= int(token_budget) * 0.97:
            print("Sangraha cache hit.")
            return meta
    if not download:
        raise DataUnavailable(
            f"Sangraha cache missing under {cache}. Pass download=True on Colab."
        )

    load_dataset, tok_mod = _require_extras()
    print("Step 1/3 -- downloading Sangraha (~250 MB, 8-12 min on Colab)")
    files = {
        lang: _download_language(
            lang,
            int((token_budget + val_tokens) * share * 5),
            cache / f"raw_{lang}.txt",
            load_dataset,
        )
        for lang, share in SANGRAHA_MIX.items()
    }

    print("\nStep 2/3 -- training an 8192-piece byte-level tokenizer")
    tok = _train_bpe(files, vocab_size, tok_mod, tok_path)
    eot = tok.token_to_id(EOT)

    print("\nStep 3/3 -- turning text into numbers")
    pieces, stats = [], {}
    for lang, path in files.items():
        docs = [d for d in path.read_text(encoding="utf-8").split(EOT) if d.strip()]
        arr = np.concatenate(
            [
                np.array(enc.ids + [eot], dtype=np.uint16)
                for i in range(0, len(docs), 2000)
                for enc in tok.encode_batch(docs[i : i + 2000])
            ]
        )
        stats[lang] = {
            "tokens": int(len(arr)),
            "bytes_per_token": round(sum(len(d.encode()) for d in docs) / max(len(arr), 1), 2),
        }
        pieces.append(arr)
        print(f"  {lang}: {len(arr) / 1e6:5.2f}M tokens, {stats[lang]['bytes_per_token']} bytes/token")

    ids = np.concatenate(pieces)
    cuts = np.flatnonzero(ids == eot) + 1
    docs_ids = np.split(ids, cuts[:-1]) if cuts.size else [ids]
    np.random.default_rng(1337).shuffle(docs_ids)
    ids = np.concatenate(docs_ids)[: token_budget + val_tokens]
    ids[:val_tokens].tofile(val_bin)
    ids[val_tokens:].tofile(train_bin)
    meta = {
        "vocab": vocab_size,
        "train_tokens": int(len(ids) - val_tokens),
        "val_tokens": int(min(val_tokens, len(ids))),
        "per_language": stats,
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "tokenizer": str(tok_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"\nReady: {meta['train_tokens'] / 1e6:.1f}M training tokens, "
        f"{meta['val_tokens'] / 1e6:.1f}M held out"
    )
    return meta


def load_uint16_bin(path: str | Path) -> np.ndarray:
    """Load the whole file into RAM. Colab disk is network-backed; memmap is slow."""
    return np.fromfile(str(path), dtype=np.uint16)


def windows_for_budget(token_budget: int, context: int, micro_batch: int) -> int:
    per_example = max(int(context), 1)
    n = max(int(token_budget) // per_example, micro_batch)
    return int(n)


def build_sangraha_loaders(
    config: ExperimentConfig,
    *,
    download: bool = True,
):
    """Return (tokenizer, train_loader, val_loader, meta, discarded)."""
    from tokenizers import Tokenizer as HfTokenizer

    cache_dir = config.dataset_cache_dir or "outputs/data/sangraha"
    if "wikitext" in str(cache_dir):
        cache_dir = "outputs/data/sangraha"
    val_tokens = max(int(config.eval_token_budget), 1)
    meta = build_sangraha_cache(
        cache_dir,
        token_budget=int(config.token_budget),
        vocab_size=int(config.vocab_size),
        val_tokens=val_tokens,
        download=download,
    )
    tok_path = Path(meta.get("tokenizer") or (Path(cache_dir) / f"bpe_{config.vocab_size}.json"))
    hf_tok = HfTokenizer.from_file(str(tok_path))
    adapter = BpeTokenizerAdapter(hf_tok, config.vocab_size, tok_path)
    train_ids = load_uint16_bin(meta.get("train_bin") or Path(cache_dir) / "train.bin")
    val_ids = load_uint16_bin(meta.get("val_bin") or Path(cache_dir) / "val.bin")
    ctx = int(config.block_size)
    n_train = windows_for_budget(config.token_budget, ctx, config.micro_batch_size)
    n_val = max(config.micro_batch_size, windows_for_budget(val_tokens, ctx, 1))
    train_ds = BinaryWindowDataset(train_ids, ctx, n_train, seed=config.seed)
    val_ds = BinaryWindowDataset(val_ids, ctx, n_val, seed=config.seed + 7)
    if len(train_ds) < config.micro_batch_size:
        raise DataError("Sangraha training set is smaller than micro_batch_size")
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
        batch_size=max(1, min(config.micro_batch_size, len(val_ds))),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_causal,
    )
    discarded = 0
    return adapter, train_loader, val_loader, meta, discarded, train_ids, val_ids, ctx
