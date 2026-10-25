"""Dataset, tokenizer, packing, and token-accounting tests."""

from __future__ import annotations

import pytest

from helpers.config import tiny_config
from helpers.data import (
    DataError,
    PackedCausalDataset,
    collate_causal,
    encode_corpus,
    pack_token_ids,
    prepare_data,
    split_texts,
    tokenize_text,
    train_tokenizer,
)
from helpers.model import shift_input_target


def test_empty_and_short_inputs():
    assert tokenize_text("") == []
    assert tokenize_text("   ") == []
    tok = train_tokenizer(["hello world"], vocab_size=16)
    assert encode_corpus([""], tok) == []
    packed, discarded = pack_token_ids([1, 2, 3], sequence_length=8)
    assert packed.shape[0] == 0
    assert discarded == 3
    with pytest.raises(DataError):
        tok.encode(None)  # type: ignore[arg-type]


def test_split_has_no_overlap():
    texts = [f"doc {i} unique token{i}" for i in range(20)]
    train, val = split_texts(texts, val_fraction=0.2)
    assert set(train).isdisjoint(set(val))
    assert len(train) + len(val) == 20


def test_shift_and_packing_shapes():
    ids = list(range(27))
    packed, discarded = pack_token_ids(ids, sequence_length=8)
    assert packed.shape == (3, 9)
    assert discarded == 0
    dataset = PackedCausalDataset(packed)
    batch = collate_causal([dataset[0], dataset[1]])
    assert tuple(batch["input_ids"].shape) == (2, 8)
    assert tuple(batch["labels"].shape) == (2, 8)
    assert int(batch["n_target_tokens"].item()) == 16
    assert int(batch["n_padding_tokens"].item()) == 0
    inputs, targets = shift_input_target(packed[0])
    assert torch_equal_shift(inputs, targets, packed[0])


def torch_equal_shift(inputs, targets, row):
    return list(inputs.tolist()) == row[:-1].tolist() and list(targets.tolist()) == row[1:].tolist()


def test_tokenizer_trained_on_train_split_only():
    train_texts = ["alpha beta gamma alpha"]
    tok = train_tokenizer(train_texts, vocab_size=16)
    assert "alpha" in tok.stoi
    # Validation-only token must not enter the vocab.
    assert "zebraunique" not in tok.stoi
    ids = tok.encode("zebraunique alpha")
    assert tok.unk_id in ids


def test_prepare_data_token_accounting(tmp_path):
    config = tiny_config(output_dir=str(tmp_path), token_budget=64, max_steps=1)
    data = prepare_data(config, download=False)
    assert data.source == "synthetic"
    assert data.train_target_tokens_per_epoch > 0
    assert data.accounting.padding_tokens == 0
    batch = next(iter(data.train_loader))
    assert batch["input_ids"].shape[1] == config.sequence_length
    assert batch["labels"].shape == batch["input_ids"].shape
    # Drop-last means every train batch is a full micro-batch of packed tokens.
    assert int(batch["n_target_tokens"].item()) == config.micro_batch_size * config.sequence_length


def test_binary_windows_match_causal_shift(tmp_path):
    import numpy as np
    from helpers.sangraha import BinaryWindowDataset

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 100, size=64, dtype=np.uint16)
    ds = BinaryWindowDataset(ids, context=8, n_windows=4, seed=1)
    assert len(ds) == 4
    row = ds[0]
    assert tuple(row.shape) == (9,)
    batch = collate_causal([ds[0], ds[1]])
    assert tuple(batch["input_ids"].shape) == (2, 8)
    assert int(batch["n_padding_tokens"].item()) == 0
    assert list(batch["labels"][0].tolist()) == row[1:].tolist()


def test_too_short_corpus_raises():
    config = tiny_config(block_size=8, vocab_size=16)
    with pytest.raises(DataError, match="no complete training examples"):
        prepare_data(config, texts=["hi"])
