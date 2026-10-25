"""Deterministic synthetic datasets shared across experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def classification_tensors(
    n: int,
    input_dim: int,
    n_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = make_generator(seed)
    features = torch.randn(n, input_dim, generator=gen)
    # Linear separator in a random projection, then integer labels.
    proj = torch.randn(input_dim, n_classes, generator=gen)
    logits = features @ proj
    labels = torch.argmax(logits, dim=-1)
    return features, labels


def regression_from_teacher(
    n: int,
    input_dim: int,
    n_classes: int,
    seed: int,
    noise: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Labels from a frozen width-64 ReLU teacher plus small Gaussian noise."""
    gen = make_generator(seed)
    x = torch.randn(n, input_dim, generator=gen)
    w1 = torch.randn(input_dim, 64, generator=gen) / (input_dim**0.5)
    w2 = torch.randn(64, n_classes, generator=gen) / (64**0.5)
    hidden = torch.relu(x @ w1)
    y = hidden @ w2
    y = y + noise * torch.randn(n, n_classes, generator=gen)
    return x, y


def repeating_stream(
    n_steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Causal next-token batches: token t+1 = (token t + 1) mod vocab.

    The first token of each row is random; the rest is a deterministic walk.
    The same stream is replayed for every schedule so data order is matched.
    """
    gen = make_generator(seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_steps):
        start = torch.randint(0, vocab_size, (batch_size, 1), generator=gen)
        offsets = torch.arange(seq_len + 1).view(1, -1)
        tokens = (start + offsets) % vocab_size
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
        batches.append((inputs, targets))
    return batches


@dataclass
class TensorBatch:
    x: torch.Tensor
    y: torch.Tensor


def cycle_dataset(x: torch.Tensor, y: torch.Tensor, batch_size: int, steps: int, seed: int) -> list[TensorBatch]:
    gen = make_generator(seed)
    n = x.shape[0]
    batches: list[TensorBatch] = []
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        batches.append(TensorBatch(x=x[idx], y=y[idx]))
    return batches


def as_dataset(x: torch.Tensor, y: torch.Tensor) -> TensorDataset:
    return TensorDataset(x, y)
