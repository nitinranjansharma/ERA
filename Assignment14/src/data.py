"""Reproducible synthetic regression data with a known generating function.

Targets are a nonlinear combination of the eight input features plus a small
amount of Gaussian noise. The split is performed after a single seeded draw,
so there is no train/validation/test leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig


def generating_function(features: torch.Tensor) -> torch.Tensor:
    """Deterministic nonlinear target. ``features`` is [N, input_dim >= 8]."""
    if features.ndim != 2 or features.size(-1) < 8:
        raise ValueError(f"generating_function expected [N, >=8], got {tuple(features.shape)}")
    # Named slices keep the formula readable.
    x0 = features[:, 0]
    x1 = features[:, 1]
    x2 = features[:, 2]
    x3 = features[:, 3]
    x4 = features[:, 4]
    x5 = features[:, 5]
    x6 = features[:, 6]
    x7 = features[:, 7]
    target = (
        torch.sin(x0)
        + 0.5 * x1.square()
        + 0.3 * x0 * x2
        + 0.2 * x3
        - 0.4 * torch.tanh(x4)
        + 0.15 * x5 * x6
        + 0.1 * x7
    )
    return target.unsqueeze(-1)  # [N, 1]


@dataclass
class DataBundle:
    """Train / validation / test tensors plus ready-to-iterate loaders."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    seed: int

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "x_train": tuple(self.x_train.shape),
            "y_train": tuple(self.y_train.shape),
            "x_val": tuple(self.x_val.shape),
            "y_val": tuple(self.y_val.shape),
            "x_test": tuple(self.x_test.shape),
            "y_test": tuple(self.y_test.shape),
        }


def _make_loader(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(features, targets)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


def make_synthetic_data(config: ExperimentConfig, seed: int | None = None) -> DataBundle:
    """Draw features, apply the generating function, split, and wrap loaders."""
    seed = config.seed if seed is None else seed
    total = config.n_train + config.n_val + config.n_test
    generator = torch.Generator()
    generator.manual_seed(seed)

    features = torch.randn(total, config.input_dim, generator=generator)
    noise = torch.randn(total, config.output_dim, generator=generator) * config.noise_std
    targets = generating_function(features) + noise

    permutation = torch.randperm(total, generator=generator)
    features = features[permutation]
    targets = targets[permutation]

    n_train = config.n_train
    n_val = config.n_val
    x_train = features[:n_train].contiguous()
    y_train = targets[:n_train].contiguous()
    x_val = features[n_train : n_train + n_val].contiguous()
    y_val = targets[n_train : n_train + n_val].contiguous()
    x_test = features[n_train + n_val :].contiguous()
    y_test = targets[n_train + n_val :].contiguous()

    train_loader = _make_loader(
        x_train, y_train, config.batch_size, True, seed, config.num_workers
    )
    val_loader = _make_loader(
        x_val, y_val, config.batch_size, False, seed + 1, config.num_workers
    )
    test_loader = _make_loader(
        x_test, y_test, config.batch_size, False, seed + 2, config.num_workers
    )
    return DataBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        seed=seed,
    )


def iter_tensors(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(features, targets)`` pairs without extra wrapping."""
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise RuntimeError("Expected TensorDataset batches of (features, targets)")
        yield batch[0], batch[1]
