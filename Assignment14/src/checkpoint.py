"""Save and restore model, optimizer, configuration, and RNG state."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.config import ExperimentConfig
from src.dense_model import build_dense_model
from src.moe_model import build_moe_model, is_moe_model


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: MappingLike | None) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


MappingLike = dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ExperimentConfig,
    epoch: int,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_kind": type(model).__name__,
        "is_moe": is_moe_model(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "config": config.to_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "rng_state": capture_rng_state(),
        "extra": extra or {},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = None,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    destination = Path(path)
    if not destination.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {destination}")
    try:
        blob = torch.load(
            destination, map_location=map_location or "cpu", weights_only=False
        )
    except TypeError:
        blob = torch.load(destination, map_location=map_location or "cpu")
    config = ExperimentConfig.from_dict(blob["config"])
    if model is None:
        if blob.get("is_moe"):
            model = build_moe_model(config)
        else:
            model = build_dense_model(config)
        if map_location is not None:
            model.to(map_location)
    missing, unexpected = model.load_state_dict(blob["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state-dict mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    if optimizer is not None and blob.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(blob["optimizer_state_dict"])
    if restore_rng:
        restore_rng_state(blob.get("rng_state"))
    blob["model"] = model
    blob["config_obj"] = config
    return blob


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return destination
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(h, "")) for h in headers))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in ",\"\n"):
        return '"' + text.replace('"', '""') + '"'
    return text
