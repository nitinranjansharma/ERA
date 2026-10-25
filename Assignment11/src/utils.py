"""Reproducibility, serialization, and filesystem helpers."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return ROOT


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} must be a mapping")
    data["_config_path"] = str(cfg_path)
    return data


def resolve_device(name: str = "cpu") -> torch.device:
    requested = str(name).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in {"mps", "gpu"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = True) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except (RuntimeError, AttributeError):
            # Some CPU ops have no deterministic kernel; keep the rest pinned.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                pass
    return seed


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}")
    return mapping[key]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def environment_record(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": int(seed),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cwd": str(Path.cwd()),
        "timestamp_utc": utc_stamp(),
    }


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"non_finite": True, "repr": str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if torch.is_tensor(value):
        if value.ndim == 0:
            return jsonable(value.detach().cpu().item())
        return [jsonable(v) for v in value.detach().cpu().tolist()]
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=False)
        handle.write("\n")
    return target


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    names = fieldnames or list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in names})
    return target


def _csv_cell(value: Any) -> str:
    converted = jsonable(value)
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, separators=(",", ":"))
    if converted is None:
        return ""
    return str(converted)


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
