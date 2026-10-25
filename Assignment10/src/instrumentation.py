"""Environment recording, seeds, shape tracing, and artifact I/O."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


@dataclass
class TensorRecord:
    name: str
    shape: list[int]
    dtype: str
    dims: list[str]
    explanation: str
    requires_grad: Optional[bool] = None
    device: str = "cpu"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        shape_s = "(" + ", ".join(str(s) for s in self.shape) + ("," if len(self.shape) == 1 else "") + ")"
        dims_s = " × ".join(self.dims) if self.dims else "scalar"
        return (
            f"{self.name:28s} shape={shape_s:18s} dtype={self.dtype:12s} "
            f"dims={dims_s:32s} {self.explanation}"
        )


class ShapeTracer:
    """Collect named tensors with an explicit interpretation of each axis."""

    def __init__(self) -> None:
        self.records: list[TensorRecord] = []

    def add(
        self,
        name: str,
        tensor: torch.Tensor,
        dims: Iterable[str],
        explanation: str,
    ) -> torch.Tensor:
        dims_list = list(dims)
        if tensor.ndim != len(dims_list) and not (tensor.ndim == 0 and dims_list == []):
            raise ValueError(
                f"Dimension labels for {name!r} do not match rank: "
                f"shape={tuple(tensor.shape)} labels={dims_list}"
            )
        rec = TensorRecord(
            name=name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            dims=dims_list,
            explanation=explanation,
            requires_grad=bool(tensor.requires_grad) if tensor.requires_grad is not None else None,
            device=str(tensor.device),
        )
        self.records.append(rec)
        return tensor

    def as_dicts(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.records]

    def format_report(self) -> str:
        lines = [
            "Complete training-step tensor inventory",
            "=" * 72,
        ]
        for rec in self.records:
            lines.append(rec.line())
        return "\n".join(lines) + "\n"


def set_seed(seed: int, deterministic: bool = True) -> None:
    if seed is None:
        raise ValueError("seed is required for a reproducible experiment configuration")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "cpu") -> torch.device:
    requested = (requested or "cpu").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise RuntimeError(f"Unsupported device {requested!r}")


def record_environment(seed: int, device: torch.device) -> dict[str, Any]:
    mps = getattr(torch.backends, "mps", None)
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": seed,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_built": bool(mps.is_built()) if mps is not None else False,
        "mps_available": bool(mps.is_available()) if mps is not None else False,
        "deterministic_requested": True,
    }
    try:
        import matplotlib

        env["matplotlib"] = matplotlib.__version__
    except Exception:
        env["matplotlib"] = None
    return env


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    path = path or (ROOT / "configs" / "default.yaml")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def create_run_dir(overwrite: bool = False, run_id: Optional[str] = None) -> Path:
    """Create a timestamped run directory. Never clobber a previous run unless overwrite=True
    *and* the caller is targeting the stable latest copies under outputs/{figures,logs,metrics}.
    """
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUTS / "runs" / stamp
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    for sub in ("figures", "logs", "metrics"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def mirror_to_stable(run_dir: Path, overwrite: bool = False) -> None:
    """Copy this run's artifacts into the stable outputs/ tree.

    The timestamped run directory is always unique. The stable copies are only
    replaced when overwrite is True, or when the destination file does not exist.
    """
    for sub in ("figures", "logs", "metrics"):
        src = run_dir / sub
        dst = OUTPUTS / sub
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            continue
        for path in src.iterdir():
            target = dst / path.name
            if target.exists() and not overwrite:
                continue
            target.write_bytes(path.read_bytes())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
        fh.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def now_seconds() -> float:
    return time.perf_counter()


@dataclass
class StepLog:
    step: int
    loss: float
    grad_norm: float
    lr: float
    elapsed_s: float
    throughput_samples_per_s: float
    tokens_per_s: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra") or {}
        d.update(extra)
        return d
