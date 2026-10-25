"""Validated experiment configuration, seeds, and runtime diagnostics.

The default model is a decoder-only causal Transformer sized for
approximately 20 million trainable parameters with tied embeddings:

    vocab_size=8192, n_embd=384, n_layer=12, n_head=6, n_inner=1024, block_size=512

Closed-form count with no Linear bias (LayerNorm still affine) is
19,876,608 (~0.6% below 20M). Tests compare the live PyTorch count to
this formula and to the 20M target within PARAM_COUNT_TOLERANCE.

Primary reversible variant is the midpoint / leapfrog recurrence:

    p[l+1] = p[l-1] + 2 * h * f_theta_l(p[l])
    p[l-1] = p[l+1] - 2 * h * f_theta_l(p[l])

Default h=0.5 so that 2h=1 and residual-function magnitude matches the
ordinary skip connection. This is not a first-order residual block.
"""

from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch

TARGET_PARAM_COUNT = 20_000_000
PARAM_COUNT_TOLERANCE = 0.15  # |count - 20M| / 20M
DEFAULT_TOKEN_BUDGET = 50_000_000
DEFAULT_SEED = 1337
VALID_PRECISIONS = ("fp32", "fp16", "bf16", "auto")
VALID_VARIANTS = ("baseline", "midpoint", "euler")
VALID_EXPERIMENTS = ("A", "B", "C")
VALID_DATASETS = ("wikitext2", "synthetic", "sangraha")
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# WikiText-2 (already split). Cached under output_dir/data/.
WIKITEXT2_BASE_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/"
    "master/word_language_model/data/wikitext-2"
)
WIKITEXT2_SPLITS = ("train.txt", "valid.txt", "test.txt")


PrecisionName = Literal["fp32", "fp16", "bf16"]
VariantName = Literal["baseline", "midpoint", "euler"]
ExperimentName = Literal["A", "B", "C"]


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _prob(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float in [0, 1), got {value!r}") from exc
    if not 0.0 <= number < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {number}")
    return number


@dataclass
class ExperimentConfig:
    """Serializable configuration for one training or smoke run."""

    experiment: str = "A"
    variant: str = "baseline"
    vocab_size: int = 8192
    n_embd: int = 384
    n_layer: int = 12
    n_head: int = 6
    n_inner: int = 1024
    block_size: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    midpoint_step_size: float = 0.5
    euler_step_size: float | None = None
    micro_batch_size: int = 32
    accumulation_steps: int = 1
    token_budget: int = DEFAULT_TOKEN_BUDGET
    eval_token_budget: int = 50_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_tokens: int = 5_000_000
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    precision: str = "auto"
    seed: int = DEFAULT_SEED
    dataset: str = "wikitext2"
    num_workers: int = 0
    log_interval_steps: int = 100
    eval_interval_tokens: int = 2_000_000
    checkpoint_interval_tokens: int = 10_000_000
    reconstruction_check_interval: int = 50
    reconstruction_atol: float = 1e-3
    reconstruction_rtol: float = 1e-3
    max_steps: int | None = None
    output_dir: str = "outputs"
    run_name: str = ""
    device: str = "auto"
    compile_model: bool = False
    mode: str = "full"
    resume_path: str | None = None
    checkpoint_name: str = "checkpoint.pt"
    optimizer_name: str = "adamw"
    notes: str = ""
    dataset_cache_dir: str = "outputs/data/wikitext2"
    batch_search_start: int = 1
    max_search_batch: int = 64
    confirm_batch_steps: int = 3
    headroom: bool = True

    def __post_init__(self) -> None:
        validate_config(self)

    @property
    def effective_batch_size(self) -> int:
        """Sequences per optimizer step (not tokens, not micro-batch)."""
        return self.micro_batch_size * self.accumulation_steps

    @property
    def tokens_per_micro_batch(self) -> int:
        """Non-padding target tokens in one packed micro-batch."""
        return self.micro_batch_size * self.block_size

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.effective_batch_size * self.block_size

    @property
    def two_h(self) -> float:
        return 2.0 * float(self.midpoint_step_size)

    @property
    def euler_h(self) -> float:
        """Lecture default: h = 1/sqrt(depth). Override with euler_step_size."""
        if self.euler_step_size is not None:
            return float(self.euler_step_size)
        return 1.0 / (float(self.n_layer) ** 0.5)

    @property
    def is_reversible(self) -> bool:
        return self.variant in ("midpoint", "euler")

    def resolved_precision(self) -> str:
        if self.precision != "auto":
            return self.precision
        return recommend_precision(self.resolved_device())

    @property
    def torch_dtype(self) -> torch.dtype:
        # Weights stay fp32. AMP (if any) only wraps F.
        return torch.float32

    @property
    def use_autocast(self) -> bool:
        return self.resolved_precision() in ("fp16", "bf16")

    @property
    def sequence_length(self) -> int:
        return self.block_size

    @property
    def dataset_name(self) -> str:
        return self.dataset

    @property
    def dtype(self) -> torch.dtype:
        return self.torch_dtype

    @property
    def min_learning_rate(self) -> float:
        return float(self.learning_rate) * float(self.min_lr_ratio)

    @property
    def log_every_steps(self) -> int:
        return self.log_interval_steps

    @property
    def eval_every_tokens(self) -> int:
        return self.eval_interval_tokens

    def resolved_device(self) -> torch.device:
        return resolve_device(self.device)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effective_batch_size"] = self.effective_batch_size
        payload["tokens_per_micro_batch"] = self.tokens_per_micro_batch
        payload["tokens_per_optimizer_step"] = self.tokens_per_optimizer_step
        payload["two_h"] = self.two_h
        payload["target_param_count"] = TARGET_PARAM_COUNT
        payload["param_count_tolerance"] = PARAM_COUNT_TOLERANCE
        payload["sequence_length"] = self.sequence_length
        payload["dataset_name"] = self.dataset_name
        payload["min_learning_rate"] = self.min_learning_rate
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        allowed = {item.name for item in fields(cls)}
        # Derived keys are allowed in snapshots.
        derived = {
            "effective_batch_size",
            "tokens_per_micro_batch",
            "tokens_per_optimizer_step",
            "two_h",
            "target_param_count",
            "param_count_tolerance",
            "sequence_length",
            "dataset_name",
            "dtype",
            "min_learning_rate",
            "log_every_steps",
            "eval_every_tokens",
            "is_reversible",
            "torch_dtype",
            "use_autocast",
            "resolved_precision",
        }
        data = dict(data)
        if "sequence_length" in data and "block_size" not in data:
            data["block_size"] = data["sequence_length"]
        if "dataset_name" in data and "dataset" not in data:
            data["dataset"] = data["dataset_name"]
        if "log_every_steps" in data and "log_interval_steps" not in data:
            data["log_interval_steps"] = data["log_every_steps"]
        if "eval_every_tokens" in data and "eval_interval_tokens" not in data:
            data["eval_interval_tokens"] = data["eval_every_tokens"]
        unknown = set(data) - allowed
        unknown -= derived
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must contain a JSON object: {path}")
        return cls.from_dict(payload)

    def with_updates(self, **changes: Any) -> "ExperimentConfig":
        return replace(self, **changes)


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Raise ValueError with an actionable message for invalid settings."""
    if config.experiment not in VALID_EXPERIMENTS:
        raise ValueError(
            f"experiment must be one of {VALID_EXPERIMENTS}, got {config.experiment!r}"
        )
    if config.variant not in VALID_VARIANTS:
        raise ValueError(
            f"variant must be one of {VALID_VARIANTS}, got {config.variant!r}"
        )
    expected_variant = {"A": "baseline", "B": "midpoint", "C": "midpoint"}
    if config.variant != expected_variant[config.experiment] and config.experiment in expected_variant:
        # Allow explicit override only when the pair is internally consistent
        # for tests; primary experiments still validate the intended pairing.
        pass
    if config.experiment == "A" and config.variant != "baseline":
        raise ValueError("Experiment A must use variant='baseline'")
    if config.experiment in ("B", "C") and config.variant == "baseline":
        raise ValueError(f"Experiment {config.experiment} must use a reversible variant")

    config.vocab_size = _positive_int("vocab_size", config.vocab_size, minimum=len(SPECIAL_TOKENS))
    config.n_embd = _positive_int("n_embd", config.n_embd, minimum=2)
    config.n_layer = _positive_int("n_layer", config.n_layer)
    config.n_head = _positive_int("n_head", config.n_head)
    config.n_inner = _positive_int("n_inner", config.n_inner, minimum=2)
    config.block_size = _positive_int("block_size", config.block_size, minimum=2)
    config.micro_batch_size = _positive_int("micro_batch_size", config.micro_batch_size)
    config.accumulation_steps = _positive_int("accumulation_steps", config.accumulation_steps)
    config.token_budget = _positive_int("token_budget", config.token_budget)
    config.eval_token_budget = _positive_int("eval_token_budget", config.eval_token_budget)
    config.seed = _non_negative_int("seed", config.seed)
    config.num_workers = _non_negative_int("num_workers", config.num_workers)
    config.dropout = _prob("dropout", config.dropout)

    if config.n_embd % config.n_head != 0:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
    if config.euler_step_size is not None and not 0.0 < float(config.euler_step_size) <= 2.0:
        raise ValueError(f"euler_step_size must be in (0, 2], got {config.euler_step_size}")
    if config.is_reversible and config.dropout != 0.0:
        raise ValueError(
            "Reversible reconstruction requires dropout=0 so the inverse can "
            "recompute f_theta deterministically. Preserve and replay RNG if you "
            "need stochastic layers; that path is not implemented here."
        )
    if config.precision not in VALID_PRECISIONS:
        raise ValueError(
            f"precision must be one of {VALID_PRECISIONS}, got {config.precision!r}"
        )
    if not 0.0 < float(config.midpoint_step_size) <= 2.0:
        raise ValueError(
            f"midpoint_step_size must be in (0, 2], got {config.midpoint_step_size}"
        )
    if not 0.0 < float(config.learning_rate) < 1.0:
        raise ValueError(f"learning_rate out of range: {config.learning_rate}")
    if not 0.0 <= float(config.min_lr_ratio) <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {config.min_lr_ratio}")
    if float(config.grad_clip) < 0.0:
        raise ValueError("grad_clip must be >= 0")
    if config.max_steps is not None:
        config.max_steps = _positive_int("max_steps", config.max_steps)
    if config.mode not in ("smoke", "full"):
        raise ValueError(f"mode must be 'smoke' or 'full', got {config.mode!r}")
    if config.dataset in ("wikitext-2", "wikitext2-raw"):
        config.dataset = "wikitext2"
    if config.dataset in ("indic", "sangraha-indic", "ai4bharat"):
        config.dataset = "sangraha"
    if config.dataset not in VALID_DATASETS:
        raise ValueError(f"Unsupported dataset {config.dataset!r}")
    config.batch_search_start = _positive_int("batch_search_start", config.batch_search_start)
    config.max_search_batch = _positive_int("max_search_batch", config.max_search_batch)
    config.confirm_batch_steps = _positive_int("confirm_batch_steps", config.confirm_batch_steps)
    if config.max_search_batch < config.batch_search_start:
        raise ValueError("max_search_batch must be >= batch_search_start")
    return config


def tiny_config(**overrides: Any) -> ExperimentConfig:
    """Deterministic unit-test configuration. Not the 20M assignment model."""
    values: dict[str, Any] = dict(
        experiment="A",
        variant="baseline",
        vocab_size=32,
        n_embd=16,
        n_layer=2,
        n_head=4,
        n_inner=32,
        block_size=8,
        dropout=0.0,
        bias=True,
        tie_embeddings=True,
        midpoint_step_size=0.5,
        micro_batch_size=2,
        accumulation_steps=1,
        token_budget=256,
        eval_token_budget=64,
        learning_rate=1e-3,
        warmup_tokens=32,
        precision="fp32",
        seed=0,
        dataset="synthetic",
        log_interval_steps=1,
        eval_interval_tokens=10_000,
        checkpoint_interval_tokens=10_000,
        reconstruction_check_interval=1,
        max_steps=2,
        output_dir="outputs/tests",
        run_name="tiny",
        mode="smoke",
        max_search_batch=16,
        confirm_batch_steps=2,
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def assignment_config(experiment: str, **overrides: Any) -> ExperimentConfig:
    """Primary ~20M configuration used by Experiments A/B/C on Colab GPU."""
    experiment = experiment.upper()
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(f"experiment must be one of {VALID_EXPERIMENTS}")
    variant = "baseline" if experiment == "A" else "midpoint"
    values: dict[str, Any] = dict(
        experiment=experiment,
        variant=variant,
        run_name=f"experiment_{experiment}",
        dataset="wikitext2",
        token_budget=DEFAULT_TOKEN_BUDGET,
        micro_batch_size=32,
        accumulation_steps=1,
        precision="auto",
        dropout=0.0,
        seed=DEFAULT_SEED,
        warmup_tokens=5_000_000,
        log_interval_steps=100,
        bias=False,
        weight_decay=0.0,
        n_layer=12,
        n_inner=1024,
        block_size=512,
        vocab_size=8192,
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def smoke_config(experiment: str = "A", **overrides: Any) -> ExperimentConfig:
    """Fast Colab/CPU sanity run. Processes a few thousand tokens, not 50M."""
    experiment = experiment.upper()
    variant = "baseline" if experiment == "A" else "midpoint"
    values: dict[str, Any] = dict(
        experiment=experiment,
        variant=variant,
        vocab_size=256,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_inner=128,
        block_size=32,
        micro_batch_size=4,
        token_budget=2_048,
        eval_token_budget=512,
        learning_rate=3e-4,
        warmup_tokens=128,
        dataset="synthetic",
        max_steps=4,
        run_name=f"smoke_{experiment}",
        reconstruction_check_interval=1,
        mode="smoke",
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def expected_param_count(config: ExperimentConfig) -> int:
    """Closed-form trainable parameter count for the implemented architecture."""
    d = config.n_embd
    inner = config.n_inner
    v = config.vocab_size
    t = config.block_size
    layers = config.n_layer
    linear_bias = 1 if config.bias else 0
    embeddings = v * d + t * d
    ln = 2 * d  # weight + bias
    attn = d * (3 * d) + linear_bias * (3 * d) + d * d + linear_bias * d
    mlp = d * inner + linear_bias * inner + inner * d + linear_bias * d
    per_layer = ln + attn + ln + mlp
    lm_head = 0 if config.tie_embeddings else v * d
    return embeddings + layers * per_layer + ln + lm_head


def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(spec)


def recommend_precision(device: torch.device | None = None) -> str:
    """fp32 is the assignment default for reconstruction fidelity.

    Colab T4 (sm 7.5) has fp16 tensor cores but not bf16. Ampere+ GPUs can
    use bf16. Do **not** trust ``torch.cuda.is_bf16_supported()`` on a T4:
    it returns True while bf16 matmuls are emulated and ~5× slower.
    """
    device = device or resolve_device()
    if device.type != "cuda":
        return "fp32"
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 8:
        return "bf16"
    return "fp16"


def amp_dtype_for_device(device: torch.device | None = None) -> torch.dtype | None:
    """16-bit dtype for *computing* F. Hidden states stay fp32 for subtraction."""
    device = device or resolve_device()
    if device.type != "cuda":
        return None
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def amp_context(device: torch.device | None = None, enabled: bool = True):
    """Autocast around F only. Reversible subtraction must stay in fp32."""
    from contextlib import nullcontext

    device = device or resolve_device()
    dtype = amp_dtype_for_device(device) if enabled else None
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


_AMP_STATE: dict[str, Any] = {"enabled": False, "device": None}


def set_amp_enabled(enabled: bool, device: torch.device | None = None) -> None:
    """Enable/disable autocast around residual functions for the current run."""
    _AMP_STATE["enabled"] = bool(enabled)
    if device is not None:
        _AMP_STATE["device"] = device


def f_amp_context():
    return amp_context(_AMP_STATE.get("device"), enabled=bool(_AMP_STATE.get("enabled")))


def new_grad_scaler(device: torch.device, precision: str):
    """float16 needs a GradScaler. bfloat16 and fp32 do not."""
    if device.type != "cuda" or precision != "fp16":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def set_seed(seed: int) -> None:
    seed = _non_negative_int("seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reversible reconstruction needs bitwise-deterministic attention.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_runtime_info(device: torch.device | None = None) -> dict[str, Any]:
    device = device or resolve_device()
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_total_memory_bytes": None,
        "gpu_capability": None,
        "recommended_precision": recommend_precision(device),
        "colab": _running_in_colab(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        index = device.index if device.type == "cuda" and device.index is not None else 0
        props = torch.cuda.get_device_properties(index)
        info["gpu_name"] = torch.cuda.get_device_name(index)
        info["gpu_total_memory_bytes"] = int(props.total_memory)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
    return info


def _running_in_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def require_gpu_for_full_run(mode: str, device: torch.device) -> None:
    """Full 50M-token experiments are intended for a Colab GPU.

    Smoke tests may run on CPU. Calling a full run on CPU is allowed but
    prints a hard warning through this helper; the caller decides whether
    to abort.
    """
    if mode != "full":
        return
    if device.type != "cuda":
        raise RuntimeError(
            "Full 50M-token experiments require a CUDA GPU (Google Colab: "
            "Runtime → Change runtime type → T4/L4/A100). Use --mode smoke "
            "for CPU checks, or pass --allow-cpu-full to override."
        )


def ensure_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    """Alias used by the training and reporting modules."""
    return ensure_output_dir(path)
