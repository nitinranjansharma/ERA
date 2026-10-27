"""Dense-to-MoE conversion (sparse upcycling).

Default strategy ``copy`` writes the trained dense feed-forward weights into
every expert. Identical clones do **not** create specialization by themselves:
the router and the experts have to keep training. They can, however, preserve
the dense block's output for any normalized top-k weights, because

    sum_k w_k * Expert(x) = (sum_k w_k) * FFN(x) = FFN(x)

when every expert equals FFN and the selected weights sum to 1.

``perturb`` copies first, then adds seeded Gaussian noise so the experts start
slightly broken-symmetric. ``partition`` and ``drop-upcycling`` are documented
in the chapter but are not implemented.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import nn

from src.config import ExperimentConfig
from src.dense_model import (
    DenseRegressor,
    StrictLinearRegressor,
    count_parameters,
    dense_block_module,
)
from src.moe_model import MoERegressor, StrictLinearMoE, build_moe_model, parameter_report


class InitializationStrategy(Protocol):
    """Strategy interface for expert initialization after the empty MoE is built."""

    name: str

    def initialize_expert(
        self,
        expert: nn.Module,
        source_block: nn.Module,
        expert_index: int,
        generator: torch.Generator,
    ) -> None: ...


class CopyStrategy:
    """Copy every source parameter into the expert. Independent storage."""

    name = "copy"

    def initialize_expert(
        self,
        expert: nn.Module,
        source_block: nn.Module,
        expert_index: int,
        generator: torch.Generator,
    ) -> None:
        _copy_state(expert, source_block)


class PerturbStrategy:
    """Copy, then add N(0, std) noise to every expert parameter."""

    name = "perturb"

    def __init__(self, std: float) -> None:
        if std < 0:
            raise ValueError(f"perturb std must be >= 0, got {std}")
        self.std = std

    def initialize_expert(
        self,
        expert: nn.Module,
        source_block: nn.Module,
        expert_index: int,
        generator: torch.Generator,
    ) -> None:
        _copy_state(expert, source_block)
        if self.std == 0:
            return
        with torch.no_grad():
            for param in expert.parameters():
                noise = torch.randn(
                    param.shape,
                    dtype=param.dtype,
                    generator=generator,
                ) * self.std
                param.add_(noise.to(device=param.device))


STRATEGIES = ("copy", "perturb")


def get_strategy(name: str, perturb_std: float = 0.02) -> InitializationStrategy:
    if name == "copy":
        return CopyStrategy()
    if name == "perturb":
        return PerturbStrategy(std=perturb_std)
    if name in {"partition", "drop_upcycling", "drop-upcycling"}:
        raise ValueError(
            f"initialization_strategy={name!r} is described in the Session 14 chapter "
            "but is not implemented. Use 'copy' or 'perturb'."
        )
    raise ValueError(f"Unknown initialization strategy {name!r}. Available: {STRATEGIES}")


def _copy_state(destination: nn.Module, source: nn.Module) -> None:
    source_state = source.state_dict()
    dest_state = destination.state_dict()
    missing = set(source_state) - set(dest_state)
    extra = set(dest_state) - set(source_state)
    if missing or extra:
        raise RuntimeError(
            "Cannot copy dense block into expert; state-dict keys do not match. "
            f"missing_in_expert={sorted(missing)} extra_in_expert={sorted(extra)}"
        )
    cloned = {}
    for key, tensor in source_state.items():
        target = dest_state[key]
        if tuple(tensor.shape) != tuple(target.shape):
            raise RuntimeError(
                f"Shape mismatch copying {key!r}: source {tuple(tensor.shape)} vs "
                f"expert {tuple(target.shape)}"
            )
        cloned[key] = tensor.detach().clone()
    destination.load_state_dict(cloned)


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone() for key, tensor in module.state_dict().items()}


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def _init_router(
    router: nn.Module,
    std: float,
    generator: torch.Generator,
    bias_noise: float = 0.0,
) -> None:
    gate = router.gate
    with torch.no_grad():
        weight = torch.randn(gate.weight.shape, dtype=gate.weight.dtype, generator=generator) * std
        gate.weight.copy_(weight.to(device=gate.weight.device))
        if gate.bias is not None:
            gate.bias.zero_()
        if hasattr(router, "expert_bias"):
            router.expert_bias.zero_()
            if bias_noise > 0:
                noise = torch.randn(
                    router.expert_bias.shape, dtype=router.expert_bias.dtype, generator=generator
                ) * bias_noise
                router.expert_bias.add_(noise.to(device=router.expert_bias.device))


def _expert_dest(dense_model: nn.Module, expert: nn.Module) -> nn.Module:
    return expert if isinstance(dense_model, DenseRegressor) else expert.linear


def _scale_output_projection(module: nn.Module, scale: float) -> None:
    """Halve the last linear map so a shared expert plus routed mix still equals F."""
    if scale == 1.0:
        return
    with torch.no_grad():
        if hasattr(module, "fc2") and isinstance(module.fc2, nn.Linear):
            module.fc2.weight.mul_(scale)
            if module.fc2.bias is not None:
                module.fc2.bias.mul_(scale)
            return
        linear = module if isinstance(module, nn.Linear) else getattr(module, "linear", None)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Cannot scale output projection of {type(module)!r}")
        linear.weight.mul_(scale)
        if linear.bias is not None:
            linear.bias.mul_(scale)


def _copy_shared_layers(dense: nn.Module, moe: nn.Module) -> list[str]:
    copied: list[str] = []
    if isinstance(dense, DenseRegressor) and isinstance(moe, MoERegressor):
        moe.input_proj.load_state_dict(
            {k: v.detach().clone() for k, v in dense.input_proj.state_dict().items()}
        )
        moe.output_proj.load_state_dict(
            {k: v.detach().clone() for k, v in dense.output_proj.state_dict().items()}
        )
        copied.extend(["input_proj", "output_proj"])
        return copied
    if isinstance(dense, StrictLinearRegressor) and isinstance(moe, StrictLinearMoE):
        return copied
    raise TypeError(
        f"Architecture mismatch: dense={type(dense).__name__}, moe={type(moe).__name__}"
    )


def _source_for_expert(dense: nn.Module) -> nn.Module:
    if isinstance(dense, DenseRegressor):
        return dense.ffn
    if isinstance(dense, StrictLinearRegressor):
        # ExpertLinear.state_dict keys are ``linear.weight`` / ``linear.bias``,
        # matching the dense module's ``linear`` submodule, not the wrapper.
        return dense.linear
    raise TypeError(f"Unsupported dense model {type(dense)!r}")


@dataclass
class ConversionReport:
    strategy: str
    num_experts: int
    top_k: int
    architecture: str
    dense_unchanged: bool
    independent_storage: bool
    all_copied_exactly: bool
    parameter_counts: dict[str, Any]
    copy_checks: list[dict[str, Any]] = field(default_factory=list)
    shared_layers_copied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    output_scale: float = 1.0
    used_shared_expert: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "architecture": self.architecture,
            "dense_unchanged": self.dense_unchanged,
            "independent_storage": self.independent_storage,
            "all_copied_exactly": self.all_copied_exactly,
            "parameter_counts": self.parameter_counts,
            "copy_checks": self.copy_checks,
            "shared_layers_copied": self.shared_layers_copied,
            "notes": self.notes,
            "output_scale": self.output_scale,
            "used_shared_expert": self.used_shared_expert,
        }


def upcycle_dense_to_moe(
    dense_model: nn.Module,
    config: ExperimentConfig,
    initialization_strategy: str | None = None,
    seed: int | None = None,
) -> tuple[nn.Module, ConversionReport]:
    """Convert a trained dense model into an MoE without mutating the dense model.

    The original dense parameters are snapshotted before conversion. Experts
    receive cloned tensors, never views of the dense weights.
    """
    strategy_name = initialization_strategy or config.initialization_strategy
    strategy = get_strategy(strategy_name, perturb_std=config.perturb_std)
    seed = config.seed if seed is None else seed

    _validate_architecture(dense_model, config)
    before = _snapshot(dense_model)

    device = next(dense_model.parameters()).device
    moe = build_moe_model(config).to(device)

    shared_copied = _copy_shared_layers(dense_model, moe)
    source_block = _source_for_expert(dense_model)
    generator = torch.Generator()
    generator.manual_seed(seed)

    copy_checks: list[dict[str, Any]] = []
    independent = True
    exact_copies = True

    def _record(dest_block: nn.Module, expert_index: int | str) -> None:
        nonlocal independent, exact_copies
        for key, source_tensor in source_block.state_dict().items():
            dest_tensor = dest_block.state_dict()[key]
            diff = _max_abs_diff(dest_tensor, source_tensor)
            same_storage = dest_tensor.data_ptr() == source_tensor.data_ptr()
            if same_storage:
                independent = False
            expected_zero = strategy_name == "copy"
            if expected_zero and diff > 0:
                exact_copies = False
            copy_checks.append(
                {
                    "expert": expert_index,
                    "parameter": key,
                    "max_abs_diff": diff,
                    "independent_storage": not same_storage,
                    "matches_source": diff == 0.0,
                }
            )

    for expert_index, expert in enumerate(moe.experts):
        dest_block = _expert_dest(dense_model, expert)
        strategy.initialize_expert(dest_block, source_block, expert_index, generator)
        _record(dest_block, expert_index)

    shared_mod = getattr(moe, "shared_expert", None)
    if shared_mod is not None:
        dest_shared = _expert_dest(dense_model, shared_mod)
        strategy.initialize_expert(dest_shared, source_block, -1, generator)
        _record(dest_shared, "shared")

    output_scale = 0.5 if config.use_shared_expert else 1.0
    if output_scale != 1.0:
        for expert in moe.experts:
            _scale_output_projection(_expert_dest(dense_model, expert), output_scale)
        if shared_mod is not None:
            _scale_output_projection(_expert_dest(dense_model, shared_mod), output_scale)

    _init_router(
        moe.router,
        std=config.router_init_std,
        generator=generator,
        bias_noise=config.bias_init_noise,
    )

    after = dense_model.state_dict()
    dense_unchanged = all(torch.equal(before[k], after[k]) for k in before)

    notes = [
        "Identical copied experts preserve the dense block output for any "
        "normalized top-k weights; they do not by themselves create specialization.",
        "Router weights are drawn from N(0, router_init_std) with the conversion seed.",
        "partition and drop-upcycling are chapter concepts, not implemented here.",
        (
            "A shared expert would compute F+F=2F, so every expert's output projection "
            "is scaled by 0.5 when use_shared_expert=True: 0.5 F + 0.5 F = F."
            if config.use_shared_expert
            else "No always-on shared expert in this conversion (use_shared_expert=False)."
        ),
    ]
    if strategy_name == "perturb":
        notes.append(
            f"Experts were copied then perturbed with N(0, {config.perturb_std}); "
            "dense-vs-MoE outputs will differ before fine-tuning."
        )
        exact_copies = False

    report = ConversionReport(
        strategy=strategy_name,
        num_experts=config.num_experts,
        top_k=config.top_k,
        architecture=config.architecture,
        dense_unchanged=dense_unchanged,
        independent_storage=independent,
        all_copied_exactly=exact_copies and strategy_name == "copy",
        parameter_counts=parameter_report(dense_model, moe),
        copy_checks=copy_checks,
        shared_layers_copied=shared_copied,
        notes=notes,
        output_scale=output_scale,
        used_shared_expert=bool(config.use_shared_expert),
    )
    if not dense_unchanged:
        raise RuntimeError("Upcycling mutated the original dense model; aborting.")
    if not independent:
        raise RuntimeError("Expert parameters share storage with the dense block; aborting.")
    return moe, report


def _validate_architecture(dense_model: nn.Module, config: ExperimentConfig) -> None:
    if config.architecture == "ffn" and not isinstance(dense_model, DenseRegressor):
        raise TypeError("architecture='ffn' requires a DenseRegressor")
    if config.architecture == "strict_linear" and not isinstance(
        dense_model, StrictLinearRegressor
    ):
        raise TypeError("architecture='strict_linear' requires a StrictLinearRegressor")
    block = dense_block_module(dense_model)
    if isinstance(dense_model, DenseRegressor):
        if dense_model.hidden_dim != config.hidden_dim:
            raise ValueError("Dense hidden_dim does not match config.hidden_dim")
        if dense_model.ffn.ffn_hidden_dim != config.ffn_hidden_dim:
            raise ValueError("Dense FFN width does not match config.ffn_hidden_dim")
    _ = block
    _ = count_parameters(dense_model)


def clone_model(model: nn.Module) -> nn.Module:
    """Deep copy a module tree. Used so ablation runs do not share parameters."""
    return deepcopy(model)


@torch.no_grad()
def conversion_invariance_report(
    dense_model: nn.Module,
    moe_model: nn.Module,
    features: torch.Tensor,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    """Check that upcycling preserved the dense function, even after scrambling the router.

    If every routed (and shared) expert is a scaled copy of the dense FFN and selected
    weights sum to 1, which experts the router picks cannot change the output.
    """
    dense_model.eval()
    moe_model.eval()
    device = next(dense_model.parameters()).device
    features = features.to(device)
    dense_out = dense_model(features)
    moe_out = moe_model(features)
    if isinstance(moe_out, tuple):
        moe_out = moe_out[0]
    pre = (moe_out - dense_out).abs()

    router = moe_model.router
    original = router.gate.weight.detach().clone()
    original_bias = None if router.gate.bias is None else router.gate.bias.detach().clone()
    router.gate.weight.normal_(0.0, 1.0)
    if router.gate.bias is not None:
        router.gate.bias.normal_(0.0, 1.0)
    scrambled = moe_model(features)
    if isinstance(scrambled, tuple):
        scrambled = scrambled[0]
    post = (scrambled - dense_out).abs()
    router.gate.weight.copy_(original)
    if original_bias is not None:
        router.gate.bias.copy_(original_bias)

    return {
        "max_abs_diff": float(pre.max().item()),
        "mean_abs_diff": float(pre.mean().item()),
        "scrambled_max_abs_diff": float(post.max().item()),
        "scrambled_mean_abs_diff": float(post.mean().item()),
        "preserved": bool(torch.allclose(moe_out, dense_out, rtol=rtol, atol=atol)),
        "router_irrelevant": bool(torch.allclose(scrambled, dense_out, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
    }
