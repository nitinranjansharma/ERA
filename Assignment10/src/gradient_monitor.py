"""Global gradient-norm logging, clipping, and a controlled spike experiment.

The ordering "grad-norm moves before loss" is *not* universal. In this module
it is manufactured deterministically: at ``spike_step`` we inflate the
gradients after ``backward()`` and before ``optimizer.step()``. The loss at
that step is still the pre-update loss; the inflated update hits the loss on
the *next* step. Whether the same ordering appears in a real run depends on
the optimiser, clipping, and the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from models import TinyMLP
from safety import TrainingSafetyError, assert_loss_ok, warn_or_raise_zero_norm
from training_loop import calculate_grad_norm


@dataclass
class ClipResult:
    pre_clip_norm: float
    threshold: float
    scale: float
    post_clip_norm: float

    def as_dict(self) -> dict:
        return asdict(self)


def clip_grad_norm_(
    parameters,
    max_norm: float,
    *,
    missing: str = "skip",
) -> ClipResult:
    """Scale grads by min(1, C / global_norm). Zero norms are not inverted."""
    if max_norm < 0:
        raise TrainingSafetyError(f"clip threshold must be non-negative, got {max_norm}")
    parameters = list(parameters)
    pre = calculate_grad_norm(parameters, missing=missing)
    pre_f = float(pre.detach().cpu()) if torch.is_tensor(pre) else float(pre)
    if not torch.isfinite(torch.tensor(pre_f)):
        raise TrainingSafetyError(f"Non-finite pre-clip gradient norm: {pre_f}")
    if pre_f == 0.0:
        warn_or_raise_zero_norm(pre_f, strict=False)
        return ClipResult(pre_clip_norm=0.0, threshold=float(max_norm), scale=1.0, post_clip_norm=0.0)
    scale = min(1.0, float(max_norm) / pre_f)
    if scale < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(scale)
    post = calculate_grad_norm(parameters, missing=missing)
    post_f = float(post.detach().cpu())
    return ClipResult(
        pre_clip_norm=pre_f,
        threshold=float(max_norm),
        scale=scale,
        post_clip_norm=post_f,
    )


def run_spike_experiment(
    *,
    steps: int = 40,
    spike_step: int = 25,
    spike_factor: float = 50.0,
    lr: float = 0.05,
    batch_size: int = 16,
    in_features: int = 8,
    hidden: int = 16,
    clip_threshold: Optional[float] = 5.0,
    detection_norm_ratio: float = 5.0,
    seed: int = 42,
) -> dict:
    """Train a tiny MLP; at ``spike_step`` multiply all grads by ``spike_factor``.

    Detection rule (this run only):
      Let baseline_norm be the median pre-clip norm on steps [1, spike_step).
      The first qualifying step is the first t where
          pre_clip_norm[t] >= detection_norm_ratio * baseline_norm
      and the loss at t is still within 25% of the previous step's loss.
      We then check that loss[t+1] has moved by more than loss[t] did.
    """
    if not (1 <= spike_step < steps):
        raise ValueError("spike_step must be a training step strictly before the last step")
    g = torch.Generator().manual_seed(seed)
    model = TinyMLP(in_features, hidden, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    rows = []
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, in_features, generator=g)
        y = torch.sin(x.sum(dim=-1, keepdim=True))
        logits = model(x)
        loss = F.mse_loss(logits, y)
        assert_loss_ok(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == spike_step:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach().mul_(spike_factor)
        clip = clip_grad_norm_(model.parameters(), clip_threshold if clip_threshold is not None else float("inf"))
        opt.step()
        rows.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pre_clip_norm": clip.pre_clip_norm,
                "clip_threshold": clip.threshold,
                "scale": clip.scale,
                "post_clip_norm": clip.post_clip_norm,
                "lr": lr,
                "injected_spike": step == spike_step,
            }
        )

    pre_spike = [r["pre_clip_norm"] for r in rows if r["step"] < spike_step]
    baseline = float(torch.tensor(pre_spike).median()) if pre_spike else 0.0
    qualifying = None
    for i, row in enumerate(rows):
        if baseline > 0 and row["pre_clip_norm"] >= detection_norm_ratio * baseline:
            prev_loss = rows[i - 1]["loss"] if i > 0 else row["loss"]
            loss_still_quiet = abs(row["loss"] - prev_loss) <= 0.25 * max(abs(prev_loss), 1e-12)
            if loss_still_quiet:
                qualifying = row["step"]
                break

    next_loss_jump = None
    if qualifying is not None and qualifying < steps:
        here = rows[qualifying - 1]
        nxt = rows[qualifying]
        prev = rows[qualifying - 2] if qualifying >= 2 else here
        delta_here = abs(here["loss"] - prev["loss"])
        delta_next = abs(nxt["loss"] - here["loss"])
        next_loss_jump = {
            "step": nxt["step"],
            "loss": nxt["loss"],
            "delta_vs_qualifying": delta_next,
            "delta_at_qualifying": delta_here,
            "loss_moved_more_after": delta_next > delta_here,
        }

    leading = bool(
        qualifying == spike_step
        and next_loss_jump is not None
        and next_loss_jump["loss_moved_more_after"]
    )
    return {
        "rows": rows,
        "spike_step": spike_step,
        "spike_factor": spike_factor,
        "clip_threshold": clip_threshold,
        "baseline_pre_clip_norm": baseline,
        "detection_rule": (
            f"first step whose pre-clip norm is ≥ {detection_norm_ratio}× the "
            "median pre-spike norm, while the loss is still within 25% of the "
            "previous step; then compare the loss move on the following step"
        ),
        "first_qualifying_step": qualifying,
        "next_loss_jump": next_loss_jump,
        "norm_changed_before_loss_in_this_run": leading,
        "caveat": (
            "This ordering is a property of this controlled injection, not a "
            "universal training law. Clipping, adaptive optimisers, and data "
            "noise can hide or reverse it."
        ),
        "passed": qualifying is not None,
    }
