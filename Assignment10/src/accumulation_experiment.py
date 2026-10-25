"""Incorrect vs correct gradient accumulation with unequal valid-token counts.

Assignment micro-batches
------------------------
    MB1: 4 valid tokens, average loss 2.0
    MB2: 4 valid tokens, average loss 2.0
    MB3: 2 valid tokens, average loss 5.0   (plus padding that must be excluded)

    correct   = (4*2 + 4*2 + 2*5) / 10 = 2.6
    incorrect = (2 + 2 + 5) / 3        = 3.0
    discrepancy = |3.0 - 2.6| / 2.6   ≈ 15.3846%

Why averaging averages is not a token mean
------------------------------------------
The mean of per-micro-batch means weights every micro-batch equally, regardless
of how many valid tokens it contributed. The global token-mean weights every
*token* equally. Those coincide if and only if every micro-batch has the same
number of valid tokens (or the micro-batch means happen to be identical).

Backprop scaling
----------------
Correct:  backward(sum_valid(loss_i) / N_total) on each micro-batch, then step.
Incorrect: backward(mean_valid_in_mb / n_microbatches) on each micro-batch.

Gradients are computed with a real scalar parameter, not by evaluating the
two arithmetic expressions alone.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch

from models import ScalarTokenModel


EXPECTED_CORRECT = 2.6
EXPECTED_INCORRECT = 3.0
EXPECTED_DISCREPANCY = (3.0 - 2.6) / 2.6  # ≈ 0.153846


def arithmetic_losses(
    token_counts: Sequence[int] = (4, 4, 2),
    mb_means: Sequence[float] = (2.0, 2.0, 5.0),
) -> dict:
    if len(token_counts) != len(mb_means):
        raise ValueError("token_counts and mb_means must have the same length")
    total_tokens = sum(token_counts)
    if total_tokens == 0:
        raise ValueError("total valid tokens must be positive")
    correct = sum(n * m for n, m in zip(token_counts, mb_means)) / total_tokens
    incorrect = sum(mb_means) / len(mb_means)
    discrepancy = abs(incorrect - correct) / abs(correct)
    return {
        "token_counts": list(token_counts),
        "mb_means": list(mb_means),
        "total_tokens": total_tokens,
        "correct_loss": correct,
        "incorrect_loss": incorrect,
        "relative_discrepancy": discrepancy,
        "relative_discrepancy_percent": 100.0 * discrepancy,
    }


def make_microbatches(
    token_counts: Sequence[int] = (4, 4, 2),
    mb_means: Sequence[float] = (2.0, 2.0, 5.0),
    *,
    pad_to: int | None = None,
    dtype=torch.float64,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Each micro-batch is (coefficients, valid_mask).

    Padding tokens get coefficient 999 so that accidentally including them
    would move the loss far from 2.6/3.0 — tests catch that.
    """
    pad_to = pad_to if pad_to is not None else max(token_counts)
    batches = []
    for n, mean in zip(token_counts, mb_means):
        coef = torch.full((pad_to,), 999.0, dtype=dtype)
        mask = torch.zeros(pad_to, dtype=dtype)
        coef[:n] = float(mean)
        mask[:n] = 1.0
        batches.append((coef, mask))
    return batches


def _masked_sum(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (per_token * mask).sum()


def _masked_mean(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if float(denom) == 0:
        raise ValueError("micro-batch has zero valid tokens")
    return _masked_sum(per_token, mask) / denom


def incorrect_accumulation(model: ScalarTokenModel, batches) -> dict:
    """Average of micro-batch averages: every micro-batch gets equal weight."""
    model.zero_grad(set_to_none=True)
    n_mb = len(batches)
    mb_means = []
    for coef, mask in batches:
        per_token = model.per_token_loss(coef)
        mb_mean = _masked_mean(per_token, mask)
        mb_means.append(float(mb_mean.detach()))
        (mb_mean / n_mb).backward()
    grad = float(model.theta.grad.detach())
    loss = sum(mb_means) / n_mb
    return {"loss": loss, "grad": grad, "mb_means": mb_means}


def correct_accumulation(model: ScalarTokenModel, batches) -> dict:
    """Token-weighted mean: accumulate summed token losses, divide by N_total."""
    model.zero_grad(set_to_none=True)
    total_tokens = sum(int(mask.sum().item()) for _, mask in batches)
    if total_tokens == 0:
        raise ValueError("effective batch has zero valid tokens")
    summed = 0.0
    for coef, mask in batches:
        per_token = model.per_token_loss(coef)
        token_sum = _masked_sum(per_token, mask)
        summed += float(token_sum.detach())
        (token_sum / total_tokens).backward()
    grad = float(model.theta.grad.detach())
    loss = summed / total_tokens
    return {"loss": loss, "grad": grad, "total_tokens": total_tokens}


def concatenated_reference(model: ScalarTokenModel, batches) -> dict:
    """Single-batch token mean over the concatenated valid tokens — the intended objective."""
    ref = ScalarTokenModel(init=float(model.theta.detach()), dtype=model.theta.dtype)
    valid_coef = torch.cat([c[m.bool()] for c, m in batches])
    ref.zero_grad(set_to_none=True)
    per_token = ref.per_token_loss(valid_coef)
    loss = per_token.mean()
    loss.backward()
    return {"loss": float(loss.detach()), "grad": float(ref.theta.grad.detach()), "n_tokens": int(valid_coef.numel())}


def padded_tokens_included_loss(model: ScalarTokenModel, batches) -> float:
    """Bug: mean over the padded length, treating pad coefficients as real tokens."""
    losses = []
    for coef, _mask in batches:
        losses.append(float(model.per_token_loss(coef).mean().detach()))
    return sum(losses) / len(losses)


def running_reduction_curves(
    token_counts: Sequence[int] = (4, 4, 2),
    mb_means: Sequence[float] = (2.0, 2.0, 5.0),
) -> list[dict]:
    """Cumulative incorrect vs correct loss after 1, 2, … micro-batches.

    The two curves coincide until the short batch arrives; that is the gap.
    """
    rows = []
    for k in range(1, len(mb_means) + 1):
        a = arithmetic_losses(token_counts[:k], mb_means[:k])
        rows.append(
            {
                "microbatches_included": k,
                "tokens_so_far": a["total_tokens"],
                "token_counts": a["token_counts"],
                "incorrect_loss": a["incorrect_loss"],
                "correct_loss": a["correct_loss"],
                "gap": a["incorrect_loss"] - a["correct_loss"],
            }
        )
    return rows


def two_method_training_curves(
    token_counts: Sequence[int] = (4, 4, 2),
    mb_means: Sequence[float] = (2.0, 2.0, 5.0),
    *,
    steps: int = 25,
    lr: float = 0.05,
    theta_init: float = 1.0,
) -> dict:
    """SGD on the toy objective under each reduction. The gap is a pair of curves."""
    batches = make_microbatches(token_counts, mb_means)

    def run(method: str) -> list[dict]:
        model = ScalarTokenModel(init=theta_init)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        rows = []
        for step in range(1, steps + 1):
            if method == "incorrect":
                out = incorrect_accumulation(model, batches)
            else:
                out = correct_accumulation(model, batches)
            opt.step()
            rows.append(
                {
                    "step": step,
                    "loss": out["loss"],
                    "grad": out["grad"],
                    "theta": float(model.theta.detach()),
                }
            )
        return rows

    return {"incorrect": run("incorrect"), "correct": run("correct"), "lr": lr}


def run_accumulation_experiment(
    token_counts: Sequence[int] = (4, 4, 2),
    mb_means: Sequence[float] = (2.0, 2.0, 5.0),
    *,
    equal_token_count: int = 4,
    theta_init: float = 1.0,
) -> dict:
    arith = arithmetic_losses(token_counts, mb_means)
    batches = make_microbatches(token_counts, mb_means)

    incorrect_model = ScalarTokenModel(init=theta_init)
    correct_model = ScalarTokenModel(init=theta_init)
    inc = incorrect_accumulation(incorrect_model, batches)
    cor = correct_accumulation(correct_model, batches)
    ref = concatenated_reference(deepcopy(correct_model), batches)

    # Equal-token control: three micro-batches of the same valid length.
    eq_counts = (equal_token_count,) * len(mb_means)
    eq_arith = arithmetic_losses(eq_counts, mb_means)
    eq_batches = make_microbatches(eq_counts, mb_means)
    eq_inc = incorrect_accumulation(ScalarTokenModel(init=theta_init), eq_batches)
    eq_cor = correct_accumulation(ScalarTokenModel(init=theta_init), eq_batches)

    pad_bug_loss = padded_tokens_included_loss(ScalarTokenModel(init=theta_init), batches)

    running = running_reduction_curves(token_counts, mb_means)
    trajectories = two_method_training_curves(
        token_counts, mb_means, steps=25, lr=0.05, theta_init=theta_init
    )

    loss_discrepancy = abs(inc["loss"] - cor["loss"]) / abs(cor["loss"])
    grad_discrepancy = abs(inc["grad"] - cor["grad"]) / abs(cor["grad"]) if cor["grad"] != 0 else float("inf")
    eq_loss_discrepancy = abs(eq_inc["loss"] - eq_cor["loss"]) / abs(eq_cor["loss"]) if eq_cor["loss"] != 0 else 0.0

    passed = (
        abs(cor["loss"] - EXPECTED_CORRECT) < 1e-12
        and abs(inc["loss"] - EXPECTED_INCORRECT) < 1e-12
        and abs(loss_discrepancy - EXPECTED_DISCREPANCY) < 1e-10
        and abs(cor["grad"] - ref["grad"]) < 1e-12
        and abs(cor["loss"] - ref["loss"]) < 1e-12
        and eq_loss_discrepancy < 1e-12
        and abs(pad_bug_loss - cor["loss"]) > 1.0
    )
    return {
        "arithmetic": arith,
        "incorrect": inc,
        "correct": cor,
        "reference_concatenated": ref,
        "equal_token": {
            "counts": list(eq_counts),
            "arithmetic": eq_arith,
            "incorrect": eq_inc,
            "correct": eq_cor,
            "relative_discrepancy": eq_loss_discrepancy,
        },
        "running_curves": running,
        "training_trajectories": trajectories,
        "pad_included_loss": pad_bug_loss,
        "loss_relative_discrepancy": loss_discrepancy,
        "grad_relative_discrepancy": grad_discrepancy,
        "explanation": (
            "Averaging micro-batch averages weights batches, not tokens. "
            "With counts (4, 4, 2) the last batch's mean of 5.0 is over-weighted, "
            "producing 3.0 instead of the token mean 2.6. When every micro-batch "
            "has the same valid-token count the two reductions coincide."
        ),
        "passed": passed,
    }
