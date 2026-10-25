"""Phase 9 failure-mode tests. Invalid states must not be silently sanitised."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gradient_monitor import clip_grad_norm_
from instrumentation import resolve_device, set_seed
from models import TinyGPT, TinyMLP
from safety import (
    MFUError,
    PrecisionError,
    TrainingSafetyError,
    assert_expected_grads,
    assert_grads_finite,
    assert_params_finite,
    assert_reduction,
    assert_supported_precision,
)
from training_loop import calculate_grad_norm, load_checkpoint, save_checkpoint, training_step


def test_nan_gradients_raise():
    p = nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([1.0, float("nan")])
    with pytest.raises(TrainingSafetyError, match="Non-finite gradients"):
        assert_grads_finite([p])


def test_inf_gradients_raise():
    p = nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([float("inf"), 0.0])
    with pytest.raises(TrainingSafetyError, match="Non-finite gradients"):
        assert_grads_finite([p])


def test_extremely_large_gradients_are_clipped_not_hidden():
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([1e12, 1e12])
    pre = float(calculate_grad_norm([p]))
    result = clip_grad_norm_([p], max_norm=1.0)
    assert pre > 1e12
    assert result.pre_clip_norm == pre
    assert result.scale < 1.0
    assert result.post_clip_norm <= 1.0 + 1e-5
    # The pre-clip value is preserved in the log; we do not replace it with the clipped one.


def test_missing_gradients_on_trainable_params():
    model = TinyMLP()
    # No backward: every grad is None.
    with pytest.raises(TrainingSafetyError, match="missing gradients"):
        assert_expected_grads(model)


def test_non_finite_parameters_after_update():
    p = nn.Parameter(torch.tensor([float("nan")]))
    with pytest.raises(TrainingSafetyError, match="Non-finite parameters"):
        assert_params_finite([p])


def test_incorrect_loss_reduction():
    with pytest.raises(TrainingSafetyError, match="Unsupported loss reduction"):
        assert_reduction("average")  # the correct names are mean/sum/none


def test_unsupported_precision_on_cpu():
    with pytest.raises(PrecisionError, match="float16"):
        assert_supported_precision("fp16", torch.device("cpu"))
    with pytest.raises(PrecisionError, match="FP8"):
        assert_supported_precision("fp8_e4m3fn", torch.device("cpu"))
    with pytest.raises(PrecisionError, match="Unsupported precision"):
        assert_supported_precision("fp4", torch.device("cpu"))


def test_unsupported_device():
    with pytest.raises(RuntimeError, match="Unsupported device"):
        resolve_device("tpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_checkpoint_shape_mismatch(tmp_path):
    small = TinyMLP(in_features=4, hidden=8, out_features=1)
    large = TinyMLP(in_features=8, hidden=16, out_features=1)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(small, path)
    with pytest.raises(TrainingSafetyError, match="mismatch"):
        load_checkpoint(large, path)


def test_non_reproducible_config_rejected():
    with pytest.raises(ValueError, match="seed is required"):
        set_seed(None)  # type: ignore[arg-type]


def test_zero_grad_norm_logged_not_converted():
    p = nn.Parameter(torch.ones(3))
    p.grad = torch.zeros(3)
    norm = float(calculate_grad_norm([p]))
    assert norm == 0.0
    result = clip_grad_norm_([p], 1.0)
    assert result.pre_clip_norm == 0.0
    assert result.scale == 1.0  # must not divide by zero or invent a 1-norm


def test_mfu_does_not_invent_peak():
    from mfu import mfu_percent

    with pytest.raises(MFUError):
        mfu_percent(1.0, -7_912)


@pytest.mark.gpu
def test_gpu_marker_is_skippable():
    assert torch.cuda.is_available() or torch.backends.mps.is_available()
