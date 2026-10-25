"""The ten prioritized acceptance tests from the Session 11 test plan."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from adam_manual import ScalarAdam, ScalarAdamNoBiasCorrection
from experiments import controlled_training_step, run_task_c, run_task_d, run_task_e
from metrics import NonFiniteError, assert_grads_finite, require_finite, update_to_weight_ratio
from model import InstrumentedMLP
from run_experiments import main as run_main
from schedules import make_schedule
from utils import load_config, set_seed


GRADIENTS = [0.50, 0.40, 0.60, 0.45, 0.55]
ATOL = 1e-10
RTOL = 1e-8


def _adam_kwargs():
    return dict(lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False)


def test_01_manual_adam_numerical_correctness():
    """Every intermediate and the float64 weight trajectory match torch.optim.Adam."""
    manual = ScalarAdam(w0=1.0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    try:
        reference = torch.optim.Adam([param], **_adam_kwargs())
    except TypeError:
        kwargs = _adam_kwargs()
        kwargs.pop("foreach", None)
        reference = torch.optim.Adam([param], **kwargs)

    beta1, beta2, lr, eps = 0.9, 0.999, 0.001, 1e-8
    m_prev, v_prev, w_prev = 0.0, 0.0, 1.0
    for t, gradient in enumerate(GRADIENTS, start=1):
        rec = manual.step(gradient)
        reference.zero_grad(set_to_none=True)
        param.grad = torch.tensor(gradient, dtype=torch.float64)
        reference.step()

        m = beta1 * m_prev + (1.0 - beta1) * gradient
        v = beta2 * v_prev + (1.0 - beta2) * gradient * gradient
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        delta = -lr * m_hat / (math.sqrt(v_hat) + eps)
        w = w_prev + delta

        assert rec.g == pytest.approx(gradient)
        assert rec.m == pytest.approx(m, rel=RTOL, abs=ATOL)
        assert rec.v == pytest.approx(v, rel=RTOL, abs=ATOL)
        assert rec.m_hat == pytest.approx(m_hat, rel=RTOL, abs=ATOL)
        assert rec.v_hat == pytest.approx(v_hat, rel=RTOL, abs=ATOL)
        assert rec.delta_w == pytest.approx(delta, rel=RTOL, abs=ATOL)
        assert rec.w == pytest.approx(w, rel=RTOL, abs=ATOL)
        torch.testing.assert_close(
            torch.tensor(rec.w, dtype=torch.float64),
            param.detach(),
            atol=ATOL,
            rtol=RTOL,
        )
        state = reference.state[param]
        assert rec.m == pytest.approx(float(state["exp_avg"]), rel=RTOL, abs=ATOL)
        assert rec.v == pytest.approx(float(state["exp_avg_sq"]), rel=RTOL, abs=ATOL)
        m_prev, v_prev, w_prev = m, v, w


def test_02_bias_correction_on_the_first_step():
    """Corrected first step is ~lr; uncorrected is ~3.16× larger for g=0.5."""
    g, lr, beta1, beta2, eps = 0.5, 1e-3, 0.9, 0.999, 1e-8
    on = ScalarAdam(w0=1.0, lr=lr, beta1=beta1, beta2=beta2, eps=eps).step(g)
    off = ScalarAdamNoBiasCorrection(w0=1.0, lr=lr, beta1=beta1, beta2=beta2, eps=eps).step(g)
    assert off.m_hat is None and off.v_hat is None
    assert abs(on.delta_w) == pytest.approx(lr, rel=1e-5, abs=1e-9)
    ratio = abs(off.delta_w) / abs(on.delta_w)
    assert ratio == pytest.approx((1.0 - beta1) / math.sqrt(1.0 - beta2), rel=1e-3)
    assert ratio == pytest.approx(3.16, rel=5e-3)


def test_03_zero_and_near_zero_gradients():
    opt = ScalarAdam(w0=1.0, lr=1e-3, eps=1e-8)
    rec = opt.step(0.0)
    assert rec.m == 0.0 and rec.v == 0.0
    assert rec.delta_w == pytest.approx(0.0)
    assert rec.w == pytest.approx(1.0)
    assert math.isfinite(rec.delta_w) and math.isfinite(rec.w)

    tiny = ScalarAdam(w0=1.0, lr=1e-3, eps=1e-8)
    small = tiny.step(1e-20)
    assert math.isfinite(small.m) and math.isfinite(small.v)
    assert math.isfinite(small.m_hat) and math.isfinite(small.v_hat)
    assert math.isfinite(small.delta_w) and math.isfinite(small.w)
    assert small.w != float("inf")


def test_04_warmup_boundary_and_schedule_behavior():
    peak = 3e-3
    warmup = 30
    spec = make_schedule("warmup_stable", total_steps=80, peak_lr=peak, warmup_steps=warmup)
    step_zero = spec.lr(0)
    midpoint = spec.lr(warmup // 2)
    boundary = spec.lr(warmup)
    after = [spec.lr(t) for t in (warmup + 1, warmup + 10, 80)]
    assert step_zero >= 0.0
    assert spec.lr(1) >= 0.0
    assert spec.lr(1) < midpoint < boundary
    assert boundary == pytest.approx(peak)
    assert spec.lr_peak_step() == warmup
    assert all(v == pytest.approx(peak) for v in after)
    assert all(spec.lr(t) >= 0.0 for t in range(0, 81))


def test_05_update_to_weight_ratio_correctness():
    before = torch.tensor([3.0, 4.0], dtype=torch.float64)
    after = torch.tensor([6.0, 8.0], dtype=torch.float64)
    delta = after - before
    manual_num = float(torch.linalg.vector_norm(delta))
    manual_den = float(torch.linalg.vector_norm(before))
    rec = update_to_weight_ratio(before, after, eps=1e-12, near_zero_eps=1e-8, name="weight")
    assert rec.update_norm == pytest.approx(manual_num)
    assert rec.param_norm == pytest.approx(manual_den)
    assert rec.ratio == pytest.approx(manual_num / manual_den)
    assert rec.excluded_from_aggregate is False

    bias_before = torch.zeros(3, dtype=torch.float64)
    bias_after = torch.tensor([0.2, -0.1, 0.0], dtype=torch.float64)
    bias = update_to_weight_ratio(bias_before, bias_after, eps=1e-12, near_zero_eps=1e-8, name="bias")
    expected = float(torch.linalg.vector_norm(bias_after - bias_before)) / 1e-12
    assert bias.near_zero_weight
    assert bias.excluded_from_aggregate
    assert bias.policy == "clamp_denom_and_flag"
    assert bias.ratio == pytest.approx(expected)


def test_06_reproducibility(fast_cfg):
    set_seed(fast_cfg["seed"], deterministic=True)
    first = run_task_c(fast_cfg)
    set_seed(fast_cfg["seed"], deterministic=True)
    second = run_task_c(fast_cfg)
    for a, b in zip(first["with_warmup"]["logs"], second["with_warmup"]["logs"]):
        assert a["training_loss"] == pytest.approx(b["training_loss"], rel=0, abs=1e-12)
        assert a["learning_rate"] == pytest.approx(b["learning_rate"], rel=0, abs=0.0)
        assert a["update_to_weight_ratio"] == pytest.approx(b["update_to_weight_ratio"], rel=0, abs=1e-12)
        assert a["gradient_norm"] == pytest.approx(b["gradient_norm"], rel=0, abs=1e-12)
    assert first["lr_peak_step"] == second["lr_peak_step"]
    assert first["loss_with_warmup"] == pytest.approx(second["loss_with_warmup"], rel=0, abs=1e-12)


def test_07_cosine_versus_wsd_checkpoint_integrity(fast_cfg, tmp_path):
    cfg = copy.deepcopy(fast_cfg)
    cfg["schedules"]["total_steps"] = 300
    cfg["schedules"]["early_stop_step"] = 200
    cfg["schedules"]["eval_every"] = 50
    cfg["schedules"]["warmup_fraction"] = 0.10
    cfg["schedules"]["wsd_decay_fraction"] = 0.20
    cfg["_checkpoint_dir"] = str(tmp_path / "ckpts")
    payload = run_task_d(cfg)
    for arm in ("cosine_300", "wsd_300"):
        run = payload[arm]
        assert run["history"][-1]["global_step"] == 300
        row200 = next(r for r in run["history"] if r["global_step"] == 200)
        snap = run["step200"]
        assert snap["checkpoint_saved"] is True
        assert snap["step"] == 200
        assert snap["training_loss"] == pytest.approx(row200["training_loss"])
        assert snap["learning_rate"] == pytest.approx(row200["learning_rate"])
        assert snap["validation_loss"] == pytest.approx(row200["validation_loss"])
        assert math.isfinite(snap["training_loss"])
        assert math.isfinite(snap["learning_rate"])
        path = Path(snap["checkpoint_path"])
        assert path.is_file()
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        assert blob["step"] == 200
        assert blob["train_loss"] == pytest.approx(snap["training_loss"])
        assert blob["learning_rate"] == pytest.approx(snap["learning_rate"])
        assert blob["state_dict"]
        assert snap["checkpoint"]["finite"] is True
        assert snap["checkpoint"]["n_params"] > 0
    assert payload["cosine_300"]["step200"]["learning_rate"] != pytest.approx(
        payload["wsd_300"]["step200"]["learning_rate"]
    )


def test_08_learning_rate_sweep_validity(fast_cfg):
    cfg = copy.deepcopy(fast_cfg)
    grid = [1.0e-4, 1.0e-2, 1.0e-1]
    cfg["width_sweep"]["widths"] = [256, 512, 1024]
    cfg["width_sweep"]["lr_grid"] = grid
    cfg["width_sweep"]["steps"] = 3
    cfg["width_sweep"]["batch_size"] = 8
    cfg["width_sweep"]["n_train"] = 32
    cfg["width_sweep"]["n_val"] = 16
    cfg["width_sweep"]["input_dim"] = 8
    cfg["width_sweep"]["n_classes"] = 4
    cfg["width_sweep"]["hidden_layers"] = 1
    cfg["width_sweep"]["refine"] = False
    cfg["width_sweep"]["max_expansions"] = 0
    payload = run_task_e(cfg)
    for width in (256, 512, 1024):
        block = payload["per_width"][str(width)]
        lrs = [cell["lr"] for cell in block["cells"]]
        assert lrs == pytest.approx(grid)
        finite = [c for c in block["cells"] if c.get("finite")]
        assert finite, f"width {width} produced no finite results"
        actual_best = min(finite, key=lambda c: c["val_loss"])
        reported = block["best"]
        assert reported["lr"] == pytest.approx(actual_best["lr"])
        assert reported["val_loss"] == pytest.approx(actual_best["val_loss"])
        idx = lrs.index(reported["lr"])
        expected_boundary = idx in {0, len(lrs) - 1}
        assert reported["on_boundary"] is expected_boundary


def test_09_numerical_stability_and_failure_detection():
    with pytest.raises(NonFiniteError, match="not finite"):
        require_finite(float("nan"), "training_loss")
    with pytest.raises(NonFiniteError, match="not finite"):
        require_finite(float("inf"), "training_loss")

    set_seed(0)
    model = InstrumentedMLP(6, 8, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(4, 6)
    y = torch.randint(0, 3, (4,))
    with pytest.raises(NonFiniteError, match="not finite"):
        controlled_training_step(model, x, y, opt, inject_nan_loss=True)

    model2 = InstrumentedMLP(6, 8, 3)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
    with pytest.raises(NonFiniteError, match="Non-finite"):
        controlled_training_step(model2, x, y, opt2, inject_inf_grad=True)

    p = nn.Parameter(torch.ones(3))
    p.grad = torch.tensor([1.0, float("inf"), 0.0])
    with pytest.raises(NonFiniteError, match="Non-finite"):
        assert_grads_finite([p])


def test_10_end_to_end_artifact_validation(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "fast.yaml"
    cfg = load_config(cfg_path)
    out = tmp_path / "artifacts"
    rc = run_main(["--config", str(cfg_path), "--tasks", "A,B,C,D,E", "--output-dir", str(out)])
    assert rc == 0
    run_dirs = sorted((out / "runs").glob("*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    results_path = run_dir / "results.json"
    config_path = run_dir / "config.json"
    assert results_path.is_file() and config_path.is_file()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["environment"]["seed"] == cfg["seed"]
    assert saved_cfg["seed"] == cfg["seed"]
    assert payload["A"]["agrees_with_torch"] is True
    assert math.isfinite(payload["A"]["manual_vs_torch"]["worst_abs_error"])
    for rec in payload["A"]["manual_corrected"]:
        for key in ("m", "v", "m_hat", "v_hat", "delta_w", "w"):
            assert math.isfinite(rec[key])
    required_plots = {
        "bias_correction.png",
        "update_to_weight_warmup.png",
        "cosine_vs_wsd.png",
        "lr_sweep_widths.png",
    }
    plot_names = {p.name for p in (run_dir / "plots").glob("*.png")}
    assert required_plots.issubset(plot_names)
    for name in required_plots:
        assert (run_dir / "plots" / name).stat().st_size > 0
    logs = run_dir / "logs"
    assert (logs / "adam_steps.csv").is_file()
    assert (logs / "bias_correction.csv").is_file()
    assert (logs / "update_to_weight.csv").is_file()
    assert (logs / "schedules.csv").is_file()
    assert (logs / "width_lr_sweep.csv").is_file()
    csv_lrs = (logs / "update_to_weight.csv").read_text(encoding="utf-8").splitlines()
    json_steps = [row["global_step"] for row in payload["C"]["with_warmup"]["logs"]]
    assert str(json_steps[0]) in csv_lrs[1]
    assert payload["D"]["cosine_300"]["history"][-1]["global_step"] == cfg["schedules"]["total_steps"]
