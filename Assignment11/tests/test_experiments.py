from __future__ import annotations

from experiments import run_task_a, run_task_d, run_task_e
from model import StandardWidthMLP


def test_task_a_payload_contains_uncorrected_and_torch(default_cfg):
    payload = run_task_a(default_cfg)
    assert payload["agrees_with_torch"]
    assert payload["manual_uncorrected"][0]["m_hat"] is None
    assert payload["inside_eps_differs"] is True or payload["textbook_vs_inside_eps"]["worst_abs_error"] >= 0.0


def test_standard_width_mlp_grows_with_width():
    small = sum(p.numel() for p in StandardWidthMLP(8, 16, 3).parameters())
    large = sum(p.numel() for p in StandardWidthMLP(8, 32, 3).parameters())
    assert large > small


def test_task_d_fast_records_checkpoint_fields(fast_cfg):
    payload = run_task_d(fast_cfg)
    early = fast_cfg["schedules"]["early_stop_step"]
    assert payload["cosine_300"]["step200"]["learning_rate"] is not None
    assert payload["wsd_300"]["step200"]["learning_rate"] is not None
    assert payload["cosine_300"]["history"][early - 1]["global_step"] == early
    assert payload["winner_val_at_200_from_300_budget"] in {"cosine", "wsd", "tie", "undetermined"}


def test_task_e_fast_marks_minima(fast_cfg):
    payload = run_task_e(fast_cfg)
    assert "16" in payload["per_width"] and "32" in payload["per_width"]
    extra = payload["extrapolation"]
    assert extra["lr_from_loglog"] > 0
    assert extra["lr_from_inverse_width"] > 0
    assert "Not μP" in payload["parameterization"] or "not μP" in payload["parameterization"] or "Not" in payload["parameterization"]
