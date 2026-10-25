from __future__ import annotations

import math

import pytest

from adam_manual import ScalarAdam, closed_form_uncorrected_ratio, compare_records, run_manual_adam


def test_pytorch_form_matches_textbook_on_assignment_gradients():
    grads = [0.50, 0.40, 0.60, 0.45, 0.55]
    a = ScalarAdam(epsilon_placement="after_vhat").run(grads)
    b = ScalarAdam(epsilon_placement="pytorch_form").run(grads)
    cmp = compare_records(a, b)
    assert cmp["worst_abs_error"] < 1e-15


def test_inside_eps_is_a_different_algorithm_for_large_eps():
    grads = [0.5, 0.5, 0.5]
    a = ScalarAdam(eps=1e-4, epsilon_placement="after_vhat").run(grads)
    b = ScalarAdam(eps=1e-4, epsilon_placement="inside_vhat").run(grads)
    assert abs(a[0].delta_w - b[0].delta_w) > 0.0


def test_closed_form_ratio_matches_measured_constant_gradient():
    g = 0.5
    on = ScalarAdam().run([g] * 8)
    off = ScalarAdam(bias_correction=False).run([g] * 8)
    for a, b in zip(on, off):
        measured = abs(b.delta_w) / abs(a.delta_w)
        formula = closed_form_uncorrected_ratio(a.t, 0.9, 0.999)
        assert measured == pytest.approx(formula, rel=1e-6, abs=1e-8)


def test_step_index_is_one_based():
    rec = run_manual_adam([0.1])[0]
    assert rec.t == 1
    assert math.isfinite(rec.w)
