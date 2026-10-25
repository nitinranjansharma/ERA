from __future__ import annotations

import pytest

from schedules import make_schedule


def test_warmup_completes_exactly_at_warmup_steps():
    spec = make_schedule("warmup_stable", total_steps=80, peak_lr=0.002, warmup_steps=30)
    assert spec.lr(29) < spec.peak_lr
    assert spec.lr(30) == spec.peak_lr
    assert spec.lr(80) == spec.peak_lr


def test_cosine_and_wsd_diverge_at_step_200_of_300():
    cosine = make_schedule("cosine", 300, 1e-3, warmup_fraction=0.1, min_lr_fraction=0.1)
    wsd = make_schedule("wsd", 300, 1e-3, warmup_fraction=0.1, decay_fraction=0.2, min_lr_fraction=0.1)
    assert wsd.lr(200) > cosine.lr(200)
    assert cosine.mean_lr(until=300) < wsd.mean_lr(until=300)


def test_wsd_designed_for_200_has_started_decay_by_200():
    wsd = make_schedule("wsd", 200, 1e-3, warmup_fraction=0.1, decay_fraction=0.2, min_lr_fraction=0.1)
    assert wsd.lr(200) == pytest.approx(wsd.min_lr)
    assert wsd.lr(int(0.5 * 200)) == wsd.peak_lr
