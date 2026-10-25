"""Memory arithmetic, unit conversions, and the 8/32-GPU ladders."""

from __future__ import annotations

import tracemalloc

import pytest

from zero_sim.config import (
    DEFAULT_BYTES_PER_PARAM,
    DEFAULT_NUM_PARAMS,
    VALID_WORLD_SIZES,
    gpu_capacity_gib,
)
from zero_sim.memory import (
    bytes_per_parameter,
    bytes_to_gb,
    bytes_to_gib,
    gb_to_bytes,
    gib_to_bytes,
    memory_breakdown,
    memory_fit,
    memory_per_gpu_gib,
    round_gib,
)

EXPECTED_8 = {"dp": 447.0, "zero1": 153.7, "zero2": 104.8, "zero3": 55.9}
EXPECTED_32 = {"dp": 447.0, "zero1": 122.2, "zero2": 68.1, "zero3": 14.0}


def test_gb_gib_conversion():
    assert gb_to_bytes(1) == 1_000_000_000
    assert gib_to_bytes(1) == 1024**3
    assert bytes_to_gb(gb_to_bytes(2.5)) == pytest.approx(2.5)
    assert bytes_to_gib(gib_to_bytes(3.25)) == pytest.approx(3.25)
    assert bytes_to_gib(80 * 1_000_000_000) == pytest.approx(74.5058, rel=1e-5)
    assert gpu_capacity_gib(80.0) == pytest.approx(74.5, abs=0.01)


def test_assignment_byte_identity():
    assert DEFAULT_BYTES_PER_PARAM == 16
    assert DEFAULT_NUM_PARAMS == 30_000_000_000


def test_memory_formulas_match_assignment():
    n = 8
    assert bytes_per_parameter("dp", n) == pytest.approx(16.0)
    assert bytes_per_parameter("zero1", n) == pytest.approx(4 + 12 / n)
    assert bytes_per_parameter("zero2", n) == pytest.approx(2 + 14 / n)
    assert bytes_per_parameter("zero3", n) == pytest.approx(16 / n)
    n = 32
    assert bytes_per_parameter("zero1", n) == pytest.approx(4 + 12 / n)
    assert bytes_per_parameter("zero2", n) == pytest.approx(2 + 14 / n)
    assert bytes_per_parameter("zero3", n) == pytest.approx(16 / n)


def test_world_size_one_collapses_to_data_parallel():
    for strategy in ("dp", "zero1", "zero2", "zero3"):
        assert bytes_per_parameter(strategy, 1) == pytest.approx(16.0)


@pytest.mark.parametrize("world_size", VALID_WORLD_SIZES)
def test_formulas_are_monotonic_in_world_size(world_size: int):
    dp = memory_per_gpu_gib("dp", world_size)
    z1 = memory_per_gpu_gib("zero1", world_size)
    z2 = memory_per_gpu_gib("zero2", world_size)
    z3 = memory_per_gpu_gib("zero3", world_size)
    assert dp >= z1 >= z2 >= z3


def test_eight_and_thirty_two_gpu_memory_ladder():
    for strategy, expected in EXPECTED_8.items():
        assert round_gib(memory_per_gpu_gib(strategy, 8)) == expected
    for strategy, expected in EXPECTED_32.items():
        assert round_gib(memory_per_gpu_gib(strategy, 32)) == expected


def test_memory_fit_decisions_at_32_gpus():
    capacity = gpu_capacity_gib(80.0)
    dp_fit, dp_head = memory_fit("dp", 32)
    z1_fit, z1_head = memory_fit("zero1", 32)
    z2_fit, z2_head = memory_fit("zero2", 32)
    z3_fit, z3_head = memory_fit("zero3", 32)
    assert dp_fit is False and dp_head < 0
    assert z1_fit is False and z1_head < 0
    assert z2_fit is True and z2_head > 0
    assert z3_fit is True and z3_head > z2_head
    assert memory_per_gpu_gib("zero2", 32) < capacity
    assert memory_per_gpu_gib("zero1", 32) > capacity


def test_activations_can_turn_a_theoretical_fit_into_a_miss():
    fits_without, _ = memory_fit("zero2", 32, activation_gib=0.0)
    fits_with, headroom = memory_fit("zero2", 32, activation_gib=20.0)
    assert fits_without is True
    assert fits_with is False
    assert headroom < 0


def test_thirty_billion_analysis_does_not_allocate_giant_tensors():
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[1]
    value = memory_per_gpu_gib("zero3", 32, num_params=30_000_000_000)
    breakdown = memory_breakdown("dp", 8, num_params=30_000_000_000)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert value > 0 and breakdown.training_state_gib > 0
    assert (peak - before) < 5_000_000  # well under 5 MiB
