"""Configuration validation and strategy comparison coverage."""

from __future__ import annotations

import pytest

from zero_sim.config import SimulationConfig, validate_num_params, validate_world_size
from zero_sim.metrics import strategy_comparison_table
from zero_sim.memory import memory_fit


@pytest.mark.parametrize("bad", [0, -1, 3, 24, 128, 31])
def test_invalid_world_sizes(bad: int):
    with pytest.raises(ValueError):
        validate_world_size(bad)
    with pytest.raises(ValueError):
        SimulationConfig(world_size=bad)


@pytest.mark.parametrize("bad", [0, -1, -30_000_000_000])
def test_invalid_parameter_counts(bad: int):
    with pytest.raises(ValueError):
        validate_num_params(bad)
    with pytest.raises(ValueError):
        SimulationConfig(num_params=bad)


def test_non_integer_and_topology_validation():
    with pytest.raises(ValueError):
        validate_world_size(8.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_num_params(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SimulationConfig(world_size=32, gpus_per_node=12)
    with pytest.raises(ValueError):
        SimulationConfig(gpu_capacity_gb=0)
    with pytest.raises(ValueError):
        SimulationConfig(activation_gib=-0.5)


def test_comparison_table_includes_all_four_strategies():
    table = strategy_comparison_table(world_size=32)
    assert list(table["strategy_id"]) == ["dp", "zero1", "zero2", "zero3"]
    assert set(table["fit_status"]) == {"Fits (training state)", "Does not fit"}
    assert table.loc[table["strategy_id"] == "dp", "fit_status"].iloc[0] == "Does not fit"
    assert table.loc[table["strategy_id"] == "zero1", "fit_status"].iloc[0] == "Does not fit"
    assert table.loc[table["strategy_id"] == "zero2", "fit_status"].iloc[0] == "Fits (training state)"
    assert table.loc[table["strategy_id"] == "zero3", "fit_status"].iloc[0] == "Fits (training state)"


def test_memory_fit_with_runtime_overhead():
    fits, _ = memory_fit("zero3", 32, runtime_overhead_gib=2.0)
    assert fits is True
    fits_heavy, headroom = memory_fit("zero2", 32, runtime_overhead_gib=10.0)
    assert fits_heavy is False
    assert headroom < 0
