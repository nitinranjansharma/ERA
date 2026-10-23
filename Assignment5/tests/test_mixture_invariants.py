"""Executable specification. A passing run is the evidence the README cites."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledger import (
    ANN_B, ANN_SHARE, CEILING_MULT, COMPOSER, EPOCH_CAP, FLOOR_SEQ, LANES,
    LONG_COMP, MAIN_B, MAIN_SHARE, NONNATIVE_CAP, RUN_FLOOR, STAGES, SUPPLY,
    TIERS_ANN, TIERS_MAIN, TOTAL_B, build,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ledger():
    return build()


def test_shares_sum_to_100():
    assert abs(sum(MAIN_SHARE.values()) - 100) < 1e-9
    assert abs(sum(ANN_SHARE.values()) - 100) < 1e-9
    assert abs(sum(COMPOSER.values()) - 100) < 1e-9
    assert abs(sum(LONG_COMP.values()) - 1.0) < 1e-9
    assert abs(sum(TIERS_MAIN.values()) - 100) < 1e-9
    assert abs(sum(TIERS_ANN.values()) - 100) < 1e-9
    assert abs(MAIN_B + ANN_B - TOTAL_B) < 1e-9


def test_stage_mixtures_sum_and_integrate(ledger):
    M = ledger["M"]
    W = [w for _, w in STAGES]
    for i, (name, _) in enumerate(STAGES):
        assert abs(sum(M[k][i] for k in LANES) - 100) < 0.05, name
        assert all(M[k][i] >= 0 for k in LANES)
    for k in LANES:
        integ = sum(M[k][i] * W[i] for i in range(4))
        assert abs(integ - MAIN_SHARE[k]) < 0.05, k


def test_effective_shares_sum_to_100(ledger):
    for s, eff in ledger["effective"].items():
        assert abs(sum(eff.values()) - 100) < 1e-6, s
        assert all(v >= -1e-9 for v in eff.values()), s


def test_lane_totals_reconcile(ledger):
    assert abs(sum(ledger["tot"].values()) - TOTAL_B) < 1e-6


def test_run_integral_floors(ledger):
    M = ledger["M"]
    W = [w for _, w in STAGES]
    for lane, fl in RUN_FLOOR.items():
        integ = sum(M[lane][i] * W[i] for i in range(4))
        assert integ + 1e-6 >= fl, (lane, integ, fl)


def test_per_batch_floors(ledger):
    for s, floors in FLOOR_SEQ.items():
        for lane, fl in floors.items():
            assert ledger["named"][s][lane] + 1e-6 >= fl, (s, lane)


def test_indic_cap_is_sequence_token_and_holds(ledger):
    tot = ledger["tot"]["indic"]
    nn_run = 100.0 * (ledger["indic_tot"]["C"] + ledger["indic_tot"]["D"]) / tot
    assert nn_run <= NONNATIVE_CAP + 1e-6
    for s, split in ledger["indic_per"].items():
        indic = sum(split.values())
        nn = 100.0 * (split["C"] + split["D"]) / indic
        assert nn <= NONNATIVE_CAP + 1e-6, (s, nn)


def test_reasoning_sft_holdout_is_disjoint():
    assert abs(SUPPLY["reason_sft"] + SUPPLY["reason_escrow"] + SUPPLY["reason_pt_main"]
               - SUPPLY["reason_total"]) < 1e-9
    assert SUPPLY["reason_sft"] > 0


def test_tier_a_is_a_partition_not_a_subset():
    assert abs(SUPPLY["indic_A_escrow"] + SUPPLY["indic_A_main"] - SUPPLY["indic_A"]) < 1e-6
    assert SUPPLY["indic_A_escrow"] > 0 and SUPPLY["indic_A_main"] > 0


def test_epoch_caps(ledger):
    tot = ledger["tot"]
    main_reason = sum(ledger["stage_tok"][s]["reason"] for s in ledger["stage_tok"] if s != "S5 anneal")
    assert main_reason / SUPPLY["reason_pt_main"] <= EPOCH_CAP + 1e-9
    assert ledger["stage_tok"]["S5 anneal"]["reason"] / SUPPLY["reason_escrow"] <= EPOCH_CAP + 1e-9
    main_A = sum(ledger["indic_per"][s]["A"] for s in ledger["indic_per"] if s != "S5 anneal")
    assert main_A / SUPPLY["indic_A_main"] <= EPOCH_CAP + 1e-9
    assert ledger["indic_per"]["S5 anneal"]["A"] / SUPPLY["indic_A_escrow"] <= EPOCH_CAP + 1e-9
    for t, key in [("B", "indic_B"), ("C", "indic_C"), ("D", "indic_D")]:
        assert ledger["indic_tot"][t] / SUPPLY[key] <= EPOCH_CAP + 1e-9, t
    for k in ("web", "code", "stem", "longctx"):
        assert tot[k] / SUPPLY[k] <= EPOCH_CAP + 1e-9, k


def test_longctx_overlay_is_inside_the_stage_budget(ledger):
    for s, row in ledger["stage_tok"].items():
        overlay_tokens = row["longctx"]
        folded = sum(row["longctx"] * frac for frac in LONG_COMP.values())
        assert abs(overlay_tokens - folded) < 1e-9
        assert overlay_tokens <= row["_pool"] + 1e-9


def test_samanantar_is_zero_and_english_is_not_indic():
    inv = ledger_inv()
    by = {d["id"]: d for d in inv["datasets"]}
    assert by["samanantar"]["unique_tokens_b"] == 0
    assert "indic" not in by["sangraha-verified-en"]["slots"]
    assert by["aon-v4-sft"]["slots"] == ["sft_reason"]


def ledger_inv():
    return json.loads((ROOT / "data" / "inventory.json").read_text())


def test_inventory_ids_unique():
    ids = [d["id"] for d in ledger_inv()["datasets"]]
    assert len(ids) == len(set(ids))


def test_agentic_shapes_sum():
    assert abs(SUPPLY["agentic_fc"] + SUPPLY["agentic_traj"] - SUPPLY["agentic"]) < 1e-9


def test_anneal_agentic_loss_floor(ledger):
    assert ledger["loss"]["S5 anneal"]["pct"]["agentic"] + 1e-6 >= 3.5


def test_build_does_not_raise():
    build()  # validate() raises SpecError on any broken invariant
