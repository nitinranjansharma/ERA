"""Selector simulation: floors hold, ceilings hold, starvation arm is measurable."""
from __future__ import annotations

import random

import pytest

from ledger import CEILING_MULT, FLOOR_SEQ, LANES, STAGES
from selector_sim import KEEP_FRAC, run_one, sample_from_mix, score, select, stage_target


@pytest.mark.parametrize("stage", [n for n, _ in STAGES] + ["S5 anneal"])
def test_v5_proxy_meets_floors(stage):
    r = run_one(stage, "v5", n=4000, seed=7, fallback="fail")
    assert r["ok"], r.get("error")
    for lane, fl in FLOOR_SEQ[stage].items():
        assert r["share"][lane] + 1.0 >= fl, (stage, lane, r["share"][lane], fl)


@pytest.mark.parametrize("stage", [n for n, _ in STAGES] + ["S5 anneal"])
def test_ceiling_1_25x(stage):
    r = run_one(stage, "v5", n=4000, seed=7, fallback="fail")
    assert r["ok"], r.get("error")
    for k in LANES:
        cap = r["target"][k] * CEILING_MULT
        # 2 pp slack for integer-token rounding on a 4000-token batch.
        assert r["share"][k] <= cap + 2.0, (k, r["share"][k], cap)


def test_english_no_floor_starves_indic_relative_to_v5():
    from selector_sim import sample_from_mix, stage_target
    rng = random.Random(7)
    mix = stage_target("S2 general")
    cand = sample_from_mix(mix, int(4000 / KEEP_FRAC), rng)
    ranked = sorted(cand, key=lambda c: score(c, "english", rng), reverse=True)
    kept = ranked[:4000]
    indic_nf = 100.0 * sum(1 for c in kept if c["lane"] == "indic") / len(kept)
    r = run_one("S2 general", "v5", n=4000, seed=7, fallback="fail")
    assert r["ok"], r.get("error")
    assert r["share"]["indic"] >= FLOOR_SEQ["S2 general"]["indic"] - 1.0
    # Unprotected english ranking must not be what sets the Indic share.
    assert r["share"]["indic"] + 1e-6 >= indic_nf


def test_fail_fallback_is_loud_when_supply_missing():
    rng = random.Random(0)
    # A candidate stream with no agentic tokens cannot meet an agentic floor.
    cand = [dict(id=i, lane="web", eligible=True, tokens=1.0) for i in range(400)]
    with pytest.raises(RuntimeError, match="floor"):
        select(cand, "S5 anneal", "v5", rng, batch_size=200, fallback="fail")
