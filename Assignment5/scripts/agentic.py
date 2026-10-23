"""Size the agentic synthesis programme in samples, not token wishes."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import EPOCH_CAP, LOSS_FRAC, MAIN_SHARE, ANN_SHARE, MAIN_B, ANN_B, SUPPLY

FC = {
    "Glaive v2": (113_000, 50e6),
    "ToolBench": (120_000, 80e6),
    "ToolACE": (110_000, 60e6),
    "Nexus/NexusRaven": (40_000, 30e6),
    "xLAM/APIGen": (60_000, 25e6),
    "Hermes FC": (15_000, 22e6),
}
TRAJ = {
    "SWE-Gym": (2_400, 150e6),
    "SWE-smith": (26_000, 120e6),
    "OpenHands rollouts": (10_000, 90e6),
}


def report(name, d):
    s = sum(v[0] for v in d.values())
    t = sum(v[1] for v in d.values())
    print(f"\n{name}: {s:,} samples · {t/1e6:.0f}M tokens · mean {t/s:,.0f} tok/sample")
    for k, (ns, nt) in sorted(d.items(), key=lambda x: -x[1][1] / x[1][0]):
        print(f"   {k:<22}{ns:>9,} samples {nt/1e6:>7.0f}M tok   {nt/ns:>8,.0f} tok/sample")
    return s, t


def main():
    print("=" * 80)
    print("EXISTING PUBLIC AGENTIC SUPPLY, split by shape  (stated: course inventory)")
    print("=" * 80)
    fc_s, fc_t = report("Short function-calling", FC)
    tr_s, tr_t = report("Long trajectories", TRAJ)
    print(f"\n   TOTAL {(fc_t+tr_t)/1e6:.0f}M tokens — the entire public agentic lane")
    assert abs((fc_t + tr_t) / 1e9 - SUPPLY["agentic"]) < 1e-6

    main_demand = MAIN_SHARE["agentic"] / 100 * MAIN_B * 1e9
    ann_demand = ANN_SHARE["agentic"] / 100 * ANN_B * 1e9
    yield_ = LOSS_FRAC["agentic"]

    print(f"\n{'='*80}\nSYNTHESIS TARGETS at the {EPOCH_CAP:.0f}-epoch ceiling\n{'='*80}")
    for label, demand, have_t, mean_tok, kind in [
        ("MAIN RUN  (function-calling)", main_demand, fc_t, fc_t / fc_s, "calls"),
        ("ANNEAL    (trajectories)", ann_demand, tr_t, tr_t / tr_s, "trajectories"),
    ]:
        unique = demand / EPOCH_CAP
        gap = unique - have_t
        print(f"\n{label}")
        print(f"   demand           {demand/1e9:>8.1f}B seen tokens  "
              f"({demand*yield_/1e9:.1f}B supervised at {100*yield_:.0f}% yield — STATED)")
        print(f"   unique needed    {unique/1e9:>8.1f}B   (repeat {EPOCH_CAP:.0f}x)")
        print(f"   have             {have_t/1e9:>8.3f}B")
        print(f"   MUST SYNTHESIZE  {gap/1e9:>8.1f}B  = {gap/have_t:>5.1f}x existing public supply of this shape")
        print(f"   => {gap/mean_tok:>12,.0f} new {kind} at the observed mean of {mean_tok:,.0f} tok each")
        print("   QA gate: executed tool returns, verifier labels, correct masks, decontaminated vs SWE-bench.")


if __name__ == "__main__":
    main()
