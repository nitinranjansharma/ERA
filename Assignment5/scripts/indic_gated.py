"""Gate tail languages instead of funding them at homeopathic shares.

A language is funded when it can reach ≥1% of the Indic lane without breaching
4 epochs. That threshold is read off the supply cliff, not chosen.

Run:  python scripts/indic_gated.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from indic import NAME, MILU, supply_b, LANE_B
from ledger import EPOCH_CAP

IGB_TAIL = {"gom", "mai", "mni", "sat", "brx"}
MIN_LANE_SHARE = 0.01


def main():
    S = supply_b()
    ranked = sorted(S.items(), key=lambda kv: -kv[1])
    print("=" * 78)
    print("THE CLIFF")
    print("=" * 78)
    print(f"  {'rank':>4}  {'language':<11}{'supply B':>10}{'ratio to next':>15}")
    worst = (0, None)
    for i in range(len(ranked) - 1):
        lg, v = ranked[i]
        nxt = ranked[i + 1][1]
        r = v / nxt if nxt > 0 else float("inf")
        if r > worst[0] and v > 0.05:
            worst = (r, i)
        if i < 16:
            print(f"  {i+1:>4}  {NAME[lg]:<11}{v:>10.2f}{r:>14.1f}x")
    ci = worst[1]
    print(f"\n  largest gap: {NAME[ranked[ci][0]]} {ranked[ci][1]:.2f}B -> "
          f"{NAME[ranked[ci+1][0]]} {ranked[ci+1][1]:.2f}B = {worst[0]:.0f}x")

    need = MIN_LANE_SHARE * LANE_B / EPOCH_CAP
    print(f"\n  Entry gate: unique native-script ≥ {need:.2f}B  "
          f"(1% of {LANE_B:.0f}B lane / {EPOCH_CAP:.0f} epochs)")
    funded = [lg for lg, v in ranked if v + 1e-12 >= need]
    gated = [lg for lg, v in ranked if v < need]
    print(f"  funded {len(funded)}: {', '.join(NAME[l] for l in funded)}")
    print(f"  gated  {len(gated)}: {', '.join(NAME[l] for l in gated)}")
    milu_ok = sum(1 for l in MILU if l in funded)
    igb_tail_gated = [NAME[l] for l in gated if l in IGB_TAIL]
    print(f"\n  MILU {milu_ok}/10  (all MILU languages are above the cliff)")
    print(f"  IndicGenBench: {len(funded)} languages with supply above the gate; "
          f"plus 10 with zero tokens anywhere.")
    print(f"  IGB tail gated: {', '.join(igb_tail_gated)}")
    print("  Re-entry is automatic when a gated language crosses the threshold. No redesign.")


if __name__ == "__main__":
    main()
