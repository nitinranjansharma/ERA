"""Allocate Indic over CONTENT, then convert back to tokens.

MUTANT-Indic (stated via Session 3): Hindi 1.375 tok/word, Malayalam 3.025.
Equal-token allocation gives Malayalam ~45% of Hindi's content. MILU scores
comprehension, not token counts.

The 4-epoch cap is fertility-invariant. Only the allocation shape moves.
Stated fertilities, not measured on our tokenizer.

Run:  python scripts/indic_fertility.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from indic import supply_b, LANE_B
from ledger import EPOCH_CAP

FERT_GROUP = {
    1.375: ["hin", "pan", "urd", "kas", "snd", "doi"],
    1.725: ["mar", "ori", "mai", "npi", "ben", "guj", "asm"],
    2.050: ["tel", "brx", "tam", "gom", "kan"],
    3.025: ["mni", "mal", "san", "sat"],
}
FERT = {lg: f for f, lgs in FERT_GROUP.items() for lg in lgs}


def content_supply(S):
    return {lg: S[lg] / FERT[lg] for lg in S}


def main():
    S = supply_b()
    C = content_supply(S)
    print("=" * 78)
    print("FERTILITY ADJUSTMENT  (stated: MUTANT-Indic group midpoints)")
    print("=" * 78)
    print(f"  Hindi tok/word {FERT['hin']:.3f}  Malayalam {FERT['mal']:.3f}  "
          f"ratio {FERT['mal']/FERT['hin']:.2f}x")
    print("\n  Operational rule: p_i ∝ (S_i / f_i)^α, token cap still 4 × S_i.")
    def alloc_content_weighted(alpha):
        weights = {lg: (S[lg] / FERT[lg]) ** alpha for lg in S}
        cap = {lg: EPOCH_CAP * S[lg] for lg in S}
        alloc = {lg: 0.0 for lg in S}
        free = set(S)
        remaining = LANE_B
        for _ in range(200):
            if remaining <= 1e-9 or not free:
                break
            wsum = sum(weights[lg] for lg in free)
            newly = []
            for lg in free:
                alloc[lg] += remaining * weights[lg] / wsum
                if alloc[lg] >= cap[lg] - 1e-12:
                    alloc[lg] = cap[lg]
                    newly.append(lg)
            remaining = LANE_B - sum(alloc.values())
            for lg in newly:
                free.discard(lg)
        return alloc, max(0.0, remaining)

    print(f"  {'alpha':>5} {'hin tok':>8} {'mal tok':>8} {'hin content':>12} {'mal content':>12} {'capped':>7}")
    star = None
    mal_best = -1
    for i in range(71):
        alpha = 1.00 - i * 0.01
        a, un = alloc_content_weighted(alpha)
        hc = a["hin"] / FERT["hin"]
        mc = a["mal"] / FERT["mal"]
        cappd = sum(1 for lg in S if a[lg] >= EPOCH_CAP * S[lg] - 1e-9)
        if abs(alpha * 100 - round(alpha * 100 / 10) * 10) < 1e-6:
            print(f"  {alpha:>5.2f} {a['hin']:8.1f} {a['mal']:8.1f} {hc:12.1f} {mc:12.1f} {cappd:7d}")
        if mc > mal_best + 1e-9:
            mal_best = mc
            star = alpha
    print(f"\n  α* (content) ≈ {star:.2f}  — Malayalam content stops rising here because the token cap binds.")
    a, _ = alloc_content_weighted(star)
    print(f"  Malayalam tokens {a['mal']:.1f}B = {a['mal']/LANE_B*100:.1f}% of lane, "
          f"{a['mal']/S['mal']:.2f} epochs")
    print(f"  Malayalam content / Hindi content = {(a['mal']/FERT['mal'])/(a['hin']/FERT['hin']):.2f}")
    print("  Content parity is not reachable at this lane size: high-fertility languages hit 4 epochs first.")


if __name__ == "__main__":
    main()
