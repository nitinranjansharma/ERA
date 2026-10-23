"""Per-language allocation inside the Indic lane.

Supply is the AI4Bharat Sangraha per-language token table (HF dataset card).
Its totals — 64,306.1M verified / 24,307.7M unverified / 162,707.9M synthetic —
reproduce the S5 inventory widget rows, so each language row is auditable
against a number the session already uses.

Two corrections are applied here, not later in prose:

  1. 12,759.9M of "verified" Sangraha is ENGLISH. It is routed to web.
  2. Synthetic native/Latin splits have identical row counts. Only the native
     half is scored by MILU / IndicGenBench.

Allocation: temperature sampling p_i ∝ S_i^α (XLM-R, arXiv:1911.02116),
water-filled against a per-language cap of 4 × supply (arXiv:2305.16264).

Run:  python scripts/indic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import EPOCH_CAP, TOTAL_B, MAIN_SHARE, ANN_SHARE, MAIN_B, ANN_B

# Sangraha per-language tokens, millions. (verified, synthetic, unverified)
# https://huggingface.co/datasets/ai4bharat/sangraha
SANGRAHA = {
    "asm": (292.1, 11696.4, 17.5),      "ben": (10604.4, 13814.1, 5608.8),
    "brx": (1.5, 0.0, 0.0),             "doi": (0.06, 0.0, 0.0),
    "eng": (12759.9, 0.0, 0.0),         "gom": (10.1, 0.0, 0.0),
    "guj": (3647.9, 12934.5, 597.0),    "hin": (12617.3, 9578.7, 12348.3),
    "kan": (1778.3, 12087.4, 388.8),    "kas": (0.5, 0.0, 0.0),
    "mai": (14.6, 0.0, 0.0),            "mal": (2730.8, 13130.0, 547.8),
    "mar": (2827.0, 10816.7, 652.1),    "mni": (7.4, 0.0, 0.0),
    "npi": (1822.5, 10588.7, 485.5),    "ori": (1177.1, 11338.0, 23.7),
    "pan": (1075.3, 9969.6, 136.9),     "san": (1329.0, 13553.5, 9.8),
    "sat": (0.3, 0.0, 0.0),             "snd": (258.2, 0.0, 0.0),
    "tam": (3985.1, 11859.3, 1515.9),   "tel": (3706.8, 11924.5, 647.4),
    "urd": (3658.1, 9415.8, 1328.2),
}
NAME = {
    "asm": "Assamese", "ben": "Bengali", "brx": "Bodo", "doi": "Dogri", "gom": "Konkani",
    "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri", "mai": "Maithili",
    "mal": "Malayalam", "mar": "Marathi", "mni": "Manipuri", "npi": "Nepali", "ori": "Odia",
    "pan": "Punjabi", "san": "Sanskrit", "sat": "Santali", "snd": "Sindhi", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}
MILU = {"ben", "guj", "hin", "kan", "mal", "mar", "ori", "pan", "tam", "tel"}
IGB_HIGH = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Tamil", "Telugu", "Urdu"]
IGB_MED = ["Assamese", "Bhojpuri", "Nepali", "Odia", "Punjabi", "Pashto", "Sanskrit"]
IGB_LOW = ["Awadhi", "Haryanvi", "Tibetan", "Garhwali", "Konkani", "Chhattisgarhi",
           "Rajasthani", "Maithili", "Manipuri", "Malvi", "Marwari", "Santali", "Bodo"]
NATIVE_SHARE = 0.50
LANE_B = MAIN_SHARE["indic"] / 100 * MAIN_B + ANN_SHARE["indic"] / 100 * ANN_B


def supply_b():
    out = {}
    for lg, (ver, syn, unv) in SANGRAHA.items():
        if lg == "eng":
            continue
        out[lg] = (ver + unv + NATIVE_SHARE * syn) / 1000.0
    return out


def allocate(S, total, alpha, cap_mult=EPOCH_CAP):
    cap = {lg: cap_mult * s for lg, s in S.items()}
    alloc = {lg: 0.0 for lg in S}
    free = set(S)
    remaining = total
    for _ in range(200):
        if remaining <= 1e-9 or not free:
            break
        wsum = sum(S[lg] ** alpha for lg in free)
        newly = []
        for lg in free:
            alloc[lg] += remaining * (S[lg] ** alpha) / wsum
            if alloc[lg] >= cap[lg] - 1e-12:
                alloc[lg] = cap[lg]
                newly.append(lg)
        remaining = total - sum(alloc.values())
        for lg in newly:
            free.discard(lg)
    return alloc, max(0.0, remaining)


def fmt(x):
    return f"{x:,.2f}" if x < 10 else f"{x:,.1f}"


def main():
    S = supply_b()
    tot_ver = sum(v for lg, (v, _, _) in SANGRAHA.items() if lg != "eng")
    print("=" * 78)
    print("SUPPLY  (stated: Sangraha card; derived: English removed, native-script half)")
    print("=" * 78)
    print(f"  Sangraha verified, all configs : {sum(v for v,_,_ in SANGRAHA.values())/1000:6.1f}B")
    print(f"  ...of which ENGLISH            : {SANGRAHA['eng'][0]/1000:6.1f}B")
    print(f"  verified Indic only            : {tot_ver/1000:6.1f}B")
    syn = sum(s for _, s, _ in SANGRAHA.values()) / 1000
    print(f"  synthetic, both scripts        : {syn:6.1f}B")
    print(f"  synthetic, native script only  : {syn*NATIVE_SHARE:6.1f}B")
    print(f"  usable native-script Indic     : {sum(S.values()):6.1f}B  "
          f"vs {LANE_B:.1f}B lane demand -> {LANE_B/sum(S.values()):.2f} epochs")

    ranked = sorted(S.items(), key=lambda kv: -kv[1])
    top2 = sum(v for _, v in ranked[:2])
    print(f"\n  top-2 (hin+ben) hold {top2/sum(S.values())*100:.1f}% of usable supply")
    bottom = [lg for lg, v in ranked if v < 1.0]
    print(f"  {len(bottom)} languages under 1B: {', '.join(NAME[l] for l in bottom)}")
    print(f"  combined {sum(S[l] for l in bottom)*1000:.1f}M tokens")

    have = {NAME[lg] for lg in S}
    print("\n" + "=" * 78)
    print("BENCHMARK COVERAGE vs SUPPLY")
    print("=" * 78)
    for tier, langs in (("high", IGB_HIGH), ("medium", IGB_MED), ("low", IGB_LOW)):
        missing = [l for l in langs if l not in have]
        print(f"  IndicGenBench {tier:6s}: {len(langs)-len(missing):2d}/{len(langs)} have supply"
              + (f"   ZERO: {', '.join(missing)}" if missing else ""))
    all_igb = IGB_HIGH + IGB_MED + IGB_LOW
    zero = [l for l in all_igb if l not in have]
    print(f"  IndicGenBench total : {len(all_igb)-len(zero)}/{len(all_igb)} have any tokens")
    print(f"  MILU (10 Indic)     : {sum(1 for l in MILU if l in S)}/10  "
          f"weakest {min((S[l], NAME[l]) for l in MILU)[1]} at {min(S[l] for l in MILU):.2f}B")

    print("\n" + "=" * 78)
    print("TEMPERATURE SWEEP  p_i ~ S_i^alpha, cap 4 epochs")
    print("=" * 78)
    print(f"  {'alpha':>5} {'hin %':>7} {'ben %':>7} {'weakest MILU':>13} {'capped':>7} {'unplaceable':>12}")
    star = None
    wk_max = min(4 * S[lg] for lg in MILU)
    for i in range(41):
        alpha = 1.00 - i * 0.01
        a, un = allocate(S, LANE_B, alpha)
        wk = min(a[lg] for lg in MILU)
        cappd = sum(1 for lg in S if a[lg] >= 4 * S[lg] - 1e-9)
        if wk >= wk_max - 1e-6 and star is None:
            star = alpha
        if abs(alpha * 100 - round(alpha * 100 / 5) * 5) < 1e-6:
            print(f"  {alpha:>5.2f} {a['hin']/LANE_B*100:>6.1f}% {a['ben']/LANE_B*100:>6.1f}% "
                  f"{wk:>11.1f}B {cappd:>7d} {un:>11.1f}B")
    print(f"\n  alpha* = {star:.2f}  (largest α at which the weakest MILU language already hits its cap)")
    print(f"  Punjabi ceiling = 4 × {min(S[lg] for lg in MILU):.2f}B = {wk_max:.1f}B")

    for alpha in (1.0, star, 0.3):
        a, un = allocate(S, LANE_B, alpha)
        print("\n" + "=" * 78)
        print(f"ALLOCATION at alpha = {alpha}    unplaceable {un:.1f}B")
        print("=" * 78)
        print(f"  {'lang':<11}{'supply B':>10}{'alloc B':>10}{'share':>8}{'epochs':>8}{'ver%':>7}  bench")
        for lg, _ in sorted(S.items(), key=lambda kv: -kv[1]):
            ver, syn_, unv = SANGRAHA[lg]
            usable = S[lg]
            verpct = ver / (usable * 1000) * 100 if usable else 0
            ep = a[lg] / usable if usable else float("inf")
            tag = "MILU" if lg in MILU else ""
            flag = "  <-CAP" if ep >= 3.999 else ""
            print(f"  {NAME[lg]:<11}{fmt(usable):>10}{fmt(a[lg]):>10}"
                  f"{a[lg]/LANE_B*100:>7.2f}%{ep:>8.2f}{verpct:>6.0f}%  {tag}{flag}")
        print(f"  {'TOTAL':<11}{sum(S.values()):>10.1f}{sum(a.values()):>10.1f}")


if __name__ == "__main__":
    main()
