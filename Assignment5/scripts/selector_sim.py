"""Selector simulation: retention, floors, ceilings, loud failure.

The loader proposes a stage mixture. Candidates are oversampled by 1/keep_frac.
OPUS-style scoring keeps ~40% of *eligible* candidates (web, code, stem, longctx,
unverified Indic). Protected lanes (agentic, reasoning, verified/translated/synthetic
Indic) bypass the selector. Floors inject outside the selector. A lane may not
exceed 1.25× its stage target. If remaining eligible supply cannot meet a floor,
the loader raises — it does not silently undershoot.

Domain vectors and the 40% keep-rate are STATED from the session OPUS widget.
Selected/rejected counts are MEASURED here.

Run:  python scripts/selector_sim.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import ANN_SHARE, CEILING_MULT, FLOOR_SEQ, LANES, ROOT, ipf_stages, STAGES

KEEP_FRAC = 0.40
SEED = 7
INDIC_ELIGIBLE_FRAC = 0.32  # tier B only; A/C/D bypass

DOMAIN = {
    "web":     (0.92, 0.20, 0.08),
    "code":    (0.78, 0.52, 0.06),
    "reason":  (0.50, 0.82, 0.10),
    "agentic": (0.34, 0.50, 0.30),
    "indic":   (0.18, 0.26, 0.92),
    "stem":    (0.70, 0.75, 0.12),
    "longctx": (0.60, 0.40, 0.15),
}
PROXY = {
    "english":  (0.70, 0.53, 0.33),
    "balanced": (0.577, 0.577, 0.577),
    "v5":       (0.40, 0.30, 0.30),
}
PROTECTED = {"reason", "agentic"}  # always-on, invisible to OPUS
ELIGIBLE_LANES = {"web", "code", "stem", "longctx", "indic"}


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def stage_target(stage):
    if stage == "S5 anneal":
        return {k: float(ANN_SHARE[k]) for k in LANES}
    idx = [n for n, _ in STAGES].index(stage)
    M = ipf_stages()
    return {k: M[k][idx] for k in LANES}


def sample_from_mix(mix, n, rng):
    keys = list(mix)
    weights = [max(mix[k], 0.0) for k in keys]
    if sum(weights) == 0:
        raise ValueError("empty mix")
    lanes = rng.choices(keys, weights=weights, k=n)
    out = []
    for i, lane in enumerate(lanes):
        if lane in PROTECTED:
            eligible = False
        elif lane == "indic":
            eligible = rng.random() < INDIC_ELIGIBLE_FRAC
        else:
            eligible = lane in ELIGIBLE_LANES
        out.append(dict(id=i, lane=lane, eligible=eligible, tokens=1.0))
    return out


def score(c, proxy, rng, noise=0.04):
    return dot(DOMAIN[c["lane"]], PROXY[proxy]) + rng.gauss(0, noise)


def _share(sel):
    tot = len(sel) or 1
    c = Counter(x["lane"] for x in sel)
    return {k: 100.0 * c[k] / tot for k in LANES}


def select(candidates, stage, proxy, rng, batch_size, fallback="fail"):
    """Fill a batch of `batch_size` tokens from an oversampled candidate pool."""
    floors = FLOOR_SEQ[stage]
    target = stage_target(stage)
    eligible = [c for c in candidates if c["eligible"]]
    protected = [c for c in candidates if not c["eligible"]]
    eligible.sort(key=lambda c: score(c, proxy, rng), reverse=True)
    n_keep = max(1, int(round(KEEP_FRAC * len(eligible))))
    kept_eligible = eligible[:n_keep]
    rejected = eligible[n_keep:] + []

    selected = (kept_eligible + protected)[:]
    rng.shuffle(selected)

    # Downsample to batch_size, preferring to drop lowest-score eligible.
    if len(selected) > batch_size:
        selected.sort(key=lambda c: (c["eligible"], score(c, proxy, rng)))
        extra = selected[:(len(selected) - batch_size)]
        selected = selected[(len(selected) - batch_size):]
        rejected.extend(extra)

    def enforce_ceiling(sel, rej):
        sh = _share(sel)
        for k in LANES:
            cap = target[k] * CEILING_MULT
            if sh[k] <= cap + 1e-6:
                continue
            n_drop = int(round((sh[k] - cap) / 100.0 * len(sel)))
            droppable = [c for c in sel if c["lane"] == k]
            droppable.sort(key=lambda c: score(c, proxy, rng))
            drop = droppable[:n_drop]
            drop_ids = {id(c) for c in drop}
            rej.extend(drop)
            sel = [c for c in sel if id(c) not in drop_ids]
            sh = _share(sel)
        return sel, rej

    selected, rejected = enforce_ceiling(selected, rejected)

    def enforce_floors(sel, rej):
        sh = _share(sel)
        for k, fl in floors.items():
            have = sum(1 for c in sel if c["lane"] == k)
            need = int(round(fl / 100.0 * max(len(sel), batch_size))) - have
            if need <= 0:
                continue
            pool = [c for c in rej if c["lane"] == k]
            if len(pool) < need:
                if fallback == "fail":
                    raise RuntimeError(
                        f"{stage}: floor {k}={fl}% unmet; have {sh[k]:.2f}%; "
                        f"need {need} more, {len(pool)} remaining. "
                        f"Silent undershoot is forbidden."
                    )
                if fallback == "steal_web":
                    web = [c for c in sel if c["lane"] == "web"]
                    n_swap = min(len(web), need - len(pool))
                    if len(pool) + n_swap < need:
                        raise RuntimeError(
                            f"{stage}: steal_web cannot meet {k} floor {fl}%"
                        )
                    take_web = web[:n_swap]
                    sel = [c for c in sel if c not in take_web]
                    rej.extend(take_web)
                    need_left = need - n_swap
                    take = pool[:need_left] if need_left > 0 else []
                    sel.extend(take + pool[:0])
                    sel.extend(pool[: min(len(pool), need)])
                    pool_ids = {id(c) for c in pool[: min(len(pool), need)]}
                    rej = [c for c in rej if id(c) not in pool_ids]
                    sh = _share(sel)
                    continue
            take = pool[:need]
            sel.extend(take)
            take_ids = {id(c) for c in take}
            rej = [c for c in rej if id(c) not in take_ids]
            # drop web to keep batch_size if we grew
            overflow = len(sel) - batch_size
            if overflow > 0:
                web = [c for c in sel if c["lane"] == "web"]
                web.sort(key=lambda c: score(c, proxy, rng))
                drop = web[:overflow]
                drop_ids = {id(c) for c in drop}
                rej.extend(drop)
                sel = [c for c in sel if id(c) not in drop_ids]
            sh = _share(sel)
        return sel, rej

    selected, rejected = enforce_floors(selected, rejected)
    selected, rejected = enforce_ceiling(selected, rejected)

    # Pad from the most under-target lane so web does not absorb the remainder
    # and breach its ceiling. Trim the most over-target lane if we overflowed.
    def deficit(sel):
        sh = _share(sel)
        return sorted(((target[k] - sh[k], k) for k in LANES), reverse=True)

    while len(selected) < batch_size and rejected:
        want = [k for _, k in deficit(selected)]
        pad = None
        for k in want:
            pool = [c for c in rejected if c["lane"] == k]
            if pool:
                pad = pool[0]
                break
        if pad is None:
            pad = rejected[0]
        selected.append(pad)
        rejected = [c for c in rejected if id(c) != id(pad)]
    while len(selected) > batch_size:
        sh = _share(selected)
        over = max(LANES, key=lambda k: sh[k] - target[k] * CEILING_MULT)
        droppable = [c for c in selected if c["lane"] == over]
        if not droppable:
            droppable = selected
        drop = droppable[0]
        rejected.append(drop)
        selected = [c for c in selected if id(c) != id(drop)]
    selected, rejected = enforce_ceiling(selected, rejected)
    selected, rejected = enforce_floors(selected, rejected)
    if abs(len(selected) - batch_size) > 5:
        raise RuntimeError(f"{stage}: selected {len(selected)} != batch_size {batch_size}")
    if len(selected) != batch_size and rejected:
        # final integer trim/pad, ignoring mix
        if len(selected) < batch_size:
            selected.extend(rejected[: batch_size - len(selected)])
        else:
            selected = selected[:batch_size]
    return selected, rejected, _share(selected), target


def run_one(stage, proxy, n=5000, seed=SEED, fallback="fail"):
    rng = random.Random(seed)
    target = stage_target(stage)
    batch_size = n
    # Oversample so that 40% keep of eligible plus protected can fill the batch.
    cand = sample_from_mix(target, int(n / KEEP_FRAC), rng)
    try:
        selected, rejected, sh, target = select(
            cand, stage, proxy, rng, batch_size, fallback=fallback
        )
        return dict(
            ok=True, share=sh, target=target, n_sel=len(selected), n_rej=len(rejected),
            by_lane=dict(Counter(c["lane"] for c in selected)),
            rej_by_lane=dict(Counter(c["lane"] for c in rejected)),
        )
    except RuntimeError as e:
        return dict(ok=False, error=str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    stages = [n for n, _ in STAGES] + ["S5 anneal"]
    print("=" * 88)
    print(f"SELECTOR SIMULATION  keep={100*KEEP_FRAC:.0f}%  ceiling=1.25×  floors=FLOOR_SEQ")
    print("Proxy vectors STATED (session widget). Counts MEASURED here.")
    print("=" * 88)
    summary = {}
    for proxy in ("english", "balanced", "v5"):
        print(f"\n-- proxy={proxy}  {PROXY[proxy]}")
        print(f"  {'stage':<16}{'ok':>4}  " + "".join(f"{k[:6]:>8}" for k in LANES))
        summary[proxy] = {}
        for st in stages:
            r = run_one(st, proxy, n=args.n, seed=args.seed, fallback="fail")
            summary[proxy][st] = {k: v for k, v in r.items() if k != "by_lane"}
            if not r["ok"]:
                print(f"  {st:<16} FAIL  {r['error'][:90]}")
                continue
            print(f"  {st:<16}{'OK':>4}  " + "".join(f"{r['share'][k]:7.1f}%" for k in LANES))

    print("\n-- starvation arm: english proxy, NO floors, keep top 40% of all candidates")
    rng = random.Random(args.seed)
    mix = stage_target("S2 general")
    cand = sample_from_mix(mix, int(args.n / KEEP_FRAC), rng)
    ranked = sorted(cand, key=lambda c: score(c, "english", rng), reverse=True)
    kept = ranked[: args.n]
    sh = {k: 100.0 * sum(1 for c in kept if c["lane"] == k) / len(kept) for k in LANES}
    print(f"  {'no-floor':<16}      " + "".join(f"{sh[k]:7.1f}%" for k in LANES))
    print(f"  Indic {sh['indic']:.2f}%  agentic {sh['agentic']:.2f}%  "
          f"(with floors: Indic ≥{FLOOR_SEQ['S2 general']['indic']}%  "
          f"agentic ≥{FLOOR_SEQ['S2 general']['agentic']}%)")

    out = ROOT / "data" / "selector_sim_results.json"
    payload = dict(summary=summary, starvation=sh, n=args.n, seed=args.seed,
                   keep=KEEP_FRAC, proxy=PROXY)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
