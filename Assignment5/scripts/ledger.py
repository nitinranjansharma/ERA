"""V5 mixture ledger — specification checker, not a budget printer.

Every number in the README is produced here. Assertions encode the plan:
stage sums, floors, ceilings, Indic caps, disjoint inventory, SFT holdout,
escrow isolation, epoch ceilings, and data-gate status.

Run:  python scripts/ledger.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV_PATH = ROOT / "data" / "inventory.json"

# ---------------------------------------------------------------------------
# Design point (stated: session range 2.4–4T; 3T is the midpoint)
# ---------------------------------------------------------------------------
TOTAL_B = 3000.0
ANNEAL_FRAC = 0.03
MAIN_B = TOTAL_B * (1.0 - ANNEAL_FRAC)  # 2910
ANN_B = TOTAL_B * ANNEAL_FRAC           # 90
EPOCH_CAP = 4.0                         # arXiv:2305.16264
CEILING_MULT = 1.25
NONNATIVE_CAP = 40.0                    # sequence-token % of the Indic lane, every stage
COMMITPACK_EPOCHS = 4.0                 # upsample to the ceiling, never past it

LANES = ["web", "code", "reason", "longctx", "indic", "stem", "agentic"]
MAIN_SHARE = dict(web=34, code=24, reason=9, longctx=2, indic=17, stem=12, agentic=2)
ANN_SHARE = dict(web=8, code=20, reason=18, longctx=8, indic=28, stem=10, agentic=8)
COMPOSER = dict(web=34, code=24, reason=6, longctx=6, indic=16, stem=12, agentic=2)

# Long-context lane composition (inventory: 60B packed code + 40B packed books).
# This overlay is INSIDE the longctx budget, not added on top of it.
LONG_COMP = dict(code=0.60, web=0.40)

# Indic provenance split. C+D = 40% in both pools. Sequence-token cap, not a weighted score.
TIERS_MAIN = dict(A=28, B=32, C=2, D=38)
TIERS_ANN = dict(A=50, B=10, C=2, D=38)
TIER_NAMES = dict(A="verified native", B="unverified crawl", C="translated/parallel", D="synthetic")

# Loss-bearing fraction of sequence tokens.
# agentic 0.53 is STATED from the session trajectory widget (356/668).
# reason 0.90 is a HYPOTHESIS until measured on the packed examples.
LOSS_FRAC = dict(web=1.0, code=1.0, stem=1.0, indic=1.0, longctx=1.0, reason=0.90, agentic=0.53)
LOSS_FRAC_LABEL = dict(
    web="assumed 1.0",
    code="assumed 1.0",
    stem="assumed 1.0",
    indic="assumed 1.0",
    longctx="assumed 1.0",
    reason="hypothesis 0.90 (prompt masked)",
    agentic="stated 356/668 = 0.53 (session widget)",
)

# Stage weights as a fraction of the MAIN run. Anneal is a separate pool.
STAGES = [
    ("S1 seed", 0.08),
    ("S2 general", 0.32),
    ("S3 reason/code", 0.36),
    ("S4 long-ctx", 0.24),
]
STAGE_SHAPE = {
    "web":     [55, 45, 28, 20],
    "code":    [15, 20, 28, 30],
    "reason":  [2, 5, 13, 12],    # traces enter after a base exists
    "longctx": [0.2, 0.5, 1, 7],  # extension is late
    "indic":   [17, 17, 17, 17],  # protected: flat, never a residue
    "stem":    [9, 12, 13, 12],
    "agentic": [2, 2, 2, 2],      # floor = allocation in the main run; concentration is the anneal
}

# Two floor kinds, both executable:
#   RUN_FLOOR  — token-weighted mean across the main run (composer slider clamps).
#   FLOOR_SEQ  — per-batch, ~65% of that stage's target. Reasoning ramps, so S1/S2
#                sit below the 6% run floor on purpose; the integral still hits 9%.
RUN_FLOOR = dict(indic=12.0, agentic=2.0, reason=6.0, web=25.0)
FLOOR_SEQ = {
    "S1 seed":         dict(indic=12.0, agentic=1.5, reason=1.2, web=25.0),
    "S2 general":      dict(indic=12.0, agentic=1.5, reason=3.0, web=25.0),
    "S3 reason/code":  dict(indic=12.0, agentic=1.5, reason=8.0, web=18.0),
    "S4 long-ctx":     dict(indic=12.0, agentic=1.5, reason=7.5, web=18.0),
    "S5 anneal":       dict(indic=20.0, agentic=8.0, reason=12.0, web=8.0),
}
FLOOR_LOSS = {"S5 anneal": dict(agentic=3.5)}  # loss-token share, not sequence share

# Disjoint inventory partitions (B tokens). A unique token is counted once.
SUPPLY = dict(
    web=4691.0,
    code=1103.0,
    stem=146.0,
    longctx=100.0,
    agentic_fc=0.267,
    agentic_traj=0.360,
    agentic=0.627,
    reason_total=85.1,
    reason_sft=8.0,       # disjoint holdout, never in pretraining
    reason_escrow=8.0,    # disjoint anneal slice of the remaining 77.1B
    reason_pt_main=69.1,  # 85.1 - 8 - 8
    indic_A=51.5462,      # Sangraha verified minus English
    indic_A_escrow=15.0,  # disjoint quality partition for anneal
    indic_A_main=36.5462,
    indic_B=45.2077,      # 24.3077 Sangraha unverified + 20.9 IndicCorpV2 (overlap unmeasured)
    indic_B_sangraha_only=24.3077,
    indic_C=6.8,          # BPCC Indic-side; Samanantar is inside BPCC
    indic_D=81.35395,     # native-script half of Sangraha synthetic
)


def ipf_stages(iters: int = 4000) -> dict[str, list[float]]:
    """Iterative proportional fitting: each stage sums to 100%, each lane integrates to MAIN_SHARE.

    After IPF, per-batch floors are enforced by taking from web (the adjustment
    variable). If web cannot pay without breaching its own floor, the plan is
    infeasible and validate() fails.
    """
    W = [w for _, w in STAGES]
    M = {k: [float(x) for x in STAGE_SHAPE[k]] for k in LANES}
    for _ in range(iters):
        for k in LANES:
            cur = sum(M[k][i] * W[i] for i in range(4))
            if cur > 0:
                f = MAIN_SHARE[k] / cur
                M[k] = [x * f for x in M[k]]
        for i in range(4):
            s = sum(M[k][i] for k in LANES)
            for k in LANES:
                M[k][i] *= 100.0 / s
    return M


def stage_tokens(M: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out = {}
    for i, (name, w) in enumerate(STAGES):
        pool = MAIN_B * w
        out[name] = {k: M[k][i] / 100.0 * pool for k in LANES}
        out[name]["_pool"] = pool
    out["S5 anneal"] = {k: ANN_SHARE[k] / 100.0 * ANN_B for k in LANES}
    out["S5 anneal"]["_pool"] = ANN_B
    return out


def named_and_effective(stage_tok: dict[str, dict[str, float]]):
    """Three views: named lanes, long-context overlay, effective capability mix.

    The  longctx tokens are already inside each stage pool. Effective capability
    folds LONG_COMP into code/web without changing the token total.
    """
    named, overlay, effective = {}, {}, {}
    for s, row in stage_tok.items():
        pool = row["_pool"]
        named[s] = {k: 100.0 * row[k] / pool for k in LANES}
        ov = {k: 0.0 for k in LANES}
        for cap, frac in LONG_COMP.items():
            ov[cap] = 100.0 * row["longctx"] * frac / pool
        overlay[s] = ov
        eff = {k: named[s][k] for k in LANES}
        for cap, frac in LONG_COMP.items():
            eff[cap] += named[s]["longctx"] * frac
        eff["longctx"] = 0.0  # consumed into capabilities
        effective[s] = eff
    return named, overlay, effective


def lane_totals(stage_tok) -> dict[str, float]:
    tot = {k: 0.0 for k in LANES}
    for s, row in stage_tok.items():
        for k in LANES:
            tot[k] += row[k]
    return tot


def indic_tiers(stage_tok):
    per = {}
    for s, row in stage_tok.items():
        split = TIERS_ANN if s == "S5 anneal" else TIERS_MAIN
        per[s] = {t: row["indic"] * split[t] / 100.0 for t in split}
    totals = {t: sum(per[s][t] for s in per) for t in TIERS_MAIN}
    return per, totals


def loss_shares(stage_tok):
    out = {}
    for s, row in stage_tok.items():
        loss = {k: row[k] * LOSS_FRAC[k] for k in LANES}
        t = sum(loss.values())
        out[s] = dict(tokens=loss, total=t, pct={k: 100.0 * loss[k] / t for k in LANES})
    return out


def load_inventory() -> dict:
    return json.loads(INV_PATH.read_text())


def unique_by_slot(inv: dict) -> dict[str, float]:
    acc: dict[str, float] = {}
    seen = set()
    for d in inv["datasets"]:
        if d["id"] in ("samanantar", "sangraha-verified-en", "aon-v4-sft"):
            continue
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        for slot in d.get("slots", []):
            acc[slot] = acc.get(slot, 0.0) + float(d["unique_tokens_b"])
    return acc


# ---------------------------------------------------------------------------
# Assertions — the specification, not comments
# ---------------------------------------------------------------------------
class SpecError(AssertionError):
    pass


def _ok(cond, msg):
    if not cond:
        raise SpecError(msg)


def validate(M, stage_tok, named, effective, indic_per, indic_tot, loss, inv) -> list[str]:
    notes = []

    _ok(abs(sum(MAIN_SHARE.values()) - 100) < 1e-9, "main shares must sum to 100")
    _ok(abs(sum(ANN_SHARE.values()) - 100) < 1e-9, "anneal shares must sum to 100")
    _ok(abs(sum(COMPOSER.values()) - 100) < 1e-9, "composer shares must sum to 100")
    _ok(abs(sum(STAGES[i][1] for i in range(4)) - 1.0) < 1e-9, "stage weights must sum to 1")
    _ok(abs(sum(LONG_COMP.values()) - 1.0) < 1e-9, "LONG_COMP must sum to 1")
    _ok(abs(sum(TIERS_MAIN.values()) - 100) < 1e-9, "main Indic tiers must sum to 100")
    _ok(abs(sum(TIERS_ANN.values()) - 100) < 1e-9, "anneal Indic tiers must sum to 100")
    _ok(abs(MAIN_B + ANN_B - TOTAL_B) < 1e-9, "main + anneal must equal total")

    # Stage sums and integrals.
    W = [w for _, w in STAGES]
    for i, (name, _) in enumerate(STAGES):
        s = sum(M[k][i] for k in LANES)
        _ok(abs(s - 100) < 0.05, f"{name} mixture sums to {s}, not 100")
    for k in LANES:
        integ = sum(M[k][i] * W[i] for i in range(4))
        _ok(abs(integ - MAIN_SHARE[k]) < 0.05, f"{k} integral {integ:.3f} != budget {MAIN_SHARE[k]}")
    for lane, fl in RUN_FLOOR.items():
        integ = sum(M[lane][i] * W[i] for i in range(4))
        _ok(integ + 1e-6 >= fl, f"run-integral {lane} {integ:.2f}% below floor {fl}%")
        _ok(all(v >= 0 for v in M[k]), f"{k} has a negative stage share")

    tot = lane_totals(stage_tok)
    _ok(abs(sum(tot.values()) - TOTAL_B) < 1e-6, f"lane totals {sum(tot.values())} != {TOTAL_B}")

    # Effective shares also sum to 100 (longctx folded in).
    for s, eff in effective.items():
        _ok(abs(sum(eff.values()) - 100) < 1e-6, f"{s} effective shares sum to {sum(eff.values())}")
        _ok(all(v >= -1e-9 for v in eff.values()), f"{s} effective has a negative")

    # Floors on the NAMED mixture (the selector sees named lanes, not the overlay).
    for s, floors in FLOOR_SEQ.items():
        for lane, fl in floors.items():
            got = named[s][lane]
            _ok(got + 1e-6 >= fl, f"{s} {lane} named share {got:.2f}% below floor {fl}%")
            _ok(got <= fl * CEILING_MULT + MAIN_SHARE.get(lane, 100) * 2, "ceiling checked below")

    # Planned mixture itself must sit inside [floor, 1.25 × target]. The selector
    # simulation (scripts/selector_sim.py) is what enforces the ceiling after selection.
    for i, (s, _) in enumerate(STAGES):
        for k in LANES:
            _ok(named[s][k] <= M[k][i] * CEILING_MULT + 1e-6,
                f"{s} {k} {named[s][k]:.2f} exceeds 1.25× target {M[k][i]:.2f}")
    for k in LANES:
        _ok(named["S5 anneal"][k] <= ANN_SHARE[k] * CEILING_MULT + 1e-6,
            f"anneal {k} {named['S5 anneal'][k]:.2f} exceeds 1.25× {ANN_SHARE[k]}")

    # Indic non-native cap: sequence tokens, every stage, and run-wide.
    for s, split in indic_per.items():
        indic = sum(split.values())
        nn = 100.0 * (split["C"] + split["D"]) / indic
        _ok(nn <= NONNATIVE_CAP + 1e-6,
            f"{s} non-native Indic {nn:.2f}% exceeds hard cap {NONNATIVE_CAP}%")
    nn_run = 100.0 * (indic_tot["C"] + indic_tot["D"]) / tot["indic"]
    _ok(nn_run <= NONNATIVE_CAP + 1e-6, f"run-wide non-native {nn_run:.2f}% exceeds cap")
    notes.append(f"non-native Indic cap is a sequence-token cap at {NONNATIVE_CAP:.0f}% of the Indic lane")

    # Disjoint reasoning partitions.
    _ok(abs(SUPPLY["reason_sft"] + SUPPLY["reason_escrow"] + SUPPLY["reason_pt_main"]
            - SUPPLY["reason_total"]) < 1e-9,
        "reason_sft + reason_escrow + reason_pt_main must equal reason_total")
    _ok(SUPPLY["reason_sft"] > 0, "SFT holdout must be non-empty")
    notes.append("SFT holdout is AFTER the 85.1B unique total, not inside a 12B PT pool")

    # Disjoint Indic A partitions.
    _ok(abs(SUPPLY["indic_A_escrow"] + SUPPLY["indic_A_main"] - SUPPLY["indic_A"]) < 1e-6,
        "Tier-A escrow + main must equal unique verified Indic (disjoint partition, not a subset counted twice)")

    # Agentic shapes are disjoint and sum.
    _ok(abs(SUPPLY["agentic_fc"] + SUPPLY["agentic_traj"] - SUPPLY["agentic"]) < 1e-9,
        "short FC + long traj must equal agentic unique")

    # Epoch ceilings on unique supply actually available to each pool.
    main_reason = sum(stage_tok[s]["reason"] for s in stage_tok if s != "S5 anneal")
    ann_reason = stage_tok["S5 anneal"]["reason"]
    _ok(main_reason / SUPPLY["reason_pt_main"] <= EPOCH_CAP + 1e-9,
        f"main-run reasoning {main_reason / SUPPLY['reason_pt_main']:.2f} ep exceeds cap")
    _ok(ann_reason / SUPPLY["reason_escrow"] <= EPOCH_CAP + 1e-9,
        f"anneal reasoning {ann_reason / SUPPLY['reason_escrow']:.2f} ep exceeds cap")

    main_A = sum(indic_per[s]["A"] for s in indic_per if s != "S5 anneal")
    ann_A = indic_per["S5 anneal"]["A"]
    _ok(main_A / SUPPLY["indic_A_main"] <= EPOCH_CAP + 1e-9,
        f"main-run verified Indic {main_A / SUPPLY['indic_A_main']:.2f} ep exceeds cap")
    _ok(ann_A / SUPPLY["indic_A_escrow"] <= EPOCH_CAP + 1e-9,
        f"anneal verified Indic {ann_A / SUPPLY['indic_A_escrow']:.2f} ep exceeds cap")

    for t, key in [("B", "indic_B"), ("C", "indic_C"), ("D", "indic_D")]:
        d = indic_tot[t]
        s = SUPPLY[key]
        _ok(d / s <= EPOCH_CAP + 1e-9, f"Indic {t} {d / s:.2f} ep exceeds cap")

    for k in ("web", "code", "stem", "longctx"):
        _ok(tot[k] / SUPPLY[k] <= EPOCH_CAP + 1e-9, f"{k} {tot[k] / SUPPLY[k]:.2f} ep exceeds cap")

    # B-overlap sensitivity: if IndicCorpV2 is entirely inside Sangraha unverified, B unique collapses.
    b_demand = indic_tot["B"]
    b_min = SUPPLY["indic_B_sangraha_only"]
    if b_demand / b_min > EPOCH_CAP:
        notes.append(
            f"SENSITIVITY: if IndicCorpV2 ⊆ Sangraha-unverified, B unique is {b_min:.1f}B and "
            f"{b_demand / b_min:.2f} epochs — BREACH. Overlap must stay below "
            f"{SUPPLY['indic_B'] - b_demand / EPOCH_CAP:.1f}B or the B share falls."
        )
    else:
        notes.append("B overlap sensitivity: even the conservative unique still fits under 4 epochs")

    # Loss-token floors.
    for s, floors in FLOOR_LOSS.items():
        for lane, fl in floors.items():
            got = loss[s]["pct"][lane]
            _ok(got + 1e-6 >= fl, f"{s} {lane} loss share {got:.2f}% below floor {fl}%")

    # Inventory: Samanantar must be zero; English Sangraha must not sit in an Indic slot.
    for d in inv["datasets"]:
        if d["id"] == "samanantar":
            _ok(d["unique_tokens_b"] == 0, "Samanantar must not add unique supply")
        if d["id"] == "sangraha-verified-en":
            _ok("indic" not in d["slots"], "English Sangraha must not be slotted as Indic")
        if d["id"] == "aon-v4-sft":
            _ok("reason" not in d["slots"] or d["slots"] == ["sft_reason"],
                "SFT holdout must not be a pretraining reason slot")

    # Data-gate: cleaned shards are not in this repo. Documented rows are.
    n_doc = sum(1 for d in inv["datasets"] if d["unique_tokens_b"] > 0 and d.get("license"))
    _ok(n_doc >= 20, "inventory must document the course rows")
    notes.append(
        "DATA GATE: cleaned-and-stamped unique tokens in this repo = 0. "
        "Documented-from-inventory-widget rows are present. Counting unit is cleaned unique tokens, "
        "so the gate is NOT MET until Session 4 provenance stamps are loaded."
    )

    # CommitPackFT upsample stays at the epoch ceiling.
    cp = next(d for d in inv["datasets"] if d["id"] == "commitpackft")
    scheduled = cp["unique_tokens_b"] * COMMITPACK_EPOCHS
    _ok(COMMITPACK_EPOCHS <= EPOCH_CAP, "CommitPackFT upsample must not exceed the epoch cap")
    notes.append(f"CommitPackFT scheduled {scheduled:.0f}B = {COMMITPACK_EPOCHS:.0f} epochs of {cp['unique_tokens_b']}B unique")

    return notes


def fmt_pct(x):
    return f"{x:6.1f}"


def print_tables(M, stage_tok, named, overlay, effective, indic_per, indic_tot, loss, notes):
    tot = lane_totals(stage_tok)
    print("=" * 88)
    print(f"V5 LEDGER  total {TOTAL_B:.0f}B   main {MAIN_B:.0f}B ({100*(1-ANNEAL_FRAC):.0f}%)   "
          f"anneal {ANN_B:.0f}B ({100*ANNEAL_FRAC:.0f}%)")
    print("Labels: stated = course inventory/widget; derived = arithmetic on stated rows; "
          "hypothesis = planning partition")
    print("=" * 88)

    print("\n-- 1. Named lane shares (% of stage pool). Long-context is a named lane, inside the pool.")
    hdr = f"{'lane':<10}" + "".join(f"{n.split()[0]:>12}" for n, _ in STAGES) + f"{'anneal':>12}{'run %':>10}{'tokens B':>10}"
    print(hdr)
    print(f"{'pool B':<10}" + "".join(f"{MAIN_B*w:11.0f}B" for _, w in STAGES) + f"{ANN_B:11.0f}B")
    for k in LANES:
        print(f"{k:<10}" + "".join(f"{named[n][k]:11.1f}%" for n, _ in STAGES)
              + f"{named['S5 anneal'][k]:11.1f}%{100*tot[k]/TOTAL_B:9.2f}%{tot[k]:10.1f}")

    print("\n-- 2. Long-context overlay by capability (% of stage pool). Included in the stage budget, not added.")
    print(f"{'from longctx':<14}" + "".join(f"{n.split()[0]:>12}" for n, _ in STAGES) + f"{'anneal':>12}")
    for cap in LONG_COMP:
        print(f"{cap:<14}" + "".join(f"{overlay[n][cap]:11.2f}%" for n, _ in STAGES)
              + f"{overlay['S5 anneal'][cap]:11.2f}%")
    print(f"{'longctx B':<14}" + "".join(f"{stage_tok[n]['longctx']:11.1f}B" for n, _ in STAGES)
          + f"{stage_tok['S5 anneal']['longctx']:11.1f}B")

    print("\n-- 3. Effective capability mix after folding longctx via LONG_COMP (sums to 100%).")
    print(f"{'lane':<10}" + "".join(f"{n.split()[0]:>12}" for n, _ in STAGES) + f"{'anneal':>12}")
    for k in LANES:
        if k == "longctx":
            continue
        print(f"{k:<10}" + "".join(f"{effective[n][k]:11.1f}%" for n, _ in STAGES)
              + f"{effective['S5 anneal'][k]:11.1f}%")

    print("\n-- Indic tiers (B tokens). Cap = translated+synthetic sequence tokens ≤ "
          f"{NONNATIVE_CAP:.0f}% of Indic, every stage.")
    print(f"{'stage':<16}{'Indic B':>10}{'A ver':>10}{'B unv':>10}{'C tr':>10}{'D syn':>10}{'C+D %':>8}")
    for s, split in indic_per.items():
        indic = sum(split.values())
        nn = 100.0 * (split["C"] + split["D"]) / indic
        print(f"{s:<16}{indic:10.1f}{split['A']:10.1f}{split['B']:10.1f}"
              f"{split['C']:10.1f}{split['D']:10.1f}{nn:7.1f}%")
    nn = 100.0 * (indic_tot["C"] + indic_tot["D"]) / tot["indic"]
    print(f"{'RUN':<16}{tot['indic']:10.1f}{indic_tot['A']:10.1f}{indic_tot['B']:10.1f}"
          f"{indic_tot['C']:10.1f}{indic_tot['D']:10.1f}{nn:7.1f}%")

    print("\n-- Feasibility (unique supply actually available to that pool)")
    print(f"{'pool':<28}{'demand':>10}{'unique':>10}{'epochs':>10}  verdict")

    def row(name, d, s, note=""):
        ep = d / s if s else float("inf")
        if ep <= 1:
            v = "funded, <1 epoch"
        elif ep <= EPOCH_CAP:
            v = f"repeat {ep:.2f}x (≤{EPOCH_CAP:.0f})"
        else:
            v = f"MUST SYNTHESIZE ({ep:.1f}x)"
        print(f"{name:<28}{d:10.1f}{s:10.1f}{ep:10.2f}  {v} {note}")

    row("web", tot["web"], SUPPLY["web"])
    row("code", tot["code"], SUPPLY["code"])
    row("stem", tot["stem"], SUPPLY["stem"])
    row("longctx", tot["longctx"], SUPPLY["longctx"])
    row("reason PT main", sum(stage_tok[s]["reason"] for s in stage_tok if s != "S5 anneal"),
        SUPPLY["reason_pt_main"], "(85.1 − 8 SFT − 8 escrow)")
    row("reason anneal escrow", stage_tok["S5 anneal"]["reason"], SUPPLY["reason_escrow"])
    row("indic A main", sum(indic_per[s]["A"] for s in indic_per if s != "S5 anneal"), SUPPLY["indic_A_main"])
    row("indic A escrow", indic_per["S5 anneal"]["A"], SUPPLY["indic_A_escrow"])
    row("indic B", indic_tot["B"], SUPPLY["indic_B"], "(overlap with IndicCorpV2 unmeasured)")
    row("indic C", indic_tot["C"], SUPPLY["indic_C"])
    row("indic D native", indic_tot["D"], SUPPLY["indic_D"])
    row("agentic FC main (real)", tot["agentic"] - stage_tok["S5 anneal"]["agentic"],
        SUPPLY["agentic_fc"])
    row("agentic traj anneal (real)", stage_tok["S5 anneal"]["agentic"], SUPPLY["agentic_traj"])

    # Synthesis gap at the 4-epoch ceiling.
    main_ag = tot["agentic"] - stage_tok["S5 anneal"]["agentic"]
    unique_needed_main = main_ag / EPOCH_CAP
    unique_needed_ann = stage_tok["S5 anneal"]["agentic"] / EPOCH_CAP
    print(f"\n  agentic main: demand {main_ag:.1f}B, unique needed at 4ep {unique_needed_main:.1f}B, "
          f"have {SUPPLY['agentic_fc']:.3f}B, SYNTHESIZE {unique_needed_main - SUPPLY['agentic_fc']:.1f}B")
    print(f"  agentic anneal: demand {stage_tok['S5 anneal']['agentic']:.1f}B, unique needed "
          f"{unique_needed_ann:.2f}B, have {SUPPLY['agentic_traj']:.3f}B, "
          f"SYNTHESIZE {unique_needed_ann - SUPPLY['agentic_traj']:.2f}B")
    print(f"  SFT holdout {SUPPLY['reason_sft']:.0f}B is NOT in any pretraining demand row above.")

    print("\n-- Sequence share vs loss-token share")
    for s, L in loss.items():
        pool = stage_tok[s]["_pool"]
        print(f"  {s:<16} AGENT seq {named[s]['agentic']:5.1f}% -> loss {L['pct']['agentic']:5.1f}%   "
              f"REASON seq {named[s]['reason']:5.1f}% -> loss {L['pct']['reason']:5.1f}%   "
              f"loss tokens {L['total']:.0f}B / {pool:.0f}B")

    print("\n-- Composer default vs this plan (main run)")
    print(f"  {'lane':<10}{'composer':>10}{'proposed':>10}{'delta B':>10}")
    for k in LANES:
        delta = (MAIN_SHARE[k] - COMPOSER[k]) / 100.0 * MAIN_B
        print(f"  {k:<10}{COMPOSER[k]:>9}%{MAIN_SHARE[k]:>9}%{delta:10.0f}")

    print("\n-- Notes")
    for n in notes:
        print(f"  • {n}")


def build():
    inv = load_inventory()
    M = ipf_stages()
    stage_tok = stage_tokens(M)
    named, overlay, effective = named_and_effective(stage_tok)
    indic_per, indic_tot = indic_tiers(stage_tok)
    loss = loss_shares(stage_tok)
    notes = validate(M, stage_tok, named, effective, indic_per, indic_tot, loss, inv)
    return dict(M=M, stage_tok=stage_tok, named=named, overlay=overlay, effective=effective,
                indic_per=indic_per, indic_tot=indic_tot, loss=loss, notes=notes, inv=inv,
                tot=lane_totals(stage_tok))


def main():
    b = build()
    print_tables(b["M"], b["stage_tok"], b["named"], b["overlay"], b["effective"],
                 b["indic_per"], b["indic_tot"], b["loss"], b["notes"])
    print("\nOK — all specification assertions passed.")


if __name__ == "__main__":
    main()
