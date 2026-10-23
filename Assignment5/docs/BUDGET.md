# Budget proposal

Reproduce: `python scripts/ledger.py`. Design point 3T, 97/3 split. Epoch cap 4 (arXiv:2305.16264).

## Main run 2,910B vs composer default

| lane | composer | proposed | demand | unique | epochs |
|---|---|---|---|---|---|
| web | 34 | 34 | 989B | 4,691B | 0.21 |
| code | 24 | 24 | 698B | 1,103B | 0.63 |
| indic | 16 | 17 | 495B | 187B partitioned | 2.8 |
| stem | 12 | 12 | 349B | 146B | 2.39 |
| reason | 6 | 9 | 262B | 69.1B PT-main | 3.79 |
| longctx | 6 | 2 | 58B | 100B | 0.58 |
| agentic | 2 | 2 | 58B | 0.267B FC | synthesize 14.3B |

Long-context 6%→2% frees **116B**. Literature requirement for 128K retrieval is 0.5–5B (arXiv:2402.10171). 58B is still 12× the upper bound.

Reasoning 6%→9% is stated jointly with 12% STEM (=21%), in the Llama 3 math+reasoning band.

## Anneal 90B, sized as its own pool

The course widget applies anneal shares to the full run size and overstates anneal demand ~30×. Demand is share × 90B.

| lane | share | demand | unique | epochs |
|---|---|---|---|---|
| indic | 28 | 25.2B | 15.0B A-escrow | 0.84 on A |
| reason | 18 | 16.2B | 8.0B escrow | 2.02 |
| code | 20 | 18.0B | 1,103B | 0.02 |
| agentic | 8 | 7.2B | 0.36B traj | 20 → synthesize 1.44B |
| longctx | 8 | 7.2B | 100B | 0.07 |
| stem | 10 | 9.0B | 146B | 0.06 |
| web | 8 | 7.2B | 4,691B | 0.00 |

Same 8% agentic share is a fantasy in the main run and tractable in the anneal because the pool is 90B rather than 2.91T.

## Indic tiers vs composer default

Composer A40/B25/C20/D15: C demand 99B vs 6.8B unique = 14 epochs. Impossible.

Proposed main A28/B32/C2/D38: C+D = 40% sequence tokens, every stage. Cap is a sequence-token cap, tested in `tests/test_mixture_invariants.py`.

## Funded / programme / fallback

| lane | asked | max at 4 epochs | status |
|---|---|---|---|
| web | 34% | 645% | funded; adjustment variable; hard floor 25% |
| code | 24% | 152% | funded |
| indic | 17% | ~21% | funded |
| stem | 12% | 20% | funded |
| reason | 9% | 10.6% of PT unique | funded, tight |
| longctx | 2% | 14% | funded |
| agentic | 2% | 0.037% | programme, 54× short on unique |

Web is 34% because it is the only lane with slack, not because 34% is optimal.

Public-only rebuild: reasoning falls to 1.0%. 78B of 85.1B is V4 AON.
