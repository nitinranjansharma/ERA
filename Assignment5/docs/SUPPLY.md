# Supply audit

Labels: **stated** — course inventory widget or dataset card. **derived** — arithmetic on stated rows. **measured** — queried or computed in this repo. **gated** — real, no number without an MoU. **hypothesis** — planning partition.

Reproduce: `python scripts/supply_audit.py`, `data/inventory.json`.

## What auditing changed

| claim as given | used here | direction |
|---|---|---|
| Indic verified 64.3B | 51.5B (12.76B English split) | ↓ derived |
| Indic synthetic 162.7B | 81.4B native-script | ↓ derived |
| C = BPCC 3B + Samanantar 2B | Samanantar ⊆ BPCC; C = 6.8B Indic-side | mixed |
| STEM 250B (composer) | 146B inventory rows | ↓ stated |
| Reasoning 85.1B all in PT | 8B SFT + 8B escrow + 69.1B PT-main | partition |

## Live queries (measured 2026-10-23)

- `ai4bharat/sangraha`: partial=True, 177,355,164 / 268,356,447 rows, 65 splits, 1.018 TB `num_bytes_memory`.
- `ai4bharat/IndicCorpV2`: 37.6% converted. Extrapolated 167GB / 20.9B published tokens = **8.0 bytes/token**.
- `ai4bharat/BPCC`: HTTP 401 Unauthorized. Stated 6.8B Indic-side stands.

## Sangraha synthetic is one encyclopedia rendered 28 ways

AI4Bharat describe the synthetic config as Wikimedia English translated to 14 Indic languages and romanized. Native/Latin pairs have identical row counts. Distinct source content is ~4.5B English Wikipedia words, not 162.7B tokens of Indic knowledge. Epoch figures count token strings; the repetition ceiling in arXiv:2305.16264 is about repeated **content**. T2 is therefore per-subject on MILU (India-specific vs general), not a single "synthetic is worse" average.

## Disjoint partitions (hypothesis, then tested)

- Verified Indic 51.5462B = 15.0B escrow + 36.5462B main. Not "top 18% of verified" counted twice.
- Reasoning 85.1B = 8B SFT holdout + 8B anneal escrow + 69.1B PT-main. The 12B figure from the previous draft is unused.
- Agentic 0.627B = 0.267B short FC + 0.360B long traj.

`tests/test_mixture_invariants.py` asserts the sums.

## B overlap

Sangraha unverified 24.3B + IndicCorpV2 20.9B = 45.2B used as B unique. If IndicCorpV2 is contained in Sangraha unverified, B is 24.3B and 160.8B demand = 6.62 epochs — a breach. Overlap must stay below 5.0B or the B share falls. The ledger prints this every run.

## Gated (named, no number)

NCERT/SCERT, NPTEL/IGNOU, India Code, e-Gazette, Digital Sansad, Census, RBI DBIE, National Archives, KrishiKosh. Programme dependencies, not supply. No budget line rests on them.
