# Indic allocation

Reproduce: `python scripts/indic.py`, `python scripts/indic_gated.py`, `python scripts/indic_fertility.py`.

Lane demand = 17% of 2,910B + 28% of 90B = **519.9B**.

## Provenance split (sequence-token cap)

Main A28/B32/C2/D38. Anneal A50/B10/C2/D38. **C+D = 40% of Indic sequence tokens in every stage.** Tested. Not a weighted fluency score.

C is existing parallel (BPCC Indic-side 6.8B). Generated translation is D.

## Two card-level corrections

1. Sangraha verified `eng` = 12,759.9M. Verified Indic = 51.5B.
2. Synthetic native/Latin row counts are 1:1. Native-script synthetic = 81.4B.

Usable Sangraha native-script (A+B_sangraha+D_native) = **157.2B**. Adding IndicCorpV2 20.9B and BPCC 6.8B (no per-language table) raises unique to ~187B.

## Temperature sampling

`p_i ∝ S_i^α`, water-filled against 4 × per-language supply.

α = 1 is uniform-epoch allocation (every language at 3.31 on the 157.2B Sangraha-native pool). **α\* = 0.72** is the largest α at which Punjabi (weakest MILU language, 6.20B) already hits 4 epochs. XLM-R's published α = 0.3 pins 17 of 22 languages to the ceiling here; we differ because the binding constraint differs.

Per-language verified share runs 5% (Assamese) to 46% (Bengali). A global A28/B32/C2/D38 is an average, not a per-language specification.

## Gate

Supply is 14 languages and then a **24× cliff** (Assamese 6.16B → Sindhi 0.26B). Entry: unique ≥ 1.30B (1% of the lane / 4 epochs).

**Funded 14:** Hindi, Bengali, Tamil, Gujarati, Telugu, Malayalam, Urdu, Marathi, Kannada, Sanskrit, Nepali, Odia, Punjabi, Assamese.

**Gated 8:** Sindhi, Maithili, Konkani, Manipuri, Bodo, Kashmiri, Santali, Dogri.

MILU 10/10. IndicGenBench **14/29** (10 languages have zero tokens anywhere; 5 more are gated). Claiming 29/29 is the same wishful accounting this plan exists to avoid. Re-entry is automatic on crossing 1.30B.

## Fertility

Stated MUTANT-Indic: Hindi 1.375 tok/word, Malayalam 3.025. Equal-token allocation gives Malayalam ~45% of Hindi's content. Content-weighted α still cannot reach parity: high-fertility languages hit the 4-epoch token cap first. Measured in `indic_fertility.py`: at α=0.30, Malayalam/Hindi content ratio = 0.26. Content parity is not reachable at this lane size.

## Romanized Indic

The Latin half of Sangraha synthetic is in-scope as a **1% sub-lane** (~5B) for input robustness (Dakshina), not for generation fluency. It is not counted in the 81.4B native D supply.
