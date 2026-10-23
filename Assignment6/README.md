# Helix — Training Data Execution System (ERA V5, Session 6)

Session 5 produced a mixture plan. A plan is only useful if the training system can execute it accurately, consistently, and reproducibly. Helix is that system: a small but complete data path that **proves what it consumed**, why it consumed it, what the model learned from it, and how the run can be reconstructed.

```bash
python run_demo.py            # full demonstration, writes submission_artifacts/
python -m unittest tests.test_invariants
python verify_evidence.py     # recompute every claim from artifacts alone
python audit.py --step 7
python audit.py --interval 8 12
python audit.py --branches
```

Python 3.10+, `numpy` only (`pip install -r requirements.txt`). No network, no downloads.

`run_demo.py` also runs the 21 invariant tests. The process **exits non-zero** if any invariant fails.

---

## What this is

```
documents
  → NFC grapheme tokenizer (frozen, hashed)
  → document-level admission (PII, license, language, repeats, dedup)
  → eval / validation firewall
  → immutable tokenized shards + manifests
  → mixture timeline (curriculum, floors, anneal vault)
  → packing (six real policies, loss / attention / position)
  → OPUS tournament (accept / reject / defer / protected override)
  → batches → tiny masked-attention LM
  → consumption ledger (hash-chained) + learning ledger
  → checkpoint bound to ledger offset
  → crash → resume (world size 2 → 4)
  → replay → fork → audit
```

The corpus is synthetic and tiny. The goal is not scale. The goal is that a grader can regenerate `submission_artifacts/` from one command and **recompute** every PASS from the files, not from a hardcoded boolean.

---

## Design decisions (where Helix is not a widget)

**1. Intent then delivery, not a cursor that pretends to be a log.**
Before a step is trained, the planner can preview the next global batch as a pure function of `(plan, corpus, tokenizer, seed, planner state)`. After the step, a delivery record is appended to a hash-chained ledger. Resume restores planner state from the checkpoint and checks that the next delivery hash equals the preview taken *before* the crash. A shadow never-crashed run is compared at the same step so a “self-comparison of a re-partitioned list” cannot pass vacuously.

Rank is derived at load: `rank = sample_index % world_size`. The demo crashes at 2 ranks and resumes at 4. The global batch identity is what must match, not the per-rank assignment.

**2. The attention mask is in the arithmetic.**
Disallowed `(query, key)` pairs are set to `-inf` before softmax. Rewriting document 1 leaves document 2’s per-token losses **bit-identical**. Merging segment ids (the control) makes them move. Without the control, a model that simply never looks across the boundary would also “pass”.

**3. Protected floors are a run-level deficit, not a per-batch quota.**
With `global_batch = 4` the smallest expressible share is 25%. A 3% agentic floor enforced per batch would become 25% and starve every other lane. Helix tracks `floor * tokens_so_far - delivered` and lets OPUS issue a `protected_override` when a protected lane is in deficit. Floors are never “raised then renormalised”, which would put them straight back under the floor.

**4. Admission is document-first; shards are sealed.**
Gating at shard level lets one near-duplicate destroy an entire shard and silently delete a lane. Documents are screened, exclusions are recorded, then admitted documents are packed into immutable shards whose `content_hash` is SHA-256 over the token bytes. Modify the tokens and it is a different shard.

**5. Anneal data lives in a vault, not a boolean.**
Reserved Indic / agentic / reasoning documents are sealed with `sha256("vault:{plan_hash}:anneal")`. The midtrain stage cannot list them. Only the anneal stage produces the matching key. A negative control tries to unseal from `midtrain` and must raise.

**6. OPUS is top-k over an overprovisioned pool, with four recorded outcomes.**
A fixed score threshold in a high-dimensional proxy rejects everything. Helix scores candidates by unigram surprise (updated from loss-bearing tokens actually seen), takes top-k, defers the mid-score remainder, rejects the tail, and rescues starving protected lanes *before* top-k. Rejected clean data is kept in the audit trail; it is not deleted.

**7. Checkpoints refuse to load when they are wrong.**
Three hashes fail differently: `model_weight_hash`, `optimizer_hash` (Nesterov velocity + step counter), `checkpoint_hash` (step, branch, ledger offset, planner cursors, tokenizer/plan identity). Nine induced corruptions must be refused. Dropping momentum buffers leaves the weights valid at restore and **diverges after three steps** — so “resume restores the optimizer” is a load-bearing sentence, not a copy of the config.

---

## Packing policies (six algorithms, not six names)

| Policy | Behaviour |
|---|---|
| `pad_only` | One example per window, pad the rest. Preserves structure, wastes compute. |
| `concat_chop` | Join with EOS, cut fixed windows. Efficient for web pretraining; splits documents. |
| `greedy` | First-fit: place each example in the first bin with room. Order-dependent. |
| `best_fit` | Best-fit decreasing: tightest remaining hole. Distinct from greedy (tested). |
| `structure_preserving` | Do not split structured traces; **segment-isolated causal attention**; position ids reset at document boundaries. Used for Indic, agentic, reasoning. |
| `long_context` | Long documents only; shorts are not mixed in. Unused positions are an explicit cost. |

Loss masks: pad, user, and tool observations are context. Assistant, plan, tool calls, and plain pretrain tokens receive next-token loss. Position 0 of an isolated segment has no in-segment history and is not trained.

Pure-lane shards: an Indic document is never smuggled into a web shard. A floor you cannot audit from the ledger is not a floor.

---

## Tokenizer

NFC normalization, then grapheme clusters (base character + combining marks). Specials occupy the first ids; `<pad>` is id 0. The tokenizer hash covers algorithm + vocab + specials. The same raw text always produces the same ids. This is Indic-safe by construction: a Devanagari matra cannot be split off its consonant because combining marks never start a cluster.

---

## Ledgers

**Consumption** (append-only, hash-chained): run, branch, global step, rank, packed sample ids, shard ids, token spans, loss-mask hash, attention/position policy, lane, stage, tokenizer version, OPUS decision id, batch content hash.

**Learning**: per packed sample, mean loss / perplexity / grad norm / classification; per loss-bearing token, id, position, doc, shard, lane, role, loss, ppl. Full token traces are affordable here; at 3T tokens they would not be — the demo still stores the sample-level aggregates a large run would keep.

**experiment = checkpoint + optimizer state + data stream + code/config.** Silently changing any one of these makes a comparison untrustworthy. Replay restores an old checkpoint and re-feeds the historical stream. Fork restores the same checkpoint, declares `opus_disabled`, and records the divergence point.

---

## Layout

```
run_demo.py                 one command, full demonstration
verify_evidence.py          recompute claims from artifacts (no trainer import)
audit.py                    which shards trained a step / interval / branch
plan/mixture_plan.json      stages, shares, floors, anneal reserves, packing map
helix/
  hashutil.py               canonical JSON, SHA-256, hash chain, derived RNG
  tokenizer.py              NFC grapheme tokenizer, frozen hash
  corpus.py                 6-lane corpus + planted failures
  clean.py                  PII, license, script, repeats, exact + Jaccard dedup
  firewall.py               train / validate / test registry, four gates
  shards.py                 sealed shards, manifests, integrity check
  packing.py                six policies, masks, position ids, span certificates
  schedule.py               mixture compiler, deficit floors, anneal vault
  opus.py                   tournament selector, four decision types
  model.py                  masked-attention LM, Nesterov SGD
  ledger.py                 dual hash-chained logs
  trainer.py                loop, checkpoint, restore, corruption probes
  evidence.py               evidence.json / evidence.md from measured values
  build.py                  corpus → admitted shards
tests/test_invariants.py
```

---

## Planted failures (negative controls)

A gate that is not load-bearing looks green. Each plant isolates one gate:

| Plant | Gate that must fire |
|---|---|
| `plant-pii-email` / `plant-pii-phone` | PII screen |
| `plant-license-nc` | license allowlist |
| `plant-ngram-repeat` (`"xy"*60`, unique so Jaccard is blind) | repeated n-gram run |
| `plant-exact-dup` / `plant-near-dup` | exact hash / shingle Jaccard |
| `plant-eval-gsm` | never-train / test registry |
| `plant-canary` | canary string |
| `plant-containment-host` | 8-gram containment (Jaccard against the eval item is low) |
| `plant-indic-wrong-script` | language/script validator |
| `plant-val-holdout` | validation may be read, never loss-bearing |

---

## Artifacts

```
submission_artifacts/
  run.log
  evidence.json
  evidence.md
  performance.json
  manifests/
  shards/
  ledgers/          consumption, learning, tokens, opus, firewall, fork
  checkpoints/
  reports/          packing, mixture, replay, audit, isolation, corruption
```

`evidence.json` records the values compared, the artifact paths, and `passed` computed from those values. `verify_evidence.py` reads the same files independently.

---

## Measured on this run

Regenerated by `python run_demo.py` (numbers will match the artifacts, not this paragraph, if you change the seed).

| Check | Result |
|---|---|
| Tokenizer hash unique across 24 shards | PASS |
| 10 planted failures blocked (PII, license, repeats, exact/near dup, eval item, canary, 8-gram containment, wrong script, validation) | PASS |
| Attention isolation: rewrite doc 1 → doc 2 loss delta **0**; merged-mask control moves **4.9e-2** | PASS |
| Greedy vs best-fit bins on `[8,3,8,3]` cap 10 | `[8]+[3,3]+[8]` vs `[8]+[8]+[3,3]` |
| Short-doc concat vs pad-only | concat utilization **>** pad-only (0.25) |
| Protected floors | indic 16.1% ≥ 8%, agentic 8.6% ≥ 3%, reasoning 21.5% ≥ 5% |
| All four curriculum stages executed, anneal vault sealed against midtrain | PASS |
| OPUS statuses | accepted, rejected, deferred, protected_override (272 decisions) |
| Crash at step 16, resume world size 2 → 4, preview = resumed = shadow | PASS |
| Replay steps 8–11 sample ids match original | PASS |
| Fork at step 8 with `opus_disabled` diverges | PASS |
| 9 checkpoint corruptions refused; dropped Nesterov moments diverge after 3 steps | PASS |
| Hash-chained consumption ledger, 96 deliveries, 4689 token traces | PASS |

---

## Scoring map (1,000 points)

| Area | How Helix proves it |
|---|---|
| End-to-end execution | `python run_demo.py` regenerates the tree; log contains the required sequence |
| Shards / tokenizer | Content hash over token bytes; one tokenizer hash on every manifest |
| Packing / masks | Six policies, isolation experiment with a merged-mask control |
| Mixture / floors / OPUS | Planned vs actual shares; four OPUS statuses; vault unseal test |
| Ledgers | Hash chain verified; token loss linked to doc/shard/lane/role |
| Checkpoint / crash / resume / replay / fork | Preview = resumed batch = shadow batch; replay ids match; fork diverges |
| Eval firewall | Plants blocked; no never-train id in an admitted shard |
| Throughput | `performance.json` reconstructable from the consumption ledger |
| Tests / evidence / docs | `tests/test_invariants.py`, `verify_evidence.py`, this README |

A subsystem receives credit only when its result is supported by reproducible evidence. If resume skipped or repeated a batch, if replay hashes diverged, if eval data entered a loss-bearing window, or if utilization cannot be reconstructed from the ledger, that section fails — by construction of the verifier, not by inspection of a screenshot.
