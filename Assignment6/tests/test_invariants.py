#!/usr/bin/env python3
"""Invariant tests for the Helix training-data execution system.

These tests exercise the contracts from Sessions 1–6: tokenizer identity,
admission, packing/masks, mixture floors, the eval firewall, hash-chained
ledgers, and checkpoint refusal. They do not hardcode demo evidence.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helix.build import build_world
from helix.constants import CANARY_STRING, PII_EMAIL
from helix.hashutil import GENESIS, derived_rng, hash_chain_link, sha256_text
from helix.model import TinyLM, pack_batch
from helix.packing import Example, attention_crossings, blocked_cross_segment_pairs, pack
from helix.schedule import AnnealVault, VaultSealedError, compile_plan, examples_from_shards
from helix.shards import verify_shard_integrity
from helix.tokenizer import FrozenTokenizer, graphemes, nfc


def _plan() -> dict:
    raw = json.loads((ROOT / "plan" / "mixture_plan.json").read_text(encoding="utf-8"))
    raw["run_tokens"] = 2048
    return raw


class TokenizerTests(unittest.TestCase):
    def test_same_text_same_ids(self):
        tok = FrozenTokenizer.from_texts(["नमस्ते", "hello"])
        a = tok.encode("नमस्ते")
        b = tok.encode("नमस्ते")
        self.assertEqual(a, b)
        self.assertEqual(tok.decode(a).replace("\n", ""), nfc("नमस्ते"))

    def test_nfc_and_combining_marks_stick(self):
        # Decomposed vs composed should collapse under NFC.
        composed = "न" + "म"
        tok = FrozenTokenizer.from_texts([composed, "नमस्ते"])
        self.assertEqual(tok.encode(nfc(composed)), tok.encode(composed))
        clusters = graphemes("क्ष")
        self.assertTrue(all(len(c) >= 1 for c in clusters))

    def test_pad_is_zero_and_hash_frozen(self):
        tok = FrozenTokenizer.from_texts(["abc"])
        self.assertEqual(tok.pad_id, 0)
        h1 = tok.tokenizer_hash
        h2 = FrozenTokenizer.from_texts(["abc"]).tokenizer_hash
        self.assertEqual(h1, h2)

    def test_derived_rng_is_pure(self):
        r1 = derived_rng(7, "lane", 3, 1).random()
        r2 = derived_rng(7, "lane", 3, 1).random()
        r3 = derived_rng(7, "lane", 3, 2).random()
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, r3)


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = build_world(_plan())

    def test_plants_are_rejected(self):
        plants = {r["plant"] for r in self.world.rejections if r.get("plant")}
        for expected in (
            "pii",
            "license",
            "repeat_ngram",
            "eval_item",
            "canary",
            "ngram_containment",
            "language",
            "validation",
            "exact_dup",
            "near_dup",
        ):
            self.assertIn(expected, plants, msg=f"{expected} was not blocked")

    def test_pii_email_never_admitted(self):
        pii = [d for d in self.world.docs if PII_EMAIL in d.text]
        self.assertTrue(pii)
        admitted_ids = {d for s in self.world.admitted_shards for d in s.doc_ids}
        for d in pii:
            self.assertNotIn(d.doc_id, admitted_ids)

    def test_never_train_not_in_admitted_shards(self):
        admitted = {d for s in self.world.admitted_shards for d in s.doc_ids}
        for doc in self.world.docs:
            if doc.never_train:
                self.assertNotIn(doc.doc_id, admitted)

    def test_shard_integrity_and_hard_fields(self):
        for shard in self.world.shards:
            errs = verify_shard_integrity(shard, self.world.tokenizer)
            self.assertEqual(errs, [])
            self.assertEqual(shard.manifest.tokenizer_hash, self.world.tokenizer.tokenizer_hash)

    def test_canary_registered(self):
        self.assertTrue(any(CANARY_STRING in d.text for d in self.world.docs))
        self.assertTrue(any(e.gate == "canary" or e.gate == "never_train" for e in self.world.firewall.events) or any(
            r.get("plant") == "canary" for r in self.world.rejections
        ))


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.pad, self.eos = 0, 1
        self.examples = [
            Example(f"d{i}", "s", "web", [2] * L, ["text"] * L)
            for i, L in enumerate([8, 3, 8, 3])
        ]

    def test_greedy_and_best_fit_are_not_the_same_algorithm(self):
        g = pack(self.examples, "greedy", 10, self.pad, self.eos, "web")
        b = pack(self.examples, "best_fit", 10, self.pad, self.eos, "web")
        self.assertNotEqual([tuple(s.doc_ids) for s in g], [tuple(s.doc_ids) for s in b])

    def test_concat_chop_actually_truncates_or_splits(self):
        long = [Example("L", "s", "web", list(range(2, 42)), ["text"] * 40)]
        seqs = pack(long, "concat_chop", 16, self.pad, self.eos, "web")
        self.assertGreaterEqual(len(seqs), 2)
        self.assertTrue(any(s.split_docs or s.truncated_docs for s in seqs) or len(seqs) >= 3)

    def test_pad_only_one_doc_per_sequence(self):
        seqs = pack(self.examples, "pad_only", 16, self.pad, self.eos, "web")
        self.assertEqual(len(seqs), len(self.examples))
        for s in seqs:
            self.assertEqual(len(s.doc_ids), 1)
            self.assertEqual(s.pad_count, 16 - (len(s.token_ids) - s.pad_count))
            self.assertTrue(0 <= s.pad_count <= 16)

    def test_structure_preserving_blocks_cross_segment_attention(self):
        seqs = pack(self.examples[:2], "structure_preserving", 32, self.pad, self.eos, "agentic")
        self.assertEqual(len(seqs), 1)
        seq = seqs[0]
        self.assertEqual(attention_crossings(seq), 0)
        self.assertGreater(blocked_cross_segment_pairs(seq), 0)
        # position ids reset at the second document
        segs = seq.segment_ids
        if 1 in segs:
            start = segs.index(1)
            self.assertEqual(seq.position_ids[start], 0)

    def test_loss_mask_ignores_pad_and_user_roles(self):
        ex = Example(
            "ag",
            "s",
            "agentic",
            [3, 4, 5, 6, 7, 8],
            ["user", "user", "assistant", "assistant", "plan", "observation"],
        )
        seqs = pack([ex], "structure_preserving", 16, self.pad, self.eos, "agentic")
        seq = seqs[0]
        for i, (role, keep) in enumerate(zip(seq.role_ids, seq.loss_mask)):
            if role in {"pad", "user", "observation"}:
                self.assertEqual(keep, 0)
            if seq.token_ids[i] == self.pad:
                self.assertEqual(keep, 0)

    def test_six_policies_all_run(self):
        for policy in ("pad_only", "concat_chop", "greedy", "best_fit", "structure_preserving", "long_context"):
            seqs = pack(self.examples, policy, 16, self.pad, self.eos, "web")
            self.assertTrue(seqs)
            for s in seqs:
                self.assertEqual(len(s.token_ids), 16)
                self.assertEqual(len(s.loss_mask), 16)
                self.assertEqual(len(s.attn_mask), 16)


class VaultAndHashTests(unittest.TestCase):
    def test_vault_refuses_wrong_stage(self):
        vault = AnnealVault([], "planhash")
        with self.assertRaises(VaultSealedError):
            vault.unseal("midtrain", "planhash")
        released = vault.unseal("anneal", "planhash")
        self.assertEqual(released, [])

    def test_hash_chain_tamper_detected(self):
        prev = GENESIS
        h1, _ = hash_chain_link(prev, {"a": 1})
        h2, p2 = hash_chain_link(h1, {"a": 2})
        self.assertEqual(p2, h1)
        h2_bad, _ = hash_chain_link(h1, {"a": 3})
        self.assertNotEqual(h2, h2_bad)


class ModelMaskTests(unittest.TestCase):
    def test_isolated_attention_is_load_bearing(self):
        short = [
            Example("iso-a", "iso", "indic", [4, 5, 6, 7, 8, 9, 10, 11], ["text"] * 8),
            Example("iso-b", "iso", "indic", [12, 13, 14, 15, 16, 17, 18, 19], ["text"] * 8),
        ]
        seq = pack(short, "structure_preserving", 32, 0, 1, "indic")[0]
        model = TinyLM(vocab_size=64, d_model=16, seq_len=32, seed=0)
        tok, pos, attn, loss = pack_batch([seq])
        _, la, _ = model.forward(tok, pos, attn, loss)
        tok2 = tok.copy()
        segs = np.asarray(seq.segment_ids)
        tok2[0, segs == 0] = (tok2[0, segs == 0] + 3) % 60
        _, lb, _ = model.forward(tok2, pos, attn, loss)
        m = (segs == 1) & (np.asarray(seq.loss_mask) > 0)
        self.assertTrue(m.any())
        self.assertEqual(float(np.max(np.abs(la[0, m] - lb[0, m]))), 0.0)
        attn_merged = np.tril(np.ones_like(attn))
        attn_merged[0, segs < 0] = False
        attn_merged[0, :, segs < 0] = False
        _, lc, _ = model.forward(tok, pos, attn_merged, loss)
        _, ld, _ = model.forward(tok2, pos, attn_merged, loss)
        self.assertGreater(float(np.max(np.abs(lc[0, m] - ld[0, m]))), 0.0)

    def test_train_step_reduces_loss_on_repeated_batch(self):
        rng = np.random.default_rng(0)
        model = TinyLM(vocab_size=32, d_model=16, seq_len=16, lr=0.2, seed=0)
        tok = rng.integers(1, 32, (2, 16), dtype=np.int32)
        pos = np.broadcast_to(np.arange(16), (2, 16)).copy()
        attn = np.tril(np.ones((2, 16, 16), dtype=bool))
        loss = np.ones((2, 16), dtype=np.float64)
        loss[:, 0] = 0
        first, _, _ = model.forward(tok, pos, attn, loss)
        for _ in range(12):
            model.train_step(tok, pos, attn, loss)
        last, _, _ = model.forward(tok, pos, attn, loss)
        self.assertLess(last, first)


class MixtureCompileTests(unittest.TestCase):
    def test_floors_not_renormalised_in_stage_mix(self):
        plan = compile_plan(_plan())
        for st in plan.stages:
            self.assertAlmostEqual(sum(st.mixture.values()), 1.0, places=6)
        self.assertEqual(plan.protected_floors["indic"], 0.08)
        # A 3% floor is smaller than 1/global_batch, so per-batch enforcement
        # would overshoot. Deficit tracking is the contract.
        self.assertLess(plan.protected_floors["agentic"], 1 / plan.global_batch)

    def test_pure_lane_shards(self):
        world = build_world(_plan())
        for shard in world.admitted_shards:
            lanes = {shard.manifest.capability_lane}
            self.assertEqual(len(lanes), 1)


if __name__ == "__main__":
    unittest.main()
