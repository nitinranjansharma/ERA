"""Lecture-style proofs 1 and 2 (CPU). Proof 3 skips without CUDA."""

from __future__ import annotations

from helpers.proofs import proof_gradients_match, proof_invert_stack, proof_memory_vs_depth


def test_proof_1_stacks_invert():
    rows = proof_invert_stack()
    assert rows
    assert all(row["ok"] for row in rows), rows


def test_proof_2_gradients_match_reference():
    rows = proof_gradients_match()
    assert rows
    assert all(row["ok"] for row in rows), rows


def test_proof_3_skips_or_runs_on_gpu():
    result = proof_memory_vs_depth(depths=(2, 4))
    if result["skipped"]:
        assert "GPU" in result["reason"]
    else:
        assert 2 in result["table"]
        assert result["growth"]["baseline"] > result["growth"]["midpoint"]
