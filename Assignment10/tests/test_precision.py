"""Independent checks of FP32 / BF16 / OCP E4M3FN encodings of 0.1."""

from __future__ import annotations

import math
import struct

import numpy as np
import torch

from precision import (
    BF16,
    FP8_E4M3FN,
    FP32,
    analyze_decimal_0p1,
    decode_int,
    encode_rational,
    fp32_bits_numpy,
    fp32_bits_struct,
    ocp_e4m3fn_test_vectors,
)
from fractions import Fraction


def test_fp32_0p1_matches_ieee_references():
    enc = encode_rational(Fraction(1, 10), FP32)
    packed = struct.unpack(">I", struct.pack(">f", 0.1))[0]
    assert enc.integer == packed == 0x3DCCCCCD
    assert enc.integer == fp32_bits_struct(0.1) == fp32_bits_numpy(0.1)
    assert enc.grouped_binary == "0 01111011 10011001100110011001101"
    assert abs(enc.represented_value - np.float32(0.1).astype(np.float64)) < 1e-15


def test_bf16_0p1_matches_torch():
    enc = encode_rational(Fraction(1, 10), BF16)
    t = torch.tensor(0.1, dtype=torch.float32).to(torch.bfloat16)
    bits = int(t.view(torch.int16).item()) & 0xFFFF
    assert enc.integer == bits == 0x3DCD
    assert abs(enc.represented_value - float(t.to(torch.float32))) < 1e-12


def test_fp8_e4m3fn_0p1_matches_torch():
    enc = encode_rational(Fraction(1, 10), FP8_E4M3FN)
    t = torch.tensor(0.1, dtype=torch.float32).to(torch.float8_e4m3fn)
    bits = int(t.view(torch.uint8).item()) & 0xFF
    assert enc.integer == bits == 0x1D
    assert abs(enc.represented_value - 0.1015625) < 1e-12
    assert enc.grouped_binary == "0 0011 101"


def test_fp8_convention_is_ocp_fn():
    # +1.0
    assert decode_int(0x38, FP8_E4M3FN) == 1.0
    # max finite 448, not inf
    assert decode_int(0x7E, FP8_E4M3FN) == 448.0
    # NaN only at 0x7F
    assert math.isnan(decode_int(0x7F, FP8_E4M3FN))
    # exp=all-ones with other mantissa is finite (256)
    assert decode_int(0x78, FP8_E4M3FN) == 256.0


def test_ocp_table_vectors():
    vectors = ocp_e4m3fn_test_vectors()
    assert all(v["ok"] for v in vectors)


def test_analyze_payload_passes_independent_checks():
    payload = analyze_decimal_0p1()
    assert payload["passed"]
    assert payload["matches"]["fp32_matches_struct"]
    assert payload["matches"]["bf16_matches_torch"]
    assert payload["matches"]["fp8_matches_torch"]
    rec = payload["recommendation"]["recommended"]
    assert rec in {"fp32", "bf16_mixed_fp32_master"}


def test_encoder_is_not_its_only_check():
    """The derivation and the reference library must be independent sources."""
    derived = encode_rational(Fraction(1, 10), FP32).integer
    reference = struct.unpack(">I", struct.pack(">f", 0.1))[0]
    # struct.pack is a C library conversion, not our encoder.
    assert derived == reference
    assert encode_rational is not struct.pack
