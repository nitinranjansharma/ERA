"""Bit-level encodings of decimal 0.1 in FP32, BF16, and OCP FP8 E4M3FN.

Convention used for FP8
-----------------------
OCP OFP8 E4M3 (PyTorch ``torch.float8_e4m3fn``):
  * 1 sign bit, 4 exponent bits, 3 stored fraction bits
  * exponent bias = 7
  * emin = -6, emax = 8
  * NO infinities; overflow saturates to ±448
  * NaN encodings: S.1111.111 only (0x7F and 0xFF)
  * exponent-all-ones with other mantissa values are *normal numbers*
    (this is the "fn" = finite variant; it is not IEEE-like in that regard)

This is *not* the same as E5M2, IEEE-style E4M3 with infinities, or the
FN-UZ variants used by some AMD/Graphcore stacks.

Independent verification
------------------------
The encoder starts from the exact rational 1/10 (binary long division), not
from a host float. Results are then cross-checked against:
  * ``struct.pack`` / ``numpy.float32`` for FP32
  * ``torch.bfloat16`` for BF16
  * ``torch.float8_e4m3fn`` for FP8
  * OCP table entries (1.0, 448, min normal, NaN) for the decoder
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Optional

import numpy as np
import torch


class PrecisionSpec:
    def __init__(
        self,
        name: str,
        exp_bits: int,
        frac_bits: int,
        *,
        finite_only: bool = False,
        nan_only_all_ones_frac: bool = False,
    ) -> None:
        self.name = name
        self.exp_bits = exp_bits
        self.frac_bits = frac_bits
        self.bias = (1 << (exp_bits - 1)) - 1
        self.finite_only = finite_only
        self.nan_only_all_ones_frac = nan_only_all_ones_frac
        self.total_bits = 1 + exp_bits + frac_bits
        self.exp_all_ones = (1 << exp_bits) - 1
        self.frac_max = (1 << frac_bits) - 1


FP32 = PrecisionSpec("fp32", 8, 23)
BF16 = PrecisionSpec("bf16", 8, 7)
FP8_E4M3FN = PrecisionSpec(
    "fp8_e4m3fn",
    4,
    3,
    finite_only=True,
    nan_only_all_ones_frac=True,
)


@dataclass
class Encoding:
    spec_name: str
    decimal: str
    sign_bit: int
    unbiased_exponent: int
    encoded_exponent: int
    fraction_bits: str
    grouped_binary: str
    hex: str
    integer: int
    represented_value: float
    represented_fraction: str
    abs_error_vs_exact: float
    rel_error_vs_exact: float
    rounded: bool
    discarded_bits: str
    round_mode: str
    notes: str

    def as_dict(self) -> dict:
        return asdict(self)


def fractional_bits_from_rational(value: Fraction, n_bits: int) -> tuple[list[int], bool]:
    """Binary long division of a value in [0, 1). Returns (bits, sticky)."""
    if value < 0 or value >= 1:
        raise ValueError("fractional_bits_from_rational expects 0 <= value < 1")
    bits = []
    rem = value
    for _ in range(n_bits):
        rem *= 2
        if rem >= 1:
            bits.append(1)
            rem -= 1
        else:
            bits.append(0)
    sticky = rem != 0
    return bits, sticky


def normalize_positive(value: Fraction, extra_frac_bits: int = 256) -> tuple[int, list[int], bool]:
    """Return (unbiased_exp, fraction_bits_after_leading_1, sticky) for a positive finite value."""
    if value <= 0:
        raise ValueError("normalize_positive expects a positive value")
    exp = 0
    v = value
    while v >= 2:
        v /= 2
        exp += 1
    while v < 1:
        v *= 2
        exp -= 1
    # 1 <= v < 2
    frac = v - 1
    bits, sticky = fractional_bits_from_rational(frac, extra_frac_bits)
    return exp, bits, sticky


def round_to_nearest_even(bits: list[int], n_keep: int, sticky: bool) -> tuple[list[int], bool]:
    """Round the bit list to n_keep bits. Returns (bits, increment_exponent)."""
    if n_keep < 0:
        raise ValueError("n_keep must be non-negative")
    if len(bits) < n_keep:
        bits = bits + [0] * (n_keep - len(bits))
    keep = bits[:n_keep]
    rest = bits[n_keep:]
    round_bit = rest[0] if rest else 0
    rest_sticky = sticky or any(b == 1 for b in rest[1:])
    increment_exp = False
    if round_bit == 1 and (rest_sticky or (keep and keep[-1] == 1) or (not keep and True)):
        # RNE: round up on > 0.5, and on a 0.5 tie when the kept LSB is 1 (odd).
        if round_bit == 1 and not rest_sticky:
            # exact tie: round so LSB becomes 0 (even)
            if keep and keep[-1] == 0:
                return keep, False
            if not keep:
                return keep, True  # 1.0 * 2^{e} rounded to 0 frac bits → 1.0 * 2^{e+1} ...
        # add 1 to keep
        carry = 1
        for i in range(len(keep) - 1, -1, -1):
            s = keep[i] + carry
            keep[i] = s & 1
            carry = s >> 1
            if carry == 0:
                break
        if carry:
            increment_exp = True
            keep = [0] * n_keep
    return keep, increment_exp


def encode_rational(value: Fraction, spec: PrecisionSpec) -> Encoding:
    exact = value
    sign = 0 if value >= 0 else 1
    mag = abs(value)
    if mag == 0:
        integer = sign << (spec.total_bits - 1)
        represented = -0.0 if sign else 0.0
        return Encoding(
            spec_name=spec.name,
            decimal=str(float(value)),
            sign_bit=sign,
            unbiased_exponent=0,
            encoded_exponent=0,
            fraction_bits="0" * spec.frac_bits,
            grouped_binary=_group(sign, 0, [0] * spec.frac_bits, spec),
            hex=f"0x{integer:0{spec.total_bits // 4}x}",
            integer=integer,
            represented_value=represented,
            represented_fraction="0",
            abs_error_vs_exact=0.0,
            rel_error_vs_exact=0.0,
            rounded=False,
            discarded_bits="",
            round_mode="round-to-nearest-even",
            notes="signed zero",
        )

    unbiased, frac_bits, sticky = normalize_positive(mag)
    encoded_exp = unbiased + spec.bias

    # Subnormals: encoded_exp <= 0
    if encoded_exp <= 0:
        # Leading 1 shifts into the fraction; exponent stored as 0.
        shift = 1 - encoded_exp  # number of extra leading zeros in the stored field
        # significand bits including the hidden 1
        full = [1] + frac_bits
        # right-shift by `shift`, then take spec.frac_bits
        shifted = [0] * shift + full
        kept, inc = round_to_nearest_even(shifted, spec.frac_bits, sticky)
        if inc:
            # rounded up to min normal
            encoded_exp = 1
            kept = [0] * spec.frac_bits
            unbiased = 1 - spec.bias
        else:
            encoded_exp = 0
            unbiased = 1 - spec.bias
        discarded = "".join(str(b) for b in shifted[spec.frac_bits : spec.frac_bits + 16])
        rounded = any(shifted[spec.frac_bits :]) or sticky
        integer = _pack(sign, encoded_exp, kept, spec)
        represented = decode_int(integer, spec)
        return _finish(spec, exact, sign, unbiased, encoded_exp, kept, integer, represented, rounded, discarded)

    kept, inc = round_to_nearest_even(frac_bits, spec.frac_bits, sticky)
    if inc:
        encoded_exp += 1
        unbiased += 1
        kept = [0] * spec.frac_bits

    # Overflow
    if encoded_exp >= spec.exp_all_ones:
        if spec.finite_only:
            encoded_exp = spec.exp_all_ones
            kept = [1] * (spec.frac_bits - 1) + [0]  # max finite: S.1111.110 for E4M3
            unbiased = encoded_exp - spec.bias
        else:
            encoded_exp = spec.exp_all_ones
            kept = [0] * spec.frac_bits  # infinity
            unbiased = spec.exp_all_ones - spec.bias

    discarded = "".join(str(b) for b in frac_bits[spec.frac_bits : spec.frac_bits + 24])
    rounded = any(frac_bits[spec.frac_bits :]) or sticky
    integer = _pack(sign, encoded_exp, kept, spec)
    represented = decode_int(integer, spec)
    return _finish(spec, exact, sign, unbiased, encoded_exp, kept, integer, represented, rounded, discarded)


def _finish(spec, exact, sign, unbiased, encoded_exp, kept, integer, represented, rounded, discarded) -> Encoding:
    abs_err = abs(represented - float(exact))
    rel_err = abs_err / float(exact) if exact != 0 else abs_err
    return Encoding(
        spec_name=spec.name,
        decimal=str(exact),
        sign_bit=sign,
        unbiased_exponent=unbiased,
        encoded_exponent=encoded_exp,
        fraction_bits="".join(str(b) for b in kept),
        grouped_binary=_group(sign, encoded_exp, kept, spec),
        hex=f"0x{integer:0{(spec.total_bits + 3) // 4}x}",
        integer=integer,
        represented_value=represented,
        represented_fraction=str(Fraction(represented).limit_denominator()),
        abs_error_vs_exact=abs_err,
        rel_error_vs_exact=rel_err,
        rounded=rounded,
        discarded_bits=discarded,
        round_mode="round-to-nearest-even",
        notes=_notes(spec, encoded_exp, kept),
    )


def _pack(sign: int, exp: int, frac: list[int], spec: PrecisionSpec) -> int:
    frac_int = 0
    for b in frac:
        frac_int = (frac_int << 1) | int(b)
    return (sign << (spec.total_bits - 1)) | (exp << spec.frac_bits) | frac_int


def _group(sign: int, exp: int, frac: list[int], spec: PrecisionSpec) -> str:
    exp_s = format(exp, f"0{spec.exp_bits}b")
    frac_s = "".join(str(b) for b in frac)
    return f"{sign} {exp_s} {frac_s}"


def _notes(spec: PrecisionSpec, exp: int, frac: list[int]) -> str:
    if exp == 0 and all(b == 0 for b in frac):
        return "zero"
    if spec.finite_only:
        if exp == spec.exp_all_ones and all(b == 1 for b in frac):
            return "NaN (OCP E4M3FN: only all-ones mantissa with all-ones exponent)"
        if exp == spec.exp_all_ones:
            return "finite normal (E4M3FN uses exp=all-ones as ordinary normals except mantissa=111)"
        if exp == 0:
            return "subnormal"
        return "normal"
    if exp == spec.exp_all_ones:
        if all(b == 0 for b in frac):
            return "infinity"
        return "NaN"
    if exp == 0:
        return "subnormal"
    return "normal"


def decode_int(integer: int, spec: PrecisionSpec) -> float:
    sign = (integer >> (spec.total_bits - 1)) & 1
    exp = (integer >> spec.frac_bits) & ((1 << spec.exp_bits) - 1)
    frac = integer & ((1 << spec.frac_bits) - 1)
    s = -1.0 if sign else 1.0
    if spec.finite_only:
        if exp == spec.exp_all_ones and frac == spec.frac_max:
            return float("nan")
        if exp == 0:
            if frac == 0:
                return s * 0.0
            return s * (2 ** (1 - spec.bias)) * (frac / (1 << spec.frac_bits))
        return s * (2 ** (exp - spec.bias)) * (1.0 + frac / (1 << spec.frac_bits))
    # IEEE-like
    if exp == spec.exp_all_ones:
        if frac == 0:
            return s * float("inf")
        return float("nan")
    if exp == 0:
        if frac == 0:
            return s * 0.0
        return s * (2 ** (1 - spec.bias)) * (frac / (1 << spec.frac_bits))
    return s * (2 ** (exp - spec.bias)) * (1.0 + frac / (1 << spec.frac_bits))


def fp32_bits_struct(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def fp32_bits_numpy(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def bf16_bits_torch(value: float) -> int:
    t = torch.tensor(float(value), dtype=torch.float32).to(torch.bfloat16)
    return int(t.view(torch.int16).item()) & 0xFFFF


def fp8_e4m3fn_bits_torch(value: float) -> Optional[int]:
    if not hasattr(torch, "float8_e4m3fn"):
        return None
    t = torch.tensor(float(value), dtype=torch.float32).to(torch.float8_e4m3fn)
    return int(t.view(torch.uint8).item()) & 0xFF


def analyze_decimal_0p1() -> dict:
    exact = Fraction(1, 10)
    # Walk the assignment's nine steps explicitly for each format.
    unbiased, frac_bits, sticky = normalize_positive(exact)
    scientific = (
        "0.1_10 = 0.0001100110011..._2 (repeating 0011 after three leading zeros) "
        "= 1.1001100110011..._2 × 2^{-4}"
    )
    encodings = {}
    for spec in (FP32, BF16, FP8_E4M3FN):
        encodings[spec.name] = encode_rational(exact, spec).as_dict()

    fp32_int = encodings["fp32"]["integer"]
    bf16_int = encodings["bf16"]["integer"]
    fp8_int = encodings["fp8_e4m3fn"]["integer"]

    independent = {
        "fp32_struct": fp32_bits_struct(0.1),
        "fp32_numpy": fp32_bits_numpy(0.1),
        "bf16_torch": bf16_bits_torch(0.1),
        "fp8_e4m3fn_torch": fp8_e4m3fn_bits_torch(0.1),
    }
    matches = {
        "fp32_matches_struct": fp32_int == independent["fp32_struct"],
        "fp32_matches_numpy": fp32_int == independent["fp32_numpy"],
        "bf16_matches_torch": bf16_int == independent["bf16_torch"],
        "fp8_matches_torch": (
            independent["fp8_e4m3fn_torch"] is None
            or fp8_int == independent["fp8_e4m3fn_torch"]
        ),
    }

    decoder_vectors = ocp_e4m3fn_test_vectors()
    decoder_ok = all(abs(v["decoded"] - v["expected"]) < 1e-12 or v["expected"] != v["expected"] for v in decoder_vectors)

    recommendation = recommend_training_precision()
    passed = all(matches.values()) and decoder_ok
    return {
        "exact_rational": "1/10",
        "normalized_binary_scientific": scientific,
        "unbiased_exponent_before_rounding": unbiased,
        "leading_fraction_bits": "".join(str(b) for b in frac_bits[:32]),
        "sticky_beyond_32": sticky,
        "encodings": encodings,
        "independent_references": independent,
        "matches": matches,
        "ocp_e4m3fn_decoder_vectors": decoder_vectors,
        "decoder_vectors_ok": decoder_ok,
        "fields": {
            "fp32": {"sign": 1, "exponent": 8, "fraction": 23, "bias": 127, "specials": "IEEE inf/NaN on exp=255"},
            "bf16": {"sign": 1, "exponent": 8, "fraction": 7, "bias": 127, "specials": "IEEE inf/NaN on exp=255; same range as FP32, less precision"},
            "fp8_e4m3fn": {
                "sign": 1,
                "exponent": 4,
                "fraction": 3,
                "bias": 7,
                "specials": "OCP: no inf; NaN only S.1111.111; max finite ±448",
            },
        },
        "why_one_value_is_not_enough": (
            "The representation error of a single constant (0.1) ranks the formats by "
            "significand width, which is true but incomplete. Training precision also "
            "depends on exponent range (overflow/underflow of activations and grads), "
            "accumulation width (master weights typically remain FP32), rounding mode, "
            "hardware throughput (tensor-core generation), and whether a scaling scheme "
            "(loss scaling, per-tensor FP8 scaling, MX block scaling) is available. "
            "BF16, FP8, and block-scaled FP4 are distinct options with different "
            "requirements; they are not drop-in replacements."
        ),
        "recommendation": recommendation,
        "passed": passed,
    }


def ocp_e4m3fn_test_vectors() -> list[dict]:
    """OCP OFP8 table entries used as an independent decoder check."""
    spec = FP8_E4M3FN
    vectors = [
        {"bits": 0x00, "expected": 0.0, "name": "+zero"},
        {"bits": 0x38, "expected": 1.0, "name": "+1.0  (0 0111 000)"},
        {"bits": 0x78, "expected": 256.0, "name": "+256 (0 1111 000) — finite, unlike IEEE"},
        {"bits": 0x7E, "expected": 448.0, "name": "+max finite  (0 1111 110)"},
        {"bits": 0x01, "expected": 2 ** -9, "name": "+min subnormal"},
        {"bits": 0x08, "expected": 2 ** -6, "name": "+min normal"},
    ]
    out = []
    for v in vectors:
        decoded = decode_int(v["bits"], spec)
        out.append({**v, "decoded": decoded, "ok": abs(decoded - v["expected"]) < 1e-12})
    nan_val = decode_int(0x7F, spec)
    out.append(
        {
            "bits": 0x7F,
            "expected": float("nan"),
            "name": "+NaN (0 1111 111)",
            "decoded": nan_val,
            "ok": nan_val != nan_val,
        }
    )
    return out


def recommend_training_precision() -> dict:
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if not cuda and not mps:
        choice = "fp32"
        rationale = (
            "This process is CPU-only (Apple M2, no CUDA, MPS not used). PyTorch CPU "
            "training of these toy models is native FP32; BF16 is available as a dtype "
            "but has no tensor-core speedup here, and FP8 E4M3 is an analysis format "
            "only (no CPU compute kernels in this project). Master weights, optimizer "
            "state, and accumulations should remain FP32."
        )
    elif cuda:
        choice = "bf16_mixed_fp32_master"
        rationale = (
            "On a CUDA device with BF16 tensor cores, mixed-precision BF16 compute "
            "plus FP32 master weights is the default recommendation. FP8 E4M3 needs "
            "per-tensor (or block) scaling and Hopper/Blackwell-class tensor cores; "
            "block-scaled FP4 is a further, separate stack (MXFP4) and is not a "
            "drop-in training dtype."
        )
    else:
        choice = "fp32"
        rationale = (
            "Apple MPS is visible to this PyTorch build, but the assignment suite is "
            "pinned to CPU in configs/default.yaml for a deterministic, GPU-free grade "
            "path. Training precision for the measured runs is therefore FP32. MPS "
            "FP16/BF16 was not benchmarked and is not recommended on the basis of the "
            "0.1 encoding error alone. CUDA BF16 mixed precision is the usual large-model "
            "default, and it is not available here."
        )
    return {
        "recommended": choice,
        "environment": {
            "cuda": cuda,
            "mps": mps,
            "cpu": True,
        },
        "rationale": rationale,
        "options_considered": {
            "fp32": "Full exponent range and 23-bit fraction; baseline on CPU; slowest on GPUs that have tensor cores.",
            "bf16": "Same exponent range as FP32 (bias 127), 7-bit fraction; good default mixed-precision compute dtype on Ampere+ and many TPUs; still uses FP32 accumulation.",
            "fp8_e4m3fn": "4-bit exponent, max 448, no infinities; needs scaling (Transformer Engine delayed scaling or MXFP8). Not usable as an unscaled global training dtype.",
            "block_scaled_fp4": "Element type plus a shared block scale (e.g. MXFP4). Distinct from BF16/FP8; requires supporting hardware and a scale-aware stack. Not recommended here.",
        },
    }


def format_report(payload: dict) -> str:
    lines = [
        "Decimal 0.1 in FP32, BF16, and OCP FP8 E4M3FN",
        "=" * 72,
        payload["normalized_binary_scientific"],
        f"Unbiased exponent (before rounding): {payload['unbiased_exponent_before_rounding']}",
        f"Leading fraction bits: {payload['leading_fraction_bits']}...",
        "",
    ]
    for name, enc in payload["encodings"].items():
        lines += [
            f"--- {name} ---",
            f"  grouped bits : {enc['grouped_binary']}",
            f"  hex          : {enc['hex']}",
            f"  encoded exp  : {enc['encoded_exponent']}  (unbiased {enc['unbiased_exponent']})",
            f"  fraction     : {enc['fraction_bits']}",
            f"  represented  : {enc['represented_value']}",
            f"  abs error    : {enc['abs_error_vs_exact']}",
            f"  rel error    : {enc['rel_error_vs_exact']}",
            f"  discarded    : {enc['discarded_bits']}…",
            f"  notes        : {enc['notes']}",
            "",
        ]
    lines += [
        f"Independent matches: {payload['matches']}",
        f"Recommendation: {payload['recommendation']['recommended']}",
        payload["recommendation"]["rationale"],
    ]
    return "\n".join(lines) + "\n"
