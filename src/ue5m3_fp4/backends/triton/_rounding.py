# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton rounding primitives used by the public UE5M3 backend.

The floating-point decomposition and rounding rules are adapted from NVIDIA
Transformer Engine's Apache-2.0 custom Triton quantization implementation.
Graphcore modifications specialize the implementation to E2M1 payloads,
unsigned E5M3 scales, and the two rounding modes used in the reported runs.
"""

from __future__ import annotations

import triton
import triton.language as tl

ROUND_TIES_TO_EVEN = 0
ROUND_STOCHASTIC_FAST = 7


@triton.jit
def _ldexp(value, exponent):
    offset = 24.0
    exponent_f32 = exponent.to(tl.float32)
    low = value * 16777216.0 * tl.exp2(exponent_f32 - offset)
    high = value * 5.9604645e-8 * tl.exp2(exponent_f32 + offset)
    return tl.where(tl.abs(value) < 1.0, low, high)


@triton.jit
def _round_finite_float(
    value,
    random_bits,
    precision: tl.constexpr,
    bias: tl.constexpr,
    signed: tl.constexpr,
    max_finite: tl.constexpr,
    rounding: tl.constexpr,
):
    """Round finite values using the exact Transformer Engine arithmetic."""

    is_zero = value == 0.0
    finite_nonzero = ~is_zero
    is_negative = value < 0.0 if signed else value < -float("inf")
    magnitude = tl.abs(value) if signed else value
    safe_magnitude = tl.where(finite_nonzero, magnitude, 1.0)

    exponent = tl.floor(tl.log2(safe_magnitude)).to(tl.int32)
    exponent = tl.maximum(exponent, 1 - bias)
    exponent = exponent - precision + 1

    significand = _ldexp(safe_magnitude, -exponent)
    lower = tl.floor(significand)
    lower_integer = lower.to(tl.int64)
    delta = significand - lower
    odd = (lower_integer & 1) != 0

    if rounding == 0:
        round_away = (delta > 0.5) | ((delta == 0.5) & odd)
    else:
        midpoint = (2 * random_bits + 1).to(tl.float32) * tl.exp2(-9.0)
        round_away = (delta + midpoint) >= 1.0

    rounded_integer = lower_integer + round_away.to(tl.int64)
    rounded = _ldexp(rounded_integer.to(tl.float32), exponent)
    rounded = tl.where(finite_nonzero, rounded, magnitude)
    rounded = tl.minimum(rounded, max_finite)
    return tl.where(is_negative, -rounded, rounded)


@triton.jit
def round_e2m1(value, random_bits, rounding: tl.constexpr):
    return _round_finite_float(
        value,
        random_bits,
        precision=2,
        bias=1,
        signed=True,
        max_finite=6.0,
        rounding=rounding,
    )


@triton.jit
def round_ue5m3_ties_to_even(value):
    zero_bits = tl.zeros(value.shape, dtype=tl.int32)
    return _round_finite_float(
        value,
        zero_bits,
        precision=4,
        bias=15,
        signed=False,
        max_finite=61440.0,
        rounding=0,
    )
