# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete probe-matched fake-FP4 GEMM composition."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ue5m3_fp4.backends.triton.gemm import issue_rz_bf16_gemm
from ue5m3_fp4.backends.triton.quantization import (
    _encoded_global_alpha,
    _require_operand,
    _resolve_tensor_amax,
    fake_quantize_gemm_operands,
)
from ue5m3_fp4.formats import RoundingMode


def probe_matched_fp4_gemm(
    a: Tensor,
    b: Tensor,
    *,
    block_size: Literal[16, 32],
    scale_target_a: float = 448.0,
    scale_target_b: float = 448.0,
    tensor_amax_a: Tensor | float | None = None,
    tensor_amax_b: Tensor | float | None = None,
    rounding_a: str | RoundingMode = RoundingMode.TIES_TO_EVEN,
    rounding_b: str | RoundingMode = RoundingMode.TIES_TO_EVEN,
    random_bits_a: Tensor | None = None,
    random_bits_b: Tensor | None = None,
    generator: torch.Generator | None = None,
    two_dimensional_b: bool = False,
    snap_to_1_over_1024: bool = True,
) -> Tensor:
    """Run the fake-FP4 GEMM numerical model used for reported comparisons.

    Operands are quantized into the encoded domain, materialized as BF16,
    accumulated using the K=64 issue-RZ rule, optionally snapped to the final
    1/1024 grid, and finally decoded with the product of the tensor scales.
    """

    _require_operand(a, name="a")
    _require_operand(b, name="b")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a and b have incompatible K dimensions")
    resolved_amax_a = _resolve_tensor_amax(a, tensor_amax_a, name="tensor_amax_a")
    resolved_amax_b = _resolve_tensor_amax(b, tensor_amax_b, name="tensor_amax_b")
    encoded_a, encoded_b = fake_quantize_gemm_operands(
        a,
        b,
        block_size=block_size,
        scale_target_a=scale_target_a,
        scale_target_b=scale_target_b,
        tensor_amax_a=resolved_amax_a,
        tensor_amax_b=resolved_amax_b,
        rounding_a=rounding_a,
        rounding_b=rounding_b,
        random_bits_a=random_bits_a,
        random_bits_b=random_bits_b,
        generator=generator,
        two_dimensional_b=two_dimensional_b,
        output_domain="encoded",
        output_dtype=torch.bfloat16,
    )
    encoded_output = issue_rz_bf16_gemm(
        encoded_a,
        encoded_b,
        snap_to_1_over_1024=snap_to_1_over_1024,
    )
    alpha = _encoded_global_alpha(
        resolved_amax_a,
        resolved_amax_b,
        float(scale_target_a),
        float(scale_target_b),
    )
    return encoded_output * alpha


__all__ = ["probe_matched_fp4_gemm"]
