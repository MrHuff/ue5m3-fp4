# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe-matched E2M1 payload quantization with UE5M3 block scales.

The numerical decomposition and rounding behavior are adapted from NVIDIA
Transformer Engine's Apache-2.0 custom Triton quantization implementation.
Graphcore modifications specialize the public path to the configuration used
by the reported block-16 and block-32 experiments.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import triton
import triton.language as tl
from torch import Tensor

from ue5m3_fp4.backends.triton._rounding import (
    ROUND_STOCHASTIC_FAST,
    ROUND_TIES_TO_EVEN,
    round_e2m1,
    round_ue5m3_ties_to_even,
)
from ue5m3_fp4.formats import RoundingMode, normalize_rounding

OutputDomain = Literal["decoded", "encoded"]


@triton.jit
def _fake_quantize_operands_kernel(
    a_ptr,
    a_out_ptr,
    b_ptr,
    b_out_ptr,
    tensor_amax_a_ptr,
    tensor_amax_b_ptr,
    random_bits_a_ptr,
    random_bits_b_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_out_am: tl.constexpr,
    stride_out_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_out_bk: tl.constexpr,
    stride_out_bn: tl.constexpr,
    scale_target_a,
    scale_target_b,
    rounding_a: tl.constexpr,
    rounding_b: tl.constexpr,
    stochastic_a: tl.constexpr,
    stochastic_b: tl.constexpr,
    two_dimensional_b: tl.constexpr,
    decoded_output: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    max_f32 = 3.4028235e38
    one = 1.0
    payload_max = 6.0
    scale_max = 61440.0

    tensor_amax_a = tl.load(tensor_amax_a_ptr).to(tl.float32)
    one_a = tl.full(tensor_amax_a.shape, one, tl.float32)
    factor_a = tl.full(tensor_amax_a.shape, payload_max, tl.float32) * scale_target_a
    global_encode_a = tl.extra.cuda.libdevice.div_rn(factor_a, tensor_amax_a)
    global_encode_a = tl.minimum(
        global_encode_a, tl.full(global_encode_a.shape, max_f32, tl.float32)
    )
    global_encode_a = tl.where(tensor_amax_a == 0.0, one_a, global_encode_a)
    global_decode_a = tl.extra.cuda.libdevice.div_rn(one_a, global_encode_a)
    global_decode_a = tl.where(global_encode_a == 0.0, one_a, global_decode_a)

    tensor_amax_b = tl.load(tensor_amax_b_ptr).to(tl.float32)
    one_b = tl.full(tensor_amax_b.shape, one, tl.float32)
    factor_b = tl.full(tensor_amax_b.shape, payload_max, tl.float32) * scale_target_b
    global_encode_b = tl.extra.cuda.libdevice.div_rn(factor_b, tensor_amax_b)
    global_encode_b = tl.minimum(
        global_encode_b, tl.full(global_encode_b.shape, max_f32, tl.float32)
    )
    global_encode_b = tl.where(tensor_amax_b == 0.0, one_b, global_encode_b)
    global_decode_b = tl.extra.cuda.libdevice.div_rn(one_b, global_encode_b)
    global_decode_b = tl.where(global_encode_b == 0.0, one_b, global_decode_b)

    if pid_row < num_pid_m:
        offsets_m = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_a = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
        pointers_a = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        values_a = tl.load(pointers_a, mask=mask_a, other=0.0).to(tl.float32)

        bits_a = tl.zeros(values_a.shape, dtype=tl.int32)
        if stochastic_a:
            bit_pointers_a = (
                random_bits_a_ptr
                + offsets_m[:, None] * stride_am
                + offsets_k[None, :] * stride_ak
            )
            bits_a = tl.load(bit_pointers_a, mask=mask_a, other=0)

        block_amax_a = tl.max(tl.abs(values_a), axis=1)
        reciprocal_payload_max = tl.extra.cuda.libdevice.div_rn(
            tl.full(block_amax_a.shape, one, tl.float32),
            tl.full(block_amax_a.shape, payload_max, tl.float32),
        )
        global_multiplier_a = tl.extra.cuda.libdevice.mul_rn(
            global_encode_a, reciprocal_payload_max
        )
        scale_a = tl.extra.cuda.libdevice.mul_rn(block_amax_a, global_multiplier_a)
        scale_a = tl.where(block_amax_a == 0.0, 0.0, scale_a)
        scale_a = tl.minimum(scale_a, scale_max)
        scale_a = round_ue5m3_ties_to_even(scale_a[:, None])
        scale_a = tl.where(scale_a == 0.0, 1.0, scale_a)

        denominator_a = tl.extra.cuda.libdevice.mul_rn(scale_a, global_decode_a)
        multiplier_a = tl.extra.cuda.libdevice.div_rn(
            tl.full(denominator_a.shape, one, tl.float32), denominator_a
        )
        scaled_a = tl.extra.cuda.libdevice.mul_rn(values_a, multiplier_a)
        payload_a = round_e2m1(scaled_a, bits_a, rounding=rounding_a)
        output_a = payload_a * scale_a
        if decoded_output:
            output_a = tl.extra.cuda.libdevice.mul_rn(output_a, global_decode_a)

        output_pointers_a = (
            a_out_ptr + offsets_m[:, None] * stride_out_am + offsets_k[None, :] * stride_out_ak
        )
        tl.store(output_pointers_a, output_a, mask=mask_a)

    if pid_row < num_pid_n:
        offsets_n = pid_row * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_b = (offsets_k[:, None] < K) & (offsets_n[None, :] < N)
        pointers_b = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        values_b = tl.load(pointers_b, mask=mask_b, other=0.0).to(tl.float32)

        bits_b = tl.zeros(values_b.shape, dtype=tl.int32)
        if stochastic_b:
            bit_pointers_b = (
                random_bits_b_ptr
                + offsets_k[:, None] * stride_bk
                + offsets_n[None, :] * stride_bn
            )
            bits_b = tl.load(bit_pointers_b, mask=mask_b, other=0)

        if two_dimensional_b:
            values_b_4d = tl.reshape(
                values_b,
                (1, BLOCK_K, BLOCK_N // BLOCK_K, BLOCK_K),
            )
            first_max = tl.max(tl.abs(values_b_4d), axis=1)
            tile_max = tl.max(first_max, axis=2)
            expanded_max = tile_max[:, None, :, None] * tl.full(
                (1, BLOCK_K, 1, BLOCK_K), 1.0, tl.float32
            )
            block_amax_b = tl.reshape(expanded_max, (BLOCK_K, BLOCK_N))
        else:
            block_amax_b = tl.max(tl.abs(values_b), axis=0)[None, :]

        reciprocal_payload_max = tl.extra.cuda.libdevice.div_rn(
            tl.full(block_amax_b.shape, one, tl.float32),
            tl.full(block_amax_b.shape, payload_max, tl.float32),
        )
        global_multiplier_b = tl.extra.cuda.libdevice.mul_rn(
            global_encode_b, reciprocal_payload_max
        )
        scale_b = tl.extra.cuda.libdevice.mul_rn(block_amax_b, global_multiplier_b)
        scale_b = tl.where(block_amax_b == 0.0, 0.0, scale_b)
        scale_b = tl.minimum(scale_b, scale_max)
        scale_b = round_ue5m3_ties_to_even(scale_b)
        scale_b = tl.where(scale_b == 0.0, 1.0, scale_b)

        denominator_b = tl.extra.cuda.libdevice.mul_rn(scale_b, global_decode_b)
        multiplier_b = tl.extra.cuda.libdevice.div_rn(
            tl.full(denominator_b.shape, one, tl.float32), denominator_b
        )
        scaled_b = tl.extra.cuda.libdevice.mul_rn(values_b, multiplier_b)
        payload_b = round_e2m1(scaled_b, bits_b, rounding=rounding_b)
        output_b = payload_b * scale_b
        if decoded_output:
            output_b = tl.extra.cuda.libdevice.mul_rn(output_b, global_decode_b)

        output_pointers_b = (
            b_out_ptr + offsets_k[:, None] * stride_out_bk + offsets_n[None, :] * stride_out_bn
        )
        tl.store(output_pointers_b, output_b, mask=mask_b)


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _require_operand(tensor: Tensor, *, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"{name} must have dtype bfloat16 or float32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _resolve_tensor_amax(tensor: Tensor, value: Tensor | float | None, *, name: str) -> Tensor:
    if value is None:
        return tensor.detach().float().abs().amax().reshape(1)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a scalar tensor or real number")
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain exactly one value")
        if value.device != tensor.device:
            raise ValueError(f"{name} must be on the same device as its operand")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        result = value.detach().to(dtype=torch.float32).reshape(1).contiguous()
    elif isinstance(value, (int, float)):
        result = torch.tensor([float(value)], device=tensor.device, dtype=torch.float32)
    else:
        raise TypeError(f"{name} must be a scalar tensor or real number")
    scalar = float(result.item())
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _resolve_random_bits(
    operand: Tensor,
    bits: Tensor | None,
    *,
    rounding: RoundingMode,
    generator: torch.Generator | None,
    name: str,
) -> tuple[Tensor, bool]:
    stochastic = rounding is RoundingMode.STOCHASTIC_8BIT_MIDPOINT
    if not stochastic:
        if bits is not None:
            raise ValueError(f"{name} is only valid with stochastic rounding")
        return torch.zeros(1, dtype=torch.int32, device=operand.device), False

    if bits is None:
        return torch.randint(
            0,
            256,
            operand.shape,
            dtype=torch.int32,
            device=operand.device,
            generator=generator,
        ), True
    if not isinstance(bits, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if bits.shape != operand.shape:
        raise ValueError(f"{name} must have the same shape as its operand")
    if bits.device != operand.device:
        raise ValueError(f"{name} must be on the same device as its operand")
    if bits.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype int32")
    if not bits.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not bool(((bits >= 0) & (bits <= 255)).all()):
        raise ValueError(f"{name} values must be in [0, 255]")
    return bits, True


def _rounding_constant(rounding: RoundingMode) -> int:
    if rounding is RoundingMode.TIES_TO_EVEN:
        return ROUND_TIES_TO_EVEN
    if rounding is RoundingMode.STOCHASTIC_8BIT_MIDPOINT:
        return ROUND_STOCHASTIC_FAST
    raise AssertionError(f"unhandled rounding mode: {rounding}")


def fake_quantize_gemm_operands(
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
    output_domain: OutputDomain = "decoded",
    output_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Fake-quantize logical GEMM operands ``A[M,K]`` and ``B[K,N]``.

    The payload is E2M1 and each block scale is unsigned E5M3 (UE5M3).  This
    exposes only the decode-centric, tensor-scaled, zero-scale-to-one path used
    for the paper's reported block-16 and block-32 experiments.  ``encoded``
    output contains ``payload * block_scale``; ``decoded`` output additionally
    applies the tensor-scale decoder.
    """

    _require_operand(a, name="a")
    _require_operand(b, name="b")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a and b have incompatible K dimensions")
    if block_size not in {16, 32}:
        raise ValueError("block_size must be 16 or 32")
    if a.shape[1] == 0 or a.shape[0] == 0 or b.shape[1] == 0:
        raise ValueError("a and b must have non-empty dimensions")
    if a.shape[1] % block_size:
        raise ValueError("K must be divisible by block_size")
    if not isinstance(scale_target_a, (int, float)) or isinstance(scale_target_a, bool):
        raise TypeError("scale_target_a must be a real number")
    if not isinstance(scale_target_b, (int, float)) or isinstance(scale_target_b, bool):
        raise TypeError("scale_target_b must be a real number")
    if not math.isfinite(scale_target_a) or scale_target_a <= 0:
        raise ValueError("scale_target_a must be finite and positive")
    if not math.isfinite(scale_target_b) or scale_target_b <= 0:
        raise ValueError("scale_target_b must be finite and positive")
    _require_bool(two_dimensional_b, name="two_dimensional_b")
    if output_domain not in {"decoded", "encoded"}:
        raise ValueError("output_domain must be 'decoded' or 'encoded'")
    if output_dtype is None:
        output_dtype = a.dtype
    if output_dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError("output_dtype must be torch.bfloat16 or torch.float32")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    normalized_a = normalize_rounding(rounding_a)
    normalized_b = normalize_rounding(rounding_b)
    bits_a, stochastic_a = _resolve_random_bits(
        a,
        random_bits_a,
        rounding=normalized_a,
        generator=generator,
        name="random_bits_a",
    )
    bits_b, stochastic_b = _resolve_random_bits(
        b,
        random_bits_b,
        rounding=normalized_b,
        generator=generator,
        name="random_bits_b",
    )
    resolved_amax_a = _resolve_tensor_amax(a, tensor_amax_a, name="tensor_amax_a")
    resolved_amax_b = _resolve_tensor_amax(b, tensor_amax_b, name="tensor_amax_b")

    output_a = torch.empty_like(a, dtype=output_dtype)
    output_b = torch.empty_like(b, dtype=output_dtype)
    block_m = 64
    block_n = 64
    grid = (
        max(triton.cdiv(a.shape[0], block_m), triton.cdiv(b.shape[1], block_n)),
        triton.cdiv(a.shape[1], block_size),
    )
    _fake_quantize_operands_kernel[grid](
        a,
        output_a,
        b,
        output_b,
        resolved_amax_a,
        resolved_amax_b,
        bits_a,
        bits_b,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        a.stride(0),
        a.stride(1),
        output_a.stride(0),
        output_a.stride(1),
        b.stride(0),
        b.stride(1),
        output_b.stride(0),
        output_b.stride(1),
        float(scale_target_a),
        float(scale_target_b),
        rounding_a=_rounding_constant(normalized_a),
        rounding_b=_rounding_constant(normalized_b),
        stochastic_a=stochastic_a,
        stochastic_b=stochastic_b,
        two_dimensional_b=two_dimensional_b,
        decoded_output=output_domain == "decoded",
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_size,
    )
    return output_a, output_b


def _encoded_global_alpha(
    tensor_amax_a: Tensor,
    tensor_amax_b: Tensor,
    scale_target_a: float,
    scale_target_b: float,
) -> Tensor:
    return (tensor_amax_a * tensor_amax_b).to(torch.float32) / (
        (scale_target_a * 6.0) * (scale_target_b * 6.0)
    )


__all__ = ["OutputDomain", "fake_quantize_gemm_operands"]
