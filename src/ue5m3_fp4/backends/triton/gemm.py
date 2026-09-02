# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton GEMM matching the measured native FP4 accumulation probe."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _add_rz_f32(left, right):
    return tl.inline_asm_elementwise(
        asm="add.rz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _issue_rz_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    mask_m = offsets_m < M
    mask_n = offsets_n < N

    total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_indices = k_start + offsets_k
        if ALIGNED:
            a = tl.load(a_ptr + offsets_m[:, None] * stride_am + k_indices[None, :] * stride_ak)
            b = tl.load(b_ptr + k_indices[:, None] * stride_bk + offsets_n[None, :] * stride_bn)
        else:
            a = tl.load(
                a_ptr + offsets_m[:, None] * stride_am + k_indices[None, :] * stride_ak,
                mask=mask_m[:, None],
                other=0.0,
            )
            b = tl.load(
                b_ptr + k_indices[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
                mask=mask_n[None, :],
                other=0.0,
            )
        partial = tl.dot(a, b, out_dtype=tl.float32)
        total = _add_rz_f32(total, partial)

    output_pointers = (
        output_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on
    )
    if ALIGNED:
        tl.store(output_pointers, total)
    else:
        tl.store(output_pointers, total, mask=mask_m[:, None] & mask_n[None, :])


def _validate_operands(a: Tensor, b: Tensor) -> None:
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("a and b must be torch.Tensor instances")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("a and b must have dtype bfloat16")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a and b have incompatible K dimensions")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    if a.shape[1] % 64:
        raise ValueError("K must be divisible by 64")


def issue_rz_bf16_gemm(
    a: Tensor,
    b: Tensor,
    *,
    snap_to_1_over_1024: bool = False,
) -> Tensor:
    """Multiply BF16 operands with an FP32 round-toward-zero add every K=64.

    The optional final snap applies ``round(C * 1024) / 1024`` after all K=64
    partials have accumulated.  This is the probe-matched output model used in
    the paper; it is not a general replacement for ``torch.mm``.
    """

    _validate_operands(a, b)
    if not isinstance(snap_to_1_over_1024, bool):
        raise TypeError("snap_to_1_over_1024 must be bool")
    m, k = a.shape
    n = b.shape[1]
    if m == 0 or n == 0 or k == 0:
        output = torch.empty((m, n), dtype=torch.float32, device=a.device)
        return output

    block_m = 32 if min(m, n) <= 512 else 128
    block_n = 64
    group_m = 4
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _issue_rz_kernel[grid](
        a,
        b,
        output,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        GROUP_M=group_m,
        ALIGNED=m % block_m == 0 and n % block_n == 0,
        num_warps=4,
        num_stages=3,
    )
    if snap_to_1_over_1024:
        output = torch.round(output * 1024.0) / 1024.0
    return output


__all__ = ["issue_rz_bf16_gemm"]
