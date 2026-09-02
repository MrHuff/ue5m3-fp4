# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from ue5m3_fp4.backends.triton import (
    issue_rz_bf16_gemm,
    probe_matched_fp4_gemm,
    triton_available,
)

pytestmark = pytest.mark.skipif(
    not triton_available(), reason="Triton CUDA backend unavailable"
)


def test_issue_rz_output_and_optional_snap() -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    a = torch.randn((33, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn((128, 67), device="cuda", dtype=torch.bfloat16, generator=generator)
    raw = issue_rz_bf16_gemm(a, b)
    snapped = issue_rz_bf16_gemm(a, b, snap_to_1_over_1024=True)
    assert raw.shape == (33, 67)
    assert raw.dtype is torch.float32
    assert torch.equal(snapped, torch.round(raw * 1024.0) / 1024.0)


def test_issue_rz_rejects_wrong_k_and_dtype() -> None:
    a = torch.ones((2, 65), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((65, 2), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 64"):
        issue_rz_bf16_gemm(a, b)
    with pytest.raises(TypeError, match="bfloat16"):
        issue_rz_bf16_gemm(a.float(), b.float())


def _historical_functions():
    root_text = os.environ.get("UE5M3_FP4_HISTORICAL_ROOT")
    if root_text is None:
        pytest.skip("set UE5M3_FP4_HISTORICAL_ROOT for recovered-kernel differential tests")
    root = Path(root_text).resolve()
    sys.path.insert(0, str(root))
    from low_bits_training.quantization.fused_quant_triton_v2 import (
        fake_quant_simultaneous,
    )
    from low_bits_training.quantization.triton_issue_rz_gemm import (
        issue_rz_bf16_gemm_triton,
    )

    return fake_quant_simultaneous, issue_rz_bf16_gemm_triton


@pytest.mark.parametrize("shape", [(32, 64, 64), (33, 128, 67), (600, 128, 700)])
def test_issue_rz_exact_differential_against_recovered_kernel(
    shape: tuple[int, int, int],
) -> None:
    _, historical_gemm = _historical_functions()
    m, k, n = shape
    generator = torch.Generator(device="cuda").manual_seed(m + k + n)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16, generator=generator)
    expected = historical_gemm(a, b)
    actual = issue_rz_bf16_gemm(a, b)
    assert torch.equal(actual, expected)


def test_probe_matched_gemm_exact_differential_against_recovered_composition() -> None:
    historical_quantize, historical_gemm = _historical_functions()
    generator = torch.Generator(device="cuda").manual_seed(882)
    a = torch.randn((35, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn((128, 69), device="cuda", dtype=torch.bfloat16, generator=generator)
    amax_a = a.float().abs().amax().reshape(1)
    amax_b = b.float().abs().amax().reshape(1)
    encoded_a, encoded_b = historical_quantize(
        a,
        b,
        scale_max_a=2048.0,
        scale_max_b=448.0,
        use_global_scale=True,
        ga_a=amax_a,
        ga_b=amax_b,
        scale_type="E5M3",
        data_dtype=torch.bfloat16,
        round_mode_a="TiesToEven",
        scale_round_mode_a="TiesToEven",
        round_mode_b="TiesToEven",
        scale_round_mode_b="TiesToEven",
        use_2d_b=True,
        encode_centric=False,
        block_size=16,
        return_encoded=True,
        nan_handling_mode="to_one",
    )
    expected = torch.round(historical_gemm(encoded_a, encoded_b) * 1024.0) / 1024.0
    alpha = (amax_a * amax_b).float() / ((2048.0 * 6.0) * (448.0 * 6.0))
    expected = expected * alpha
    actual = probe_matched_fp4_gemm(
        a,
        b,
        block_size=16,
        scale_target_a=2048.0,
        scale_target_b=448.0,
        tensor_amax_a=amax_a,
        tensor_amax_b=amax_b,
        two_dimensional_b=True,
        snap_to_1_over_1024=True,
    )
    assert torch.equal(actual, expected)
