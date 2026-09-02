# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from ue5m3_fp4.backends.triton import fake_quantize_gemm_operands, triton_available

pytestmark = pytest.mark.skipif(
    not triton_available(), reason="Triton CUDA backend unavailable"
)


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("two_dimensional_b", [False, True])
@pytest.mark.parametrize("output_domain", ["decoded", "encoded"])
def test_quantized_operands_are_reproducible_and_finite(
    block_size: int,
    two_dimensional_b: bool,
    output_domain: str,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234)
    a = torch.randn((37, 64), device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn((64, 71), device="cuda", dtype=torch.bfloat16, generator=generator)
    bits_a = torch.randint(0, 256, a.shape, device="cuda", dtype=torch.int32)
    bits_b = torch.randint(0, 256, b.shape, device="cuda", dtype=torch.int32)
    kwargs = {
        "block_size": block_size,
        "scale_target_a": 2048.0,
        "scale_target_b": 448.0,
        "rounding_a": "StochasticFast",
        "rounding_b": "StochasticFast",
        "random_bits_a": bits_a,
        "random_bits_b": bits_b,
        "two_dimensional_b": two_dimensional_b,
        "output_domain": output_domain,
        "output_dtype": torch.bfloat16,
    }
    first = fake_quantize_gemm_operands(a, b, **kwargs)
    second = fake_quantize_gemm_operands(a, b, **kwargs)
    for actual, repeated, source in zip(first, second, (a, b), strict=True):
        assert actual.shape == source.shape
        assert actual.dtype is torch.bfloat16
        assert torch.equal(actual, repeated)
        assert bool(torch.isfinite(actual).all())


def test_zero_blocks_use_unit_scale_and_remain_zero() -> None:
    a = torch.zeros((3, 64), device="cuda", dtype=torch.float32)
    b = torch.zeros((64, 5), device="cuda", dtype=torch.float32)
    quantized_a, quantized_b = fake_quantize_gemm_operands(
        a,
        b,
        block_size=16,
        output_domain="decoded",
        output_dtype=torch.float32,
    )
    assert torch.count_nonzero(quantized_a).item() == 0
    assert torch.count_nonzero(quantized_b).item() == 0


def test_quantizer_rejects_unreported_configuration() -> None:
    a = torch.ones((2, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((64, 2), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="block_size"):
        fake_quantize_gemm_operands(a, b, block_size=64)
    with pytest.raises(ValueError, match="output_domain"):
        fake_quantize_gemm_operands(a, b, block_size=16, output_domain="packed")
    with pytest.raises(ValueError, match="only valid with stochastic"):
        fake_quantize_gemm_operands(
            a,
            b,
            block_size=16,
            random_bits_a=torch.zeros_like(a, dtype=torch.int32),
        )


def _load_historical_quantizer():
    root_text = os.environ.get("UE5M3_FP4_HISTORICAL_ROOT")
    if root_text is None:
        pytest.skip("set UE5M3_FP4_HISTORICAL_ROOT for recovered-kernel differential tests")
    root = Path(root_text).resolve()
    if not (root / "low_bits_training/quantization/fused_quant_triton_v2.py").is_file():
        pytest.fail("UE5M3_FP4_HISTORICAL_ROOT does not contain the recovered quantizer")
    sys.path.insert(0, str(root))
    from low_bits_training.quantization.fused_quant_triton_v2 import (
        fake_quant_simultaneous,
    )

    return fake_quant_simultaneous


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("two_dimensional_b", [False, True])
@pytest.mark.parametrize("output_domain", ["decoded", "encoded"])
def test_exact_differential_against_recovered_deterministic_kernel(
    block_size: int,
    two_dimensional_b: bool,
    output_domain: str,
) -> None:
    historical = _load_historical_quantizer()
    generator = torch.Generator(device="cuda").manual_seed(
        9000 + block_size + int(two_dimensional_b)
    )
    a = torch.randn((77, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn((128, 91), device="cuda", dtype=torch.bfloat16, generator=generator)
    amax_a = a.float().abs().amax().reshape(1)
    amax_b = b.float().abs().amax().reshape(1)
    expected = historical(
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
        use_2d_b=two_dimensional_b,
        encode_centric=False,
        block_size=block_size,
        return_encoded=output_domain == "encoded",
        nan_handling_mode="to_one",
    )
    actual = fake_quantize_gemm_operands(
        a,
        b,
        block_size=block_size,
        scale_target_a=2048.0,
        scale_target_b=448.0,
        tensor_amax_a=amax_a,
        tensor_amax_b=amax_b,
        two_dimensional_b=two_dimensional_b,
        output_domain=output_domain,
        output_dtype=torch.bfloat16,
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("output_domain", ["decoded", "encoded"])
def test_exact_differential_with_explicit_stochastic_bits(output_domain: str) -> None:
    historical = _load_historical_quantizer()
    generator = torch.Generator(device="cuda").manual_seed(731)
    a = torch.randn((65, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn((128, 67), device="cuda", dtype=torch.bfloat16, generator=generator)
    bits_a = torch.randint(
        0, 256, a.shape, device="cuda", dtype=torch.int32, generator=generator
    )
    bits_b = torch.randint(
        0, 256, b.shape, device="cuda", dtype=torch.int32, generator=generator
    )
    amax_a = a.float().abs().amax().reshape(1)
    amax_b = b.float().abs().amax().reshape(1)
    expected = historical(
        a,
        b,
        scale_max_a=448.0,
        scale_max_b=448.0,
        use_global_scale=True,
        ga_a=amax_a,
        ga_b=amax_b,
        scale_type="E5M3",
        data_dtype=torch.bfloat16,
        round_mode_a="StochasticFast",
        scale_round_mode_a="TiesToEven",
        round_mode_b="StochasticFast",
        scale_round_mode_b="TiesToEven",
        srbits_a=bits_a,
        srbits_b=bits_b,
        use_2d_b=True,
        encode_centric=False,
        block_size=16,
        return_encoded=output_domain == "encoded",
        nan_handling_mode="to_one",
    )
    actual = fake_quantize_gemm_operands(
        a,
        b,
        block_size=16,
        tensor_amax_a=amax_a,
        tensor_amax_b=amax_b,
        rounding_a="StochasticFast",
        rounding_b="StochasticFast",
        random_bits_a=bits_a,
        random_bits_b=bits_b,
        two_dimensional_b=True,
        output_domain=output_domain,
        output_dtype=torch.bfloat16,
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
