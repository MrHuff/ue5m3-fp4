# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from ue5m3_fp4.formats import (
    E2M1,
    UE5M3,
    RoundingMode,
    StochasticFast,
    quantize_dequantize_blocks_with_metadata,
    round_to_format,
    stochastic_8bit_midpoint,
)


def test_format_descriptors_match_the_value_sets() -> None:
    assert (E2M1.total_bits, E2M1.exponent_bits, E2M1.fraction_bits) == (4, 2, 1)
    assert E2M1.signed
    assert E2M1.precision == 2
    assert E2M1.smallest_subnormal == 0.5
    assert E2M1.max_finite == 6.0

    assert (UE5M3.total_bits, UE5M3.exponent_bits, UE5M3.fraction_bits) == (8, 5, 3)
    assert not UE5M3.signed
    assert UE5M3.precision == 4
    assert UE5M3.smallest_normal == 2.0**-14
    assert UE5M3.smallest_subnormal == 2.0**-17
    assert UE5M3.max_finite == 61_440.0


def test_e2m1_ties_to_even_golden_values_and_saturation() -> None:
    values = torch.tensor(
        [-100.0, -0.75, -0.25, 0.0, 0.25, 0.5, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 100.0]
    )
    expected = torch.tensor(
        [-6.0, -1.0, -0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    )
    torch.testing.assert_close(round_to_format(values, E2M1), expected)


def test_ue5m3_rounding_covers_subnormal_normal_and_maximum() -> None:
    values = torch.tensor(
        [0.0, 2.0**-18, 2.0**-17, 2.0**-14, 61_440.0, 100_000.0],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [0.0, 0.0, 2.0**-17, 2.0**-14, 61_440.0, 61_440.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(round_to_format(values, UE5M3), expected)


def test_stochastic_fast_uses_exact_8bit_midpoints_at_boundary() -> None:
    values = torch.tensor([0.75, 0.75])
    random_bits = torch.tensor([127, 128], dtype=torch.int32)
    result = stochastic_8bit_midpoint(values, random_bits=random_bits)
    torch.testing.assert_close(result, torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(StochasticFast(values, random_bits=random_bits), result)


def test_stochastic_fast_exhaustive_8bit_distribution_is_unbiased_on_grid() -> None:
    random_bits = torch.arange(256, dtype=torch.int32)
    halfway = stochastic_8bit_midpoint(
        torch.full((256,), 0.75), random_bits=random_bits
    )
    quarter = stochastic_8bit_midpoint(
        torch.full((256,), 0.625), random_bits=random_bits
    )
    assert int((halfway == 1.0).sum()) == 128
    assert int((quarter == 1.0).sum()) == 64
    assert halfway.mean().item() == pytest.approx(0.75)
    assert quarter.mean().item() == pytest.approx(0.625)


def test_stochastic_fast_generator_is_reproducible() -> None:
    values = torch.linspace(0.51, 0.99, 128)
    first = stochastic_8bit_midpoint(
        values, generator=torch.Generator().manual_seed(1729)
    )
    second = stochastic_8bit_midpoint(
        values, generator=torch.Generator().manual_seed(1729)
    )
    torch.testing.assert_close(first, second)


def test_one_dimensional_blocks_are_independent_along_last_axis() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 100.0]])
    result, metadata = quantize_dequantize_blocks_with_metadata(
        values,
        block_size=4,
        scale_format=UE5M3,
        scale_target=448.0,
        rounding=RoundingMode.TIES_TO_EVEN,
    )
    assert result.shape == values.shape
    assert metadata.block_amax.shape == (1, 2)
    torch.testing.assert_close(metadata.block_amax, torch.tensor([[4.0, 100.0]]))
    assert metadata.tensor_reference.item() == 100.0


def test_two_dimensional_weight_blocks_share_square_tile_scales() -> None:
    values = torch.arange(1.0, 21.0).reshape(4, 5)
    result, metadata = quantize_dequantize_blocks_with_metadata(
        values,
        block_size=2,
        scale_format=UE5M3,
        scale_target=448.0,
        rounding="TiesToEven",
        two_dimensional=True,
    )
    assert result.shape == values.shape
    assert metadata.block_amax.shape == (2, 3)
    torch.testing.assert_close(
        metadata.block_amax,
        torch.tensor([[7.0, 9.0, 10.0], [17.0, 19.0, 20.0]]),
    )


def test_explicit_stale_tensor_reference_changes_scaling_without_mutating_it() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 40.0]])
    reference = torch.tensor(4.0)
    result, metadata = quantize_dequantize_blocks_with_metadata(
        values,
        block_size=4,
        scale_format=UE5M3,
        scale_target=448.0,
        rounding="TTE",
        tensor_reference=reference,
    )
    assert torch.isfinite(result).all()
    assert reference.item() == 4.0
    # The 10x stale-growth case remains below UE5M3's approximately 137x
    # target-relative headroom and therefore does not saturate the block scale.
    assert metadata.block_scale_codes.item() < UE5M3.max_finite
    assert result[-1, -1].item() == pytest.approx(40.0, rel=0.04)


def test_zero_tensor_has_deterministic_zero_output() -> None:
    values = torch.zeros(2, 17)
    result, metadata = quantize_dequantize_blocks_with_metadata(
        values,
        block_size=16,
        scale_format=UE5M3,
        scale_target=448.0,
        rounding="ties_to_even",
    )
    torch.testing.assert_close(result, values)
    assert bool((metadata.block_scale_codes == 1).all())


def test_tiny_reference_and_extreme_stale_growth_remain_finite() -> None:
    values = torch.tensor([[1.0e-40, 1.0e-20]], dtype=torch.float32)
    result, metadata = quantize_dequantize_blocks_with_metadata(
        values,
        block_size=2,
        scale_format=UE5M3,
        scale_target=448.0,
        rounding="ties_to_even",
        tensor_reference=torch.tensor(1.0e-40),
    )

    assert bool(torch.isfinite(result).all())
    assert bool(torch.isfinite(metadata.global_encode_multiplier))
    assert metadata.block_scale_codes.item() == UE5M3.max_finite


def test_block_scale_uses_source_exact_fp32_operation_order() -> None:
    # In FP32, reassociating this scale computation gives 38,911.99609375 and
    # rounds down to 36,864. The kernel order lands on the exact 38,912
    # midpoint and ties-to-even rounds to 40,960.
    values = torch.tensor([[7.209771652307365e-15]], dtype=torch.float32)
    reference = torch.tensor(8.30072435974513e-17, dtype=torch.float32)
    _, metadata = quantize_dequantize_blocks_with_metadata(
        values,
        block_size=1,
        scale_format=UE5M3,
        scale_target=448.0,
        rounding="ties_to_even",
        tensor_reference=reference,
    )

    assert metadata.block_scale_codes.item() == 40_960.0


def test_payload_uses_source_exact_reciprocal_then_multiply_order() -> None:
    values = torch.tensor(
        [[2.7757528175464845e-18, 8.410989948970382e-08]],
        dtype=torch.float32,
    )
    result, metadata = quantize_dequantize_blocks_with_metadata(
        values,
        block_size=2,
        scale_format=UE5M3,
        scale_target=448.0,
        rounding="ties_to_even",
        tensor_reference=torch.tensor(4.857566965417253e-19),
    )

    assert metadata.block_scale_codes.item() == UE5M3.max_finite
    # Reciprocal-then-multiply lands exactly on the 0.25 E2M1 midpoint and
    # ties-to-even rounds to zero. Direct division lands just above it.
    assert result[0, 0].item() == 0.0


def test_invalid_rounding_and_unsigned_negative_input_fail_loudly() -> None:
    with pytest.raises(ValueError, match="unknown rounding mode"):
        round_to_format(torch.tensor([1.0]), E2M1, rounding="ideal_stochastic")
    with pytest.raises(ValueError, match="cannot represent negative"):
        round_to_format(torch.tensor([-1.0]), UE5M3)
    with pytest.raises(ValueError, match="same shape"):
        stochastic_8bit_midpoint(
            torch.tensor([0.75, 0.75]), random_bits=torch.tensor([1])
        )


@pytest.mark.parametrize("target", [True, float("inf"), float("nan")])
def test_block_quantizer_rejects_nonfinite_or_boolean_scale_targets(target: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        quantize_dequantize_blocks_with_metadata(
            torch.ones(1, 4),
            block_size=4,
            scale_format=UE5M3,
            scale_target=target,
            rounding="ties_to_even",
        )
