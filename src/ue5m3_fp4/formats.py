# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable Torch reference operations for FP4 with UE5M3 block scales.

This module deliberately models values, rather than packed storage.  It is a
small, device-independent numerical reference for tests, fake quantization,
and comparison with optimized kernels.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import torch
from torch import Tensor


class RoundingMode(StrEnum):
    """Rounding modes implemented by the public numerical reference."""

    TIES_TO_EVEN = "ties_to_even"
    STOCHASTIC_8BIT_MIDPOINT = "stochastic_8bit_midpoint"


_ROUNDING_ALIASES: Final[dict[str, RoundingMode]] = {
    "tiestoeven": RoundingMode.TIES_TO_EVEN,
    "ties_to_even": RoundingMode.TIES_TO_EVEN,
    "tte": RoundingMode.TIES_TO_EVEN,
    "stochasticfast": RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
    "stochastic_fast": RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
    "stochastic8bitmidpoint": RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
    "stochastic_8bit_midpoint": RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
}


def normalize_rounding(rounding: str | RoundingMode) -> RoundingMode:
    """Return the canonical rounding mode, accepting documented aliases."""

    if isinstance(rounding, RoundingMode):
        return rounding
    if not isinstance(rounding, str):
        raise TypeError("rounding must be a string or RoundingMode")
    key = rounding.strip().replace("-", "_").lower()
    key_without_underscores = key.replace("_", "")
    try:
        return _ROUNDING_ALIASES[key]
    except KeyError as error:
        try:
            return _ROUNDING_ALIASES[key_without_underscores]
        except KeyError:
            choices = ", ".join(mode.value for mode in RoundingMode)
            raise ValueError(
                f"unknown rounding mode {rounding!r}; expected {choices}"
            ) from error


@dataclass(frozen=True, slots=True)
class FloatFormat:
    """Description of a finite floating-point value set.

    ``max_finite`` is explicit because UE5M3 reserves the all-ones exponent
    field for special values.  Its largest finite value is therefore 61,440,
    despite using an unsigned 5-exponent-bit, 3-fraction-bit encoding.
    """

    name: str
    total_bits: int
    exponent_bits: int
    fraction_bits: int
    exponent_bias: int
    signed: bool
    has_subnormals: bool
    max_finite: float

    def __post_init__(self) -> None:
        sign_bits = int(self.signed)
        if self.total_bits != sign_bits + self.exponent_bits + self.fraction_bits:
            raise ValueError("total_bits does not match the format fields")
        if self.exponent_bits <= 0 or self.fraction_bits < 0:
            raise ValueError("format bit counts must be non-negative")
        if (
            self.exponent_bias <= 0
            or not math.isfinite(self.max_finite)
            or self.max_finite <= 0
        ):
            raise ValueError("format bias and maximum must be positive")

    @property
    def precision(self) -> int:
        """Number of significand bits, including the implicit leading bit."""

        return self.fraction_bits + 1

    @property
    def smallest_normal(self) -> float:
        """Smallest positive normal value."""

        return 2.0 ** (1 - self.exponent_bias)

    @property
    def smallest_subnormal(self) -> float:
        """Smallest positive value, including subnormals when present."""

        if not self.has_subnormals:
            return self.smallest_normal
        return 2.0 ** (1 - self.exponent_bias - self.fraction_bits)


E2M1: Final[FloatFormat] = FloatFormat(
    name="E2M1",
    total_bits=4,
    exponent_bits=2,
    fraction_bits=1,
    exponent_bias=1,
    signed=True,
    has_subnormals=True,
    max_finite=6.0,
)

UE5M3: Final[FloatFormat] = FloatFormat(
    name="UE5M3",
    total_bits=8,
    exponent_bits=5,
    fraction_bits=3,
    exponent_bias=15,
    signed=False,
    has_subnormals=True,
    max_finite=61_440.0,
)


def _working_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _validate_float_tensor(tensor: Tensor, *, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _round_magnitudes(
    magnitude: Tensor,
    *,
    format: FloatFormat,
    rounding: RoundingMode,
    generator: torch.Generator | None,
    random_bits: Tensor | None,
) -> Tensor:
    """Round finite non-negative values to ``format``."""

    nonzero = magnitude != 0
    safe_magnitude = torch.where(nonzero, magnitude, torch.ones_like(magnitude))
    _, binary_exponent = torch.frexp(safe_magnitude)
    value_exponent = binary_exponent - 1
    if format.has_subnormals:
        value_exponent = torch.clamp_min(value_exponent, 1 - format.exponent_bias)

    quantum_exponent = value_exponent - format.precision + 1
    significand = torch.ldexp(safe_magnitude, -quantum_exponent)
    lower_significand = torch.floor(significand)
    delta = significand - lower_significand

    if rounding is RoundingMode.TIES_TO_EVEN:
        lower_is_odd = lower_significand.to(torch.int64).bitwise_and(1).bool()
        round_away = (delta > 0.5) | ((delta == 0.5) & lower_is_odd)
    else:
        if random_bits is None:
            random_bits = torch.randint(
                0,
                256,
                delta.shape,
                dtype=torch.int32,
                device=delta.device,
                generator=generator,
            )
        else:
            if not isinstance(random_bits, Tensor):
                raise TypeError("random_bits must be a torch.Tensor")
            if tuple(random_bits.shape) != tuple(delta.shape):
                raise ValueError("random_bits must have the same shape as the input")
            if random_bits.is_floating_point() or random_bits.is_complex():
                raise TypeError("random_bits must have an integer dtype")
            if not bool(((random_bits >= 0) & (random_bits <= 255)).all()):
                raise ValueError("random_bits values must be in [0, 255]")
            random_bits = random_bits.to(device=delta.device, dtype=torch.int32)

        # Exact StochasticFast rule used by the training implementation.  The
        # integer r is uniform on [0, 255], and (2r+1)/512 is the midpoint of
        # one of 256 equal subintervals of [0, 1].
        midpoint = (2 * random_bits + 1).to(delta.dtype) * (2.0**-9)
        round_away = delta + midpoint >= 1.0

    rounded_significand = lower_significand + round_away.to(delta.dtype)
    rounded = torch.ldexp(rounded_significand, quantum_exponent)
    rounded = torch.where(nonzero, rounded, torch.zeros_like(rounded))
    return torch.clamp(rounded, max=format.max_finite)


def round_to_format(
    tensor: Tensor,
    format: FloatFormat,
    *,
    rounding: str | RoundingMode = RoundingMode.TIES_TO_EVEN,
    generator: torch.Generator | None = None,
    random_bits: Tensor | None = None,
) -> Tensor:
    """Round ``tensor`` to values representable by ``format``.

    ``random_bits`` is an explicit test/provenance hook.  Production callers
    normally pass a device-compatible ``generator`` and let this function draw
    one integer uniformly from ``[0, 255]`` for each value.
    """

    _validate_float_tensor(tensor, name="tensor")
    if not isinstance(format, FloatFormat):
        raise TypeError("format must be a FloatFormat")
    if not format.signed and bool((tensor < 0).any()):
        raise ValueError(f"{format.name} cannot represent negative values")

    mode = normalize_rounding(rounding)
    # The training kernel forms tensor and block scales in FP32 even when the
    # learned operand uses another floating dtype.
    work = tensor.to(torch.float32)
    negative = (
        torch.signbit(work) if format.signed else torch.zeros_like(work, dtype=torch.bool)
    )
    magnitude = work.abs() if format.signed else work
    rounded = _round_magnitudes(
        magnitude,
        format=format,
        rounding=mode,
        generator=generator,
        random_bits=random_bits,
    )
    result = torch.where(negative, -rounded, rounded)
    return result.to(tensor.dtype)


def stochastic_8bit_midpoint(
    tensor: Tensor,
    format: FloatFormat = E2M1,
    *,
    generator: torch.Generator | None = None,
    random_bits: Tensor | None = None,
) -> Tensor:
    """Apply the exact 8-bit-midpoint ``StochasticFast`` rounding rule.

    For fractional distance ``delta`` above the lower representable value and
    integer ``r ~ Uniform({0, ..., 255})``, round away from zero iff
    ``delta + (2*r + 1) * 2**-9 >= 1``.  This intentionally is not ideal or
    full-precision stochastic rounding.
    """

    return round_to_format(
        tensor,
        format,
        rounding=RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
        generator=generator,
        random_bits=random_bits,
    )


# Public compatibility name for the training implementation's mode.
StochasticFast = stochastic_8bit_midpoint


@dataclass(frozen=True, slots=True)
class BlockQuantizationMetadata:
    """Inspectable scale values from the portable block quantizer."""

    tensor_reference: Tensor
    global_encode_multiplier: Tensor
    block_amax: Tensor
    block_scale_codes: Tensor


def _block_amax_and_expand(
    magnitude: Tensor,
    *,
    block_size: int,
    two_dimensional: bool,
) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    """Return block maxima and a closure expanding block values to the input."""

    if two_dimensional:
        if magnitude.ndim < 2:
            raise ValueError(
                "two_dimensional=True requires a tensor with at least 2 dimensions"
            )
        rows, columns = magnitude.shape[-2:]
        padded_rows = ((rows + block_size - 1) // block_size) * block_size
        padded_columns = ((columns + block_size - 1) // block_size) * block_size
        padded = torch.nn.functional.pad(
            magnitude,
            (0, padded_columns - columns, 0, padded_rows - rows),
        )
        leading = padded.shape[:-2]
        row_blocks = padded_rows // block_size
        column_blocks = padded_columns // block_size
        tiled = padded.reshape(
            *leading,
            row_blocks,
            block_size,
            column_blocks,
            block_size,
        )
        block_amax = tiled.amax(dim=(-3, -1))

        def expand(values: Tensor) -> Tensor:
            expanded = values[..., :, None, :, None].expand(
                *leading,
                row_blocks,
                block_size,
                column_blocks,
                block_size,
            )
            return expanded.reshape(*leading, padded_rows, padded_columns)[..., :rows, :columns]

        return block_amax, expand

    if magnitude.ndim == 0:
        raise ValueError("block quantization requires a tensor with at least 1 dimension")
    width = magnitude.shape[-1]
    padded_width = ((width + block_size - 1) // block_size) * block_size
    padded = torch.nn.functional.pad(magnitude, (0, padded_width - width))
    leading = padded.shape[:-1]
    block_count = padded_width // block_size
    blocked = padded.reshape(*leading, block_count, block_size)
    block_amax = blocked.amax(dim=-1)

    def expand(values: Tensor) -> Tensor:
        return (
            values[..., :, None]
            .expand(*leading, block_count, block_size)
            .reshape(*leading, padded_width)[..., :width]
        )

    return block_amax, expand


def _quantize_dequantize_blocks_impl(
    tensor: Tensor,
    *,
    block_size: int,
    scale_format: FloatFormat,
    scale_target: float,
    rounding: str | RoundingMode,
    generator: torch.Generator | None,
    tensor_reference: Tensor | float | None,
    two_dimensional: bool,
) -> tuple[Tensor, BlockQuantizationMetadata]:
    _validate_float_tensor(tensor, name="tensor")
    if tensor.numel() == 0:
        raise ValueError("block quantization requires a non-empty tensor")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(scale_format, FloatFormat) or scale_format.signed:
        raise ValueError("scale_format must be an unsigned FloatFormat")
    if (
        type(scale_target) not in (int, float)
        or not math.isfinite(float(scale_target))
        or not float(scale_target) > 0
    ):
        raise ValueError("scale_target must be a finite positive number")

    work = tensor.to(_working_dtype(tensor))
    if tensor_reference is None:
        reference = work.detach().abs().amax()
    else:
        reference = torch.as_tensor(
            tensor_reference,
            dtype=work.dtype,
            device=work.device,
        )
        if reference.numel() != 1:
            raise ValueError("tensor_reference must be scalar")
        reference = reference.reshape(())
        if not bool(torch.isfinite(reference)) or bool(reference < 0):
            raise ValueError("tensor_reference must be finite and non-negative")

    payload_max = E2M1.max_finite
    target = float(scale_target)
    # The zero-reference fallback matches the implementation's deterministic
    # path and avoids division by zero if a previously sampled tensor was zero.
    encode_numerator = torch.as_tensor(
        target * payload_max,
        device=work.device,
        dtype=work.dtype,
    )
    raw_encode_multiplier = encode_numerator / reference
    global_encode_multiplier = torch.where(
        reference == 0,
        torch.ones_like(reference),
        torch.minimum(
            raw_encode_multiplier,
            torch.full_like(reference, torch.finfo(work.dtype).max),
        ),
    )
    global_decode_multiplier = global_encode_multiplier.reciprocal()

    block_amax, expand = _block_amax_and_expand(
        work.abs(),
        block_size=block_size,
        two_dimensional=two_dimensional,
    )
    # Preserve the kernel's FP32 operation order. Reassociating this as
    # ``(block_amax * global_encode_multiplier) / payload_max`` can move an
    # exact UE5M3 midpoint to the opposite side before ties-to-even rounding.
    inverse_payload_max = torch.as_tensor(
        1.0 / payload_max,
        device=work.device,
        dtype=work.dtype,
    )
    scale_multiplier = global_encode_multiplier * inverse_payload_max
    ideal_scale_codes = block_amax * scale_multiplier
    ideal_scale_codes = torch.minimum(
        ideal_scale_codes,
        torch.full_like(ideal_scale_codes, scale_format.max_finite),
    )
    scale_codes = round_to_format(
        ideal_scale_codes,
        scale_format,
        rounding=RoundingMode.TIES_TO_EVEN,
    )
    # A zero block, or a positive scale that underflows during UE5M3 rounding,
    # receives code one before reciprocal scaling.  Zero payloads still decode
    # exactly to zero.
    scale_codes = torch.where(scale_codes == 0, torch.ones_like(scale_codes), scale_codes)
    expanded_scales = expand(scale_codes)
    decoded_scale = expanded_scales * global_decode_multiplier
    # Match the kernel's FP32 reciprocal-then-multiply sequence. A direct
    # division can cross an E2M1 midpoint for extreme stale references.
    payload_encode_multiplier = torch.reciprocal(decoded_scale)
    payload_input = work * payload_encode_multiplier
    payload = round_to_format(
        payload_input,
        E2M1,
        rounding=rounding,
        generator=generator,
    )
    dequantized = payload * decoded_scale
    metadata = BlockQuantizationMetadata(
        tensor_reference=reference.detach().clone(),
        global_encode_multiplier=global_encode_multiplier.detach().clone(),
        block_amax=block_amax.detach().clone(),
        block_scale_codes=scale_codes.detach().clone(),
    )
    return dequantized.to(tensor.dtype), metadata


def quantize_dequantize_blocks(
    tensor: Tensor,
    *,
    block_size: int,
    scale_format: FloatFormat,
    scale_target: float,
    rounding: str | RoundingMode,
    generator: torch.Generator | None = None,
    tensor_reference: Tensor | float | None = None,
    two_dimensional: bool = False,
) -> Tensor:
    """Fake-quantize E2M1 payloads with rounded unsigned block scales.

    Blocks are contiguous along the final dimension by default.  With
    ``two_dimensional=True``, each matrix in the tensor is tiled over its final
    two dimensions and shares one scale per ``block_size`` square tile.  The
    latter is the proposed B=16 weight path; activations and upstream gradients
    use one-dimensional blocks.

    ``tensor_reference`` is the tensor-scale amax supplied by a scaling
    lifecycle.  If omitted, the current tensor's global amax is used (D=1).
    """

    result, _ = _quantize_dequantize_blocks_impl(
        tensor,
        block_size=block_size,
        scale_format=scale_format,
        scale_target=scale_target,
        rounding=rounding,
        generator=generator,
        tensor_reference=tensor_reference,
        two_dimensional=two_dimensional,
    )
    return result


def quantize_dequantize_blocks_with_metadata(
    tensor: Tensor,
    *,
    block_size: int,
    scale_format: FloatFormat,
    scale_target: float,
    rounding: str | RoundingMode,
    generator: torch.Generator | None = None,
    tensor_reference: Tensor | float | None = None,
    two_dimensional: bool = False,
) -> tuple[Tensor, BlockQuantizationMetadata]:
    """As :func:`quantize_dequantize_blocks`, also returning scale metadata."""

    return _quantize_dequantize_blocks_impl(
        tensor,
        block_size=block_size,
        scale_format=scale_format,
        scale_target=scale_target,
        rounding=rounding,
        generator=generator,
        tensor_reference=tensor_reference,
        two_dimensional=two_dimensional,
    )


__all__ = [
    "E2M1",
    "UE5M3",
    "BlockQuantizationMetadata",
    "FloatFormat",
    "RoundingMode",
    "StochasticFast",
    "normalize_rounding",
    "quantize_dequantize_blocks",
    "quantize_dequantize_blocks_with_metadata",
    "round_to_format",
    "stochastic_8bit_midpoint",
]
