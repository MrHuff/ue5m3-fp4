# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, explicit CPU/GPU oracles for the reconstructed accumulator rule."""

from __future__ import annotations

import math
from typing import Any

import torch

VARIANTS = ("rn", "rn_grid", "rz", "rz_grid")


def add_rz_f32(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Add FP32 tensors with IEEE round-toward-zero semantics.

    PyTorch does not expose an RTZ mode for ordinary tensor addition. The exact
    sum is therefore formed in FP64, rounded to FP32 by PyTorch, and stepped one
    FP32 value toward zero only when round-to-nearest moved away from the exact
    value. Inputs must be finite FP32 tensors with equal shapes.
    """

    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise TypeError("add_rz_f32 expects float32 inputs")
    if left.shape != right.shape:
        raise ValueError("add_rz_f32 inputs must have equal shapes")
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("add_rz_f32 inputs must be finite")
    exact = left.double() + right.double()
    rounded = left + right
    inexact = exact != rounded.double()
    rounded_away = rounded.abs().double() > exact.abs()
    toward_zero = torch.nextafter(rounded, torch.zeros_like(rounded))
    return torch.where(inexact & rounded_away, toward_zero, rounded)


def snap_rne(value: torch.Tensor, denominator: int) -> torch.Tensor:
    """Round to the nearest-even multiple of ``1 / denominator``."""

    if type(denominator) is not int or denominator <= 0:
        raise ValueError("denominator must be a positive integer")
    return torch.round(value * denominator) / denominator


def accumulate_issue_terms(
    terms: torch.Tensor,
    *,
    issue_size: int = 64,
    grid_denominator: int = 1024,
) -> dict[str, torch.Tensor]:
    """Reduce ``[rows, K]`` products under RN/RTZ and optional final grid."""

    if terms.ndim != 2:
        raise ValueError("terms must have shape [rows, K]")
    if type(issue_size) is not int or issue_size <= 0:
        raise ValueError("issue_size must be a positive integer")
    if not terms.is_floating_point() or not bool(torch.isfinite(terms).all()):
        raise ValueError("terms must be finite floating-point values")
    remainder = terms.shape[1] % issue_size
    if remainder:
        terms = torch.nn.functional.pad(terms, (0, issue_size - remainder))
    reference = terms.double().sum(dim=1)
    partials = (
        terms.float()
        .reshape(terms.shape[0], -1, issue_size)
        .sum(
            dim=2,
            dtype=torch.float32,
        )
    )
    issue_reference = partials.double().sum(dim=1)
    rn = torch.zeros(terms.shape[0], device=terms.device, dtype=torch.float32)
    rz = torch.zeros_like(rn)
    for partial in partials.unbind(dim=1):
        rn = rn + partial
        rz = add_rz_f32(rz, partial)
    return {
        "reference": reference,
        "issue_reference": issue_reference,
        "rn": rn,
        "rn_grid": snap_rne(rn, grid_denominator),
        "rz": rz,
        "rz_grid": snap_rne(rz, grid_denominator),
    }


def accumulate_issue_partials(
    partials: torch.Tensor,
    *,
    grid_denominator: int = 1024,
) -> dict[str, torch.Tensor]:
    """Reduce already-formed FP32 issue partials."""

    if partials.ndim != 2:
        raise ValueError("partials must have shape [rows, issues]")
    if not partials.is_floating_point() or not bool(torch.isfinite(partials).all()):
        raise ValueError("partials must be finite floating-point values")
    values = partials.float()
    reference = values.double().sum(dim=1)
    rn = torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
    rz = torch.zeros_like(rn)
    for partial in values.unbind(dim=1):
        rn = rn + partial
        rz = add_rz_f32(rz, partial)
    return {
        "reference": reference,
        "rn": rn,
        "rn_grid": snap_rne(rn, grid_denominator),
        "rz": rz,
        "rz_grid": snap_rne(rz, grid_denominator),
    }


def comparison_metrics(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Return scalar error, gain, and directional metrics."""

    estimate64 = estimate.detach().double().cpu()
    reference64 = reference.detach().double().cpu()
    if estimate64.shape != reference64.shape:
        raise ValueError("estimate and reference must have equal shapes")
    if estimate64.numel() == 0:
        raise ValueError("comparison inputs must be non-empty")
    error = estimate64 - reference64
    reference_energy = torch.sum(reference64.square())
    estimate_energy = torch.sum(estimate64.square())
    dot = torch.sum(estimate64 * reference64)
    gain = dot / max(float(reference_energy), 1.0e-300)
    cosine = dot / max(
        math.sqrt(float(reference_energy) * float(estimate_energy)),
        1.0e-300,
    )
    return {
        "mean_error": float(error.mean()),
        "mean_absolute_error": float(error.abs().mean()),
        "root_mean_square_error": float(torch.sqrt(torch.mean(error.square()))),
        "maximum_absolute_error": float(error.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(error)
            / max(float(torch.linalg.vector_norm(reference64)), 1.0e-300)
        ),
        "gain": float(gain),
        "cosine_similarity": float(cosine),
        "magnitude_bias": float(
            (estimate64.abs().mean() - reference64.abs().mean())
            / max(float(reference64.abs().mean()), 1.0e-300)
        ),
        "exact_count": int(torch.sum(estimate64 == reference64)),
        "count": estimate64.numel(),
    }
