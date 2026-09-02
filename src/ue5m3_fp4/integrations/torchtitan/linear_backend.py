# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe-matched training-linear backend seam for the TorchTitan adapter."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from ue5m3_fp4.nn.convert import ConversionRecord, convert_linear_modules
from ue5m3_fp4.nn.linear import LinearBackend, normalize_linear_backend
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState

LINEAR_BACKEND_NAME = LinearBackend.PROBE_MATCHED_TRITON.value
LINEAR_BACKEND_PROBE_MATCHED = True
_REPORTED_LINEAR_MARKER = "_ue5m3_reported_internal_linear"


def convert_reported_linears(
    model: nn.Module,
    *,
    recipe: UE5M3Recipe,
    scale_state: TrainingScaleState,
    selector: Callable[[str, nn.Linear], bool],
    backend: str | LinearBackend = LinearBackend.PROBE_MATCHED_TRITON,
) -> tuple[ConversionRecord, ...]:
    """Convert selected linears through an explicit public backend seam."""

    resolved_backend = normalize_linear_backend(backend)
    records = convert_linear_modules(
        model,
        recipe=recipe,
        scale_state=scale_state,
        backend=resolved_backend,
        selector=selector,
    )
    for record in records:
        setattr(
            model.get_submodule(record.module_name),
            _REPORTED_LINEAR_MARKER,
            resolved_backend.value,
        )
    return records


def is_reported_quantized_linear(module: object) -> bool:
    """Return whether a module was installed through this backend seam."""

    return isinstance(module, nn.Module) and isinstance(
        getattr(module, _REPORTED_LINEAR_MARKER, None), str
    )


__all__ = [
    "LINEAR_BACKEND_NAME",
    "LINEAR_BACKEND_PROBE_MATCHED",
    "convert_reported_linears",
    "is_reported_quantized_linear",
]
