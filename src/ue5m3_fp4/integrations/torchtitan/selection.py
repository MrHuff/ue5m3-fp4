# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reported Nemotron-H 8B module selection for TorchTitan.

The selector is deliberately architecture-specific.  It will not reinterpret
an unknown ``Linear`` as part of the published experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from ue5m3_fp4.integrations.torchtitan.linear_backend import (
    convert_reported_linears,
    is_reported_quantized_linear,
)
from ue5m3_fp4.nn.convert import ConversionRecord
from ue5m3_fp4.nn.linear import LinearBackend, normalize_linear_backend
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState

REPORTED_NEMOTRON_H_8B_PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
REPORTED_NEMOTRON_H_LAYER_COUNT = 52
REPORTED_ELIGIBLE_LINEAR_COUNT = 112
REPORTED_OUTPUT_HEAD_COUNT = 1

_LINEAR_FQN = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.mixer\."
    r"(?P<projection>in_proj|out_proj|q_proj|k_proj|v_proj|o_proj|up_proj|down_proj)$"
)

_PROJECTIONS_BY_BLOCK = {
    "M": frozenset({"in_proj", "out_proj"}),
    "*": frozenset({"q_proj", "k_proj", "v_proj", "o_proj"}),
    "-": frozenset({"up_proj", "down_proj"}),
}


class NemotronHSelectionError(RuntimeError):
    """The loaded model does not match the reported 8B module layout."""


class FP32OutputLinear(nn.Linear):
    """Vocabulary head evaluated in FP32 while retaining master parameters.

    The reported setup keeps this head outside the set of 112 FP4-eligible
    internal projections.  Parameters are not copied during conversion, which
    preserves checkpoint and optimizer identity.
    """

    @classmethod
    def from_float(cls, module: nn.Linear) -> FP32OutputLinear:
        if type(module) is not nn.Linear:
            raise TypeError("the FP32 output-head adapter expects an exact nn.Linear")
        converted = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
            dtype=module.weight.dtype,
        )
        converted.weight = module.weight
        if module.bias is not None:
            converted.bias = module.bias
        converted.train(module.training)
        return converted

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(
                inputs.to(torch.float32),
                self.weight.to(torch.float32),
                self.bias.to(torch.float32) if self.bias is not None else None,
            )


def reported_projection(module_name: str) -> tuple[int, str] | None:
    """Return ``(layer, projection)`` for one reported eligible linear."""

    if not isinstance(module_name, str):
        raise TypeError("module_name must be a string")
    match = _LINEAR_FQN.search(module_name)
    if match is None:
        return None
    layer = int(match.group("layer"))
    if layer >= len(REPORTED_NEMOTRON_H_8B_PATTERN):
        return None
    projection = match.group("projection")
    block_type = REPORTED_NEMOTRON_H_8B_PATTERN[layer]
    if projection not in _PROJECTIONS_BY_BLOCK[block_type]:
        return None
    return layer, projection


def select_reported_nemotron_h_linear(
    module_name: str,
    _module: nn.Linear,
) -> bool:
    """Select exactly the 112 internal linears used by the proposed recipe."""

    return reported_projection(module_name) is not None


def _find_output_head(model: nn.Module) -> tuple[str, nn.Linear]:
    candidates = [
        (name, module)
        for name, module in model.named_modules(remove_duplicate=False)
        if name
        and name.rsplit(".", 1)[-1] in {"lm_head", "output"}
        and isinstance(module, nn.Linear)
        and ".mixer." not in name
    ]
    if len(candidates) != REPORTED_OUTPUT_HEAD_COUNT:
        names = [name for name, _ in candidates]
        raise NemotronHSelectionError(
            "expected exactly one Nemotron-H vocabulary output head named "
            f"'lm_head' or 'output', found {names}"
        )
    name, module = candidates[0]
    if type(module) is not nn.Linear:
        raise NemotronHSelectionError(
            f"output head {name!r} is {type(module).__qualname__}, expected nn.Linear"
        )
    return name, module


def _replace_output_head(model: nn.Module, name: str, module: nn.Linear) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, FP32OutputLinear.from_float(module))


def _validate_reported_architecture(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None:
        inner = getattr(model, "model", None)
        config = getattr(inner, "config", None)
    if config is None:
        raise NemotronHSelectionError("Nemotron-H model does not expose a config")

    layer_count = getattr(config, "num_hidden_layers", None)
    pattern = getattr(config, "hybrid_override_pattern", None)
    if layer_count != REPORTED_NEMOTRON_H_LAYER_COUNT:
        raise NemotronHSelectionError(
            f"reported recipe requires 52 layers, model reports {layer_count!r}"
        )
    if pattern != REPORTED_NEMOTRON_H_8B_PATTERN:
        raise NemotronHSelectionError(
            "model hybrid_override_pattern does not match the reported 8B architecture"
        )


def _validate_mamba_output_dispatch(model: nn.Module) -> None:
    """Reject a remote-code fast path that bypasses converted ``out_proj``.

    Compatible Nemotron-H implementations either always call ``out_proj`` or
    expose ``_can_fuse_dense_out_proj`` and use it to disable the fused output
    projection for non-dense linear modules.
    """

    import inspect
    import sys

    for name, module in model.named_modules():
        out_proj = getattr(module, "out_proj", None)
        if not is_reported_quantized_linear(out_proj):
            continue
        source_module = sys.modules.get(type(module).__module__)
        fusion_guard = (
            getattr(source_module, "_can_fuse_dense_out_proj", None)
            if source_module is not None
            else None
        )
        if callable(fusion_guard):
            if fusion_guard(out_proj):
                raise NemotronHSelectionError(
                    f"{name}.out_proj would be bypassed by the Mamba fused projection"
                )
            continue

        fast_forward = getattr(type(module), "cuda_kernels_forward", None)
        if fast_forward is None:
            continue
        try:
            source = inspect.getsource(fast_forward)
        except (OSError, TypeError) as error:
            raise NemotronHSelectionError(
                f"cannot verify quantized out_proj dispatch for {name}"
            ) from error
        # The pinned vanilla remote code uses direct-weight fusion only in its
        # training branch.  Evaluation dispatches through self.out_proj and is
        # therefore safe; public training must use the hash-locked patch in
        # remote_code.py.
        narrow_patch_present = (
            "if type(self.out_proj) is nn.Linear:" in source
            and "out = self.out_proj(scan_output)" in source
        )
        if module.training and "outproj_weight" in source and not narrow_patch_present:
            raise NemotronHSelectionError(
                "the loaded Nemotron-H remote code fuses out_proj by reading its "
                f"weight directly at {name}; use a public revision that dispatches "
                "non-dense output projections through module.forward"
            )


@dataclass(frozen=True, slots=True)
class ReportedConversion:
    """Auditable result of applying the reported 8B precision placement."""

    fp4_linears: tuple[ConversionRecord, ...]
    fp32_output_head: str
    scale_state: TrainingScaleState
    linear_backend: str
    probe_matched_backend: bool


def convert_reported_nemotron_h(
    model: nn.Module,
    *,
    recipe: UE5M3Recipe | None = None,
    backend: str | LinearBackend = LinearBackend.PROBE_MATCHED_TRITON,
) -> ReportedConversion:
    """Apply the reported B=16 or B=32 precision placement in place."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    resolved_recipe = recipe or UE5M3Recipe.proposed()
    if resolved_recipe.block_size not in {16, 32}:
        raise ValueError("the reported TorchTitan converter requires block size 16 or 32")
    expected_recipe = replace(
        UE5M3Recipe.proposed(),
        name=resolved_recipe.name,
        block_size=resolved_recipe.block_size,
    )
    if resolved_recipe != expected_recipe:
        raise ValueError(
            "the reported TorchTitan converter permits only the proposed recipe's "
            "block-16/block-32 variants"
        )
    resolved_backend = normalize_linear_backend(backend)

    _validate_reported_architecture(model)
    head_name, head = _find_output_head(model)
    state = TrainingScaleState(resolved_recipe)
    records = convert_reported_linears(
        model,
        recipe=resolved_recipe,
        scale_state=state,
        selector=select_reported_nemotron_h_linear,
        backend=resolved_backend,
    )
    if len(records) != REPORTED_ELIGIBLE_LINEAR_COUNT:
        names = [record.module_name for record in records]
        raise NemotronHSelectionError(
            "reported 8B conversion requires exactly 112 eligible internal linears; "
            f"converted {len(records)}: {names}"
        )
    _replace_output_head(model, head_name, head)
    _validate_mamba_output_dispatch(model)

    # Plain Python metadata: no Parameter or buffer registration and therefore
    # no effect on checkpoint state dictionaries.
    model._ue5m3_reported_conversion = {  # type: ignore[attr-defined]
        "schema": "ue5m3_fp4_reported_nemotron_h_conversion_v1",
        "fp4_linear_count": len(records),
        "fp32_output_head": head_name,
        "linear_backend": resolved_backend.value,
        "probe_matched_backend": (resolved_backend is LinearBackend.PROBE_MATCHED_TRITON),
        "recipe": resolved_recipe.to_dict(),
    }
    return ReportedConversion(
        fp4_linears=records,
        fp32_output_head=head_name,
        scale_state=state,
        linear_backend=resolved_backend.value,
        probe_matched_backend=(resolved_backend is LinearBackend.PROBE_MATCHED_TRITON),
    )


__all__ = [
    "REPORTED_ELIGIBLE_LINEAR_COUNT",
    "REPORTED_NEMOTRON_H_8B_PATTERN",
    "REPORTED_NEMOTRON_H_LAYER_COUNT",
    "FP32OutputLinear",
    "NemotronHSelectionError",
    "ReportedConversion",
    "convert_reported_nemotron_h",
    "reported_projection",
    "select_reported_nemotron_h_linear",
]
