# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public implementations of the reported FP4 comparator configurations.

Transformer Engine is imported only when a native NVFP4 converter is selected.
The native path requires the exact reported build and supported hardware; it
never substitutes the software UE5M3 path when either requirement is absent.
"""

from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import nn

from ue5m3_fp4.integrations.torchtitan.linear_backend import convert_reported_linears
from ue5m3_fp4.integrations.torchtitan.selection import (
    NemotronHSelectionError,
    _find_output_head,
    _replace_output_head,
    _validate_mamba_output_dispatch,
    _validate_reported_architecture,
    reported_projection,
)
from ue5m3_fp4.nn.convert import ConversionRecord
from ue5m3_fp4.nn.linear import LinearBackend
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState

TRANSFORMER_ENGINE_REVISION = "01aef4fc721bd12fd09cd56d53a314aee1b953d6"
TRANSFORMER_ENGINE_VERSION = "2.16.0.dev0+01aef4fc"

REPORTED_FINAL_BLOCK_EXEMPTION = 8
REPORTED_FINAL_BLOCK_START = 52 - REPORTED_FINAL_BLOCK_EXEMPTION
REPORTED_ALL_ELIGIBLE_LINEAR_COUNT = 112
REPORTED_FINAL_EXEMPT_LINEAR_COUNT = 16
REPORTED_TE_ELIGIBLE_LINEAR_COUNT = (
    REPORTED_ALL_ELIGIBLE_LINEAR_COUNT - REPORTED_FINAL_EXEMPT_LINEAR_COUNT
)

_REPORTED_LINEAR_MARKER = "_ue5m3_reported_internal_linear"


class NativeNVFP4Variant(StrEnum):
    """Native NVFP4 configurations reported by the paper."""

    TRANSFORMER_ENGINE_RECIPE = "transformer_engine_recipe"
    NO_RHT_ALL_LINEARS = "no_rht_all_linears"


@dataclass(frozen=True, slots=True)
class TransformerEngineRuntime:
    """Validated native modules needed by the public NVFP4 adapter."""

    version: str
    api: Any
    recipe_type: type[Any]
    base_module: Any
    linear_module: Any


@dataclass(frozen=True, slots=True)
class NativeNVFP4Conversion:
    """Auditable result of one native Transformer Engine conversion."""

    fp4_linears: tuple[ConversionRecord, ...]
    fp32_output_head: str
    variant: NativeNVFP4Variant
    transformer_engine_version: str
    native_backend: str
    current_tensor_interval: int
    final_bf16_linears: int


@dataclass(frozen=True, slots=True)
class UE5M3TESettingsConversion:
    """Auditable result of the UE5M3 probe-matched TE-settings comparator."""

    fp4_linears: tuple[ConversionRecord, ...]
    fp32_output_head: str
    scale_state: TrainingScaleState
    linear_backend: str
    current_tensor_interval: int
    final_bf16_linears: int


def _normalize_native_variant(value: str | NativeNVFP4Variant) -> NativeNVFP4Variant:
    if isinstance(value, NativeNVFP4Variant):
        return value
    if not isinstance(value, str):
        raise TypeError("variant must be a string or NativeNVFP4Variant")
    try:
        return NativeNVFP4Variant(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in NativeNVFP4Variant)
        raise ValueError(
            f"unknown native NVFP4 variant {value!r}; expected {choices}"
        ) from error


def require_pinned_transformer_engine() -> TransformerEngineRuntime:
    """Import and validate the exact native runtime used by the reported runs."""

    try:
        package = importlib.import_module("transformer_engine")
        api = importlib.import_module("transformer_engine.pytorch")
        recipe_module = importlib.import_module("transformer_engine.common.recipe")
        base_module = importlib.import_module("transformer_engine.pytorch.module.base")
        linear_module = importlib.import_module("transformer_engine.pytorch.module.linear")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "native NVFP4 requires a CUDA build of the pinned Transformer Engine "
            f"revision {TRANSFORMER_ENGINE_REVISION}"
        ) from error

    version = getattr(package, "__version__", None)
    if version != TRANSFORMER_ENGINE_VERSION:
        raise RuntimeError(
            "native NVFP4 reproduction requires Transformer Engine "
            f"{TRANSFORMER_ENGINE_VERSION}; found {version!r}"
        )
    linear_type = getattr(api, "Linear", None)
    recipe_type = getattr(recipe_module, "NVFP4BlockScaling", None)
    availability = getattr(api, "is_nvfp4_available", None)
    autocast = getattr(api, "fp8_autocast", None)
    if not isinstance(linear_type, type) or not issubclass(linear_type, nn.Module):
        raise RuntimeError(  # noqa: TRY004 - malformed external runtime, not caller input
            "pinned Transformer Engine does not expose pytorch.Linear"
        )
    if not isinstance(recipe_type, type):
        raise RuntimeError(  # noqa: TRY004 - malformed external runtime, not caller input
            "pinned Transformer Engine does not expose NVFP4BlockScaling"
        )
    if not callable(availability) or not callable(autocast):
        raise RuntimeError(  # noqa: TRY004 - malformed external runtime, not caller input
            "pinned Transformer Engine has an incomplete NVFP4 API"
        )

    available = availability(return_reason=True)
    if not (
        isinstance(available, tuple) and len(available) == 2 and isinstance(available[0], bool)
    ):
        raise RuntimeError("Transformer Engine returned an invalid NVFP4 capability result")
    if not available[0]:
        reason = available[1] if isinstance(available[1], str) else "unknown reason"
        raise RuntimeError(f"native NVFP4 is unavailable on this runtime: {reason}")

    parameters = inspect.signature(recipe_type).parameters
    required_controls = {
        "disable_rht",
        "disable_stochastic_rounding",
        "disable_2d_quantization",
    }
    missing = required_controls - set(parameters)
    if missing:
        raise RuntimeError(
            "pinned NVFP4BlockScaling lacks required controls: " + ", ".join(sorted(missing))
        )
    return TransformerEngineRuntime(
        version=version,
        api=api,
        recipe_type=recipe_type,
        base_module=base_module,
        linear_module=linear_module,
    )


def configure_reported_te_accumulation(
    runtime: TransformerEngineRuntime | None = None,
) -> dict[str, Any]:
    """Disable TE's two-times dgrad/wgrad accumulation as in the reported run."""

    resolved = runtime or require_pinned_transformer_engine()
    values: dict[str, bool] = {}
    for module_name, module in (
        ("module.base", resolved.base_module),
        ("module.linear", resolved.linear_module),
    ):
        for attribute in ("_2X_ACC_DGRAD", "_2X_ACC_WGRAD"):
            if not hasattr(module, attribute):
                raise RuntimeError(
                    f"pinned Transformer Engine is missing {module_name}.{attribute}"
                )
            setattr(module, attribute, False)
            value = getattr(module, attribute)
            if value is not False:
                raise RuntimeError(f"failed to disable {module_name}.{attribute}")
            values[f"{module_name}.{attribute}"] = value
    return {
        "schema": "ue5m3_reported_te_accumulation_v1",
        "transformer_engine_version": resolved.version,
        "controls": values,
    }


def _assert_qparams(
    qparams: Any,
    *,
    path: str,
    rht: bool,
    stochastic: bool,
    two_dimensional: bool,
) -> None:
    expected = {
        "random_hadamard_transform": rht,
        "stochastic_rounding": stochastic,
        "fp4_2d_quantization": two_dimensional,
    }
    for attribute, value in expected.items():
        actual = getattr(qparams, attribute, None)
        if actual is not value:
            raise RuntimeError(
                f"native NVFP4 {path}.{attribute} is {actual!r}, expected {value!r}"
            )


def build_reported_nvfp4_recipe(
    variant: str | NativeNVFP4Variant,
    *,
    runtime: TransformerEngineRuntime | None = None,
) -> Any:
    """Construct and verify one pinned native NVFP4 recipe.

    NVFP4 uses current-tensor global scaling. No delayed-amax fields are set;
    in particular, the no-RHT ablation is an effective D=1 path rather than an
    inert requested D=50 setting from a historical launcher.
    """

    resolved_variant = _normalize_native_variant(variant)
    resolved_runtime = runtime or require_pinned_transformer_engine()
    disable_rht = resolved_variant is NativeNVFP4Variant.NO_RHT_ALL_LINEARS
    recipe = resolved_runtime.recipe_type(
        disable_rht=disable_rht,
        disable_stochastic_rounding=False,
        disable_2d_quantization=False,
    )
    _assert_qparams(
        recipe.fp4_quant_fwd_inp,
        path="forward_input",
        rht=not disable_rht,
        stochastic=False,
        two_dimensional=False,
    )
    _assert_qparams(
        recipe.fp4_quant_fwd_weight,
        path="forward_weight",
        rht=False,
        stochastic=False,
        two_dimensional=True,
    )
    _assert_qparams(
        recipe.fp4_quant_bwd_grad,
        path="backward_gradient",
        rht=not disable_rht,
        stochastic=True,
        two_dimensional=False,
    )
    return recipe


def _linear_reset_parameters_weight_init(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


_BOUND_LINEAR_TYPES: dict[tuple[type[Any], type[Any]], type[nn.Module]] = {}


def _bound_recipe_linear_type(runtime: TransformerEngineRuntime) -> type[nn.Module]:
    key = (runtime.api.Linear, runtime.recipe_type)
    cached = _BOUND_LINEAR_TYPES.get(key)
    if cached is not None:
        return cached

    te_api = runtime.api
    recipe_type = runtime.recipe_type
    te_version = runtime.version

    class ReportedNativeNVFP4Linear(te_api.Linear):  # type: ignore[misc, valid-type]
        """Transformer Engine Linear carrying one verified native NVFP4 recipe."""

        _NVFP4_INFERENCE_ALIGNMENT_MULTIPLE = 16

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            bias: bool,
            params_dtype: torch.dtype,
            recipe: Any,
            module_name: str,
            device: torch.device | str,
        ) -> None:
            super().__init__(
                in_features,
                out_features,
                bias=bias,
                params_dtype=params_dtype,
                init_method=_linear_reset_parameters_weight_init,
                device=device,
            )
            if not isinstance(recipe, recipe_type):
                raise TypeError("bound recipe must be a pinned NVFP4BlockScaling")
            self.bound_recipe = recipe
            self.reported_module_name = module_name
            setattr(self, _REPORTED_LINEAR_MARKER, "native_transformer_engine_nvfp4")
            self._nvfp4_inference_alignment_enabled = False
            self._nvfp4_inference_alignment_forward_count = 0
            self._nvfp4_inference_alignment_padded_forward_count = 0
            self._nvfp4_inference_alignment_original_row_count = 0
            self._nvfp4_inference_alignment_padding_row_count = 0
            self._nvfp4_inference_alignment_maximum_padding_rows = 0

        @classmethod
        def from_float(
            cls,
            module: nn.Linear,
            *,
            recipe: Any,
            module_name: str,
        ) -> nn.Module:
            if type(module) is not nn.Linear:
                raise TypeError("native NVFP4 conversion expects an exact nn.Linear")
            converted = cls(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                params_dtype=module.weight.dtype,
                recipe=recipe,
                module_name=module_name,
                device=module.weight.device,
            )
            if module.weight.device.type != "meta":
                with torch.no_grad():
                    converted.weight.copy_(module.weight.to(converted.weight.dtype))
                    if module.bias is not None:
                        converted.bias.copy_(module.bias.to(converted.bias.dtype))
            converted.weight.requires_grad_(module.weight.requires_grad)
            if module.bias is not None:
                converted.bias.requires_grad_(module.bias.requires_grad)
            converted.train(module.training)
            return converted

        @staticmethod
        def nvfp4_inference_alignment_policy() -> dict[str, Any]:
            return {
                "schema": "ue5m3_native_nvfp4_internal_alignment_v1",
                "scope": "native Transformer Engine Linear input only",
                "flat_first_dim": "product(input.shape[:-1])",
                "required_multiple": 16,
                "padding": "append all-zero rows after flattening",
                "output": "slice appended rows and restore original leading dimensions",
                "top_level_model_inputs_modified": False,
            }

        def enable_nvfp4_inference_alignment(self) -> None:
            if self.training:
                raise RuntimeError("NVFP4 alignment may only be enabled after module.eval()")
            if self._nvfp4_inference_alignment_enabled:
                raise RuntimeError("NVFP4 alignment was enabled twice")
            if getattr(self, "fp8_meta_tensors_initialized", None) is not False:
                raise RuntimeError("enable alignment before native quantizers are initialized")
            quantizers = getattr(self, "quantizers", None)
            workspaces = getattr(self, "_fp8_workspaces", None)
            if (
                not isinstance(quantizers, Mapping)
                or any(list(values) for values in quantizers.values())
                or not isinstance(workspaces, Mapping)
                or workspaces
            ):
                raise RuntimeError("NVFP4 alignment requires fresh native quantizer state")
            self._nvfp4_inference_alignment_enabled = True

        def nvfp4_inference_alignment_report(self) -> dict[str, Any]:
            return {
                "enabled": self._nvfp4_inference_alignment_enabled,
                "policy": self.nvfp4_inference_alignment_policy(),
                "counters": {
                    "forward_count": self._nvfp4_inference_alignment_forward_count,
                    "padded_forward_count": self._nvfp4_inference_alignment_padded_forward_count,
                    "original_row_count": self._nvfp4_inference_alignment_original_row_count,
                    "padding_row_count": self._nvfp4_inference_alignment_padding_row_count,
                    "maximum_padding_rows": (
                        self._nvfp4_inference_alignment_maximum_padding_rows
                    ),
                },
            }

        def native_nvfp4_report(self) -> dict[str, Any]:
            return {
                "schema": "ue5m3_native_nvfp4_module_v1",
                "module_name": self.reported_module_name,
                "numeric_backend": "native_transformer_engine_nvfp4",
                "native_hardware": True,
                "transformer_engine_version": te_version,
                "tensor_scaling": "current_tensor",
                "effective_interval": 1,
                "delayed_amax_fields_applied": False,
                "alignment": self.nvfp4_inference_alignment_report(),
            }

        def _align_input(
            self,
            inputs: torch.Tensor,
        ) -> tuple[torch.Tensor, tuple[int, ...], int]:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("NVFP4 alignment is restricted to no-grad evaluation")
            if inputs.ndim < 2:
                raise ValueError("native NVFP4 Linear requires a rank-two-or-higher input")
            original_shape = tuple(inputs.shape)
            original_rows = math.prod(original_shape[:-1])
            if original_rows <= 0:
                raise ValueError("native NVFP4 Linear rejects empty inputs")
            padding_rows = (-original_rows) % self._NVFP4_INFERENCE_ALIGNMENT_MULTIPLE
            self._nvfp4_inference_alignment_forward_count += 1
            self._nvfp4_inference_alignment_original_row_count += original_rows
            if padding_rows:
                self._nvfp4_inference_alignment_padded_forward_count += 1
                self._nvfp4_inference_alignment_padding_row_count += padding_rows
                self._nvfp4_inference_alignment_maximum_padding_rows = max(
                    self._nvfp4_inference_alignment_maximum_padding_rows,
                    padding_rows,
                )
                flattened = inputs.reshape(original_rows, original_shape[-1])
                inputs = torch.cat(
                    (flattened, inputs.new_zeros((padding_rows, original_shape[-1]))),
                    dim=0,
                )
            return inputs, original_shape, padding_rows

        @staticmethod
        def _restore_output(
            output: torch.Tensor,
            original_shape: tuple[int, ...],
            padding_rows: int,
        ) -> torch.Tensor:
            if not padding_rows:
                return output
            original_rows = math.prod(original_shape[:-1])
            if output.ndim != 2 or output.shape[0] != original_rows + padding_rows:
                raise RuntimeError("native NVFP4 Linear returned an invalid aligned shape")
            return output[:original_rows].reshape(*original_shape[:-1], output.shape[-1])

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if not isinstance(inputs, torch.Tensor):
                raise TypeError("native NVFP4 Linear input must be a torch.Tensor")
            if inputs.dtype is not torch.bfloat16:
                inputs = inputs.to(torch.bfloat16)
            original_shape = tuple(inputs.shape)
            padding_rows = 0
            if self._nvfp4_inference_alignment_enabled:
                inputs, original_shape, padding_rows = self._align_input(inputs)
            with te_api.fp8_autocast(enabled=True, fp8_recipe=self.bound_recipe):
                output = super().forward(inputs)
            return self._restore_output(output, original_shape, padding_rows)

    ReportedNativeNVFP4Linear.__name__ = "ReportedNativeNVFP4Linear"
    ReportedNativeNVFP4Linear.__qualname__ = "ReportedNativeNVFP4Linear"
    ReportedNativeNVFP4Linear.__module__ = __name__
    _BOUND_LINEAR_TYPES[key] = ReportedNativeNVFP4Linear
    return ReportedNativeNVFP4Linear


def _eligible_inventory(model: nn.Module) -> tuple[tuple[str, nn.Linear, int], ...]:
    inventory: list[tuple[str, nn.Linear, int]] = []
    invalid: list[str] = []
    for name, module in model.named_modules(remove_duplicate=False):
        projection = reported_projection(name)
        if projection is None:
            continue
        if not isinstance(module, nn.Linear) or type(module) is not nn.Linear:
            invalid.append(f"{name} ({type(module).__qualname__})")
            continue
        inventory.append((name, module, projection[0]))
    if invalid:
        raise NemotronHSelectionError(
            "comparator conversion requires exact nn.Linear modules: " + ", ".join(invalid)
        )
    if len(inventory) != REPORTED_ALL_ELIGIBLE_LINEAR_COUNT:
        raise NemotronHSelectionError(
            "reported Nemotron-H 8B requires exactly 112 eligible linears; "
            f"found {len(inventory)}"
        )
    aliases: dict[int, list[str]] = {}
    for name, module, _ in inventory:
        aliases.setdefault(id(module), []).append(name)
    duplicate_aliases = [names for names in aliases.values() if len(names) > 1]
    if duplicate_aliases:
        raise NemotronHSelectionError(
            "aliased eligible linears are unsupported: "
            + "; ".join(", ".join(names) for names in duplicate_aliases)
        )
    return tuple(inventory)


def _replace_child(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)


def _final_exempt_selector(module_name: str, _module: nn.Linear) -> bool:
    projection = reported_projection(module_name)
    return projection is not None and projection[0] < REPORTED_FINAL_BLOCK_START


def convert_native_nvfp4_nemotron_h(
    model: nn.Module,
    *,
    variant: str | NativeNVFP4Variant,
) -> NativeNVFP4Conversion:
    """Apply a pinned native NVFP4 comparator to the reported 8B architecture."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    resolved_variant = _normalize_native_variant(variant)
    runtime = require_pinned_transformer_engine()
    configure_reported_te_accumulation(runtime)
    recipe = build_reported_nvfp4_recipe(resolved_variant, runtime=runtime)
    _validate_reported_architecture(model)
    inventory = _eligible_inventory(model)
    head_name, head = _find_output_head(model)
    selected = (
        inventory
        if resolved_variant is NativeNVFP4Variant.NO_RHT_ALL_LINEARS
        else tuple(item for item in inventory if item[2] < REPORTED_FINAL_BLOCK_START)
    )
    expected_count = (
        REPORTED_ALL_ELIGIBLE_LINEAR_COUNT
        if resolved_variant is NativeNVFP4Variant.NO_RHT_ALL_LINEARS
        else REPORTED_TE_ELIGIBLE_LINEAR_COUNT
    )
    if len(selected) != expected_count:
        raise NemotronHSelectionError(
            f"native NVFP4 {resolved_variant.value} requires {expected_count} linears; "
            f"selected {len(selected)}"
        )

    linear_type = _bound_recipe_linear_type(runtime)
    records: list[ConversionRecord] = []
    for module_name, module, _ in selected:
        replacement = linear_type.from_float(  # type: ignore[attr-defined]
            module,
            recipe=recipe,
            module_name=module_name,
        )
        _replace_child(model, module_name, replacement)
        records.append(
            ConversionRecord(
                module_name=module_name,
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        )
    _replace_output_head(model, head_name, head)
    _validate_mamba_output_dispatch(model)
    final_bf16 = REPORTED_ALL_ELIGIBLE_LINEAR_COUNT - len(records)
    model._ue5m3_reported_conversion = {  # type: ignore[attr-defined]
        "schema": "ue5m3_native_nvfp4_nemotron_h_conversion_v1",
        "variant": resolved_variant.value,
        "fp4_linear_count": len(records),
        "bf16_final_linear_count": final_bf16,
        "fp32_output_head": head_name,
        "numeric_backend": "native_transformer_engine_nvfp4",
        "native_hardware": True,
        "transformer_engine_version": runtime.version,
        "transformer_engine_revision": TRANSFORMER_ENGINE_REVISION,
        "tensor_scaling": "current_tensor",
        "effective_interval": 1,
        "delayed_amax_fields_applied": False,
        "rht_disabled": resolved_variant is NativeNVFP4Variant.NO_RHT_ALL_LINEARS,
        "stochastic_gradient_rounding": True,
        "two_dimensional_weight_scaling": True,
    }
    return NativeNVFP4Conversion(
        fp4_linears=tuple(records),
        fp32_output_head=head_name,
        variant=resolved_variant,
        transformer_engine_version=runtime.version,
        native_backend="native_transformer_engine_nvfp4",
        current_tensor_interval=1,
        final_bf16_linears=final_bf16,
    )


def convert_ue5m3_te_settings_nemotron_h(model: nn.Module) -> UE5M3TESettingsConversion:
    """Apply the reported UE5M3 probe-matched TE-settings comparator."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    _validate_reported_architecture(model)
    _eligible_inventory(model)
    head_name, head = _find_output_head(model)
    recipe = UE5M3Recipe.transformer_engine_settings()
    state = TrainingScaleState(recipe)
    records = convert_reported_linears(
        model,
        recipe=recipe,
        scale_state=state,
        selector=_final_exempt_selector,
        backend=LinearBackend.PROBE_MATCHED_TRITON,
    )
    if len(records) != REPORTED_TE_ELIGIBLE_LINEAR_COUNT:
        raise NemotronHSelectionError(
            "UE5M3 TE-settings conversion requires exactly 96 FP4 linears; "
            f"converted {len(records)}"
        )
    _replace_output_head(model, head_name, head)
    _validate_mamba_output_dispatch(model)
    model._ue5m3_reported_conversion = {  # type: ignore[attr-defined]
        "schema": "ue5m3_te_settings_nemotron_h_conversion_v1",
        "fp4_linear_count": len(records),
        "bf16_final_linear_count": REPORTED_FINAL_EXEMPT_LINEAR_COUNT,
        "fp32_output_head": head_name,
        "numeric_backend": LinearBackend.PROBE_MATCHED_TRITON.value,
        "native_hardware": False,
        "tensor_scaling": "current_tensor",
        "effective_interval": 1,
        "rht": {
            "forward_gemm": False,
            "data_gradient_gemm": False,
            "weight_gradient_upstream_gradient": True,
            "weight_gradient_activation": True,
            "weights": False,
            "block_size": 16,
        },
        "recipe": recipe.to_dict(),
    }
    return UE5M3TESettingsConversion(
        fp4_linears=records,
        fp32_output_head=head_name,
        scale_state=state,
        linear_backend=LinearBackend.PROBE_MATCHED_TRITON.value,
        current_tensor_interval=1,
        final_bf16_linears=REPORTED_FINAL_EXEMPT_LINEAR_COUNT,
    )


__all__ = [
    "REPORTED_FINAL_BLOCK_EXEMPTION",
    "REPORTED_FINAL_EXEMPT_LINEAR_COUNT",
    "REPORTED_TE_ELIGIBLE_LINEAR_COUNT",
    "TRANSFORMER_ENGINE_REVISION",
    "TRANSFORMER_ENGINE_VERSION",
    "NativeNVFP4Conversion",
    "NativeNVFP4Variant",
    "TransformerEngineRuntime",
    "UE5M3TESettingsConversion",
    "build_reported_nvfp4_recipe",
    "configure_reported_te_accumulation",
    "convert_native_nvfp4_nemotron_h",
    "convert_ue5m3_te_settings_nemotron_h",
    "require_pinned_transformer_engine",
]
