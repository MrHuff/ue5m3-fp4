# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration with pinned upstream TorchTitan.

Importing this package is side-effect free.  Call :func:`register_torchtitan`,
or use ``ue5m3_fp4.integrations.torchtitan.plugin`` as TorchTitan's explicit
custom-import module.
"""

from ue5m3_fp4.integrations.torchtitan.comparators import (
    TRANSFORMER_ENGINE_REVISION,
    TRANSFORMER_ENGINE_VERSION,
    NativeNVFP4Conversion,
    NativeNVFP4Variant,
    UE5M3TESettingsConversion,
    build_reported_nvfp4_recipe,
    configure_reported_te_accumulation,
    convert_native_nvfp4_nemotron_h,
    convert_ue5m3_te_settings_nemotron_h,
    require_pinned_transformer_engine,
)
from ue5m3_fp4.integrations.torchtitan.linear_backend import (
    LINEAR_BACKEND_NAME,
    LINEAR_BACKEND_PROBE_MATCHED,
)
from ue5m3_fp4.integrations.torchtitan.nemotron_h import (
    MODEL_FLAVOR,
    MODEL_NAME,
    TORCHTITAN_REVISION,
    NemotronH8BArgs,
    NemotronHForTorchTitan,
    parallelize_nemotron_h,
)
from ue5m3_fp4.integrations.torchtitan.registration import (
    CONVERTER_B16_NAME,
    CONVERTER_B32_NAME,
    CONVERTER_NAME,
    CONVERTER_NATIVE_NVFP4_NO_RHT_ALL_NAME,
    CONVERTER_NATIVE_NVFP4_TE_NAME,
    CONVERTER_TORCH_CONTROL_NAME,
    CONVERTER_UE5M3_TE_SETTINGS_NAME,
    NativeNVFP4NoRHTAllConverter,
    NativeNVFP4TEConverter,
    ReportedUE5M3B32Converter,
    ReportedUE5M3Converter,
    UE5M3TESettingsConverter,
    UE5M3TorchControlConverter,
    register_torchtitan,
)
from ue5m3_fp4.integrations.torchtitan.selection import (
    REPORTED_ELIGIBLE_LINEAR_COUNT,
    REPORTED_NEMOTRON_H_8B_PATTERN,
    REPORTED_NEMOTRON_H_LAYER_COUNT,
    FP32OutputLinear,
    NemotronHSelectionError,
    ReportedConversion,
    convert_reported_nemotron_h,
    reported_projection,
    select_reported_nemotron_h_linear,
)
from ue5m3_fp4.integrations.torchtitan.trainer import (
    begin_training_step,
    install_trainer_step_hook,
    training_scale_states,
)

__all__ = [
    "CONVERTER_B16_NAME",
    "CONVERTER_B32_NAME",
    "CONVERTER_NAME",
    "CONVERTER_NATIVE_NVFP4_NO_RHT_ALL_NAME",
    "CONVERTER_NATIVE_NVFP4_TE_NAME",
    "CONVERTER_TORCH_CONTROL_NAME",
    "CONVERTER_UE5M3_TE_SETTINGS_NAME",
    "LINEAR_BACKEND_NAME",
    "LINEAR_BACKEND_PROBE_MATCHED",
    "MODEL_FLAVOR",
    "MODEL_NAME",
    "REPORTED_ELIGIBLE_LINEAR_COUNT",
    "REPORTED_NEMOTRON_H_8B_PATTERN",
    "REPORTED_NEMOTRON_H_LAYER_COUNT",
    "TORCHTITAN_REVISION",
    "TRANSFORMER_ENGINE_REVISION",
    "TRANSFORMER_ENGINE_VERSION",
    "FP32OutputLinear",
    "NativeNVFP4Conversion",
    "NativeNVFP4NoRHTAllConverter",
    "NativeNVFP4TEConverter",
    "NativeNVFP4Variant",
    "NemotronH8BArgs",
    "NemotronHForTorchTitan",
    "NemotronHSelectionError",
    "ReportedConversion",
    "ReportedUE5M3B32Converter",
    "ReportedUE5M3Converter",
    "UE5M3TESettingsConversion",
    "UE5M3TESettingsConverter",
    "UE5M3TorchControlConverter",
    "begin_training_step",
    "build_reported_nvfp4_recipe",
    "configure_reported_te_accumulation",
    "convert_native_nvfp4_nemotron_h",
    "convert_reported_nemotron_h",
    "convert_ue5m3_te_settings_nemotron_h",
    "install_trainer_step_hook",
    "parallelize_nemotron_h",
    "register_torchtitan",
    "reported_projection",
    "require_pinned_transformer_engine",
    "select_reported_nemotron_h_linear",
    "training_scale_states",
]
