# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registration with the public extension APIs of pinned upstream TorchTitan."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from torch import nn

from ue5m3_fp4.integrations.torchtitan.comparators import (
    NativeNVFP4Variant,
    convert_native_nvfp4_nemotron_h,
    convert_ue5m3_te_settings_nemotron_h,
)
from ue5m3_fp4.integrations.torchtitan.data import build_public_two_stream_dataloader
from ue5m3_fp4.integrations.torchtitan.nemotron_h import (
    MODEL_FLAVOR,
    MODEL_NAME,
    NemotronH8BArgs,
    NemotronHForTorchTitan,
    parallelize_nemotron_h,
)
from ue5m3_fp4.integrations.torchtitan.selection import convert_reported_nemotron_h
from ue5m3_fp4.integrations.torchtitan.trainer import install_trainer_step_hook
from ue5m3_fp4.nn.linear import LinearBackend
from ue5m3_fp4.recipe import UE5M3Recipe

CONVERTER_B16_NAME = "ue5m3_fp4.reported_b16_d50"
CONVERTER_B32_NAME = "ue5m3_fp4.reported_b32_d50"
CONVERTER_TORCH_CONTROL_NAME = "ue5m3_fp4.torch_control_b16_d50"
CONVERTER_NATIVE_NVFP4_TE_NAME = "ue5m3_fp4.native_nvfp4_te"
CONVERTER_NATIVE_NVFP4_NO_RHT_ALL_NAME = "ue5m3_fp4.native_nvfp4_no_rht_all"
CONVERTER_UE5M3_TE_SETTINGS_NAME = "ue5m3_fp4.ue5m3_te_settings_d1"
# Backward-compatible singular spelling for the paper's primary B=16 recipe.
CONVERTER_NAME = CONVERTER_B16_NAME

_REGISTERED = False


class ReportedUE5M3Converter:
    """TorchTitan converter for the reported B=16, D=50 precision placement."""

    def __init__(self, _job_config: Any, _parallel_dims: Any) -> None:
        self.conversion = None

    def convert(self, model: nn.Module) -> None:
        self.conversion = convert_reported_nemotron_h(model)

    def post_optimizer_hook(self, _model: nn.Module | list[nn.Module]) -> None:
        return None


class ReportedUE5M3B32Converter(ReportedUE5M3Converter):
    """TorchTitan converter for the reported B=32, D=50 ablation."""

    def convert(self, model: nn.Module) -> None:
        recipe = replace(
            UE5M3Recipe.proposed(),
            name="proposed_ue5m3_b32_d50",
            block_size=32,
        )
        self.conversion = convert_reported_nemotron_h(model, recipe=recipe)


class UE5M3TorchControlConverter(ReportedUE5M3Converter):
    """B=16 control with identical quantization and generic FP32 Torch GEMMs."""

    def convert(self, model: nn.Module) -> None:
        self.conversion = convert_reported_nemotron_h(
            model,
            backend=LinearBackend.TRITON_QUANT_TORCH,
        )


class NativeNVFP4TEConverter(ReportedUE5M3Converter):
    """Pinned native NVFP4 baseline with the final-eight-block BF16 exemption."""

    def convert(self, model: nn.Module) -> None:
        self.conversion = convert_native_nvfp4_nemotron_h(
            model,
            variant=NativeNVFP4Variant.TRANSFORMER_ENGINE_RECIPE,
        )


class NativeNVFP4NoRHTAllConverter(ReportedUE5M3Converter):
    """Pinned native NVFP4 no-RHT ablation over all 112 eligible linears."""

    def convert(self, model: nn.Module) -> None:
        self.conversion = convert_native_nvfp4_nemotron_h(
            model,
            variant=NativeNVFP4Variant.NO_RHT_ALL_LINEARS,
        )


class UE5M3TESettingsConverter(ReportedUE5M3Converter):
    """Probe-matched UE5M3 comparator using TE-style recipe controls."""

    def convert(self, model: nn.Module) -> None:
        self.conversion = convert_ue5m3_te_settings_nemotron_h(model)


def register_torchtitan() -> None:
    """Register the model, converter, and logical-step hook exactly once."""

    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from torchtitan.components.loss import build_cross_entropy_loss
        from torchtitan.components.lr_scheduler import build_lr_schedulers
        from torchtitan.components.optimizer import build_optimizers
        from torchtitan.components.tokenizer import build_hf_tokenizer
        from torchtitan.components.validate import build_validator
        from torchtitan.protocols.model_converter import register_model_converter
        from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

        from ue5m3_fp4.integrations.torchtitan.state_dict import (
            NemotronHStateDictAdapter,
        )
    except ImportError as error:
        raise ImportError(
            "TorchTitan is required; install upstream revision "
            "e37f83f58b35fdbceed9a5916b3490c16247ac9c"
        ) from error

    register_model_converter(ReportedUE5M3Converter, CONVERTER_B16_NAME)
    register_model_converter(ReportedUE5M3B32Converter, CONVERTER_B32_NAME)
    register_model_converter(UE5M3TorchControlConverter, CONVERTER_TORCH_CONTROL_NAME)
    register_model_converter(NativeNVFP4TEConverter, CONVERTER_NATIVE_NVFP4_TE_NAME)
    register_model_converter(
        NativeNVFP4NoRHTAllConverter,
        CONVERTER_NATIVE_NVFP4_NO_RHT_ALL_NAME,
    )
    register_model_converter(UE5M3TESettingsConverter, CONVERTER_UE5M3_TE_SETTINGS_NAME)
    register_train_spec(
        MODEL_NAME,
        TrainSpec(
            model_cls=NemotronHForTorchTitan,
            model_args={MODEL_FLAVOR: NemotronH8BArgs()},
            parallelize_fn=parallelize_nemotron_h,
            pipelining_fn=None,
            build_optimizers_fn=build_optimizers,
            build_lr_schedulers_fn=build_lr_schedulers,
            build_dataloader_fn=build_public_two_stream_dataloader,
            build_tokenizer_fn=build_hf_tokenizer,
            build_loss_fn=build_cross_entropy_loss,
            build_validator_fn=build_validator,
            state_dict_adapter=NemotronHStateDictAdapter,
        ),
    )
    install_trainer_step_hook()
    _REGISTERED = True


__all__ = [
    "CONVERTER_B16_NAME",
    "CONVERTER_B32_NAME",
    "CONVERTER_NAME",
    "CONVERTER_NATIVE_NVFP4_NO_RHT_ALL_NAME",
    "CONVERTER_NATIVE_NVFP4_TE_NAME",
    "CONVERTER_TORCH_CONTROL_NAME",
    "CONVERTER_UE5M3_TE_SETTINGS_NAME",
    "NativeNVFP4NoRHTAllConverter",
    "NativeNVFP4TEConverter",
    "ReportedUE5M3B32Converter",
    "ReportedUE5M3Converter",
    "UE5M3TESettingsConverter",
    "UE5M3TorchControlConverter",
    "register_torchtitan",
]
