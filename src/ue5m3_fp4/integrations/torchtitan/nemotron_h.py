# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Hugging Face Nemotron-H adapter for upstream TorchTitan.

This module implements TorchTitan's public model protocol without importing
the private training repository.  Model implementation and tokenizer assets
come from a caller-supplied Hugging Face path or model identifier.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from ue5m3_fp4.integrations.torchtitan.remote_code import (
    require_patched_nemotron_h_assets,
)
from ue5m3_fp4.integrations.torchtitan.selection import (
    REPORTED_NEMOTRON_H_8B_PATTERN,
    REPORTED_NEMOTRON_H_LAYER_COUNT,
)

TORCHTITAN_REVISION = "e37f83f58b35fdbceed9a5916b3490c16247ac9c"
MODEL_NAME = "nemotron_h_ue5m3"
MODEL_FLAVOR = "8B_reported"

_SDPA_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
_SDPA_BACKENDS_WITHOUT_CUDNN = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _nemotron_layers_and_config(model: nn.Module) -> tuple[nn.ModuleList, Any]:
    layers = getattr(model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        backbone = getattr(model, "backbone", None)
        layers = getattr(backbone, "layers", None)
    config = getattr(model, "config", None)
    if not isinstance(layers, nn.ModuleList) or config is None:
        raise TypeError("model must expose Nemotron-H layers and config")
    return layers, config


def verify_nemotron_h_sdpa(model: nn.Module) -> tuple[str, ...]:
    """Verify that every reported attention block uses the registered SDPA class."""

    layers, config = _nemotron_layers_and_config(model)
    attention_layers = [
        (index, layer)
        for index, layer in enumerate(layers)
        if getattr(layer, "block_type", None) == "attention"
    ]
    if not attention_layers:
        raise RuntimeError("reported Nemotron-H architecture contains no attention blocks")
    source_module = importlib.import_module(type(attention_layers[0][1].mixer).__module__)
    attention_classes = getattr(source_module, "NEMOTRONH_ATTENTION_CLASSES", None)
    if not isinstance(attention_classes, Mapping) or "sdpa" not in attention_classes:
        raise RuntimeError("pinned Nemotron-H remote code has no registered SDPA mixer")
    sdpa_cls = attention_classes["sdpa"]
    incorrect = [
        f"layers.{index}.mixer"
        for index, layer in attention_layers
        if not isinstance(layer.mixer, sdpa_cls)
    ]
    if incorrect:
        raise RuntimeError(f"Nemotron-H attention mixers are not SDPA: {incorrect}")
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("Nemotron-H config does not record SDPA after mixer conversion")
    return tuple(f"layers.{index}.mixer" for index, _ in attention_layers)


def configure_nemotron_h_sdpa(
    model: nn.Module,
    *,
    cudnn_enabled: bool,
) -> dict[str, Any]:
    """Replace eager attention mixers and fail closed unless all are SDPA.

    This helper supports both the TorchTitan wrapper and a standard Hugging
    Face ``NemotronHForCausalLM`` loaded for held-out evaluation.
    """

    if type(cudnn_enabled) is not bool:
        raise TypeError("cudnn_enabled must be bool")
    layers, config = _nemotron_layers_and_config(model)
    if config.hidden_size != config.num_attention_heads * config.head_dim:
        raise RuntimeError(
            "Nemotron-H SDPA requires hidden_size == num_attention_heads * head_dim"
        )
    attention_layers = [
        (index, layer)
        for index, layer in enumerate(layers)
        if getattr(layer, "block_type", None) == "attention"
    ]
    if not attention_layers:
        raise RuntimeError("reported Nemotron-H architecture contains no attention blocks")
    source_module = importlib.import_module(type(attention_layers[0][1].mixer).__module__)
    attention_classes = getattr(source_module, "NEMOTRONH_ATTENTION_CLASSES", None)
    if not isinstance(attention_classes, Mapping) or "sdpa" not in attention_classes:
        raise RuntimeError("pinned Nemotron-H remote code has no registered SDPA mixer")
    sdpa_cls = attention_classes["sdpa"]

    converted = []
    for index, layer in attention_layers:
        old_mixer = layer.mixer
        if isinstance(old_mixer, sdpa_cls):
            continue
        new_mixer = sdpa_cls(config, layer_idx=old_mixer.layer_idx)
        target_device = old_mixer.q_proj.weight.device
        target_dtype = old_mixer.q_proj.weight.dtype
        if target_device.type == "meta":
            if not hasattr(new_mixer, "to_empty"):
                raise RuntimeError("SDPA mixer does not support meta-device construction")
            new_mixer = new_mixer.to_empty(device=target_device)
        else:
            new_mixer = new_mixer.to(device=target_device, dtype=target_dtype)
            new_mixer.load_state_dict(old_mixer.state_dict())
        new_mixer.train(old_mixer.training)
        layer.mixer = new_mixer
        converted.append(f"layers.{index}.mixer")

    config._attn_implementation = "sdpa"
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(cudnn_enabled)
    mixers = verify_nemotron_h_sdpa(model)
    return {
        "schema": "ue5m3_nemotron_h_sdpa_configuration_v1",
        "implementation": "sdpa",
        "cudnn_enabled": cudnn_enabled,
        "attention_mixers": list(mixers),
        "converted_mixers": converted,
    }


@dataclass
class NemotronH8BArgs:
    """The Nemotron-H 8B architecture used by the reported trajectories."""

    _enforced: str = "all fields have defaults for TorchTitan BaseModelArgs compatibility"
    hf_assets_path: str = ""
    vocab_size: int = 131_072
    hidden_size: int = 4_096
    intermediate_size: int = 21_504
    num_hidden_layers: int = REPORTED_NEMOTRON_H_LAYER_COUNT
    hybrid_override_pattern: str = REPORTED_NEMOTRON_H_8B_PATTERN
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 8_192
    ssm_state_size: int = 128
    mamba_num_heads: int = 128
    mamba_head_dim: int = 64
    n_groups: int = 8
    conv_kernel: int = 4
    chunk_size: int = 256
    mamba_expand: int = 2
    mamba_hidden_act: str = "silu"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: tuple[float, float] = (0.0, float("inf"))
    use_mamba_kernels: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_hidden_act: str = "relu2"
    use_bias: bool = False
    tie_word_embeddings: bool = False
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = False
    initializer_range: float = 0.02
    rescale_prenorm_residual: bool = True
    use_cache: bool = False
    use_conv_bias: bool = True
    mamba_proj_bias: bool = False

    def update_from_config(self, job_config: Any, **_kwargs: Any) -> None:
        assets = getattr(job_config.model, "hf_assets_path", "")
        if not isinstance(assets, str) or not assets.strip():
            raise ValueError(
                "model.hf_assets_path must name public/local Nemotron-H Hugging Face assets"
            )
        self.hf_assets_path = assets
        seq_len = getattr(job_config.training, "seq_len", None)
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("training.seq_len must be a positive integer")
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                "the reported Nemotron-H 8B adapter supports sequence lengths up to 8192"
            )

    def get_nparams_and_flops(self, model: nn.Module, _seq_len: int) -> tuple[int, float]:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        return parameter_count, 6.0 * parameter_count

    def apply_to_config(self, config: Any) -> None:
        """Set and validate every architecture-defining field."""

        fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "hybrid_override_pattern",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "ssm_state_size",
            "mamba_num_heads",
            "mamba_head_dim",
            "n_groups",
            "conv_kernel",
            "chunk_size",
            "mamba_expand",
            "mamba_hidden_act",
            "time_step_min",
            "time_step_max",
            "time_step_floor",
            "time_step_limit",
            "use_mamba_kernels",
            "attention_bias",
            "attention_dropout",
            "hidden_dropout",
            "mlp_bias",
            "mlp_hidden_act",
            "use_bias",
            "tie_word_embeddings",
            "layer_norm_epsilon",
            "residual_in_fp32",
            "initializer_range",
            "rescale_prenorm_residual",
            "use_cache",
            "use_conv_bias",
            "mamba_proj_bias",
        )
        for field in fields:
            setattr(config, field, getattr(self, field))
        if hasattr(config, "rms_norm_eps"):
            config.rms_norm_eps = self.layer_norm_epsilon
        # Public Nemotron-H configuration revisions use both serialized Mamba
        # names and normalized runtime aliases; set both explicitly.
        config.mamba_n_groups = self.n_groups
        config.mamba_d_conv = self.conv_kernel
        config.mamba_chunk_size = self.chunk_size
        config.mamba_dt_min = self.time_step_min
        config.mamba_dt_max = self.time_step_max
        config.mamba_dt_init_floor = self.time_step_floor
        config.mamba_dt_limit = self.time_step_limit
        config.mamba_conv_bias = self.use_conv_bias
        # The pinned remote config exposes ``layers_block_type`` as a derived,
        # read-only property of ``hybrid_override_pattern``.  Setting the
        # architecture-defining pattern above is therefore both necessary and
        # sufficient; attempting to assign the property prevents construction.


class NemotronHForTorchTitan(nn.Module):
    """Storage-neutral version of the Nemotron-H wrapper used for the runs."""

    def __init__(self, model_args: NemotronH8BArgs) -> None:
        super().__init__()
        if not isinstance(model_args, NemotronH8BArgs):
            raise TypeError("model_args must be NemotronH8BArgs")
        if not model_args.hf_assets_path:
            raise ValueError("Nemotron-H Hugging Face assets were not configured")
        assets_root = require_patched_nemotron_h_assets(model_args.hf_assets_path)
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except ImportError as error:
            raise ImportError(
                "the Nemotron-H TorchTitan adapter requires transformers"
            ) from error

        config = AutoConfig.from_pretrained(
            assets_root,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_args.apply_to_config(config)
        # The pinned remote code rejects automatic SDPA selection.  Construct
        # eager modules first, then replace only attention mixers with its
        # registered SDPA implementation, matching the historical wrapper.
        config._attn_implementation = "eager"
        hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        backbone = getattr(hf_model, "backbone", None)
        if not isinstance(backbone, nn.Module):
            raise TypeError("public Nemotron-H model does not expose .backbone")

        self.config = config
        self.model_args = model_args
        self.tok_embeddings = backbone.embeddings
        self.layers = backbone.layers
        self.norm = backbone.norm_f
        self.output = hf_model.lm_head
        self.gradient_checkpointing = False
        self._cudnn_sdpa_enabled = _env_flag("TORCH_CUDNN_SDPA_ENABLED", True)
        self._sdpa_backends = (
            _SDPA_BACKENDS if self._cudnn_sdpa_enabled else _SDPA_BACKENDS_WITHOUT_CUDNN
        )
        self.sdpa_configuration = configure_nemotron_h_sdpa(
            self,
            cudnn_enabled=self._cudnn_sdpa_enabled,
        )
        self.attention_impl = "sdpa"
        self.attention_backend = (
            "nemotron_sdpa_cudnn_first"
            if self._cudnn_sdpa_enabled
            else "nemotron_sdpa_no_cudnn"
        )

    def _attention_context(self, device: torch.device):
        if device.type == "cuda":
            return sdpa_kernel(self._sdpa_backends)
        return contextlib.nullcontext()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        if buffer_device is not None:
            for buffer_name, buffer in self.named_buffers(recurse=True):
                if buffer is None or buffer.device == buffer_device:
                    continue
                parent_name, _, local_name = buffer_name.rpartition(".")
                parent = self.get_submodule(parent_name) if parent_name else self
                setattr(parent, local_name, buffer.to(buffer_device))

        # TorchTitan materializes a meta model with to_empty before invoking
        # this hook. Hugging Face init_weights does not initialize every such
        # tensor, so reset each public module explicitly as in the training
        # implementation used for the paper.
        for module in self.modules():
            if module is self:
                continue
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
            elif hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
                nn.init.ones_(module.weight)

        for module in self.modules():
            if all(hasattr(module, name) for name in ("A_log", "D", "dt_bias")):
                module.A_log._no_weight_decay = True
                module.D._no_weight_decay = True
                dt = torch.exp(
                    torch.rand(
                        self.config.mamba_num_heads,
                        device=module.dt_bias.device,
                        dtype=torch.float32,
                    )
                    * (
                        math.log(self.config.time_step_max)
                        - math.log(self.config.time_step_min)
                    )
                    + math.log(self.config.time_step_min)
                ).clamp(min=self.config.time_step_floor)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                with torch.no_grad():
                    inv_dt = inv_dt.to(module.dt_bias.dtype)
                    if isinstance(module.dt_bias, DTensor):
                        inv_dt = distribute_tensor(
                            inv_dt,
                            module.dt_bias.device_mesh,
                            module.dt_bias.placements,
                        )
                    module.dt_bias.copy_(inv_dt)
                module.dt_bias._no_reinit = True

            if isinstance(module, nn.Linear):
                if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            elif hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
                nn.init.ones_(module.weight)

            # Preserve the historical condition verbatim. With ordinary
            # modules recurse=False yields local parameter names, so this is
            # normally a no-op; changing it would change initialization.
            if self.config.rescale_prenorm_residual:
                for name, parameter in module.named_parameters(recurse=False):
                    if name == "out_proj.weight":
                        nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                        with torch.no_grad():
                            parameter /= math.sqrt(self.config.num_hidden_layers)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor | None:
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = int(cache_position[-1].item()) + 1
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(
            -1, 1
        )
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :, None, None, :
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                    padding_mask,
                    min_dtype,
                )
        if attention_mask is not None and attention_mask.device.type == "cuda":
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

    @staticmethod
    def _update_mamba_mask(
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
    ) -> torch.Tensor | None:
        if cache_position[0] > 0 or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            return None
        return attention_mask

    def forward_hidden_states_for_cce(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: object | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.tok_embeddings(input_ids)
        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)
        for mixer_block in self.layers:
            block_type = mixer_block.block_type
            if block_type == "mamba":
                layer_mask = mamba_mask
            elif block_type == "attention":
                layer_mask = causal_mask
            elif block_type == "mlp":
                layer_mask = None
            else:
                raise ValueError(f"invalid Nemotron-H block type: {block_type!r}")
            if block_type == "attention":
                with self._attention_context(hidden_states.device):
                    hidden_states = mixer_block(
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=layer_mask,
                    )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                )
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_params: object | None = None,
        cache_position: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = self.forward_hidden_states_for_cce(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_params=cache_params,
            cache_position=cache_position,
        )
        logits = self.output(hidden_states.to(self.output.weight.dtype)).float()
        if labels is not None:
            labels = labels.to(logits.device)
            return nn.functional.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
            )
        return logits


def _hf_backbone(model: NemotronHForTorchTitan) -> nn.Module:
    layers = getattr(model, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != 52:
        raise RuntimeError("public Nemotron-H wrapper does not expose 52 layers")
    return model


def parallelize_nemotron_h(
    model: NemotronHForTorchTitan,
    parallel_dims: Any,
    job_config: Any,
) -> NemotronHForTorchTitan:
    """Apply the subset of upstream TorchTitan parallelism used by this adapter."""

    if not isinstance(model, NemotronHForTorchTitan):
        raise TypeError("model must be NemotronHForTorchTitan")
    for name in ("tp_enabled", "cp_enabled", "pp_enabled"):
        if getattr(parallel_dims, name, False):
            raise NotImplementedError(f"Nemotron-H public adapter does not support {name}")
    if getattr(job_config.activation_checkpoint, "mode", "none") != "none":
        raise NotImplementedError(
            "Nemotron-H public adapter currently requires activation_checkpoint.mode='none'"
        )

    if getattr(parallel_dims, "fsdp_enabled", False):
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
        )
        from torchtitan.config import TORCH_DTYPE_MAP

        world_mesh = parallel_dims.world_mesh
        dimensions = (
            ("dp_replicate", "dp_shard_cp")
            if getattr(parallel_dims, "dp_replicate_enabled", False)
            else ("dp_shard_cp",)
        )
        mesh = world_mesh[dimensions]
        policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )
        options: dict[str, Any] = {"mesh": mesh, "mp_policy": policy}
        if job_config.training.enable_cpu_offload:
            options["offload_policy"] = CPUOffloadPolicy()
        match job_config.parallelism.fsdp_reshard_after_forward:
            case "always" | "default":
                reshard_after_forward = True
            case "never":
                reshard_after_forward = False
            case value:
                raise ValueError(f"invalid FSDP reshard policy: {value!r}")
        backbone = _hf_backbone(model)
        fully_shard(
            backbone.tok_embeddings,
            **options,
            reshard_after_forward=reshard_after_forward,
        )
        for layer in backbone.layers:
            fully_shard(layer, **options, reshard_after_forward=reshard_after_forward)
        fully_shard(
            [backbone.norm, backbone.output],
            **options,
            reshard_after_forward=(
                job_config.parallelism.fsdp_reshard_after_forward == "always"
            ),
        )
        fully_shard(model, **options)
    elif getattr(parallel_dims, "dp_replicate_enabled", False):
        from torchtitan.models.llama3.infra.parallelize import apply_ddp

        apply_ddp(
            model,
            parallel_dims.world_mesh,
            enable_compile=(
                job_config.compile.enable and "model" in job_config.compile.components
            ),
        )
    return model


__all__ = [
    "MODEL_FLAVOR",
    "MODEL_NAME",
    "TORCHTITAN_REVISION",
    "NemotronH8BArgs",
    "NemotronHForTorchTitan",
    "configure_nemotron_h_sdpa",
    "parallelize_nemotron_h",
    "verify_nemotron_h_sdpa",
]
