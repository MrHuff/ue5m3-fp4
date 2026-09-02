# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable fake-quantized linear for the proposed UE5M3 FP4 recipe."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import torch

from ue5m3_fp4.formats import RoundingMode, quantize_dequantize_blocks
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState, sample_global_amax

_INFERENCE_ACTIVATION_MODES = {
    "current_tensor",
    "training_replay",
    "calibrated_frozen",
}

_TE_SETTINGS_RHT_SIGNS_B16 = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)


class LinearBackend(StrEnum):
    """Numerical backend used by :class:`UE5M3Linear`."""

    PORTABLE = "portable_decoded_torch_reference"
    PROBE_MATCHED_TRITON = "probe_matched_triton_issue_rz"
    TRITON_QUANT_TORCH = "triton_quantized_torch_fp32"


def normalize_linear_backend(value: str | LinearBackend) -> LinearBackend:
    """Parse a public backend name without silently selecting a fallback."""

    if isinstance(value, LinearBackend):
        return value
    if not isinstance(value, str):
        raise TypeError("backend must be a string or LinearBackend")
    aliases = {
        "portable": LinearBackend.PORTABLE,
        LinearBackend.PORTABLE.value: LinearBackend.PORTABLE,
        "probe_matched": LinearBackend.PROBE_MATCHED_TRITON,
        "triton": LinearBackend.PROBE_MATCHED_TRITON,
        LinearBackend.PROBE_MATCHED_TRITON.value: LinearBackend.PROBE_MATCHED_TRITON,
        "torch_control": LinearBackend.TRITON_QUANT_TORCH,
        LinearBackend.TRITON_QUANT_TORCH.value: LinearBackend.TRITON_QUANT_TORCH,
    }
    try:
        return aliases[value.strip().lower()]
    except KeyError as error:
        choices = ", ".join(item.value for item in LinearBackend)
        raise ValueError(
            f"unknown UE5M3 linear backend {value!r}; expected {choices}"
        ) from error


def _finite_amax(tensor: torch.Tensor) -> torch.Tensor:
    return sample_global_amax(tensor, label="UE5M3 operand amax")


def _te_settings_rht_last_dim_b16(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the recovered fixed-sign, normalized B=16 Hadamard transform.

    Transformer Engine's NVFP4 recipe transforms only columnwise operand
    representations. The reported UE5M3 comparator materialized the transform
    through a BF16 Torch matmul, so this helper intentionally preserves that
    rounding boundary instead of replacing it with an algebraically equivalent
    FP32 implementation.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("RHT input must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError("the reported B=16 RHT expects a rank-2 operand")
    if tensor.shape[-1] % 16:
        raise ValueError("the reported B=16 RHT requires the last dimension divisible by 16")
    signs = torch.tensor(
        _TE_SETTINGS_RHT_SIGNS_B16,
        dtype=torch.float32,
        device=tensor.device,
    )
    hadamard = torch.ones((1, 1), dtype=torch.float32, device=tensor.device)
    for _ in range(4):
        hadamard = torch.cat(
            (
                torch.cat((hadamard, hadamard), dim=1),
                torch.cat((hadamard, -hadamard), dim=1),
            ),
            dim=0,
        )
    transformed = torch.matmul(
        (tensor.reshape(-1, 16).to(torch.float32) * signs).to(torch.bfloat16),
        (hadamard / 4.0).to(torch.bfloat16),
    )
    return transformed.reshape(tensor.shape).contiguous()


class _UE5M3LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        owner: UE5M3Linear,
    ) -> torch.Tensor:
        if owner._inference["active"] and torch.is_grad_enabled():
            raise RuntimeError("Post-load FP4 inference is forward-only")

        activation_reference = owner._forward_reference("activation", inputs)
        weight_reference = owner._forward_reference("weight", weight)
        q_inputs = owner._quantize(
            inputs,
            role="activation",
            tensor_reference=activation_reference,
        )
        q_weight = owner._quantize(
            weight,
            role="weight",
            tensor_reference=weight_reference,
        )

        input_shape = inputs.shape
        flat_inputs = q_inputs.reshape(-1, q_inputs.shape[-1]).to(torch.float32)
        output = flat_inputs @ q_weight.to(torch.float32).transpose(0, 1)
        if bias is not None:
            output = output + bias.to(torch.float32)
        output = output.reshape(*input_shape[:-1], weight.shape[0]).to(inputs.dtype)

        ctx.owner = owner
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.activation_reference = activation_reference.detach()
        ctx.weight_reference = weight_reference.detach()
        ctx.save_for_backward(inputs, weight)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        owner: UE5M3Linear = ctx.owner
        if owner._inference["active"]:
            raise RuntimeError("Backward is forbidden after FP4 inference is configured")

        inputs, weight = ctx.saved_tensors
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])

        # The same delayed dY amax is reused by both backward GEMMs.  Only the
        # payload draws are separate, and only dY uses stochastic rounding.
        dy_reference = owner.scale_state.reference(
            "upstream_gradient",
            owner.module_name,
            flat_grad,
        )
        q_dy_dgrad = owner._quantize(
            flat_grad,
            role="upstream_gradient",
            tensor_reference=dy_reference,
        )
        # Wgrad sees the transposed operands: dY.T is row-scaled along the
        # token/reduction dimension, while X is column-scaled along that same
        # dimension. Quantizing before the transpose would create different
        # block boundaries and would not model the training GEMM.
        dy_transposed = flat_grad.transpose(0, 1).contiguous()
        q_dy_wgrad = owner._quantize(
            dy_transposed,
            role="wgrad_upstream_gradient",
            tensor_reference=dy_reference,
        )
        q_weight = owner._quantize(
            weight,
            role="weight",
            tensor_reference=ctx.weight_reference,
        )
        saved_inputs_transposed = flat_inputs.transpose(0, 1).contiguous()
        q_saved_inputs_transposed = owner._quantize(
            saved_inputs_transposed,
            role="activation",
            tensor_reference=ctx.activation_reference,
        )

        grad_input = (q_dy_dgrad.to(torch.float32) @ q_weight.to(torch.float32)).reshape_as(
            inputs
        )
        grad_weight = q_dy_wgrad.to(torch.float32) @ q_saved_inputs_transposed.to(
            torch.float32
        ).transpose(0, 1)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = flat_grad.to(torch.float32).sum(dim=0).to(ctx.bias_dtype)
        return (
            grad_input.to(inputs.dtype),
            grad_weight.to(weight.dtype),
            grad_bias,
            None,
        )


def _reported_quantized_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    owner: UE5M3Linear,
    role_a: str,
    role_b: str,
    tensor_reference_a: torch.Tensor,
    tensor_reference_b: torch.Tensor,
    two_dimensional_b: bool,
) -> torch.Tensor:
    """Run one reported CUDA GEMM without any implicit backend fallback."""

    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError(
            "The probe-matched Triton backend requires CUDA operands; "
            "select backend='portable' explicitly for the CPU reference"
        )
    try:
        from ue5m3_fp4.backends.triton import (
            fake_quantize_gemm_operands,
            probe_matched_fp4_gemm,
            triton_available,
        )
    except ImportError as error:
        raise RuntimeError(
            "The probe-matched Triton backend was requested but its dependencies "
            "could not be imported"
        ) from error
    if not triton_available():
        raise RuntimeError(
            "The probe-matched Triton backend was requested but CUDA/Triton is unavailable"
        )
    common = {
        "block_size": owner.recipe.block_size,
        "scale_target_a": owner.recipe.scale_target_for(role_a, owner.module_name),
        "scale_target_b": owner.recipe.scale_target_for(role_b, owner.module_name),
        "tensor_amax_a": tensor_reference_a,
        "tensor_amax_b": tensor_reference_b,
        "rounding_a": owner.recipe.rounding_for(role_a),
        "rounding_b": owner.recipe.rounding_for(role_b),
        "two_dimensional_b": two_dimensional_b,
    }
    a = a.contiguous()
    b = b.contiguous()
    if owner.backend is LinearBackend.PROBE_MATCHED_TRITON:
        return probe_matched_fp4_gemm(
            a,
            b,
            **common,
            snap_to_1_over_1024=True,
        )
    if owner.backend is LinearBackend.TRITON_QUANT_TORCH:
        encoded_a, encoded_b = fake_quantize_gemm_operands(
            a,
            b,
            **common,
            output_domain="encoded",
            output_dtype=torch.bfloat16,
        )
        with torch.autocast(device_type="cuda", enabled=False):
            encoded_output = torch.mm(encoded_a.float(), encoded_b.float())
        alpha = (tensor_reference_a.float() * tensor_reference_b.float()) / (
            (common["scale_target_a"] * 6.0) * (common["scale_target_b"] * 6.0)
        )
        return encoded_output * alpha
    raise AssertionError(f"backend {owner.backend.value!r} is not a reported CUDA backend")


class _ProbeMatchedUE5M3LinearFunction(torch.autograd.Function):
    """Autograd definition for the reported Triton-quantized CUDA paths."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        owner: UE5M3Linear,
    ) -> torch.Tensor:
        if owner._inference["active"] and torch.is_grad_enabled():
            raise RuntimeError("Post-load FP4 inference is forward-only")
        if inputs.dtype is not torch.bfloat16:
            raise TypeError(
                "The probe-matched reported path requires BF16 activations at the "
                "linear boundary"
            )
        if weight.dtype not in {torch.bfloat16, torch.float32}:
            raise TypeError("The probe-matched path requires BF16 or FP32 master weights")

        input_shape = inputs.shape
        input_flat = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        weight_for_gemm = weight.to(torch.float32).transpose(0, 1).contiguous()
        activation_reference = owner._forward_reference("activation", input_flat)
        weight_reference = owner._forward_reference("weight", weight)
        activation_rht_reference = None
        if owner.recipe.randomized_hadamard_transform and not owner._inference["active"]:
            # The forward GEMM itself remains untransformed. This current amax
            # belongs to the transformed X.T representation retained for wgrad.
            input_transposed_rht = _te_settings_rht_last_dim_b16(
                input_flat.to(torch.float32).transpose(0, 1).contiguous()
            )
            activation_rht_reference = _finite_amax(input_transposed_rht)
        output = _reported_quantized_gemm(
            input_flat,
            weight_for_gemm,
            owner=owner,
            role_a="activation",
            role_b="weight",
            tensor_reference_a=activation_reference,
            tensor_reference_b=weight_reference,
            two_dimensional_b=owner.recipe.two_dimensional_weights,
        )
        if bias is not None:
            output = output + bias.to(torch.float32)
        output = output.reshape(*input_shape[:-1], weight.shape[0]).to(torch.bfloat16)

        ctx.owner = owner
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.activation_reference = activation_reference.detach()
        ctx.activation_rht_reference = (
            activation_rht_reference.detach() if activation_rht_reference is not None else None
        )
        ctx.weight_reference = weight_reference.detach()
        ctx.save_for_backward(inputs, weight)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        owner: UE5M3Linear = ctx.owner
        if owner._inference["active"]:
            raise RuntimeError("Backward is forbidden after FP4 inference is configured")

        inputs, weight = ctx.saved_tensors
        dy = grad_output.to(torch.bfloat16).reshape(-1, grad_output.shape[-1])
        dy = dy.to(torch.float32).contiguous()
        saved_inputs = inputs.to(torch.bfloat16).reshape(-1, inputs.shape[-1])
        saved_inputs = saved_inputs.to(torch.float32).contiguous()
        weight_for_dgrad = weight.to(torch.float32).contiguous()

        # The proposed no-RHT recipe shares this held dY reference between
        # dgrad and wgrad. The TE-settings comparator uses it only for dgrad;
        # wgrad samples the amax of transformed dY.T below. StochasticFast
        # payload draws remain separate for the two GEMMs.
        dy_reference = owner.scale_state.reference(
            "upstream_gradient",
            owner.module_name,
            dy,
        )
        grad_input = _reported_quantized_gemm(
            dy,
            weight_for_dgrad,
            owner=owner,
            role_a="upstream_gradient",
            role_b="weight",
            tensor_reference_a=dy_reference,
            tensor_reference_b=ctx.weight_reference,
            two_dimensional_b=owner.recipe.two_dimensional_weights,
        )

        dy_transposed = dy.transpose(0, 1).contiguous()
        wgrad_inputs = saved_inputs
        wgrad_dy_reference = dy_reference
        wgrad_activation_reference = ctx.activation_reference
        if owner.recipe.randomized_hadamard_transform:
            if ctx.activation_rht_reference is None:
                raise RuntimeError("missing saved transformed-activation amax for RHT wgrad")
            # This is the only transformed GEMM in the reported UE5M3
            # TE-settings path. Dgrad above uses untransformed dY and W.
            dy_transposed = _te_settings_rht_last_dim_b16(dy_transposed)
            saved_inputs_transposed = _te_settings_rht_last_dim_b16(
                saved_inputs.transpose(0, 1).contiguous()
            )
            wgrad_inputs = saved_inputs_transposed.transpose(0, 1).contiguous()
            wgrad_dy_reference = _finite_amax(dy_transposed)
            wgrad_activation_reference = ctx.activation_rht_reference
        grad_weight = _reported_quantized_gemm(
            dy_transposed,
            wgrad_inputs,
            owner=owner,
            role_a="wgrad_upstream_gradient",
            role_b="activation",
            tensor_reference_a=wgrad_dy_reference,
            tensor_reference_b=wgrad_activation_reference,
            two_dimensional_b=False,
        )

        grad_bias = None
        if ctx.has_bias:
            grad_bias = dy.sum(dim=0).to(ctx.bias_dtype)
        return (
            grad_input.reshape_as(inputs).to(torch.bfloat16),
            grad_weight.to(weight.dtype),
            grad_bias,
            None,
        )


class UE5M3Linear(torch.nn.Module):
    """An ``nn.Linear`` compatible UE5M3/E2M1 software reference.

    Call ``scale_state.begin_step(step)`` once before each training step.  For
    quantized evaluation, configure all converted modules through
    :class:`FP4InferenceScalingController` after loading the checkpoint.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        recipe: UE5M3Recipe | None = None,
        scale_state: TrainingScaleState | None = None,
        backend: str | LinearBackend = LinearBackend.PORTABLE,
        module_name: str = "linear",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        _initialize_parameters: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not module_name:
            raise ValueError("module_name must be non-empty")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.recipe = recipe or UE5M3Recipe.proposed()
        if scale_state is not None and scale_state.recipe != self.recipe:
            raise ValueError("scale_state.recipe must match the linear recipe")
        self.scale_state = scale_state or TrainingScaleState(self.recipe)
        self.backend = normalize_linear_backend(backend)
        if self.recipe.randomized_hadamard_transform:
            expected = UE5M3Recipe.transformer_engine_settings()
            if self.recipe != expected:
                raise ValueError(
                    "the public RHT implementation accepts only the reported "
                    "UE5M3 Transformer-Engine-settings recipe"
                )
            if self.backend is not LinearBackend.PROBE_MATCHED_TRITON:
                raise ValueError(
                    "the reported UE5M3 RHT comparator requires the probe-matched "
                    "Triton backend"
                )
        self.module_name = module_name
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        if _initialize_parameters:
            self.reset_parameters()
        self._inference_generation = 0
        self._inference: dict[str, Any] = {}
        self.reset_fp4_inference_scaling(clear_training_state=False)

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_float(
        cls,
        module: torch.nn.Linear,
        *,
        recipe: UE5M3Recipe,
        scale_state: TrainingScaleState,
        backend: str | LinearBackend = LinearBackend.PORTABLE,
        module_name: str,
    ) -> UE5M3Linear:
        if not isinstance(module, torch.nn.Linear):
            raise TypeError("from_float expects torch.nn.Linear")
        converted = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            recipe=recipe,
            scale_state=scale_state,
            backend=backend,
            module_name=module_name,
            device="meta",
            dtype=module.weight.dtype,
            _initialize_parameters=False,
        )
        # Reuse the exact Parameter objects. This preserves requires_grad,
        # optimizer references, meta parameters, and weight tying across
        # separately registered Linear modules.
        converted.weight = module.weight
        if module.bias is not None:
            converted.bias = module.bias
        return converted

    def _quantize(
        self,
        tensor: torch.Tensor,
        *,
        role: str,
        tensor_reference: torch.Tensor,
    ) -> torch.Tensor:
        return quantize_dequantize_blocks(
            tensor,
            block_size=self.recipe.block_size,
            scale_format=self.recipe.scale_format,
            scale_target=self.recipe.scale_target_for(role, self.module_name),
            rounding=self.recipe.rounding_for(role),
            tensor_reference=tensor_reference,
            two_dimensional=(role == "weight" and self.recipe.two_dimensional_weights),
        ).to(tensor.dtype)

    @staticmethod
    def _reference_tensor(value: float, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(value, device=tensor.device, dtype=torch.float32)

    def _forward_reference(self, role: str, tensor: torch.Tensor) -> torch.Tensor:
        if not self._inference["active"]:
            return self.scale_state.reference(role, self.module_name, tensor)

        state = self._inference
        if role == "weight":
            value = state["frozen_weights"].get("forward.w")
            if value is None:
                raise RuntimeError("Weight amax has not been frozen after checkpoint load")
            state["counters"]["weight_frozen_resolutions"] += 1
            return self._reference_tensor(value, tensor)

        if role != "activation":
            raise RuntimeError(f"Inference does not resolve role {role!r}")
        phase = state["phase"]
        mode = state["activation_mode"]
        if phase == "activation_calibration":
            current = float(_finite_amax(tensor).item())
            state["activation_calibration_max"] = max(
                state["activation_calibration_max"], current
            )
            state["activation_calibration_count"] += 1
            state["counters"]["activation_calibration_resolutions"] += 1
            return self._reference_tensor(current, tensor)
        if phase != "measuring":
            raise RuntimeError(f"FP4 inference forward is not allowed in phase {phase!r}")
        if mode == "current_tensor":
            current = float(_finite_amax(tensor).item())
            state["counters"]["activation_current_resolutions"] += 1
            return self._reference_tensor(current, tensor)
        if mode == "calibrated_frozen":
            value = state["frozen_activations"].get("forward.x")
            if value is None:
                raise RuntimeError("Activation amax has not been frozen")
            state["counters"]["activation_frozen_resolutions"] += 1
            return self._reference_tensor(value, tensor)
        if mode == "training_replay":
            logical_step = state["training_replay_logical_step"]
            if logical_step <= 0:
                raise RuntimeError("Advance the replay work unit before each forward")
            if state["training_replay_last_consumed_step"] == logical_step:
                raise RuntimeError(f"Replay step {logical_step} was consumed twice")
            last_refresh = state["training_replay_last_refresh_step"]
            if last_refresh == 0 or logical_step - last_refresh >= 50:
                current = float(_finite_amax(tensor).item())
                state["training_replay_cache"] = current
                state["training_replay_last_refresh_step"] = logical_step
                state["training_replay_refresh_trace"].append(
                    {"logical_step": logical_step, "amax": current}
                )
                state["counters"]["activation_replay_refreshes"] += 1
            else:
                state["counters"]["activation_replay_reuses"] += 1
            state["training_replay_last_consumed_step"] = logical_step
            return self._reference_tensor(state["training_replay_cache"], tensor)
        raise RuntimeError(f"Unsupported inference activation mode {mode!r}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"Expected input width {self.in_features}, got {inputs.shape[-1]}")
        # ``torch.autograd.Function.forward`` itself runs with grad mode
        # disabled, so enforce the forward-only inference contract here at the
        # public module boundary.
        if self._inference["active"] and torch.is_grad_enabled():
            raise RuntimeError(
                "Post-load FP4 inference requires torch.no_grad() or torch.inference_mode()"
            )
        if self.backend is LinearBackend.PORTABLE:
            function = _UE5M3LinearFunction
        elif self.backend in {
            LinearBackend.PROBE_MATCHED_TRITON,
            LinearBackend.TRITON_QUANT_TORCH,
        }:
            function = _ProbeMatchedUE5M3LinearFunction
        else:  # pragma: no cover - enum normalization makes this unreachable.
            raise AssertionError(f"unhandled linear backend: {self.backend}")
        return function.apply(inputs, self.weight, self.bias, self)

    # Post-load inference lifecycle.  These method names intentionally match
    # FP4InferenceScalingController's small module protocol.
    def reset_fp4_inference_scaling(
        self,
        *,
        clear_training_state: bool = True,
    ) -> dict[str, Any]:
        if type(clear_training_state) is not bool:
            raise TypeError("clear_training_state must be bool")
        cleared_entries = self.scale_state.clear_runtime() if clear_training_state else 0
        self._inference_generation += 1
        self._inference = {
            "active": False,
            "generation": self._inference_generation,
            "phase": "inactive",
            "activation_mode": None,
            "frozen_weights": {},
            "frozen_activations": {},
            "activation_calibration_max": 0.0,
            "activation_calibration_count": 0,
            "training_replay_logical_step": 0,
            "training_replay_last_consumed_step": 0,
            "training_replay_cache": 0.0,
            "training_replay_last_refresh_step": 0,
            "training_replay_refresh_trace": [],
            "legacy_training_state_cleared": {
                "runtime_cache_cleared": clear_training_state,
                "delayed_amax_entries": cleared_entries,
                "training_cache_inherited": False,
            },
            "counters": {
                "resets": self._inference_generation,
                "weight_calibrations": 0,
                "weight_frozen_resolutions": 0,
                "activation_current_resolutions": 0,
                "activation_calibration_resolutions": 0,
                "activation_frozen_resolutions": 0,
                "activation_replay_refreshes": 0,
                "activation_replay_reuses": 0,
            },
        }
        return {
            "generation": self._inference_generation,
            "legacy_training_state_cleared": dict(
                self._inference["legacy_training_state_cleared"]
            ),
        }

    def configure_fp4_inference_scaling(self, activation_mode: str) -> None:
        if self.training:
            raise RuntimeError("Call model.eval() before configuring FP4 inference")
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise RuntimeError(
                "The deterministic FP4 inference scaling protocol requires one process"
            )
        forward_rounding = {
            "activation_rounding": self.recipe.activation_rounding,
            "weight_rounding": self.recipe.weight_rounding,
            "scale_rounding": self.recipe.scale_rounding,
        }
        stochastic = {
            name: mode.value
            for name, mode in forward_rounding.items()
            if mode is RoundingMode.STOCHASTIC_8BIT_MIDPOINT
        }
        if stochastic:
            details = ", ".join(f"{name}={mode}" for name, mode in sorted(stochastic.items()))
            raise ValueError(
                f"Deterministic FP4 inference rejects stochastic forward rounding: {details}"
            )
        if activation_mode not in _INFERENCE_ACTIVATION_MODES:
            raise ValueError(f"Unknown FP4 inference mode {activation_mode!r}")
        if activation_mode == "training_replay" and self.recipe.delayed_scale_interval != 50:
            raise ValueError(
                "training_replay implements the recorded D=50 protocol and "
                "requires delayed_scale_interval=50"
            )
        state = self._inference
        if state["active"]:
            raise RuntimeError("FP4 inference scaling was configured twice")
        state["active"] = True
        state["activation_mode"] = activation_mode
        state["phase"] = "configured"

    def calibrate_and_freeze_fp4_inference_weight_scale(self) -> dict[str, Any]:
        state = self._inference
        if state["phase"] != "configured":
            raise RuntimeError("Weight calibration requires configured inference")
        value = float(_finite_amax(self.weight).item())
        state["frozen_weights"]["forward.w"] = value
        state["counters"]["weight_calibrations"] += 1
        state["phase"] = "weights_frozen"
        return {"key": "forward.w", "amax": value, "device": self.weight.device.type}

    def begin_fp4_inference_activation_calibration(self) -> None:
        state = self._inference
        if state["activation_mode"] != "calibrated_frozen":
            raise RuntimeError("Activation calibration requires calibrated_frozen mode")
        if state["phase"] != "weights_frozen":
            raise RuntimeError("Freeze weights before activation calibration")
        state["activation_calibration_max"] = 0.0
        state["activation_calibration_count"] = 0
        state["phase"] = "activation_calibration"

    def freeze_fp4_inference_activation_scales(self) -> dict[str, Any]:
        state = self._inference
        if state["phase"] != "activation_calibration":
            raise RuntimeError("No activation calibration is active")
        if state["activation_calibration_count"] <= 0:
            raise RuntimeError("No activation operands were observed")
        value = state["activation_calibration_max"]
        count = state["activation_calibration_count"]
        state["frozen_activations"]["forward.x"] = value
        state["phase"] = "activations_frozen"
        return {"forward.x": {"amax": value, "observations": count}}

    def begin_fp4_inference_measurement(self) -> None:
        state = self._inference
        expected = (
            "activations_frozen"
            if state["activation_mode"] == "calibrated_frozen"
            else "weights_frozen"
        )
        if state["phase"] != expected:
            raise RuntimeError(
                f"Measurement requires phase {expected!r}, got {state['phase']!r}"
            )
        state["phase"] = "measuring"

    def advance_fp4_inference_training_replay_step(self, logical_step: int) -> None:
        state = self._inference
        if state["activation_mode"] != "training_replay":
            raise RuntimeError("Replay advancement requires training_replay mode")
        expected = state["training_replay_logical_step"] + 1
        if logical_step != expected:
            raise RuntimeError(f"Expected replay step {expected}, got {logical_step}")
        state["training_replay_logical_step"] = logical_step

    def reset_fp4_inference_measurement_counters(self) -> None:
        counters = self._inference["counters"]
        for name in tuple(counters):
            if name not in {"resets", "weight_calibrations"}:
                counters[name] = 0

    def fp4_inference_scaling_report(self) -> dict[str, Any]:
        state = self._inference
        probe_matched = self.backend is LinearBackend.PROBE_MATCHED_TRITON
        triton_torch_control = self.backend is LinearBackend.TRITON_QUANT_TORCH
        matmul_policy = {
            "input_dtype": "bfloat16" if probe_matched else "float32",
            "device_type": self.weight.device.type,
            "torch_float32_matmul_precision": None,
            "cuda_matmul_fp32_precision": None,
            "cuda_matmul_allow_tf32": None,
        }
        if self.weight.device.type == "cuda":
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                matmul_policy["cuda_matmul_fp32_precision"] = str(
                    torch.backends.cuda.matmul.fp32_precision
                )
            else:  # PyTorch 2.7 and earlier precision API.
                matmul_policy["torch_float32_matmul_precision"] = (
                    torch.get_float32_matmul_precision()
                )
                matmul_policy["cuda_matmul_allow_tf32"] = bool(
                    torch.backends.cuda.matmul.allow_tf32
                )
        return {
            "schema": "ue5m3_fp4_inference_module_scaling_v1",
            "active": state["active"],
            "generation": state["generation"],
            "phase": state["phase"],
            "activation_mode": state["activation_mode"],
            "frozen_global_amax": {
                "weights": dict(state["frozen_weights"]),
                "activations": dict(state["frozen_activations"]),
            },
            "activation_calibration_counts": {
                "forward.x": state["activation_calibration_count"]
            },
            "counters": dict(state["counters"]),
            "legacy_training_state_cleared": dict(state["legacy_training_state_cleared"]),
            "rounding_modes": {
                "round_mode_activations": self.recipe.activation_rounding,
                "round_mode_weights": self.recipe.weight_rounding,
            },
            "training_replay": {
                "interval": self.recipe.delayed_scale_interval,
                "logical_step": state["training_replay_logical_step"],
                "last_consumed_step": state["training_replay_last_consumed_step"],
                "cache": {"forward.x": state["training_replay_cache"]},
                "last_refresh_steps": {"forward.x": state["training_replay_last_refresh_step"]},
                "refresh_trace": {"forward.x": list(state["training_replay_refresh_trace"])},
            },
            "format": {
                "payload": self.recipe.payload_format.name,
                "block_size": self.recipe.block_size,
                "scale_type": self.recipe.scale_format.name,
                "scale_max_activations": self.recipe.scale_target_for(
                    "activation", self.module_name
                ),
                "scale_max_weights": self.recipe.scale_target_for("weight", self.module_name),
                "gemm_output_model": (
                    "encoded_operand_k64_issue_rz_bf16_gemm_final_snap_1_over_1024"
                    if probe_matched
                    else (
                        "encoded_operand_torch_fp32_matmul"
                        if triton_torch_control
                        else "decoded_operand_torch_matmul"
                    )
                ),
                "torch_matmul_policy": matmul_policy,
                "native_hardware": False,
                "encode_centric": probe_matched or triton_torch_control,
                "use_encoded_gemm": probe_matched or triton_torch_control,
                "linear_backend": self.backend.value,
                "randomized_hadamard_transform": {
                    "forward_gemm": False,
                    "data_gradient_gemm": False,
                    "weight_gradient_operands": bool(self.recipe.randomized_hadamard_transform),
                    "block_size": (16 if self.recipe.randomized_hadamard_transform else None),
                },
            },
        }


__all__ = [
    "LinearBackend",
    "UE5M3Linear",
    "normalize_linear_backend",
]
