# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import ue5m3_fp4.integrations.torchtitan.comparators as comparators
import ue5m3_fp4.nn.linear as linear_module
from ue5m3_fp4.integrations.torchtitan.comparators import (
    NativeNVFP4Variant,
    TransformerEngineRuntime,
    convert_native_nvfp4_nemotron_h,
    convert_ue5m3_te_settings_nemotron_h,
)
from ue5m3_fp4.integrations.torchtitan.selection import (
    REPORTED_NEMOTRON_H_8B_PATTERN,
    FP32OutputLinear,
    reported_projection,
)
from ue5m3_fp4.nn.linear import LinearBackend, UE5M3Linear
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState


class _SyntheticMixer(nn.Module):
    def __init__(self, block_type: str) -> None:
        super().__init__()
        projections = {
            "M": ("in_proj", "out_proj"),
            "*": ("q_proj", "k_proj", "v_proj", "o_proj"),
            "-": ("up_proj", "down_proj"),
        }[block_type]
        for projection in projections:
            setattr(self, projection, nn.Linear(1, 1, bias=False))


class _SyntheticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=52,
            hybrid_override_pattern=REPORTED_NEMOTRON_H_8B_PATTERN,
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict({"mixer": _SyntheticMixer(block_type)})
                for block_type in REPORTED_NEMOTRON_H_8B_PATTERN
            ]
        )
        self.output = nn.Linear(1, 2, bias=False)


class _FakeQParams:
    def __init__(self, *, rht: bool, stochastic: bool, two_dimensional: bool) -> None:
        self.random_hadamard_transform = rht
        self.stochastic_rounding = stochastic
        self.fp4_2d_quantization = two_dimensional


class _FakeNVFP4Recipe:
    def __init__(
        self,
        *,
        disable_rht: bool = False,
        disable_stochastic_rounding: bool = False,
        disable_2d_quantization: bool = False,
    ) -> None:
        self.disable_rht = disable_rht
        self.disable_stochastic_rounding = disable_stochastic_rounding
        self.disable_2d_quantization = disable_2d_quantization
        self.fp4_quant_fwd_inp = _FakeQParams(
            rht=not disable_rht,
            stochastic=False,
            two_dimensional=False,
        )
        self.fp4_quant_fwd_weight = _FakeQParams(
            rht=False,
            stochastic=False,
            two_dimensional=not disable_2d_quantization,
        )
        self.fp4_quant_bwd_grad = _FakeQParams(
            rht=not disable_rht,
            stochastic=not disable_stochastic_rounding,
            two_dimensional=False,
        )


class _FakeTELinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        params_dtype: torch.dtype,
        init_method: Any,
        device: torch.device | str,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=params_dtype,
        )
        init_method(self.weight)
        self.fp8_meta_tensors_initialized = False
        self.quantizers = {"forward": [], "backward": []}
        self._fp8_workspaces = {}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(
            inputs.to(torch.float32),
            self.weight.to(torch.float32),
            self.bias.to(torch.float32) if self.bias is not None else None,
        ).to(inputs.dtype)


class _FakeTEAPI:
    Linear = _FakeTELinear

    @staticmethod
    @contextlib.contextmanager
    def fp8_autocast(*, enabled: bool, fp8_recipe: Any):
        assert enabled is True
        assert isinstance(fp8_recipe, _FakeNVFP4Recipe)
        yield


def _fake_runtime() -> TransformerEngineRuntime:
    return TransformerEngineRuntime(
        version=comparators.TRANSFORMER_ENGINE_VERSION,
        api=_FakeTEAPI,
        recipe_type=_FakeNVFP4Recipe,
        base_module=SimpleNamespace(_2X_ACC_DGRAD=True, _2X_ACC_WGRAD=True),
        linear_module=SimpleNamespace(_2X_ACC_DGRAD=True, _2X_ACC_WGRAD=True),
    )


@pytest.mark.parametrize(
    ("variant", "expected_count", "expected_final_bf16", "rht_disabled"),
    [
        (NativeNVFP4Variant.TRANSFORMER_ENGINE_RECIPE, 96, 16, False),
        (NativeNVFP4Variant.NO_RHT_ALL_LINEARS, 112, 0, True),
    ],
)
def test_native_comparators_have_exact_coverage_controls_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    variant: NativeNVFP4Variant,
    expected_count: int,
    expected_final_bf16: int,
    rht_disabled: bool,
) -> None:
    runtime = _fake_runtime()
    monkeypatch.setattr(comparators, "require_pinned_transformer_engine", lambda: runtime)
    model = _SyntheticModel()

    conversion = convert_native_nvfp4_nemotron_h(model, variant=variant)

    native_linears = [
        module
        for module in model.modules()
        if getattr(module, "_ue5m3_reported_internal_linear", None)
        == "native_transformer_engine_nvfp4"
    ]
    assert len(conversion.fp4_linears) == expected_count
    assert len(native_linears) == expected_count
    assert conversion.final_bf16_linears == expected_final_bf16
    assert conversion.current_tensor_interval == 1
    assert isinstance(model.output, FP32OutputLinear)
    assert model._ue5m3_reported_conversion["effective_interval"] == 1
    assert model._ue5m3_reported_conversion["delayed_amax_fields_applied"] is False
    assert model._ue5m3_reported_conversion["rht_disabled"] is rht_disabled
    assert all(module.bound_recipe.disable_rht is rht_disabled for module in native_linears)
    assert all(not module.bound_recipe.disable_stochastic_rounding for module in native_linears)
    assert all(not module.bound_recipe.disable_2d_quantization for module in native_linears)
    assert runtime.base_module._2X_ACC_DGRAD is False
    assert runtime.base_module._2X_ACC_WGRAD is False
    assert runtime.linear_module._2X_ACC_DGRAD is False
    assert runtime.linear_module._2X_ACC_WGRAD is False

    final_eligible = [
        module
        for name, module in model.named_modules()
        if (projection := reported_projection(name)) is not None and projection[0] >= 44
    ]
    if variant is NativeNVFP4Variant.TRANSFORMER_ENGINE_RECIPE:
        assert len(final_eligible) == 16
        assert all(type(module) is nn.Linear for module in final_eligible)
    else:
        assert not final_eligible or all(module in native_linears for module in final_eligible)


def test_native_alignment_is_explicit_eval_only_and_shape_preserving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_runtime()
    monkeypatch.setattr(comparators, "require_pinned_transformer_engine", lambda: runtime)
    model = _SyntheticModel().eval()
    convert_native_nvfp4_nemotron_h(
        model,
        variant=NativeNVFP4Variant.TRANSFORMER_ENGINE_RECIPE,
    )
    layer = model.layers[0]["mixer"].in_proj

    layer.enable_nvfp4_inference_alignment()
    with torch.no_grad():
        output = layer(torch.ones((1, 3, 1), dtype=torch.bfloat16))

    assert output.shape == (1, 3, 1)
    report = layer.native_nvfp4_report()
    assert report["numeric_backend"] == "native_transformer_engine_nvfp4"
    assert report["tensor_scaling"] == "current_tensor"
    assert report["effective_interval"] == 1
    assert report["delayed_amax_fields_applied"] is False
    assert report["alignment"]["counters"]["padding_row_count"] == 13


def test_ue5m3_te_settings_has_exact_coverage_and_current_d1() -> None:
    model = _SyntheticModel()

    conversion = convert_ue5m3_te_settings_nemotron_h(model)

    converted = [module for module in model.modules() if isinstance(module, UE5M3Linear)]
    assert len(conversion.fp4_linears) == 96
    assert len(converted) == 96
    assert conversion.final_bf16_linears == 16
    assert conversion.current_tensor_interval == 1
    assert {module.backend for module in converted} == {LinearBackend.PROBE_MATCHED_TRITON}
    assert {module.recipe for module in converted} == {
        UE5M3Recipe.transformer_engine_settings()
    }
    assert isinstance(model.output, FP32OutputLinear)
    assert model._ue5m3_reported_conversion["rht"] == {
        "forward_gemm": False,
        "data_gradient_gemm": False,
        "weight_gradient_upstream_gradient": True,
        "weight_gradient_activation": True,
        "weights": False,
        "block_size": 16,
    }
    final_eligible = [
        module
        for name, module in model.named_modules()
        if (projection := reported_projection(name)) is not None and projection[0] >= 44
    ]
    assert len(final_eligible) == 16
    assert all(type(module) is nn.Linear for module in final_eligible)


def test_ue5m3_te_settings_transforms_only_wgrad_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = UE5M3Recipe.transformer_engine_settings()
    state = TrainingScaleState(recipe)
    layer = UE5M3Linear(
        64,
        64,
        bias=False,
        recipe=recipe,
        scale_state=state,
        backend=LinearBackend.PROBE_MATCHED_TRITON,
        module_name="layers.0.mixer.in_proj",
        dtype=torch.float32,
    )
    calls: list[dict[str, Any]] = []
    direct_amax_samples: list[tuple[torch.Tensor, str]] = []

    original_sample_global_amax = linear_module.sample_global_amax

    def record_direct_amax_sample(
        tensor: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        direct_amax_samples.append((tensor.detach().clone(), label))
        return original_sample_global_amax(tensor, label=label)

    def fake_gemm(a: torch.Tensor, b: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append({"a": a.detach().clone(), "b": b.detach().clone(), **kwargs})
        return a.to(torch.float32) @ b.to(torch.float32)

    monkeypatch.setattr(linear_module, "sample_global_amax", record_direct_amax_sample)
    monkeypatch.setattr(linear_module, "_reported_quantized_gemm", fake_gemm)
    state.begin_step(1)
    inputs = torch.randn((64, 64), dtype=torch.bfloat16, requires_grad=True)
    layer(inputs).sum().backward()

    assert [call["role_a"] for call in calls] == [
        "activation",
        "upstream_gradient",
        "wgrad_upstream_gradient",
    ]
    assert torch.equal(calls[0]["a"], inputs.detach())
    assert torch.equal(calls[1]["a"], torch.ones_like(calls[1]["a"]))
    expected_dy_t = linear_module._te_settings_rht_last_dim_b16(
        calls[1]["a"].transpose(0, 1).contiguous()
    )
    expected_x_t = linear_module._te_settings_rht_last_dim_b16(
        inputs.detach().to(torch.float32).transpose(0, 1).contiguous()
    )
    assert torch.equal(calls[2]["a"], expected_dy_t)
    assert torch.equal(calls[2]["b"], expected_x_t.transpose(0, 1).contiguous())
    assert float(calls[2]["tensor_reference_a"]) == float(expected_dy_t.abs().amax())
    assert float(calls[2]["tensor_reference_b"]) == float(expected_x_t.abs().amax())
    assert [label for _, label in direct_amax_samples] == [
        "UE5M3 operand amax",
        "UE5M3 operand amax",
    ]
    assert torch.equal(direct_amax_samples[0][0], expected_x_t)
    assert torch.equal(direct_amax_samples[1][0], expected_dy_t)
    assert all(
        recipe.scale_target_for(call["role_a"], layer.module_name) == 448.0 for call in calls
    )


@pytest.mark.parametrize(
    ("filename", "converter"),
    [
        ("nemotron_h_8b_nvfp4_te.toml", "ue5m3_fp4.native_nvfp4_te"),
        (
            "nemotron_h_8b_nvfp4_no_rht_all_linears.toml",
            "ue5m3_fp4.native_nvfp4_no_rht_all",
        ),
        (
            "nemotron_h_8b_ue5m3_te_settings.toml",
            "ue5m3_fp4.ue5m3_te_settings_d1",
        ),
        (
            "nemotron_h_8b_ue5m3_torch_control.toml",
            "ue5m3_fp4.torch_control_b16_d50",
        ),
    ],
)
def test_comparator_configs_are_public_torchtitan_configs(
    filename: str,
    converter: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "reproduce" / "configs" / filename).open("rb") as handle:
        config = tomllib.load(handle)

    assert config["model"]["converters"] == [converter]
    assert config["model"]["name"] == "nemotron_h_ue5m3"
    assert config["experimental"]["custom_import"] == (
        "ue5m3_fp4.integrations.torchtitan.plugin"
    )
    assert "te_fp4" not in config
    assert "mxfp_custom" not in config


def test_native_runtime_fails_closed_on_the_unpinned_host_version() -> None:
    try:
        import transformer_engine
    except (ImportError, OSError):
        with pytest.raises(RuntimeError, match="pinned Transformer Engine"):
            comparators.require_pinned_transformer_engine()
        return
    if transformer_engine.__version__ == comparators.TRANSFORMER_ENGINE_VERSION:
        pytest.skip("host already has the exact pinned Transformer Engine build")
    with pytest.raises(RuntimeError, match="requires Transformer Engine"):
        comparators.require_pinned_transformer_engine()
