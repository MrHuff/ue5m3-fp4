# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import ue5m3_fp4.integrations.torchtitan.remote_code as remote_code
import ue5m3_fp4.nn.linear as linear_module
from ue5m3_fp4.integrations.torchtitan.selection import (
    REPORTED_ELIGIBLE_LINEAR_COUNT,
    REPORTED_NEMOTRON_H_8B_PATTERN,
    FP32OutputLinear,
    convert_reported_nemotron_h,
)
from ue5m3_fp4.integrations.torchtitan.trainer import begin_training_step
from ue5m3_fp4.nn.linear import LinearBackend, UE5M3Linear
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.inference import FP4InferenceScalingController
from ue5m3_fp4.scaling.training import TrainingScaleState


def test_shared_trainer_hook_is_a_noop_for_bf16_model() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))

    assert begin_training_step([model], 1) == 0


def test_remote_code_patch_is_hash_locked_atomic_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = f"prefix\n{remote_code._ORIGINAL_FUSED_BLOCK}suffix\n".encode()
    patched = f"prefix\n{remote_code._PATCHED_FUSED_BLOCK}suffix\n".encode()
    monkeypatch.setattr(
        remote_code,
        "ORIGINAL_MODELING_SHA256",
        hashlib.sha256(original).hexdigest(),
    )
    monkeypatch.setattr(
        remote_code,
        "PATCHED_MODELING_SHA256",
        hashlib.sha256(patched).hexdigest(),
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    modeling = assets / "modeling_nemotron_h.py"
    modeling.write_bytes(original)

    assert (
        remote_code.patch_nemotron_h_remote_code(modeling)
        == hashlib.sha256(patched).hexdigest()
    )
    assert modeling.read_bytes() == patched
    assert (
        remote_code.patch_nemotron_h_remote_code(modeling)
        == hashlib.sha256(patched).hexdigest()
    )
    assert remote_code.require_patched_nemotron_h_assets(assets) == assets


def test_remote_code_patch_rejects_unknown_source_without_writing(tmp_path: Path) -> None:
    modeling = tmp_path / "modeling_nemotron_h.py"
    unknown = b"not the pinned source"
    modeling.write_bytes(unknown)

    with pytest.raises(RuntimeError, match="unrecognized"):
        remote_code.patch_nemotron_h_remote_code(modeling)

    assert modeling.read_bytes() == unknown


@pytest.mark.parametrize(
    "backend",
    [LinearBackend.PROBE_MATCHED_TRITON, LinearBackend.TRITON_QUANT_TORCH],
)
def test_cuda_backends_fail_closed_on_cpu(backend: LinearBackend) -> None:
    layer = UE5M3Linear(
        64,
        64,
        dtype=torch.bfloat16,
        backend=backend,
    )
    layer.scale_state.begin_step(1)

    with pytest.raises(RuntimeError, match="requires CUDA operands"):
        layer(torch.randn(64, 64, dtype=torch.bfloat16))


def test_exact_autograd_routes_all_three_gemms_with_reported_operand_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = UE5M3Recipe.proposed()
    state = TrainingScaleState(recipe)
    layer = UE5M3Linear(
        4,
        3,
        recipe=recipe,
        scale_state=state,
        backend=LinearBackend.PROBE_MATCHED_TRITON,
        module_name="layers.45.mixer.down_proj",
        dtype=torch.float32,
    )
    calls: list[dict[str, Any]] = []

    def fake_gemm(a: torch.Tensor, b: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append({"a": a, "b": b, **kwargs})
        return a.to(torch.float32) @ b.to(torch.float32)

    monkeypatch.setattr(linear_module, "_reported_quantized_gemm", fake_gemm)
    state.begin_step(1)
    inputs = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
    output = layer(inputs)
    output.sum().backward()

    assert [call["role_a"] for call in calls] == [
        "activation",
        "upstream_gradient",
        "wgrad_upstream_gradient",
    ]
    assert [call["role_b"] for call in calls] == ["weight", "weight", "activation"]
    assert [call["two_dimensional_b"] for call in calls] == [True, True, False]
    assert [tuple(call["a"].shape) for call in calls] == [(2, 4), (2, 3), (3, 2)]
    assert [tuple(call["b"].shape) for call in calls] == [(4, 3), (3, 4), (2, 4)]
    assert calls[1]["tensor_reference_a"] is calls[2]["tensor_reference_a"]
    assert recipe.scale_target_for(calls[0]["role_a"], layer.module_name) == 448.0
    assert recipe.scale_target_for(calls[1]["role_a"], layer.module_name) == 448.0
    assert recipe.scale_target_for(calls[2]["role_a"], layer.module_name) == 2_048.0
    assert output.dtype is torch.bfloat16
    assert inputs.grad is not None and inputs.grad.dtype is torch.bfloat16
    assert layer.weight.grad is not None and layer.weight.grad.dtype is torch.float32


class _SyntheticMixer(torch.nn.Module):
    def __init__(self, block_type: str) -> None:
        super().__init__()
        projections = {
            "M": ("in_proj", "out_proj"),
            "*": ("q_proj", "k_proj", "v_proj", "o_proj"),
            "-": ("up_proj", "down_proj"),
        }[block_type]
        for projection in projections:
            setattr(self, projection, torch.nn.Linear(1, 1, bias=False))


class _VanillaFusedSyntheticMixer(_SyntheticMixer):
    def cuda_kernels_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.training:
            outproj_weight = self.out_proj.weight
            return hidden_states @ outproj_weight
        return self.out_proj(hidden_states)


class _PatchedFusedSyntheticMixer(_SyntheticMixer):
    def cuda_kernels_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if type(self.out_proj) is nn.Linear:
            outproj_weight = self.out_proj.weight
            return hidden_states @ outproj_weight
        scan_output = hidden_states
        out = self.out_proj(scan_output)
        return out


class _SyntheticReportedModel(torch.nn.Module):
    def __init__(self, *, mamba_mixer: type[_SyntheticMixer] = _SyntheticMixer) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=52,
            hybrid_override_pattern=REPORTED_NEMOTRON_H_8B_PATTERN,
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "mixer": (
                            mamba_mixer(block_type)
                            if block_type == "M"
                            else _SyntheticMixer(block_type)
                        )
                    }
                )
                for block_type in REPORTED_NEMOTRON_H_8B_PATTERN
            ]
        )
        self.output = torch.nn.Linear(1, 2, bias=False)


@pytest.mark.parametrize("block_size", [16, 32])
def test_reported_converter_installs_exact_backend_and_fp32_head(block_size: int) -> None:
    model = _SyntheticReportedModel()
    recipe = replace(
        UE5M3Recipe.proposed(),
        name=f"proposed_ue5m3_b{block_size}_d50",
        block_size=block_size,
    )

    conversion = convert_reported_nemotron_h(model, recipe=recipe)

    assert len(conversion.fp4_linears) == REPORTED_ELIGIBLE_LINEAR_COUNT
    assert conversion.linear_backend == LinearBackend.PROBE_MATCHED_TRITON.value
    assert conversion.probe_matched_backend is True
    assert isinstance(model.output, FP32OutputLinear)
    converted = [module for module in model.modules() if isinstance(module, UE5M3Linear)]
    assert len(converted) == REPORTED_ELIGIBLE_LINEAR_COUNT
    assert {module.backend for module in converted} == {LinearBackend.PROBE_MATCHED_TRITON}
    assert {module.recipe.block_size for module in converted} == {block_size}


def test_torch_control_converter_metadata_does_not_claim_probe_matching() -> None:
    model = _SyntheticReportedModel()

    conversion = convert_reported_nemotron_h(
        model,
        backend=LinearBackend.TRITON_QUANT_TORCH,
    )

    assert conversion.linear_backend == LinearBackend.TRITON_QUANT_TORCH.value
    assert conversion.probe_matched_backend is False
    assert model._ue5m3_reported_conversion["probe_matched_backend"] is False


def test_training_conversion_rejects_unpatched_direct_weight_mamba_fusion() -> None:
    model = _SyntheticReportedModel(mamba_mixer=_VanillaFusedSyntheticMixer)

    with pytest.raises(RuntimeError, match="fuses out_proj"):
        convert_reported_nemotron_h(model)


def test_eval_conversion_allows_vanilla_source_because_eval_dispatches_module() -> None:
    model = _SyntheticReportedModel(mamba_mixer=_VanillaFusedSyntheticMixer).eval()

    conversion = convert_reported_nemotron_h(model)

    assert len(conversion.fp4_linears) == REPORTED_ELIGIBLE_LINEAR_COUNT


def test_training_conversion_accepts_hash_locked_dispatch_shape() -> None:
    model = _SyntheticReportedModel(mamba_mixer=_PatchedFusedSyntheticMixer)

    conversion = convert_reported_nemotron_h(model)

    assert len(conversion.fp4_linears) == REPORTED_ELIGIBLE_LINEAR_COUNT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("block_size", [16, 32])
def test_exact_backend_runs_finite_forward_dgrad_and_wgrad(block_size: int) -> None:
    recipe = replace(
        UE5M3Recipe.proposed(),
        name=f"proposed_ue5m3_b{block_size}_d50",
        block_size=block_size,
    )
    state = TrainingScaleState(recipe)
    layer = UE5M3Linear(
        64,
        64,
        recipe=recipe,
        scale_state=state,
        backend=LinearBackend.PROBE_MATCHED_TRITON,
        module_name="layers.45.mixer.down_proj",
        device="cuda",
        dtype=torch.float32,
    )
    state.begin_step(1)
    inputs = torch.randn(
        64,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    torch.manual_seed(1234)
    output = layer(inputs)
    output.float().square().mean().backward()

    assert output.shape == (64, 64)
    assert output.dtype is torch.bfloat16
    assert bool(torch.isfinite(output).all())
    assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
    assert layer.weight.grad is not None and bool(torch.isfinite(layer.weight.grad).all())
    entries = state.report()["entries"]
    assert {(entry["role"], entry["module_name"]) for entry in entries} == {
        ("activation", "layers.45.mixer.down_proj"),
        ("upstream_gradient", "layers.45.mixer.down_proj"),
        ("weight", "layers.45.mixer.down_proj"),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_linear_forward_is_the_clean_probe_matched_composition() -> None:
    from ue5m3_fp4.backends.triton import probe_matched_fp4_gemm

    recipe = UE5M3Recipe.proposed()
    state = TrainingScaleState(recipe)
    layer = UE5M3Linear(
        64,
        64,
        recipe=recipe,
        scale_state=state,
        backend=LinearBackend.PROBE_MATCHED_TRITON,
        device="cuda",
        dtype=torch.float32,
        bias=False,
    )
    inputs = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    input_amax = inputs.float().abs().amax().reshape(1)
    weight_amax = layer.weight.detach().float().abs().amax().reshape(1)
    expected = probe_matched_fp4_gemm(
        inputs,
        layer.weight.float().transpose(0, 1).contiguous(),
        block_size=16,
        tensor_amax_a=input_amax,
        tensor_amax_b=weight_amax,
        two_dimensional_b=True,
        snap_to_1_over_1024=True,
    ).to(torch.bfloat16)

    state.begin_step(1)
    actual = layer(inputs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_torch_control_uses_encoded_operands_and_fp32_torch_mm() -> None:
    from ue5m3_fp4.backends.triton import fake_quantize_gemm_operands

    recipe = UE5M3Recipe.proposed()
    state = TrainingScaleState(recipe)
    layer = UE5M3Linear(
        64,
        64,
        recipe=recipe,
        scale_state=state,
        backend=LinearBackend.TRITON_QUANT_TORCH,
        device="cuda",
        dtype=torch.float32,
        bias=False,
    )
    inputs = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    input_amax = inputs.float().abs().amax().reshape(1)
    weight_amax = layer.weight.detach().float().abs().amax().reshape(1)
    encoded_x, encoded_w = fake_quantize_gemm_operands(
        inputs,
        layer.weight.float().transpose(0, 1).contiguous(),
        block_size=16,
        tensor_amax_a=input_amax,
        tensor_amax_b=weight_amax,
        two_dimensional_b=True,
        output_domain="encoded",
        output_dtype=torch.bfloat16,
    )
    alpha = (input_amax * weight_amax) / ((448.0 * 6.0) ** 2)
    expected = (torch.mm(encoded_x.float(), encoded_w.float()) * alpha).to(torch.bfloat16)

    state.begin_step(1)
    actual = layer(inputs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_inference_provenance_names_the_probe_matched_numeric_path() -> None:
    model = torch.nn.Sequential(
        UE5M3Linear(
            64,
            64,
            backend=LinearBackend.PROBE_MATCHED_TRITON,
            device="cuda",
            dtype=torch.float32,
        )
    ).eval()
    controller = FP4InferenceScalingController(
        model,
        activation_mode="current_tensor",
        checkpoint_identity={"checkpoint_id": "synthetic-exact-test"},
    )
    controller.reset_after_checkpoint_load()
    controller.calibrate_and_freeze_weights()
    controller.begin_measurement()
    with torch.no_grad():
        model(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))

    provenance = controller.provenance()

    assert provenance["numeric_path"] == ("quantized_ue5m3_fp4_probe_matched_k64_issue_rz")
    assert provenance["gemm_output_model"] == (
        "encoded_operand_k64_issue_rz_bf16_gemm_final_snap_1_over_1024"
    )
