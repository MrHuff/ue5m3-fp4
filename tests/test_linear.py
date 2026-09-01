# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

import ue5m3_fp4.nn.linear as linear_module
from ue5m3_fp4.formats import RoundingMode
from ue5m3_fp4.nn.linear import UE5M3Linear
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.inference import FP4InferenceScalingController
from ue5m3_fp4.scaling.training import TrainingScaleState


def test_training_gemms_use_selective_dy_rounding_and_one_delayed_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = UE5M3Recipe.proposed()
    scale_state = TrainingScaleState(recipe)
    layer = UE5M3Linear(
        4,
        3,
        recipe=recipe,
        scale_state=scale_state,
        module_name="layers.44.mixer.down_proj",
    )
    scale_state.begin_step(1)
    calls: list[dict[str, Any]] = []

    def identity_quantizer(tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append({"tensor": tensor, **kwargs})
        return tensor

    monkeypatch.setattr(linear_module, "quantize_dequantize_blocks", identity_quantizer)
    inputs = torch.randn(2, 4, requires_grad=True)
    expected_weight = layer.weight.detach().clone()
    output = layer(inputs)
    output.sum().backward()

    assert len(calls) == 6
    assert [call["rounding"] for call in calls] == [
        RoundingMode.TIES_TO_EVEN,
        RoundingMode.TIES_TO_EVEN,
        RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
        RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
        RoundingMode.TIES_TO_EVEN,
        RoundingMode.TIES_TO_EVEN,
    ]
    assert [call["scale_target"] for call in calls] == [
        448.0,
        448.0,
        448.0,
        2_048.0,
        448.0,
        448.0,
    ]
    assert [call["two_dimensional"] for call in calls] == [
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    assert calls[2]["tensor_reference"] is calls[3]["tensor_reference"]
    assert calls[2]["tensor"].shape == (2, 3)
    assert calls[3]["tensor"].shape == (3, 2)
    torch.testing.assert_close(
        calls[5]["tensor"], inputs.detach().transpose(0, 1).contiguous()
    )
    torch.testing.assert_close(
        inputs.grad,
        torch.ones(2, 3) @ expected_weight,
    )
    torch.testing.assert_close(
        layer.weight.grad,
        torch.ones(3, 2) @ inputs.detach(),
    )


def test_training_scale_cache_is_not_serialized_with_parameters() -> None:
    recipe = UE5M3Recipe.proposed()
    scale_state = TrainingScaleState(recipe)
    layer = UE5M3Linear(4, 3, recipe=recipe, scale_state=scale_state)
    scale_state.begin_step(1)
    layer(torch.ones(1, 4))

    assert scale_state.state_dict() == {}
    assert set(layer.state_dict()) == {"weight", "bias"}


def test_real_linear_matches_shape_and_produces_finite_gradients() -> None:
    recipe = UE5M3Recipe.proposed()
    scale_state = TrainingScaleState(recipe)
    layer = UE5M3Linear(7, 5, recipe=recipe, scale_state=scale_state)
    scale_state.begin_step(1)
    inputs = torch.randn(2, 3, 7, requires_grad=True)

    output = layer(inputs)
    output.square().mean().backward()

    assert output.shape == (2, 3, 5)
    assert bool(torch.isfinite(output).all())
    assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
    assert layer.weight.grad is not None and bool(torch.isfinite(layer.weight.grad).all())


def test_real_linear_implements_current_tensor_inference_lifecycle() -> None:
    model = torch.nn.Sequential(UE5M3Linear(4, 3)).eval()
    controller = FP4InferenceScalingController(
        model,
        activation_mode="current_tensor",
        checkpoint_identity={"checkpoint_id": "synthetic-test-fixture"},
    )
    controller.reset_after_checkpoint_load()
    controller.calibrate_and_freeze_weights()
    controller.begin_measurement()

    with pytest.raises(RuntimeError, match="torch.no_grad"):
        model(torch.randn(2, 4))
    with torch.no_grad():
        output = model(torch.randn(2, 4))

    assert output.shape == (2, 3)
    provenance = controller.provenance()
    assert provenance["numeric_path"] == "quantized_ue5m3_fp4_decoded_torch"
    assert provenance["gemm_output_model"] == "decoded_operand_torch_matmul"
    assert provenance["native_hardware"] is False
    assert provenance["resolved_formats"][0]["torch_matmul_policy"][
        "device_type"
    ] == "cpu"
    assert provenance["fp4_quantization_applied"] is True
    assert provenance["inference_scaling_protocol"][
        "training_step_or_cache_inherited"
    ] is False


def test_inference_reset_discards_process_local_training_cache() -> None:
    recipe = UE5M3Recipe.proposed()
    scale_state = TrainingScaleState(recipe)
    model = torch.nn.Sequential(
        UE5M3Linear(4, 3, recipe=recipe, scale_state=scale_state)
    ).eval()
    scale_state.begin_step(1)
    scale_state.reference("activation", "0", torch.tensor([5.0]))
    controller = FP4InferenceScalingController(
        model,
        activation_mode="current_tensor",
        checkpoint_identity={"checkpoint_id": "synthetic-test-fixture"},
    )

    reset = controller.reset_after_checkpoint_load()

    assert reset["0"]["legacy_training_state_cleared"]["delayed_amax_entries"] == 1
    assert scale_state.step is None
    assert scale_state.report()["entries"] == []


def test_inference_rejects_stochastic_forward_rounding() -> None:
    recipe = replace(
        UE5M3Recipe.proposed(),
        activation_rounding=RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
    )
    model = torch.nn.Sequential(UE5M3Linear(4, 3, recipe=recipe)).eval()
    controller = FP4InferenceScalingController(
        model,
        activation_mode="current_tensor",
        checkpoint_identity={"checkpoint_id": "synthetic-test-fixture"},
    )

    with pytest.raises(ValueError, match="stochastic forward rounding"):
        controller.reset_after_checkpoint_load()


def test_d50_replay_rejects_a_different_recipe_interval() -> None:
    recipe = replace(UE5M3Recipe.proposed(), delayed_scale_interval=1)
    model = torch.nn.Sequential(UE5M3Linear(4, 3, recipe=recipe)).eval()
    controller = FP4InferenceScalingController(
        model,
        activation_mode="training_replay",
        checkpoint_identity={"checkpoint_id": "synthetic-test-fixture"},
        replay_work_unit={"kind": "fixed_forward_batch", "size": 1},
    )

    with pytest.raises(ValueError, match="recorded D=50"):
        controller.reset_after_checkpoint_load()


def test_inference_rejects_multiprocess_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Sequential(UE5M3Linear(4, 3)).eval()
    controller = FP4InferenceScalingController(
        model,
        activation_mode="current_tensor",
        checkpoint_identity={"checkpoint_id": "synthetic-test-fixture"},
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    with pytest.raises(RuntimeError, match="requires one process"):
        controller.reset_after_checkpoint_load()


def test_replay_only_measures_activation_amax_on_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Sequential(UE5M3Linear(4, 3)).eval()
    controller = FP4InferenceScalingController(
        model,
        activation_mode="training_replay",
        checkpoint_identity={"checkpoint_id": "synthetic-test-fixture"},
        replay_work_unit={"kind": "fixed_forward_batch", "size": 1},
    )
    controller.reset_after_checkpoint_load()
    controller.calibrate_and_freeze_weights()
    controller.begin_measurement(
        evaluation_order={"order": "synthetic ascending", "seed": None}
    )
    calls = 0
    original = linear_module._finite_amax

    def counted_amax(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(tensor)

    monkeypatch.setattr(linear_module, "_finite_amax", counted_amax)
    with torch.no_grad():
        for step in range(1, 52):
            inputs = torch.full((1, 4), float(step))
            controller.advance_training_replay_work_unit(
                inputs,
                effective_token_count=4,
            )
            model(inputs)

    assert calls == 2


def test_frozen_activation_measurement_does_not_rescan_amax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Sequential(UE5M3Linear(4, 3)).eval()
    controller = FP4InferenceScalingController(
        model,
        activation_mode="calibrated_frozen",
        checkpoint_identity={"checkpoint_id": "synthetic-test-fixture"},
    )
    controller.reset_after_checkpoint_load()
    controller.calibrate_and_freeze_weights()
    controller.begin_activation_calibration({"id": "synthetic-disjoint-calibration"})
    calibration = torch.ones(1, 4)
    controller.record_activation_calibration_batch(calibration)
    with torch.no_grad():
        model(calibration)
    controller.freeze_activation_scales()
    controller.begin_measurement()

    def unexpected_amax(_tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("frozen measurement must not scan activation amax")

    monkeypatch.setattr(linear_module, "_finite_amax", unexpected_amax)
    with torch.no_grad():
        output = model(torch.full((1, 4), 99.0))

    assert output.shape == (1, 3)
