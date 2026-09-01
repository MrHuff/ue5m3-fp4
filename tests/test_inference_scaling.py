# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch

from ue5m3_fp4.scaling.inference import FP4InferenceScalingController


class FakeFP4Module(torch.nn.Module):
    """Small implementation of the public inference-scaling contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.bfloat16))
        self.generation = 0
        self.reset_fp4_inference_scaling()

    def reset_fp4_inference_scaling(self) -> dict[str, Any]:
        self.generation += 1
        self.active = False
        self.phase = "inactive"
        self.activation_mode: str | None = None
        self.activation_observations = 0
        self.activation_amax = 0.0
        self.logical_step = 0
        self.last_consumed_step = 0
        self.last_refresh_step = 0
        self.refresh_trace: list[dict[str, Any]] = []
        self.replay_cache: dict[str, float] = {}
        self.counters = {
            "resets": self.generation,
            "weight_calibrations": 0,
            "weight_frozen_resolutions": 0,
            "activation_current_resolutions": 0,
            "activation_calibration_resolutions": 0,
            "activation_frozen_resolutions": 0,
            "activation_replay_refreshes": 0,
            "activation_replay_reuses": 0,
        }
        return {"generation": self.generation, "legacy_training_state_cleared": {}}

    def configure_fp4_inference_scaling(self, activation_mode: str) -> None:
        self.active = True
        self.activation_mode = activation_mode
        self.phase = "configured"

    def calibrate_and_freeze_fp4_inference_weight_scale(self) -> dict[str, Any]:
        self.phase = "weights_frozen"
        self.counters["weight_calibrations"] += 1
        return {"key": "forward.w", "amax": 2.0, "device": "cpu"}

    def begin_fp4_inference_activation_calibration(self) -> None:
        self.phase = "activation_calibration"

    def freeze_fp4_inference_activation_scales(self) -> dict[str, Any]:
        self.phase = "activations_frozen"
        return {
            "forward.x": {
                "amax": self.activation_amax,
                "observations": self.activation_observations,
            }
        }

    def begin_fp4_inference_measurement(self) -> None:
        self.phase = "measuring"

    def advance_fp4_inference_training_replay_step(self, logical_step: int) -> None:
        self.logical_step = logical_step

    def reset_fp4_inference_measurement_counters(self) -> None:
        for name in tuple(self.counters):
            if name not in {"resets", "weight_calibrations"}:
                self.counters[name] = 0

    def fp4_inference_scaling_report(self) -> dict[str, Any]:
        return {
            "schema": "fake_fp4_scaling_v1",
            "active": self.active,
            "generation": self.generation,
            "phase": self.phase,
            "activation_mode": self.activation_mode,
            "frozen_global_amax": {
                "weights": {"forward.w": 2.0},
                "activations": (
                    {"forward.x": self.activation_amax}
                    if self.activation_mode == "calibrated_frozen"
                    else {}
                ),
            },
            "activation_calibration_counts": {
                "forward.x": self.activation_observations
            },
            "counters": dict(self.counters),
            "legacy_training_state_cleared": {},
            "rounding_modes": {
                "round_mode_activations": "TiesToEven",
                "round_mode_weights": "TiesToEven",
            },
            "training_replay": {
                "interval": 50,
                "logical_step": self.logical_step,
                "last_consumed_step": self.last_consumed_step,
                "cache": dict(self.replay_cache),
                "last_refresh_steps": {"forward.x": self.last_refresh_step},
                "refresh_trace": {"forward.x": list(self.refresh_trace)},
            },
            "format": {
                "payload": "E2M1",
                "block_size": 16,
                "scale_type": "E5M3",
                "scale_max_activations": 448.0,
                "scale_max_weights": 448.0,
                "gemm_output_model": "decoded_operand_torch_matmul",
                "torch_matmul_policy": {
                    "input_dtype": "float32",
                    "device_type": "cpu",
                    "torch_float32_matmul_precision": None,
                    "cuda_matmul_fp32_precision": None,
                    "cuda_matmul_allow_tf32": None,
                },
                "native_hardware": False,
                "encode_centric": False,
                "use_encoded_gemm": False,
            },
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        operand_amax = float(inputs.detach().abs().max().item())
        self.counters["weight_frozen_resolutions"] += 1
        if self.phase == "activation_calibration":
            self.activation_observations += 1
            self.activation_amax = max(self.activation_amax, operand_amax)
            self.counters["activation_calibration_resolutions"] += 1
        elif self.phase == "measuring" and self.activation_mode == "current_tensor":
            self.counters["activation_current_resolutions"] += 1
        elif self.phase == "measuring" and self.activation_mode == "calibrated_frozen":
            self.counters["activation_frozen_resolutions"] += 1
        elif self.phase == "measuring" and self.activation_mode == "training_replay":
            refresh = not self.replay_cache or self.logical_step - self.last_refresh_step >= 50
            if refresh:
                self.replay_cache["forward.x"] = operand_amax
                self.last_refresh_step = self.logical_step
                self.refresh_trace.append(
                    {"logical_step": self.logical_step, "amax": operand_amax}
                )
                self.counters["activation_replay_refreshes"] += 1
            else:
                self.counters["activation_replay_reuses"] += 1
            self.last_consumed_step = self.logical_step
        return inputs


class FakeFP4Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant = FakeFP4Module()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quant(inputs)


def make_controller(
    activation_mode: str,
) -> tuple[FakeFP4Model, FP4InferenceScalingController]:
    model = FakeFP4Model().eval()
    replay = (
        {"kind": "fixed_forward_batch", "size": 1}
        if activation_mode == "training_replay"
        else None
    )
    return model, FP4InferenceScalingController(
        model,
        activation_mode=activation_mode,
        checkpoint_identity={"metadata_sha256": "a" * 64},
        replay_work_unit=replay,
    )


def test_current_tensor_lifecycle() -> None:
    model, controller = make_controller("current_tensor")
    controller.reset_after_checkpoint_load()
    assert controller.calibrate_and_freeze_weights()["quant"]["amax"] == 2.0
    controller.begin_measurement()
    model(torch.tensor([[1.0, -3.0]]))
    controller.assert_ready_for_measurement()

    protocol = controller.provenance()["inference_scaling_protocol"]
    assert protocol["activation_policy"]["mode"] == "current_tensor"
    assert protocol["training_step_or_cache_inherited"] is False


def test_calibrated_frozen_lifecycle() -> None:
    model, controller = make_controller("calibrated_frozen")
    controller.reset_after_checkpoint_load()
    controller.calibrate_and_freeze_weights()
    controller.begin_activation_calibration({"manifest_sha256": "b" * 64})
    for batch in (torch.tensor([[1, 2]]), torch.tensor([[3, 4]])):
        controller.record_activation_calibration_batch(batch)
        model(batch)
    assert controller.freeze_activation_scales()["quant"]["forward.x"] == {
        "amax": 4.0,
        "observations": 2,
    }
    controller.begin_measurement()
    model(torch.tensor([[99, 100]]))
    assert controller.counters()["quant"]["activation_frozen_resolutions"] == 1


def test_cold_replay_refreshes_at_work_units_one_and_fifty_one() -> None:
    model, controller = make_controller("training_replay")
    controller.reset_after_checkpoint_load()
    controller.calibrate_and_freeze_weights()
    controller.begin_measurement(evaluation_order={"order": "fixed", "seed": None})

    for logical_step in range(1, 52):
        batch = torch.tensor([[float(logical_step)]])
        controller.advance_training_replay_work_unit(batch, effective_token_count=1)
        model(batch)

    assert model.quant.refresh_trace == [
        {"logical_step": 1, "amax": 1.0},
        {"logical_step": 51, "amax": 51.0},
    ]
    assert controller.counters()["quant"]["activation_replay_reuses"] == 49


def test_controller_rejects_measurement_before_setup() -> None:
    _, controller = make_controller("current_tensor")
    with pytest.raises(RuntimeError):
        controller.begin_measurement()


def test_controller_centrally_rejects_multiprocess_modules_without_own_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, controller = make_controller("current_tensor")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    with pytest.raises(RuntimeError, match="requires one process"):
        controller.reset_after_checkpoint_load()


def test_provenance_rejects_model_switched_back_to_training() -> None:
    model, controller = make_controller("current_tensor")
    controller.reset_after_checkpoint_load()
    controller.calibrate_and_freeze_weights()
    controller.begin_measurement()
    model(torch.ones(1, 2))
    model.train()

    with pytest.raises(RuntimeError, match="every module in eval mode"):
        controller.provenance()
