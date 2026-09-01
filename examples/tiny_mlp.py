# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one synthetic training step and one post-load quantized forward."""

from __future__ import annotations

import argparse

import torch

from ue5m3_fp4.nn import convert_linear_modules, select_all_linears
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling import FP4InferenceScalingController, TrainingScaleState


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation-mode",
        choices=("current_tensor", "training_replay", "calibrated_frozen"),
        default="current_tensor",
        help="post-load activation-scale lifecycle to exercise",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    recipe = UE5M3Recipe.proposed()
    training_scales = TrainingScaleState(recipe)
    model = TinyMLP()
    convert_linear_modules(
        model,
        recipe=recipe,
        scale_state=training_scales,
        selector=select_all_linears,
    )

    training_scales.begin_step(1)
    inputs = torch.randn(4, 16)
    loss = model(inputs).square().mean()
    loss.backward()

    # A checkpoint contains learned parameters, not delayed-amax caches.
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inference_model = TinyMLP()
    inference_model.load_state_dict(checkpoint)
    convert_linear_modules(
        inference_model,
        recipe=recipe,
        selector=select_all_linears,
    )
    inference_model.eval()

    controller_kwargs: dict[str, object] = {
        "activation_mode": args.activation_mode,
        "checkpoint_identity": {"id": "in-memory-synthetic-fixture"},
    }
    if args.activation_mode == "training_replay":
        controller_kwargs["replay_work_unit"] = {
            "kind": "fixed_forward_batch",
            "size": int(inputs.shape[0]),
        }
    controller = FP4InferenceScalingController(inference_model, **controller_kwargs)
    controller.reset_after_checkpoint_load()
    controller.calibrate_and_freeze_weights()
    if args.activation_mode == "calibrated_frozen":
        calibration_inputs = torch.randn_like(inputs)
        controller.begin_activation_calibration(
            {"id": "disjoint-in-memory-synthetic-calibration"}
        )
        with torch.inference_mode():
            controller.record_activation_calibration_batch(calibration_inputs)
            inference_model(calibration_inputs)
        controller.freeze_activation_scales()
    evaluation_order = None
    if args.activation_mode == "training_replay":
        evaluation_order = {"order": "single synthetic batch", "seed": 7}
    controller.begin_measurement(evaluation_order=evaluation_order)
    with torch.inference_mode():
        if args.activation_mode == "training_replay":
            controller.advance_training_replay_work_unit(
                inputs,
                effective_token_count=inputs.numel(),
            )
        output = inference_model(inputs)

    provenance = controller.provenance()
    print(f"training loss: {loss.item():.6f}")
    print(f"activation mode: {args.activation_mode}")
    print(f"inference output shape: {tuple(output.shape)}")
    print(f"numeric path: {provenance['numeric_path']}")


if __name__ == "__main__":
    main()
