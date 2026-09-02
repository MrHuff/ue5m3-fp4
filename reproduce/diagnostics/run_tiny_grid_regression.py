#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the 81,920-parameter teacher/student final-grid ablation.

The public UE5M3 quantizer and K64 issue-RZ GEMM are used for forward, data
gradient, and weight gradient. Only the optional output-grid denominator
changes between quantized trajectories.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ue5m3_fp4.backends.triton import (
    fake_quantize_gemm_operands,
    issue_rz_bf16_gemm,
    triton_available,
)
from ue5m3_fp4.formats import RoundingMode
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState

try:
    from .common import make_payload, write_json
except ImportError:  # Direct ``python reproduce/diagnostics/...py`` execution.
    from common import make_payload, write_json


def _parse_denominators(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected unique non-negative integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("denominators must be unique")
    return values


class _DiagnosticLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        owner: DiagnosticLinear,
    ) -> torch.Tensor:
        x = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        weight_for_gemm = weight.transpose(0, 1).contiguous()
        x_amax = owner.scale_state.reference("activation", owner.module_name, x)
        weight_amax = owner.scale_state.reference("weight", owner.module_name, weight)
        output = owner.gemm(
            x,
            weight_for_gemm,
            amax_a=x_amax,
            amax_b=weight_amax,
            rounding_a=RoundingMode.TIES_TO_EVEN,
            rounding_b=RoundingMode.TIES_TO_EVEN,
            two_dimensional_b=True,
        )
        ctx.owner = owner
        ctx.x_amax = x_amax
        ctx.weight_amax = weight_amax
        ctx.save_for_backward(inputs, weight)
        return output.reshape(*inputs.shape[:-1], weight.shape[0]).to(torch.bfloat16)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        owner: DiagnosticLinear = ctx.owner
        inputs, weight = ctx.saved_tensors
        dy = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        x = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        dy_amax = owner.scale_state.reference("upstream_gradient", owner.module_name, dy)
        gradient_rounding = owner.gradient_rounding
        grad_input = owner.gemm(
            dy,
            weight.contiguous(),
            amax_a=dy_amax,
            amax_b=ctx.weight_amax,
            rounding_a=gradient_rounding,
            rounding_b=RoundingMode.TIES_TO_EVEN,
            two_dimensional_b=True,
        )
        grad_weight = owner.gemm(
            dy.transpose(0, 1).contiguous(),
            x.contiguous(),
            amax_a=dy_amax,
            amax_b=ctx.x_amax,
            rounding_a=gradient_rounding,
            rounding_b=RoundingMode.TIES_TO_EVEN,
            two_dimensional_b=False,
        )
        return (
            grad_input.reshape_as(inputs).to(inputs.dtype),
            grad_weight.to(weight.dtype),
            None,
        )


class DiagnosticLinear(nn.Module):
    """Bias-free linear exposing the report's optional final-grid control."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        denominator: int,
        gradient_rounding: RoundingMode,
        scale_state: TrainingScaleState,
        generator: torch.Generator,
        module_name: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32, device=device)
        )
        self.denominator = denominator
        self.gradient_rounding = gradient_rounding
        self.scale_state = scale_state
        self.generator = generator
        self.module_name = module_name

    def gemm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        amax_a: torch.Tensor,
        amax_b: torch.Tensor,
        rounding_a: RoundingMode,
        rounding_b: RoundingMode,
        two_dimensional_b: bool,
    ) -> torch.Tensor:
        encoded_a, encoded_b = fake_quantize_gemm_operands(
            a,
            b,
            block_size=16,
            scale_target_a=448.0,
            scale_target_b=448.0,
            tensor_amax_a=amax_a,
            tensor_amax_b=amax_b,
            rounding_a=rounding_a,
            rounding_b=rounding_b,
            generator=self.generator,
            two_dimensional_b=two_dimensional_b,
            output_domain="encoded",
            output_dtype=torch.bfloat16,
        )
        encoded_output = issue_rz_bf16_gemm(
            encoded_a,
            encoded_b,
            snap_to_1_over_1024=False,
        )
        if self.denominator:
            encoded_output = torch.round(encoded_output * self.denominator) / self.denominator
        alpha = (amax_a.float() * amax_b.float()) / ((448.0 * 6.0) ** 2)
        return encoded_output * alpha

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _DiagnosticLinearFunction.apply(inputs, self.weight, self)


class QuantizedStudent(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        denominator: int,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        recipe = UE5M3Recipe.proposed()
        self.scale_state = TrainingScaleState(recipe)
        generator = torch.Generator(device=device).manual_seed(seed + 991)
        rounding = (
            RoundingMode.TIES_TO_EVEN
            if config["gradient_rounding"] == "ties_to_even"
            else RoundingMode.STOCHASTIC_8BIT_MIDPOINT
        )
        self.fc1 = DiagnosticLinear(
            config["input_dim"],
            config["hidden_dim"],
            denominator=denominator,
            gradient_rounding=rounding,
            scale_state=self.scale_state,
            generator=generator,
            module_name="diagnostic.fc1",
            device=device,
        )
        self.fc2 = DiagnosticLinear(
            config["hidden_dim"],
            config["output_dim"],
            denominator=denominator,
            gradient_rounding=rounding,
            scale_state=self.scale_state,
            generator=generator,
            module_name="diagnostic.fc2",
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(inputs).float()).to(torch.bfloat16)).float()


class BF16Student(nn.Module):
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config["input_dim"], config["hidden_dim"], bias=False, device=device
        )
        self.fc2 = nn.Linear(
            config["hidden_dim"], config["output_dim"], bias=False, device=device
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.fc2(F.silu(self.fc1(inputs))).float()


def _make_problem(config: dict[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    weight1 = torch.randn(
        (config["hidden_dim"], config["input_dim"]),
        generator=generator,
        device=device,
    ) / math.sqrt(config["input_dim"])
    weight2 = torch.randn(
        (config["output_dim"], config["hidden_dim"]),
        generator=generator,
        device=device,
    ) / math.sqrt(config["hidden_dim"])
    train_x = torch.randn(
        (config["train_examples"], config["input_dim"]),
        generator=generator,
        device=device,
    ).to(torch.bfloat16)
    eval_x = torch.randn(
        (config["eval_examples"], config["input_dim"]),
        generator=generator,
        device=device,
    ).to(torch.bfloat16)
    with torch.no_grad():
        train_y = F.linear(F.silu(F.linear(train_x.float(), weight1)), weight2)
        eval_y = F.linear(F.silu(F.linear(eval_x.float(), weight1)), weight2)
    initial1 = weight1 + config["initial_noise"] * torch.randn(
        weight1.shape, generator=generator, device=device
    ) / math.sqrt(config["input_dim"])
    initial2 = weight2 + config["initial_noise"] * torch.randn(
        weight2.shape, generator=generator, device=device
    ) / math.sqrt(config["hidden_dim"])
    batches = torch.randint(
        config["train_examples"],
        (config["steps"], config["batch_size"]),
        generator=generator,
        device=device,
    )
    return {
        "initial": (initial1, initial2),
        "train": (train_x, train_y),
        "eval": (eval_x, eval_y),
        "batches": batches,
    }


def _copy_initial(model: nn.Module, problem: dict[str, Any]) -> None:
    with torch.no_grad():
        model.fc1.weight.copy_(problem["initial"][0])
        model.fc2.weight.copy_(problem["initial"][1])


@torch.no_grad()
def _evaluate(model: nn.Module, data: tuple[torch.Tensor, torch.Tensor], batch: int) -> float:
    inputs, targets = data
    losses = [
        F.mse_loss(model(inputs[start : start + batch]), targets[start : start + batch])
        for start in range(0, inputs.shape[0], batch)
    ]
    return float(torch.stack(losses).mean())


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_difference(
    candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, Any]:
    differing = 0
    total = 0
    maximum = 0.0
    for name, value in candidate.items():
        other = reference[name]
        differing += int(torch.sum(value != other))
        total += value.numel()
        maximum = max(maximum, float((value.float() - other.float()).abs().max()))
    return {
        "bit_exact": differing == 0,
        "differing_elements": differing,
        "total_elements": total,
        "maximum_absolute_difference": maximum,
    }


def _run_one(
    config: dict[str, Any],
    problem: dict[str, Any],
    *,
    seed: int,
    denominator: int | None,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if denominator is None:
        model: nn.Module = BF16Student(config, device)
        scale_state = None
        label = "bf16"
    else:
        model = QuantizedStudent(config, denominator=denominator, seed=seed, device=device)
        scale_state = model.scale_state
        label = "no_grid" if denominator == 0 else f"1/{denominator}"
    _copy_initial(model, problem)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.0
    )
    if scale_state is not None:
        scale_state.begin_step(1)
    initial_eval = _evaluate(model, problem["eval"], config["batch_size"])
    if scale_state is not None:
        scale_state.clear_runtime()
    train_x, train_y = problem["train"]
    history: list[dict[str, float | int]] = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, config["steps"] + 1):
        if scale_state is not None:
            scale_state.begin_step(step)
        indices = problem["batches"][step - 1]
        prediction = model(train_x[indices])
        loss = F.mse_loss(prediction, train_y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss for {label} at step {step}")
        if (step - 1) % config["history_stride"] == 0 or step == config["steps"]:
            history.append({"step": step, "loss": float(loss.detach())})
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if scale_state is not None:
        scale_state.begin_step(config["steps"] + 1)
    final_eval = _evaluate(model, problem["eval"], config["batch_size"])
    return (
        {
            "seed": seed,
            "denominator": denominator,
            "label": label,
            "initial_evaluation_mse": initial_eval,
            "final_evaluation_mse": final_eval,
            "elapsed_seconds": elapsed,
            "history": history,
            "scale_state": scale_state.report() if scale_state is not None else None,
        },
        _state(model),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument(
        "--denominators", type=_parse_denominators, default="0,256,512,1024,2048,4096"
    )
    parser.add_argument(
        "--gradient-rounding", choices=("ties_to_even", "stochastic"), default="ties_to_even"
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    denominators = (
        args.denominators
        if isinstance(args.denominators, list)
        else _parse_denominators(args.denominators)
    )
    config: dict[str, Any] = {
        "device": args.device,
        "seeds": args.seeds,
        "denominators": denominators,
        "gradient_rounding": args.gradient_rounding,
        "steps": args.steps,
        "train_examples": 2048,
        "eval_examples": 1024,
        "batch_size": 128,
        "input_dim": 256,
        "hidden_dim": 256,
        "output_dim": 64,
        "parameter_count": 81_920,
        "initial_noise": 0.25,
        "learning_rate": 0.002,
        "history_stride": 5,
        "delayed_scale_interval": 50,
        "block_size": 16,
        "scale_target": 448.0,
    }
    if args.quick:
        config.update(
            {
                "seeds": [args.seeds[0]],
                "denominators": denominators[:2],
                "steps": min(args.steps, 2),
                "train_examples": 128,
                "eval_examples": 128,
                "history_stride": 1,
            }
        )
    if not config["seeds"] or config["steps"] <= 0:
        raise ValueError("at least one seed and a positive step count are required")
    device = torch.device(config["device"])
    if device.type != "cuda" or not triton_available():
        raise RuntimeError("the grid regression requires CUDA and Triton")
    results: list[dict[str, Any]] = []
    all_differences: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        problem = _make_problem(config, seed, device)
        bf16, _ = _run_one(config, problem, seed=seed, denominator=None, device=device)
        results.append(bf16)
        quantized: dict[int, tuple[dict[str, Any], dict[str, torch.Tensor]]] = {}
        for denominator in config["denominators"]:
            quantized[denominator] = _run_one(
                config,
                problem,
                seed=seed,
                denominator=denominator,
                device=device,
            )
            results.append(quantized[denominator][0])
        if 0 in quantized:
            reference_state = quantized[0][1]
            for denominator, (record, state) in quantized.items():
                difference = _state_difference(state, reference_state)
                record["state_difference_from_no_grid"] = difference
                all_differences.append(difference)
    write_json(
        args.output_json,
        make_payload(
            experiment="tiny_81920_parameter_final_grid_regression",
            config=config,
            results={
                "all_parameter_states_bit_exact_to_no_grid": (
                    bool(all_differences) and all(item["bit_exact"] for item in all_differences)
                ),
                "runs": results,
            },
            device=device,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
