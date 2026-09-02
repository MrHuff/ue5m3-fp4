#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the report's storage-neutral RTZ/grid statistical sweep.

The experiment changes only cross-K64 accumulation and the optional final
grid. It does not infer or claim a hardware design rationale.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

try:
    from .common import make_payload, write_json
    from .numerics import (
        VARIANTS,
        accumulate_issue_partials,
        accumulate_issue_terms,
        comparison_metrics,
    )
except ImportError:  # Direct ``python reproduce/diagnostics/...py`` execution.
    from common import make_payload, write_json
    from numerics import (
        VARIANTS,
        accumulate_issue_partials,
        accumulate_issue_terms,
        comparison_metrics,
    )


_DISTRIBUTIONS = (
    "gaussian",
    "gaussian_sigma32",
    "laplace",
    "student_t3",
    "contaminated_gaussian",
)
_MODES = ("bf16", "fp4_like_nvfp4")


def _sample(
    name: str,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if name == "gaussian":
        return torch.randn(shape, generator=generator, device=device)
    if name == "gaussian_sigma32":
        return 32.0 * torch.randn(shape, generator=generator, device=device)
    if name == "laplace":
        positive = torch.empty(shape, device=device).exponential_(generator=generator)
        negative = torch.empty(shape, device=device).exponential_(generator=generator)
        return (positive - negative) / math.sqrt(2.0)
    if name == "student_t3":
        normal = torch.randn(shape, generator=generator, device=device)
        gamma_shape = torch.full(shape, 1.5, device=device)
        chi_square = torch._standard_gamma(gamma_shape, generator=generator) * 2.0
        return (normal / torch.sqrt(chi_square / 3.0)) / math.sqrt(3.0)
    if name == "contaminated_gaussian":
        values = torch.randn(shape, generator=generator, device=device)
        outliers = torch.rand(shape, generator=generator, device=device) < 0.01
        values = torch.where(outliers, values * 25.0, values)
        return values / math.sqrt(0.99 + 0.01 * 25.0**2)
    raise ValueError(f"unknown distribution: {name!r}")


def _nvfp4_like_dequantize(value: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Apply the simple E2M1/E4M3 block quantizer used by this diagnostic.

    This deliberately is not the UE5M3 training quantizer: the accumulator
    investigation was motivated by native NVFP4, whose local scales are E4M3.
    """

    if value.shape[-1] % block_size:
        raise ValueError("the last dimension must be divisible by block_size")
    blocks = value.float().reshape(*value.shape[:-1], -1, block_size)
    scale = (blocks.abs().amax(dim=-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn).float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = (blocks / scale).abs().clamp_max(6.0)
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=value.device,
    )
    nearest = (normalized[..., None] - levels).abs().argmin(dim=-1)
    return (levels[nearest] * torch.sign(blocks) * scale).reshape_as(value).to(torch.bfloat16)


def _prepare(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "bf16":
        return value.to(torch.bfloat16)
    if mode == "fp4_like_nvfp4":
        return _nvfp4_like_dequantize(value)
    raise ValueError(f"unknown operand mode: {mode!r}")


def _moments(value: torch.Tensor) -> dict[str, float]:
    data = value.detach().double().cpu().flatten()
    mean = data.mean()
    centered = data - mean
    std = torch.sqrt(torch.mean(centered.square()))
    normalized = centered / max(float(std), torch.finfo(torch.float64).tiny)
    return {
        "mean": float(mean),
        "standard_deviation": float(std),
        "skewness": float(torch.mean(normalized**3)),
        "excess_kurtosis": float(torch.mean(normalized**4) - 3.0),
    }


def run_dot_sweep(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    case_index = 0
    for distribution in config["distributions"]:
        for correlation in config["correlations"]:
            for mode in config["operand_modes"]:
                generator = torch.Generator(device=device)
                generator.manual_seed(config["seed"] + case_index * 100_003)
                case_index += 1
                gathered: dict[str, list[torch.Tensor]] = {
                    key: []
                    for key in ("full_precision", "reference", "issue_reference", *VARIANTS)
                }
                for start in range(0, config["trials"], config["chunk_size"]):
                    rows = min(config["chunk_size"], config["trials"] - start)
                    x = _sample(
                        distribution,
                        (rows, config["k"]),
                        generator=generator,
                        device=device,
                    )
                    independent = _sample(
                        distribution,
                        (rows, config["k"]),
                        generator=generator,
                        device=device,
                    )
                    w = correlation * x + math.sqrt(1.0 - correlation**2) * independent
                    gathered["full_precision"].append(
                        (x.double() * w.double()).sum(dim=1).cpu()
                    )
                    xq, wq = _prepare(x, mode), _prepare(w, mode)
                    accumulated = accumulate_issue_terms(
                        xq.float() * wq.float(),
                        issue_size=config["issue_size"],
                        grid_denominator=config["grid_denominator"],
                    )
                    for key, value in accumulated.items():
                        gathered[key].append(value.cpu())
                merged = {key: torch.cat(parts) for key, parts in gathered.items()}
                records.append(
                    {
                        "distribution": distribution,
                        "correlation": correlation,
                        "operand_mode": mode,
                        "reference_moments": _moments(merged["reference"]),
                        "accumulator_error": {
                            key: comparison_metrics(merged[key], merged["reference"])
                            for key in VARIANTS
                        },
                        "cross_issue_error": {
                            key: comparison_metrics(merged[key], merged["issue_reference"])
                            for key in VARIANTS
                        },
                        "end_to_end_error": {
                            key: comparison_metrics(merged[key], merged["full_precision"])
                            for key in VARIANTS
                        },
                        "quantization_only": comparison_metrics(
                            merged["reference"], merged["full_precision"]
                        ),
                    }
                )
    return records


def run_regression_probe(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dist_index, distribution in enumerate(config["regression_distributions"]):
        for mode_index, mode in enumerate(config["operand_modes"]):
            generator = torch.Generator(device=device)
            generator.manual_seed(config["seed"] + 9_000_001 + dist_index * 101 + mode_index)
            shape = (config["regression_batch"], config["regression_features"])
            x = _sample(distribution, shape, generator=generator, device=device)
            beta_shape = (1, config["regression_features"])
            true_beta = torch.randn(beta_shape, generator=generator, device=device) / math.sqrt(
                config["regression_features"]
            )
            beta = true_beta + 0.25 * torch.randn(
                beta_shape, generator=generator, device=device
            ) / math.sqrt(config["regression_features"])
            target = (x.double() * true_beta.double()).sum(dim=1)
            full_prediction = (x.double() * beta.double()).sum(dim=1)
            full_residual = full_prediction - target
            full_gradient = (x.double().T @ full_residual[:, None]).squeeze(1) / config[
                "regression_batch"
            ]
            xq, betaq = _prepare(x, mode), _prepare(beta, mode)
            forward = accumulate_issue_terms(
                xq.float() * betaq.float(),
                issue_size=config["issue_size"],
                grid_denominator=config["grid_denominator"],
            )
            quantized_residual = forward["reference"] - target
            quantized_gradient = (xq.double().T @ quantized_residual[:, None]).squeeze(
                1
            ) / config["regression_batch"]
            variants: dict[str, Any] = {}
            for variant in VARIANTS:
                residual = forward[variant].double() - target
                gradient_terms = xq.float().T * residual.float()[None, :]
                gradient = (
                    accumulate_issue_terms(
                        gradient_terms,
                        issue_size=config["issue_size"],
                        grid_denominator=config["grid_denominator"],
                    )[variant]
                    / config["regression_batch"]
                )
                variants[variant] = {
                    "mean_squared_error": float(torch.mean(residual.square())),
                    "forward_accumulator_error": comparison_metrics(
                        forward[variant], forward["reference"]
                    ),
                    "gradient_accumulator_error": comparison_metrics(
                        gradient, quantized_gradient
                    ),
                    "end_to_end_gradient_error": comparison_metrics(gradient, full_gradient),
                }
            records.append(
                {
                    "distribution": distribution,
                    "operand_mode": mode,
                    "full_precision_mean_squared_error": float(
                        torch.mean(full_residual.square())
                    ),
                    "quantized_exact_accumulation_mean_squared_error": float(
                        torch.mean(quantized_residual.square())
                    ),
                    "variants": variants,
                }
            )
    return records


def run_cancellation_sweep(
    config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    if config["cancellation_issues"] % 2:
        raise ValueError("cancellation_issues must be even")
    records: list[dict[str, Any]] = []
    case_index = 0
    half = config["cancellation_issues"] // 2
    for distribution in config["cancellation_distributions"]:
        for partial_scale in config["cancellation_partial_scales"]:
            for residual_sigma in config["cancellation_residual_sigmas"]:
                generator = torch.Generator(device=device)
                generator.manual_seed(config["seed"] + 19_000_001 + case_index * 1009)
                case_index += 1
                values = (
                    _sample(
                        distribution,
                        (config["cancellation_trials"], half),
                        generator=generator,
                        device=device,
                    ).float()
                    * partial_scale
                )
                partials = torch.cat((values, -values), dim=1)
                order = torch.rand(partials.shape, generator=generator, device=device).argsort(
                    dim=1
                )
                partials = torch.gather(partials, dim=1, index=order)
                residual = (
                    torch.randn(
                        (config["cancellation_trials"],),
                        generator=generator,
                        device=device,
                    ).float()
                    * residual_sigma
                )
                partials[:, -1] += residual
                accumulated = accumulate_issue_partials(
                    partials, grid_denominator=config["grid_denominator"]
                )
                reference = accumulated["reference"]
                records.append(
                    {
                        "distribution": distribution,
                        "partial_scale": partial_scale,
                        "residual_standard_deviation": residual_sigma,
                        "reference_moments": _moments(reference),
                        "variants": {
                            key: {
                                **comparison_metrics(accumulated[key], reference),
                                "zero_rate": float(
                                    (accumulated[key] == 0).double().mean().cpu()
                                ),
                            }
                            for key in VARIANTS
                        },
                    }
                )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--trials", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--issue-size", type=int, default=64)
    parser.add_argument("--grid-denominator", type=int, default=1024)
    parser.add_argument("--regression-batch", type=int, default=2048)
    parser.add_argument("--regression-features", type=int, default=2048)
    parser.add_argument("--cancellation-trials", type=int, default=4096)
    parser.add_argument("--cancellation-issues", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "device": args.device,
        "seed": args.seed,
        "trials": args.trials,
        "k": args.k,
        "chunk_size": args.chunk_size,
        "issue_size": args.issue_size,
        "grid_denominator": args.grid_denominator,
        "distributions": list(_DISTRIBUTIONS),
        "correlations": [0.0, 0.25],
        "operand_modes": list(_MODES),
        "regression_batch": args.regression_batch,
        "regression_features": args.regression_features,
        "regression_distributions": [
            "gaussian",
            "student_t3",
            "contaminated_gaussian",
        ],
        "cancellation_trials": args.cancellation_trials,
        "cancellation_issues": args.cancellation_issues,
        "cancellation_distributions": [
            "gaussian",
            "student_t3",
            "contaminated_gaussian",
        ],
        "cancellation_partial_scales": [1.0, 32.0],
        "cancellation_residual_sigmas": [0.0, 1 / 4096, 1 / 1024, 1 / 64],
    }
    if args.quick:
        config.update(
            {
                "trials": 32,
                "k": 128,
                "chunk_size": 16,
                "distributions": ["gaussian"],
                "correlations": [0.0],
                "operand_modes": ["bf16"],
                "regression_batch": 64,
                "regression_features": 64,
                "regression_distributions": ["gaussian"],
                "cancellation_trials": 64,
                "cancellation_distributions": ["gaussian"],
                "cancellation_partial_scales": [1.0],
                "cancellation_residual_sigmas": [0.0, 1 / 1024],
            }
        )
    for name in ("trials", "k", "chunk_size", "regression_batch", "regression_features"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["k"] % 16 or config["regression_features"] % 16:
        raise ValueError("k and regression_features must be divisible by 16")
    return config


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    results = {
        "interpretation_guardrail": (
            "This experiment measures consequences of the reconstructed rule; "
            "it does not establish NVIDIA's hardware design intent."
        ),
        "dot_products": run_dot_sweep(config, device),
        "linear_regression": run_regression_probe(config, device),
        "near_cancellation": run_cancellation_sweep(config, device),
    }
    write_json(
        args.output_json,
        make_payload(
            experiment="issue_rz_accumulator_statistics",
            config=config,
            results=results,
            device=device,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
