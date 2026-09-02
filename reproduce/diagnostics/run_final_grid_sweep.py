#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep final-grid granularity on paired near-cancellation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .common import make_payload, write_json
    from .numerics import add_rz_f32, comparison_metrics, snap_rne
except ImportError:  # Direct ``python reproduce/diagnostics/...py`` execution.
    from common import make_payload, write_json
    from numerics import add_rz_f32, comparison_metrics, snap_rne


def _parse_denominators(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique non-negative denominators")
    return result


def _accumulate_rz(partials: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference = partials.double().sum(dim=1)
    estimate = torch.zeros(partials.shape[0], dtype=torch.float32, device=partials.device)
    for partial in partials.float().unbind(dim=1):
        estimate = add_rz_f32(estimate, partial)
    return reference, estimate


def run_sweep(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    if config["issues"] % 2:
        raise ValueError("issues must be even")
    half = config["issues"] // 2
    records: list[dict[str, Any]] = []
    for scale_index, partial_scale in enumerate(config["partial_scales"]):
        generator = torch.Generator(device=device).manual_seed(
            config["seed"] + scale_index * 1009
        )
        values = (
            torch.randn((config["trials"], half), generator=generator, device=device).float()
            * partial_scale
        )
        base = torch.cat((values, -values), dim=1)
        order = torch.rand(base.shape, generator=generator, device=device).argsort(dim=1)
        base = torch.gather(base, dim=1, index=order)
        residual_unit = torch.randn(
            (config["trials"],), generator=generator, device=device
        ).float()
        for sigma in config["residual_sigmas"]:
            partials = base.clone()
            partials[:, -1] += residual_unit * sigma
            reference, rz = _accumulate_rz(partials)
            for denominator in config["denominators"]:
                estimate = rz if denominator == 0 else snap_rne(rz, denominator)
                records.append(
                    {
                        "partial_scale": partial_scale,
                        "residual_standard_deviation": sigma,
                        "denominator": denominator,
                        "quantum": None if denominator == 0 else 1.0 / denominator,
                        "dead_zone_half_width": (
                            None if denominator == 0 else 1.0 / (2.0 * denominator)
                        ),
                        "zero_rate": float((estimate == 0).double().mean().cpu()),
                        "error": comparison_metrics(estimate, reference),
                    }
                )
    return records


def _archived_native_matches(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    study = payload["studies"]["final_grid_native_witnesses"]
    return {
        "evidence_class": "archived_report_evidence",
        "summary_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_artifact_sha256": study["source_artifact_sha256"],
        "trials": study["trials"],
        "matches_by_denominator": study["matches_by_denominator"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--denominators", type=_parse_denominators, default="0,256,512,1024,2048,4096"
    )
    parser.add_argument("--trials", type=int, default=4096)
    parser.add_argument("--issues", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--archived-summary",
        type=Path,
        default=Path("reproduce/diagnostics/archived/report_summary.json"),
    )
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
        "denominators": denominators,
        "trials": args.trials,
        "issues": args.issues,
        "partial_scales": [1.0, 32.0],
        "residual_sigmas": [0.0, 1 / 8192, 1 / 4096, 1 / 2048, 1 / 1024, 1 / 512],
        "seed": args.seed,
    }
    if args.quick:
        config.update(
            {
                "trials": 64,
                "partial_scales": [1.0],
                "residual_sigmas": [0.0, 1 / 1024],
            }
        )
    if config["trials"] <= 0 or config["issues"] <= 0:
        raise ValueError("trials and issues must be positive")
    archived = _archived_native_matches(args.archived_summary)
    missing = [
        str(item)
        for item in denominators
        if str(item) not in archived["matches_by_denominator"]
    ]
    if missing:
        raise ValueError(f"archived witness summary lacks denominators: {missing}")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    write_json(
        args.output_json,
        make_payload(
            experiment="final_grid_granularity",
            config=config,
            results={
                "archived_native_witnesses": archived,
                "rerunnable_near_cancellation": run_sweep(config, device),
                "interpretation_guardrail": (
                    "The historical corpus rejects coarser grids but does not distinguish "
                    "1/1024 from finer grids or no final snap. The synthetic cancellation "
                    "rerun measures how each grid treats small residuals."
                ),
            },
            device=device,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
