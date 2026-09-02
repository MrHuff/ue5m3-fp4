#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a synthetic full-model construction or forward/backward smoke.

This is an integration check, not a quality or throughput benchmark.  It uses
random token IDs, performs no optimizer update, and never reads a checkpoint.
The quantized choices exercise the same converters and numerical paths used by
the public TorchTitan configurations.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ue5m3_fp4.integrations.torchtitan.comparators import (
    NativeNVFP4Variant,
    convert_native_nvfp4_nemotron_h,
    convert_ue5m3_te_settings_nemotron_h,
)
from ue5m3_fp4.integrations.torchtitan.nemotron_h import (
    NemotronH8BArgs,
    NemotronHForTorchTitan,
)
from ue5m3_fp4.integrations.torchtitan.selection import convert_reported_nemotron_h
from ue5m3_fp4.integrations.torchtitan.trainer import (
    begin_training_step,
    training_scale_states,
)
from ue5m3_fp4.nn.linear import LinearBackend
from ue5m3_fp4.recipe import UE5M3Recipe

NUMERIC_PATHS = (
    "bf16",
    "ue5m3-proposed-b16",
    "ue5m3-proposed-b32",
    "ue5m3-torch-control",
    "ue5m3-te-settings",
    "native-nvfp4-te",
    "native-nvfp4-no-rht-all",
)


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _convert(model: torch.nn.Module, numeric_path: str) -> dict[str, Any]:
    if numeric_path == "bf16":
        return {
            "numeric_path": numeric_path,
            "fp4_quantization_applied": False,
            "fp4_linears": 0,
        }
    if numeric_path in {"ue5m3-proposed-b16", "ue5m3-proposed-b32"}:
        block_size = 32 if numeric_path.endswith("b32") else 16
        recipe = replace(
            UE5M3Recipe.proposed(),
            name=f"proposed_ue5m3_b{block_size}_d50",
            block_size=block_size,
        )
        conversion = convert_reported_nemotron_h(model, recipe=recipe)
        return {
            "numeric_path": numeric_path,
            "fp4_quantization_applied": True,
            "fp4_linears": len(conversion.fp4_linears),
            "backend": conversion.linear_backend,
            "recipe": recipe.to_dict(),
        }
    if numeric_path == "ue5m3-torch-control":
        conversion = convert_reported_nemotron_h(
            model,
            backend=LinearBackend.TRITON_QUANT_TORCH,
        )
        return {
            "numeric_path": numeric_path,
            "fp4_quantization_applied": True,
            "fp4_linears": len(conversion.fp4_linears),
            "backend": conversion.linear_backend,
            "recipe": conversion.scale_state.recipe.to_dict(),
        }
    if numeric_path == "ue5m3-te-settings":
        conversion = convert_ue5m3_te_settings_nemotron_h(model)
        return {
            "numeric_path": numeric_path,
            "fp4_quantization_applied": True,
            "fp4_linears": len(conversion.fp4_linears),
            "bf16_final_linears": conversion.final_bf16_linears,
            "backend": conversion.linear_backend,
            "recipe": conversion.scale_state.recipe.to_dict(),
        }
    variant = (
        NativeNVFP4Variant.TRANSFORMER_ENGINE_RECIPE
        if numeric_path == "native-nvfp4-te"
        else NativeNVFP4Variant.NO_RHT_ALL_LINEARS
    )
    conversion = convert_native_nvfp4_nemotron_h(model, variant=variant)
    return {
        "numeric_path": numeric_path,
        "fp4_quantization_applied": True,
        "fp4_linears": len(conversion.fp4_linears),
        "bf16_final_linears": conversion.final_bf16_linears,
        "backend": conversion.native_backend,
        "transformer_engine_version": conversion.transformer_engine_version,
    }


def _architecture(model: NemotronHForTorchTitan) -> dict[str, Any]:
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "layer_count": len(model.layers),
        "attention_mixer_count": len(model.sdpa_configuration["attention_mixers"]),
        "state_dict_roots": sorted({name.split(".", 1)[0] for name in model.state_dict()}),
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--numeric-path", choices=NUMERIC_PATHS, default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--meta-only", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.sequence_length <= 1:
        raise ValueError("batch size must be positive and sequence length must exceed one")
    if args.sequence_length > 8192:
        raise ValueError("the reported model supports at most 8,192 tokens")
    if (
        args.numeric_path != "bf16"
        and not args.meta_only
        and not args.forward_only
        and (args.batch_size * args.sequence_length) % 64
    ):
        raise ValueError(
            "quantized backward requires batch_size * sequence_length divisible by 64"
        )
    assets = args.assets.expanduser().resolve()
    if not assets.is_dir():
        raise FileNotFoundError(f"asset directory does not exist: {assets}")

    torch.manual_seed(args.seed)
    started = time.perf_counter()
    with torch.device("meta"), _default_dtype(torch.bfloat16):
        model = NemotronHForTorchTitan(NemotronH8BArgs(hf_assets_path=str(assets)))
        conversion = _convert(model, args.numeric_path)
    architecture = _architecture(model)
    if architecture != {
        "parameter_count": 8_084_075_520,
        "layer_count": 52,
        "attention_mixer_count": 4,
        "state_dict_roots": ["layers", "norm", "output", "tok_embeddings"],
    }:
        raise RuntimeError(f"constructed architecture differs from the paper: {architecture}")

    result: dict[str, Any] = {
        "schema": "ue5m3_fp4_nemotron_h_smoke_v1",
        "scope": "synthetic integration smoke; not quality or throughput evidence",
        "seed": args.seed,
        "numeric_path": args.numeric_path,
        "architecture": architecture,
        "sdpa_configuration": model.sdpa_configuration,
        "conversion": conversion,
        "meta_only": args.meta_only,
        "forward_only": args.forward_only,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": _version("transformers"),
            "mamba_ssm": _version("mamba-ssm"),
            "causal_conv1d": _version("causal-conv1d"),
        },
    }
    if not args.meta_only:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the full-model smoke requires a CUDA device")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        model.to_empty(device=device)
        model.init_weights(buffer_device=device)
        model.train(not args.forward_only)
        begun_states = begin_training_step([model], 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        tokens = torch.randint(
            0,
            131_072,
            (args.batch_size, args.sequence_length),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        loss = model(tokens, labels=tokens)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("full-model smoke produced a non-finite loss")
        if not args.forward_only:
            loss.backward()
        torch.cuda.synchronize(device)
        gradients = tuple(
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        )
        finite_gradients = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        if not args.forward_only and not finite_gradients:
            raise RuntimeError("full-model smoke produced a non-finite gradient")
        scale_reports = [state.report() for state in training_scale_states([model])]
        result["measurement"] = {
            "loss": float(loss.detach()),
            "begun_training_scale_states": begun_states,
            "gradient_parameter_elements": sum(gradient.numel() for gradient in gradients),
            "all_gradients_finite": finite_gradients,
            "training_scale_state_count": len(scale_reports),
            "training_scale_entry_count": sum(
                len(report["entries"]) for report in scale_reports
            ),
            "training_scale_refresh_count": sum(
                entry["refreshes"] for report in scale_reports for entry in report["entries"]
            ),
            "training_scale_reuse_count": sum(
                entry["reuses"] for report in scale_reports for entry in report["entries"]
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "device_name": torch.cuda.get_device_name(device),
        }
    result["elapsed_seconds"] = time.perf_counter() - started

    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output is not None:
        _write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
