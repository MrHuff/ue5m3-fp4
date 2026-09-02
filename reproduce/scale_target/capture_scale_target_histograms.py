#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the late-layer value distributions used by the UE5M3 report.

The command accepts a local, standard Hugging Face safetensors checkpoint and
a local safetensors file containing ``tokens`` with shape ``[N, 8193]``.  It
runs exactly one selected sequence through a training-mode forward/backward
pass without an optimizer update.  Model parameters are frozen; gradients are
propagated to a detached embedding input so the upstream gradient at each
selected linear is preserved without allocating gradients for 8B parameters.

For ``backbone.layers.{45,47,49,51}.mixer.down_proj`` the output records
streaming log-histograms of the forward input ``X``, BF16 master weight ``W``,
and upstream gradient ``dY``.  It also forms the block-16 maxima for ``dY.T``
as used by the weight-gradient GEMM and diagnoses the raw/rounded UE5M3 scale
codes for targets 448 and 2,048.  Those diagnostics use each module's current
``dY`` amax; they are not a delayed-amax replay.

Only hashes and semantic identities are written.  Local checkpoint, asset,
token, and output paths are deliberately excluded from the result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from ue5m3_fp4.checkpoint import (
    NEMOTRON_H_ASSET_REPOSITORY,
    NEMOTRON_H_ASSET_REVISION,
    load_hf_nemotron_h_checkpoint,
)
from ue5m3_fp4.formats import UE5M3, round_to_format

SCHEMA = "ue5m3_fp4_public_scale_target_capture_v1"
TARGET_LAYERS = (45, 47, 49, 51)
SCALE_TARGETS = (448.0, 2048.0)
SEQUENCE_WIDTH = 8193
BLOCK_SIZE = 16
DEFAULT_HISTOGRAM_CHUNK_ELEMENTS = 4 << 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    identity = {
        "dtype": str(canonical.dtype).removeprefix("torch."),
        "shape": list(canonical.shape),
        "bytes_sha256": hashlib.sha256(
            canonical.view(torch.uint8).numpy().tobytes(order="C")
        ).hexdigest(),
    }
    return _canonical_sha256(identity)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class StreamingLogHistogram:
    """Accumulate exact counts without retaining the source tensors."""

    def __init__(
        self,
        *,
        log10_min: float,
        log10_max: float,
        bins: int,
        chunk_elements: int,
    ) -> None:
        if not math.isfinite(log10_min) or not math.isfinite(log10_max):
            raise ValueError("histogram bounds must be finite")
        if log10_min >= log10_max or bins <= 0 or chunk_elements <= 0:
            raise ValueError("histogram bounds, bins, and chunk size must be valid")
        self.log10_min = float(log10_min)
        self.log10_max = float(log10_max)
        self.bins = int(bins)
        self.chunk_elements = int(chunk_elements)
        self.counts = torch.zeros(self.bins, dtype=torch.int64)
        self.total_count = 0
        self.finite_count = 0
        self.nonfinite_count = 0
        self.zero_count = 0
        self.positive_count = 0
        self.underflow_count = 0
        self.overflow_count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.min_positive: float | None = None
        self.max_abs = 0.0
        self.update_calls = 0

    def clone_empty(self) -> StreamingLogHistogram:
        return StreamingLogHistogram(
            log10_min=self.log10_min,
            log10_max=self.log10_max,
            bins=self.bins,
            chunk_elements=self.chunk_elements,
        )

    @torch.no_grad()
    def update(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError("histogram input must be a floating-point tensor")
        flat = tensor.detach().reshape(-1)
        self.update_calls += 1
        local_counts = torch.zeros(self.bins, dtype=torch.int64, device=flat.device)
        local_total = torch.zeros((), dtype=torch.int64, device=flat.device)
        local_finite = torch.zeros((), dtype=torch.int64, device=flat.device)
        local_nonfinite = torch.zeros((), dtype=torch.int64, device=flat.device)
        local_zero = torch.zeros((), dtype=torch.int64, device=flat.device)
        local_positive = torch.zeros((), dtype=torch.int64, device=flat.device)
        local_underflow = torch.zeros((), dtype=torch.int64, device=flat.device)
        local_overflow = torch.zeros((), dtype=torch.int64, device=flat.device)
        local_sum_abs = torch.zeros((), dtype=torch.float64, device=flat.device)
        local_sum_sq = torch.zeros((), dtype=torch.float64, device=flat.device)
        local_min_positive = torch.full(
            (), float("inf"), dtype=torch.float32, device=flat.device
        )
        local_max_abs = torch.zeros((), dtype=torch.float32, device=flat.device)
        width = (self.log10_max - self.log10_min) / self.bins

        for start in range(0, flat.numel(), self.chunk_elements):
            chunk = flat[start : start + self.chunk_elements].float()
            finite = torch.isfinite(chunk)
            finite_values = chunk[finite]
            absolute = finite_values.abs()
            positive = absolute[absolute > 0]

            local_total += chunk.numel()
            local_finite += finite.sum(dtype=torch.int64)
            local_nonfinite += (~finite).sum(dtype=torch.int64)
            local_zero += (absolute == 0).sum(dtype=torch.int64)
            local_positive += positive.numel()
            local_sum_abs += absolute.sum(dtype=torch.float64)
            local_sum_sq += (absolute * absolute).sum(dtype=torch.float64)
            if absolute.numel():
                local_max_abs = torch.maximum(local_max_abs, absolute.max())
            if not positive.numel():
                continue
            local_min_positive = torch.minimum(local_min_positive, positive.min())
            log_values = torch.log10(positive)
            underflow = log_values < self.log10_min
            overflow = log_values >= self.log10_max
            in_range = ~(underflow | overflow)
            local_underflow += underflow.sum(dtype=torch.int64)
            local_overflow += overflow.sum(dtype=torch.int64)
            if in_range.any():
                indices = torch.floor((log_values[in_range] - self.log10_min) / width).to(
                    torch.int64
                )
                indices.clamp_(0, self.bins - 1)
                local_counts += torch.bincount(indices, minlength=self.bins)

        self.counts += local_counts.cpu()
        self.total_count += int(local_total.item())
        self.finite_count += int(local_finite.item())
        self.nonfinite_count += int(local_nonfinite.item())
        self.zero_count += int(local_zero.item())
        self.positive_count += int(local_positive.item())
        self.underflow_count += int(local_underflow.item())
        self.overflow_count += int(local_overflow.item())
        self.sum_abs += float(local_sum_abs.item())
        self.sum_sq += float(local_sum_sq.item())
        candidate_min = float(local_min_positive.item())
        if math.isfinite(candidate_min):
            self.min_positive = (
                candidate_min
                if self.min_positive is None
                else min(self.min_positive, candidate_min)
            )
        self.max_abs = max(self.max_abs, float(local_max_abs.item()))

    def result(self) -> dict[str, Any]:
        in_range_count = int(self.counts.sum().item())
        if self.total_count != self.finite_count + self.nonfinite_count:
            raise RuntimeError("finite and non-finite histogram counts do not sum")
        if self.finite_count != self.zero_count + self.positive_count:
            raise RuntimeError("zero and positive-magnitude histogram counts do not sum")
        if self.positive_count != (self.underflow_count + in_range_count + self.overflow_count):
            raise RuntimeError("histogram range counts do not sum")
        edges = torch.linspace(
            self.log10_min,
            self.log10_max,
            self.bins + 1,
            dtype=torch.float64,
        ).tolist()
        finite_denominator = max(1, self.finite_count)
        return {
            "domain": "log10(abs(value)); positive finite values only",
            "bin_convention": "left_closed_right_open",
            "log10_bin_edges": edges,
            "counts": self.counts.tolist(),
            "in_range_count": in_range_count,
            "underflow_count": self.underflow_count,
            "overflow_count": self.overflow_count,
            "total_count": self.total_count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "zero_count": self.zero_count,
            "positive_count": self.positive_count,
            "zero_fraction_of_finite": self.zero_count / finite_denominator,
            "mean_abs": self.sum_abs / finite_denominator,
            "rms": math.sqrt(self.sum_sq / finite_denominator),
            "min_positive": self.min_positive,
            "max_abs": self.max_abs,
            "update_calls": self.update_calls,
        }


class RawScaleCodeMetrics:
    """Stream raw and ties-to-even UE5M3 block-scale diagnostics."""

    def __init__(
        self,
        *,
        target: float,
        histogram_template: StreamingLogHistogram,
    ) -> None:
        if not math.isfinite(target) or target <= 0:
            raise ValueError("scale target must be positive and finite")
        self.target = float(target)
        self.raw_histogram = histogram_template.clone_empty()
        self.block_count = 0
        self.zero_block_count = 0
        self.raw_below_min_positive_count = 0
        self.raw_below_half_min_positive_count = 0
        self.rounded_zero_count = 0
        self.rounded_saturated_count = 0
        self.sum_abs_rounding_error = 0.0
        self.sum_sq_rounding_error = 0.0
        self.sum_abs_raw = 0.0

    @torch.no_grad()
    def update(self, block_amax: torch.Tensor, global_amax: torch.Tensor) -> None:
        global_value = float(global_amax.detach().float().item())
        if not math.isfinite(global_value) or global_value < 0:
            raise ValueError(f"invalid global amax {global_value}")
        raw = block_amax.detach().float()
        raw = raw * (self.target / global_value) if global_value else torch.zeros_like(raw)
        rounded = round_to_format(raw, UE5M3)
        error = rounded.float() - raw
        self.raw_histogram.update(raw)
        self.block_count += raw.numel()
        self.zero_block_count += int((block_amax == 0).sum().item())
        self.raw_below_min_positive_count += int(
            ((raw > 0) & (raw < UE5M3.smallest_subnormal)).sum().item()
        )
        self.raw_below_half_min_positive_count += int(
            ((raw > 0) & (raw < UE5M3.smallest_subnormal / 2)).sum().item()
        )
        self.rounded_zero_count += int(((raw > 0) & (rounded == 0)).sum().item())
        self.rounded_saturated_count += int((rounded >= UE5M3.max_finite).sum().item())
        self.sum_abs_rounding_error += float(error.abs().double().sum().item())
        self.sum_sq_rounding_error += float((error.double() ** 2).sum().item())
        self.sum_abs_raw += float(raw.abs().double().sum().item())

    def result(self) -> dict[str, Any]:
        denominator = max(1, self.block_count)
        positive_blocks = max(1, self.block_count - self.zero_block_count)
        return {
            "target": self.target,
            "scale_format": UE5M3.name,
            "scale_format_min_positive": UE5M3.smallest_subnormal,
            "scale_format_max": UE5M3.max_finite,
            "stale_growth_headroom": UE5M3.max_finite / self.target,
            "block_count": self.block_count,
            "zero_block_count": self.zero_block_count,
            "zero_block_fraction": self.zero_block_count / denominator,
            "raw_below_min_positive_count": self.raw_below_min_positive_count,
            "raw_below_min_positive_fraction_of_positive_blocks": (
                self.raw_below_min_positive_count / positive_blocks
            ),
            "raw_below_half_min_positive_count": self.raw_below_half_min_positive_count,
            "raw_below_half_min_positive_fraction_of_positive_blocks": (
                self.raw_below_half_min_positive_count / positive_blocks
            ),
            "rounded_zero_count_before_zero_scale_repair": self.rounded_zero_count,
            "rounded_zero_fraction_of_positive_blocks_before_zero_scale_repair": (
                self.rounded_zero_count / positive_blocks
            ),
            "rounded_saturated_count": self.rounded_saturated_count,
            "rounded_saturated_fraction": self.rounded_saturated_count / denominator,
            "mean_abs_rounding_error": self.sum_abs_rounding_error / denominator,
            "rms_rounding_error": math.sqrt(self.sum_sq_rounding_error / denominator),
            "relative_l1_rounding_error": self.sum_abs_rounding_error
            / max(self.sum_abs_raw, 1e-30),
            "raw_code_histogram": self.raw_histogram.result(),
        }


def iter_wgrad_dy_block_amax(
    dy: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
    output_chunk: int = 512,
) -> Iterable[torch.Tensor]:
    """Yield block maxima for ``dY.T`` along the Wgrad reduction dimension."""

    if dy.ndim < 2:
        raise ValueError(f"expected dY with at least two dimensions, got {dy.shape}")
    if block_size <= 0 or output_chunk <= 0:
        raise ValueError("block_size and output_chunk must be positive")
    dy_2d = dy.detach().reshape(-1, dy.shape[-1])
    reduction_size, output_features = dy_2d.shape
    padded_reduction = (reduction_size + block_size - 1) // block_size * block_size
    for start in range(0, output_features, output_chunk):
        chunk = dy_2d[:, start : start + output_chunk].float().abs()
        if padded_reduction != reduction_size:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, 0, 0, padded_reduction - reduction_size),
            )
        # This is equivalent to blocking [output_features, reduction_size]
        # without materializing the full transpose.
        maxima = chunk.reshape(-1, block_size, chunk.shape[-1]).amax(dim=1)
        yield maxima.transpose(0, 1).contiguous().reshape(-1)


def _histogram(
    *,
    log10_min: float,
    log10_max: float,
    bins: int,
    chunk_elements: int,
) -> StreamingLogHistogram:
    return StreamingLogHistogram(
        log10_min=log10_min,
        log10_max=log10_max,
        bins=bins,
        chunk_elements=chunk_elements,
    )


def load_token_row(
    path: Path,
    *,
    tensor_key: str,
    sequence_index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load one row from a local ``[N, 8193]`` safetensors token file."""

    if sequence_index < 0:
        raise ValueError("sequence_index must be nonnegative")
    if not path.is_file():
        raise FileNotFoundError(path)
    file_sha256 = _sha256_file(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != {tensor_key}:
            raise ValueError(
                f"token file tensors are {sorted(keys)}, expected only {tensor_key!r}"
            )
        tensor = handle.get_tensor(tensor_key)
    if tensor.dtype not in {torch.int32, torch.int64}:
        raise TypeError("token tensor must have int32 or int64 dtype")
    if tensor.ndim != 2 or tensor.shape[1] != SEQUENCE_WIDTH:
        raise ValueError(
            f"token tensor must have shape [N, {SEQUENCE_WIDTH}], got {list(tensor.shape)}"
        )
    if not 0 <= sequence_index < tensor.shape[0]:
        raise IndexError(f"sequence_index {sequence_index} is outside N={tensor.shape[0]} rows")
    if _sha256_file(path) != file_sha256:
        raise RuntimeError("token file changed while it was being read")
    row = tensor[sequence_index].contiguous()
    return row, {
        "schema": "ue5m3_fp4_public_token_row_identity_v1",
        "file_sha256": file_sha256,
        "file_bytes": path.stat().st_size,
        "tensor_key": tensor_key,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype).removeprefix("torch."),
        "sequence_index": sequence_index,
        "sequence_sha256": _tensor_sha256(row),
        "prediction_tokens": SEQUENCE_WIDTH - 1,
    }


def _capture(
    *,
    model: torch.nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    histogram_log10_min: float,
    histogram_log10_max: float,
    histogram_bins: int,
    raw_code_log10_min: float,
    raw_code_log10_max: float,
    raw_code_bins: int,
    chunk_elements: int,
) -> dict[str, Any]:
    def value_histogram() -> StreamingLogHistogram:
        return _histogram(
            log10_min=histogram_log10_min,
            log10_max=histogram_log10_max,
            bins=histogram_bins,
            chunk_elements=chunk_elements,
        )

    def code_histogram() -> StreamingLogHistogram:
        return _histogram(
            log10_min=raw_code_log10_min,
            log10_max=raw_code_log10_max,
            bins=raw_code_bins,
            chunk_elements=chunk_elements,
        )

    target_names = {
        layer: f"backbone.layers.{layer}.mixer.down_proj" for layer in TARGET_LAYERS
    }
    modules = dict(model.named_modules())
    missing = [name for name in target_names.values() if name not in modules]
    if missing:
        raise KeyError(f"checkpoint model lacks expected target modules: {missing}")

    per_layer: dict[int, dict[str, Any]] = {}
    pooled = {
        "x": value_histogram(),
        "weight": value_histogram(),
        "dy": value_histogram(),
        "wgrad_dy_block_amax": value_histogram(),
        "raw_scale_codes": {
            target: RawScaleCodeMetrics(
                target=target,
                histogram_template=code_histogram(),
            )
            for target in SCALE_TARGETS
        },
    }
    handles: list[Any] = []
    capture_original_forward = {"active": False}

    for layer, module_name in target_names.items():
        module = modules[module_name]
        if not hasattr(module, "weight") or not isinstance(module.weight, torch.Tensor):
            raise TypeError(f"target module {module_name} has no tensor weight")
        record: dict[str, Any] = {
            "module_name": module_name,
            "x_histogram": value_histogram(),
            "weight_histogram": value_histogram(),
            "dy_histogram": value_histogram(),
            "wgrad_dy_block_amax_histogram": value_histogram(),
            "raw_scale_codes": {
                target: RawScaleCodeMetrics(
                    target=target,
                    histogram_template=code_histogram(),
                )
                for target in SCALE_TARGETS
            },
            "x_shape": None,
            "weight_shape": list(module.weight.shape),
            "dy_shape": None,
            "dy_current_tensor_amax": None,
        }
        record["weight_histogram"].update(module.weight)
        pooled["weight"].update(module.weight)
        per_layer[layer] = record

        def forward_pre_hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            *,
            layer_index: int = layer,
        ) -> None:
            # Gradient checkpointing recomputes this module during backward.
            # Count X only on the original forward.
            if not capture_original_forward["active"]:
                return
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"layer {layer_index} down_proj has no tensor input")
            x = inputs[0]
            layer_record = per_layer[layer_index]
            layer_record["x_shape"] = list(x.shape)
            layer_record["x_histogram"].update(x)
            pooled["x"].update(x)

        def backward_hook(
            _module: torch.nn.Module,
            _grad_input: tuple[Any, ...],
            grad_output: tuple[Any, ...],
            *,
            layer_index: int = layer,
        ) -> None:
            if not grad_output or not isinstance(grad_output[0], torch.Tensor):
                raise TypeError(f"layer {layer_index} down_proj has no tensor dY")
            dy = grad_output[0]
            layer_record = per_layer[layer_index]
            layer_record["dy_shape"] = list(dy.shape)
            layer_record["dy_histogram"].update(dy)
            pooled["dy"].update(dy)
            global_amax = dy.detach().float().abs().max()
            layer_record["dy_current_tensor_amax"] = float(global_amax.item())
            for block_amax in iter_wgrad_dy_block_amax(dy):
                layer_record["wgrad_dy_block_amax_histogram"].update(block_amax)
                pooled["wgrad_dy_block_amax"].update(block_amax)
                for target in SCALE_TARGETS:
                    layer_record["raw_scale_codes"][target].update(block_amax, global_amax)
                    pooled["raw_scale_codes"][target].update(block_amax, global_amax)

        handles.extend(
            (
                module.register_forward_pre_hook(forward_pre_hook),
                module.register_full_backward_hook(backward_hook),
            )
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    model.config.use_cache = False
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise TypeError("loaded model does not support gradient checkpointing")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dropout_probabilities = sorted(
        {float(module.p) for module in model.modules() if isinstance(module, torch.nn.Dropout)}
    )
    if any(dropout_probabilities):
        raise RuntimeError(
            f"deterministic capture requires zero dropout; got {dropout_probabilities}"
        )

    input_ids = tokens[:-1].to(device=device, dtype=torch.long).unsqueeze(0)
    labels = tokens[1:].to(device=device, dtype=torch.long).unsqueeze(0)
    input_embeddings = model.get_input_embeddings()(input_ids).detach()
    input_embeddings.requires_grad_(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        capture_original_forward["active"] = True
        try:
            hidden_states = model.backbone(
                inputs_embeds=input_embeddings,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        finally:
            capture_original_forward["active"] = False
        try:
            from cut_cross_entropy import linear_cross_entropy
        except ImportError as error:
            raise RuntimeError(
                "the 8B histogram capture requires cut-cross-entropy; install the "
                "pinned reproduction environment before running this command"
            ) from error
        loss = linear_cross_entropy(
            hidden_states.to(torch.bfloat16),
            model.lm_head.weight.to(torch.bfloat16),
            labels,
            shift=False,
            reduction="mean",
            filter_c_grad=False,
        )
        loss.backward()
        if input_embeddings.grad is None:
            raise RuntimeError("backward did not reach the detached embedding input")
        input_gradient_norm = float(
            torch.linalg.vector_norm(input_embeddings.grad.float()).item()
        )
    finally:
        for handle in handles:
            handle.remove()

    layers: dict[str, Any] = {}
    for layer, record in per_layer.items():
        for name in ("x_histogram", "weight_histogram", "dy_histogram"):
            if record[name].update_calls != 1:
                raise RuntimeError(
                    f"layer {layer} {name} was captured {record[name].update_calls} "
                    "times; expected exactly once"
                )
        if record["x_shape"] is None or record["dy_shape"] is None:
            raise RuntimeError(f"layer {layer} capture is incomplete")
        layers[str(layer)] = {
            "module_name": record["module_name"],
            "x_shape": record["x_shape"],
            "weight_shape": record["weight_shape"],
            "dy_shape": record["dy_shape"],
            "dy_current_tensor_amax": record["dy_current_tensor_amax"],
            "x_histogram": record["x_histogram"].result(),
            "weight_histogram": record["weight_histogram"].result(),
            "dy_histogram": record["dy_histogram"].result(),
            "wgrad_dy_block_amax_histogram": record["wgrad_dy_block_amax_histogram"].result(),
            "raw_scale_codes": {
                f"{target:g}": record["raw_scale_codes"][target].result()
                for target in SCALE_TARGETS
            },
        }

    result = {
        "loss": float(loss.detach().float().item()),
        "prediction_tokens": int(labels.numel()),
        "input_gradient_norm": input_gradient_norm,
        "dropout_probabilities": dropout_probabilities,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
        "parameter_requires_grad_count": sum(
            int(parameter.requires_grad) for parameter in model.parameters()
        ),
        "target_layers": list(TARGET_LAYERS),
        "layers": layers,
        "pooled": {
            "x_histogram": pooled["x"].result(),
            "weight_histogram": pooled["weight"].result(),
            "dy_histogram": pooled["dy"].result(),
            "wgrad_dy_block_amax_histogram": pooled["wgrad_dy_block_amax"].result(),
            "raw_scale_codes": {
                f"{target:g}": pooled["raw_scale_codes"][target].result()
                for target in SCALE_TARGETS
            },
        },
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return result


def _architecture_identity(model: torch.nn.Module) -> dict[str, Any]:
    fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "hybrid_override_pattern",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "ssm_state_size",
        "mamba_num_heads",
        "mamba_head_dim",
        "n_groups",
        "conv_kernel",
        "chunk_size",
        "mamba_expand",
        "mamba_hidden_act",
        "attention_dropout",
        "hidden_dropout",
        "mlp_hidden_act",
        "tie_word_embeddings",
    )
    architecture = {field: getattr(model.config, field) for field in fields}
    architecture["model_class"] = type(model).__name__
    architecture["config_class"] = type(model.config).__name__
    architecture["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    architecture["sha256"] = _canonical_sha256(architecture)
    return architecture


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-label",
        required=True,
        help="Public semantic label, e.g. bf16 or ue5m3-proposed-b16-step-30000.",
    )
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--tensor-key", default="tokens")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-assets", default=NEMOTRON_H_ASSET_REPOSITORY)
    parser.add_argument("--hf-revision", default=NEMOTRON_H_ASSET_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--histogram-log10-min", type=float, default=-16.0)
    parser.add_argument("--histogram-log10-max", type=float, default=8.0)
    parser.add_argument("--histogram-bins", type=int, default=192)
    parser.add_argument("--raw-code-log10-min", type=float, default=-16.0)
    parser.add_argument("--raw-code-log10-max", type=float, default=5.0)
    parser.add_argument("--raw-code-bins", type=int, default=168)
    parser.add_argument(
        "--histogram-chunk-elements",
        type=int,
        default=DEFAULT_HISTOGRAM_CHUNK_ELEMENTS,
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    safe_label = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
    if safe_label.fullmatch(args.checkpoint_label) is None:
        raise ValueError(
            "--checkpoint-label must be a location-free identifier using only "
            "letters, digits, '.', '_', '+', or '-'"
        )
    if safe_label.fullmatch(args.tensor_key) is None:
        raise ValueError("--tensor-key must be a location-free tensor identifier")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the 8B forward/backward capture requires a CUDA device")
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if cublas_workspace_config not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8' for a "
            "deterministic CUDA capture"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    torch.cuda.set_device(0 if device.index is None else device.index)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tokens, token_identity = load_token_row(
        args.tokens,
        tensor_key=args.tensor_key,
        sequence_index=args.sequence_index,
    )
    model, load_provenance = load_hf_nemotron_h_checkpoint(
        args.checkpoint,
        hf_assets=args.hf_assets,
        hf_revision=args.hf_revision,
        device=device,
        local_files_only=args.local_files_only,
    )
    architecture = _architecture_identity(model)
    if int(tokens.min().item()) < 0 or int(tokens.max().item()) >= int(model.config.vocab_size):
        raise ValueError("selected token row contains an ID outside the model vocabulary")
    floating_dtypes = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    if floating_dtypes != ["bfloat16"]:
        raise TypeError(f"expected BF16 master weights, got {floating_dtypes}")
    try:
        capture = _capture(
            model=model,
            tokens=tokens,
            device=device,
            histogram_log10_min=args.histogram_log10_min,
            histogram_log10_max=args.histogram_log10_max,
            histogram_bins=args.histogram_bins,
            raw_code_log10_min=args.raw_code_log10_min,
            raw_code_log10_max=args.raw_code_log10_max,
            raw_code_bins=args.raw_code_bins,
            chunk_elements=args.histogram_chunk_elements,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "schema": SCHEMA,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "scope": {
            "checkpoint_label": args.checkpoint_label,
            "mode": "training-mode forward/backward; no optimizer step",
            "parameter_gradient_policy": (
                "all model parameters frozen; gradient propagated to detached input embeddings"
            ),
            "interpretation": (
                "BF16 master-weight value-distribution probe; not an FP4 execution "
                "trace and not a delayed-amax replay"
            ),
        },
        "checkpoint": load_provenance["checkpoint"],
        "assets": load_provenance["assets"],
        "token_row": token_identity,
        "model": {
            "architecture": architecture,
            "parameter_dtypes": floating_dtypes,
            "attention": load_provenance["attention"],
            "attention_verified": load_provenance["attention_verified"],
        },
        "protocol": {
            "seed": args.seed,
            "dropout": 0.0,
            "single_device": True,
            "torch_deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
            "cublas_workspace_config": cublas_workspace_config,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "block_size": BLOCK_SIZE,
            "scale_format": UE5M3.name,
            "scale_targets": list(SCALE_TARGETS),
            "scale_reference": "current dY tensor amax, separately per module",
            "wgrad_dy_layout": ("dY.T with block scaling along the token reduction dimension"),
            "numeric_path": "BF16 HF-export forward/backward",
            "histogram": {
                "log10_min": args.histogram_log10_min,
                "log10_max": args.histogram_log10_max,
                "bins": args.histogram_bins,
                "raw_code_log10_min": args.raw_code_log10_min,
                "raw_code_log10_max": args.raw_code_log10_max,
                "raw_code_bins": args.raw_code_bins,
                "chunk_elements": args.histogram_chunk_elements,
            },
            "capture_source_sha256": _sha256_file(Path(__file__)),
        },
        "runtime": {
            "cuda_device_name": torch.cuda.get_device_name(device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "transformers_version": _distribution_version("transformers"),
            "mamba_ssm_version": _distribution_version("mamba-ssm"),
            "causal_conv1d_version": _distribution_version("causal-conv1d"),
            "cut_cross_entropy_version": _distribution_version("cut-cross-entropy"),
            "ue5m3_fp4_version": _distribution_version("ue5m3-fp4"),
        },
        "capture": capture,
    }
    _atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "status": "succeeded",
                "result_sha256": _sha256_file(args.output),
                "loss": capture["loss"],
                "prediction_tokens": capture["prediction_tokens"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
