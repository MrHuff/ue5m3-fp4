#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native NVFP4 GEMM with the public K64 issue-RZ emulator.

This diagnostic requires Blackwell, CUDA, Triton, and the repository-pinned
Transformer Engine build. No native-to-software fallback is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ue5m3_fp4.backends.triton import issue_rz_bf16_gemm, triton_available
from ue5m3_fp4.integrations.torchtitan.comparators import (
    TRANSFORMER_ENGINE_REVISION,
    require_pinned_transformer_engine,
)

try:
    from .common import make_payload, write_json
except ImportError:  # Direct ``python reproduce/diagnostics/...py`` execution.
    from common import make_payload, write_json


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class PackedCase:
    activation_nibbles: torch.Tensor
    weight_nibbles: torch.Tensor
    activation_scales: torch.Tensor
    weight_scales: torch.Tensor
    alpha: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_native_runtime(expected_library_sha256: str | None) -> tuple[Any, ...]:
    requested = os.environ.get("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING")
    if requested not in {None, "1"}:
        raise RuntimeError(
            "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING must be unset or 1 for this oracle"
        )
    os.environ["NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] = "1"
    if not torch.cuda.is_available() or not triton_available():
        raise RuntimeError("native comparison requires CUDA and Triton")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 10:
        raise RuntimeError(
            f"native NVFP4 requires Blackwell (compute capability >= 10.0); found {capability}"
        )
    runtime = require_pinned_transformer_engine()
    import transformer_engine

    package_dir = Path(transformer_engine.__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor

    native_library = Path(tex.__file__).resolve()
    native_sha256 = _sha256_file(native_library)
    if expected_library_sha256 is not None and native_sha256 != expected_library_sha256:
        raise RuntimeError(
            "Transformer Engine native-library SHA256 mismatch: "
            f"expected {expected_library_sha256}, found {native_sha256}"
        )
    return runtime, tex, TE_DType, NVFP4Quantizer, NVFP4Tensor, native_sha256


def _make_quantizers(tex: Any, quantizer_type: type[Any]) -> tuple[Any, Any]:
    options = {
        "fp4_dtype": tex.DType.kFloat4E2M1,
        "rowwise": True,
        "columnwise": False,
        "with_rht": False,
        "with_post_rht_amax": False,
        "with_2d_quantization": False,
        "stochastic_rounding": False,
    }
    activation = quantizer_type(**options)
    weight = quantizer_type(**options)
    activation.internal = True
    weight.internal = True
    return activation, weight


def _make_case(m: int, n: int, k: int, seed: int, device: torch.device) -> PackedCase:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    # Sign occupies bit 3; magnitude codes 0..7 follow the E2M1 lookup table.
    activation = torch.randint(0, 16, (m, k), generator=generator, device=device).to(
        torch.uint8
    )
    weight = torch.randint(0, 16, (n, k), generator=generator, device=device).to(torch.uint8)
    blocks = k // 16
    activation_exponents = torch.randint(-5, 7, (m, blocks), generator=generator, device=device)
    weight_exponents = torch.randint(-5, 7, (n, blocks), generator=generator, device=device)
    activation_scales = torch.exp2(activation_exponents.float()).to(torch.float8_e4m3fn)
    weight_scales = torch.exp2(weight_exponents.float()).to(torch.float8_e4m3fn)
    return PackedCase(
        activation_nibbles=activation,
        weight_nibbles=weight,
        activation_scales=activation_scales,
        weight_scales=weight_scales,
        alpha=1.0,
    )


def _permute_issue_groups(case: PackedCase, order: torch.Tensor) -> PackedCase:
    groups = case.activation_nibbles.shape[1] // 64
    if order.shape != (groups,) or set(order.cpu().tolist()) != set(range(groups)):
        raise ValueError("order must be a permutation of all K64 issue groups")
    activation = case.activation_nibbles.reshape(case.activation_nibbles.shape[0], groups, 64)[
        :, order, :
    ].reshape_as(case.activation_nibbles)
    weight = case.weight_nibbles.reshape(case.weight_nibbles.shape[0], groups, 64)[
        :, order, :
    ].reshape_as(case.weight_nibbles)
    # CPU indexing is not implemented for Float8 in all supported Torch
    # versions, so reorder through FP32 and restore the exact Float8 values.
    activation_scales = (
        case.activation_scales.float()
        .reshape(case.activation_scales.shape[0], groups, 4)[:, order, :]
        .reshape_as(case.activation_scales)
        .to(case.activation_scales.dtype)
    )
    weight_scales = (
        case.weight_scales.float()
        .reshape(case.weight_scales.shape[0], groups, 4)[:, order, :]
        .reshape_as(case.weight_scales)
        .to(case.weight_scales.dtype)
    )
    return PackedCase(activation, weight, activation_scales, weight_scales, case.alpha)


def _pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    return (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()


def _decode(nibbles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    magnitude = torch.tensor(_E2M1, device=nibbles.device, dtype=torch.float32)
    values = magnitude[(nibbles & 7).long()]
    values = torch.where((nibbles & 8).bool(), -values, values)
    expanded_scales = torch.repeat_interleave(scales.float(), 16, dim=1)
    return (values * expanded_scales).to(torch.bfloat16).contiguous()


def _scale_storage(
    values: torch.Tensor,
    *,
    storage_shape: tuple[int, ...],
) -> torch.Tensor:
    output = torch.zeros(storage_shape, dtype=torch.uint8, device=values.device)
    output[: values.shape[0], : values.shape[1]] = values.view(torch.uint8)
    return output


def _wrap_case(
    case: PackedCase,
    *,
    tex: Any,
    quantizer_type: type[Any],
    tensor_type: type[Any],
) -> tuple[Any, Any]:
    activation_quantizer, weight_quantizer = _make_quantizers(tex, quantizer_type)
    m, k = case.activation_nibbles.shape
    n = case.weight_nibbles.shape[0]
    activation_scales = _scale_storage(
        case.activation_scales,
        storage_shape=activation_quantizer.get_scale_shape((m, k), False),
    )
    weight_scales = _scale_storage(
        case.weight_scales,
        storage_shape=weight_quantizer.get_scale_shape((n, k), False),
    )

    def wrap(
        *,
        shape: tuple[int, int],
        nibbles: torch.Tensor,
        scales: torch.Tensor,
        amax: float,
        quantizer: Any,
    ) -> Any:
        return tensor_type(
            shape=torch.Size(shape),
            dtype=torch.bfloat16,
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise_data=_pack_nibbles(nibbles),
            rowwise_scale_inv=scales,
            columnwise_data=None,
            columnwise_scale_inv=None,
            amax_rowwise=torch.tensor([amax], dtype=torch.float32, device=nibbles.device),
            amax_columnwise=None,
            quantizer=quantizer,
            requires_grad=False,
            with_gemm_swizzled_scales=False,
        )

    activation = wrap(
        shape=(m, k),
        nibbles=case.activation_nibbles,
        scales=activation_scales,
        amax=6.0 * 448.0,
        quantizer=activation_quantizer,
    )
    weight = wrap(
        shape=(n, k),
        nibbles=case.weight_nibbles,
        scales=weight_scales,
        amax=case.alpha * 6.0 * 448.0,
        quantizer=weight_quantizer,
    )
    return activation, weight


def _native_gemm(
    activation: Any,
    weight: Any,
    *,
    tex: Any,
    te_dtype: Any,
    shape: tuple[int, int],
    workspace: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(shape, dtype=torch.float32, device=workspace.device)
    tex.generic_gemm(
        weight,
        True,
        activation,
        False,
        output,
        None,
        te_dtype[torch.float32],
        None,
        te_dtype[torch.float32],
        False,
        None,
        False,
        workspace,
        workspace.numel(),
        False,
        False,
    )
    torch.cuda.synchronize(workspace.device)
    return output


def _run_case(
    case: PackedCase,
    *,
    tex: Any,
    te_dtype: Any,
    quantizer_type: type[Any],
    tensor_type: type[Any],
    workspace: torch.Tensor,
) -> dict[str, torch.Tensor]:
    activation_q, weight_q = _wrap_case(
        case,
        tex=tex,
        quantizer_type=quantizer_type,
        tensor_type=tensor_type,
    )
    m, _ = case.activation_nibbles.shape
    n = case.weight_nibbles.shape[0]
    native = _native_gemm(
        activation_q,
        weight_q,
        tex=tex,
        te_dtype=te_dtype,
        shape=(m, n),
        workspace=workspace,
    )
    activation = _decode(case.activation_nibbles, case.activation_scales)
    weight = _decode(case.weight_nibbles, case.weight_scales)
    issue_rz = (
        issue_rz_bf16_gemm(
            activation,
            weight.transpose(0, 1).contiguous(),
            snap_to_1_over_1024=False,
        )
        * case.alpha
    )
    issue_rz_grid = torch.round((issue_rz / case.alpha) * 1024.0) / 1024.0 * case.alpha
    rn = torch.mm(activation.float(), weight.float().transpose(0, 1)) * case.alpha
    return {"native": native, "issue_rz": issue_rz, "issue_rz_grid": issue_rz_grid, "rn": rn}


def _comparison(candidate: torch.Tensor, native: torch.Tensor) -> dict[str, Any]:
    difference = candidate.float() - native.float()
    return {
        "fp32_exact": int(torch.sum(candidate.float() == native.float()).item()),
        "bf16_exact": int(
            torch.sum(candidate.to(torch.bfloat16) == native.to(torch.bfloat16)).item()
        ),
        "elements": native.numel(),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "maximum_absolute_error": float(difference.abs().max().item()),
    }


def _sample_indices(m: int, n: int, count: int, seed: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("samples must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(m * n, generator=generator)[: min(count, m * n)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=20_260_622)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--permutations", type=int, default=258)
    parser.add_argument("--expected-te-library-sha256")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.m, args.n, args.k, args.samples) <= 0 or args.permutations < 0:
        raise ValueError(
            "matrix dimensions/samples must be positive; permutations non-negative"
        )
    if args.m % 16 or args.n % 16 or args.k % 64:
        raise ValueError("m and n must be divisible by 16; k must be divisible by 64")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("native NVFP4 comparison requires a CUDA device")
    runtime, tex, te_dtype, quantizer_type, tensor_type, library_sha256 = (
        _require_native_runtime(args.expected_te_library_sha256)
    )
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    case = _make_case(args.m, args.n, args.k, args.seed, device)
    matrices = _run_case(
        case,
        tex=tex,
        te_dtype=te_dtype,
        quantizer_type=quantizer_type,
        tensor_type=tensor_type,
        workspace=workspace,
    )
    indices = _sample_indices(args.m, args.n, args.samples, args.seed + 17).to(device)
    sampled = {name: value.flatten()[indices] for name, value in matrices.items()}
    large_comparison = {
        name: _comparison(value, sampled["native"])
        for name, value in sampled.items()
        if name != "native"
    }

    permutation_counts = {name: 0 for name in ("issue_rz", "issue_rz_grid", "rn")}
    groups = args.k // 64
    permutation_generator = torch.Generator(device="cpu").manual_seed(args.seed + 29)
    for _ in range(args.permutations):
        order = torch.randperm(groups, generator=permutation_generator).to(device)
        permuted = _permute_issue_groups(case, order)
        outputs = _run_case(
            permuted,
            tex=tex,
            te_dtype=te_dtype,
            quantizer_type=quantizer_type,
            tensor_type=tensor_type,
            workspace=workspace,
        )
        native_cell = outputs["native"][0, 0]
        for name in permutation_counts:
            permutation_counts[name] += int(outputs[name][0, 0] == native_cell)

    config = {
        "device": str(device),
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "seed": args.seed,
        "samples": min(args.samples, args.m * args.n),
        "permutations": args.permutations,
        "operand_generation": "uniform E2M1 codes; power-of-two E4M3 block-16 scales",
        "native_output_dtype": "float32",
        "transformer_engine_revision": TRANSFORMER_ENGINE_REVISION,
        "transformer_engine_version": runtime.version,
        "transformer_engine_native_library_sha256": library_sha256,
    }
    write_json(
        args.output_json,
        make_payload(
            experiment="native_nvfp4_vs_probe_matched_issue_rz",
            config=config,
            results={
                "large_matrix_sample": large_comparison,
                "k64_group_permutations": {
                    "matches": permutation_counts,
                    "trials": args.permutations,
                    "cell": [0, 0],
                    "note": (
                        "This is a new deterministic synthetic corpus. It validates the "
                        "public implementation but is not the archived 258-case witness corpus."
                    ),
                },
            },
            device=device,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
