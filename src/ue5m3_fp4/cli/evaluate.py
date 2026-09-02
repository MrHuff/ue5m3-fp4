# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an HF-exported Nemotron-H checkpoint on local token shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ue5m3_fp4.checkpoint import (
    NEMOTRON_H_ASSET_REPOSITORY,
    NEMOTRON_H_ASSET_REVISION,
    load_hf_nemotron_h_checkpoint,
)
from ue5m3_fp4.eval.validation import discover_validation_paths, evaluate_validation
from ue5m3_fp4.integrations.torchtitan.comparators import (
    NativeNVFP4Variant,
    convert_native_nvfp4_nemotron_h,
    convert_ue5m3_te_settings_nemotron_h,
)
from ue5m3_fp4.integrations.torchtitan.selection import (
    FP32OutputLinear,
    convert_reported_nemotron_h,
)
from ue5m3_fp4.nn.linear import LinearBackend
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.inference import (
    FP4InferenceScalingController,
    learned_weight_bf16_numeric_path,
)

EVALUATION_RUN_SCHEMA = "ue5m3_fp4_public_validation_run_v1"
_NUMERIC_PATH_ALIASES = {
    "bf16": "bf16",
    "ue5m3-b16": "ue5m3-proposed-b16",
    "ue5m3-proposed-b16": "ue5m3-proposed-b16",
    "ue5m3-b32": "ue5m3-proposed-b32",
    "ue5m3-proposed-b32": "ue5m3-proposed-b32",
    "ue5m3-torch-control": "ue5m3-torch-control",
    "ue5m3-te-settings": "ue5m3-te-settings",
    "native-nvfp4-te": "native-nvfp4-te",
    "native-nvfp4-no-rht-all": "native-nvfp4-no-rht-all",
}
_NUMERIC_PATHS = tuple(_NUMERIC_PATH_ALIASES)
_D50_UE5M3_CONFIGURATIONS = {
    "ue5m3-proposed-b16",
    "ue5m3-proposed-b32",
    "ue5m3-torch-control",
}
_CURRENT_UE5M3_CONFIGURATIONS = {"ue5m3-te-settings"}
_NATIVE_NVFP4_CONFIGURATIONS = {
    "native-nvfp4-te",
    "native-nvfp4-no-rht-all",
}
_ACTIVATION_MODES = ("current_tensor", "training_replay", "calibrated_frozen")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_token_tensor(path: Path, *, tensor_key: str) -> torch.Tensor:
    values = load_file(path, device="cpu")
    if tensor_key not in values:
        raise KeyError(f"{path} does not contain tensor {tensor_key!r}")
    tokens = values[tensor_key]
    if tokens.dtype not in {torch.int32, torch.int64}:
        raise TypeError(f"{path}:{tensor_key} must contain int32 or int64 tokens")
    if tokens.ndim != 2 or tokens.shape[0] <= 0 or tokens.shape[1] <= 1:
        raise ValueError(
            f"{path}:{tensor_key} must have shape [N, input_tokens + 1], "
            f"got {tuple(tokens.shape)}"
        )
    return tokens.contiguous()


def token_dataset_identity(
    inputs: Sequence[str | Path],
    *,
    tensor_key: str,
    expected_input_tokens: int,
    expected_sequences: int | None,
) -> tuple[list[Path], dict[str, Any], frozenset[str]]:
    """Resolve ordered shards and hash both files and individual token rows."""

    paths = discover_validation_paths(inputs)
    source_records: list[dict[str, Any]] = []
    row_hashes: list[str] = []
    sequence_count = 0
    for source_index, path in enumerate(paths):
        file_sha256 = _sha256_file(path)
        tokens = _load_token_tensor(path, tensor_key=tensor_key)
        input_tokens = int(tokens.shape[1]) - 1
        if input_tokens != expected_input_tokens:
            raise ValueError(
                f"{path} contains {input_tokens} input tokens per sequence; "
                f"expected {expected_input_tokens}"
            )
        source_records.append(
            {
                "source_index": source_index,
                "bytes": path.stat().st_size,
                "sha256": file_sha256,
                "shape": list(tokens.shape),
                "dtype": str(tokens.dtype).removeprefix("torch."),
            }
        )
        for row in tokens:
            canonical = row.contiguous().view(torch.uint8).numpy().tobytes()
            row_hashes.append(hashlib.sha256(canonical).hexdigest())
        sequence_count += int(tokens.shape[0])
    if len(row_hashes) != len(set(row_hashes)):
        raise ValueError("the token dataset contains duplicate sequence records")
    if expected_sequences is not None and sequence_count != expected_sequences:
        raise ValueError(
            f"token dataset contains {sequence_count} sequences; expected {expected_sequences}"
        )
    identity = {
        "schema": "ue5m3_fp4_local_token_dataset_identity_v1",
        "tensor_key": tensor_key,
        "ordered_sources": source_records,
        "source_count": len(source_records),
        "sequence_count": sequence_count,
        "input_tokens_per_sequence": expected_input_tokens,
        "target_only_tokens_per_sequence": 1,
        "ordered_record_hashes_sha256": _canonical_sha256(row_hashes),
    }
    identity["sha256"] = _canonical_sha256(identity)
    return paths, identity, frozenset(row_hashes)


def _token_batches(
    paths: Sequence[Path],
    *,
    tensor_key: str,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    for path in paths:
        expected_sha256 = _sha256_file(path)
        tokens = _load_token_tensor(path, tensor_key=tensor_key)
        if _sha256_file(path) != expected_sha256:
            raise RuntimeError(f"token shard changed while being loaded: {path}")
        for start in range(0, int(tokens.shape[0]), batch_size):
            yield tokens[start : start + batch_size]


def _run_activation_calibration(
    model: torch.nn.Module,
    controller: FP4InferenceScalingController,
    paths: Sequence[Path],
    *,
    identity: dict[str, Any],
    tensor_key: str,
    device: torch.device,
) -> None:
    controller.begin_activation_calibration(identity)
    with torch.inference_mode():
        for tokens in _token_batches(paths, tensor_key=tensor_key, batch_size=1):
            input_ids = tokens[:, :-1].to(device=device, dtype=torch.long).contiguous()
            controller.record_activation_calibration_batch(
                input_ids,
                token_count=input_ids.numel(),
            )
            model.backbone(input_ids=input_ids, use_cache=False, return_dict=True)
    controller.freeze_activation_scales()


def _enable_native_nvfp4_alignment(
    model: torch.nn.Module,
    *,
    expected_modules: int,
) -> tuple[tuple[str, torch.nn.Module], ...]:
    modules = tuple(
        sorted(
            (
                (name, module)
                for name, module in model.named_modules()
                if callable(getattr(module, "native_nvfp4_report", None))
            ),
            key=lambda item: item[0],
        )
    )
    if len(modules) != expected_modules:
        raise RuntimeError(
            f"native NVFP4 conversion exposed {len(modules)} modules; "
            f"expected {expected_modules}"
        )
    with torch.no_grad():
        for name, module in modules:
            enable = getattr(module, "enable_nvfp4_inference_alignment", None)
            if not callable(enable):
                raise TypeError(f"native NVFP4 module {name} has no alignment API")
            enable()
            report = module.native_nvfp4_report()
            alignment = report.get("alignment", {})
            if (
                report.get("tensor_scaling") != "current_tensor"
                or report.get("effective_interval") != 1
                or report.get("delayed_amax_fields_applied") is not False
                or alignment.get("enabled") is not True
                or any(alignment.get("counters", {}).values())
            ):
                raise RuntimeError(f"native NVFP4 module {name} has invalid fresh state")
    return modules


def _native_nvfp4_provenance(
    modules: Sequence[tuple[str, torch.nn.Module]],
    *,
    configuration: str,
    expected_forwards: int,
) -> dict[str, Any]:
    reports = {name: module.native_nvfp4_report() for name, module in modules}
    for name, report in reports.items():
        counters = report.get("alignment", {}).get("counters", {})
        if (
            report.get("numeric_backend") != "native_transformer_engine_nvfp4"
            or report.get("native_hardware") is not True
            or report.get("tensor_scaling") != "current_tensor"
            or report.get("effective_interval") != 1
            or report.get("delayed_amax_fields_applied") is not False
            or counters.get("forward_count") != expected_forwards
        ):
            raise RuntimeError(
                f"native NVFP4 module {name} has incomplete measurement evidence"
            )
    return {
        "schema": "ue5m3_fp4_evaluation_numeric_path_v1",
        "numeric_path": "native_nvfp4_transformer_engine",
        "configuration": configuration,
        "learned_weight_dtype": "bfloat16",
        "fp4_quantization_applied": True,
        "inference_scaling_protocol": {
            "mode": "current_tensor",
            "effective_interval": 1,
            "checkpoint_cache_inherited": False,
            "delayed_amax_fields_applied": False,
            "module_count": len(reports),
            "module_reports": reports,
            "module_reports_sha256": _canonical_sha256(reports),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one validated, single-process evaluation."""

    configuration = _NUMERIC_PATH_ALIASES[args.numeric_path]
    device = torch.device(args.device)
    if configuration != "bf16" and device.type != "cuda":
        raise ValueError("the released FP4 evaluation requires a CUDA device")
    if configuration in {"bf16", *_NATIVE_NVFP4_CONFIGURATIONS} and (
        args.activation_mode is not None
    ):
        raise ValueError("--activation-mode applies only to UE5M3 software numeric paths")
    activation_mode = args.activation_mode
    if configuration in _D50_UE5M3_CONFIGURATIONS and activation_mode is None:
        activation_mode = "training_replay"
    if configuration in _CURRENT_UE5M3_CONFIGURATIONS:
        if activation_mode not in {None, "current_tensor"}:
            raise ValueError("UE5M3 TE-settings evaluation requires current_tensor D=1")
        activation_mode = "current_tensor"
    if activation_mode == "calibrated_frozen" and not args.calibration:
        raise ValueError("calibrated_frozen requires one or more --calibration inputs")
    if activation_mode != "calibrated_frozen" and args.calibration:
        raise ValueError("--calibration is valid only with calibrated_frozen")
    if args.batch_size != 1:
        raise ValueError("the reported held-out validation protocol requires batch_size=1")

    validation_paths, validation_identity, validation_records = token_dataset_identity(
        args.validation,
        tensor_key=args.tensor_key,
        expected_input_tokens=args.expected_input_tokens,
        expected_sequences=(None if args.allow_partial_data else 768),
    )
    calibration_paths: list[Path] = []
    calibration_identity: dict[str, Any] | None = None
    if args.calibration:
        calibration_paths, calibration_identity, calibration_records = token_dataset_identity(
            args.calibration,
            tensor_key=args.tensor_key,
            expected_input_tokens=args.expected_input_tokens,
            expected_sequences=(None if args.allow_partial_data else 64),
        )
        overlap = validation_records & calibration_records
        if overlap:
            raise ValueError(
                f"calibration and validation data overlap in {len(overlap)} token records"
            )

    model, load_provenance = load_hf_nemotron_h_checkpoint(
        args.checkpoint,
        hf_assets=args.hf_assets,
        hf_revision=args.hf_revision,
        device=device,
        local_files_only=args.local_files_only,
    )
    checkpoint_sha256 = load_provenance["checkpoint"]["sha256"]
    controller: FP4InferenceScalingController | None = None
    native_modules: tuple[tuple[str, torch.nn.Module], ...] = ()
    conversion_record: dict[str, Any] | None = None
    before_forward = None

    if configuration == "bf16":
        output_head = getattr(model, "lm_head", None)
        if type(output_head) is not torch.nn.Linear:
            raise TypeError("BF16 Nemotron-H evaluation requires an exact nn.Linear lm_head")
        checkpoint_parameter = output_head.weight
        model.lm_head = FP32OutputLinear.from_float(output_head)
        if model.lm_head.weight is not checkpoint_parameter:
            raise RuntimeError(
                "FP32 output-head conversion changed checkpoint parameter identity"
            )
        conversion_record = {
            "fp4_linear_count": 0,
            "fp32_output_head": "lm_head",
            "output_parameter_dtype": str(model.lm_head.weight.dtype).removeprefix("torch."),
            "output_compute_dtype": "float32",
            "checkpoint_parameter_identity_preserved": True,
        }
        numeric_provenance = learned_weight_bf16_numeric_path()
    elif configuration in _NATIVE_NVFP4_CONFIGURATIONS:
        variant = (
            NativeNVFP4Variant.TRANSFORMER_ENGINE_RECIPE
            if configuration == "native-nvfp4-te"
            else NativeNVFP4Variant.NO_RHT_ALL_LINEARS
        )
        checkpoint_output_head = getattr(model, "lm_head", None)
        if type(checkpoint_output_head) is not torch.nn.Linear:
            raise TypeError("Nemotron-H evaluation requires an exact nn.Linear lm_head")
        checkpoint_output_parameter = checkpoint_output_head.weight
        conversion = convert_native_nvfp4_nemotron_h(model, variant=variant)
        if not isinstance(model.lm_head, FP32OutputLinear):
            raise RuntimeError("native conversion did not install the FP32 output head")
        if model.lm_head.weight is not checkpoint_output_parameter:
            raise RuntimeError("native conversion changed output checkpoint identity")
        native_modules = _enable_native_nvfp4_alignment(
            model,
            expected_modules=len(conversion.fp4_linears),
        )
        conversion_record = {
            "fp4_linear_count": len(conversion.fp4_linears),
            "fp4_linear_names": [record.module_name for record in conversion.fp4_linears],
            "fp32_output_head": conversion.fp32_output_head,
            "bf16_final_linear_count": conversion.final_bf16_linears,
            "output_parameter_dtype": str(model.lm_head.weight.dtype).removeprefix("torch."),
            "output_compute_dtype": "float32",
            "checkpoint_parameter_identity_preserved": True,
            "linear_backend": conversion.native_backend,
            "variant": conversion.variant.value,
            "transformer_engine_version": conversion.transformer_engine_version,
            "effective_interval": conversion.current_tensor_interval,
        }
        numeric_provenance = None
    else:
        block_size = 32 if configuration == "ue5m3-proposed-b32" else 16
        checkpoint_output_head = getattr(model, "lm_head", None)
        if type(checkpoint_output_head) is not torch.nn.Linear:
            raise TypeError("Nemotron-H evaluation requires an exact nn.Linear lm_head")
        checkpoint_output_parameter = checkpoint_output_head.weight
        if configuration == "ue5m3-te-settings":
            conversion = convert_ue5m3_te_settings_nemotron_h(model)
            final_bf16_linears = conversion.final_bf16_linears
        else:
            recipe = replace(
                UE5M3Recipe.proposed(),
                name=f"proposed_ue5m3_b{block_size}_d50",
                block_size=block_size,
            )
            backend = (
                LinearBackend.TRITON_QUANT_TORCH
                if configuration == "ue5m3-torch-control"
                else LinearBackend.PROBE_MATCHED_TRITON
            )
            conversion = convert_reported_nemotron_h(
                model,
                recipe=recipe,
                backend=backend,
            )
            final_bf16_linears = 0
        if not isinstance(model.lm_head, FP32OutputLinear):
            raise RuntimeError("quantized conversion did not install the FP32 output head")
        if model.lm_head.weight is not checkpoint_output_parameter:
            raise RuntimeError("quantized conversion changed output checkpoint identity")
        conversion_record = {
            "fp4_linear_count": len(conversion.fp4_linears),
            "fp4_linear_names": [record.module_name for record in conversion.fp4_linears],
            "fp32_output_head": conversion.fp32_output_head,
            "bf16_final_linear_count": final_bf16_linears,
            "output_parameter_dtype": str(model.lm_head.weight.dtype).removeprefix("torch."),
            "output_compute_dtype": "float32",
            "checkpoint_parameter_identity_preserved": True,
            "linear_backend": conversion.linear_backend,
            "recipe": conversion.scale_state.recipe.to_dict(),
        }
        controller = FP4InferenceScalingController(
            model,
            activation_mode=activation_mode,
            checkpoint_identity=load_provenance["checkpoint"],
            replay_work_unit=(
                {"kind": "fixed_forward_batch", "size": 1}
                if activation_mode == "training_replay"
                else None
            ),
        )
        # The loader has completed and every module is already in eval mode.
        # Only now is fresh inference state created and each learned weight's
        # global amax sampled once and frozen.
        controller.reset_after_checkpoint_load()
        controller.calibrate_and_freeze_weights()
        if activation_mode == "calibrated_frozen":
            if calibration_identity is None:
                raise AssertionError("calibration identity was not constructed")
            _run_activation_calibration(
                model,
                controller,
                calibration_paths,
                identity=calibration_identity,
                tensor_key=args.tensor_key,
                device=device,
            )
        controller.begin_measurement(
            evaluation_order=(
                {
                    "order": "ordered local safetensors sources, then ascending row",
                    "seed": None,
                    "validation_data_sha256": validation_identity["sha256"],
                }
                if activation_mode == "training_replay"
                else None
            )
        )
        controller.reset_counters()
        if activation_mode == "training_replay":

            def advance_replay(input_ids: torch.Tensor, _forward_count: int) -> None:
                assert controller is not None
                controller.advance_training_replay_work_unit(
                    input_ids,
                    effective_token_count=input_ids.numel(),
                )

            before_forward = advance_replay
        numeric_provenance = None

    def progress(completed: int, total: int) -> None:
        if args.quiet:
            return
        print(f"validation sequences: {completed}/{total}", file=sys.stderr, flush=True)

    result = evaluate_validation(
        model,
        validation_paths,
        checkpoint_id=checkpoint_sha256,
        device=device,
        batch_size=args.batch_size,
        ce_chunk_tokens=args.ce_chunk_tokens,
        tensor_key=args.tensor_key,
        model_provenance={
            "load": load_provenance,
            "numeric_path_requested": args.numeric_path,
            "configuration": configuration,
            "conversion": conversion_record,
        },
        data_provenance={"identity": validation_identity},
        progress_callback=progress,
        before_forward_callback=before_forward,
        forward_kwargs={"use_cache": False, "return_dict": True},
        logit_path="chunked_lm_head",
    )
    if controller is not None:
        numeric_provenance = controller.provenance()
        numeric_provenance["configuration"] = configuration
    elif native_modules:
        numeric_provenance = _native_nvfp4_provenance(
            native_modules,
            configuration=configuration,
            expected_forwards=result["evaluation"]["forward_count"],
        )
    result["schema"] = EVALUATION_RUN_SCHEMA
    result["model"]["numeric_path"] = numeric_provenance
    result["calibration_data"] = calibration_identity
    result["result_sha256"] = _canonical_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--numeric-path", required=True, choices=_NUMERIC_PATHS)
    parser.add_argument("--activation-mode", choices=_ACTIVATION_MODES)
    parser.add_argument("--calibration", nargs="+", type=Path)
    parser.add_argument("--hf-assets", default=NEMOTRON_H_ASSET_REPOSITORY)
    parser.add_argument("--hf-revision", default=NEMOTRON_H_ASSET_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tensor-key", default="tokens")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--ce-chunk-tokens", default=256, type=int)
    parser.add_argument("--expected-input-tokens", default=8192, type=int)
    parser.add_argument(
        "--allow-partial-data",
        action="store_true",
        help="allow a smoke subset instead of 768 validation/64 calibration records",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    if not args.quiet:
        metrics = result["metrics"]
        print(
            f"wrote {output}: NLL={metrics['nll']:.10f}, "
            f"perplexity={metrics['perplexity']:.10f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
