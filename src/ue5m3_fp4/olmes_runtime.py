# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed numeric controls for the pinned public OLMES runner."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ue5m3_fp4.checkpoint import checkpoint_identity
from ue5m3_fp4.integrations.torchtitan.comparators import (
    NativeNVFP4Variant,
    convert_native_nvfp4_nemotron_h,
    convert_ue5m3_te_settings_nemotron_h,
)
from ue5m3_fp4.integrations.torchtitan.nemotron_h import (
    configure_nemotron_h_sdpa,
    verify_nemotron_h_sdpa,
)
from ue5m3_fp4.integrations.torchtitan.selection import (
    FP32OutputLinear,
    convert_reported_nemotron_h,
)
from ue5m3_fp4.nn.linear import LinearBackend
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.inference import FP4InferenceScalingController

RUNTIME_ENABLE_ENV = "UE5M3_PUBLIC_OLMES_RUNTIME"
RUNTIME_MODEL_ENV = "UE5M3_PUBLIC_OLMES_MODEL_DIRECTORY"
RUNTIME_RESULT_ENV = "UE5M3_PUBLIC_OLMES_RUNTIME_RESULT"
RUNTIME_NUMERIC_PATH_ENV = "UE5M3_OLMES_NUMERIC_PATH"
RUNTIME_REQUEST_MODE_ENV = "UE5M3_OLMES_REQUEST_MODE"
RUNTIME_FROZEN_VERIFIED_ENV = "UE5M3_OLMES_FROZEN_BUNDLE_VERIFIED"
RUNTIME_RESULT_SCHEMA = "ue5m3_fp4_public_olmes_runtime_v1"
PUBLIC_OLMES_BASE_REVISION = "8e2743734066b073c5d8498d1b8220f67a21a2d6"
PUBLIC_OLMES_BASE_TREE = "991940b9a2b37f8491ff29d1d22487b209fe750f"
HISTORICAL_OLMES_REVISION = "3d80ebb0f08706a5d2dd3fb0be72100735b5f5c6"
HISTORICAL_OLMES_TREE = "6403093f39b09e3dd6980bee7d60863a7714de8f"

OLMES_NUMERIC_PATHS = (
    "bf16",
    "ue5m3-proposed-b16",
    "ue5m3-proposed-b32",
    "ue5m3-torch-control",
    "ue5m3-te-settings",
    "native-nvfp4-te",
    "native-nvfp4-no-rht-all",
)
_D50_PATHS = {
    "ue5m3-proposed-b16",
    "ue5m3-proposed-b32",
    "ue5m3-torch-control",
}
_NATIVE_PATHS = {"native-nvfp4-te", "native-nvfp4-no-rht-all"}
_FP4_MODULE_METHODS = (
    "reset_fp4_inference_scaling",
    "configure_fp4_inference_scaling",
    "calibrate_and_freeze_fp4_inference_weight_scale",
    "begin_fp4_inference_measurement",
    "advance_fp4_inference_training_replay_step",
    "reset_fp4_inference_measurement_counters",
    "fp4_inference_scaling_report",
)


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _input_identity(input_ids: torch.Tensor) -> dict[str, Any]:
    if input_ids.dtype is not torch.long or input_ids.ndim != 2:
        raise TypeError("OLMES forwards require rank-two torch.long input_ids")
    canonical = input_ids.detach().to(device="cpu").contiguous()
    identity = {
        "dtype": "int64",
        "shape": list(canonical.shape),
        "bytes_sha256": hashlib.sha256(
            canonical.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    identity["sha256"] = _canonical_sha256(identity)
    return identity


def controlled_hflm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Disable request coalescing so OLMES forwards remain explicit."""

    if kwargs.get("logits_cache") not in {None, False}:
        raise RuntimeError("the public OLMES runtime rejects an enabled logits cache")
    controlled = dict(kwargs)
    controlled["logits_cache"] = False
    return controlled


def install_olmes_filename_compatibility(olmes_utils: Any | None = None) -> dict[str, Any]:
    """Restore the historical colon-free result/request naming behavior.

    The selected 146 likelihood tasks and Hugging Face model path are unchanged
    between the released public OLMES ancestor and the historical descendant.
    The descendant also replaced colons in generated filenames. Reproduce that
    storage-only behavior here so frozen request names remain compatible without
    depending on a non-public Git remote.
    """

    if olmes_utils is None:
        import oe_eval.utils as olmes_utils

    if getattr(olmes_utils, "_ue5m3_colon_free_names", False):
        raise RuntimeError("OLMES filename compatibility was installed twice")

    def sanitise_name(name: str) -> str:
        return name.replace(":", "-")

    def load_jsonl(file_name: str) -> list[Any]:
        with open(sanitise_name(file_name)) as file:
            return [json.loads(line.strip()) for line in file]

    def load_json(file_name: str) -> Any:
        with open(sanitise_name(file_name)) as file:
            return json.loads(file.read())

    def save_jsonl(file_name: str, data: Any) -> None:
        with open(sanitise_name(file_name), "w") as file:
            for item in data:
                file.write(json.dumps(item))
                file.write("\n")

    def save_json(file_name: str, data: Any) -> None:
        with open(sanitise_name(file_name), "w") as file:
            file.write(json.dumps(data))

    def task_file_name(
        output_dir: str,
        task_idx: int,
        task_name: str,
        file_name: str,
    ) -> str:
        return sanitise_name(
            os.path.join(output_dir, f"task-{task_idx:03d}-{task_name}-{file_name}")
        )

    olmes_utils.sanitise_name = sanitise_name
    olmes_utils.load_jsonl = load_jsonl
    olmes_utils.load_json = load_json
    olmes_utils.save_jsonl = save_jsonl
    olmes_utils.save_json = save_json
    olmes_utils.task_file_name = task_file_name
    olmes_utils._ue5m3_colon_free_names = True
    return {
        "public_base_revision": PUBLIC_OLMES_BASE_REVISION,
        "public_base_tree": PUBLIC_OLMES_BASE_TREE,
        "historical_revision": HISTORICAL_OLMES_REVISION,
        "historical_tree": HISTORICAL_OLMES_TREE,
        "colon_free_request_and_result_names": True,
    }


def install_fp32_output_head(model: torch.nn.Module) -> dict[str, Any]:
    """Install the paper's FP32-compute output projection without copying it."""

    output_head = getattr(model, "lm_head", None)
    if type(output_head) is not torch.nn.Linear:
        raise TypeError("OLMES requires an exact nn.Linear Nemotron-H lm_head")
    checkpoint_parameter = output_head.weight
    model.lm_head = FP32OutputLinear.from_float(output_head)
    if model.lm_head.weight is not checkpoint_parameter:
        raise RuntimeError("OLMES output-head conversion changed parameter identity")
    return {
        "module": "lm_head",
        "parameter_dtype": str(model.lm_head.weight.dtype).removeprefix("torch."),
        "compute_dtype": "float32",
        "checkpoint_parameter_identity_preserved": True,
    }


def _converted_output_record(
    model: torch.nn.Module,
    *,
    checkpoint_parameter: torch.nn.Parameter,
    output_name: str,
) -> dict[str, Any]:
    output_head = getattr(model, "lm_head", None)
    if not isinstance(output_head, FP32OutputLinear):
        raise TypeError("OLMES conversion did not install the FP32 output head")
    if output_head.weight is not checkpoint_parameter:
        raise RuntimeError("OLMES conversion changed output checkpoint identity")
    return {
        "module": output_name,
        "parameter_dtype": str(output_head.weight.dtype).removeprefix("torch."),
        "compute_dtype": "float32",
        "checkpoint_parameter_identity_preserved": True,
    }


class _OLMESD50Controller:
    """Cold D=50 lifecycle over one uninterrupted OLMES model-forward stream."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.modules = tuple(
            sorted(
                (
                    (name, module)
                    for name, module in model.named_modules()
                    if all(
                        callable(getattr(module, method, None))
                        for method in _FP4_MODULE_METHODS
                    )
                ),
                key=lambda item: item[0],
            )
        )
        if not self.modules:
            raise RuntimeError("converted OLMES model exposes no FP4 lifecycle modules")
        if any(module.training for module in model.modules()):
            raise RuntimeError("D=50 setup requires model.eval()")
        for _, module in self.modules:
            module.reset_fp4_inference_scaling()
        try:
            for _, module in self.modules:
                module.configure_fp4_inference_scaling("training_replay")
        except Exception:
            for _, module in self.modules:
                module.reset_fp4_inference_scaling()
            raise
        for _, module in self.modules:
            module.calibrate_and_freeze_fp4_inference_weight_scale()
            module.begin_fp4_inference_measurement()
            module.reset_fp4_inference_measurement_counters()
        self.logical_step = 0

    def advance(self) -> int:
        next_step = self.logical_step + 1
        if next_step > 1:
            for name, module in self.modules:
                replay = module.fp4_inference_scaling_report()["training_replay"]
                if replay["last_consumed_step"] != self.logical_step:
                    raise RuntimeError(
                        f"FP4 module {name} did not consume D=50 step {self.logical_step}"
                    )
        for _, module in self.modules:
            module.advance_fp4_inference_training_replay_step(next_step)
        self.logical_step = next_step
        return next_step

    def provenance(self, expected_forwards: int) -> dict[str, Any]:
        if self.logical_step != expected_forwards:
            raise RuntimeError("D=50 logical counter differs from OLMES forwards")
        expected_refreshes = (expected_forwards + 49) // 50
        reports = {name: module.fp4_inference_scaling_report() for name, module in self.modules}
        for name, report in reports.items():
            replay = report["training_replay"]
            counters = report["counters"]
            if (
                replay["logical_step"] != expected_forwards
                or replay["last_consumed_step"] != expected_forwards
                or counters["activation_replay_refreshes"] != expected_refreshes
                or counters["activation_replay_reuses"]
                != expected_forwards - expected_refreshes
                or counters["weight_frozen_resolutions"] != expected_forwards
            ):
                raise RuntimeError(f"FP4 module {name} has incomplete D=50 evidence")
        return {
            "mode": "training_replay",
            "interval": 50,
            "initial_cache": "cold_after_checkpoint_load",
            "work_unit": "one top-level OLMES model forward",
            "logical_work_units": expected_forwards,
            "counter_scope": "one entire OLMES process",
            "reset_between_tasks": False,
            "module_count": len(reports),
            "module_reports": reports,
            "module_reports_sha256": _canonical_sha256(reports),
        }


def _enable_native_alignment(
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
            f"native OLMES conversion exposed {len(modules)} modules; "
            f"expected {expected_modules}"
        )
    with torch.no_grad():
        for name, module in modules:
            module.enable_nvfp4_inference_alignment()
            report = module.native_nvfp4_report()
            if (
                report.get("tensor_scaling") != "current_tensor"
                or report.get("effective_interval") != 1
                or report.get("delayed_amax_fields_applied") is not False
                or report.get("alignment", {}).get("enabled") is not True
                or any(report.get("alignment", {}).get("counters", {}).values())
            ):
                raise RuntimeError(f"native NVFP4 module {name} has invalid fresh state")
    return modules


class _RuntimeSession:
    def __init__(
        self,
        expected_model: Path,
        result_path: Path,
        *,
        configuration: str,
        request_mode: str,
    ) -> None:
        if configuration not in OLMES_NUMERIC_PATHS:
            raise ValueError(f"unsupported OLMES numeric path {configuration!r}")
        if request_mode not in {"public_task_rebuild", "frozen_request_archive"}:
            raise ValueError(f"unsupported OLMES request mode {request_mode!r}")
        if (
            request_mode == "frozen_request_archive"
            and os.environ.get(RUNTIME_FROZEN_VERIFIED_ENV) != "1"
        ):
            raise ValueError("frozen OLMES mode requires a hash-verified request bundle")
        self.expected_model = expected_model
        self.result_path = result_path
        self.configuration = configuration
        self.request_mode = request_mode
        self.checkpoint: dict[str, Any] | None = None
        self.attach_count = 0
        self.forward_hashes: list[str] = []
        self.record: dict[str, Any] | None = None
        self.controller: _OLMESD50Controller | FP4InferenceScalingController | None = None
        self.native_modules: tuple[tuple[str, torch.nn.Module], ...] = ()

    def _convert(self, model: torch.nn.Module) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.checkpoint is None:
            raise RuntimeError("OLMES checkpoint identity was not recorded")
        if self.configuration == "bf16":
            output = install_fp32_output_head(model)
            return output, {
                "fp4_linear_count": 0,
                "bf16_final_linear_count": 112,
                "numeric_backend": "learned_weight_bf16",
            }

        output_head = getattr(model, "lm_head", None)
        if type(output_head) is not torch.nn.Linear:
            raise TypeError("quantized OLMES requires an exact nn.Linear lm_head")
        checkpoint_output_parameter = output_head.weight
        if self.configuration in _NATIVE_PATHS:
            variant = (
                NativeNVFP4Variant.TRANSFORMER_ENGINE_RECIPE
                if self.configuration == "native-nvfp4-te"
                else NativeNVFP4Variant.NO_RHT_ALL_LINEARS
            )
            conversion = convert_native_nvfp4_nemotron_h(model, variant=variant)
            self.native_modules = _enable_native_alignment(
                model,
                expected_modules=len(conversion.fp4_linears),
            )
            record = {
                "fp4_linear_count": len(conversion.fp4_linears),
                "bf16_final_linear_count": conversion.final_bf16_linears,
                "fp32_output_head": conversion.fp32_output_head,
                "numeric_backend": conversion.native_backend,
                "variant": conversion.variant.value,
                "transformer_engine_version": conversion.transformer_engine_version,
                "effective_interval": conversion.current_tensor_interval,
            }
        elif self.configuration == "ue5m3-te-settings":
            conversion = convert_ue5m3_te_settings_nemotron_h(model)
            record = {
                "fp4_linear_count": len(conversion.fp4_linears),
                "bf16_final_linear_count": conversion.final_bf16_linears,
                "fp32_output_head": conversion.fp32_output_head,
                "numeric_backend": conversion.linear_backend,
                "effective_interval": conversion.current_tensor_interval,
                "recipe": conversion.scale_state.recipe.to_dict(),
            }
            self.controller = FP4InferenceScalingController(
                model,
                activation_mode="current_tensor",
                checkpoint_identity=self.checkpoint,
            )
            self.controller.reset_after_checkpoint_load()
            self.controller.calibrate_and_freeze_weights()
            self.controller.begin_measurement()
            self.controller.reset_counters()
        else:
            block_size = 32 if self.configuration == "ue5m3-proposed-b32" else 16
            recipe = replace(
                UE5M3Recipe.proposed(),
                name=f"proposed_ue5m3_b{block_size}_d50",
                block_size=block_size,
            )
            backend = (
                LinearBackend.TRITON_QUANT_TORCH
                if self.configuration == "ue5m3-torch-control"
                else LinearBackend.PROBE_MATCHED_TRITON
            )
            conversion = convert_reported_nemotron_h(
                model,
                recipe=recipe,
                backend=backend,
            )
            record = {
                "fp4_linear_count": len(conversion.fp4_linears),
                "bf16_final_linear_count": 0,
                "fp32_output_head": conversion.fp32_output_head,
                "numeric_backend": conversion.linear_backend,
                "effective_interval": 50,
                "recipe": conversion.scale_state.recipe.to_dict(),
            }
            self.controller = _OLMESD50Controller(model)
        output = _converted_output_record(
            model,
            checkpoint_parameter=checkpoint_output_parameter,
            output_name=record["fp32_output_head"],
        )
        return output, record

    def attach(self, model: torch.nn.Module, hflm: Any, pretrained: Any) -> None:
        if self.attach_count:
            raise RuntimeError("the public OLMES runtime permits one model instance")
        if not isinstance(pretrained, str):
            raise TypeError("the OLMES model source must be a local directory string")
        if Path(pretrained).expanduser().resolve() != self.expected_model:
            raise RuntimeError("OLMES loaded a model outside the requested HF directory")
        if getattr(hflm, "logits_cache", None) is not False:
            raise RuntimeError("OLMES did not disable its logits cache")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("HFLM did not expose a torch.nn.Module")
        model.eval()
        if any(module.training for module in model.modules()):
            raise RuntimeError("a loaded OLMES module remains in training mode")

        attention = configure_nemotron_h_sdpa(model, cudnn_enabled=False)
        verified_mixers = verify_nemotron_h_sdpa(model)
        if tuple(attention["attention_mixers"]) != verified_mixers:
            raise RuntimeError("configured and verified OLMES attention paths disagree")
        self.checkpoint = checkpoint_identity(self.expected_model)
        output, conversion = self._convert(model)

        self.attach_count += 1
        self.record = {
            "schema": RUNTIME_RESULT_SCHEMA,
            "configuration": self.configuration,
            "fp4_quantization_applied": self.configuration != "bf16",
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_eval": True,
            "logits_cache": False,
            "checkpoint": self.checkpoint,
            "attention": attention,
            "attention_verified": True,
            "output_projection": output,
            "conversion": conversion,
            "olmes_compatibility": {
                "public_base_revision": PUBLIC_OLMES_BASE_REVISION,
                "public_base_tree": PUBLIC_OLMES_BASE_TREE,
                "historical_revision": HISTORICAL_OLMES_REVISION,
                "historical_tree": HISTORICAL_OLMES_TREE,
                "colon_free_request_and_result_names": True,
                "selected_likelihood_tasks_use_chat_format": False,
            },
        }

        def before_forward(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            if module.training:
                raise RuntimeError("the OLMES model entered training mode")
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("could not identify OLMES input_ids")
            if isinstance(self.controller, _OLMESD50Controller):
                logical_step = self.controller.advance()
                if logical_step != len(self.forward_hashes) + 1:
                    raise RuntimeError("D=50 and OLMES forward counters diverged")
            identity = _input_identity(input_ids)
            self.forward_hashes.append(identity["sha256"])

        model.register_forward_pre_hook(before_forward, with_kwargs=True)

    def _numeric_provenance(self, forward_count: int) -> dict[str, Any]:
        if self.configuration == "bf16":
            return {
                "numeric_path": "learned_weight_bf16",
                "fp4_quantization_applied": False,
                "inference_scaling_protocol": None,
            }
        if isinstance(self.controller, _OLMESD50Controller):
            protocol = self.controller.provenance(forward_count)
        elif isinstance(self.controller, FP4InferenceScalingController):
            protocol = self.controller.provenance()
            reports = protocol["inference_scaling_protocol"]["module_reports"]
            for name, report in reports.items():
                counters = report["counters"]
                if (
                    report.get("activation_mode") != "current_tensor"
                    or counters["activation_current_resolutions"] != forward_count
                    or counters["weight_frozen_resolutions"] != forward_count
                ):
                    raise RuntimeError(
                        f"FP4 module {name} has incomplete current-tensor evidence"
                    )
            return protocol
        elif self.native_modules:
            reports = {
                name: module.native_nvfp4_report() for name, module in self.native_modules
            }
            for name, report in reports.items():
                if (
                    report.get("tensor_scaling") != "current_tensor"
                    or report.get("effective_interval") != 1
                    or report.get("delayed_amax_fields_applied") is not False
                    or report.get("alignment", {}).get("counters", {}).get("forward_count")
                    != forward_count
                ):
                    raise RuntimeError(
                        f"native NVFP4 module {name} has incomplete forward evidence"
                    )
            protocol = {
                "mode": "current_tensor",
                "effective_interval": 1,
                "checkpoint_cache_inherited": False,
                "delayed_amax_fields_applied": False,
                "module_count": len(reports),
                "module_reports": reports,
                "module_reports_sha256": _canonical_sha256(reports),
            }
        else:
            raise RuntimeError("quantized OLMES configured no numeric controller")
        return {
            "numeric_path": (
                "native_nvfp4_transformer_engine"
                if self.native_modules
                else "quantized_ue5m3_fp4"
            ),
            "fp4_quantization_applied": True,
            "inference_scaling_protocol": protocol,
        }

    def finalize(self) -> None:
        # The launch process also imports sitecustomize before spawning the
        # actual evaluator.  Only the evaluator attaches a model and emits an
        # attestation.
        if self.attach_count == 0:
            return
        if self.attach_count != 1 or self.record is None:
            raise RuntimeError("the public OLMES runtime attachment is inconsistent")
        forward_count = len(self.forward_hashes)
        if forward_count <= 0:
            raise RuntimeError("OLMES completed without a measured model forward")
        if self.request_mode == "frozen_request_archive" and forward_count != 46_136:
            raise RuntimeError(
                "frozen OLMES replay produced a different top-level forward count"
            )
        document = dict(self.record)
        document["numeric_path"] = self._numeric_provenance(forward_count)
        document["request_identity"] = {
            "mode": self.request_mode,
            "byte_identical_frozen_request_archive": (
                self.request_mode == "frozen_request_archive"
            ),
            "frozen_manifest_sha256": (
                "b7cd708300b7b63edd45e4d973de7195b2c98384f1a9b0773f49c5a8d0e47898"
                if self.request_mode == "frozen_request_archive"
                else None
            ),
            "frozen_archive_sha256": (
                "0bf27af57eb1bb1b98872c4af12d419498652d935a6b745cc7ec4ecdb32d7483"
                if self.request_mode == "frozen_request_archive"
                else None
            ),
            "top_level_model_forward_count": forward_count,
            "ordered_model_forward_inputs_sha256": _canonical_sha256(self.forward_hashes),
        }
        document["runtime_attested"] = True
        _write_json_atomic(self.result_path, document)


_INSTALLED = False


def install_from_environment() -> None:
    """Patch HFLM before OLMES imports its verbose adapter."""

    global _INSTALLED
    if os.environ.get(RUNTIME_ENABLE_ENV) != "1":
        return
    if _INSTALLED:
        raise RuntimeError("the public OLMES runtime was installed twice")
    _INSTALLED = True
    expected_model = Path(os.environ[RUNTIME_MODEL_ENV]).expanduser().resolve()
    result_path = Path(os.environ[RUNTIME_RESULT_ENV]).expanduser().resolve()
    configuration = os.environ.get(RUNTIME_NUMERIC_PATH_ENV, "bf16")
    request_mode = os.environ.get(RUNTIME_REQUEST_MODE_ENV, "public_task_rebuild")
    install_olmes_filename_compatibility()
    session = _RuntimeSession(
        expected_model,
        result_path,
        configuration=configuration,
        request_mode=request_mode,
    )

    from lm_eval.models.huggingface import HFLM

    original_init = HFLM.__init__

    def wrapped_init(
        instance: Any,
        pretrained: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        original_init(
            instance,
            pretrained,
            *args,
            **controlled_hflm_kwargs(kwargs),
        )
        session.attach(instance._model, instance, pretrained)

    HFLM.__init__ = wrapped_init
    atexit.register(session.finalize)


__all__ = [
    "HISTORICAL_OLMES_REVISION",
    "HISTORICAL_OLMES_TREE",
    "OLMES_NUMERIC_PATHS",
    "PUBLIC_OLMES_BASE_REVISION",
    "PUBLIC_OLMES_BASE_TREE",
    "RUNTIME_ENABLE_ENV",
    "RUNTIME_FROZEN_VERIFIED_ENV",
    "RUNTIME_MODEL_ENV",
    "RUNTIME_NUMERIC_PATH_ENV",
    "RUNTIME_REQUEST_MODE_ENV",
    "RUNTIME_RESULT_ENV",
    "RUNTIME_RESULT_SCHEMA",
    "controlled_hflm_kwargs",
    "install_fp32_output_head",
    "install_from_environment",
    "install_olmes_filename_compatibility",
]
