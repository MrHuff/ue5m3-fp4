#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Deterministic scale lifecycle for post-training FP4 evaluation.

Training checkpoints contain learned weights, not an inference-time scale
protocol. This controller therefore starts only after checkpoint loading. It
creates fresh inference-only state, never consults a process-local training
cache, calibrates each loaded weight tensor once, and freezes the resulting
weight reference. Activation scaling is then either computed from the current
operand for every measured forward or calibrated as the maximum over an
explicitly identified batch sequence and frozen before measurement.

The existing BF16 evaluation path remains a different numeric path: it evaluates
the learned checkpoint weights without applying FP4 fake quantization.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch

EVALUATION_NUMERIC_PATH_SCHEMA = "ue5m3_fp4_evaluation_numeric_path_v1"
FP4_INFERENCE_SCALING_PROTOCOL_SCHEMA = "ue5m3_fp4_inference_scaling_v1"
FP4_INFERENCE_ACTIVATION_MODES = frozenset(
    {"current_tensor", "calibrated_frozen", "training_replay"}
)
FP4_INFERENCE_TRAINING_REPLAY_INTERVAL = 50
FP4_INFERENCE_TRAINING_REPLAY_EFFECTIVE_TOKENS = 8192

_REQUIRED_MODULE_METHODS = (
    "reset_fp4_inference_scaling",
    "configure_fp4_inference_scaling",
    "calibrate_and_freeze_fp4_inference_weight_scale",
    "begin_fp4_inference_activation_calibration",
    "freeze_fp4_inference_activation_scales",
    "begin_fp4_inference_measurement",
    "advance_fp4_inference_training_replay_step",
    "reset_fp4_inference_measurement_counters",
    "fp4_inference_scaling_report",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be finite, JSON-serializable data") from error


def _tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Calibration batches must be torch.Tensor instances")
    canonical = tensor.detach().to(device="cpu").contiguous()
    byte_view = canonical.view(torch.uint8)
    payload = {
        "dtype": str(canonical.dtype).removeprefix("torch."),
        "shape": list(canonical.shape),
        "bytes_sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def learned_weight_bf16_numeric_path() -> dict[str, Any]:
    """Provenance label for the existing, non-quantized BF16 evaluator."""

    return {
        "schema": EVALUATION_NUMERIC_PATH_SCHEMA,
        "numeric_path": "learned_weight_bf16",
        "learned_weight_dtype": "bfloat16",
        "fp4_quantization_applied": False,
        "inference_scaling_protocol": None,
    }


def _is_fp4_inference_module(module: torch.nn.Module) -> bool:
    return all(callable(getattr(module, name, None)) for name in _REQUIRED_MODULE_METHODS)


class FP4InferenceScalingController:
    """Coordinate and attest deterministic FP4 scales across a model.

    Lifecycle::

        controller.reset_after_checkpoint_load()
        controller.calibrate_and_freeze_weights()
        # current_tensor:
        controller.begin_measurement()

        # calibrated_frozen instead:
        controller.begin_activation_calibration(calibration_identity)
        for input_ids in calibration_batches:
            controller.record_activation_calibration_batch(input_ids)
            model(input_ids)
        controller.freeze_activation_scales()
        controller.begin_measurement()

    No method reads a training step, and no measured forward is allowed before
    the selected lifecycle reaches ``measuring``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        activation_mode: str,
        checkpoint_identity: Mapping[str, Any],
        replay_work_unit: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if activation_mode not in FP4_INFERENCE_ACTIVATION_MODES:
            valid = ", ".join(sorted(FP4_INFERENCE_ACTIVATION_MODES))
            raise ValueError(
                f"Unknown activation_mode {activation_mode!r}; expected one of {valid}"
            )
        if not isinstance(checkpoint_identity, Mapping) or not checkpoint_identity:
            raise ValueError("checkpoint_identity must be a non-empty mapping")
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if _is_fp4_inference_module(module)
        ]
        if not modules:
            raise ValueError(
                "The model contains no modules implementing the FP4 inference scaling lifecycle"
            )
        if any(not name for name, _ in modules):
            raise ValueError("The model root cannot itself be the FP4 linear module")

        self.model = model
        self.activation_mode = activation_mode
        self.checkpoint_identity = _json_copy(
            dict(checkpoint_identity), label="checkpoint_identity"
        )
        if activation_mode == "training_replay":
            self.replay_work_unit = self._validate_replay_work_unit(replay_work_unit)
        elif replay_work_unit is not None:
            raise ValueError("replay_work_unit is only valid for training_replay mode")
        else:
            self.replay_work_unit = None
        self.modules = tuple(sorted(modules, key=lambda item: item[0]))
        self.phase = "created"
        self.reset_count = 0
        self.calibration_identity: dict[str, Any] | None = None
        self.calibration_batches: list[dict[str, Any]] = []
        self.calibration_token_count = 0
        self.evaluation_order: dict[str, Any] | None = None
        self.training_replay_work_units: list[dict[str, Any]] = []

    @staticmethod
    def _validate_replay_work_unit(
        value: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            # Preserve the public configuration-validation exception contract.
            raise ValueError(  # noqa: TRY004
                "training_replay requires an explicit replay_work_unit mapping"
            )
        result = _json_copy(dict(value), label="replay_work_unit")
        kind = result.get("kind")
        size = result.get("size")
        if kind not in {"effective_tokens", "fixed_forward_batch"}:
            raise ValueError(
                "replay_work_unit.kind must be effective_tokens or fixed_forward_batch"
            )
        if type(size) is not int or size <= 0:
            raise ValueError("replay_work_unit.size must be a positive integer")
        if (
            kind == "effective_tokens"
            and size != FP4_INFERENCE_TRAINING_REPLAY_EFFECTIVE_TOKENS
        ):
            raise ValueError(
                "The token-based training replay uses exactly 8192 nonpadding/"
                "effective tokens per logical step; use fixed_forward_batch when "
                "the evaluator cannot provide that unit"
            )
        return result

    def _require_phase(self, expected: str) -> None:
        if self.phase != expected:
            raise RuntimeError(
                f"FP4 inference controller phase is {self.phase!r}, expected {expected!r}"
            )

    def _require_deterministic_runtime(self) -> None:
        training_modules = [
            name or "<root>" for name, module in self.model.named_modules() if module.training
        ]
        if training_modules:
            raise RuntimeError(
                "FP4 inference requires every module in eval mode; training "
                f"modules: {', '.join(training_modules[:5])}"
            )
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise RuntimeError(
                "The deterministic FP4 inference scaling protocol requires one process"
            )

    def reset_after_checkpoint_load(self) -> dict[str, Any]:
        """Reset all scale state; safe to call again before a new evaluation."""

        self._require_deterministic_runtime()
        resets: dict[str, Any] = {}
        for name, module in self.modules:
            resets[name] = module.reset_fp4_inference_scaling()
        try:
            for _, module in self.modules:
                module.configure_fp4_inference_scaling(self.activation_mode)
        except Exception:
            # Do not leave a partially configured model when one module fails
            # deterministic-rounding or runtime validation.
            for _, module in self.modules:
                module.reset_fp4_inference_scaling()
            raise
        self.reset_count += 1
        self.calibration_identity = None
        self.calibration_batches = []
        self.calibration_token_count = 0
        self.evaluation_order = None
        self.training_replay_work_units = []
        self.phase = "reset"
        return resets

    def calibrate_and_freeze_weights(self) -> dict[str, Any]:
        """Calibrate loaded weights exactly once per explicit reset."""

        self._require_deterministic_runtime()
        self._require_phase("reset")
        records = {
            name: module.calibrate_and_freeze_fp4_inference_weight_scale()
            for name, module in self.modules
        }
        self.phase = "weights_frozen"
        return records

    def begin_activation_calibration(
        self,
        calibration_identity: Mapping[str, Any],
    ) -> None:
        self._require_deterministic_runtime()
        self._require_phase("weights_frozen")
        if self.activation_mode != "calibrated_frozen":
            raise RuntimeError("Activation calibration is not used by current_tensor mode")
        if not isinstance(calibration_identity, Mapping) or not calibration_identity:
            raise ValueError("calibration_identity must be a non-empty mapping")
        self.calibration_identity = _json_copy(
            dict(calibration_identity), label="calibration_identity"
        )
        self.calibration_batches = []
        self.calibration_token_count = 0
        for _, module in self.modules:
            module.begin_fp4_inference_activation_calibration()
        self.phase = "activation_calibration"

    def record_activation_calibration_batch(
        self,
        input_ids: torch.Tensor,
        *,
        token_count: int | None = None,
    ) -> str:
        """Record exact ordered input identity immediately before its forward."""

        self._require_deterministic_runtime()
        self._require_phase("activation_calibration")
        identity = _tensor_identity(input_ids)
        if token_count is None:
            token_count = input_ids.numel()
        if type(token_count) is not int or token_count <= 0:
            raise ValueError("token_count must be a positive integer")
        identity["index"] = len(self.calibration_batches)
        identity["token_count"] = token_count
        identity["record_sha256"] = _canonical_sha256(identity)
        self.calibration_batches.append(identity)
        self.calibration_token_count += token_count
        return identity["record_sha256"]

    def freeze_activation_scales(self) -> dict[str, Any]:
        self._require_deterministic_runtime()
        self._require_phase("activation_calibration")
        if not self.calibration_batches:
            raise RuntimeError("No activation calibration batches were recorded")
        records: dict[str, Any] = {}
        for name, module in self.modules:
            record = module.freeze_fp4_inference_activation_scales()
            observations = int(record.get("forward.x", {}).get("observations", 0))
            if observations < len(self.calibration_batches):
                raise RuntimeError(
                    f"FP4 module {name!r} observed {observations} activation "
                    f"operands for {len(self.calibration_batches)} recorded batches"
                )
            records[name] = record
        self.phase = "activations_frozen"
        return records

    def begin_measurement(
        self,
        *,
        evaluation_order: Mapping[str, Any] | None = None,
    ) -> None:
        self._require_deterministic_runtime()
        expected = (
            "activations_frozen"
            if self.activation_mode == "calibrated_frozen"
            else "weights_frozen"
        )
        self._require_phase(expected)
        if evaluation_order is None:
            if self.activation_mode == "training_replay":
                raise ValueError(
                    "training_replay requires explicit evaluation_order provenance"
                )
            self.evaluation_order = None
        else:
            record = _json_copy(dict(evaluation_order), label="evaluation_order")
            if not isinstance(record.get("order"), str) or not record["order"].strip():
                raise ValueError("evaluation_order.order must be a non-empty string")
            if "seed" not in record or (
                record["seed"] is not None and type(record["seed"]) is not int
            ):
                raise ValueError("evaluation_order.seed must be an integer or null")
            self.evaluation_order = record
        for _, module in self.modules:
            module.begin_fp4_inference_measurement()
        self.phase = "measuring"

    def advance_training_replay_work_unit(
        self,
        input_ids: torch.Tensor,
        *,
        effective_token_count: int,
    ) -> str:
        """Advance the cold replay counter once, immediately before model work."""

        self._require_deterministic_runtime()
        self._require_phase("measuring")
        if self.activation_mode != "training_replay" or self.replay_work_unit is None:
            raise RuntimeError("Work-unit advancement is only valid for training_replay mode")
        if type(effective_token_count) is not int or effective_token_count <= 0:
            raise ValueError("effective_token_count must be a positive integer")
        kind = self.replay_work_unit["kind"]
        size = int(self.replay_work_unit["size"])
        if kind == "effective_tokens" and effective_token_count != size:
            raise ValueError(
                f"Expected exactly {size} effective tokens, got {effective_token_count}"
            )
        if kind == "fixed_forward_batch" and (
            input_ids.ndim < 1 or int(input_ids.shape[0]) != size
        ):
            raise ValueError(
                f"Expected fixed forward batch size {size}, got shape {tuple(input_ids.shape)}"
            )

        logical_step = len(self.training_replay_work_units) + 1
        if logical_step > 1:
            for name, module in self.modules:
                replay = module.fp4_inference_scaling_report()["training_replay"]
                if replay["last_consumed_step"] != logical_step - 1:
                    raise RuntimeError(
                        f"FP4 module {name!r} did not consume replay step {logical_step - 1}"
                    )
        tensor_identity = _tensor_identity(input_ids)
        record = {
            "logical_step": logical_step,
            "effective_token_count": effective_token_count,
            "input": tensor_identity,
        }
        record["record_sha256"] = _canonical_sha256(record)
        for _, module in self.modules:
            module.advance_fp4_inference_training_replay_step(logical_step)
        self.training_replay_work_units.append(record)
        return record["record_sha256"]

    def assert_ready_for_measurement(self) -> None:
        self._require_deterministic_runtime()
        self._require_phase("measuring")
        for name, module in self.modules:
            report = module.fp4_inference_scaling_report()
            if report.get("phase") != "measuring" or not report.get("active"):
                raise RuntimeError(f"FP4 module {name!r} is not ready for measurement")

    def reset_counters(self) -> None:
        """Reset only resolver counters; frozen scales and lifecycle stay intact."""

        self.assert_ready_for_measurement()
        for _, module in self.modules:
            module.reset_fp4_inference_measurement_counters()

    def counters(self) -> dict[str, dict[str, int]]:
        return {
            name: module.fp4_inference_scaling_report()["counters"]
            for name, module in self.modules
        }

    def provenance(self) -> dict[str, Any]:
        """Return JSON provenance suitable for validation/downstream results."""

        self.assert_ready_for_measurement()
        module_reports = {
            name: module.fp4_inference_scaling_report() for name, module in self.modules
        }
        frozen_references = {
            name: report["frozen_global_amax"] for name, report in module_reports.items()
        }
        if self.activation_mode == "current_tensor":
            activation_calibration = None
            activation_policy = {
                "mode": "current_tensor",
                "definition": "amax of the current activation operand",
                "cache_reuse": False,
            }
            training_replay = None
        elif self.activation_mode == "calibrated_frozen":
            if self.calibration_identity is None:
                raise RuntimeError("Missing calibrated activation-set identity")
            ordered_records = [record["record_sha256"] for record in self.calibration_batches]
            activation_calibration = {
                "identity": self.calibration_identity,
                "identity_sha256": _canonical_sha256(self.calibration_identity),
                "batch_count": len(self.calibration_batches),
                "token_count": self.calibration_token_count,
                "ordered_batch_record_sha256": ordered_records,
                "ordered_batches_sha256": _canonical_sha256(ordered_records),
                "aggregation": "maximum_amax_over_calibration_forwards",
                "calibration_execution_scaling": ("current_tensor_per_operand_while_observing"),
            }
            activation_policy = {
                "mode": "calibrated_frozen",
                "definition": "maximum amax over the identified calibration batches",
                "cache_reuse_during_measurement": True,
            }
            training_replay = None
        else:
            if self.evaluation_order is None:
                raise RuntimeError("Missing training-replay evaluation order")
            if not self.training_replay_work_units:
                raise RuntimeError("No training-replay work units were measured")
            module_replay = {
                name: report["training_replay"] for name, report in module_reports.items()
            }
            final_step = len(self.training_replay_work_units)
            for name, replay in module_replay.items():
                if (
                    replay["logical_step"] != final_step
                    or replay["last_consumed_step"] != final_step
                ):
                    raise RuntimeError(
                        f"FP4 module {name!r} did not consume final replay step {final_step}"
                    )
            work_unit_hashes = [
                record["record_sha256"] for record in self.training_replay_work_units
            ]
            refresh_traces = {
                name: replay["refresh_trace"] for name, replay in module_replay.items()
            }
            final_caches = {name: replay["cache"] for name, replay in module_replay.items()}
            activation_calibration = None
            activation_policy = {
                "mode": "training_replay",
                "definition": (
                    "cold delayed-amax replay refreshed at logical steps 1, 51, 101, ..."
                ),
                "cache_reuse": True,
                "order_sensitive": True,
            }
            training_replay = {
                "interval": FP4_INFERENCE_TRAINING_REPLAY_INTERVAL,
                "initial_logical_step": 0,
                "checkpoint_step_used_as_initial_state": False,
                "refresh_rule": "step == 1 or step - last_refresh_step >= 50",
                "work_unit": self.replay_work_unit,
                "work_unit_sha256": _canonical_sha256(self.replay_work_unit),
                "evaluation_order": self.evaluation_order,
                "evaluation_order_sha256": _canonical_sha256(self.evaluation_order),
                "logical_step_count": len(work_unit_hashes),
                "ordered_work_unit_record_sha256": work_unit_hashes,
                "ordered_work_units_sha256": _canonical_sha256(work_unit_hashes),
                "refresh_traces_sha256": _canonical_sha256(refresh_traces),
                "final_caches_sha256": _canonical_sha256(final_caches),
            }

        protocol = {
            "schema": FP4_INFERENCE_SCALING_PROTOCOL_SCHEMA,
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_identity_sha256": _canonical_sha256(self.checkpoint_identity),
            "post_load_reset_count": self.reset_count,
            "training_step_or_cache_inherited": False,
            "distributed_scope": "single_process",
            "weight_policy": {
                "mode": "post_load_calibrated_frozen",
                "definition": "amax sampled once from each loaded learned weight tensor",
                "updates_during_measurement": False,
            },
            "activation_policy": activation_policy,
            "activation_calibration": activation_calibration,
            "training_replay": training_replay,
            "module_count": len(module_reports),
            "module_reports": module_reports,
            "module_reports_sha256": _canonical_sha256(module_reports),
            "frozen_global_amax_sha256": _canonical_sha256(frozen_references),
        }
        format_records = sorted(
            {
                json.dumps(
                    report["format"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for report in module_reports.values()
            }
        )
        formats = [json.loads(record) for record in format_records]
        scale_types = {record["scale_type"] for record in formats}
        gemm_output_models = {record.get("gemm_output_model") for record in formats}
        native_hardware_values = {record.get("native_hardware") for record in formats}
        if None in gemm_output_models or len(gemm_output_models) != 1:
            raise RuntimeError("FP4 modules must report one GEMM output model")
        if native_hardware_values != {False}:
            raise RuntimeError(
                "The software inference controller requires native_hardware=false"
            )
        gemm_output_model = next(iter(gemm_output_models))
        torch_matmul_policies = sorted(
            {
                json.dumps(
                    record.get("torch_matmul_policy"),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for record in formats
            }
        )
        if "null" in torch_matmul_policies:
            raise RuntimeError("FP4 modules must attest the Torch matmul policy")
        torch_matmul_policies = [json.loads(record) for record in torch_matmul_policies]
        if scale_types in ({"E5M3"}, {"UE5M3"}):
            if gemm_output_model == (
                "encoded_operand_k64_issue_rz_bf16_gemm_final_snap_1_over_1024"
            ):
                numeric_path = "quantized_ue5m3_fp4_probe_matched_k64_issue_rz"
            elif gemm_output_model == "encoded_operand_torch_fp32_matmul":
                numeric_path = "quantized_ue5m3_fp4_encoded_torch_fp32"
            else:
                numeric_path = "quantized_ue5m3_fp4_decoded_torch"
        elif scale_types == {"E4M3"}:
            numeric_path = "quantized_nvfp4_custom_emulator_decoded_torch"
        else:
            numeric_path = "quantized_custom_fp4_mixed_format_decoded_torch"
        return {
            "schema": EVALUATION_NUMERIC_PATH_SCHEMA,
            "numeric_path": numeric_path,
            "gemm_output_model": gemm_output_model,
            "torch_matmul_policies": torch_matmul_policies,
            "torch_matmul_policies_sha256": _canonical_sha256(torch_matmul_policies),
            "native_hardware": False,
            "resolved_formats": formats,
            "resolved_formats_sha256": _canonical_sha256(formats),
            "learned_weight_dtype": sorted(
                {
                    str(parameter.dtype).removeprefix("torch.")
                    for parameter in self.model.parameters()
                    if parameter.is_floating_point()
                }
            ),
            "fp4_quantization_applied": True,
            "inference_scaling_protocol": protocol,
        }


__all__ = [
    "EVALUATION_NUMERIC_PATH_SCHEMA",
    "FP4_INFERENCE_ACTIVATION_MODES",
    "FP4_INFERENCE_SCALING_PROTOCOL_SCHEMA",
    "FP4InferenceScalingController",
    "learned_weight_bf16_numeric_path",
]
