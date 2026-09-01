#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Exact next-token loss over local, already-tokenized sequences.

Every sequence contains one more token than is forwarded to the model.  For a
row named ``tokens``, ``tokens[:-1]`` is the model input and ``tokens[1:]`` is
the position-aligned target.  Labels are never passed to the model because
causal language-model implementations commonly shift labels internally.

Inputs may be in-memory integer tensors or local ``.safetensors`` files.  A
tensor can have shape ``[sequence_length + 1]`` or
``[num_sequences, sequence_length + 1]``.  A safetensors file must contain a
rank-two integer tensor under ``tensor_key`` (``"tokens"`` by default).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load

VALIDATION_RESULT_SCHEMA = "ue5m3_fp4_local_validation_v1"

ValidationInput: TypeAlias = torch.Tensor | str | Path
ValidationInputs: TypeAlias = ValidationInput | Iterable[ValidationInput]
ProgressCallback: TypeAlias = Callable[[int, int], None]
BeforeForwardCallback: TypeAlias = Callable[[torch.Tensor, int], None]

_SAFE_INTEGER_DTYPES = {
    "I32": (torch.int32, "int32"),
    "I64": (torch.int64, "int64"),
}
_TORCH_INTEGER_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


@dataclass(frozen=True)
class ValidationSource:
    """Stable metadata for one in-memory tensor or local safetensors file."""

    kind: Literal["tensor", "safetensors"]
    identifier: str
    sha256: str
    shape: tuple[int, int]
    dtype: Literal["int32", "int64"]
    sequence_offset: int


@dataclass(frozen=True)
class _PreparedSource:
    metadata: ValidationSource
    tensor: torch.Tensor | None = None
    path: Path | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(repr(tuple(canonical.shape)).encode("ascii"))
    digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def discover_validation_paths(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand local files and directories while preserving explicit order.

    Directory contents are appended in sorted recursive order.  Repeated files
    are rejected so a duplicated path cannot silently double the token count.
    """

    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Validation input does not exist: {path}")
        candidates = (
            sorted(path.rglob("*.safetensors"), key=lambda item: str(item))
            if path.is_dir()
            else [path]
        )
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files found under {path}")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix != ".safetensors":
                raise ValueError(
                    f"Validation files must use the .safetensors suffix; got {candidate}"
                )
            resolved = candidate.resolve()
            if resolved in seen:
                raise ValueError(f"Validation file was provided more than once: {resolved}")
            seen.add(resolved)
            paths.append(resolved)
    return paths


def _as_input_list(inputs: ValidationInputs) -> list[ValidationInput]:
    if isinstance(inputs, (torch.Tensor, str, Path)):
        return [inputs]
    result = list(inputs)
    if not result:
        raise ValueError("At least one validation input is required")
    if any(not isinstance(item, (torch.Tensor, str, Path)) for item in result):
        invalid = next(
            item for item in result if not isinstance(item, (torch.Tensor, str, Path))
        )
        raise TypeError(
            f"Validation inputs must be tensors or local paths; got {type(invalid)!r}"
        )
    return result


def _normalize_tensor(tensor: torch.Tensor, *, label: str) -> torch.Tensor:
    if tensor.dtype not in _TORCH_INTEGER_DTYPES:
        raise TypeError(
            f"{label} must use torch.int32 or torch.int64 storage; got {tensor.dtype}"
        )
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"{label} must have shape [N, sequence_length + 1]; got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 1:
        raise ValueError(
            f"{label} must contain a sequence with at least two tokens; "
            f"got {tuple(tensor.shape)}"
        )
    # Own the storage so a caller cannot mutate the evaluated data after its
    # identity hash has been recorded.
    return tensor.detach().to(device="cpu").contiguous().clone()


def _prepare_sources(
    inputs: ValidationInputs,
    *,
    tensor_key: str,
) -> list[_PreparedSource]:
    raw_inputs = _as_input_list(inputs)
    expanded: list[ValidationInput] = []
    pending_paths: list[str | Path] = []

    def flush_paths() -> None:
        if pending_paths:
            expanded.extend(discover_validation_paths(pending_paths))
            pending_paths.clear()

    for item in raw_inputs:
        if isinstance(item, torch.Tensor):
            flush_paths()
            expanded.append(item)
        else:
            pending_paths.append(item)
    flush_paths()

    seen_paths: set[Path] = set()
    for item in expanded:
        if isinstance(item, torch.Tensor):
            continue
        resolved = Path(item).resolve()
        if resolved in seen_paths:
            raise ValueError(f"Validation file was provided more than once: {resolved}")
        seen_paths.add(resolved)

    prepared: list[_PreparedSource] = []
    sequence_offset = 0
    expected_width: int | None = None
    for source_index, item in enumerate(expanded):
        if isinstance(item, torch.Tensor):
            tensor = _normalize_tensor(item, label=f"tensor source {source_index}")
            shape = (int(tensor.shape[0]), int(tensor.shape[1]))
            dtype = _TORCH_INTEGER_DTYPES[tensor.dtype]
            identifier = f"tensor-{source_index:05d}"
            metadata = ValidationSource(
                kind="tensor",
                identifier=identifier,
                sha256=_tensor_sha256(tensor),
                shape=shape,
                dtype=dtype,
                sequence_offset=sequence_offset,
            )
            prepared_source = _PreparedSource(metadata=metadata, tensor=tensor)
        else:
            path = Path(item).resolve()
            with safe_open(path, framework="pt", device="cpu") as handle:
                # ``safe_open`` exposes ``keys()`` but is not itself iterable.
                if tensor_key not in handle.keys():  # noqa: SIM118
                    raise KeyError(f"{path} does not contain tensor {tensor_key!r}")
                tensor_slice = handle.get_slice(tensor_key)
                raw_shape = tuple(tensor_slice.get_shape())
                safe_dtype = tensor_slice.get_dtype()
            if safe_dtype not in _SAFE_INTEGER_DTYPES:
                raise TypeError(
                    f"{path}:{tensor_key} must use int32 or int64 storage; "
                    f"got safetensors {safe_dtype}"
                )
            if len(raw_shape) != 2 or raw_shape[0] <= 0 or raw_shape[1] <= 1:
                raise ValueError(
                    f"{path}:{tensor_key} must have non-empty shape "
                    f"[N, sequence_length + 1]; got {raw_shape}"
                )
            shape = (int(raw_shape[0]), int(raw_shape[1]))
            _, dtype = _SAFE_INTEGER_DTYPES[safe_dtype]
            file_sha256 = _file_sha256(path)
            metadata = ValidationSource(
                kind="safetensors",
                identifier=f"safetensors-{source_index:05d}-{file_sha256[:12]}",
                sha256=file_sha256,
                shape=shape,
                dtype=dtype,
                sequence_offset=sequence_offset,
            )
            prepared_source = _PreparedSource(metadata=metadata, path=path)

        if expected_width is None:
            expected_width = shape[1]
        elif shape[1] != expected_width:
            raise ValueError(
                "All validation inputs must use the same sequence length; "
                f"expected width {expected_width}, got {shape[1]} in "
                f"{metadata.identifier}"
            )
        prepared.append(prepared_source)
        sequence_offset += shape[0]

    if not prepared:
        raise ValueError("At least one validation input is required")
    return prepared


def _load_source(source: _PreparedSource, *, tensor_key: str) -> torch.Tensor:
    if source.tensor is not None:
        tensor = source.tensor
    else:
        if source.path is None:
            raise AssertionError("A prepared file source has no path")
        serialized = source.path.read_bytes()
        observed_sha256 = hashlib.sha256(serialized).hexdigest()
        if observed_sha256 != source.metadata.sha256:
            raise RuntimeError(
                f"Validation source {source.metadata.identifier} changed after "
                "its identity was recorded"
            )
        tensor = load(serialized)[tensor_key]
        tensor = _normalize_tensor(tensor, label=f"{source.path}:{tensor_key}")
    expected_dtype = {
        "int32": torch.int32,
        "int64": torch.int64,
    }[source.metadata.dtype]
    if tensor.dtype != expected_dtype or tuple(tensor.shape) != source.metadata.shape:
        raise RuntimeError(
            f"Validation source {source.metadata.identifier} changed after inspection; "
            f"expected {expected_dtype} {source.metadata.shape}, got "
            f"{tensor.dtype} {tuple(tensor.shape)}"
        )
    return tensor


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, Mapping) and "logits" in outputs:
        logits = outputs["logits"]
    elif isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    else:
        raise TypeError("The model output does not expose a logits tensor")
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Expected model logits to be a tensor, got {type(logits)!r}")
    return logits


def _extract_hidden_states(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        hidden_states = outputs
    elif hasattr(outputs, "last_hidden_state"):
        hidden_states = outputs.last_hidden_state
    elif isinstance(outputs, Mapping) and "last_hidden_state" in outputs:
        hidden_states = outputs["last_hidden_state"]
    elif isinstance(outputs, (tuple, list)) and outputs:
        hidden_states = outputs[0]
    else:
        raise TypeError("The model backbone output does not expose hidden states")
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(
            f"Expected backbone hidden states to be a tensor, got {type(hidden_states)!r}"
        )
    return hidden_states


def _validate_targets(targets: torch.Tensor, vocab_size: int) -> None:
    if targets.ndim != 2:
        raise ValueError(f"Expected targets with shape [B, S], got {targets.shape}")
    if vocab_size <= 0:
        raise ValueError("The logits vocabulary dimension must be non-empty")
    target_min = int(targets.min().item())
    target_max = int(targets.max().item())
    if target_min < 0 or target_max >= vocab_size:
        raise ValueError(
            f"Target token IDs [{target_min}, {target_max}] are outside "
            f"vocabulary [0, {vocab_size})"
        )


def _cross_entropy_sums(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    if not bool(torch.isfinite(token_losses).all().item()):
        raise FloatingPointError("Non-finite validation loss encountered")
    return token_losses.sum(dim=1, dtype=torch.float64)


def _sequence_loss_sums(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    """Return one FP64-accumulated sum of FP32 token losses per sequence."""

    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, S, V], got {logits.shape}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "Logit positions must align one-to-one with targets: "
            f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
        )
    _validate_targets(targets, logits.shape[-1])

    sums = torch.zeros(logits.shape[0], dtype=torch.float64, device=logits.device)
    for start in range(0, targets.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, targets.shape[1])
        sums += _cross_entropy_sums(
            logits[:, start:stop, :],
            targets[:, start:stop],
        )
    return sums.cpu()


def _sequence_loss_sums_from_hidden_states(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    """Apply the language-model head and FP32 CE in bounded position chunks."""

    if hidden_states.ndim != 3 or hidden_states.shape[:2] != targets.shape:
        raise ValueError(
            "Backbone positions must align one-to-one with targets: "
            f"hidden_states={tuple(hidden_states.shape)}, "
            f"targets={tuple(targets.shape)}"
        )
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, torch.nn.Module):
        raise TypeError("chunked_lm_head mode requires model.lm_head")
    vocab_size = getattr(lm_head, "out_features", None)
    if not isinstance(vocab_size, int):
        raise TypeError("model.lm_head must expose an integer out_features")
    _validate_targets(targets, vocab_size)
    try:
        head_dtype = next(lm_head.parameters()).dtype
    except StopIteration as error:
        raise TypeError("model.lm_head must have floating-point parameters") from error

    sums = torch.zeros(
        hidden_states.shape[0],
        dtype=torch.float64,
        device=hidden_states.device,
    )
    for start in range(0, targets.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, targets.shape[1])
        chunk_hidden = hidden_states[:, start:stop, :].to(dtype=head_dtype)
        chunk_logits = lm_head(chunk_hidden)
        if not isinstance(chunk_logits, torch.Tensor):
            raise TypeError(
                f"Expected model.lm_head output to be a tensor, got {type(chunk_logits)!r}"
            )
        sums += _cross_entropy_sums(chunk_logits, targets[:, start:stop])
    return sums.cpu()


def evaluate_validation(
    model: torch.nn.Module,
    inputs: ValidationInputs,
    *,
    checkpoint_id: str,
    device: str | torch.device,
    batch_size: int = 1,
    ce_chunk_tokens: int = 256,
    tensor_key: str = "tokens",
    model_provenance: Mapping[str, Any] | None = None,
    data_provenance: Mapping[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    before_forward_callback: BeforeForwardCallback | None = None,
    forward_kwargs: Mapping[str, Any] | None = None,
    logit_path: Literal["chunked_lm_head", "model_forward"] = "model_forward",
) -> dict[str, Any]:
    """Evaluate exact next-token NLL on local token sequences.

    Cross-entropy is computed in FP32 and per-sequence sums are accumulated in
    FP64.  ``before_forward_callback`` runs immediately before each model
    forward and can advance a cold D=50 inference replay controller exactly
    once per evaluation work unit.  Its integer argument is one-based.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("checkpoint_id must be a non-empty string")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if ce_chunk_tokens <= 0:
        raise ValueError(f"ce_chunk_tokens must be positive, got {ce_chunk_tokens}")
    if not isinstance(tensor_key, str) or not tensor_key:
        raise ValueError("tensor_key must be a non-empty string")
    if logit_path not in {"chunked_lm_head", "model_forward"}:
        raise ValueError(f"Unknown logit_path: {logit_path!r}")
    if logit_path == "chunked_lm_head" and not isinstance(
        getattr(model, "backbone", None), torch.nn.Module
    ):
        raise TypeError("chunked_lm_head mode requires model.backbone")
    kwargs = dict(forward_kwargs or {})
    if "labels" in kwargs:
        raise ValueError("Do not pass labels: validation targets are already shifted")
    if "input_ids" in kwargs:
        raise ValueError("forward_kwargs must not contain input_ids")

    sources = _prepare_sources(inputs, tensor_key=tensor_key)
    total_sequences = sum(source.metadata.shape[0] for source in sources)
    target_tokens_per_sequence = sources[0].metadata.shape[1] - 1
    expected_total_tokens = total_sequences * target_tokens_per_sequence

    model.eval()
    evaluation_device = torch.device(device)
    per_sequence_loss_sums: list[float] = []
    per_sequence_token_counts: list[int] = []
    per_sequence_provenance: list[dict[str, Any]] = []
    completed_sequences = 0
    forward_count = 0
    started_at = time.perf_counter()

    with torch.inference_mode():
        for source_index, source in enumerate(sources):
            tokens = _load_source(source, tensor_key=tensor_key)
            for row_start in range(0, tokens.shape[0], batch_size):
                row_stop = min(row_start + batch_size, tokens.shape[0])
                batch = tokens[row_start:row_stop].to(
                    device=evaluation_device,
                    dtype=torch.long,
                    non_blocking=evaluation_device.type == "cuda",
                )
                input_ids = batch[:, :-1].contiguous()
                targets = batch[:, 1:].contiguous()
                forward_count += 1
                if before_forward_callback is not None:
                    before_forward_callback(input_ids, forward_count)

                if logit_path == "chunked_lm_head":
                    backbone_outputs = model.backbone(input_ids=input_ids, **kwargs)
                    hidden_states = _extract_hidden_states(backbone_outputs)
                    loss_sums = _sequence_loss_sums_from_hidden_states(
                        model,
                        hidden_states,
                        targets,
                        chunk_tokens=ce_chunk_tokens,
                    )
                else:
                    outputs = model(input_ids=input_ids, **kwargs)
                    logits = _extract_logits(outputs)
                    loss_sums = _sequence_loss_sums(
                        logits,
                        targets,
                        chunk_tokens=ce_chunk_tokens,
                    )

                for batch_index, loss_sum_tensor in enumerate(loss_sums):
                    source_row = row_start + batch_index
                    loss_sum = float(loss_sum_tensor.item())
                    per_sequence_loss_sums.append(loss_sum)
                    per_sequence_token_counts.append(target_tokens_per_sequence)
                    per_sequence_provenance.append(
                        {
                            "sequence_index": (source.metadata.sequence_offset + source_row),
                            "source_index": source_index,
                            "source_identifier": source.metadata.identifier,
                            "source_sha256": source.metadata.sha256,
                            "row_index": source_row,
                        }
                    )

                completed_sequences += row_stop - row_start
                if progress_callback is not None:
                    progress_callback(completed_sequences, total_sequences)

    token_count = sum(per_sequence_token_counts)
    if completed_sequences != total_sequences or token_count != expected_total_tokens:
        raise RuntimeError(
            "Validation accounting mismatch: "
            f"completed {completed_sequences}/{total_sequences} sequences and "
            f"{token_count}/{expected_total_tokens} target tokens"
        )
    total_loss_sum = math.fsum(per_sequence_loss_sums)
    nll = total_loss_sum / token_count
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "schema": VALIDATION_RESULT_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "model": dict(model_provenance or {}),
        "validation_data": {
            **dict(data_provenance or {}),
            "tensor_key": tensor_key,
            "sources": [asdict(source.metadata) for source in sources],
            "sequence_length": target_tokens_per_sequence,
            "sequences": total_sequences,
            "target_tokens": token_count,
        },
        "evaluation": {
            "batch_size": batch_size,
            "forward_count": forward_count,
            "ce_chunk_tokens": ce_chunk_tokens,
            "cross_entropy_dtype": "float32",
            "accumulation_dtype": "float64",
            "logit_path": logit_path,
            "elapsed_seconds": elapsed_seconds,
        },
        "metrics": {
            "loss_sum": total_loss_sum,
            "token_count": token_count,
            "nll": nll,
            "perplexity": math.exp(nll),
        },
        "per_sequence": {
            "loss_sums": per_sequence_loss_sums,
            "token_counts": per_sequence_token_counts,
            "provenance": per_sequence_provenance,
        },
    }


__all__ = [
    "VALIDATION_RESULT_SCHEMA",
    "ValidationSource",
    "discover_validation_paths",
    "evaluate_validation",
]
