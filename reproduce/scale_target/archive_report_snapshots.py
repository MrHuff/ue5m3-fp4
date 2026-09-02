#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create storage-neutral public archives from the report histogram captures.

The source report captures contain storage and orchestration provenance that is
irrelevant to reproducing the numerical figure.  This one-way exporter keeps
the complete numerical capture, checkpoint/data identities, and source hashes,
while deliberately dropping locations, job metadata, and nested runtime-wheel
records.  It refuses source files other than the two hash-pinned report inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "ue5m3_fp4_bf16_scale_target_histograms_v1"
ARCHIVE_SCHEMA = "ue5m3_fp4_public_archived_scale_target_histograms_v1"
MANIFEST_SCHEMA = "ue5m3_fp4_public_scale_target_archive_manifest_v1"
REFERENCE_RENDER_SHA256 = {
    "pdf": "fd6d42dae417c7708e9c8ac485032909ac999d4ac35c81096fde647876c6307b",
    "png": "219cfb96b690dba693912e56c3b7271176fb595a650b32dc11987407e374214e",
}
EXPECTED_SOURCES = {
    "bf16": "8637d66dc9b53b8b29ceea068a16afba1e43d0f208049cd500da27205f4175c2",
    "ue5m3_proposed_b16": ("1a07a7663ee07a4f352d4c7f1a0896e9e025a295cc154066b86bfe5c18b263ee"),
}

_FORBIDDEN_LOCATION_KEYS = {
    "uri",
    "path",
    "job",
    "job_id",
    "run_id",
    "project",
    "remote_folder",
    "dump_folder",
}
_FORBIDDEN_VALUE_FRAGMENTS = (
    "s3://",
    "/work" + "space/",
    "/vo" + "lt/",
    "${wandb_name}",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("snapshot input must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise TypeError("snapshot input must contain a JSON object")
    return document


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def validate_storage_neutral(value: Any, *, location: str = "root") -> None:
    """Reject location/orchestration fields and storage-like string values."""

    if isinstance(value, Mapping):
        for raw_key, member in value.items():
            key = str(raw_key)
            normalized = key.lower()
            if normalized in _FORBIDDEN_LOCATION_KEYS or normalized.endswith("_uri"):
                raise ValueError(f"forbidden location field at {location}.{key}")
            if normalized.endswith("_path") and normalized != "numeric_path":
                raise ValueError(f"forbidden path field at {location}.{key}")
            validate_storage_neutral(member, location=f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, member in enumerate(value):
            validate_storage_neutral(member, location=f"{location}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.lower()
        for fragment in _FORBIDDEN_VALUE_FRAGMENTS:
            if fragment in lowered:
                raise ValueError(
                    f"storage/orchestration value at {location} contains {fragment!r}"
                )


def sanitize_snapshot(
    source: Mapping[str, Any],
    *,
    key: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Return the storage-neutral subset needed to inspect and plot a capture."""

    if key not in EXPECTED_SOURCES:
        raise KeyError(f"unknown report snapshot {key!r}")
    if source_sha256 != EXPECTED_SOURCES[key]:
        raise ValueError(
            f"source hash for {key} changed: {source_sha256} != {EXPECTED_SOURCES[key]}"
        )
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"source schema for {key} changed")
    checkpoint = _mapping(source.get("checkpoint"), "checkpoint")
    validation = _mapping(source.get("validation_manifest"), "validation manifest")
    sequence = _mapping(source.get("sequence"), "sequence")
    model = _mapping(source.get("model"), "model")
    protocol = _mapping(source.get("protocol"), "protocol")
    runtime = _mapping(source.get("runtime"), "runtime")
    capture = _mapping(source.get("capture"), "capture")

    archived = {
        "schema": ARCHIVE_SCHEMA,
        "archive_key": key,
        "captured_at_utc": source.get("created_at_utc"),
        "archived_source": {
            "schema": SOURCE_SCHEMA,
            "sha256": source_sha256,
            "capture_source_sha256": protocol.get("source_sha256"),
        },
        "scope": source.get("scope"),
        "checkpoint_identity": {
            "step": 30_000,
            "metadata_sha256": checkpoint.get("metadata_sha256"),
            "selected_model_fqns": checkpoint.get("selected_model_fqns"),
            "selected_model_fqns_sha256": checkpoint.get("selected_model_fqns_sha256"),
            "export_manifest_sha256": _mapping(
                checkpoint.get("local_export"), "checkpoint local export"
            ).get("manifest_sha256"),
            "export_file_count": _mapping(
                checkpoint.get("local_export"), "checkpoint local export"
            ).get("file_count"),
        },
        "validation_identity": {
            "schema": validation.get("schema"),
            "manifest_sha256": validation.get("sha256"),
            "provenance_sha256": validation.get("provenance_sha256"),
            "sequences": validation.get("sequences"),
            "validation_tokens": validation.get("validation_tokens"),
        },
        "sequence_identity": {
            "global_sequence_index": sequence.get("global_sequence_index"),
            "rank": sequence.get("rank"),
            "rank_local_sequence_index": sequence.get("rank_local_sequence_index"),
            "shard_sha256": sequence.get("shard_sha256"),
            "token_tensor_sha256": sequence.get("token_tensor_sha256"),
            "token_dtype": sequence.get("token_dtype"),
            "token_shape": sequence.get("token_shape"),
            "prediction_tokens": sequence.get("prediction_tokens"),
        },
        "model_identity": {
            "class": model.get("class"),
            "config_class": model.get("config_class"),
            "config_sha256": model.get("config_sha256"),
            "parameter_count": model.get("parameter_count"),
            "parameter_dtypes": model.get("parameter_dtypes"),
            "attention_implementation": model.get("attention_implementation"),
            "hf_assets": model.get("export_hf_assets"),
            "static_model_config_sha256": _mapping(
                model.get("export_static_model_config"), "static model config"
            ).get("sha256"),
            "export_verification": model.get("export_verification"),
        },
        "protocol": {
            field: protocol.get(field)
            for field in (
                "attention_implementation",
                "block_size",
                "histogram_chunk_elements",
                "numeric_path",
                "scale_format",
                "scale_reference",
                "scale_targets",
                "seed",
                "wgrad_dy_layout",
            )
        },
        "runtime": {
            field: runtime.get(field)
            for field in (
                "cuda_device_name",
                "cuda_version",
                "torch_version",
                "transformers_version",
                "mamba_ssm_version",
                "causal_conv1d_version",
                "cut_cross_entropy_version",
            )
        },
        # The complete per-layer and pooled numerical evidence is retained.
        "capture": capture,
    }
    validate_storage_neutral(archived)
    return archived


def _parse_snapshot(value: str) -> tuple[str, Path]:
    key, separator, raw_path = value.partition("=")
    if not separator or key not in EXPECTED_SOURCES or not raw_path:
        choices = ", ".join(EXPECTED_SOURCES)
        raise argparse.ArgumentTypeError(f"expected KEY=FILE where KEY is one of {choices}")
    return key, Path(raw_path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        action="append",
        type=_parse_snapshot,
        required=True,
        help="Hash-pinned report input as bf16=FILE or ue5m3_proposed_b16=FILE.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = dict(args.snapshot)
    if set(inputs) != set(EXPECTED_SOURCES) or len(inputs) != len(args.snapshot):
        raise ValueError("provide each expected snapshot exactly once")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    records: dict[str, Any] = {}
    for key in EXPECTED_SOURCES:
        source_path = inputs[key]
        source_sha256 = sha256_file(source_path)
        archived = sanitize_snapshot(
            _read_json(source_path),
            key=key,
            source_sha256=source_sha256,
        )
        destination = output / f"{key}.json"
        _write_json(destination, archived)
        records[key] = {
            "source_sha256": source_sha256,
            "archive_file": destination.name,
            "archive_sha256": sha256_file(destination),
        }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "scope": (
            "Storage-neutral copies of the numerical histograms used by the report; "
            "model weights and token data are not included."
        ),
        "snapshots": records,
        "reference_render_sha256": REFERENCE_RENDER_SHA256,
    }
    validate_storage_neutral(manifest)
    _write_json(output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
