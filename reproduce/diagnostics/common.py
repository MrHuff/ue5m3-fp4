# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared provenance and serialization helpers for public diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "ue5m3_fp4_diagnostic_v1"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value canonically for identity hashing."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the SHA256 identity of a JSON-compatible value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def runtime_provenance(device: torch.device | str) -> dict[str, Any]:
    """Capture stable public runtime fields without host paths or identifiers."""

    resolved = torch.device(device)
    cuda_name = None
    cuda_capability = None
    if resolved.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(resolved)
        cuda_capability = list(torch.cuda.get_device_capability(resolved))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_type": resolved.type,
        "device_name": cuda_name,
        "cuda_capability": cuda_capability,
    }


def make_payload(
    *,
    experiment: str,
    config: dict[str, Any],
    results: Any,
    device: torch.device | str,
    evidence_class: str = "rerunnable_public_experiment",
) -> dict[str, Any]:
    """Build a self-identifying diagnostic result payload."""

    if evidence_class not in {
        "rerunnable_public_experiment",
        "archived_report_evidence",
    }:
        raise ValueError(f"unsupported evidence class: {evidence_class!r}")
    return {
        "schema": SCHEMA_VERSION,
        "experiment": experiment,
        "evidence_class": evidence_class,
        "config": config,
        "config_sha256": sha256_json(config),
        "runtime": runtime_provenance(device),
        "results": results,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic, human-readable JSON result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
