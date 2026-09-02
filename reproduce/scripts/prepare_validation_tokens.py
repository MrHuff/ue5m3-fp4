#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze ordered token rows into hashed validation/calibration safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

SCHEMA = "ue5m3_fp4_public_token_freeze_v1"


def _sha256(path: Path) -> str:
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


def _load_rows(paths: list[Path], *, width: int) -> tuple[list[torch.Tensor], list[dict]]:
    rows: list[torch.Tensor] = []
    sources: list[dict[str, Any]] = []
    names: set[str] = set()
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name in names:
            raise ValueError(f"input basenames must be unique: {path.name}")
        names.add(path.name)
        source_start = len(rows)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path.name}:{line_number}") from error
                values = document.get("tokens") if isinstance(document, dict) else document
                if not isinstance(values, list) or any(
                    type(value) is not int for value in values
                ):
                    raise TypeError(
                        f"{path.name}:{line_number} must be a JSON list[int] or "
                        "an object with a list[int] field named 'tokens'"
                    )
                if len(values) != width:
                    raise ValueError(
                        f"{path.name}:{line_number} contains {len(values)} tokens; "
                        f"expected {width}"
                    )
                if min(values) < 0 or max(values) > torch.iinfo(torch.int32).max:
                    raise ValueError(f"{path.name}:{line_number} has an invalid token ID")
                rows.append(torch.tensor(values, dtype=torch.int32))
        sources.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "first_record": source_start,
                "records": len(rows) - source_start,
            }
        )
    return rows, sources


def _row_sha256(row: torch.Tensor) -> str:
    return hashlib.sha256(row.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _write_split(
    rows: list[torch.Tensor],
    *,
    name: str,
    output: Path,
    shard_rows: int,
    tensor_key: str,
) -> dict[str, Any]:
    split = output / name
    split.mkdir()
    records = []
    for shard_index, start in enumerate(range(0, len(rows), shard_rows)):
        tensor = torch.stack(rows[start : start + shard_rows]).contiguous()
        path = split / f"part-{shard_index:05d}.safetensors"
        save_file({tensor_key: tensor}, path)
        records.append(
            {
                "file": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "shape": list(tensor.shape),
                "first_record": start,
            }
        )
    row_hashes = [_row_sha256(row) for row in rows]
    return {
        "records": len(rows),
        "ordered_row_sha256": row_hashes,
        "ordered_rows_sha256": _canonical_sha256(row_hashes),
        "shards": records,
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    if args.validation_rows <= 0 or args.calibration_rows < 0:
        raise ValueError("row counts must be positive (calibration may be zero)")
    if args.sequence_length <= 0 or args.shard_rows <= 0:
        raise ValueError("sequence length and shard rows must be positive")
    if not args.tensor_key:
        raise ValueError("tensor key must be non-empty")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    required = args.validation_rows + args.calibration_rows
    rows, sources = _load_rows(
        args.input_jsonl,
        width=args.sequence_length + 1,
    )
    if len(rows) != required:
        raise ValueError(f"inputs contain {len(rows)} rows; expected exactly {required}")
    hashes = [_row_sha256(row) for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("input contains duplicate token rows")

    output.mkdir(parents=True)
    validation = _write_split(
        rows[: args.validation_rows],
        name="validation",
        output=output,
        shard_rows=args.shard_rows,
        tensor_key=args.tensor_key,
    )
    calibration_rows = rows[args.validation_rows :]
    calibration = (
        _write_split(
            calibration_rows,
            name="calibration",
            output=output,
            shard_rows=args.shard_rows,
            tensor_key=args.tensor_key,
        )
        if calibration_rows
        else None
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "historical_validation_identity_reproduced": False,
        "identity_scope": "new ordered public/caller-supplied token rows",
        "ordering": (
            "non-empty JSONL lines in input argument order; first validation_rows "
            "records are validation and the remainder are calibration"
        ),
        "tensor_key": args.tensor_key,
        "storage_dtype": "int32",
        "model_input_tokens_per_record": args.sequence_length,
        "target_only_tokens_per_record": 1,
        "sources": sources,
        "validation": validation,
        "calibration": calibration,
        "validation_calibration_disjoint": True,
    }
    manifest["identity_sha256"] = _canonical_sha256(manifest)
    path = output / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-rows", type=int, default=768)
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--shard-rows", type=int, default=64)
    parser.add_argument("--tensor-key", default="tokens")
    return parser


def main() -> int:
    manifest = freeze(build_parser().parse_args())
    print(json.dumps({"identity_sha256": manifest["identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
