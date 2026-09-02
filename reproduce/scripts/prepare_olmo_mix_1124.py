#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the public 32-way OLMo Mix 1124 source transformation.

This is a storage-neutral, W&B-free and scheduler-free version of the recovered
preparation method. It writes Hugging Face ``Dataset.save_to_disk`` directories.
Conversion of these rows into a new MDS stream is method reproduction, not a
byte-exact reconstruction of the unavailable historical MDS objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

DATASET_ID = "allenai/olmo-mix-1124"
REVISION = "8162bd79c6dc4fea470506531a8d791badc06b4b"
SHARD_COUNT = 32
SHUFFLE_SEED = 42
NO_DCLM_COMPONENTS = (
    "arxiv",
    "starcoder",
    "wiki",
    "pes2o",
    "open-web-math",
    "algebraic-stack",
)
TARGET_COLUMNS = ("text", "id", "source_ds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("no-dclm", "dclm", "inspect"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--shard",
        default="all",
        help="zero-based shard in [0,31], or 'all' (large; parallel jobs are recommended)",
    )
    parser.add_argument("--num-proc", type=int, default=10)
    parser.add_argument("--max-shard-size", default="500MB")
    return parser.parse_args()


def shard_indices(raw: str) -> Iterable[int]:
    if raw == "all":
        return range(SHARD_COUNT)
    index = int(raw)
    if not 0 <= index < SHARD_COUNT:
        raise ValueError(f"--shard must be in [0,{SHARD_COUNT - 1}], got {index}")
    return (index,)


def load_component(name: str, *, cache_dir: Path, num_proc: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the locked 'datasets==3.6.0' to prepare OLMo Mix") from exc
    return load_dataset(
        DATASET_ID,
        name=name,
        split="train",
        revision=REVISION,
        verification_mode="no_checks",
        cache_dir=str(cache_dir),
        streaming=False,
        num_proc=num_proc,
    )


def output_path(root: Path, stream: str, index: int) -> Path:
    return root / stream / f"shard-{index:03d}-of-{SHARD_COUNT:03d}"


def ensure_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def save_dataset(dataset, path: Path, *, num_proc: int, max_shard_size: str) -> None:
    dataset.save_to_disk(
        str(path),
        max_shard_size=max_shard_size,
        num_proc=num_proc,
    )
    summary = {
        "dataset": DATASET_ID,
        "revision": REVISION,
        "rows": len(dataset),
        "columns": list(dataset.column_names),
        "output": str(path),
    }
    print(json.dumps(summary, sort_keys=True))


def prepare_no_dclm(args: argparse.Namespace) -> None:
    from datasets import concatenate_datasets

    # Load once, as in the historical non-streaming transformation, then take
    # the corresponding non-contiguous 32-way shard from every component.
    components = {
        name: load_component(name, cache_dir=args.cache_dir, num_proc=args.num_proc)
        for name in NO_DCLM_COMPONENTS
    }
    for index in shard_indices(args.shard):
        destination = output_path(args.output_root, "olmo-no-dclm", index)
        ensure_destination(destination)
        pieces = []
        for name in NO_DCLM_COMPONENTS:
            piece = components[name].shard(
                num_shards=SHARD_COUNT,
                index=index,
                contiguous=False,
            )
            piece = piece.add_column("source_ds", [name] * len(piece))
            pieces.append(piece)
        merged = concatenate_datasets(pieces, axis=0)
        merged = merged.shuffle(seed=SHUFFLE_SEED)
        merged = merged.select_columns(list(TARGET_COLUMNS))
        merged = merged.flatten_indices(keep_in_memory=False, num_proc=args.num_proc)
        save_dataset(
            merged,
            destination,
            num_proc=args.num_proc,
            max_shard_size=args.max_shard_size,
        )


def prepare_dclm(args: argparse.Namespace) -> None:
    dclm = load_component("dclm", cache_dir=args.cache_dir, num_proc=args.num_proc)
    for index in shard_indices(args.shard):
        destination = output_path(args.output_root, "dclm", index)
        ensure_destination(destination)
        shard = dclm.shard(
            num_shards=SHARD_COUNT,
            index=index,
            contiguous=True,
        )
        shard = shard.add_column("source_ds", ["dclm"] * len(shard))
        shard = shard.select_columns(list(TARGET_COLUMNS))
        save_dataset(
            shard,
            destination,
            num_proc=args.num_proc,
            max_shard_size=args.max_shard_size,
        )


def inspect(args: argparse.Namespace) -> None:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("install the locked 'datasets==3.6.0' to inspect OLMo Mix") from exc

    inventory = []
    errors = []
    for stream in ("dclm", "olmo-no-dclm"):
        for index in range(SHARD_COUNT):
            path = output_path(args.output_root, stream, index)
            record = {
                "stream": stream,
                "shard": index,
                "path": str(path),
                "present": path.is_dir(),
            }
            if not record["present"]:
                errors.append(f"missing prepared shard: {path}")
                inventory.append(record)
                continue
            try:
                dataset = load_from_disk(str(path))
                record["rows"] = len(dataset)
                record["columns"] = list(dataset.column_names)
                if record["rows"] <= 0:
                    errors.append(f"prepared shard is empty: {path}")
                if tuple(record["columns"]) != TARGET_COLUMNS:
                    errors.append(
                        f"prepared shard columns differ for {path}: {record['columns']}"
                    )
            except Exception as exc:
                record["load_error"] = f"{type(exc).__name__}: {exc}"
                errors.append(f"could not load prepared shard {path}: {exc}")
            inventory.append(record)
    result = {
        "schema": "ue5m3_fp4_public_olmo_mix_inventory_v1",
        "expected": 64,
        "valid": not errors,
        "errors": errors,
        "shards": inventory,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise RuntimeError(f"prepared OLMo Mix inventory has {len(errors)} error(s)")


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.num_proc <= 0:
        raise ValueError("--num-proc must be positive")
    if args.mode == "no-dclm":
        prepare_no_dclm(args)
    elif args.mode == "dclm":
        prepare_dclm(args)
    else:
        inspect(args)


if __name__ == "__main__":
    main()
