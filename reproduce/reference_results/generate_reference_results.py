#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate public, storage-neutral tables from the paper's reviewed data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "ue5m3_fp4_public_reference_results_v1"
# Spell these test sentinels without embedding the prohibited strings verbatim in
# the public source tree.  ``validate_bundle.py`` scans source files as well as
# generated artifacts, while this module independently reconstructs the exact
# strings and applies them to every exported table.
FORBIDDEN_TEXT = (
    "s3" + "://",
    "/" + "workspace/",
    "/" + "volt/",
    "job" + "_id",
    "job" + "-id",
)

TABLES = {
    "validation_metrics.csv": {
        "source": "quantized_validation_curve_metrics.csv",
        "rows": 84,
        "remove": {"result_uri"},
    },
    "validation_comparisons.csv": {
        "source": "quantized_validation_curve_comparisons.csv",
        "rows": 144,
        "remove": {"candidate_result_uri", "reference_result_uri"},
    },
    "olmes_aggregate_scores.csv": {
        "source": "quantized_olmes/scores.csv",
        "rows": 21,
        "remove": {"checkpoint_uri"},
    },
    "olmes_leaf_task_metrics.csv": {
        "source": "quantized_olmes/leaf_task_metrics.csv",
        "rows": 1_022,
        "remove": {"checkpoint_uri"},
    },
    "olmes_paired_differences.csv": {
        "source": "quantized_olmes/paired_differences.csv",
        "rows": 36,
        "remove": set(),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(
    source: Path,
    output: Path,
    *,
    remove: set[str],
    expected_rows: int,
) -> dict[str, Any]:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        missing = remove - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"expected private columns are absent in {source}: {sorted(missing)}"
            )
        fields = [field for field in reader.fieldnames if field not in remove]
        rows = [{field: row[field] for field in fields} for row in reader]
    if len(rows) != expected_rows:
        raise ValueError(f"{source} has {len(rows)} rows; expected {expected_rows}")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "source": source.relative_to(source.parents[1]).as_posix(),
        "source_sha256": _sha256(source),
        "source_rows": len(rows),
        "removed_columns": sorted(remove),
        "public_file": output.name,
        "public_sha256": _sha256(output),
        "public_columns": fields,
        "public_rows": len(rows),
    }


def generate(report_data: Path, output: Path) -> dict[str, Any]:
    report_data = report_data.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for output_name, specification in TABLES.items():
        source = report_data / specification["source"]
        if not source.is_file():
            raise FileNotFoundError(source)
        records.append(
            _write_csv(
                source,
                output / output_name,
                remove=specification["remove"],
                expected_rows=specification["rows"],
            )
        )

    score_path = output / "olmes_aggregate_scores.csv"
    with score_path.open(newline="", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    task_keys = sorted({row["task_key"] for row in score_rows})
    benchmarks = sorted({row["benchmark"] for row in score_rows})
    if len(task_keys) != 7 or len(benchmarks) != 3:
        raise ValueError(
            f"OLMES table must be a complete 7x3 matrix, got {len(task_keys)}x{len(benchmarks)}"
        )
    if {(row["task_key"], row["benchmark"]) for row in score_rows} != {
        (task, benchmark) for task in task_keys for benchmark in benchmarks
    }:
        raise ValueError("OLMES aggregate score matrix contains a duplicate or missing cell")

    provenance = {
        "schema": SCHEMA,
        "generated_from": "reviewed UE5M3 paper data",
        "transformation": (
            "lossless CSV row/field copy except explicitly listed private storage columns"
        ),
        "tables": records,
        "invariants": {
            "validation_points": 84,
            "validation_comparisons": 144,
            "olmes_configurations": task_keys,
            "olmes_benchmarks": benchmarks,
            "olmes_aggregate_cells": 21,
            "olmes_leaf_task_cells": 1_022,
            "olmes_leaf_tasks_per_configuration": 146,
            "olmes_paired_differences": 36,
            "independent_training_seeds_per_configuration": 1,
        },
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for path in [*(output / name for name in TABLES), provenance_path]:
        payload = path.read_text(encoding="utf-8").lower()
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in payload:
                raise RuntimeError(f"forbidden private marker {forbidden!r} in {path}")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report_data",
        type=Path,
        help="path to reports/ue5m3_fp4_training/data in the reviewed paper tree",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    arguments = parser.parse_args()
    provenance = generate(arguments.report_data, arguments.output.resolve())
    print(json.dumps(provenance["invariants"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
