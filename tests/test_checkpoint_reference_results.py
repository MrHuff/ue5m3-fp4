# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _ROOT / "reproduce" / "reference_results"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_sanitized_reference_result_inventory_and_hashes() -> None:
    provenance = json.loads((_RESULTS / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["schema"] == "ue5m3_fp4_public_reference_results_v1"
    assert provenance["invariants"]["validation_points"] == 84
    assert provenance["invariants"]["validation_comparisons"] == 144
    assert provenance["invariants"]["olmes_aggregate_cells"] == 21
    assert provenance["invariants"]["olmes_leaf_task_cells"] == 1_022
    assert provenance["invariants"]["olmes_paired_differences"] == 36
    for record in provenance["tables"]:
        public_path = _RESULTS / record["public_file"]
        assert len(_rows(public_path)) == record["public_rows"]
        assert _sha256(public_path) == record["public_sha256"]


def test_olmes_scores_form_complete_seven_by_three_matrix() -> None:
    rows = _rows(_RESULTS / "olmes_aggregate_scores.csv")
    tasks = {row["task_key"] for row in rows}
    benchmarks = {row["benchmark"] for row in rows}
    assert len(tasks) == 7
    assert len(benchmarks) == 3
    assert {(row["task_key"], row["benchmark"]) for row in rows} == {
        (task, benchmark) for task in tasks for benchmark in benchmarks
    }


def test_olmes_leaf_results_cover_every_configuration_and_task() -> None:
    rows = _rows(_RESULTS / "olmes_leaf_task_metrics.csv")
    tasks = {row["task_key"] for row in rows}
    indices = {int(row["leaf_task_index"]) for row in rows}
    assert len(rows) == 1_022
    assert len(tasks) == 7
    assert indices == set(range(146))
    assert {(row["task_key"], int(row["leaf_task_index"])) for row in rows} == {
        (task, index) for task in tasks for index in range(146)
    }


def test_reference_results_contain_no_private_storage_markers() -> None:
    forbidden = (
        "s3" + "://",
        "/" + "workspace/",
        "/" + "volt/",
        "job" + "_id",
    )
    for path in sorted(_RESULTS.glob("*")):
        if path.suffix not in {".csv", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert all(marker not in text for marker in forbidden)


def test_rendered_reference_artifact_manifest() -> None:
    generated = _RESULTS / "generated"
    manifest = json.loads((generated / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "ue5m3_fp4_rendered_reference_artifacts_v1"
    assert manifest["formula"]["validation_percent"] == (
        "100 * (BF16_NLL - candidate_NLL) / BF16_NLL"
    )
    assert manifest["formula"]["olmes_percentage_points"] == (
        "candidate_score_percent - BF16_score_percent"
    )
    assert len(manifest["outputs"]) == 6
    for record in manifest["outputs"]:
        path = generated / record["file"]
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]
