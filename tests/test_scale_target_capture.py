# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SCALE_TARGET_ROOT = ROOT / "reproduce" / "scale_target"


def _load_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCALE_TARGET_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_module("scale_target_capture", "capture_scale_target_histograms.py")
archive = _load_module("scale_target_archive", "archive_report_snapshots.py")
renderer = _load_module("scale_target_renderer", "render_archived_figure.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for key, member in value.items():
            yield str(key)
            yield from _walk_strings(member)
    elif isinstance(value, list):
        for member in value:
            yield from _walk_strings(member)
    elif isinstance(value, str):
        yield value


def test_streaming_histogram_accounting() -> None:
    histogram = capture.StreamingLogHistogram(
        log10_min=-2.0,
        log10_max=2.0,
        bins=4,
        chunk_elements=2,
    )
    histogram.update(torch.tensor([float("-inf"), float("nan"), 0.0, -1e-4, 0.1, 1.0, 100.0]))
    result = histogram.result()
    assert result["total_count"] == 7
    assert result["finite_count"] == 5
    assert result["nonfinite_count"] == 2
    assert result["zero_count"] == 1
    assert result["positive_count"] == 4
    assert result["underflow_count"] == 1
    assert result["in_range_count"] == 2
    assert result["overflow_count"] == 1
    assert sum(result["counts"]) == result["in_range_count"]
    assert result["update_calls"] == 1


def test_wgrad_dy_blocking_is_along_transposed_reduction_dimension() -> None:
    dy = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]])
    maxima = torch.cat(list(capture.iter_wgrad_dy_block_amax(dy, block_size=4, output_chunk=2)))
    assert torch.equal(maxima, torch.tensor([7.0, 9.0, 8.0, 10.0]))


def test_raw_scale_code_metrics_accounting_and_targets() -> None:
    template = capture.StreamingLogHistogram(
        log10_min=-20,
        log10_max=5,
        bins=25,
        chunk_elements=2,
    )
    metric = capture.RawScaleCodeMetrics(target=448.0, histogram_template=template)
    block_amax = torch.tensor([0.0, 1e-10, 1.0, 2.0])
    metric.update(block_amax, torch.tensor(2.0))
    result = metric.result()
    assert result["block_count"] == 4
    assert result["zero_block_count"] == 1
    assert result["raw_below_min_positive_count"] == 1
    assert result["raw_below_half_min_positive_count"] == 1
    assert result["rounded_zero_count_before_zero_scale_repair"] == 1
    assert result["rounded_saturated_count"] == 0
    assert result["scale_format_min_positive"] == 2.0**-17
    assert result["stale_growth_headroom"] == pytest.approx(61_440 / 448)
    histogram = result["raw_code_histogram"]
    assert histogram["total_count"] == result["block_count"]
    assert histogram["zero_count"] == result["zero_block_count"]


def test_token_loader_requires_n_by_8193_and_records_no_path(tmp_path: Path) -> None:
    token_path = tmp_path / "caller-tokens.safetensors"
    rows = torch.arange(2 * 8193, dtype=torch.int32).reshape(2, 8193)
    save_file({"tokens": rows}, token_path)
    row, identity = capture.load_token_row(
        token_path,
        tensor_key="tokens",
        sequence_index=1,
    )
    assert torch.equal(row, rows[1])
    assert identity["tensor_shape"] == [2, 8193]
    assert identity["prediction_tokens"] == 8192
    assert "path" not in identity
    assert str(tmp_path) not in json.dumps(identity)

    invalid = tmp_path / "invalid.safetensors"
    save_file({"tokens": torch.zeros((1, 8192), dtype=torch.int32)}, invalid)
    with pytest.raises(ValueError, match="8193"):
        capture.load_token_row(invalid, tensor_key="tokens", sequence_index=0)


@pytest.mark.parametrize(
    "value",
    (
        {"uri": "example"},
        {"local_path": "example"},
        {"payload": "s3://example"},
    ),
)
def test_storage_neutral_validator_rejects_locations(value: Any) -> None:
    with pytest.raises(ValueError):
        archive.validate_storage_neutral(value)


def test_archived_histograms_are_sanitized_hashed_and_accounted() -> None:
    archive_root = SCALE_TARGET_ROOT / "archived"
    manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == archive.MANIFEST_SCHEMA
    assert manifest["reference_render_sha256"] == archive.REFERENCE_RENDER_SHA256
    archive.validate_storage_neutral(manifest)

    for key, expected_source_hash in archive.EXPECTED_SOURCES.items():
        path = archive_root / f"{key}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        archive.validate_storage_neutral(document)
        assert document["schema"] == archive.ARCHIVE_SCHEMA
        assert document["archive_key"] == key
        assert document["archived_source"]["sha256"] == expected_source_hash
        assert manifest["snapshots"][key]["archive_sha256"] == _sha256(path)

        capture_record = document["capture"]
        layers = capture_record["layers"]
        assert set(layers) == {"45", "47", "49", "51"}
        for histogram_name in (
            "x_histogram",
            "weight_histogram",
            "dy_histogram",
            "wgrad_dy_block_amax_histogram",
        ):
            pooled = renderer.validate_histogram(
                capture_record["pooled"][histogram_name],
                label=f"{key}.{histogram_name}",
            )
            members = [
                renderer.validate_histogram(
                    layer[histogram_name], label=f"{key}.{histogram_name}.layer"
                )
                for layer in layers.values()
            ]
            assert pooled["counts"] == [
                sum(values) for values in zip(*(member["counts"] for member in members))
            ]
            assert pooled["total_count"] == sum(member["total_count"] for member in members)
        for target in ("448", "2048"):
            raw = capture_record["pooled"]["raw_scale_codes"][target]
            raw_histogram = renderer.validate_histogram(
                raw["raw_code_histogram"], label=f"{key}.raw.{target}"
            )
            assert raw["block_count"] == raw_histogram["total_count"]
            assert raw["zero_block_count"] == raw_histogram["zero_count"]


def test_historical_350m_archive_matches_all_final_window_points() -> None:
    csv_path = SCALE_TARGET_ROOT / "historical_350m_final_window.csv"
    provenance_path = SCALE_TARGET_ROOT / "historical_350m_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    archive.validate_storage_neutral(provenance)
    assert provenance["artifact_lineage"]["public_final_window_csv_sha256"] == _sha256(csv_path)
    assert provenance["public_rerun_status"]["status"] == "record_only"
    assert provenance["measurement"]["independent_runs_per_configuration"] == 1

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert {row["configuration"] for row in rows} == {
        "default_T448",
        "final_four_w2_wgrad_dY_T2048",
    }
    summaries = {
        summary["configuration"]: summary for summary in provenance["measurement"]["summaries"]
    }
    means: dict[str, float] = {}
    for configuration, summary in summaries.items():
        selected = [row for row in rows if row["configuration"] == configuration]
        values = [float(row["training_loss"]) for row in selected]
        means[configuration] = sum(values) / len(values)
        population_std = math.sqrt(
            sum((value - means[configuration]) ** 2 for value in values) / len(values)
        )
        assert len(selected) == summary["points"] == 10
        assert means[configuration] == pytest.approx(
            summary["final_window_mean_training_loss"], abs=1e-15
        )
        assert population_std == pytest.approx(
            summary["final_window_population_standard_deviation"], abs=1e-15
        )
        endpoint = next(row for row in selected if int(row["step"]) == 10_000)
        assert float(endpoint["training_loss"]) == summary["step_10000_training_loss"]
    observed_difference = means["final_four_w2_wgrad_dY_T2048"] - means["default_T448"]
    assert observed_difference == pytest.approx(
        provenance["measurement"]["mean_training_loss_difference_T2048_minus_T448"],
        abs=1e-15,
    )

    public_text = "\n".join(_walk_strings(provenance)).lower()
    for forbidden in (
        "run_id",
        "job_id",
        "s3://",
        "/work" + "space/",
        "/vo" + "lt/",
    ):
        assert forbidden not in public_text
