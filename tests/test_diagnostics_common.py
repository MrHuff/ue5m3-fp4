# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from reproduce.diagnostics.common import canonical_json_bytes, make_payload, sha256_json
from reproduce.diagnostics.numerics import (
    accumulate_issue_partials,
    accumulate_issue_terms,
    add_rz_f32,
    comparison_metrics,
    snap_rne,
)


def test_canonical_config_hash_is_key_order_independent() -> None:
    left = {"z": [3, 2, 1], "a": {"flag": True}}
    right = {"a": {"flag": True}, "z": [3, 2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == hashlib.sha256(canonical_json_bytes(left)).hexdigest()


def test_payload_identifies_config_and_evidence_class() -> None:
    config = {"seed": 17, "shape": [2, 64]}
    payload = make_payload(
        experiment="unit_test",
        config=config,
        results={"exact": True},
        device="cpu",
    )

    assert payload["config_sha256"] == sha256_json(config)
    assert payload["evidence_class"] == "rerunnable_public_experiment"
    assert payload["runtime"]["device_type"] == "cpu"


def test_add_rz_steps_toward_zero_only_when_rne_rounds_away() -> None:
    one = torch.tensor([1.0], dtype=torch.float32)
    # 2^-24 is exactly halfway between 1.0 and the next FP32 value. RNE returns
    # 1.0, which is already toward zero, so RTZ agrees.
    positive_half_ulp = torch.tensor([2.0**-24], dtype=torch.float32)
    assert torch.equal(add_rz_f32(one, positive_half_ulp), one)

    # A negative half-ULP increment makes RNE select 1.0, which is away from
    # the positive exact sum; RTZ instead selects the lower neighbor.
    decrement = torch.tensor([-(2.0**-25)], dtype=torch.float32)
    rounded = one + decrement
    result = add_rz_f32(one, decrement)
    assert result < rounded
    assert result.double() <= one.double() + decrement.double()


def test_issue_reduction_and_grid_invariants() -> None:
    terms = torch.zeros((2, 128), dtype=torch.float32)
    terms[0, 0] = 1.0
    terms[0, 64] = -0.25
    terms[1, 3] = 1.0 / 4096.0
    reduced = accumulate_issue_terms(terms, issue_size=64, grid_denominator=1024)

    assert reduced["reference"].tolist() == [0.75, 1.0 / 4096.0]
    assert reduced["rn"][0].item() == 0.75
    assert reduced["rz"][0].item() == 0.75
    assert reduced["rn_grid"][1].item() == 0.0
    assert reduced["rz_grid"][1].item() == 0.0
    assert snap_rne(torch.tensor([0.5 / 1024]), 1024).item() == 0.0


def test_partial_reduction_metrics_are_json_safe() -> None:
    partials = torch.tensor([[1.0, -1.0], [2.0, -1.0]], dtype=torch.float32)
    reduced = accumulate_issue_partials(partials)
    metrics = comparison_metrics(reduced["rz"], reduced["reference"])

    assert metrics["exact_count"] == 2
    assert metrics["count"] == 2
    json.dumps(metrics, allow_nan=False)


def test_archived_summary_is_sanitized_and_internally_consistent() -> None:
    path = Path("reproduce/diagnostics/archived/report_summary.json")
    raw = path.read_text(encoding="utf-8")
    lowered = raw.lower()
    # Build sentinels from fragments so the repository-wide release audit does
    # not mistake this negative test itself for leaked infrastructure metadata.
    forbidden = (
        "s3" + "://",
        "/work" + "space/",
        "/vo" + "lt/",
        "job" + "_id",
        "access" + "_key",
        "sec" + "ret",
    )
    assert all(token not in lowered for token in forbidden)

    payload = json.loads(raw)
    assert payload["evidence_class"] == "archived_report_evidence"
    studies = payload["studies"]
    assert studies["tiny_grid_regression"]["parameter_count"] == 81_920
    assert studies["final_grid_native_witnesses"]["matches_by_denominator"]["1024"] == 258
    assert studies["model_1_292b_training_100_step"]["final_gradient_different_elements"] == 1
