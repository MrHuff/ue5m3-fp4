# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from reproduce.diagnostics.run_accumulator_statistics import config_from_args
from reproduce.diagnostics.run_final_grid_sweep import (
    _archived_native_matches,
    run_sweep,
)
from reproduce.diagnostics.run_native_gemm_oracle import (
    PackedCase,
    _decode,
    _permute_issue_groups,
    _sample_indices,
)
from reproduce.diagnostics.run_tiny_grid_regression import (
    _parse_denominators,
    _state_difference,
)


def test_statistics_quick_config_preserves_numerical_constraints() -> None:
    args = argparse.Namespace(
        device="cpu",
        seed=20260721,
        trials=4096,
        k=4096,
        chunk_size=256,
        issue_size=64,
        grid_denominator=1024,
        regression_batch=2048,
        regression_features=2048,
        cancellation_trials=4096,
        cancellation_issues=64,
        quick=True,
    )

    config = config_from_args(args)

    assert config["k"] % 64 == 0
    assert config["regression_features"] % 64 == 0
    assert config["operand_modes"] == ["bf16"]
    assert config["grid_denominator"] == 1024


def test_native_case_permutation_preserves_decoded_product_sum() -> None:
    generator = torch.Generator().manual_seed(7)
    activation = torch.randint(0, 16, (16, 128), generator=generator).to(torch.uint8)
    weight = torch.randint(0, 16, (16, 128), generator=generator).to(torch.uint8)
    activation_scales = torch.ones((16, 8), dtype=torch.float8_e4m3fn)
    weight_scales = torch.ones((16, 8), dtype=torch.float8_e4m3fn)
    case = PackedCase(activation, weight, activation_scales, weight_scales, 1.0)

    permuted = _permute_issue_groups(case, torch.tensor([1, 0]))
    original_product = (
        _decode(activation, activation_scales).float()
        @ _decode(weight, weight_scales).float().T
    )
    permuted_product = (
        _decode(permuted.activation_nibbles, permuted.activation_scales).float()
        @ _decode(permuted.weight_nibbles, permuted.weight_scales).float().T
    )

    assert torch.equal(original_product, permuted_product)


def test_native_sample_indices_are_seeded_and_bounded() -> None:
    first = _sample_indices(16, 32, 37, 11)
    second = _sample_indices(16, 32, 37, 11)

    assert torch.equal(first, second)
    assert first.unique().numel() == 37
    assert int(first.min()) >= 0
    assert int(first.max()) < 16 * 32


def test_denominator_parser_and_state_difference() -> None:
    assert _parse_denominators("0,256,1024") == [0, 256, 1024]
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_denominators("0,0")
    reference = {"weight": torch.tensor([1.0, 2.0])}
    exact = {"weight": torch.tensor([1.0, 2.0])}
    changed = {"weight": torch.tensor([1.0, 3.0])}

    assert _state_difference(exact, reference)["bit_exact"] is True
    difference = _state_difference(changed, reference)
    assert difference["bit_exact"] is False
    assert difference["differing_elements"] == 1


def test_final_grid_sweep_marks_archived_and_rerunnable_components() -> None:
    archived = _archived_native_matches(
        Path("reproduce/diagnostics/archived/report_summary.json")
    )
    config = {
        "denominators": [0, 1024],
        "trials": 32,
        "issues": 64,
        "partial_scales": [1.0],
        "residual_sigmas": [0.0, 1 / 1024],
        "seed": 9,
    }

    records = run_sweep(config, torch.device("cpu"))

    assert archived["evidence_class"] == "archived_report_evidence"
    assert archived["matches_by_denominator"]["1024"] == 258
    assert len(records) == 4
    assert {row["denominator"] for row in records} == {0, 1024}
