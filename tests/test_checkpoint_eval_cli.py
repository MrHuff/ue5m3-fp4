# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ue5m3_fp4.checkpoint import NEMOTRON_H_ASSET_REVISION
from ue5m3_fp4.cli import evaluate as evaluate_cli
from ue5m3_fp4.integrations.torchtitan.selection import FP32OutputLinear


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {"last_hidden_state": self.embedding(input_ids)}


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


def _arguments(tmp_path: Path, validation: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "checkpoint": tmp_path / "checkpoint",
        "validation": [validation],
        "output": tmp_path / "result.json",
        "numeric_path": "bf16",
        "activation_mode": None,
        "calibration": None,
        "hf_assets": "unused-in-test",
        "hf_revision": NEMOTRON_H_ASSET_REVISION,
        "local_files_only": True,
        "device": "cpu",
        "tensor_key": "tokens",
        "batch_size": 1,
        "ce_chunk_tokens": 2,
        "expected_input_tokens": 3,
        "allow_partial_data": True,
        "quiet": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_bf16_cli_run_evaluates_local_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = tmp_path / "validation.safetensors"
    save_file({"tokens": torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)}, validation)
    model = _Model().eval()
    load_provenance = {
        "checkpoint": {"sha256": "a" * 64},
        "assets": {"sha256": "b" * 64},
    }
    monkeypatch.setattr(
        evaluate_cli,
        "load_hf_nemotron_h_checkpoint",
        lambda *_args, **_kwargs: (model, load_provenance),
    )

    result = evaluate_cli.run(_arguments(tmp_path, validation))

    assert result["schema"] == evaluate_cli.EVALUATION_RUN_SCHEMA
    assert result["checkpoint_id"] == "a" * 64
    assert result["metrics"]["token_count"] == 3
    assert result["model"]["numeric_path"]["fp4_quantization_applied"] is False
    assert result["model"]["conversion"]["output_compute_dtype"] == "float32"
    assert isinstance(model.lm_head, FP32OutputLinear)
    assert result["validation_data"]["identity"]["sequence_count"] == 1
    assert len(result["result_sha256"]) == 64


def test_token_dataset_identity_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.safetensors"
    row = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    save_file({"tokens": torch.stack((row, row))}, path)
    with pytest.raises(ValueError, match="duplicate sequence"):
        evaluate_cli.token_dataset_identity(
            [path],
            tensor_key="tokens",
            expected_input_tokens=3,
            expected_sequences=None,
        )


def test_cli_rejects_fp4_on_cpu_before_loading(tmp_path: Path) -> None:
    validation = tmp_path / "validation.safetensors"
    save_file({"tokens": torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)}, validation)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        evaluate_cli.run(
            _arguments(
                tmp_path,
                validation,
                numeric_path="ue5m3-b16",
                activation_mode="current_tensor",
            )
        )


def test_cli_rejects_activation_mode_for_bf16(tmp_path: Path) -> None:
    validation = tmp_path / "validation.safetensors"
    save_file({"tokens": torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)}, validation)
    with pytest.raises(ValueError, match="only to UE5M3"):
        evaluate_cli.run(_arguments(tmp_path, validation, activation_mode="training_replay"))


def test_parser_pins_public_nemotron_revision(tmp_path: Path) -> None:
    args = evaluate_cli.build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--validation",
            str(tmp_path / "validation.safetensors"),
            "--output",
            str(tmp_path / "result.json"),
            "--numeric-path",
            "bf16",
        ]
    )
    assert args.hf_revision == NEMOTRON_H_ASSET_REVISION


def test_parser_exposes_every_reported_evaluation_path(tmp_path: Path) -> None:
    parser = evaluate_cli.build_parser()
    for numeric_path in (
        "bf16",
        "ue5m3-proposed-b16",
        "ue5m3-proposed-b32",
        "ue5m3-torch-control",
        "ue5m3-te-settings",
        "native-nvfp4-te",
        "native-nvfp4-no-rht-all",
    ):
        parsed = parser.parse_args(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--validation",
                str(tmp_path / "validation.safetensors"),
                "--output",
                str(tmp_path / "result.json"),
                "--numeric-path",
                numeric_path,
            ]
        )
        assert parsed.numeric_path == numeric_path


def test_public_token_freezer_creates_disjoint_hashed_splits(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"tokens": row}) for row in ([0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5])
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "frozen"
    script = (
        Path(__file__).resolve().parents[1]
        / "reproduce"
        / "scripts"
        / "prepare_validation_tokens.py"
    )
    subprocess.run(
        [
            "python",
            str(script),
            "--input-jsonl",
            str(source),
            "--output",
            str(output),
            "--validation-rows",
            "2",
            "--calibration-rows",
            "1",
            "--sequence-length",
            "3",
            "--shard-rows",
            "1",
        ],
        check=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_validation_identity_reproduced"] is False
    assert manifest["validation"]["records"] == 2
    assert manifest["calibration"]["records"] == 1
    _, _, validation_hashes = evaluate_cli.token_dataset_identity(
        [output / "validation"],
        tensor_key="tokens",
        expected_input_tokens=3,
        expected_sequences=2,
    )
    _, _, calibration_hashes = evaluate_cli.token_dataset_identity(
        [output / "calibration"],
        tensor_key="tokens",
        expected_input_tokens=3,
        expected_sequences=1,
    )
    assert validation_hashes.isdisjoint(calibration_hashes)
