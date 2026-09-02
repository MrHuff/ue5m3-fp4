# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ue5m3_fp4.integrations.torchtitan.selection import FP32OutputLinear
from ue5m3_fp4.olmes_runtime import (
    controlled_hflm_kwargs,
    install_fp32_output_head,
    install_olmes_filename_compatibility,
)

_ROOT = Path(__file__).resolve().parents[1]
_WRAPPER = _ROOT / "reproduce" / "evaluation" / "run_public_olmes.sh"
_REPLAY = _ROOT / "reproduce" / "scripts" / "prepare_olmes_replay.py"


def test_public_olmes_wrapper_shell_syntax_and_fixed_contract() -> None:
    subprocess.run(["bash", "-n", str(_WRAPPER)], check=True)
    source = _WRAPPER.read_text(encoding="utf-8")
    for required in (
        "8e2743734066b073c5d8498d1b8220f67a21a2d6",
        "core_9mcqa::olmes",
        "mmlu::olmes",
        "mmlu_pro:mc::none",
        "--random-subsample-seed 20260830",
        "--batch-size 8",
        "--gpus 1",
        "--num-workers 1",
        '"max_length":2048',
        "UE5M3_PUBLIC_OLMES_RUNTIME=1",
        "ue5m3-olmes-runtime.json",
    ):
        assert required in source


def test_public_olmes_wrapper_fails_closed_for_unknown_numeric_path(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["UE5M3_OLMES_NUMERIC_PATH"] = "not-a-released-path"
    completed = subprocess.run(
        [str(_WRAPPER), str(tmp_path / "model"), str(tmp_path / "output")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 2
    assert "unsupported OLMES numeric path" in completed.stderr
    assert "silently replaced with BF16" in completed.stderr


def test_public_olmes_wrapper_names_all_released_numeric_paths() -> None:
    source = _WRAPPER.read_text(encoding="utf-8")
    for numeric_path in (
        "bf16",
        "ue5m3-proposed-b16",
        "ue5m3-proposed-b32",
        "ue5m3-torch-control",
        "ue5m3-te-settings",
        "native-nvfp4-te",
        "native-nvfp4-no-rht-all",
    ):
        assert numeric_path in source


def test_public_olmes_wrapper_rejects_unavailable_frozen_replay(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "modeling_nemotron_h.py").write_text("# test\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test")
    environment = dict(os.environ)
    environment["UE5M3_OLMES_REQUEST_MODE"] = "frozen_request_archive"
    completed = subprocess.run(
        [str(_WRAPPER), str(model), str(tmp_path / "output")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 2
    assert "0bf27af57eb1bb1b98872c4af12d419498652d935a6b745cc7ec4ecdb32d7483" in (
        completed.stderr
    )


def test_frozen_replay_patch_matches_reviewed_olmes_source(tmp_path: Path) -> None:
    source = _ROOT / "third_party" / "olmes" / "oe_eval" / "run_eval.py"
    copied = tmp_path / "run_eval.py"
    shutil.copyfile(source, copied)

    subprocess.run(
        ["python", str(_REPLAY), "patch-olmes", "--source", str(copied)],
        check=True,
    )

    assert hashlib.sha256(copied.read_bytes()).hexdigest() == (
        "fde74f9c7aab5efd0ed22ddaf776d1375eef3dea55293bfb36ff3ec5f23ab80a"
    )
    assert b'os.environ.get("OLMES_REUSE_RAW_REQUESTS") == "1"' in copied.read_bytes()


def test_olmes_runtime_disables_logits_cache() -> None:
    assert controlled_hflm_kwargs({"trust_remote_code": True}) == {
        "trust_remote_code": True,
        "logits_cache": False,
    }
    assert controlled_hflm_kwargs({"logits_cache": False})["logits_cache"] is False
    with pytest.raises(RuntimeError, match="enabled logits cache"):
        controlled_hflm_kwargs({"logits_cache": True})


def test_olmes_runtime_restores_colon_free_historical_filenames(tmp_path: Path) -> None:
    olmes_utils = SimpleNamespace()

    record = install_olmes_filename_compatibility(olmes_utils)

    name = olmes_utils.task_file_name(
        str(tmp_path),
        7,
        "mmlu:mc::olmes",
        "requests.jsonl",
    )
    assert ":" not in name
    assert name.endswith("task-007-mmlu-mc--olmes-requests.jsonl")
    olmes_utils.save_jsonl(name, [{"request": 1}])
    assert olmes_utils.load_jsonl(name) == [{"request": 1}]
    assert record["public_base_revision"] == ("8e2743734066b073c5d8498d1b8220f67a21a2d6")
    assert record["historical_revision"] == ("3d80ebb0f08706a5d2dd3fb0be72100735b5f5c6")


def test_olmes_runtime_installs_identity_preserving_fp32_head() -> None:
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    checkpoint_parameter = model.lm_head.weight

    record = install_fp32_output_head(model)

    assert isinstance(model.lm_head, FP32OutputLinear)
    assert model.lm_head.weight is checkpoint_parameter
    assert record["compute_dtype"] == "float32"
    assert record["parameter_dtype"] == "bfloat16"
