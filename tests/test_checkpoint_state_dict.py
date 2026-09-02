# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TORCHTITAN_ROOT = _REPOSITORY_ROOT / "third_party" / "torchtitan"
if str(_TORCHTITAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TORCHTITAN_ROOT))

from ue5m3_fp4.checkpoint import CHECKPOINT_IDENTITY_SCHEMA, checkpoint_identity  # noqa: E402
from ue5m3_fp4.integrations.torchtitan.nemotron_h import NemotronH8BArgs  # noqa: E402
from ue5m3_fp4.integrations.torchtitan.state_dict import (  # noqa: E402
    NemotronHStateDictAdapter,
)


def _adapter(tmp_path: Path) -> NemotronHStateDictAdapter:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "backbone.embeddings.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    return NemotronHStateDictAdapter(NemotronH8BArgs(), str(tmp_path))


def test_nemotron_h_state_dict_adapter_round_trip(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    native = {
        "tok_embeddings.weight": torch.arange(6).reshape(2, 3),
        "layers.0.mixer.A_log": torch.tensor([1.0, 2.0]),
        "norm.weight": torch.ones(3),
        "output.weight": torch.arange(6).reshape(3, 2),
    }

    hf = adapter.to_hf(native)

    assert set(hf) == {
        "backbone.embeddings.weight",
        "backbone.layers.0.mixer.A_log",
        "backbone.norm_f.weight",
        "lm_head.weight",
    }
    round_trip = adapter.from_hf(hf)
    assert set(round_trip) == set(native)
    assert all(round_trip[key] is value for key, value in native.items())
    assert adapter.fqn_to_index_mapping == {
        "backbone.embeddings.weight": 1,
        "lm_head.weight": 2,
    }


@pytest.mark.parametrize(
    ("method", "state_dict"),
    [
        ("to_hf", {"backbone.embeddings.weight": torch.ones(1)}),
        ("to_hf", {"model.unrelated.weight": torch.ones(1)}),
        ("from_hf", {"model.backbone.embeddings.weight": torch.ones(1)}),
        ("from_hf", {"unrelated.weight": torch.ones(1)}),
    ],
)
def test_nemotron_h_state_dict_adapter_rejects_unknown_roots(
    tmp_path: Path,
    method: str,
    state_dict: dict[str, torch.Tensor],
) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(KeyError):
        getattr(adapter, method)(state_dict)


def test_nemotron_h_state_dict_adapter_checks_model_args(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="NemotronH8BArgs"):
        NemotronHStateDictAdapter(object(), str(tmp_path))  # type: ignore[arg-type]


def test_checkpoint_identity_validates_single_file_hf_export(tmp_path: Path) -> None:
    save_file(
        {
            "backbone.embeddings.weight": torch.arange(6).reshape(2, 3),
            "lm_head.weight": torch.arange(6).reshape(3, 2),
        },
        tmp_path / "model.safetensors",
    )

    identity = checkpoint_identity(tmp_path)

    assert identity["schema"] == CHECKPOINT_IDENTITY_SCHEMA
    assert identity["tensor_key_count"] == 2
    assert len(identity["tensor_keys_sha256"]) == 64
    assert len(identity["sha256"]) == 64


def test_checkpoint_identity_rejects_wrapper_names(tmp_path: Path) -> None:
    save_file(
        {"model.backbone.embeddings.weight": torch.ones(1)},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(ValueError, match="non-standard Nemotron-H"):
        checkpoint_identity(tmp_path)
