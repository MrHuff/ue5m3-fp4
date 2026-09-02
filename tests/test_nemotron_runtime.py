# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ue5m3_fp4.integrations.torchtitan.nemotron_h import (
    NemotronH8BArgs,
    configure_nemotron_h_sdpa,
    verify_nemotron_h_sdpa,
)


class _EagerAttention(nn.Module):
    def __init__(self, _config=None, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(4, 4, bias=False)


class _SDPAAttention(_EagerAttention):
    pass


NEMOTRONH_ATTENTION_CLASSES = {"sdpa": _SDPAAttention}


class _Layer(nn.Module):
    def __init__(self, block_type: str, index: int) -> None:
        super().__init__()
        self.block_type = block_type
        self.mixer = _EagerAttention(layer_idx=index)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            num_attention_heads=1,
            head_dim=4,
            _attn_implementation="eager",
        )
        self.layers = nn.ModuleList([_Layer("attention", 0), _Layer("mlp", 1)])


class _ReadOnlyBlockTypeConfig:
    @property
    def layers_block_type(self) -> list[str]:
        return [
            "mamba" if kind == "M" else "attention" if kind == "*" else "mlp"
            for kind in self.hybrid_override_pattern
        ]


def test_model_args_use_read_only_remote_block_type_property() -> None:
    config = _ReadOnlyBlockTypeConfig()

    NemotronH8BArgs().apply_to_config(config)

    assert len(config.layers_block_type) == 52
    assert config.layers_block_type.count("attention") == 4
    assert config.layers_block_type.count("mamba") == 24
    assert config.layers_block_type.count("mlp") == 24


def test_sdpa_conversion_copies_weights_and_verifies() -> None:
    model = _Model()
    with torch.no_grad():
        model.layers[0].mixer.q_proj.weight.fill_(0.25)
    record = configure_nemotron_h_sdpa(model, cudnn_enabled=False)
    assert record["attention_mixers"] == ["layers.0.mixer"]
    assert record["converted_mixers"] == ["layers.0.mixer"]
    assert isinstance(model.layers[0].mixer, _SDPAAttention)
    assert torch.equal(
        model.layers[0].mixer.q_proj.weight,
        torch.full((4, 4), 0.25),
    )
    assert verify_nemotron_h_sdpa(model) == ("layers.0.mixer",)


def test_sdpa_verifier_rejects_record_without_converted_modules() -> None:
    model = _Model()
    model.config._attn_implementation = "sdpa"
    with pytest.raises(RuntimeError, match="not SDPA"):
        verify_nemotron_h_sdpa(model)
