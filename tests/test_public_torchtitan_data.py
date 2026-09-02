# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ue5m3_fp4.integrations.torchtitan.config import PublicData
from ue5m3_fp4.integrations.torchtitan.data import PublicTwoStreamTextDataset

datasets = pytest.importorskip("datasets")


class _Tokenizer:
    def encode(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        assert add_bos and add_eos
        return [1, *[ord(character) for character in text], 2]


def _write_stream(root: Path, stream: str) -> None:
    for shard in range(2):
        dataset = datasets.Dataset.from_dict(
            {"text": [f"{stream}-{shard}-a", f"{stream}-{shard}-b"]}
        )
        dataset.save_to_disk(root / stream / f"shard-{shard:03d}-of-002")


def _dataset(root: Path) -> PublicTwoStreamTextDataset:
    return PublicTwoStreamTextDataset(
        root=root,
        tokenizer=_Tokenizer(),
        seq_len=7,
        dp_rank=0,
        dp_world_size=1,
        expected_shards=2,
    )


def test_public_data_configuration_rejects_non_unit_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        PublicData(dclm_weight=0.9, no_dclm_weight=0.2)


def test_document_schedule_has_exact_82_18_period(tmp_path: Path) -> None:
    for stream in ("dclm", "olmo-no-dclm"):
        _write_stream(tmp_path, stream)
    dataset = _dataset(tmp_path)
    selections = [dataset._choose_stream() for _ in range(100)]
    assert selections.count("dclm") == 82
    assert selections.count("olmo-no-dclm") == 18


def test_packing_cursor_and_buffer_round_trip(tmp_path: Path) -> None:
    for stream in ("dclm", "olmo-no-dclm"):
        _write_stream(tmp_path, stream)
    original = _dataset(tmp_path)
    iterator = iter(original)
    first_inputs, first_labels = next(iterator)
    state = original.state_dict()
    expected_inputs, expected_labels = next(iterator)

    restored = _dataset(tmp_path)
    restored.load_state_dict(state)
    actual_inputs, actual_labels = next(iter(restored))

    assert torch.equal(first_inputs["input"][1:], first_labels[:-1])
    assert torch.equal(actual_inputs["input"], expected_inputs["input"])
    assert torch.equal(actual_labels, expected_labels)
    assert restored.state_dict() == original.state_dict()
