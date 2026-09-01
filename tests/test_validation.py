#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ue5m3_fp4.eval import evaluate_validation
from ue5m3_fp4.eval.validation import _load_source, _prepare_sources


class _NextTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        self.forward_calls += 1
        predictions = (input_ids + 1) % self.vocab_size
        logits = torch.zeros(
            *input_ids.shape,
            self.vocab_size,
            device=input_ids.device,
        )
        logits.scatter_(-1, predictions.unsqueeze(-1), 2.0)
        return SimpleNamespace(logits=logits)


def test_tensor_validation_has_exact_sequence_and_token_accounting() -> None:
    tokens = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        dtype=torch.int32,
    )
    model = _NextTokenModel()
    callback_records: list[tuple[int, tuple[int, ...]]] = []

    result = evaluate_validation(
        model,
        tokens,
        checkpoint_id="toy-step-10",
        device="cpu",
        batch_size=1,
        ce_chunk_tokens=2,
        before_forward_callback=lambda input_ids, index: callback_records.append(
            (index, tuple(input_ids.shape))
        ),
    )

    expected_nll = math.log(math.exp(2.0) + 4.0) - 2.0
    assert result["checkpoint_id"] == "toy-step-10"
    assert result["metrics"]["token_count"] == 6
    assert result["validation_data"]["sequences"] == 2
    assert result["metrics"]["nll"] == pytest.approx(expected_nll)
    assert result["metrics"]["perplexity"] == pytest.approx(math.exp(expected_nll))
    assert result["per_sequence"]["token_counts"] == [3, 3]
    assert [item["sequence_index"] for item in result["per_sequence"]["provenance"]] == [0, 1]
    assert math.fsum(result["per_sequence"]["loss_sums"]) == pytest.approx(
        result["metrics"]["loss_sum"]
    )
    assert callback_records == [(1, (1, 3)), (2, (1, 3))]
    assert model.forward_calls == 2


def test_local_safetensors_and_tensor_sources_preserve_order(tmp_path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    save_file(
        {"tokens": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)},
        shard_dir / "b.safetensors",
    )
    save_file(
        {"tokens": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)},
        shard_dir / "a.safetensors",
    )
    final_tensor = torch.tensor([2, 3, 4, 0], dtype=torch.int32)

    result = evaluate_validation(
        _NextTokenModel(),
        [shard_dir, final_tensor],
        checkpoint_id="mixed-local-inputs",
        device="cpu",
        batch_size=2,
    )

    sources = result["validation_data"]["sources"]
    assert [source["kind"] for source in sources] == [
        "safetensors",
        "safetensors",
        "tensor",
    ]
    assert sources[0]["identifier"].startswith("safetensors-00000-")
    assert sources[1]["identifier"].startswith("safetensors-00001-")
    assert sources[2]["identifier"] == "tensor-00002"
    assert str(tmp_path) not in repr(sources)
    assert all(len(source["sha256"]) == 64 for source in sources)
    assert [item["sequence_index"] for item in result["per_sequence"]["provenance"]] == [
        0,
        1,
        2,
    ]
    assert result["metrics"]["token_count"] == 9
    assert result["evaluation"]["forward_count"] == 3


def test_chunked_lm_head_matches_full_model_path() -> None:
    class _Backbone(torch.nn.Module):
        def forward(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                last_hidden_state=torch.nn.functional.one_hot(
                    input_ids,
                    num_classes=5,
                ).float()
            )

    class _ChunkedModel(_NextTokenModel):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _Backbone()
            self.lm_head = torch.nn.Linear(5, 5, bias=False)
            with torch.no_grad():
                self.lm_head.weight.zero_()
                for token in range(5):
                    self.lm_head.weight[(token + 1) % 5, token] = 2.0

    tokens = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 0]], dtype=torch.int32)
    full = evaluate_validation(
        _NextTokenModel(),
        tokens,
        checkpoint_id="full",
        device="cpu",
        ce_chunk_tokens=1,
    )
    chunked = evaluate_validation(
        _ChunkedModel(),
        tokens,
        checkpoint_id="chunked",
        device="cpu",
        ce_chunk_tokens=1,
        logit_path="chunked_lm_head",
    )

    assert chunked["metrics"]["loss_sum"] == pytest.approx(full["metrics"]["loss_sum"])
    assert chunked["per_sequence"]["loss_sums"] == pytest.approx(
        full["per_sequence"]["loss_sums"]
    )


@pytest.mark.parametrize(
    ("inputs", "error_type", "message"),
    [
        (torch.ones(2, 3), TypeError, "int32 or torch.int64"),
        (torch.ones(2, 1, dtype=torch.int32), ValueError, "at least two tokens"),
        (
            [
                torch.ones(1, 4, dtype=torch.int32),
                torch.ones(1, 5, dtype=torch.int32),
            ],
            ValueError,
            "same sequence length",
        ),
    ],
)
def test_invalid_validation_inputs_fail_closed(inputs, error_type, message) -> None:
    with pytest.raises(error_type, match=message):
        evaluate_validation(
            _NextTokenModel(),
            inputs,
            checkpoint_id="invalid",
            device="cpu",
        )


def test_labels_and_empty_checkpoint_identity_are_rejected() -> None:
    tokens = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    with pytest.raises(ValueError, match="checkpoint_id"):
        evaluate_validation(
            _NextTokenModel(),
            tokens,
            checkpoint_id="",
            device="cpu",
        )
    with pytest.raises(ValueError, match="Do not pass labels"):
        evaluate_validation(
            _NextTokenModel(),
            tokens,
            checkpoint_id="toy",
            device="cpu",
            forward_kwargs={"labels": tokens[:, 1:]},
        )


def test_duplicate_path_separated_by_tensor_is_rejected(tmp_path) -> None:
    validation_path = tmp_path / "validation.safetensors"
    save_file(
        {"tokens": torch.tensor([[0, 1, 2]], dtype=torch.int32)},
        validation_path,
    )

    with pytest.raises(ValueError, match="provided more than once"):
        evaluate_validation(
            _NextTokenModel(),
            [
                validation_path,
                torch.tensor([[0, 1, 2]], dtype=torch.int32),
                validation_path,
            ],
            checkpoint_id="duplicate-path",
            device="cpu",
        )


def test_prepared_tensor_owns_storage_after_identity_is_recorded() -> None:
    tokens = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    prepared = _prepare_sources(tokens, tensor_key="tokens")
    tokens.fill_(4)

    loaded = _load_source(prepared[0], tensor_key="tokens")

    torch.testing.assert_close(loaded, torch.tensor([[0, 1, 2]], dtype=torch.int32))


def test_same_shape_file_replacement_is_rejected_after_identity(tmp_path) -> None:
    validation_path = tmp_path / "validation.safetensors"
    save_file(
        {"tokens": torch.tensor([[0, 1, 2]], dtype=torch.int32)},
        validation_path,
    )
    prepared = _prepare_sources(validation_path, tensor_key="tokens")
    save_file(
        {"tokens": torch.tensor([[2, 1, 0]], dtype=torch.int32)},
        validation_path,
    )

    with pytest.raises(RuntimeError, match="changed after"):
        _load_source(prepared[0], tensor_key="tokens")
