# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage-neutral two-stream input pipeline for public method reproduction.

The historical run used a private Mosaic stream whose byte-level shard index
and scheduler cursor were not recovered.  This module consumes the public
``Dataset.save_to_disk`` output produced by ``prepare_olmo_mix_1124.py`` and
reconstructs the documented 82/18 mix and persistent token-buffer packing.  It
is deliberately deterministic and checkpointable, but its row order is a new
public dataset identity rather than the unrecoverable historical identity.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info

PUBLIC_DATASET_NAME = "ue5m3_public_olmo_mix_1124"
_STREAMS = ("dclm", "olmo-no-dclm")


def _assigned_shards(
    root: Path,
    stream: str,
    *,
    dp_rank: int,
    dp_world_size: int,
    expected_shards: int,
    strict_inventory: bool,
) -> tuple[Path, ...]:
    stream_root = root / stream
    if not stream_root.is_dir():
        raise FileNotFoundError(f"prepared stream directory does not exist: {stream_root}")
    paths = tuple(sorted(path for path in stream_root.glob("shard-*-of-*") if path.is_dir()))
    if strict_inventory and len(paths) != expected_shards:
        raise RuntimeError(
            f"{stream_root} contains {len(paths)} prepared shards; expected {expected_shards}"
        )
    if not paths:
        raise RuntimeError(f"{stream_root} contains no prepared shard directories")
    assigned = tuple(
        path for index, path in enumerate(paths) if index % dp_world_size == dp_rank
    )
    if not assigned:
        raise RuntimeError(
            f"data-parallel rank {dp_rank} received no {stream!r} shards; "
            "use no more data-parallel ranks than prepared shards"
        )
    return assigned


def _load_assigned_stream(paths: tuple[Path, ...]):
    try:
        from datasets import concatenate_datasets, load_from_disk
    except ImportError as error:
        raise ImportError(
            "public OLMo Mix loading requires the optional 'datasets' dependency"
        ) from error

    datasets = [load_from_disk(str(path)) for path in paths]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


class PublicTwoStreamTextDataset(IterableDataset, Stateful):
    """Checkpointable 82/18 document mixer with persistent 8193-token packing."""

    def __init__(
        self,
        *,
        root: str | Path,
        tokenizer: Any,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        dclm_weight: float = 0.82,
        no_dclm_weight: float = 0.18,
        expected_shards: int = 32,
        text_column: str = "text",
        cycle: bool = True,
        strict_inventory: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"public_data.root does not exist: {self.root}")
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must expose encode(text, add_bos=..., add_eos=...)")
        if type(seq_len) is not int or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if type(dp_world_size) is not int or dp_world_size <= 0:
            raise ValueError("dp_world_size must be a positive integer")
        if type(dp_rank) is not int or not 0 <= dp_rank < dp_world_size:
            raise ValueError("dp_rank must be in [0, dp_world_size)")
        if not isinstance(text_column, str) or not text_column:
            raise ValueError("text_column must be a non-empty string")
        if type(cycle) is not bool:
            raise TypeError("cycle must be bool")
        if dclm_weight <= 0 or no_dclm_weight <= 0:
            raise ValueError("stream weights must be positive")
        if abs((dclm_weight + no_dclm_weight) - 1.0) > 1e-12:
            raise ValueError("stream weights must sum to one")

        fraction = Fraction(str(dclm_weight)).limit_denominator(10_000)
        if abs(float(fraction) - dclm_weight) > 1e-12:
            raise ValueError("dclm_weight must have a stable rational representation")

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.text_column = text_column
        self.cycle = cycle
        self._dclm_numerator = fraction.numerator
        self._mix_denominator = fraction.denominator
        self._datasets = {
            stream: _load_assigned_stream(
                _assigned_shards(
                    self.root,
                    stream,
                    dp_rank=dp_rank,
                    dp_world_size=dp_world_size,
                    expected_shards=expected_shards,
                    strict_inventory=strict_inventory,
                )
            )
            for stream in _STREAMS
        }
        for stream, dataset in self._datasets.items():
            if len(dataset) == 0:
                raise RuntimeError(f"assigned {stream!r} dataset is empty")
            if text_column not in dataset.column_names:
                raise RuntimeError(
                    f"assigned {stream!r} dataset lacks text column {text_column!r}"
                )

        self._positions = {stream: 0 for stream in _STREAMS}
        self._epochs = {stream: 0 for stream in _STREAMS}
        self._mix_position = 0
        self._token_buffer: list[int] = []
        self._windows_emitted = 0

    def _choose_stream(self) -> str:
        # The difference of consecutive ceilings gives a balanced deterministic
        # schedule with exactly numerator selections per denominator documents.
        current = self._mix_position
        numerator = self._dclm_numerator
        denominator = self._mix_denominator
        before = (current * numerator + denominator - 1) // denominator
        after = ((current + 1) * numerator + denominator - 1) // denominator
        self._mix_position += 1
        return "dclm" if after > before else "olmo-no-dclm"

    def _next_text(self, stream: str) -> str:
        dataset = self._datasets[stream]
        position = self._positions[stream]
        if position >= len(dataset):
            if not self.cycle:
                raise StopIteration
            position = 0
            self._positions[stream] = 0
            self._epochs[stream] += 1
        value = dataset[position][self.text_column]
        self._positions[stream] = position + 1
        if not isinstance(value, str):
            raise TypeError(f"{stream} row {position} column {self.text_column!r} is not text")
        return value

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError(
                "PublicTwoStreamTextDataset requires DataLoader num_workers=0 so "
                "each data-parallel rank's training process owns one exact "
                "buffer/cursor state"
            )
        window_size = self.seq_len + 1
        while True:
            while len(self._token_buffer) < window_size:
                stream = self._choose_stream()
                try:
                    text = self._next_text(stream)
                except StopIteration:
                    return
                tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                if not isinstance(tokens, list) or any(
                    type(item) is not int for item in tokens
                ):
                    raise TypeError("tokenizer.encode must return list[int]")
                self._token_buffer.extend(tokens)

            packed = torch.tensor(self._token_buffer[:window_size], dtype=torch.long)
            del self._token_buffer[:window_size]
            self._windows_emitted += 1
            yield {"input": packed[:-1]}, packed[1:]

    def state_dict(self) -> dict[str, Any]:
        """Serialize exact public row cursors and token-buffer leftovers."""

        return {
            "schema": "ue5m3_public_two_stream_state_v1",
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "positions": dict(self._positions),
            "epochs": dict(self._epochs),
            "mix_position": self._mix_position,
            "token_buffer": list(self._token_buffer),
            "windows_emitted": self._windows_emitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a state produced by the same rank and world size."""

        if state_dict.get("schema") != "ue5m3_public_two_stream_state_v1":
            raise ValueError("unsupported public two-stream dataloader state")
        if state_dict.get("dp_rank") != self.dp_rank:
            raise ValueError("dataloader state belongs to a different data-parallel rank")
        if state_dict.get("dp_world_size") != self.dp_world_size:
            raise ValueError("dataloader resharding is not supported")
        positions = state_dict.get("positions")
        epochs = state_dict.get("epochs")
        if set(positions or {}) != set(_STREAMS) or set(epochs or {}) != set(_STREAMS):
            raise ValueError("dataloader state has an invalid stream inventory")
        for stream in _STREAMS:
            position = positions[stream]
            epoch = epochs[stream]
            if type(position) is not int or not 0 <= position <= len(self._datasets[stream]):
                raise ValueError(f"invalid {stream} row position")
            if type(epoch) is not int or epoch < 0:
                raise ValueError(f"invalid {stream} epoch")
        mix_position = state_dict.get("mix_position")
        windows_emitted = state_dict.get("windows_emitted")
        token_buffer = state_dict.get("token_buffer")
        if type(mix_position) is not int or mix_position < 0:
            raise ValueError("invalid mix position")
        if type(windows_emitted) is not int or windows_emitted < 0:
            raise ValueError("invalid emitted-window count")
        if not isinstance(token_buffer, list) or any(
            type(item) is not int for item in token_buffer
        ):
            raise ValueError("invalid token buffer")
        if len(token_buffer) > self.seq_len:
            raise ValueError("token buffer is larger than one packing window")
        self._positions = dict(positions)
        self._epochs = dict(epochs)
        self._mix_position = mix_position
        self._token_buffer = list(token_buffer)
        self._windows_emitted = windows_emitted


def build_public_two_stream_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Any,
    job_config: Any,
    infinite: bool = True,
):
    """Build TorchTitan's checkpoint-aware loader over prepared public data."""

    if job_config.training.dataset != PUBLIC_DATASET_NAME:
        raise ValueError(
            f"the Nemotron-H public TrainSpec requires training.dataset={PUBLIC_DATASET_NAME!r}"
        )
    config = job_config.public_data
    dataset = PublicTwoStreamTextDataset(
        root=config.root,
        tokenizer=tokenizer,
        seq_len=job_config.training.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        dclm_weight=config.dclm_weight,
        no_dclm_weight=config.no_dclm_weight,
        expected_shards=config.expected_shards_per_stream,
        text_column=config.text_column,
        cycle=config.cycle and infinite,
        strict_inventory=config.strict_inventory,
    )
    try:
        from torchtitan.components.dataloader import ParallelAwareDataloader
    except ImportError as error:
        raise ImportError(
            "the public data builder requires the pinned TorchTitan revision"
        ) from error
    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
    )


__all__ = [
    "PUBLIC_DATASET_NAME",
    "PublicTwoStreamTextDataset",
    "build_public_two_stream_dataloader",
]
