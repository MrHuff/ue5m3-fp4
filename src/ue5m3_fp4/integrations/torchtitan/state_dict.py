# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TorchTitan/Hugging Face state-dict mapping for Nemotron-H.

The public TorchTitan wrapper exposes the recovered native roots
``tok_embeddings``, ``layers``, ``norm``, and ``output``.  Hugging Face stores
the same tensors below ``backbone.embeddings``, ``backbone.layers``,
``backbone.norm_f``, and ``lm_head``.  Tensor layouts are otherwise identical.
"""

from __future__ import annotations

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from ue5m3_fp4.integrations.torchtitan.nemotron_h import NemotronH8BArgs

_ROOTS = (
    ("tok_embeddings.", "backbone.embeddings."),
    ("layers.", "backbone.layers."),
    ("norm.", "backbone.norm_f."),
    ("output.", "lm_head."),
)


class NemotronHStateDictAdapter(StateDictAdapter):
    """Map the public TorchTitan roots to standard Nemotron-H HF names.

    The conversion is deliberately strict.  Silently passing an unexpected
    key could produce a syntactically valid but incomplete export, which is
    substantially worse than failing before checkpoint I/O begins.
    """

    def __init__(
        self,
        model_args: NemotronH8BArgs,
        hf_assets_path: str | None,
    ) -> None:
        if not isinstance(model_args, NemotronH8BArgs):
            raise TypeError("model_args must be NemotronH8BArgs")
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args
        self.hf_assets_path = hf_assets_path

    @staticmethod
    def _replace_root(key: str, *, to_hf: bool) -> str:
        if not isinstance(key, str) or not key:
            raise ValueError("state-dict keys must be non-empty strings")
        for native_root, hf_root in _ROOTS:
            source, target = (native_root, hf_root) if to_hf else (hf_root, native_root)
            if key.startswith(source):
                return f"{target}{key.removeprefix(source)}"
        side = "TorchTitan" if to_hf else "Hugging Face"
        expected = [pair[0 if to_hf else 1] for pair in _ROOTS]
        raise KeyError(f"unexpected Nemotron-H {side} key {key!r}; expected roots {expected}")

    @staticmethod
    def _insert_unique(result: dict[str, Any], key: str, value: Any) -> None:
        if key in result:
            raise ValueError(f"state-dict conversion produced duplicate key {key!r}")
        result[key] = value

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Rename recovered TorchTitan roots to standard HF roots."""

        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dict")
        converted: dict[str, Any] = {}
        for native_key, value in state_dict.items():
            hf_key = self._replace_root(native_key, to_hf=True)
            self._insert_unique(converted, hf_key, value)
        return converted

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Rename standard HF roots to recovered TorchTitan roots."""

        if not isinstance(hf_state_dict, dict):
            raise TypeError("hf_state_dict must be a dict")
        converted: dict[str, Any] = {}
        for hf_key, value in hf_state_dict.items():
            native_key = self._replace_root(hf_key, to_hf=False)
            self._insert_unique(converted, native_key, value)
        return converted


__all__ = ["NemotronHStateDictAdapter"]
