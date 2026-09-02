# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy access to the reported Triton numerical path.

Importing this module does not import Triton or initialize CUDA.  Backend
dependencies are loaded only when an operation is called.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def triton_available() -> bool:
    """Return whether Torch, Triton, and an available CUDA device are present."""

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
        return False
    import torch

    return bool(torch.version.cuda is not None and torch.cuda.is_available())


def fake_quantize_gemm_operands(*args, **kwargs) -> tuple[Tensor, Tensor]:
    """Lazy wrapper for the reported E2M1-with-UE5M3 operand quantizer."""

    from ue5m3_fp4.backends.triton.quantization import (
        fake_quantize_gemm_operands as implementation,
    )

    return implementation(*args, **kwargs)


def issue_rz_bf16_gemm(*args, **kwargs) -> Tensor:
    """Lazy wrapper for the K=64 issue-RZ BF16 GEMM."""

    from ue5m3_fp4.backends.triton.gemm import issue_rz_bf16_gemm as implementation

    return implementation(*args, **kwargs)


def probe_matched_fp4_gemm(*args, **kwargs) -> Tensor:
    """Lazy wrapper for the complete probe-matched fake-FP4 GEMM."""

    from ue5m3_fp4.backends.triton.api import probe_matched_fp4_gemm as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "fake_quantize_gemm_operands",
    "issue_rz_bf16_gemm",
    "probe_matched_fp4_gemm",
    "triton_available",
]
