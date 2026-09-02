# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys


def test_triton_backend_import_is_lazy() -> None:
    script = """
import sys
import ue5m3_fp4.backends.triton as backend
assert 'triton' not in sys.modules
assert callable(backend.fake_quantize_gemm_operands)
assert callable(backend.issue_rz_bf16_gemm)
assert callable(backend.probe_matched_fp4_gemm)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_triton_backend_exports_are_narrow() -> None:
    from ue5m3_fp4.backends import triton as backend

    assert backend.__all__ == [
        "fake_quantize_gemm_operands",
        "issue_rz_bf16_gemm",
        "probe_matched_fp4_gemm",
        "triton_available",
    ]
