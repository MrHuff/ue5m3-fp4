#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed when the public runtime does not match its checked-in lock."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

EXPECTED = {
    "PyYAML": "6.0.2",
    "datasets": "3.6.0",
    "huggingface-hub": "0.36.0",
    "safetensors": "0.6.2",
    "tensorboard": "2.20.0",
    "tokenizers": "0.21.4",
    "torchdata": "0.11.0",
    "transformers": "4.48.2",
    "tyro": "1.0.5",
    "causal-conv1d": "1.6.2.post1",
    "cut-cross-entropy": "25.9.3",
    "mamba-ssm": "2.3.2.post1",
}
TORCHTITAN_REVISION = "e37f83f58b35fdbceed9a5916b3490c16247ac9c"
TRANSFORMER_ENGINE_REVISION = "01aef4fc721bd12fd09cd56d53a314aee1b953d6"
TRANSFORMER_ENGINE_VERSION = "2.16.0.dev0+01aef4fc"
EXPECTED_TORCH = "2.9.0a0+145a3a7bda.nv25.10"
EXPECTED_TRITON = "3.5.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-no-gpu", action="store_true")
    args = parser.parse_args()

    mismatches = []
    packages = {}
    for name, expected in EXPECTED.items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = None
        packages[name] = actual
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual}")

    import torch
    import transformer_engine
    import triton

    from ue5m3_fp4.integrations.torchtitan.nemotron_h import TORCHTITAN_REVISION as recorded

    if recorded != TORCHTITAN_REVISION:
        mismatches.append(f"TorchTitan source pin differs: {recorded}")
    if transformer_engine.__version__ != TRANSFORMER_ENGINE_VERSION:
        mismatches.append(
            "Transformer Engine: expected "
            f"{TRANSFORMER_ENGINE_VERSION}, got {transformer_engine.__version__}"
        )
    repository = Path(__file__).resolve().parents[2]
    try:
        te_revision = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository / "third_party/TransformerEngine"),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        te_revision = None
        mismatches.append(f"could not verify Transformer Engine Git revision: {error}")
    if te_revision is not None and te_revision != TRANSFORMER_ENGINE_REVISION:
        mismatches.append(
            f"Transformer Engine source: expected {TRANSFORMER_ENGINE_REVISION}, got {te_revision}"
        )
    if torch.__version__ != EXPECTED_TORCH:
        mismatches.append(f"torch: expected {EXPECTED_TORCH}, got {torch.__version__}")
    if triton.__version__ != EXPECTED_TRITON:
        mismatches.append(f"triton: expected {EXPECTED_TRITON}, got {triton.__version__}")
    if torch.version.cuda is None:
        mismatches.append("PyTorch is not a CUDA build")
    if not torch.cuda.is_available() and not args.allow_no_gpu:
        mismatches.append("CUDA is unavailable")

    result = {
        "schema": "ue5m3_public_runtime_check_v1",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": packages,
        "torchtitan_revision": recorded,
        "transformer_engine_version": transformer_engine.__version__,
        "transformer_engine_revision": te_revision,
        "mismatches": mismatches,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit("runtime verification failed: " + "; ".join(mismatches))


if __name__ == "__main__":
    main()
