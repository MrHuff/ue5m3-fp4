#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export one public TorchTitan Nemotron-H DCP checkpoint to HF safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--hf-assets", required=True, type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    assets = args.hf_assets.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"DCP checkpoint directory does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")

    from ue5m3_fp4.integrations.torchtitan.registration import register_torchtitan

    register_torchtitan()
    from torchtitan.protocols.train_spec import get_train_spec

    train_spec = get_train_spec("nemotron_h_ue5m3")
    model_args = train_spec.model_args["8B_reported"]
    model_args.hf_assets_path = str(assets)

    from scripts.checkpoint_conversion.convert_to_hf import convert_to_hf

    output_dir.mkdir(parents=True, exist_ok=True)
    convert_to_hf(
        input_dir,
        output_dir,
        "nemotron_h_ue5m3",
        "8B_reported",
        assets,
    )


if __name__ == "__main__":
    main()
