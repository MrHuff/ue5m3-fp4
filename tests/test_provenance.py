# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHED_OUTPUTS = {
    "Extracted `formats.py`": "src/ue5m3_fp4/formats.py",
    "Extracted `recipe.py`": "src/ue5m3_fp4/recipe.py",
    "Extracted `scaling/training.py`": "src/ue5m3_fp4/scaling/training.py",
    "Extracted `scaling/inference.py`": "src/ue5m3_fp4/scaling/inference.py",
    "Extracted `nn/linear.py`": "src/ue5m3_fp4/nn/linear.py",
    "Extracted `nn/convert.py`": "src/ue5m3_fp4/nn/convert.py",
    "Extracted `eval/validation.py`": "src/ue5m3_fp4/eval/validation.py",
    "Packaged recipe-resource API": "src/ue5m3_fp4/recipes/__init__.py",
    "Packaged `proposed_b16_d50.yaml`": ("src/ue5m3_fp4/recipes/proposed_b16_d50.yaml"),
    "Packaged `current_tensor_d1.yaml`": (
        "src/ue5m3_fp4/recipes/inference/current_tensor_d1.yaml"
    ),
    "Packaged `training_replay_d50.yaml`": (
        "src/ue5m3_fp4/recipes/inference/training_replay_d50.yaml"
    ),
    "Packaged `calibrated_frozen.yaml`": (
        "src/ue5m3_fp4/recipes/inference/calibrated_frozen.yaml"
    ),
}


def test_documented_release_candidate_hashes_match_files() -> None:
    provenance = (ROOT / "SOURCE_PROVENANCE.md").read_text(encoding="utf-8")
    for label, relative_path in HASHED_OUTPUTS.items():
        match = re.search(
            rf"^\| {re.escape(label)} \| `([0-9a-f]{{64}})` \|$",
            provenance,
            flags=re.MULTILINE,
        )
        assert match is not None, f"missing provenance row for {label}"
        observed = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert observed == match.group(1), f"stale provenance hash for {relative_path}"
