# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHED_OUTPUTS = (
    "src/ue5m3_fp4/formats.py",
    "src/ue5m3_fp4/recipe.py",
    "src/ue5m3_fp4/scaling/training.py",
    "src/ue5m3_fp4/scaling/inference.py",
    "src/ue5m3_fp4/nn/linear.py",
    "src/ue5m3_fp4/nn/convert.py",
    "src/ue5m3_fp4/backends/triton/_rounding.py",
    "src/ue5m3_fp4/backends/triton/quantization.py",
    "src/ue5m3_fp4/backends/triton/gemm.py",
    "src/ue5m3_fp4/backends/triton/api.py",
    "src/ue5m3_fp4/integrations/torchtitan/nemotron_h.py",
    "src/ue5m3_fp4/integrations/torchtitan/remote_code.py",
    "src/ue5m3_fp4/integrations/torchtitan/selection.py",
    "src/ue5m3_fp4/integrations/torchtitan/comparators.py",
    "src/ue5m3_fp4/integrations/torchtitan/linear_backend.py",
    "src/ue5m3_fp4/integrations/torchtitan/registration.py",
    "src/ue5m3_fp4/integrations/torchtitan/config.py",
    "src/ue5m3_fp4/integrations/torchtitan/data.py",
    "src/ue5m3_fp4/integrations/torchtitan/state_dict.py",
    "src/ue5m3_fp4/integrations/torchtitan/trainer.py",
    "src/ue5m3_fp4/checkpoint.py",
    "src/ue5m3_fp4/cli/evaluate.py",
    "src/ue5m3_fp4/olmes_runtime.py",
    "reproduce/configs/nemotron_h_8b_bf16.toml",
    "reproduce/configs/nemotron_h_8b_nvfp4_no_rht_all_linears.toml",
    "reproduce/configs/nemotron_h_8b_nvfp4_te.toml",
    "reproduce/configs/nemotron_h_8b_ue5m3_b16.toml",
    "reproduce/configs/nemotron_h_8b_ue5m3_b32.toml",
    "reproduce/configs/nemotron_h_8b_ue5m3_te_settings.toml",
    "reproduce/configs/nemotron_h_8b_ue5m3_torch_control.toml",
    "reproduce/manifests/data_preparation.yaml",
    "reproduce/manifests/reported_experiments.yaml",
    "reproduce/manifests/validation.yaml",
    "reproduce/manifests/olmes.yaml",
    "reproduce/reference_results/provenance.json",
    "reproduce/reference_results/generated/artifact_manifest.json",
    "reproduce/diagnostics/archived/report_summary.json",
    "reproduce/scale_target/archived/manifest.json",
    "reproduce/scale_target/historical_350m_provenance.json",
    "reproduce/Dockerfile",
    "reproduce/environment/requirements-ngc-25.10.lock",
    "reproduce/environment/compiled-requirements-ngc-25.10.lock",
    "reproduce/environment/olmes-cp312-linux-aarch64.lock",
    "src/ue5m3_fp4/recipes/__init__.py",
    "src/ue5m3_fp4/recipes/proposed_b16_d50.yaml",
    "src/ue5m3_fp4/recipes/inference/current_tensor_d1.yaml",
    "src/ue5m3_fp4/recipes/inference/training_replay_d50.yaml",
    "src/ue5m3_fp4/recipes/inference/calibrated_frozen.yaml",
)


def test_documented_release_candidate_hashes_match_files() -> None:
    provenance = (ROOT / "SOURCE_PROVENANCE.md").read_text(encoding="utf-8")
    for relative_path in HASHED_OUTPUTS:
        match = re.search(
            rf"^\| `{re.escape(relative_path)}` \| `([0-9a-f]{{64}})` \|$",
            provenance,
            flags=re.MULTILINE,
        )
        assert match is not None, f"missing provenance row for {relative_path}"
        observed = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert observed == match.group(1), f"stale provenance hash for {relative_path}"
