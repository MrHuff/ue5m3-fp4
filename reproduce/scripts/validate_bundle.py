#!/usr/bin/env python3
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate syntax and paper-critical invariants in the reproduction bundle."""

from __future__ import annotations

import hashlib
import sys
import tomllib
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is declared by the package
    raise SystemExit(
        "PyYAML is required: install the project before running this check"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
MANIFEST_ROOT = ROOT / "manifests"
HISTORICAL_ROOT = ROOT / "historical_specs"
OLMES_LOCK = ROOT / "environment" / "olmes-cp312-linux-aarch64.lock"
OLMES_LOCK_SHA256 = "ccb078f4c1c87766931e2dafa0c6002640c2d35d5c7e41965ea3531f3bc16da5"
TRAINING_LOCK = ROOT / "environment" / "requirements-ngc-25.10.lock"
TRAINING_LOCK_SHA256 = "ecc6ebeb68655bf9f236ac1744db585bb7c3dc6b56bd497cf47e2c2c838205fc"
COMPILED_LOCK = ROOT / "environment" / "compiled-requirements-ngc-25.10.lock"
COMPILED_LOCK_SHA256 = "da1d0ecceb06b6c6d0b57da57dc70c84aa6b5cb702d73ba91256b1e5afc041b6"
NGC_BASE_MANIFEST = "sha256:5c8302e4628ac326c412368675cc462b8aee2f96326a3bf817304a83816f179a"
PUBLIC_OLMES_REPOSITORY = "https://github.com/allenai/olmes.git"
PUBLIC_OLMES_COMMIT = "8e2743734066b073c5d8498d1b8220f67a21a2d6"
PUBLIC_OLMES_TREE = "991940b9a2b37f8491ff29d1d22487b209fe750f"

EXPECTED_CONFIGS = {
    "nemotron_h_8b_bf16.toml": {"converter": [], "recipe": None},
    "nemotron_h_8b_ue5m3_b16.toml": {
        "converter": ["ue5m3_fp4.reported_b16_d50"],
        "recipe": "ue5m3",
        "block_size": 16,
    },
    "nemotron_h_8b_ue5m3_b32.toml": {
        "converter": ["ue5m3_fp4.reported_b32_d50"],
        "recipe": "ue5m3",
        "block_size": 32,
    },
    "nemotron_h_8b_ue5m3_torch_control.toml": {
        "converter": ["ue5m3_fp4.torch_control_b16_d50"],
        "recipe": "ue5m3_torch_control",
        "block_size": 16,
    },
    "nemotron_h_8b_ue5m3_te_settings.toml": {
        "converter": ["ue5m3_fp4.ue5m3_te_settings_d1"],
        "recipe": "ue5m3_te_settings",
        "block_size": 16,
    },
    "nemotron_h_8b_nvfp4_te.toml": {
        "converter": ["ue5m3_fp4.native_nvfp4_te"],
        "recipe": "native_nvfp4",
        "block_size": 16,
    },
    "nemotron_h_8b_nvfp4_no_rht_all_linears.toml": {
        "converter": ["ue5m3_fp4.native_nvfp4_no_rht_all"],
        "recipe": "native_nvfp4_no_rht",
        "block_size": 16,
    },
}
EXPECTED_HISTORICAL_SPECS = {
    "nemotron_h_8b_nvfp4_no_rht_all_linears.toml",
    "nemotron_h_8b_nvfp4_te.toml",
    "nemotron_h_8b_ue5m3_te_settings.toml",
    "nemotron_h_8b_ue5m3_torch_control.toml",
}

BANNED_PUBLIC_TEXT = (
    "us-east-1-large-model-" + "training",
    "/" + "volt/",
    "/" + "workspace/",
    "PET_" + "MASTER_ADDR",
    "WANDB_" + "API_KEY",
    "AWS_" + "SECRET_ACCESS_KEY",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_common(path: Path, config: dict) -> None:
    require(config["debug"]["seed"] == 42, f"{path.name}: seed")
    require(config["model"]["name"] == "nemotron_h_ue5m3", f"{path.name}: model")
    require(config["model"]["flavor"] == "8B_reported", f"{path.name}: flavor")
    require(
        config["job"]["custom_config_module"] == "ue5m3_fp4.integrations.torchtitan.config",
        f"{path.name}: custom config",
    )
    require(
        config["experimental"]["custom_import"] == "ue5m3_fp4.integrations.torchtitan.plugin",
        f"{path.name}: plugin",
    )
    training = config["training"]
    require(training["dtype"] == "bfloat16", f"{path.name}: dtype")
    require(training["local_batch_size"] == 1, f"{path.name}: local batch")
    require(training["global_batch_size"] == 768, f"{path.name}: global batch")
    require(training["seq_len"] == 8192, f"{path.name}: sequence length")
    require(training["steps"] == 30000, f"{path.name}: steps")
    require(
        training["dataset"] == "ue5m3_public_olmo_mix_1124",
        f"{path.name}: public dataset",
    )
    require(
        config["compile"]
        == {
            "enable": True,
            "components": ["loss"],
            "backend": "inductor",
        },
        f"{path.name}: compiled CE",
    )
    require(config["checkpoint"]["interval"] == 2500, f"{path.name}: checkpoint interval")
    require(config["checkpoint"]["keep_latest_k"] == 0, f"{path.name}: keep all")
    require(
        config["checkpoint"]["last_save_model_only"] is False,
        f"{path.name}: full final checkpoint",
    )
    public_data = config["public_data"]
    require(public_data["dclm_weight"] == 0.82, f"{path.name}: DCLM weight")
    require(public_data["no_dclm_weight"] == 0.18, f"{path.name}: non-DCLM weight")
    require(public_data["expected_shards_per_stream"] == 32, f"{path.name}: shards")


def validate_recipe(path: Path, config: dict, expected: dict) -> None:
    require(config["model"]["converters"] == expected["converter"], f"{path.name}: converter")
    require(
        "mxfp_custom" not in config and "te_fp4" not in config,
        f"{path.name}: stale private schema",
    )


def validate_manifests() -> None:
    for name in (
        "data_preparation.yaml",
        "reported_experiments.yaml",
        "validation.yaml",
        "olmes.yaml",
    ):
        path = MANIFEST_ROOT / name
        require(path.is_file(), f"missing {path}")
        with path.open(encoding="utf-8") as handle:
            require(isinstance(yaml.safe_load(handle), dict), f"{path}: YAML root")

    with (MANIFEST_ROOT / "reported_experiments.yaml").open(encoding="utf-8") as handle:
        experiments = yaml.safe_load(handle)
    common = experiments["common_training"]
    require(common["nominal_input_tokens"] == 30000 * 768 * 8192, "token arithmetic")
    require(common["independent_runs_per_configuration"] == 1, "single-run disclosure")
    data_contract = experiments["data"]
    require(data_contract["historical_dataloader_workers"] == 2, "historical workers")
    require(data_contract["public_dataloader_workers"] == 0, "public workers")
    te_settings = next(
        item
        for item in experiments["experiments"]
        if item["key"] == "ue5m3_transformer_engine_settings"
    )
    require(
        te_settings["randomized_hadamard_transform"]
        == {
            "forward_gemm": False,
            "data_gradient_gemm": False,
            "weight_gradient_upstream_gradient_dy_t": True,
            "weight_gradient_saved_activation_x_t": True,
            "weights": False,
            "block_size": 16,
        },
        "UE5M3 TE-settings RHT operand placement",
    )

    with (MANIFEST_ROOT / "validation.yaml").open(encoding="utf-8") as handle:
        validation = yaml.safe_load(handle)
    data = validation["validation_data"]
    require(
        data["records"] * data["evaluated_next_token_targets_per_record"] == 6291456,
        "validation tokens",
    )
    require(validation["checkpoint_grid"]["expected_results"] == 84, "validation task count")
    metric = validation["metric"]
    require(
        metric["percent_nll_reduction_reference"] == "bf16 at the same checkpoint step",
        "validation percent reference",
    )
    require(
        metric["percent_nll_reduction_formula"]
        == "100 * (bf16_nll - candidate_nll) / bf16_nll",
        "validation percent sign convention",
    )

    with (MANIFEST_ROOT / "olmes.yaml").open(encoding="utf-8") as handle:
        olmes = yaml.safe_load(handle)
    require(olmes["request_bundle"]["leaf_tasks"] == 146, "OLMES leaf tasks")
    require(olmes["request_bundle"]["likelihood_requests"] == 368932, "OLMES requests")
    require(olmes["execution"]["expected_forward_work_units"] == 46136, "OLMES forwards")
    historical_runtime = olmes["historical_controlled_runtime"]
    require(
        historical_runtime["container_publicly_distributed"] is False,
        "historical OLMES container availability",
    )
    public_olmes = historical_runtime["public_olmes_reconstruction"]
    require(public_olmes["repository"] == PUBLIC_OLMES_REPOSITORY, "public OLMES remote")
    require(public_olmes["commit"] == PUBLIC_OLMES_COMMIT, "public OLMES commit")
    require(public_olmes["tree"] == PUBLIC_OLMES_TREE, "public OLMES tree")
    require(
        public_olmes["selected_likelihood_tasks_use_chat_format"] is False,
        "selected OLMES chat-format behavior",
    )
    require(
        public_olmes["compatibility_overlay"] == "colon-free request and result filenames",
        "public OLMES compatibility overlay",
    )

    with (MANIFEST_ROOT / "data_preparation.yaml").open(encoding="utf-8") as handle:
        data_preparation = yaml.safe_load(handle)
    require(
        data_preparation["source"]["revision"] == "8162bd79c6dc4fea470506531a8d791badc06b4b",
        "data revision",
    )
    require(data_preparation["raw_sharding"]["shard_count"] == 32, "data shard count")
    require(
        data_preparation["runtime_document_packing"]["window_tokens"] == 8193, "packing window"
    )


def validate_public_boundary() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        # Scan bytes so generated PDF/PNG artifacts are covered without trying
        # to decode arbitrary binary data as UTF-8.
        payload = path.read_bytes()
        for needle in BANNED_PUBLIC_TEXT:
            require(
                needle.encode("utf-8") not in payload,
                f"{path}: contains disallowed internal text {needle!r}",
            )


def validate_runtime_assets() -> None:
    require(TRAINING_LOCK.is_file(), f"missing {TRAINING_LOCK}")
    require(sha256(TRAINING_LOCK) == TRAINING_LOCK_SHA256, "training lock changed")
    require(COMPILED_LOCK.is_file(), f"missing {COMPILED_LOCK}")
    require(sha256(COMPILED_LOCK) == COMPILED_LOCK_SHA256, "compiled lock changed")
    require(OLMES_LOCK.is_file(), f"missing {OLMES_LOCK}")
    require(sha256(OLMES_LOCK) == OLMES_LOCK_SHA256, "OLMES dependency lock changed")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    require(NGC_BASE_MANIFEST in dockerfile, "Docker base manifest is not pinned")
    require(
        "--no-deps --require-hashes" in dockerfile,
        "Docker training dependencies are not hash locked",
    )
    installer = (ROOT / "scripts" / "install_olmes_runtime.sh").read_text(encoding="utf-8")
    for expected in (
        OLMES_LOCK_SHA256,
        PUBLIC_OLMES_COMMIT,
        PUBLIC_OLMES_TREE,
        "090253dac6688f2532509daa7aa2eb5fae50e956",
        "db85f8065408b842100436a45f56c65d3a0dd6a6",
    ):
        require(expected in installer, f"OLMES installer is missing source pin {expected}")


def main() -> int:
    actual = {path.name for path in CONFIG_ROOT.glob("*.toml")}
    require(actual == set(EXPECTED_CONFIGS), f"config inventory differs: {sorted(actual)}")
    historical = {path.name for path in HISTORICAL_ROOT.glob("*.toml")}
    require(
        historical == EXPECTED_HISTORICAL_SPECS,
        f"historical specification inventory differs: {sorted(historical)}",
    )
    for name, expected in EXPECTED_CONFIGS.items():
        path = CONFIG_ROOT / name
        config = load_toml(path)
        validate_common(path, config)
        validate_recipe(path, config, expected)
    validate_manifests()
    validate_public_boundary()
    validate_runtime_assets()
    print(
        f"validated {len(EXPECTED_CONFIGS)} runnable configs, "
        f"{len(historical)} historical specifications, and 4 manifests under {ROOT}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"reproduction bundle validation failed: {exc}") from exc
