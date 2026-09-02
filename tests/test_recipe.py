# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from ue5m3_fp4.formats import E2M1, UE5M3, RoundingMode
from ue5m3_fp4.recipe import OperandRole, UE5M3Recipe
from ue5m3_fp4.recipes import (
    RECIPE_RESOURCES,
    available_recipes,
    load_recipe_config,
    read_recipe_text,
    recipe_path,
    recipe_resource,
)

EXPECTED_RECIPE_RESOURCES = (
    "proposed_b16_d50.yaml",
    "inference/calibrated_frozen.yaml",
    "inference/current_tensor_d1.yaml",
    "inference/training_replay_d50.yaml",
)


def test_proposed_recipe_has_explicit_numerical_controls() -> None:
    recipe = UE5M3Recipe.proposed()
    assert recipe.block_size == 16
    assert recipe.delayed_scale_interval == 50
    assert recipe.payload_format is E2M1
    assert recipe.scale_format is UE5M3
    assert recipe.scale_target == 448.0
    assert recipe.two_dimensional_weights
    assert not recipe.randomized_hadamard_transform
    assert recipe.rounding_for("activation") is RoundingMode.TIES_TO_EVEN
    assert recipe.rounding_for("weight") is RoundingMode.TIES_TO_EVEN
    assert recipe.rounding_for("upstream_gradient") is RoundingMode.STOCHASTIC_8BIT_MIDPOINT
    assert (
        recipe.rounding_for("wgrad_upstream_gradient") is RoundingMode.STOCHASTIC_8BIT_MIDPOINT
    )


@pytest.mark.parametrize("layer", [44, 45, 50, 51])
def test_t2048_override_is_only_wgrad_dy_in_selected_down_projections(layer: int) -> None:
    recipe = UE5M3Recipe.proposed()
    module = f"model.layers.{layer}.mixer.down_proj"
    assert recipe.scale_target_for("wgrad_dy", module) == 2_048.0
    assert recipe.scale_target_for("upstream_gradient", module) == 448.0
    assert recipe.scale_target_for("activation", module) == 448.0
    assert recipe.scale_target_for("weight", module) == 448.0


def test_t2048_override_does_not_leak_to_other_layers_or_linears() -> None:
    recipe = UE5M3Recipe.proposed()
    for module in (
        "model.layers.43.mixer.down_proj",
        "model.layers.52.mixer.down_proj",
        "model.layers.44.mixer.in_proj",
        "model.layers.44.mlp.down_proj",
    ):
        assert recipe.scale_target_for(OperandRole.WGRAD_UPSTREAM_GRADIENT, module) == 448.0


def test_role_aliases_normalize_to_stable_names() -> None:
    recipe = UE5M3Recipe.proposed()
    assert recipe.normalize_role("x") is OperandRole.ACTIVATION
    assert recipe.normalize_role("w") is OperandRole.WEIGHT
    assert recipe.normalize_role("dY") is OperandRole.UPSTREAM_GRADIENT
    assert recipe.normalize_role("wgrad_dY") is OperandRole.WGRAD_UPSTREAM_GRADIENT
    with pytest.raises(ValueError, match="unknown operand role"):
        recipe.normalize_role("generic_gradient")


def test_recipe_mapping_round_trip_is_lossless() -> None:
    recipe = UE5M3Recipe.proposed()
    assert UE5M3Recipe.from_dict(recipe.to_dict()) == recipe
    bad = recipe.to_dict()
    bad["silent_typo"] = True
    with pytest.raises(ValueError, match="unknown"):
        UE5M3Recipe.from_dict(bad)


def test_packaged_recipe_inventory_is_complete() -> None:
    assert RECIPE_RESOURCES == EXPECTED_RECIPE_RESOURCES
    assert available_recipes() == EXPECTED_RECIPE_RESOURCES
    for name in available_recipes():
        assert recipe_resource(name).is_file()
        assert read_recipe_text(name).startswith("# Copyright")


def test_packaged_training_recipe_matches_code_default() -> None:
    with recipe_path("proposed_b16_d50.yaml") as path:
        assert UE5M3Recipe.from_yaml(path) == UE5M3Recipe.proposed()


def test_inference_recipe_files_state_distinct_scale_lifecycles() -> None:
    current = load_recipe_config("inference/current_tensor_d1.yaml")
    replay = load_recipe_config("inference/training_replay_d50.yaml")
    calibrated = load_recipe_config("inference/calibrated_frozen.yaml")

    assert current["activation_reference"]["policy"] == "current_tensor"
    assert current["activation_reference"]["refresh_interval_forwards"] == 1
    assert replay["activation_reference"]["policy"] == "periodic_sample_and_hold"
    assert replay["activation_reference"]["refresh_interval_logical_steps"] == 50
    assert replay["activation_reference"]["initial_state"] == "cold"
    assert calibrated["activation_reference"]["policy"] == (
        "disjoint_calibration_maximum_then_frozen"
    )
    assert calibrated["activation_reference"]["measurement_data_must_be_disjoint"]
    for config in (current, replay, calibrated):
        assert config["numeric_path"] == "fp4_fake_quantized"
        assert config["gemm_output_model"] == "decoded_operand_torch_matmul"
        assert config["torch_matmul_policy"] == "runtime_attested"
        assert config["native_hardware"] is False
        assert config["training_step_or_cache_inherited"] is False


def test_public_configs_do_not_contain_internal_locations_or_credentials() -> None:
    text = "\n".join(read_recipe_text(name) for name in available_recipes()).lower()
    for forbidden in (
        "s3" + "://",
        "/" + "workspace/",
        "/" + "volt/",
        "token" + ":",
        "password" + ":",
    ):
        assert forbidden not in text


def test_packaged_recipe_lookup_rejects_unknown_or_non_string_names() -> None:
    with pytest.raises(ValueError, match="unknown packaged recipe"):
        recipe_resource("../pyproject.toml")
    with pytest.raises(TypeError, match="must be a string"):
        recipe_resource(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [True, float("inf"), float("nan")])
def test_recipe_rejects_nonfinite_or_boolean_scale_targets(target: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        replace(UE5M3Recipe.proposed(), scale_target=target)


def test_recipe_rejects_unimplemented_rht() -> None:
    with pytest.raises(ValueError, match="does not implement RHT"):
        replace(UE5M3Recipe.proposed(), randomized_hadamard_transform=True)
