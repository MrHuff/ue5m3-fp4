# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
import torch

from ue5m3_fp4.nn.convert import (
    convert_linear_modules,
    exclude_lm_head,
    select_all_linears,
)
from ue5m3_fp4.nn.linear import UE5M3Linear
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(8, 16, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(16, 8),
        )
        self.output = torch.nn.Linear(8, 4, bias=False)


def test_converter_reports_exact_selected_coverage_and_preserves_parameters() -> None:
    model = TinyModel()
    original_weight = model.block[0].weight.detach().clone()

    records = convert_linear_modules(
        model,
        selector=lambda name, _module: name != "output",
    )

    assert [record.module_name for record in records] == ["block.0", "block.2"]
    assert isinstance(model.block[0], UE5M3Linear)
    assert isinstance(model.block[2], UE5M3Linear)
    assert isinstance(model.output, torch.nn.Linear)
    torch.testing.assert_close(model.block[0].weight, original_weight)


def test_converter_is_idempotent_for_already_converted_modules() -> None:
    model = TinyModel()
    first = convert_linear_modules(model, selector=select_all_linears)
    second = convert_linear_modules(model, selector=select_all_linears)
    assert len(first) == 3
    assert second == ()


def test_converter_preserves_frozen_parameters_and_optimizer_references() -> None:
    model = TinyModel()
    model.block[0].weight.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original_parameters = {id(parameter) for parameter in model.parameters()}

    convert_linear_modules(model, selector=select_all_linears)

    assert not model.block[0].weight.requires_grad
    assert {id(parameter) for parameter in model.parameters()} == original_parameters
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } == original_parameters


def test_converter_preserves_weight_tying_between_distinct_linears() -> None:
    class TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(4, 4, bias=False)
            self.second = torch.nn.Linear(4, 4, bias=False)
            self.second.weight = self.first.weight

    model = TiedModel()
    convert_linear_modules(model, selector=select_all_linears)

    assert isinstance(model.first, UE5M3Linear)
    assert isinstance(model.second, UE5M3Linear)
    assert model.first.weight is model.second.weight


def test_converter_rejects_a_scale_state_from_another_recipe() -> None:
    recipe = UE5M3Recipe.proposed()
    mismatched = replace(recipe, delayed_scale_interval=1)

    with pytest.raises(ValueError, match="must match recipe"):
        convert_linear_modules(
            TinyModel(),
            recipe=recipe,
            scale_state=TrainingScaleState(mismatched),
            selector=select_all_linears,
        )


def test_converter_requires_explicit_coverage_and_can_exclude_lm_head() -> None:
    class LanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    model = LanguageModel()
    with pytest.raises(ValueError, match="selector is required"):
        convert_linear_modules(model)

    records = convert_linear_modules(model, selector=exclude_lm_head)

    assert [record.module_name for record in records] == ["block"]
    assert isinstance(model.block, UE5M3Linear)
    assert isinstance(model.lm_head, torch.nn.Linear)


def test_converter_rejects_aliased_linear_module_objects_before_mutation() -> None:
    model = torch.nn.Module()
    shared = torch.nn.Linear(4, 4)
    model.first = shared
    model.second = shared

    with pytest.raises(ValueError, match="Aliased nn.Linear"):
        convert_linear_modules(model, selector=select_all_linears)

    assert model.first is shared
    assert model.second is shared


def test_converter_rejects_selected_linear_subclasses_before_mutation() -> None:
    class SpecializedLinear(torch.nn.Linear):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return 2 * super().forward(inputs)

    model = torch.nn.Module()
    specialized = SpecializedLinear(4, 4)
    model.projection = specialized

    with pytest.raises(TypeError, match="architecture-specific subclasses"):
        convert_linear_modules(model, selector=select_all_linears)

    assert model.projection is specialized
