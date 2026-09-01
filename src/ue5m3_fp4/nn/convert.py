# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral replacement of selected ``torch.nn.Linear`` modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ue5m3_fp4.nn.linear import UE5M3Linear
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState

LinearSelector = Callable[[str, torch.nn.Linear], bool]


def select_all_linears(_module_name: str, _module: torch.nn.Linear) -> bool:
    """Explicitly select every child ``nn.Linear``."""

    return True


def exclude_lm_head(module_name: str, _module: torch.nn.Linear) -> bool:
    """Exclude a conventionally named ``lm_head`` from conversion."""

    return module_name.rsplit(".", 1)[-1] != "lm_head"


@dataclass(frozen=True)
class ConversionRecord:
    """One deterministic module replacement."""

    module_name: str
    in_features: int
    out_features: int
    bias: bool


def convert_linear_modules(
    model: torch.nn.Module,
    *,
    recipe: UE5M3Recipe | None = None,
    scale_state: TrainingScaleState | None = None,
    selector: LinearSelector | None = None,
) -> tuple[ConversionRecord, ...]:
    """Replace selected child linears in-place and return exact coverage.

    ``selector`` is required and receives the fully qualified child name and
    source module. Requiring it prevents a generic converter from silently
    choosing scientific coverage, such as whether an LM head remains in high
    precision. The root object is deliberately not replaced, which keeps
    object identity and optimizer/model wrappers explicit for callers.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model, torch.nn.Linear):
        # The type is valid; its placement as the root is the invalid value.
        raise ValueError(  # noqa: TRY004
            "convert_linear_modules cannot replace the model root"
        )

    recipe = recipe or UE5M3Recipe.proposed()
    if not isinstance(recipe, UE5M3Recipe):
        raise TypeError("recipe must be a UE5M3Recipe")
    if scale_state is not None and not isinstance(scale_state, TrainingScaleState):
        raise TypeError("scale_state must be a TrainingScaleState")
    if scale_state is not None and scale_state.recipe != recipe:
        raise ValueError("scale_state.recipe must match recipe")
    scale_state = scale_state or TrainingScaleState(recipe)
    if selector is None:
        raise ValueError(
            "selector is required; pass select_all_linears explicitly to "
            "convert every child linear"
        )
    if not callable(selector):
        raise TypeError("selector must be callable")
    linear_modules = [
        (module_name, module)
        for module_name, module in model.named_modules(remove_duplicate=False)
        if module_name and isinstance(module, torch.nn.Linear)
    ]
    selected_subclasses = [
        (module_name, type(module).__qualname__)
        for module_name, module in linear_modules
        if type(module) is not torch.nn.Linear and selector(module_name, module)
    ]
    if selected_subclasses:
        rendered = ", ".join(
            f"{module_name} ({class_name})" for module_name, class_name in selected_subclasses
        )
        raise TypeError(
            "Generic conversion supports exact nn.Linear modules only; "
            f"architecture-specific subclasses require an adapter: {rendered}"
        )
    candidates = [
        (module_name, module)
        for module_name, module in linear_modules
        if type(module) is torch.nn.Linear
    ]
    aliases: dict[int, list[str]] = {}
    for module_name, module in candidates:
        aliases.setdefault(id(module), []).append(module_name)
    duplicate_aliases = [names for names in aliases.values() if len(names) > 1]
    if duplicate_aliases:
        rendered = "; ".join(", ".join(names) for names in duplicate_aliases)
        raise ValueError(
            "Aliased nn.Linear module objects require architecture-specific "
            f"conversion: {rendered}"
        )

    # Evaluate every selector decision before mutating the model. A selector
    # error therefore cannot leave a partially converted module tree.
    selected = [
        (module_name, module)
        for module_name, module in candidates
        if selector(module_name, module)
    ]
    records: list[ConversionRecord] = []
    for module_name, module in selected:
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = UE5M3Linear.from_float(
            module,
            recipe=recipe,
            scale_state=scale_state,
            module_name=module_name,
        )
        replacement.train(module.training)
        setattr(parent, child_name, replacement)
        records.append(
            ConversionRecord(
                module_name=module_name,
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        )
    return tuple(records)


__all__ = [
    "ConversionRecord",
    "LinearSelector",
    "convert_linear_modules",
    "exclude_lm_head",
    "select_all_linears",
]
