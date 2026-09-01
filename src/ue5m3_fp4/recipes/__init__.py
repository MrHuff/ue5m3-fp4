# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access the versioned YAML recipes shipped with :mod:`ue5m3_fp4`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Final

import yaml

RECIPE_RESOURCES: Final[tuple[str, ...]] = (
    "proposed_b16_d50.yaml",
    "inference/calibrated_frozen.yaml",
    "inference/current_tensor_d1.yaml",
    "inference/training_replay_d50.yaml",
)


def available_recipes() -> tuple[str, ...]:
    """Return the stable relative names of all packaged YAML recipes."""

    return RECIPE_RESOURCES


def recipe_resource(name: str) -> Traversable:
    """Return an importlib resource for one allowlisted recipe.

    The returned object works for an unpacked installation and for a package
    imported directly from a zip file. Use :func:`recipe_path` when a consumer
    specifically requires a temporary filesystem path.
    """

    if not isinstance(name, str):
        raise TypeError("recipe name must be a string")
    if name not in RECIPE_RESOURCES:
        choices = ", ".join(RECIPE_RESOURCES)
        raise ValueError(f"unknown packaged recipe {name!r}; expected one of {choices}")
    resource = files(__name__).joinpath(*name.split("/"))
    if not resource.is_file():  # pragma: no cover - detects a broken installation
        raise FileNotFoundError(f"packaged recipe is missing: {name}")
    return resource


def read_recipe_text(name: str) -> str:
    """Read one packaged recipe as UTF-8 text."""

    return recipe_resource(name).read_text(encoding="utf-8")


def load_recipe_config(name: str) -> dict[str, Any]:
    """Parse one packaged YAML recipe and require a mapping at its root."""

    value = yaml.safe_load(read_recipe_text(name))
    if not isinstance(value, Mapping):
        raise TypeError(f"packaged recipe {name!r} must contain a YAML mapping")
    return dict(value)


@contextmanager
def recipe_path(name: str) -> Iterator[Path]:
    """Yield a filesystem path for one packaged recipe.

    The path is valid only inside the context, which also supports resources
    loaded from a zip importer.
    """

    with as_file(recipe_resource(name)) as path:
        yield path


__all__ = [
    "RECIPE_RESOURCES",
    "available_recipes",
    "load_recipe_config",
    "read_recipe_text",
    "recipe_path",
    "recipe_resource",
]
