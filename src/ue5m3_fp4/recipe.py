# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit configuration for the proposed UE5M3 FP4 training recipe."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from ue5m3_fp4.formats import (
    E2M1,
    UE5M3,
    FloatFormat,
    RoundingMode,
    normalize_rounding,
)


class OperandRole(StrEnum):
    """Operands with independent delayed-amax state."""

    ACTIVATION = "activation"
    WEIGHT = "weight"
    UPSTREAM_GRADIENT = "upstream_gradient"
    WGRAD_UPSTREAM_GRADIENT = "wgrad_upstream_gradient"


_ROLE_ALIASES: Final[dict[str, OperandRole]] = {
    "activation": OperandRole.ACTIVATION,
    "activations": OperandRole.ACTIVATION,
    "x": OperandRole.ACTIVATION,
    "input": OperandRole.ACTIVATION,
    "weight": OperandRole.WEIGHT,
    "weights": OperandRole.WEIGHT,
    "w": OperandRole.WEIGHT,
    "upstream_gradient": OperandRole.UPSTREAM_GRADIENT,
    "upstream_gradients": OperandRole.UPSTREAM_GRADIENT,
    "dy": OperandRole.UPSTREAM_GRADIENT,
    "d_y": OperandRole.UPSTREAM_GRADIENT,
    "wgrad_upstream_gradient": OperandRole.WGRAD_UPSTREAM_GRADIENT,
    "wgrad_dy": OperandRole.WGRAD_UPSTREAM_GRADIENT,
    "weight_gradient_upstream_gradient": OperandRole.WGRAD_UPSTREAM_GRADIENT,
}


def normalize_role(role: str | OperandRole) -> OperandRole:
    """Normalize a documented role or short operand alias."""

    if isinstance(role, OperandRole):
        return role
    if not isinstance(role, str):
        raise TypeError("role must be a string or OperandRole")
    key = role.strip().replace("-", "_").lower()
    try:
        return _ROLE_ALIASES[key]
    except KeyError as error:
        choices = ", ".join(item.value for item in OperandRole)
        raise ValueError(f"unknown operand role {role!r}; expected {choices}") from error


_NEMOTRON_DOWN_PROJ = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.mixer\.down_proj(?:$|\.)")


@dataclass(frozen=True, slots=True)
class ScaleTargetOverride:
    """A structured, auditable target override for a model region."""

    role: OperandRole
    module_kind: str
    layer_start: int
    layer_end: int
    scale_target: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", normalize_role(self.role))
        if self.module_kind != "mixer.down_proj":
            raise ValueError("the public recipe supports module_kind='mixer.down_proj'")
        if type(self.layer_start) is not int or type(self.layer_end) is not int:
            raise TypeError("layer bounds must be integers")
        if self.layer_start < 0 or self.layer_end < self.layer_start:
            raise ValueError("invalid layer bounds")
        if (
            type(self.scale_target) not in (int, float)
            or not math.isfinite(float(self.scale_target))
            or self.scale_target <= 0
        ):
            raise ValueError("override scale_target must be a finite positive number")

    def matches(self, role: str | OperandRole, module_name: str) -> bool:
        if normalize_role(role) is not self.role:
            return False
        if not isinstance(module_name, str):
            raise TypeError("module_name must be a string")
        match = _NEMOTRON_DOWN_PROJ.search(module_name)
        if match is None:
            return False
        layer = int(match.group("layer"))
        return self.layer_start <= layer <= self.layer_end

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "module_kind": self.module_kind,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "scale_target": float(self.scale_target),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScaleTargetOverride:
        if not isinstance(value, Mapping):
            raise TypeError("each scale-target override must be a mapping")
        expected = {
            "role",
            "module_kind",
            "layer_start",
            "layer_end",
            "scale_target",
        }
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError(
                f"invalid override keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            role=normalize_role(value["role"]),
            module_kind=value["module_kind"],
            layer_start=value["layer_start"],
            layer_end=value["layer_end"],
            scale_target=value["scale_target"],
        )


def _format_from_config(value: Any, *, expected: FloatFormat) -> FloatFormat:
    if isinstance(value, FloatFormat):
        result = value
    elif isinstance(value, str):
        formats = {E2M1.name.lower(): E2M1, UE5M3.name.lower(): UE5M3}
        try:
            result = formats[value.strip().lower()]
        except KeyError as error:
            raise ValueError(f"unknown floating-point format {value!r}") from error
    else:
        raise TypeError("format fields must be format names or FloatFormat instances")
    if result != expected:
        raise ValueError(f"this recipe requires {expected.name}, got {result.name}")
    return result


@dataclass(frozen=True, slots=True)
class UE5M3Recipe:
    """Complete numerical controls for one UE5M3 FP4 training recipe.

    Delayed scaling is periodic sample-and-hold: the current operand's global
    amax is sampled at a refresh and the reference remains unchanged for
    ``delayed_scale_interval`` logical training steps.  It is not a rolling or
    history-window maximum.
    """

    name: str
    block_size: int
    payload_format: FloatFormat
    scale_format: FloatFormat
    scale_target: float
    delayed_scale_interval: int
    activation_rounding: RoundingMode
    weight_rounding: RoundingMode
    upstream_gradient_rounding: RoundingMode
    scale_rounding: RoundingMode
    two_dimensional_weights: bool
    randomized_hadamard_transform: bool
    scale_target_overrides: tuple[ScaleTargetOverride, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("recipe name must be a non-empty string")
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        object.__setattr__(
            self,
            "payload_format",
            _format_from_config(self.payload_format, expected=E2M1),
        )
        object.__setattr__(
            self,
            "scale_format",
            _format_from_config(self.scale_format, expected=UE5M3),
        )
        if (
            type(self.scale_target) not in (int, float)
            or not math.isfinite(float(self.scale_target))
            or self.scale_target <= 0
        ):
            raise ValueError("scale_target must be a finite positive number")
        if type(self.delayed_scale_interval) is not int or self.delayed_scale_interval <= 0:
            raise ValueError("delayed_scale_interval must be a positive integer")
        for field_name in (
            "activation_rounding",
            "weight_rounding",
            "upstream_gradient_rounding",
            "scale_rounding",
        ):
            object.__setattr__(self, field_name, normalize_rounding(getattr(self, field_name)))
        if self.scale_rounding is not RoundingMode.TIES_TO_EVEN:
            raise ValueError("UE5M3 block scales use ties-to-even rounding")
        if type(self.two_dimensional_weights) is not bool:
            raise TypeError("two_dimensional_weights must be bool")
        if type(self.randomized_hadamard_transform) is not bool:
            raise TypeError("randomized_hadamard_transform must be bool")
        if self.randomized_hadamard_transform:
            raise ValueError("this portable recipe does not implement RHT")
        object.__setattr__(self, "scale_target_overrides", tuple(self.scale_target_overrides))
        if any(
            not isinstance(item, ScaleTargetOverride) for item in self.scale_target_overrides
        ):
            raise TypeError("scale_target_overrides must contain ScaleTargetOverride values")

    @classmethod
    def proposed(cls) -> UE5M3Recipe:
        """Return the proposed B=16, D=50 periodic-refresh recipe."""

        return cls(
            name="proposed_ue5m3_b16_d50",
            block_size=16,
            payload_format=E2M1,
            scale_format=UE5M3,
            scale_target=448.0,
            delayed_scale_interval=50,
            activation_rounding=RoundingMode.TIES_TO_EVEN,
            weight_rounding=RoundingMode.TIES_TO_EVEN,
            upstream_gradient_rounding=RoundingMode.STOCHASTIC_8BIT_MIDPOINT,
            scale_rounding=RoundingMode.TIES_TO_EVEN,
            two_dimensional_weights=True,
            randomized_hadamard_transform=False,
            scale_target_overrides=(
                ScaleTargetOverride(
                    role=OperandRole.WGRAD_UPSTREAM_GRADIENT,
                    module_kind="mixer.down_proj",
                    layer_start=44,
                    layer_end=51,
                    scale_target=2_048.0,
                ),
            ),
        )

    @staticmethod
    def normalize_role(role: str | OperandRole) -> OperandRole:
        return normalize_role(role)

    def rounding_for(self, role: str | OperandRole) -> RoundingMode:
        """Return payload rounding for an exact GEMM operand role."""

        normalized = normalize_role(role)
        if normalized is OperandRole.ACTIVATION:
            return self.activation_rounding
        if normalized is OperandRole.WEIGHT:
            return self.weight_rounding
        return self.upstream_gradient_rounding

    def scale_target_for(self, role: str | OperandRole, module_name: str) -> float:
        """Resolve the tensor-scale target for one role and module.

        The T=2,048 overlay applies only to dY in the weight-gradient GEMM for
        ``mixer.down_proj`` in Nemotron layers 44--51.  All other operands use
        the recipe-wide T=448 target.
        """

        normalized = normalize_role(role)
        if not isinstance(module_name, str):
            raise TypeError("module_name must be a string")
        for override in self.scale_target_overrides:
            if override.matches(normalized, module_name):
                return float(override.scale_target)
        return float(self.scale_target)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, YAML-safe representation."""

        return {
            "schema_version": 1,
            "name": self.name,
            "block_size": self.block_size,
            "payload_format": self.payload_format.name,
            "scale_format": self.scale_format.name,
            "scale_target": float(self.scale_target),
            "delayed_scale_interval": self.delayed_scale_interval,
            "activation_rounding": self.activation_rounding.value,
            "weight_rounding": self.weight_rounding.value,
            "upstream_gradient_rounding": self.upstream_gradient_rounding.value,
            "scale_rounding": self.scale_rounding.value,
            "two_dimensional_weights": self.two_dimensional_weights,
            "randomized_hadamard_transform": self.randomized_hadamard_transform,
            "scale_target_overrides": [item.to_dict() for item in self.scale_target_overrides],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> UE5M3Recipe:
        """Parse a strict configuration mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("recipe configuration must be a mapping")
        data = dict(value)
        schema_version = data.pop("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"unsupported recipe schema_version {schema_version!r}")
        expected = {
            "name",
            "block_size",
            "payload_format",
            "scale_format",
            "scale_target",
            "delayed_scale_interval",
            "activation_rounding",
            "weight_rounding",
            "upstream_gradient_rounding",
            "scale_rounding",
            "two_dimensional_weights",
            "randomized_hadamard_transform",
            "scale_target_overrides",
        }
        unknown = set(data) - expected
        missing = expected - set(data)
        if unknown or missing:
            raise ValueError(
                f"invalid recipe keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        overrides = tuple(
            ScaleTargetOverride.from_dict(item) for item in data["scale_target_overrides"]
        )
        return cls(
            name=data["name"],
            block_size=data["block_size"],
            payload_format=_format_from_config(data["payload_format"], expected=E2M1),
            scale_format=_format_from_config(data["scale_format"], expected=UE5M3),
            scale_target=data["scale_target"],
            delayed_scale_interval=data["delayed_scale_interval"],
            activation_rounding=normalize_rounding(data["activation_rounding"]),
            weight_rounding=normalize_rounding(data["weight_rounding"]),
            upstream_gradient_rounding=normalize_rounding(data["upstream_gradient_rounding"]),
            scale_rounding=normalize_rounding(data["scale_rounding"]),
            two_dimensional_weights=data["two_dimensional_weights"],
            randomized_hadamard_transform=data["randomized_hadamard_transform"],
            scale_target_overrides=overrides,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> UE5M3Recipe:
        """Load a recipe with ``yaml.safe_load`` and strict field validation."""

        try:
            import yaml
        except ImportError as error:  # pragma: no cover - dependency error is explicit
            raise RuntimeError("PyYAML is required to load recipe files") from error
        recipe_path = Path(path)
        with recipe_path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        return cls.from_dict(value)


__all__ = [
    "OperandRole",
    "ScaleTargetOverride",
    "UE5M3Recipe",
    "normalize_role",
]
