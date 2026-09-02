# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public configuration extension for the pinned TorchTitan revision.

TorchTitan merges this small dataclass into its own ``JobConfig`` when a TOML
sets ``job.custom_config_module`` to this module.  Numerical recipes are named
model converters; this extension only describes the public, storage-neutral
training-data reconstruction.
"""

import math
from dataclasses import dataclass, field


@dataclass
class PublicData:
    """Paths and deterministic mixing controls for prepared OLMo Mix data."""

    root: str = ""
    """Root containing ``dclm/`` and ``olmo-no-dclm/`` shard directories."""

    dclm_weight: float = 0.82
    """Document-level DCLM sampling weight in the public reconstruction."""

    no_dclm_weight: float = 0.18
    """Document-level non-DCLM sampling weight in the public reconstruction."""

    expected_shards_per_stream: int = 32
    """Number of prepared Hugging Face shards expected for each stream."""

    text_column: str = "text"
    """Dataset column passed to the tokenizer."""

    cycle: bool = True
    """Cycle an exhausted stream; required for a fixed-step pretraining run."""

    strict_inventory: bool = True
    """Reject missing or additional ``shard-*`` directories."""

    def __post_init__(self) -> None:
        for name in ("dclm_weight", "no_dclm_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"public_data.{name} must be finite")
            if value <= 0:
                raise ValueError(f"public_data.{name} must be positive")
        if not math.isclose(
            self.dclm_weight + self.no_dclm_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("public data stream weights must sum to one")
        if type(self.expected_shards_per_stream) is not int:
            raise TypeError("expected_shards_per_stream must be an integer")
        if self.expected_shards_per_stream <= 0:
            raise ValueError("expected_shards_per_stream must be positive")
        if not isinstance(self.text_column, str) or not self.text_column:
            raise ValueError("text_column must be a non-empty string")
        if type(self.cycle) is not bool or type(self.strict_inventory) is not bool:
            raise TypeError("cycle and strict_inventory must be bool")


@dataclass
class JobConfig:
    """Fields merged into upstream TorchTitan's ``JobConfig``."""

    public_data: PublicData = field(default_factory=PublicData)


__all__ = ["JobConfig", "PublicData"]
