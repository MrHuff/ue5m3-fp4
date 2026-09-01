# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit periodic sample-and-hold tensor scaling for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ue5m3_fp4.recipe import OperandRole, UE5M3Recipe


@dataclass(slots=True)
class _ScaleEntry:
    reference: Tensor
    last_refresh_step: int
    refreshes: int = 1
    reuses: int = 0


class TrainingScaleState:
    """Process-local tensor-scale references for one training trajectory.

    Call :meth:`begin_step` exactly once before using any references in a
    logical training step.  For D=50, a key first observed at step 1 samples
    the current operand's global amax at steps 1, 51, 101, and so on.  Between
    refreshes the sampled value is held unchanged.  No history window or
    rolling maximum is maintained.

    State is keyed independently by exact operand role and module name.  The
    caches are intentionally absent from :meth:`state_dict`: they are runtime
    numerical state, not learned checkpoint parameters.
    """

    def __init__(self, recipe: UE5M3Recipe) -> None:
        if not isinstance(recipe, UE5M3Recipe):
            raise TypeError("recipe must be a UE5M3Recipe")
        self.recipe = recipe
        self._step: int | None = None
        self._entries: dict[tuple[OperandRole, str], _ScaleEntry] = {}
        self._begin_step_calls = 0

    @property
    def step(self) -> int | None:
        """Current logical step, or ``None`` before the first begin_step."""

        return self._step

    def begin_step(self, step: int) -> None:
        """Advance to a strictly increasing, positive logical training step."""

        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        if self._step is not None and step <= self._step:
            raise ValueError(
                f"training steps must increase strictly; current={self._step}, got={step}"
            )
        self._step = step
        self._begin_step_calls += 1

    @staticmethod
    def _sample_amax(tensor: Tensor) -> Tensor:
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if not tensor.is_floating_point():
            raise TypeError("tensor must have a floating-point dtype")
        if tensor.numel() == 0:
            raise ValueError("cannot sample amax from an empty tensor")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("cannot sample amax from non-finite values")
        return tensor.detach().abs().amax().to(dtype=torch.float32).reshape(())

    def reference(self, role: str, module_name: str, tensor: Tensor) -> Tensor:
        """Return the held tensor amax, refreshing it when D steps elapsed."""

        if self._step is None:
            raise RuntimeError("begin_step(step) must be called before reference()")
        normalized_role = self.recipe.normalize_role(role)
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("module_name must be a non-empty string")
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a torch.Tensor")

        key = (normalized_role, module_name)
        entry = self._entries.get(key)
        refresh = (
            entry is None
            or self._step - entry.last_refresh_step >= self.recipe.delayed_scale_interval
        )
        if refresh:
            sampled = self._sample_amax(tensor)
            if entry is None:
                entry = _ScaleEntry(reference=sampled, last_refresh_step=self._step)
                self._entries[key] = entry
            else:
                entry.reference = sampled
                entry.last_refresh_step = self._step
                entry.refreshes += 1
        else:
            if entry.reference.device != tensor.device:
                raise RuntimeError(
                    "tensor device changed while a delayed reference was held; "
                    "create a new TrainingScaleState for the new runtime"
                )
            entry.reuses += 1

        # A clone prevents downstream in-place operations from modifying the
        # process-local held reference.
        return entry.reference.clone()

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state; delayed-amax caches are intentionally omitted."""

        return {}

    def clear_runtime(self) -> int:
        """Clear process-local delayed-amax state before post-load inference.

        Returns the number of cached operand references that were discarded.
        Learned parameters are owned by the model and are not touched.
        """

        cleared = len(self._entries)
        self._entries.clear()
        self._step = None
        self._begin_step_calls = 0
        return cleared

    def report(self) -> dict[str, Any]:
        """Return a JSON-safe description of the live sample-and-hold state."""

        entries = []
        for (role, module_name), entry in sorted(
            self._entries.items(), key=lambda item: (item[0][0].value, item[0][1])
        ):
            entries.append(
                {
                    "role": role.value,
                    "module_name": module_name,
                    "reference_amax": float(entry.reference.detach().cpu().item()),
                    "last_refresh_step": entry.last_refresh_step,
                    "refreshes": entry.refreshes,
                    "reuses": entry.reuses,
                }
            )
        return {
            "schema": "ue5m3_fp4_training_scale_state_v1",
            "policy": "periodic_sample_and_hold",
            "rolling_or_window_maximum": False,
            "refresh_interval_steps": self.recipe.delayed_scale_interval,
            "current_step": self._step,
            "begin_step_calls": self._begin_step_calls,
            "checkpoint_cache_entries": 0,
            "entries": entries,
        }


__all__ = ["TrainingScaleState"]
