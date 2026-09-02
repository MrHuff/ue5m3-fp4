# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit periodic sample-and-hold tensor scaling for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from ue5m3_fp4.recipe import OperandRole, UE5M3Recipe


def sample_global_amax(tensor: Tensor, *, label: str) -> Tensor:
    """Sample a finite FP32 amax and synchronize it across the default group.

    The reported training recipes use a global per-tensor reference. When a
    distributed process group has more than one rank, a local maximum would
    make each rank quantize with a different tensor scale. The scalar MAX
    collective below matches the recovered training implementation's enabled
    global-amax synchronization path.
    """

    if not isinstance(tensor, Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    if not tensor.is_floating_point():
        raise TypeError("tensor must have a floating-point dtype")
    if tensor.numel() == 0:
        raise ValueError(f"cannot sample {label} from an empty tensor")
    if tensor.device.type == "meta":
        raise RuntimeError(f"cannot sample {label} from a meta tensor")

    value = tensor.detach().to(dtype=torch.float32).abs().amax()
    # A sharded DTensor reduction can produce a scalar DTensor with a partial
    # MAX placement. Its local scalar is the correct input to the explicit
    # default-process-group MAX below.
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        value = to_local()
    value = value.to(dtype=torch.float32).reshape(())
    if not bool(torch.isfinite(value)):
        raise ValueError(f"cannot sample {label} from non-finite values")

    if not dist.is_available() or not dist.is_initialized():
        return value
    try:
        world_size = dist.get_world_size()
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(f"cannot determine process-group size for {label}") from error
    if world_size < 1:
        raise RuntimeError(f"process-group size for {label} must be positive, got {world_size}")
    if world_size == 1:
        return value

    try:
        backend = str(dist.get_backend()).lower()
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(f"cannot determine collective backend for {label}") from error
    if "nccl" in backend and value.device.type != "cuda":
        raise RuntimeError(
            f"global {label} synchronization uses NCCL but its scalar is on "
            f"{value.device.type!r}, not CUDA"
        )
    synchronized = value.clone()
    try:
        dist.all_reduce(synchronized, op=dist.ReduceOp.MAX)
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"failed to synchronize global {label} with a MAX collective"
        ) from error
    if not bool(torch.isfinite(synchronized)):
        raise RuntimeError(f"global {label} MAX collective returned a non-finite value")
    return synchronized


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
        return sample_global_amax(tensor, label="training operand amax")

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


__all__ = ["TrainingScaleState", "sample_global_amax"]
