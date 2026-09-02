# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TorchTitan logical-step bridge for delayed UE5M3 tensor scaling."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Any

from torch import nn

from ue5m3_fp4.scaling.training import TrainingScaleState

_HOOK_MARKER = "_ue5m3_fp4_logical_step_hook_v1"


def training_scale_states(model_parts: Iterable[nn.Module]) -> tuple[TrainingScaleState, ...]:
    """Collect unique scale states from converted model parts."""

    states: dict[int, TrainingScaleState] = {}
    for model in model_parts:
        if not isinstance(model, nn.Module):
            raise TypeError("every model part must be a torch.nn.Module")
        for module in model.modules():
            state = getattr(module, "scale_state", None)
            if isinstance(state, TrainingScaleState):
                states.setdefault(id(state), state)
    return tuple(states.values())


def begin_training_step(model_parts: Iterable[nn.Module], step: int) -> int:
    """Begin one logical optimizer step for every unique delayed-scale state."""

    if type(step) is not int or step < 1:
        raise ValueError("step must be a positive integer")
    states = training_scale_states(model_parts)
    # The plugin is imported for BF16 and Transformer Engine comparators too.
    # Those models deliberately contain no UE5M3 state, so the shared trainer
    # hook must be an explicit no-op for them.
    if not states:
        return 0
    for state in states:
        if state.step is not None and state.step >= step:
            raise RuntimeError(
                "TorchTitan UE5M3 step hook was called out of order: "
                f"state={state.step}, trainer={step}"
            )
        state.begin_step(step)
    return len(states)


def install_trainer_step_hook(trainer_cls: type[Any] | None = None) -> type[Any]:
    """Install the D=50 logical-step hook on pinned upstream TorchTitan.

    TorchTitan increments ``Trainer.step`` immediately before ``train_step``.
    Wrapping that method starts the sample-and-hold lifecycle once per logical
    optimizer step, rather than once per gradient-accumulation microbatch.
    """

    if trainer_cls is None:
        try:
            from torchtitan.train import Trainer
        except ImportError as error:
            raise ImportError(
                "TorchTitan is required; install the pinned upstream revision"
            ) from error
        trainer_cls = Trainer

    original = getattr(trainer_cls, "train_step", None)
    if not callable(original):
        raise TypeError("trainer class must expose a callable train_step")
    if getattr(original, _HOOK_MARKER, False):
        return trainer_cls

    @wraps(original)
    def train_step_with_ue5m3_state(self: Any, *args: Any, **kwargs: Any) -> Any:
        model_parts = getattr(self, "model_parts", None)
        step = getattr(self, "step", None)
        if model_parts is None:
            raise RuntimeError("TorchTitan Trainer has no model_parts at train_step")
        begin_training_step(model_parts, step)
        return original(self, *args, **kwargs)

    setattr(train_step_with_ue5m3_state, _HOOK_MARKER, True)
    setattr(trainer_cls, "train_step", train_step_with_ue5m3_state)
    return trainer_cls


__all__ = [
    "begin_training_step",
    "install_trainer_step_hook",
    "training_scale_states",
]
