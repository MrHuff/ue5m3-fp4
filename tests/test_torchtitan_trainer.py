# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from torch import nn

from ue5m3_fp4.integrations.torchtitan.trainer import install_trainer_step_hook
from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState


class _ScaleCarrier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_state = TrainingScaleState(UE5M3Recipe.proposed())


def _make_fake_trainer_class(microbatches: tuple[float, ...]) -> type:
    class FakeTrainer:
        def __init__(self, *, step: int) -> None:
            self.step = step
            self.model = _ScaleCarrier()
            self.model_parts = [self.model]
            self.microbatches = microbatches
            self.original_train_step_calls = 0

        def train_step(self) -> list[float]:
            self.original_train_step_calls += 1
            return [
                float(
                    self.model.scale_state.reference(
                        "activation",
                        "layers.0.mixer.in_proj",
                        torch.tensor([value]),
                    ).item()
                )
                for value in self.microbatches
            ]

    return FakeTrainer


def test_hook_advances_once_per_optimizer_step_not_per_microbatch() -> None:
    trainer_cls = install_trainer_step_hook(_make_fake_trainer_class((3.0, 100.0, 200.0)))
    trainer = trainer_cls(step=1)

    # Three simulated accumulation microbatches share the one amax sampled at
    # the optimizer-step boundary. The wrapper itself calls begin_step once.
    assert trainer.train_step() == [3.0, 3.0, 3.0]
    first_report = trainer.model.scale_state.report()
    assert first_report["begin_step_calls"] == 1
    assert first_report["entries"][0]["refreshes"] == 1
    assert first_report["entries"][0]["reuses"] == 2

    trainer.step = 2
    assert trainer.train_step() == [3.0, 3.0, 3.0]
    assert trainer.model.scale_state.report()["begin_step_calls"] == 2

    # D=50 refreshes at logical step 51 relative to the step-1 sample, still
    # once for the whole accumulated optimizer step.
    trainer.step = 51
    trainer.microbatches = (2.0, 300.0, 400.0)
    assert trainer.train_step() == [2.0, 2.0, 2.0]
    final_report = trainer.model.scale_state.report()
    assert final_report["current_step"] == 51
    assert final_report["begin_step_calls"] == 3
    activation = next(
        entry for entry in final_report["entries"] if entry["role"] == "activation"
    )
    assert activation["last_refresh_step"] == 51
    assert activation["refreshes"] == 2
    assert trainer.original_train_step_calls == 3


def test_hook_accepts_a_cold_resumed_step_and_rejects_duplicate_or_older_steps() -> None:
    trainer_cls = install_trainer_step_hook(_make_fake_trainer_class((7.0, 99.0)))
    # Installing the hook twice is idempotent and must not double-advance.
    assert install_trainer_step_hook(trainer_cls) is trainer_cls
    trainer = trainer_cls(step=27501)

    assert trainer.train_step() == [7.0, 7.0]
    report = trainer.model.scale_state.report()
    assert report["current_step"] == 27501
    assert report["begin_step_calls"] == 1
    assert trainer.original_train_step_calls == 1

    with pytest.raises(RuntimeError, match="out of order"):
        trainer.train_step()
    assert trainer.original_train_step_calls == 1

    trainer.step = 27500
    with pytest.raises(RuntimeError, match="out of order"):
        trainer.train_step()
    assert trainer.original_train_step_calls == 1
