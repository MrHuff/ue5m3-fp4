# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist

from ue5m3_fp4.recipe import UE5M3Recipe
from ue5m3_fp4.scaling.training import TrainingScaleState, sample_global_amax


def test_reference_requires_an_explicit_logical_step() -> None:
    state = TrainingScaleState(UE5M3Recipe.proposed())
    with pytest.raises(RuntimeError, match="begin_step"):
        state.reference("activation", "layers.0.mixer.in_proj", torch.ones(4))


def test_steps_must_be_positive_and_strictly_increasing() -> None:
    state = TrainingScaleState(UE5M3Recipe.proposed())
    with pytest.raises(ValueError, match="positive"):
        state.begin_step(0)
    state.begin_step(1)
    with pytest.raises(ValueError, match="strictly"):
        state.begin_step(1)
    with pytest.raises(ValueError, match="positive"):
        state.begin_step(-1)


def test_d50_is_periodic_sample_and_hold_not_a_window_maximum() -> None:
    state = TrainingScaleState(UE5M3Recipe.proposed())
    observed = {}
    for step in range(1, 52):
        state.begin_step(step)
        if step == 1:
            tensor = torch.tensor([1.0, -3.0])
        elif step == 51:
            tensor = torch.tensor([1.0, -2.0])
        else:
            tensor = torch.tensor([100.0 * step])
        observed[step] = state.reference("activation", "layers.0.mixer.in_proj", tensor)

    assert observed[1].item() == 3.0
    assert observed[2].item() == 3.0
    assert observed[50].item() == 3.0
    # The step-51 sample replaces the held value; it is not a maximum over the
    # preceding 50 values, which were much larger.
    assert observed[51].item() == 2.0

    entry = state.report()["entries"][0]
    assert entry["last_refresh_step"] == 51
    assert entry["refreshes"] == 2
    assert entry["reuses"] == 49


def test_scale_caches_are_separate_by_role_and_module() -> None:
    state = TrainingScaleState(UE5M3Recipe.proposed())
    state.begin_step(1)
    assert state.reference("x", "layers.0.mixer.in_proj", torch.tensor([2.0])).item() == 2.0
    assert state.reference("w", "layers.0.mixer.in_proj", torch.tensor([3.0])).item() == 3.0
    assert state.reference("x", "layers.1.mixer.in_proj", torch.tensor([4.0])).item() == 4.0
    # An alias resolves to the same activation/module entry and therefore
    # reuses 2.0 rather than sampling 99.0.
    assert (
        state.reference("activation", "layers.0.mixer.in_proj", torch.tensor([99.0])).item()
        == 2.0
    )
    entries = state.report()["entries"]
    assert {(item["role"], item["module_name"]) for item in entries} == {
        ("activation", "layers.0.mixer.in_proj"),
        ("activation", "layers.1.mixer.in_proj"),
        ("weight", "layers.0.mixer.in_proj"),
    }


def test_first_use_at_a_later_step_starts_that_keys_refresh_cadence() -> None:
    state = TrainingScaleState(UE5M3Recipe.proposed())
    state.begin_step(7)
    assert state.reference("weight", "head", torch.tensor([7.0])).item() == 7.0
    state.begin_step(56)
    assert state.reference("weight", "head", torch.tensor([56.0])).item() == 7.0
    state.begin_step(57)
    assert state.reference("weight", "head", torch.tensor([57.0])).item() == 57.0


def test_runtime_caches_are_not_serialized() -> None:
    state = TrainingScaleState(UE5M3Recipe.proposed())
    state.begin_step(1)
    state.reference("upstream_gradient", "head", torch.tensor([5.0]))
    assert state.state_dict() == {}
    report = state.report()
    assert report["policy"] == "periodic_sample_and_hold"
    assert report["rolling_or_window_maximum"] is False
    assert report["refresh_interval_steps"] == 50
    assert report["checkpoint_cache_entries"] == 0


def test_runtime_cache_can_be_explicitly_cleared_before_inference() -> None:
    state = TrainingScaleState(UE5M3Recipe.proposed())
    state.begin_step(1)
    state.reference("activation", "head", torch.tensor([5.0]))

    assert state.clear_runtime() == 1
    assert state.step is None
    assert state.report()["entries"] == []
    assert state.clear_runtime() == 0


def test_reference_rejects_empty_nonfinite_and_unnamed_operands() -> None:
    state = TrainingScaleState(UE5M3Recipe.proposed())
    state.begin_step(1)
    with pytest.raises(ValueError, match="non-empty"):
        state.reference("activation", "", torch.ones(1))
    with pytest.raises(ValueError, match="empty"):
        state.reference("activation", "module", torch.empty(0))
    with pytest.raises(ValueError, match="non-finite"):
        state.reference("activation", "module", torch.tensor([float("nan")]))


def test_amax_is_local_without_an_initialized_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    sampled = sample_global_amax(
        torch.tensor([-7.0, 2.0], dtype=torch.bfloat16),
        label="test operand",
    )

    assert sampled.dtype is torch.float32
    assert sampled.shape == ()
    assert sampled.item() == 7.0


def test_training_refresh_uses_global_max_and_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collective_calls: list[tuple[torch.Tensor, object]] = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    def fake_all_reduce(value: torch.Tensor, *, op: object) -> None:
        collective_calls.append((value.clone(), op))
        value.fill_(11.0)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    state = TrainingScaleState(UE5M3Recipe.proposed())

    state.begin_step(1)
    first = state.reference("activation", "layers.0.mixer.in_proj", torch.tensor([3.0]))
    state.begin_step(2)
    held = state.reference("activation", "layers.0.mixer.in_proj", torch.tensor([99.0]))

    assert first.item() == 11.0
    assert held.item() == 11.0
    assert len(collective_calls) == 1
    assert collective_calls[0][0].item() == 3.0
    assert collective_calls[0][1] == dist.ReduceOp.MAX


def test_single_rank_process_group_does_not_issue_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    def unexpected_all_reduce(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a single-rank process group needs no collective")

    monkeypatch.setattr(dist, "all_reduce", unexpected_all_reduce)

    assert sample_global_amax(torch.tensor([-5.0]), label="test operand").item() == 5.0


def test_global_amax_rejects_invalid_world_size_and_nccl_cpu_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 0)
    with pytest.raises(RuntimeError, match="must be positive"):
        sample_global_amax(torch.ones(1), label="test operand")

    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    with pytest.raises(RuntimeError, match="NCCL.*not CUDA"):
        sample_global_amax(torch.ones(1), label="test operand")
