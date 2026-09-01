<!--
Copyright (c) 2026 Graphcore Ltd. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Post-load quantized evaluation

Training checkpoints contain learned high-precision master weights. They do
not contain the process-local tensor-maximum caches used by periodic scaling.
Loading a checkpoint therefore cannot resume an unspecified training cache.
Quantized evaluation must create a new, explicit forward-only scale lifecycle.

This distinction matters. Evaluating the learned weights in BF16 measures an
unquantized control; it does not measure FP4 inference. A UE5M3 FP4 evaluation
first converts the eligible linear layers, then quantizes their loaded weights
and forward activations to FP4-representable values during every measured
forward.

## Common setup

Perform post-load setup in this order:

1. Load the learned master weights and call `model.eval()`.
2. Reset every FP4 module's process-local inference state.
3. Measure each loaded weight tensor's global `amax` once and freeze it. The
   weight is unchanged during evaluation, so recomputing it would return the
   same value.
4. Configure one of the activation policies below.
5. Enter measurement only after all required scale setup is complete.

The `FP4InferenceScalingController` enforces this ordering and rejects a
partially configured model. It also records the checkpoint identity, resolved
format, frozen references, work-unit order, refresh trace, and scale-resolution
counters.

## Activation policies

### Current tensor (`current_tensor`, D=1)

Measure the current activation operand's `amax` and use it immediately. There
is no activation-cache reuse. This policy is independent of evaluation order,
apart from ordinary model-state effects.

### Cold periodic replay (`training_replay`, D=50)

Start every per-linear activation cache empty. Advance one logical work unit
immediately before each complete model forward. The first forward samples an
`amax`; forwards 2 through 50 reuse it; forward 51 samples a new value, followed
by another 49 reuses, and so on.

The name `training_replay` describes the 50-unit sample-and-hold rule. It does
not reconstruct the missing training cache, and an inference work unit is not
an optimizer step. A work unit must be defined by the evaluator—for example,
one fixed validation sequence or one fixed forward batch. The result is
deterministic only together with the cold initial state, batch size, and exact
input order. Changing that order can change the result.

Example controller setup for a fixed batch of one:

```python
from ue5m3_fp4.scaling.inference import FP4InferenceScalingController

model.eval()
controller = FP4InferenceScalingController(
    model,
    activation_mode="training_replay",
    checkpoint_identity={"id": "step-30000"},
    replay_work_unit={"kind": "fixed_forward_batch", "size": 1},
)
controller.reset_after_checkpoint_load()
controller.calibrate_and_freeze_weights()
controller.begin_measurement(
    evaluation_order={"order": "input-list order", "seed": None}
)
```

`evaluate_validation` exposes an immediate pre-forward hook for advancing the
same counter:

```python
from ue5m3_fp4.eval import evaluate_validation

result = evaluate_validation(
    model,
    ["validation-000.safetensors"],
    checkpoint_id="step-30000",
    device="cuda",
    batch_size=1,
    before_forward_callback=lambda input_ids, _: (
        controller.advance_training_replay_work_unit(
            input_ids,
            effective_token_count=input_ids.numel(),
        )
    ),
)
result["model"] = controller.provenance()
```

The callback is invoked exactly once immediately before each model forward. Do
not advance the replay counter per layer or per token.

### Disjoint calibration, then frozen (`calibrated_frozen`)

Run an explicitly identified calibration stream before measurement. During
calibration, collect the maximum observed activation `amax` separately for
each converted linear and then freeze those values. The measured validation
stream must be disjoint from calibration data. Frozen activation references
do not depend on validation order, although the chosen calibration set remains
part of the result definition.

## Forward-only behavior

The proposed training recipe uses stochastic rounding only for the upstream
gradient `dY`. Inference has no backward GEMMs, so this rounding is inactive.
Any randomized Hadamard transform used solely to form weight gradients is also
inactive. The inference lifecycle rejects backward execution and stochastic or
stateful forward-rounding modes while it is active.

## Local next-token loss

Each validation row contains `sequence_length + 1` integer tokens. The
evaluator forwards `tokens[:-1]` and compares its logits directly with
`tokens[1:]`; it never passes `labels`, avoiding a second causal shift.
Cross-entropy is computed in FP32, per-sequence sums accumulate in FP64, and
the result retains per-sequence loss sums, token counts, source hashes, and row
indices for paired analysis. File identifiers are generated from source order
and content hash; absolute local paths are not emitted. Inputs can be in-memory
`int32`/`int64` tensors or local safetensors files containing a rank-two
`tokens` tensor.

## Fake UE5M3 versus native NVFP4

The UE5M3 path is software FP4 simulation: eligible operands are genuinely
rounded to E2M1 values using UE5M3 block scales. The initial public linear
reference then decodes those operands and passes float32 inputs to PyTorch
matrix multiplication. Runtime provenance records PyTorch's matmul-precision
policy and CUDA TF32 setting when applicable. The path therefore preserves
operand-quantization error but does not yet reproduce the probe-matched native
reduction and output-rounding model. It is not native UE5M3 execution or a
hardware-throughput measurement.

Native Transformer Engine NVFP4 is a separate path using E2M1 payloads with
E4M3 block scales on supported hardware. Results must label these numeric paths
separately; running BF16 matrix multiplication after loading a quantized-trained
checkpoint is neither of them.
