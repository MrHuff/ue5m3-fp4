<!--
Copyright (c) 2026 Graphcore Ltd. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# UE5M3 FP4 numerical contract

This package simulates FP4 training with an E2M1 payload and an unsigned E5M3
block scale. The behavior below is part of the recipe, not an interchangeable
implementation detail.

## Formats and block scaling

E2M1 uses one sign bit, two exponent bits, and one fraction bit. Its largest
finite magnitude is 6. In the proposed B=16 recipe, each one-dimensional
activation block contains 16 values and each two-dimensional weight tile is
16-by-16; the B=32 ablation uses the corresponding 32 and 32-by-32 shapes.

A block scale is a nonnegative magnitude, so a sign bit has no useful role.
UE5M3 assigns all eight bits to five exponent bits and three fraction bits. The
finite format used here has maximum 61,440, smallest normal value `2^-14`, and
smallest subnormal value `2^-17`. E4M3 and UE5M3 have the same three fraction
bits; UE5M3 gains range, not additional fractional precision.

For each operand, a tensor-wide reference first establishes the range in which
the per-block scale is represented. The default scale target is 448. The
Nemotron-H 8B recipe uses a target of 2,048 only for the `dY` operand in the
weight-gradient GEMM of `mixer.down_proj` in zero-based layers 45, 47, 49, and
51. The selector spans layers 44 through 51, but only those four layers contain
the applicable MLP module. This override is architecture-specific and must not
be applied silently to other models.

Weights use two-dimensional scaling so that the forward and transposed views
used by the backward GEMMs agree on their block organization. A block scale
that rounds to zero is replaced by the numeric scale value 1.0, matching the
proposed recipe's divide-by-zero guard, before dequantization.

## Periodic sample-and-hold tensor references

Let `a_t = max(abs(x_t))` for one operand at optimizer step `t`. Current scaling
uses `a_t` immediately. The proposed D=50 policy instead samples the maximum on
a refresh step and holds that exact sample for the next 49 steps:

```text
reference(t) = a_t                 on a cold start or refresh step
reference(t) = reference(t - 1)   otherwise
```

With one-based steps, refreshes occur at 1, 51, 101, and so on. Activations,
weights, and the upstream gradient have separate per-linear caches. The safety
factor is 1.

This is periodic sample-and-hold, not a maximum over the preceding 50 steps.
It also differs from Transformer Engine's documented FP8 delayed-scaling
history, which records an `amax` every iteration and derives a scale from an
amax-history window. The cache is process-local numerical state and is not
serialized with learned master weights.

## Selective rounding

Forward activations, weights, saved activations used by the weight-gradient
GEMM, and block scales use round-to-nearest, ties-to-even.

Only the upstream gradient `dY` uses stochastic rounding. It is quantized this
way as the first operand of both backward GEMMs, in the orientation consumed by
each GEMM:

- data gradient: row-wise `dY` in `dX = dY @ W`;
- weight gradient: row-wise `dY.T` in `dW = dY.T @ X`.

The proposed no-RHT UE5M3 recipe computes one delayed tensor reference for `dY` and reuses it
for both GEMMs. Each quantization call draws its own random rounding values. The
saved activation `X` in the weight-gradient GEMM remains deterministic and is
column-scaled along that GEMM's reduction dimension.

The recorded implementation calls its stochastic mode `StochasticFast`. For a
normalized value whose exact significand lies a fraction `delta` between two
representable values, it samples an eight-bit integer `r` uniformly from
`{0, ..., 255}` and chooses the upper value when

```text
delta + (2*r + 1)/512 >= 1.
```

This is an eight-bit midpoint discretization of ideal stochastic rounding. It
must not be silently replaced by a different random-bit width or by a
full-precision Bernoulli draw when reproducing the reported recipe.

## GEMM output models

Operand fake quantization alone does not reproduce the tested native FP4 GEMM
output. The release therefore provides three explicit UE5M3 backends:

- `probe_matched_triton_issue_rz`, the backend used for the proposed and
  Transformer-Engine-settings UE5M3 experiments. It quantizes encoded operands
  with the released Triton kernels, reduces in groups of 64, combines group
  results with FP32 round-toward-zero additions, and rounds the encoded output
  to the selected `1/1024` lattice before decoding;
- `triton_quantized_torch_fp32`, the reported generic-Torch control. It uses
  the same encoded Triton operand quantization followed by a Torch FP32 matrix
  multiplication; and
- `portable_decoded_torch_reference`, a CPU-capable decoded-operand reference
  for tests and small numerical checks. It is not a reported 8B training path.

The `1/1024` output lattice is a probe-matched comparator setting, not a claim
that the hardware implements that rule internally. None of the software paths
measures native UE5M3 throughput. Native Transformer Engine NVFP4 uses E4M3
block scales and remains a separate, fail-closed execution path.

## Randomized Hadamard transform placement

The proposed recipe does not use a randomized Hadamard transform (RHT). In the
UE5M3 comparator with Transformer Engine settings, RHT is confined to the two
columnwise representations used by the weight-gradient GEMM: `dY.T` and the
saved activation `X.T`. The forward GEMM and data-gradient GEMM remain
untransformed. The transform uses the recorded fixed signs and normalized
block-16 Hadamard matrix; it is not applied to weights.

## Recipe summary

The proposed block-16 training recipe is therefore:

- E2M1 values with UE5M3 block scales;
- two-dimensional weight scaling;
- 50-step periodic sample-and-hold references for activations, weights, and
  `dY`;
- ties-to-even forward, saved-activation, and scale rounding;
- exact `StochasticFast` rounding for `dY` in both backward GEMMs;
- no randomized Hadamard transform;
- FP4 in every selected internal linear, with the language model head excluded
  from FP4; its BF16 checkpoint parameter is retained and its matrix
  multiplication is computed in FP32.

The generic converter cannot infer a model architecture's eligible-linears
policy. It therefore requires an explicit selector and returns the complete
set of converted names. The provided `exclude_lm_head` helper preserves a
conventionally named language model head; architecture integrations should use
an exact allowlist when reproducing the reported model coverage.
