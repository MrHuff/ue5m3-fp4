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
finite magnitude is 6. Each block of 16 values in the proposed recipe shares
one eight-bit scale.

A block scale is a nonnegative magnitude, so a sign bit has no useful role.
UE5M3 assigns all eight bits to five exponent bits and three fraction bits. The
finite format used here has maximum 61,440, smallest normal value `2^-14`, and
smallest subnormal value `2^-17`. E4M3 and UE5M3 have the same three fraction
bits; UE5M3 gains range, not additional fractional precision.

For each operand, a tensor-wide reference first establishes the range in which
the per-block scale is represented. The default scale target is 448. The
Nemotron-H 8B recipe uses a target of 2,048 only for the `dY` operand in the
weight-gradient GEMM of `mixer.down_proj` in layers 44 through 51. That override is
architecture-specific and must not be applied silently to other models.

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

The no-RHT recipe computes one delayed tensor reference for `dY` and reuses it
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

## GEMM output model

Operand fake quantization alone does not reproduce the tested native FP4 GEMM
output. This initial public slice decodes the fake-quantized operands and
passes float32 inputs to PyTorch matrix multiplication. The runtime provenance
records PyTorch's configured matmul-precision policy, including the CUDA TF32
flag when applicable. It is the decoded-Torch control, not the paper's
probe-matched comparator.

The experiments also used a probe-matched software backend that models the
observed native reduction and output rounding. That backend is not yet part of
this extraction, so results from this package must not be substituted for the
paper's probe-matched results. Neither software path measures native UE5M3
throughput. Native Transformer Engine NVFP4 uses E4M3 block scales and remains
a separate format and execution path.

## Recipe summary

The proposed block-16 training recipe is therefore:

- E2M1 values with UE5M3 block scales;
- two-dimensional weight scaling;
- 50-step periodic sample-and-hold references for activations, weights, and
  `dY`;
- ties-to-even forward, saved-activation, and scale rounding;
- exact `StochasticFast` rounding for `dY` in both backward GEMMs;
- no randomized Hadamard transforms;
- FP4 in every selected internal linear, with the language-model head retained
  at its stated high precision.

The generic converter cannot infer a model architecture's eligible-linears
policy. It therefore requires an explicit selector and returns the complete
set of converted names. The provided `exclude_lm_head` helper preserves a
conventionally named language-model head; architecture integrations should use
an exact allowlist when reproducing the reported model coverage.
