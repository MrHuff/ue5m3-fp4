# Numerical reproduction protocol

## Proposed UE5M3 recipe

The proposed path uses signed E2M1 FP4 payloads and unsigned E5M3 (`UE5M3`)
block scales. Activations use one-dimensional blocks of 16 or 32 values;
weights use two-dimensional 16-by-16 or 32-by-32 blocks aligned with both GEMM
views. A global tensor scale maps the current or cached global maximum absolute
value to the configured target before block quantization.

The default tensor-scale target is 448. For UE5M3, whose largest finite scale
code is 61,440, this leaves approximately `61,440 / 448 = 137.14` times stale
growth headroom. In the proposed 8B configuration, target 2,048 is used only for
the upstream-gradient operand `dY` in the weight-gradient GEMMs of the final
four MLP `mixer.down_proj` modules (zero-based layers 45, 47, 49, and 51). The
larger target maps small `dY` block-scale values farther from zero, reducing
block-scale underflow, but reduces stale-growth headroom to `61,440 / 2,048 =
30`. This is a tradeoff, not a universal larger-is-better setting.

If a UE5M3 block-scale code rounds to zero, the implementation replaces it with
one before reciprocal computation and decoding. This deterministic zero-scale
rule is separate from stochastic payload rounding.

## Delayed tensor scaling during training

Training D=50 is periodic sample-and-hold, not a maximum over a 50-step window.
Each converted linear maintains independent activation, weight, and gradient
references. On optimizer steps 1, 51, 101, and so on, each selected operand path
samples its current global maximum absolute value. That exact value is cached
and reused for the intervening 49 optimizer steps. The safety factor is 1.0.
Each maximum is measured in FP32 and MAX-reduced across the default distributed
process group before it is cached.

The training loop must publish one logical optimizer-step index before any
quantized work for that step. Gradient-accumulation microbatches do not advance
the D=50 counter. Delayed state is process-local and must not be silently
restored from, or inferred from, a model checkpoint.

The proposed recipe delays all three training operand classes:

- forward activations;
- weights;
- upstream gradients used by the backward GEMMs.

The per-path interval fields are zero in the reconstructed TOMLs, which means
they inherit the global interval of 50; zero does not disable those paths.

## Selective stochastic rounding

For a linear `Y = X W^T`, the backward GEMMs are `dX = dY W` and
`dW = dY^T X`. The proposed recipe applies the eight-bit-midpoint
`StochasticFast` E2M1 rounding rule only to the upstream-gradient operand `dY`
in both backward GEMMs:

- data gradient: `dY` stochastic, `W` deterministic ties-to-even;
- weight gradient: `dY^T` stochastic, saved `X` deterministic ties-to-even.

Forward activations, weights, saved activations, and all block-scale codes use
deterministic round-to-nearest, ties-to-even. Stochastic rounding is not used in
the inference forward path.

The public recipe represents the `T=2048` override as a
`ScaleTargetOverride` for the weight-gradient upstream-gradient role,
`mixer.down_proj`, and the inclusive zero-based layer range 44 through 51.
Only layers 45, 47, 49, and 51 are MLP blocks with the applicable module, so
those are the four effective overrides.

## Probe-matched GEMM output model

The proposed and TE-settings UE5M3 configurations use a software GEMM model
matched to native FP4-output probes:

1. operands are quantized into encoded BF16 values;
2. the reduction dimension is split into groups of 64 terms;
3. each group is evaluated with a dot product accumulating to FP32;
4. group results are combined sequentially with FP32 round-toward-zero adds;
5. the final encoded result is rounded ties-to-even to a `1/1024` lattice;
6. the FP32 tensor decode factor is applied.

The `1/1024` grid is the selected comparator setting. The available denominator
probes rejected coarser `1/256` and `1/512` grids, but `1/1024`, finer grids,
and no final grid all matched those particular probes. It is therefore not
presented as a uniquely identified hardware rule. The comparator is a software
numerical model, not native UE5M3 execution and not evidence of UE5M3 hardware
throughput.

The generic Torch control keeps the proposed recipe and B=16 quantization. It
uses the same encoded Triton operand quantization, performs a Torch FP32 matrix
multiplication in the encoded domain, and then applies the global decode factor.
It is intended to measure sensitivity to the GEMM output model.

## Layer coverage and randomized Hadamard transforms

The proposed B16, B32, and Torch-control configurations quantize all 112
eligible FFN, attention, and Mamba input/output linears. They exclude the output
head from FP4: its BF16 checkpoint parameter is retained and its matrix
multiplication is computed in FP32. They do not use a randomized Hadamard
transform (RHT).

The UE5M3-with-Transformer-Engine-settings comparator quantizes 96 eligible
linears and retains the BF16 exemption for 16 projections in the final eight
hybrid blocks. Its randomized Hadamard transform is confined to the two
columnwise operand representations used by the weight-gradient GEMM:
transformed `dY.T` and transformed saved `X.T`. The forward GEMM and
data-gradient GEMM use untransformed operands. Tensor maxima are sampled from
the current operand on every step (D=1).

The native Transformer Engine reference uses E2M1 payloads, signed E4M3 block
scales, two-dimensional weight scaling, stochastic `dY` rounding, RHT on the
columnwise representations used for weight-gradient computation, and the same
final-eight-block BF16 exemption. It must be run through the pinned native
NVFP4 implementation on supported Blackwell hardware. The integration also
sets Transformer Engine's two-times accumulation controls to false for both
data-gradient and weight-gradient GEMMs and verifies those effective values.

## Post-load quantized inference

All reported quantized validation and OLMES paths load learned BF16 master
weights and then explicitly apply FP4 or NVFP4 inference. A BF16 checkpoint
evaluated without conversion is the BF16 reference; it must not be labelled an
FP4 result.

For software UE5M3, the post-load lifecycle is:

1. load weights into a fresh model;
2. apply the exact recorded module conversion and verify coverage;
3. call `eval()`;
4. reset inference state and any legacy delayed caches;
5. sample and freeze each unchanged weight tensor's global amax;
6. select and initialize one activation-scale policy;
7. begin measurement.

Native Transformer Engine NVFP4 follows its own fresh-module lifecycle. After
checkpoint load, the evaluator converts eligible linears to the pinned native
implementation, verifies current-tensor D=1 scaling, and records native module
state and forward counters. It does not use the software controller or freeze a
software weight-amax reference.

Current-tensor D=1 recomputes each activation operand's amax for every forward.
Cold D=50 replay refreshes activations on complete forward work units 1, 51,
101, and so on, holding the sampled value between refreshes. Validation defines
one work unit as one batch-one, 8,192-token model forward. OLMES defines one work
unit as one top-level forward on one consecutive batch of up to eight requests;
its counter spans all 146 tasks without resetting between tasks.

This inference replay does not restore the training cache and does not equate a
forward batch with a training optimizer step. The result is order-dependent, so
batch size, request order, partial final batches, and reset boundaries are part
of the experiment identity.

## Native Transformer Engine qualification

The native reference requires Transformer Engine
`2.16.0.dev0+01aef4fc` at commit
`01aef4fc721bd12fd09cd56d53a314aee1b953d6`, with two-times accumulation
disabled for both wgrad and dgrad. The audited production v12 evaluation
runtime had no delayed-amax lifecycle API. Its native results use effective
current-tensor D=1 scaling even when a historical config recorded an
experimental requested interval of 50. Any future patched D=50 native run is a
new experiment and must not be substituted for the reported result.
