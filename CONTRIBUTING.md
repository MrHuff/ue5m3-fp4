# Contributing

Thank you for improving UE5M3 FP4. Numerical behavior and experiment identity
are part of this project's public interface: an apparently mechanical change
can alter the represented format, scale refresh, stochastic distribution,
GEMM output, converted-module set, or result interpretation.

## Before opening a change

1. Describe the behavior and reproduction tier being changed.
2. Keep credentials, private storage locations, model checkpoints, private
   data, frozen non-public request bundles, and private orchestration out of
   commits, examples, logs, and issue text.
3. Preserve the boundary between portable numerical code, CUDA kernels,
   model/runtime integrations, and audited result artifacts.
4. Do not claim native execution, throughput equivalence, a byte-identical
   historical rerun, repeated-seed evidence, or statistical significance
   without the corresponding implementation and evidence.
5. Do not replace audited paper outputs with public-method rerun outputs. A new
   dataset/checkpoint/request identity is a new experiment.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

ruff check src tests examples reproduce
ruff format --check src tests examples reproduce
PYTHONPATH=src:third_party/torchtitan python -m pytest
PYTHONPATH=src python examples/tiny_mlp.py
python -m build
```

Initialize the pinned submodules and use the container/locks under
`reproduce/` for model-scale or CUDA/Triton work. Before requesting review,
run the narrowest relevant tests plus the complete suite that is available in
your environment. Record skipped GPU tests rather than presenting them as
passes.

## Numerical and kernel changes

A numerical change must include:

- a regression test that identifies the affected values or operation;
- the exact operands and forward/backward GEMMs affected;
- format, block shape/orientation, scale target, refresh rule, and rounding
  mode;
- the accumulation and output-rounding model where applicable;
- whether learned checkpoint state or process-local numerical state changes;
- updated machine-readable provenance when result identity changes; and
- a statement about whether prior measurements remain comparable.

Do not silently replace periodic sample-and-hold with a rolling maximum,
eight-bit-midpoint `StochasticFast` with an ideal Bernoulli draw, or the K=64
issue-RZ probe-matched path with an ordinary matrix multiplication.

CUDA/Triton changes should record GPU model, driver, CUDA, PyTorch, Triton,
kernel configuration, and exact test command. Passing the portable CPU
reference is necessary but does not qualify a GPU kernel.

## Model and training integrations

Architecture integrations must provide a reviewed selector or exact allowlist
of converted linears. They should fail closed on unsupported subclasses,
aliases, attention backends, fused paths that bypass converted modules, and
checkpoint layouts. Test the complete converted-module set; a count alone is
not sufficient.

Training-loop integrations must advance delayed scaling once per logical
optimizer step. Gradient-accumulation microbatches must not advance D=50.
Changes to data order, tokenization, packing, batch size, checkpoint resume, or
inference work-unit order must produce a new recorded experiment identity.

Native Transformer Engine changes require the pinned implementation and
supported hardware. A config that records a requested option is not evidence
that the runtime applied it; verify the effective numerical path.

## Result artifacts and documentation

Keep these categories explicit:

- learned-weight BF16 evaluation;
- software FP4 fake quantization;
- the probe-matched UE5M3 GEMM model;
- native Transformer Engine NVFP4; and
- sanitized outputs copied from completed paper runs.

Published measurements should carry immutable checkpoint/data/protocol
identities and machine-readable provenance. The paper has one independent
training trajectory per configuration. Sequence- or task-bootstrap intervals
must not be described as training run-to-run variability.

Scripts that regenerate sanitized artifacts must fail if expected source
hashes, row counts, columns, or privacy checks differ. Generated figures and
tables should be deterministic and listed in an artifact manifest.

When documentation changes a reproduction claim, update the README,
verification record, release checklist, and citation/version metadata together.

## Packaging and release review

Before a release request:

```bash
python reproduce/scripts/validate_bundle.py
python -m build
```

Inspect both wheel and sdist, install the wheel into a clean environment, and
verify recipe resources. A complete source checkout—not a wheel or sdist—is
required for pinned Git submodules and the public TorchTitan training workflow.
Follow [`docs/release_checklist.md`](docs/release_checklist.md).

Report security issues through [`SECURITY.md`](SECURITY.md), not a public
issue.
