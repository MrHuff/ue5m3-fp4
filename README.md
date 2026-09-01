# UE5M3 FP4

This repository is a clean extraction of the UE5M3 block-scaling method for
FP4 pretraining and post-load quantized evaluation.  It contains the portable
reference path: the implementation models FP4 numerics in software and does
not claim native UE5M3 hardware acceleration.

The initial slice makes the numerical choices explicit:

- E2M1 payloads with unsigned E5M3 block scales;
- block size 16 and tensor target 448;
- periodic sample-and-hold tensor amax with a 50-step refresh interval;
- deterministic ties-to-even rounding for activations, weights, and scales;
- the paper's 8-bit-midpoint `StochasticFast` mode only for the upstream
  gradient `dY` in the data- and weight-gradient GEMMs;
- no randomized Hadamard transform; and
- three post-load activation-scale policies: current tensor, cold D=50 replay,
  and disjoint calibration followed by frozen scales.

The initial linear reference fake-quantizes operands, decodes them, and passes
float32 operands to PyTorch matrix multiplication. The resulting provenance
records PyTorch's runtime matmul-precision policy. This slice does not yet
include the probe-matched native-GEMM accumulation/output model used for the
paper's closest hardware comparison.

## Status

This is an alpha extraction, not yet a public release.  The portable core and
tests are being separated first; TorchTitan, Nemotron-H, Transformer Engine,
native kernels, checkpoint export, and cluster launchers are deliberately not
part of this slice.  See [the release checklist](docs/release_checklist.md).

## Install and test

```bash
python -m pip install -e '.[dev]'
pytest
```

The package can also be tested without installation:

```bash
PYTHONPATH=src pytest
```

Run the synthetic end-to-end example with:

```bash
PYTHONPATH=src python examples/tiny_mlp.py
```

## Why inference has an explicit scale lifecycle

Training checkpoints contain learned master weights but not the process-local
delayed-amax cache. Quantized evaluation therefore creates fresh inference
state, never consults a training cache, measures and freezes each loaded weight
tensor's amax, and selects one explicit activation policy before measurement.
In particular, cold D=50 replay counts
ordered inference work units; it does not pretend to restore an optimizer-step
cache from training.

## Provenance

The extraction is based on `gc-training` commit
`99a96f2a345ab4a9d37904cfdcdf93777458106d`.  It is built as a fresh repository
instead of publishing the monorepo history.  File-level and third-party
provenance is recorded in [NOTICE](NOTICE) and
[SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md).

## License

Apache License 2.0.  Publication remains subject to the organizational and
third-party checks listed in the release checklist.
