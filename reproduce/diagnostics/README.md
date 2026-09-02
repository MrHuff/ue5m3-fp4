# Numerical diagnostics

This directory separates public reruns from compact summaries of historical
evidence. Every public runner uses synthetic inputs, writes a configuration
SHA256, records its runtime, and fails instead of changing numerical backends
when a prerequisite is unavailable.

## Public reruns

The statistical study is storage-neutral and defaults to the report settings:

```bash
python reproduce/diagnostics/run_accumulator_statistics.py \
  --device cuda:0 \
  --output-json outputs/issue-rz-statistics.json
```

Use `--quick --device cpu` for a small functional check. The FP4-like branch
in this particular diagnostic uses E2M1 values with E4M3 local scales because
the study isolates the accumulator behavior observed in native NVFP4. It is
not presented as the UE5M3 training quantizer.

The companion denominator sweep combines a new seeded near-cancellation run
with the explicitly labelled compact summary of the archived native witnesses:

```bash
python reproduce/diagnostics/run_final_grid_sweep.py \
  --output-json outputs/final-grid-sweep.json
```

The native oracle requires a Blackwell GPU and the exact Transformer Engine
revision pinned as the `third_party/TransformerEngine` submodule. It compares
native FP32 output with the public `issue_rz_bf16_gemm` implementation on a
large deterministic matrix and on deterministic physical-K64 permutations:

```bash
python reproduce/diagnostics/run_native_gemm_oracle.py \
  --output-json outputs/native-gemm-oracle.json
```

Pass `--expected-te-library-sha256 HEX` when reproducing a particular binary
build. The runner always records the loaded native library hash. Its synthetic
permutation corpus is a new public validation corpus; it is not relabelled as
the report's checkpoint-derived 258-case corpus.

The tiny optimizer-level control uses the public UE5M3 quantizer and K64
issue-RZ GEMM in forward, data-gradient, and weight-gradient paths:

```bash
python reproduce/diagnostics/run_tiny_grid_regression.py \
  --gradient-rounding ties_to_even \
  --output-json outputs/grid-regression-tte.json

python reproduce/diagnostics/run_tiny_grid_regression.py \
  --gradient-rounding stochastic \
  --output-json outputs/grid-regression-sr.json
```

Its defaults reproduce the 81,920-parameter shape, three seeds, 500 updates,
and denominators used in the report. This is a CUDA/Triton experiment; there is
no CPU or ordinary-matmul fallback.

## Archived evidence and limits

`archived/report_summary.json` contains small, sanitized summaries of the
completed native-witness, 1.292B parity, tiny-regression, and 8B timing runs.
It deliberately contains no weights, private storage locations, job metadata,
or machine paths. The source artifact hashes identify the report artifacts
from which the summaries were transcribed.

The original 1.292B one-step/100-step harness and the exact 8B timing harness
depended on historical model and workload integration that is not present in
this standalone repository. They are therefore labelled archived evidence,
not rerunnable public experiments. The public TorchTitan training pipeline can
be used for new independent model-level runs, but those have new experiment
identities and must not be described as byte-for-byte replays of the archived
results.
