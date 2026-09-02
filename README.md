# UE5M3 FP4 for language model pretraining

This repository contains the released UE5M3 FP4 software behind the report
*UE5M3 FP4 Block Scaling for Stable Language Model Pretraining*. It includes
the portable numerical reference, the CUDA/Triton fake-quantization and
probe-matched GEMM path used by the proposed experiments, a public
Nemotron-H 8B integration for pinned upstream TorchTitan, and sanitized result
artifacts from the reported runs.

The code is released under Apache-2.0. Version `0.2.0a0` is an alpha research
release: interfaces may change, and the supported reproduction targets below
are deliberately narrower than “rerun every historical job byte for byte.”

## Reproduction scope

Four different claims are relevant here. They are not interchangeable.

| Target | Status | What this repository supports |
|---|---|---|
| Exact released numerical and kernel path | Included | E2M1 payload quantization, UE5M3 block-scale quantization, selective stochastic rounding, D=50 sample-and-hold scaling, and the CUDA/Triton K=64 issue-RZ probe-matched GEMM implementation are released with focused tests. |
| Public TorchTitan method reproduction | Runnable, with native runtime gates | BF16, proposed UE5M3 B=16/B=32, the generic-Torch GEMM control, and the software UE5M3-with-TE-settings comparator run through pinned TorchTitan, public Nemotron-H assets, and a deterministic public OLMo Mix loader. Two native NVFP4 configs additionally require the pinned Transformer Engine build and suitable Blackwell hardware. |
| Audited paper result artifacts | Included | Sanitized held-out validation and downstream OLMES tables, provenance hashes, and deterministic figure/table renderers are checked in. These are outputs of the completed paper runs, not newly generated public reruns. |
| Byte-identical historical rerun | Unavailable | Historical model checkpoints, the private Mosaic MDS object inventory and cursor, and the frozen quantized-OLMES request archive are not distributed. The original jobs therefore cannot be replayed byte for byte from this repository. |

“Exact numerical and kernel path” means the implementation of the proposed
software numerical model is present. It does not mean native UE5M3 hardware
execution, bitwise equality across arbitrary CUDA/PyTorch/GPU versions, or a
hardware-throughput reproduction. The probe-matched GEMM is a software model
of the measured output behavior; the selected final `1/1024` grid was
consistent with the probes but was not uniquely identified by them.

The public TorchTitan run reconstructs the reported method and configuration,
but consumes a new deterministic data identity. A new run should be described
as a public method reproduction, not as the source of the checked-in paper
metrics.

## Proposed recipe

| Choice | B=16 setting |
|---|---|
| Payload | signed E2M1 FP4 |
| Block scale | unsigned E5M3 (UE5M3) |
| Block size | 16 values; a runnable B=32 ablation is also provided |
| Default tensor-scale target | 448 |
| Delayed tensor scaling | periodic sample-and-hold, `D=50` optimizer steps |
| Activation and weight rounding | round-to-nearest, ties-to-even |
| Backward rounding | eight-bit-midpoint `StochasticFast`, only for `dY` |
| Weight scaling | two-dimensional |
| Randomized Hadamard transform | disabled |
| Eligible internal linears | all 112; the vocabulary output head is excluded from FP4 and uses FP32 compute |
| Inference activation policies | current-tensor D=1, cold D=50 replay, or calibrated/frozen |

Delayed scaling is not a rolling or windowed maximum. Each operand samples its
current maximum absolute value on optimizer steps 1, 51, 101, and so on, then
holds that value for the intervening steps. The training integration advances
this state once per optimizer step, not once per gradient-accumulation
microbatch.

For `Y = X W^T`, stochastic rounding is applied only to the upstream-gradient
operand `dY` in `dX = dY W` and `dW = dY^T X`. Forward activations, weights,
saved activations, and block-scale codes use deterministic ties-to-even
rounding. The target 2,048 override applies only to `dY` in the weight-gradient
GEMMs for the final four MLP `mixer.down_proj` modules.

See [the numerical protocol](reproduce/docs/numerical_protocol.md) for the
complete operand, scale, accumulation, and inference definitions.

## Repository layout

```text
src/ue5m3_fp4/
  formats.py                    portable E2M1 and UE5M3 reference
  scaling/                      training and post-load inference lifecycles
  nn/                           autograd-aware FP4 linear conversion
  backends/triton/              released CUDA/Triton quantization and GEMM
  integrations/torchtitan/      Nemotron-H model, data, converter, and hooks
  checkpoint.py                 strict local HF checkpoint loading
  cli/evaluate.py               held-out BF16 and quantized validation CLI
reproduce/
  configs/                      runnable core and comparator TorchTitan TOMLs
  historical_specs/             record-only transcriptions of historical jobs
  reference_results/            sanitized audited paper outputs and renderers
  diagnostics/                  public accumulator/GEMM diagnostic reruns
  scale_target/                 value-distribution capture and archived histograms
  scripts/                      data, asset, training, export, and eval tools
third_party/                    pinned TorchTitan, Transformer Engine, OLMES
tests/                          CPU contracts and CUDA/Triton tests
```

Files under `reproduce/historical_specs/` are provenance records and are not
accepted by the public training launcher. Executable comparator configurations
belong under `reproduce/configs/`. The software UE5M3-with-Transformer-Engine-
settings comparator uses this project's Triton path. Native NVFP4 comparator
code is runtime gated: it requires the exact pinned Transformer Engine build
and NVFP4-capable Blackwell hardware, neither of which is supplied merely by
installing the Python package. Availability of a historical spec alone is not
evidence that its runtime behavior has been reproduced.

## Install the portable package

Python 3.11 or newer is required. For the portable reference and development
suite:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

PYTHONPATH=src:third_party/torchtitan python -m pytest
PYTHONPATH=src python examples/tiny_mlp.py
```

The CPU reference validates represented values, rounding, scale lifecycles,
autograd behavior, conversion, and provenance. CUDA/Triton tests require a
compatible NVIDIA GPU and are skipped when that runtime is unavailable.

The synthetic example supports all three inference policies:

```bash
PYTHONPATH=src python examples/tiny_mlp.py --activation-mode current_tensor
PYTHONPATH=src python examples/tiny_mlp.py --activation-mode training_replay
PYTHONPATH=src python examples/tiny_mlp.py --activation-mode calibrated_frozen
```

These examples test numerical control flow; they do not measure language model
quality.

## Run the public 8B method reproduction

Use a source checkout because the pinned upstream projects are Git submodules:

```bash
git clone --recurse-submodules https://github.com/MrHuff/ue5m3-fp4.git
cd ue5m3-fp4
git submodule update --init --recursive
docker build --platform=linux/arm64 \
  -f reproduce/Dockerfile -t ue5m3-fp4:0.2.0a0 .
```

The container targets Linux arm64, builds the pinned Mamba packages, and replaces the base image's
Transformer Engine with the source-pinned version used by the comparator
integration. A complete Git checkout is required because the build verifies
the Transformer Engine gitlink before compilation.
An x86 host needs an arm64 builder or emulation for this exact image.

Download the immutable Nemotron configuration, tokenizer, and remote code.
The downloader excludes all model-weight formats and records a content
manifest. It also applies a hash-locked Mamba `out_proj` dispatch patch needed
to keep converted FP4 linears on the quantized path.

```bash
python reproduce/scripts/download_nemotron_assets.py \
  --output-dir /data/nemotron-h-assets
python reproduce/scripts/download_nemotron_assets.py \
  --output-dir /data/nemotron-h-assets --verify-only
```

Before allocating model weights, verify that the public adapter constructs the
expected 8B architecture and applies the requested conversion on the meta
device:

```bash
PYTHONPATH=src:third_party/torchtitan python \
  reproduce/scripts/smoke_nemotron_h.py \
  --assets /data/nemotron-h-assets \
  --numeric-path ue5m3-proposed-b16 \
  --meta-only
```

Omit `--meta-only` to run a synthetic full-model integration check on CUDA.
For a backward check, use sequence length 16 for BF16 or 64 for an FP4 path:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_CUDNN_SDPA_ENABLED=0 \
  PYTHONPATH=src:third_party/torchtitan python \
  reproduce/scripts/smoke_nemotron_h.py \
  --assets /data/nemotron-h-assets \
  --numeric-path ue5m3-proposed-b16 \
  --sequence-length 64 \
  --output /results/ue5m3/smoke-b16.json
```

This uses random token IDs, performs no optimizer update, and never reads a
checkpoint. It is an architecture, conversion, execution, and finite-gradient
check—not training-quality or throughput evidence.

Prepare all 32 shards of both public OLMo Mix streams as described in
[the reproduction guide](reproduce/README.md), then launch a configuration
from `reproduce/configs/`:

```bash
export UE5M3_NEMOTRON_ASSETS=/data/nemotron-h-assets
export UE5M3_DATA_ROOT=/data/olmo-mix
export UE5M3_OUTPUT_ROOT=/results/ue5m3

bash reproduce/scripts/run_torchtitan_train.sh \
  reproduce/configs/nemotron_h_8b_ue5m3_b16.toml
```

The proposed B16/B32 and software TE-settings configs use this project's
Triton path. The generic-Torch control retains the same encoded operands but
uses a Torch FP32 matrix multiplication. The two native NVFP4 configs fail
closed unless the pinned Transformer Engine build reports NVFP4 support on
suitable hardware.

The launcher defaults to one process for a smoke launch. The paper used 32
GB200 GPUs and global batch 768; topology variables and the complete data
preparation procedure are documented in
[`reproduce/README.md`](reproduce/README.md).

The public loader implements a balanced, checkpointable 82/18 document
schedule over local Hugging Face `Dataset.save_to_disk` shards. It preserves
BOS/EOS tokenization and persistent non-overlapping 8,193-token packing, but it
cannot reconstruct the unavailable historical MDS object order or cursor.

## Export and evaluate checkpoints

Training checkpoints contain learned BF16 master weights. Process-local
delayed-amax caches are intentionally not serialized. Convert a public
TorchTitan DCP checkpoint to standard Hugging Face safetensors:

```bash
PYTHONPATH=src:third_party/torchtitan python \
  reproduce/scripts/export_hf_checkpoint.py \
  /results/ue5m3/run/checkpoint/step-30000 \
  /results/ue5m3/hf/step-30000 \
  --hf-assets /data/nemotron-h-assets
```

The validation CLI expects ordered local safetensors containing an integer
rank-two tensor named `tokens`, with one target-only token after each input
sequence. Freeze 768 validation rows and 64 disjoint calibration rows from
caller-supplied tokenized JSONL while recording a new public identity:

```bash
python reproduce/scripts/prepare_validation_tokens.py \
  --input-jsonl /data/validation/ordered-token-rows.jsonl \
  --output /data/validation/frozen
```

Evaluate the same learned checkpoint in BF16 or with post-load FP4 fake
quantization:

```bash
# Learned-weight BF16 reference
bash reproduce/scripts/run_evaluation.sh validation \
  --checkpoint /results/ue5m3/hf/step-30000 \
  --hf-assets /data/nemotron-h-assets --local-files-only \
  --validation /data/validation/frozen/validation \
  --numeric-path bf16 \
  --output /results/ue5m3/validation/bf16.json

# Proposed B=16 with a cold, ordered D=50 inference replay
bash reproduce/scripts/run_evaluation.sh validation \
  --checkpoint /results/ue5m3/hf/step-30000 \
  --hf-assets /data/nemotron-h-assets --local-files-only \
  --validation /data/validation/frozen/validation \
  --numeric-path ue5m3-b16 --activation-mode training_replay \
  --output /results/ue5m3/validation/b16-d50.json
```

The evaluator calls `model.eval()`, resets delayed state, samples and freezes
loaded weight maxima, and then initializes the selected activation policy. A
cold D=50 replay is order dependent: one work unit is one complete batch-one
forward, and input order is part of the result identity. Current-tensor D=1 is
available with `--activation-mode current_tensor`; calibrated/frozen scaling
requires a separately supplied, disjoint calibration stream.

The exact validation records used for the paper are not distributed. Running
the evaluator on a replacement set validates the released code but produces a
new experiment, not the reported validation curve.

The public OLMES wrapper runs the pinned suites through all seven released
numeric paths for a complete Hugging Face model. By default it rebuilds the
public tasks and records a new ordered-request identity. A byte-identical
frozen-request replay is enabled only when the caller supplies the exact frozen
manifest and archive and every hash, member, and request-file check passes; the
archive itself is not distributed here. See
[`reproduce/evaluation/README.md`](reproduce/evaluation/README.md).

Build its separate CPython 3.12/aarch64 system-site virtual environment with a
hash-locked project-added dependency set, without changing the training
environment:

```bash
bash reproduce/scripts/install_olmes_runtime.sh /opt/ue5m3-olmes-venv
```

## Audited paper results

[`reproduce/reference_results/`](reproduce/reference_results/) contains:

- 84 held-out-loss points: seven trajectories at twelve checkpoints;
- 144 same-step validation comparisons;
- the seven-configuration by three-benchmark OLMES aggregate matrix;
- 1,022 OLMES leaf-task metric rows and 36 paired aggregate differences;
- source/output hashes and the explicit single-training-seed limitation; and
- deterministic scripts for rendering the published comparison figures and
  compact tables.

These tables were copied from the reviewed completed-run collectors with
private storage columns removed. The generator verifies their expected shape
and source hashes when given the reviewed paper-data directory. The renderer
can be run directly against the checked-in sanitized tables:

```bash
python reproduce/reference_results/render_reference_artifacts.py
```

Sequence/task bootstrap intervals describe variation over frozen evaluation
examples. Every training configuration has one independent trajectory, so
those intervals do not estimate training run-to-run variability and do not
support statistical-significance claims about retraining.

## Numerical diagnostics and value distributions

[`reproduce/diagnostics/`](reproduce/diagnostics/) contains the public
accumulator-statistics, native-GEMM-oracle, and tiny optimizer-control runners,
plus sanitized summaries of paper diagnostics whose exact historical harnesses
are unavailable. [`reproduce/scale_target/`](reproduce/scale_target/) contains
the checkpoint-and-token interface for capturing late-layer `X`, `W`, `dY`,
and block-scale-code histograms, the archived numerical captures used by the
paper figure, and its single-seed 350M T=448/T=2048 ablation records. Neither
directory contains weights or token data; their READMEs distinguish runnable
synthetic/public captures from record-only historical evidence.

## What is not distributed

The repository intentionally contains no model weights, private storage
addresses, credentials, or cluster-specific launcher glue. It also does not
contain the historical checkpoints, historical Mosaic MDS shard/index data and
cursor, the frozen 368,932-request quantized-OLMES archive, or the historical
paper-run container image. The public Dockerfile reconstructs the released
runtime from pinned public sources for new runs.

Consequently:

- the proposed numerical implementation and public TorchTitan method can be
  rerun;
- the checked-in paper artifacts can be audited and re-rendered;
- reported metrics cannot be recomputed from the original inputs using this
  repository alone; and
- native NVFP4 comparator and throughput measurements require the separately
  pinned Transformer Engine runtime, supported Blackwell hardware, and inputs
  not bundled here.

## Verification, provenance, and citation

Run the release checks appropriate to your environment:

```bash
ruff check src tests examples reproduce
ruff format --check src tests examples reproduce
PYTHONPATH=src:third_party/torchtitan python -m pytest
python reproduce/scripts/validate_bundle.py
python -m build
```

The checks actually completed for this candidate, along with known gaps, are
recorded in [`docs/verification.md`](docs/verification.md). Release procedure
and no-overclaim gates are in
[`docs/release_checklist.md`](docs/release_checklist.md).

Source and third-party attribution are recorded in [`NOTICE`](NOTICE) and
[`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md). Cite the report and record the
software version or exact Git commit; machine-readable citation metadata is in
[`CITATION.cff`](CITATION.cff).

## License and contact

The project is licensed under the [Apache License 2.0](LICENSE). Third-party
submodules remain under their own licenses.

For correspondence, contact Robert Hu at `robert.stats.hu@gmail.com`.
