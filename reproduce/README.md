# Reproducing the 8B UE5M3 experiments

This directory replaces the private cluster launcher with pinned upstream
TorchTitan, an architecture-specific Nemotron-H adapter, the released
probe-matched Triton path, and storage-neutral scripts. The configurations
under `configs/` cover BF16, proposed UE5M3 B=16/B=32, the generic-Torch GEMM
control, software UE5M3 with Transformer Engine settings, and two native
NVFP4 variants. The native variants fail closed unless the exact pinned
Transformer Engine build and suitable Blackwell hardware are available. Files
under `historical_specs/` are record-only transcriptions; the launcher
deliberately rejects them.

The public method path fixes the model architecture, initialization, linear
coverage, FP4 operands, rounding, K=64 issue-RZ accumulation, 1/1024 output
snap, selective T=2048 override, and D=50 optimizer-step sample-and-hold
semantics. The original private MDS object inventory and cursor are unavailable.
The provided deterministic 82/18 Hugging Face loader therefore creates a new
data identity and supports method reproduction, not the exact historical token
order or a promise of bit-identical loss.

## 1. Initialize and build the runtime

Clone with all pinned public dependencies:

```bash
git clone --recurse-submodules https://github.com/MrHuff/ue5m3-fp4.git
cd ue5m3-fp4
git submodule update --init --recursive
```

The checked-in container targets Linux arm64, starts from NVIDIA PyTorch 25.10, and builds
`causal-conv1d==1.6.2.post1` and `mamba-ssm==2.3.2.post1`, and replaces the
base image's Transformer Engine with commit
`01aef4fc721bd12fd09cd56d53a314aee1b953d6` (version
`2.16.0.dev0+01aef4fc`). The Mamba versions match both the historical launcher
request and the attested validation/OLMES runtime. The Docker build verifies
the Transformer Engine Git revision before compiling it:

```bash
docker build --platform=linux/arm64 \
  -f reproduce/Dockerfile -t ue5m3-fp4:reproduce .
```

An x86 host needs an arm64 builder or emulation for this exact image.

For a host installation using an equivalent NVIDIA PyTorch 25.10 environment:

```bash
python -m pip install --no-deps --require-hashes \
  -r reproduce/environment/requirements-ngc-25.10.lock
python -m pip install --no-build-isolation --require-hashes \
  -r reproduce/environment/compiled-requirements-ngc-25.10.lock
python -m pip install \
  "cmake==3.31.6" "ninja==1.13.0" "pybind11[global]==3.0.1" \
  "packaging==25.0"
git -C third_party/TransformerEngine submodule update --init --recursive
python -m pip uninstall -y \
  transformer_engine transformer-engine-cu12 transformer-engine-cu13 \
  transformer-engine-torch || true
NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=100a MAX_JOBS=8 \
  python -m pip install --no-build-isolation --force-reinstall --no-deps \
  ./third_party/TransformerEngine
python -m pip install --no-deps -e .
PYTHONPATH=third_party/torchtitan python reproduce/scripts/verify_runtime.py
```

The verifier intentionally rejects the NVIDIA base image's preinstalled
Transformer Engine 2.9.0. A host setup must build the pinned submodule with
`NVTE_FRAMEWORK=pytorch` before the command above can pass. The Dockerfile is
the authoritative sequence for that build.

Downstream OLMES evaluation uses a separate system-site virtual environment
because its reviewed dependency set includes a different Transformers version.
On CPython 3.12 Linux aarch64, install the hash-locked project-added PyPI set
and immutable VCS sources with:

```bash
bash reproduce/scripts/install_olmes_runtime.sh /opt/ue5m3-olmes-venv
```

## 2. Download model code and tokenizer assets (never weights)

The asset downloader is pinned to
`nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base` revision
`78dc93a79e2533922ac8ad2c16f79b7fb747970d`. Its allowlist excludes every
known parameter format. It then applies and verifies a narrow source patch so a
converted Mamba `out_proj` is called through `module.forward` rather than being
bypassed by the dense fused-kernel argument.

```bash
python reproduce/scripts/download_nemotron_assets.py \
  --output-dir /data/nemotron-h-assets
python reproduce/scripts/download_nemotron_assets.py \
  --output-dir /data/nemotron-h-assets --verify-only
```

The generated `UE5M3_ASSET_MANIFEST.json` records every file digest, the
original remote-code digest, the patched digest, and `weights_included=false`.

Before a costly launch, construct the exact architecture and conversion on the
meta device:

```bash
PYTHONPATH=src:third_party/torchtitan python \
  reproduce/scripts/smoke_nemotron_h.py \
  --assets /data/nemotron-h-assets \
  --numeric-path ue5m3-proposed-b16 \
  --meta-only \
  --output /results/ue5m3/meta-smoke-b16.json
```

The script checks the 8,084,075,520-parameter, 52-layer model contract without
allocating parameter storage. To exercise a full synthetic forward/backward on
a suitable CUDA device, omit `--meta-only`. Quantized backward requires
`batch_size * sequence_length` to be divisible by 64; for example:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_CUDNN_SDPA_ENABLED=0 \
  PYTHONPATH=src:third_party/torchtitan python \
  reproduce/scripts/smoke_nemotron_h.py \
  --assets /data/nemotron-h-assets \
  --numeric-path ue5m3-proposed-b16 \
  --sequence-length 64 \
  --output /results/ue5m3/full-smoke-b16.json
```

This script uses random token IDs, performs no optimizer update, and never
loads a checkpoint. Its loss, elapsed time, and memory fields are integration
diagnostics, not model-quality or throughput measurements.

## 3. Prepare the public OLMo Mix method reconstruction

The source is `allenai/olmo-mix-1124` revision
`8162bd79c6dc4fea470506531a8d791badc06b4b`. Run both transformations for
shards 0 through 31 (normally as parallel data-preparation jobs):

```bash
for shard in $(seq 0 31); do
  python reproduce/scripts/prepare_olmo_mix_1124.py dclm \
    --cache-dir /data/hf-cache --output-root /data/olmo-mix --shard "$shard"
  python reproduce/scripts/prepare_olmo_mix_1124.py no-dclm \
    --cache-dir /data/hf-cache --output-root /data/olmo-mix --shard "$shard"
done

python reproduce/scripts/prepare_olmo_mix_1124.py inspect \
  --cache-dir /data/hf-cache --output-root /data/olmo-mix
```

Each data-parallel rank receives prepared shards deterministically. Documents
are selected with a balanced, checkpointed 82/18 schedule, tokenized with BOS
and EOS, appended to a persistent buffer, and emitted in non-overlapping
8,193-token windows as 8,192 inputs plus shifted targets. The public loader
fixes DataLoader `num_workers=0`; dataset iteration therefore runs in each
data-parallel rank's training process, giving the row cursor and token-buffer
remainder one checkpoint-state owner per rank. See
`docs/data_and_checkpoints.md` for the precise difference from the unavailable
historical MDS ordering.

## 4. Launch training

Set local paths and choose an executable config:

```bash
export UE5M3_NEMOTRON_ASSETS=/data/nemotron-h-assets
export UE5M3_DATA_ROOT=/data/olmo-mix
export UE5M3_OUTPUT_ROOT=/results/ue5m3

bash reproduce/scripts/run_torchtitan_train.sh \
  reproduce/configs/nemotron_h_8b_ue5m3_b16.toml
```

The launcher defaults to the checked-in
`third_party/torchtitan/torchtitan/train.py` and puts this package plus the
submodule on `PYTHONPATH`. A one-node, one-GPU launch is the safe default. The
reported topology used eight nodes with four GB200 GPUs per node:

```bash
export UE5M3_NNODES=8
export UE5M3_NPROC_PER_NODE=4
export UE5M3_NODE_RANK=0             # set separately on every node
export UE5M3_MASTER_ADDR=trainer-0   # shared rendezvous host
export UE5M3_MASTER_PORT=29500
```

At 32 data-parallel ranks the global batch of 768 gives 24 accumulation
microbatches per optimizer step. The public plugin advances D=50 scaling once
per optimizer step, not once per microbatch. TorchTitan writes a fully resolved
configuration and DCP checkpoints every 2,500 steps.

## 5. Export and evaluate a checkpoint

Convert a local DCP step to standard HF safetensors without copying it into the
repository:

```bash
PYTHONPATH=src:third_party/torchtitan python \
  reproduce/scripts/export_hf_checkpoint.py \
  /results/ue5m3/nemotron_h_8b_ue5m3_b16/checkpoint/step-30000 \
  /results/ue5m3/hf/ue5m3-b16-step-30000 \
  --hf-assets /data/nemotron-h-assets
```

Validation inputs are local safetensors containing an integer rank-two tensor
named `tokens`, with shape `[N, 8193]`. Evaluate BF16 learned weights or the
post-load quantized path as follows:

```bash
PYTHONPATH=src python -m ue5m3_fp4.cli.evaluate \
  --checkpoint /results/ue5m3/hf/ue5m3-b16-step-30000 \
  --hf-assets /data/nemotron-h-assets --local-files-only \
  --validation /data/validation/tokens.safetensors \
  --numeric-path ue5m3-b16 --activation-mode training_replay \
  --output /results/ue5m3/validation/b16-step-30000.json
```

Use `--numeric-path bf16` without `--activation-mode` for the learned-weight
BF16 reference, or `--activation-mode current_tensor` for D=1. The evaluator
computes FP32 token cross entropy, accumulates in FP64, records file/row and
checkpoint identities, freezes post-load weight maxima, and records the exact
inference scale lifecycle. It requires 768 ordered records unless
`--allow-partial-data` is explicitly used for a smoke test.

The paper's frozen validation records and frozen OLMES request archive are not
redistributed here. Their expected identities and ordering contracts remain in
`manifests/validation.yaml` and `manifests/olmes.yaml`; replacing either input
creates a new evaluation identity and cannot reproduce the reported metric.

## Inventory

- `configs/`: seven public TorchTitan configurations; native NVFP4 entries are
  runtime gated as described above.
- `historical_specs/`: four non-runnable comparator/control transcriptions.
- `manifests/`: architecture, data, validation, and OLMES contracts.
- `docs/`: numerical and data/checkpoint details.
- `diagnostics/`: public numerical reruns and sanitized diagnostic summaries.
- `scale_target/`: public value-distribution capture and archived histogram
  evidence.
- `environment/` and `Dockerfile`: concrete training and OLMES runtimes.
- `scripts/smoke_nemotron_h.py`: meta construction and synthetic full-model
  integration check for all seven numerical paths.
- `scripts/audit_public_tree.py`: current-tree and reachable-history privacy,
  secret, symlink, size, and model-weight audit.
- `scripts/validate_bundle.py`: syntax and cross-file invariant checks.

Run the static bundle check before launching:

```bash
python reproduce/scripts/validate_bundle.py
python reproduce/scripts/audit_public_tree.py --history
```
