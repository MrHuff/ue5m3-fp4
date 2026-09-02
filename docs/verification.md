# Release-candidate verification

This page records the checks run against the `0.2.0a0` release candidate on
2026-09-02. It separates what was exercised from what remains dependent on
hardware, data, or checkpoints that are not distributed with this repository.

## Scope

The release contains four distinct kinds of reproducibility evidence:

1. The UE5M3 format, scaling lifecycle, Triton quantization kernels, and Triton
   FP4 GEMM are exact released implementations and have differential tests.
2. The public TorchTitan integration is a runnable method reproduction for new
   training runs with OLMo-mix-1124 and the pinned Nemotron-H architecture.
3. The checked-in validation and OLMES tables are audited paper-result
   artifacts. Their figures and derived tables are deterministically rendered.
4. A byte-identical rerun of the historical experiment is not claimed because
   the original trained checkpoints, frozen data objects, and frozen OLMES
   request archive are not redistributed.

The checks below do not turn a single training trajectory into an independent
replication and do not establish statistical significance across random seeds.

## Test environment

- Linux `aarch64`, kernel 6.12.77
- Python 3.12.3
- PyTorch `2.9.0a0+145a3a7bda.nv25.10`
- Triton 3.5.1
- PyYAML 6.0.3
- safetensors 0.6.2
- causal-conv1d `1.6.2.post1`
- cut-cross-entropy 25.9.3
- mamba-ssm `2.3.2.post1`
- Transformer Engine 2.9.0 in the exercised host runtime; the native path
  correctly rejected this version
- pinned Transformer Engine source checkout at commit
  `01aef4fc721bd12fd09cd56d53a314aee1b953d6`; its commit and expected package
  identity `2.16.0.dev0+01aef4fc` were verified, but it was not built or
  executed in this release check
- four NVIDIA GB200 GPUs, CUDA capability 10.0

The exercised host is not a supported-version matrix and did not contain the
pinned native Transformer Engine build. The lock files in
`reproduce/environment/` record the project-added runtime packages; the NVIDIA
container supplies the CUDA/PyTorch base. The Linux arm64 Dockerfile declares
the intended paper-compatible native-reference recipe: it replaces Transformer
Engine 2.9.0 from the base image with the pinned source revision and verifies
the resulting package version during the build. Docker execution was
unavailable for this release check, as recorded below.

## Python and CUDA tests

The complete suite was run with:

```bash
PYTHONPATH=src:third_party/torchtitan python -m pytest -q
```

Of 192 collected tests, 178 passed and 14 historical differential tests were
skipped because their optional comparison checkout was not configured. No test
failed. The provenance-integrity tests passed against the refreshed
`SOURCE_PROVENANCE.md` digest snapshot.

The exact CUDA differential suite was then run with the audited comparison
checkout enabled:

```bash
UE5M3_FP4_HISTORICAL_ROOT=<audited-comparison-checkout> \
  PYTHONPATH=src \
  python -m pytest -q \
    tests/test_triton_gemm.py \
    tests/test_triton_quantization.py
```

Result: 26 tests passed. These tests compare the public quantizers and GEMM
against the research implementation in addition to exercising their public
APIs. PyTorch emitted one TF32 deprecation warning; no numerical test failed.

## Nemotron-H integration smoke test

The public asset downloader fetched and hash-checked the seven pinned
architecture/tokenizer files and rejected model-weight files. The generated
inventory records each accepted file's source revision and SHA-256 digest.

Using those local assets, the checked-in smoke script constructed and converted
the exact model architecture on the meta device:

```bash
PYTHONPATH=src:third_party/torchtitan \
  <verified-runtime>/bin/python reproduce/scripts/smoke_nemotron_h.py \
  --assets <verified-nemotron-assets> \
  --numeric-path ue5m3-proposed-b16 \
  --meta-only
```

The observed structure and conversion count were:

```json
{
  "attention_mixer_count": 4,
  "causal_conv1d": "1.6.2.post1",
  "fp4_linears": 112,
  "layer_count": 52,
  "mamba_ssm": "2.3.2.post1",
  "parameter_count": 8084075520,
  "state_dict_roots": ["layers", "norm", "output", "tok_embeddings"]
}
```

This qualifies model registration, pinned remote code, topology, and state-dict
shape without allocating 8B parameters. It is not a training run. Building and
installing the two pinned Mamba dependencies from source also completed in a
clean temporary environment.

Two full-parameter forward/backward smokes then ran on one GB200 with synthetic
token IDs. The BF16 path used sequence length 16 and the proposed B16 path used
sequence length 64, as its GEMM reduction dimension must be divisible by 64.
Both found all 8,084,075,520 parameter-gradient elements finite and used about
35.1 GB of peak allocated CUDA memory. The proposed path converted all 112
eligible linears and recorded 336 scale entries and 336 first-step refreshes.

The checked-in synthetic smoke command exposes the same integration path. A
stable proposed-B16 invocation is:

```bash
CUDA_VISIBLE_DEVICES=<device> TORCH_CUDNN_SDPA_ENABLED=0 \
  PYTHONPATH=src:third_party/torchtitan \
  python reproduce/scripts/smoke_nemotron_h.py \
  --assets <verified-nemotron-assets> \
  --numeric-path ue5m3-proposed-b16 \
  --sequence-length 64 \
  --output <smoke-result.json>
```

These are execution and finite-gradient checks on synthetic inputs. Their
losses and wall times are neither model-quality measurements nor throughput
comparisons, and the two paths used different sequence lengths.

The dedicated OLMES installer also completed in a fresh system-site virtual
environment. It verified the published lock hash, installed its complete
hash-checked project-added PyPI set, installed AI2 OLMo at
`090253dac6688f2532509daa7aa2eb5fae50e956`, installed alpaca-eval at
`db85f8065408b842100436a45f56c65d3a0dd6a6`, and installed OLMES 0.1.0 from
the public pinned ancestor at
`8e2743734066b073c5d8498d1b8220f67a21a2d6`. An import smoke then loaded
OLMES 0.1.0, lm-eval 0.4.3, AI2 OLMo 0.6.0, alpaca-eval 0.6.6, and
Transformers 4.51.3 from that environment. The released storage-only filename
compatibility hook was also exercised.

## Reference-result rendering

The checked-in tables were regenerated into a temporary directory with:

```bash
python reproduce/reference_results/render_reference_artifacts.py \
  --output-directory <temporary-directory>
```

All six output sizes and SHA-256 digests matched
`reproduce/reference_results/generated/artifact_manifest.json`: the validation
PDF and PNG, final-validation CSV, OLMES-difference PDF and PNG, and OLMES
score/delta CSV. The manifest also verified the two input-table digests and the
formulas used for percentage loss and percentage-point score differences.

The checked-in reference inventory contains 84 validation metrics, 144
validation comparisons, 21 OLMES aggregate scores, 1,022 OLMES leaf-task rows,
and 36 OLMES paired differences. These are result records, not bundled model
weights.

## Metadata and distribution checks

The citation metadata passed the CFF 1.2 schema validator:

```bash
uvx --from cffconvert==2.0.0 cffconvert --validate -i CITATION.cff
```

The repository audit checked the current project files plus every reachable
project Git blob, excluding independently maintained submodule contents. It
also checked the staged public submodule URLs and exact gitlinks. It found no
model/checkpoint files, private location markers, oversized project files,
escaping symlinks, or credential-shaped secrets:

```bash
python reproduce/scripts/audit_public_tree.py --history
```

`detect-secrets` 1.5.0 was also run over the staged first-party tree. Its full
default entropy scan reported 149 documented hexadecimal hashes/commits and six
long converter identifiers; inspection confirmed that none was a credential.
With only the entropy heuristics disabled, all remaining credential-specific
detectors reported zero findings across zero files.

An isolated `python -m build` produced both the `0.2.0a0` wheel and source
distribution. Inventory checks found all required Triton kernels, integration
code, recipes, configs, scripts, the OLMES lock, and reference artifacts, and
found no checkpoint or model-weight extensions. The candidate sdist contained
190 entries and the wheel 46 entries. The wheel was installed with `--no-deps`
in a fresh system-site virtual environment; all four recipe resources loaded,
the distribution reported version `0.2.0a0`, and the evaluation entry point
rendered its help successfully. A final build from the reviewed tag is still a
release gate because refreshing provenance changes the sdist.

## Checks not completed here

- The Dockerfile could not be built in this environment because Docker, Podman,
  and Skopeo were unavailable. Its exact Transformer Engine source revision and
  installed version are checked during the image build and again by
  `reproduce/scripts/verify_runtime.py`.
- The historical paper-run container is recorded by reference and digest for
  provenance but is not publicly distributed. Public reruns use the checked-in
  Dockerfile and therefore receive a new runtime identity.
- A fresh end-to-end 8B pretraining run was not launched. It requires the
  documented compute budget and public OLMo-mix-1124 preparation.
- The historical checkpoints and frozen validation/OLMES inputs are not
  redistributed, so the paper results were not recomputed from weights in this
  release check.
- Native NVFP4 comparator execution requires supported Blackwell hardware and
  the pinned Transformer Engine build. GB200 hardware was available, but the
  exercised host had Transformer Engine 2.9.0 rather than the required pinned
  2.16 source build, so native execution was not qualified.
- A released Python/PyTorch/CUDA compatibility matrix has not yet been run.

These limitations are deliberately narrower than the released method: users
can train and evaluate new runs with public inputs, inspect every reported
aggregate artifact, and test the exact numerical kernels, but cannot recreate
private inputs that are not distributed.
