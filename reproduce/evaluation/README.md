# Public evaluation commands

## Freeze local token rows

The evaluator's storage-neutral input is a safetensors tensor named `tokens`
with shape `[N, 8193]` and integer storage. Freeze exactly 768 ordered
validation rows followed by 64 disjoint calibration rows from one or more
JSONL files with:

```bash
python reproduce/scripts/prepare_validation_tokens.py \
  --input-jsonl /path/to/ordered-token-rows.jsonl \
  --output /path/to/frozen-evaluation-data
```

Each non-empty line is either a JSON array of token IDs or an object whose
`tokens` field is that array. The command rejects duplicate rows and unexpected
counts, creates deterministic shards under `validation/` and `calibration/`,
and writes `manifest.json` with source, shard, and ordered-row hashes. These
caller-supplied rows form a **new public identity**. They do not recreate the
unavailable historical post-cursor validation order and must not be described
as the paper's byte-identical validation set.

## Held-out next-token validation

The public evaluator accepts a local Hugging Face safetensors export plus the
pinned public Nemotron-H assets. It configures and verifies SDPA, retains the
output projection's BF16 checkpoint parameter while computing that matrix
multiplication in FP32, and computes FP32 cross-entropy with FP64 sums.

BF16 learned-weight reference:

```bash
python -m ue5m3_fp4.cli.evaluate \
  --checkpoint /path/to/hf-weight-export \
  --validation /path/to/ordered-validation-shards \
  --numeric-path bf16 \
  --output validation-bf16.json
```

Probe-matched UE5M3 block-16 with a cold D=50 activation cache:

```bash
python -m ue5m3_fp4.cli.evaluate \
  --checkpoint /path/to/hf-weight-export \
  --validation /path/to/ordered-validation-shards \
  --numeric-path ue5m3-proposed-b16 \
  --activation-mode training_replay \
  --output validation-b16-d50.json
```

Use `ue5m3-proposed-b32` for block size 32, `current_tensor` for D=1, or
provide a disjoint calibration set for frozen activation scales:

```bash
python -m ue5m3_fp4.cli.evaluate \
  --checkpoint /path/to/hf-weight-export \
  --validation /path/to/ordered-validation-shards \
  --calibration /path/to/ordered-disjoint-calibration-shards \
  --numeric-path ue5m3-b16 \
  --activation-mode calibrated_frozen \
  --output validation-b16-calibrated.json
```

By default, validation must contain exactly 768 ordered records and calibration
must contain exactly 64, each with 8,192 input tokens plus one target-only
token. `--allow-partial-data` is an explicit smoke-test escape hatch. The CLI
hashes all files and token rows, rejects duplicate records, verifies calibration
is disjoint from validation, loads learned BF16 weights, calls `model.eval()`,
then resets scale state and freezes each loaded weight tensor's global amax.
The D=50 counter advances once per complete batch-one forward and starts cold.

The same CLI exposes all seven reported numerical paths. In addition to BF16
and the two proposed paths, use `ue5m3-torch-control`, `ue5m3-te-settings`,
`native-nvfp4-te`, or `native-nvfp4-no-rht-all`. The TE-settings comparator is
fixed to current-tensor D=1. Native choices require the exact pinned
Transformer Engine build and capable Blackwell hardware; they fail instead of
substituting a software path. Each native module enables the required internal
row alignment before its first quantizer initialization and records its
effective current-tensor state.

The default model source is
`nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base` at revision
`78dc93a79e2533922ac8ad2c16f79b7fb747970d`; its architecture is overridden by
the checked 8B paper configuration before weights are loaded. Supply a local
snapshot with `--hf-assets` and `--local-files-only` for an offline run.

## OLMES

Install the hash-pinned submodule and its dependencies, then dry-run or execute
the public task reconstruction against a **complete** HF directory (weights,
config, tokenizer, and pinned remote code). Select one of the same seven paths
with `UE5M3_OLMES_NUMERIC_PATH`:

```bash
bash reproduce/scripts/install_olmes_runtime.sh /path/to/new-olmes-venv
source /path/to/new-olmes-venv/bin/activate
export UE5M3_OLMES_NUMERIC_PATH=ue5m3-proposed-b16
bash reproduce/evaluation/run_public_olmes.sh \
  /path/to/complete-hf-model /path/to/olmes-output --dry-run
```

The installer uses the released CPython-3.12/Linux-aarch64 hash lock and
verifies the OLMES gitlink/tree plus pinned VCS dependencies. The wrapper
verifies the public OLMES ancestor at commit
`8e2743734066b073c5d8498d1b8220f67a21a2d6` and fixes the suites, batch size,
maximum length, random-subsample seed, GPU/worker counts, and raw-request
capture used by the paper. Its fail-closed runtime hook disables the OLMES
logits cache, calls `model.eval()`, installs and verifies the required SDPA
attention path with cuDNN SDPA disabled, and computes the output projection in
FP32 without copying its BF16 checkpoint parameter. A successful non-dry run writes
`ue5m3-olmes-runtime.json` alongside the OLMES outputs; the wrapper rejects a
run that produces no runtime attestation. For the three D=50 paths, one cold
logical counter advances immediately before every top-level model forward and
is never reset between the 146 leaf tasks. The attestation hashes that complete
ordered forward-input stream.

The historical evaluator used descendant commit
`3d80ebb0f08706a5d2dd3fb0be72100735b5f5c6`. Its changes outside the public
ancestor affect HELMET/judge paths that are not among these 146 likelihood
tasks, dependency declarations covered by the released lock, default chat
handling that is inactive because all selected tasks have
`use_chat_format=false`, and colon-free output filenames. The public runtime
restores that filename behavior and records both source identities in its
attestation, avoiding any dependency on a non-public remote.

The historical sweep replayed a frozen archive of 368,932 likelihood requests.
That archive is not included in this public repository. The default
`UE5M3_OLMES_REQUEST_MODE=public_task_rebuild` therefore creates and records a
new request identity; it must not be described as a byte-identical rerun.

If the exact frozen artifacts are available, enable byte-identical
frozen-request replay:

```bash
export UE5M3_OLMES_REQUEST_MODE=frozen_request_archive
export UE5M3_OLMES_REQUEST_MANIFEST=/path/to/request-bundle-manifest.json
export UE5M3_OLMES_REQUEST_ARCHIVE=/path/to/request-bundle.tar.gz
bash reproduce/evaluation/run_public_olmes.sh \
  /path/to/complete-hf-model /path/to/empty-olmes-output
```

The staging helper requires the manifest SHA-256
`b7cd708300b7b63edd45e4d973de7195b2c98384f1a9b0773f49c5a8d0e47898`
and archive SHA-256
`0bf27af57eb1bb1b98872c4af12d419498652d935a6b745cc7ec4ecdb32d7483`.
It verifies both canonical inventories, safely extracts only declared files,
forces the bundled Hugging Face cache offline, stages all 146 request files,
and applies the reviewed replay gate only to a temporary copy of the pinned
OLMES source. It verifies that every request file remains byte-identical after
evaluation. Unsupported numeric paths fail rather than silently running BF16.
The reviewed aggregate outputs remain in `reproduce/reference_results/`.
