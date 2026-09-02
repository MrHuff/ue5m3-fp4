# Data, assets, and checkpoints

## Training data

The reported 30,000-step runs used the `shuffled-olmo-mix-1124` Mosaic-style
dataset adapter with two streams:

- `dclm`, proportion 0.82;
- `olmo-no-dclm`, proportion 0.18.

The historical loader used the training split, disabled shuffle, used two
workers, and sharded data by node. The public runnable configurations do not
depend on Mosaic or its private shard layout. They read the Hugging Face
`Dataset.save_to_disk` directories described below from `UE5M3_DATA_ROOT`.
Prepared shards are assigned to data-parallel ranks by sorted shard index
modulo the data-parallel world size.

The original shard manifest, cursor, and byte-exact data order are not included
in this repository. Matching only the 82/18 proportions does not recreate the
original samples or their order. Consequently, the public configuration
reconstructs the reported optimization setup with a new data identity. Even if
the historical data manifest became available, bitwise equality would still
depend on the complete runtime, RNG, checkpoint, and distributed-execution
state.

### Public method-reproduction command

The source dataset is pinned to `allenai/olmo-mix-1124` revision
`8162bd79c6dc4fea470506531a8d791badc06b4b`. The runtime lock uses Hugging Face
Datasets 3.6.0. Prepare individual shards locally:

```bash
python reproduce/scripts/prepare_olmo_mix_1124.py no-dclm \
  --cache-dir /path/to/hf-cache \
  --output-root /path/to/prepared-olmo \
  --shard 0

python reproduce/scripts/prepare_olmo_mix_1124.py dclm \
  --cache-dir /path/to/hf-cache \
  --output-root /path/to/prepared-olmo \
  --shard 0
```

Run both commands for shard indices 0 through 31, then run the script's
`inspect` command to validate the prepared inventory. The no-DCLM transformation
takes the matching non-contiguous shard from arxiv, starcoder, wiki, pes2o,
open-web-math, and algebraic-stack; concatenates them in that order; and applies
a seed-42 shuffle. The DCLM rewrite uses contiguous 32-way sharding and no
additional shuffle. The training loader then treats the resulting DCLM and
no-DCLM datasets as separate streams weighted 0.82 and 0.18.

The script writes local Hugging Face dataset directories. The public loader
selects documents with a balanced rational schedule: every consecutive 50
document selections contain 41 from DCLM and 9 from no-DCLM. DataLoader
`num_workers=0`, so dataset iteration runs in each data-parallel rank's training
process and the row cursor and token-buffer remainder have one checkpointable
owner. This deterministic schedule is a public reconstruction of the stated
82/18 method, not the original Mosaic scheduler.

Rewriting those rows as Mosaic MDS would be a storage conversion and would
require a new content manifest. The historical MDS `index.json`, object
boundaries, dependency lock, and complete source-cache ordering are unavailable,
so the public procedure reproduces the method but not the byte-exact historical
MDS stream.

During training, each sampled document is tokenized with both BOS and EOS,
appended to a persistent in-process dataset buffer, and consumed in non-overlapping
8,193-token windows. Each window yields its first 8,192 tokens as model input
and its last 8,192 tokens as one-token-shifted labels. Buffer leftovers cross
document boundaries and are checkpointed with the dataloader.

At global batch 768, sequence length 8,192, and 30,000 optimizer steps, each
trajectory processes 188,743,680,000 nominal input tokens. All configurations
used seed 42 once. This is one trajectory per configuration, not a multi-seed
experiment.

## Tokenizer and model assets

The model adapter uses configuration, tokenizer, and remote-code assets from
`nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base` at immutable revision
`78dc93a79e2533922ac8ad2c16f79b7fb747970d`, then instantiates the separate
reported 8B architecture recorded in `reported_experiments.yaml`. Download and
verify them with:

```bash
python reproduce/scripts/download_nemotron_assets.py \
  --output-dir /path/to/nemotron-h-assets
python reproduce/scripts/download_nemotron_assets.py \
  --output-dir /path/to/nemotron-h-assets --verify-only
```

The downloader allowlists only configuration, tokenizer, and Python files and
rejects model-weight formats. It also applies a hash-locked patch to the pinned
Mamba implementation: an ordinary dense `out_proj` retains the fused call,
whereas a converted FP4 `out_proj` receives the fused scan output through its
module `forward`. This prevents the fused kernel from silently bypassing the
quantized module during training. The generated manifest records a seven-file
inventory and `weights_included=false`.

A rerun must preserve and hash:

- `config.json` and all remote-code model/config files;
- tokenizer model, vocabulary, merges, special-token maps, and tokenizer
  configuration;
- the upstream repository identifier and immutable revision;
- the complete relative-path, size, and SHA-256 inventory.

Set `UE5M3_NEMOTRON_ASSETS` to that verified local directory. A mutable
model-hub branch name is not sufficient provenance. No checkpoint weights are
bundled or downloaded by this procedure.

## Checkpoint production

The reported runs wrote Torch distributed checkpoints every 2,500 optimizer
steps and retained all twelve checkpoints from step 2,500 through step 30,000.
The checkpoints contain learned BF16 master weights. Process-local delayed-amax
caches are numerical runtime state and are not serialized.

Resuming at one of the supplied 2,500-step checkpoint boundaries begins with a
scheduled D=50 refresh on the next optimizer step, so the missing cache is
replaced where an uninterrupted run would refresh it. A checkpoint taken and
resumed between scheduled refreshes would instead start its cache cold and
create a different numerical trajectory; that resume requires a new experiment
identity.

Use `reproduce/scripts/export_hf_checkpoint.py` to convert a public TorchTitan
DCP checkpoint into standard Hugging Face safetensors. The local evaluator
accepts that export plus one or more safetensors token files; each token file
must contain an integer rank-two tensor named `tokens` with width 8,193. See the
commands in `reproduce/README.md`.

For every new checkpoint, store the following immutable sidecars:

1. the fully resolved rank-zero training configuration and its SHA-256 digest;
2. source commit and dirty-state ledger;
3. dependency lock and container digest;
4. dataset manifest, cursor, and tokenizer/model-asset inventories;
5. distributed-checkpoint metadata and the digest/size of every referenced
   object;
6. optimizer step, consumed-token count, and resume lineage;
7. FP4 module-coverage inventory and all environment-based numerical overrides.

The native Transformer Engine reference trajectory was resumed after step
15,000. Model, optimizer, scheduler, and counters were restored, while its
dataloader/RNG stream restarted. Preserve this fact when comparing its curve or
attempting historical reproduction; do not describe the two job segments as an
uninterrupted data stream.

The UE5M3-with-Transformer-Engine-settings validation trajectory was recovered
from an audited complete log-checkpoint mirror because its canonical historical
root was incomplete. A new public run should write a normal complete checkpoint
and should not reproduce that storage accident.

## Held-out validation data

The frozen validation manifest has SHA-256
`ee78fd2c8f7e3b2516c03973105cc5fd589018e748196aa038eb234216958486`.
It orders 768 records by ascending original data-parallel rank and then ascending
row index. Each record supplies 8,192 model-input tokens plus one target-only
token, yielding 8,192 next-token losses and 6,291,456 evaluated tokens overall.
No padding or shuffle is permitted.

The manifest and its shard objects are not bundled here. A replacement held-out
set can test the public code but does not reproduce the paper's reported
validation values. For delayed-scale D=50 inference, even a permutation of the
same records changes scale-cache reuse and is a different experiment.

## Optional activation calibration data

The calibrated/frozen inference protocol uses 64 ordered sequences, each with
8,192 input tokens plus one target-only token. This calibration stream must be
disjoint from validation. It observes per-module activation maxima under
current-tensor scaling and freezes their maximum before measurement. It is not
part of the main 84-point curve unless explicitly reported as an additional
ablation.

## OLMES request bundle

The downstream likelihood sweep uses a frozen request bundle rather than
re-materializing tasks from mutable dataset versions. Its expected digests and
inventory counts are in `../manifests/olmes.yaml`. Place the verified manifest
and archive at user-controlled paths and pass them to the evaluation adapter.
If any digest differs, treat the run as a new evaluation rather than a
reproduction.
