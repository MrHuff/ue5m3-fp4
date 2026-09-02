# Scale-target value distributions

This directory contains the public, storage-neutral reproduction path for the
late-layer value-distribution figure and the archived 350M scale-target
ablation reported in the paper. It contains no model checkpoint or token data.

## What is measured

`capture_scale_target_histograms.py` loads one local BF16-master-weight
Nemotron-H 8B Hugging Face checkpoint and one selected row of a local
safetensors token file. The token tensor must be named `tokens` and have shape
`[N, 8193]`: 8,192 inputs and the next-token target.

One training-mode forward/backward pass records the following for
`mixer.down_proj` in hybrid layers 45, 47, 49, and 51:

- the forward input `X`;
- the loaded BF16 weight `W`;
- the upstream gradient `dY`; and
- the block-16 maxima of `dY.T` along the reduction dimension of the
  weight-gradient GEMM.

Every model parameter is frozen. The backward pass reaches a detached input
embedding, preserving the relevant upstream gradients while avoiding 8B
parameter-gradient allocation. The command requires `cut-cross-entropy` so it
does not materialize the full `[8192, 131072]` logit tensor.

For target `T`, the raw block-scale code is

```text
raw_code = block_amax * T / current_tensor_amax
```

before ties-to-even rounding to UE5M3. `T=2048` moves nonzero codes 4.57 times
farther from zero than `T=448`, but reduces stale-growth headroom from
`61440/448 = 137.14` to `61440/2048 = 30`. This capture uses the current `dY`
amax separately for each module. It diagnoses the scale-target tradeoff; it is
not an FP4 execution trace and does not replay the training run's delayed-amax
cache.

## Run a new public capture

First prepare the pinned public Nemotron-H assets and a standard Hugging Face
safetensors export as described in the repository reproduction guide. Freeze
caller-supplied ordered token rows with
`reproduce/scripts/prepare_validation_tokens.py`, or provide another local
safetensors file with the exact interface above.

```bash
python reproduce/scale_target/capture_scale_target_histograms.py \
  --checkpoint CHECKPOINT_DIRECTORY \
  --checkpoint-label ue5m3-proposed-b16-step-30000 \
  --tokens TOKENS.safetensors \
  --sequence-index 0 \
  --hf-assets NEMOTRON_ASSET_DIRECTORY \
  --local-files-only \
  --output capture.json
```

Repeat the command for the BF16 and proposed block-16 learned checkpoints with
the same token file and sequence index. The result records semantic checkpoint
and asset identities, token/file hashes, model configuration, numerical
protocol, runtime versions, full per-layer histograms, and pooled counts. It
does not record local filesystem locations.

## Archived report evidence

`archived/bf16.json` and `archived/ue5m3_proposed_b16.json` preserve the
complete numerical captures used by the report figure. Storage locations,
orchestration metadata, and nested wheel-location records were removed. Their
original report-input hashes and public archive hashes are recorded in
`archived/manifest.json`. The archives retain both per-layer and pooled counts,
so the accounting can be audited independently of the plot.

Render the four-panel figure with:

```bash
python reproduce/scale_target/render_archived_figure.py \
  --output scale_target_checkpoint_snapshot
```

This writes `scale_target_checkpoint_snapshot.pdf` and `.png`. Plotting-library
versions can affect file bytes and typography, but not the archived bins or
counts.

The one-way archival transformation is itself reproducible with
`archive_report_snapshots.py` when the two hash-pinned original report captures
are available. The exporter refuses any input whose SHA-256 differs from the
recorded sources and validates that its output contains no storage locations.

## Historical 350M ablation

`historical_350m_final_window.csv` contains all ten logged final-window points
for each matched configuration. `historical_350m_provenance.json` records the
architecture, optimizer, batching, seed, FP4 recipe, exact override, source
commit and source-artifact hashes.

This is a record-only artifact. The public tree does not include the historical
350M model implementation, exact ordered training stream, or original
distributed orchestration, so it would be misleading to claim an exact public
rerun from this repository alone. Both configurations are single seed 42
trajectories. Their ten final-window observations are repeated measurements
within a run, not ten independent runs, and the archived metric is training
loss rather than validation loss.
