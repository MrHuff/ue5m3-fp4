#!/usr/bin/env bash
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch a runnable public config against the pinned TorchTitan submodule.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 reproduce/configs/CONFIG.toml [additional TorchTitan arguments...]" >&2
  exit 2
fi

config=$1
shift
if [[ ! -f "$config" ]]; then
  echo "configuration not found: $config" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
config=$(cd -- "$(dirname -- "$config")" && pwd)/$(basename -- "$config")

case "$config" in
  "$repo_root"/reproduce/configs/*.toml) ;;
  *)
    echo "only runnable files under reproduce/configs are accepted" >&2
    echo "reproduce/historical_specs records non-runnable historical settings" >&2
    exit 2
    ;;
esac

: "${UE5M3_NEMOTRON_ASSETS:?Set UE5M3_NEMOTRON_ASSETS to assets from download_nemotron_assets.py}"
: "${UE5M3_DATA_ROOT:?Set UE5M3_DATA_ROOT to the prepared OLMo Mix root}"
: "${UE5M3_OUTPUT_ROOT:?Set UE5M3_OUTPUT_ROOT to a user-controlled output directory}"

for directory in "$UE5M3_NEMOTRON_ASSETS" "$UE5M3_DATA_ROOT/dclm" "$UE5M3_DATA_ROOT/olmo-no-dclm"; do
  if [[ ! -d "$directory" ]]; then
    echo "required directory not found: $directory" >&2
    exit 2
  fi
done

train_entrypoint=${UE5M3_TRAIN_ENTRYPOINT:-$repo_root/third_party/torchtitan/torchtitan/train.py}
if [[ ! -f "$train_entrypoint" ]]; then
  echo "training entry point not found: $train_entrypoint" >&2
  echo "run 'git submodule update --init --recursive' first" >&2
  exit 2
fi

config_name=$(basename "$config" .toml)
output_dir=${UE5M3_OUTPUT_ROOT%/}/$config_name

# This disables a cuDNN SDPA path that did not support the reported hybrid
# model/runtime combination. It does not select a private implementation.
export TORCH_CUDNN_SDPA_ENABLED=0

nnodes=${UE5M3_NNODES:-1}
nproc_per_node=${UE5M3_NPROC_PER_NODE:-1}
node_rank=${UE5M3_NODE_RANK:-0}
master_addr=${UE5M3_MASTER_ADDR:-127.0.0.1}
master_port=${UE5M3_MASTER_PORT:-29500}

export PYTHONPATH="$repo_root/src:$repo_root/third_party/torchtitan${PYTHONPATH:+:$PYTHONPATH}"

exec torchrun \
  --nnodes "$nnodes" \
  --nproc_per_node "$nproc_per_node" \
  --node_rank "$node_rank" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$master_addr:$master_port" \
  "$train_entrypoint" \
  --job.config_file "$config" \
  --job.dump_folder "$output_dir" \
  --model.hf_assets_path "$UE5M3_NEMOTRON_ASSETS" \
  --public-data.root "$UE5M3_DATA_ROOT" \
  "$@"
