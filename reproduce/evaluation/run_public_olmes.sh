#!/usr/bin/env bash
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

expected_olmes_commit=8e2743734066b073c5d8498d1b8220f67a21a2d6

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 COMPLETE_HF_MODEL_DIRECTORY OUTPUT_DIRECTORY [--dry-run]" >&2
  exit 2
fi

model_directory=$1
output_directory=$2
dry_run=${3:-}
if [[ -n "$dry_run" && "$dry_run" != "--dry-run" ]]; then
  echo "the only optional argument is --dry-run" >&2
  exit 2
fi

repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
olmes_root=$repository/third_party/olmes
if [[ ! -d "$olmes_root" ]]; then
  echo "missing pinned OLMES submodule; run: git submodule update --init" >&2
  exit 2
fi
observed_olmes_commit=$(git -C "$olmes_root" rev-parse HEAD)
if [[ "$observed_olmes_commit" != "$expected_olmes_commit" ]]; then
  echo "OLMES revision mismatch: expected $expected_olmes_commit, got $observed_olmes_commit" >&2
  exit 2
fi

numeric_path=${UE5M3_OLMES_NUMERIC_PATH:-bf16}
case "$numeric_path" in
  bf16|ue5m3-proposed-b16|ue5m3-proposed-b32|ue5m3-torch-control|ue5m3-te-settings|native-nvfp4-te|native-nvfp4-no-rht-all)
    ;;
  *)
    echo "unsupported OLMES numeric path: $numeric_path" >&2
    echo "no numeric path is silently replaced with BF16" >&2
    exit 2
    ;;
esac
request_mode=${UE5M3_OLMES_REQUEST_MODE:-public_task_rebuild}
if [[ "$request_mode" != "public_task_rebuild" && "$request_mode" != "frozen_request_archive" ]]; then
  echo "unsupported OLMES request mode: $request_mode" >&2
  exit 2
fi
if [[ ! -d "$model_directory" ]]; then
  echo "model directory does not exist: $model_directory" >&2
  exit 2
fi
for required in config.json modeling_nemotron_h.py; do
  if [[ ! -f "$model_directory/$required" ]]; then
    echo "complete HF model directory is missing $required" >&2
    exit 2
  fi
done
if ! compgen -G "$model_directory/*.safetensors" >/dev/null; then
  echo "complete HF model directory contains no safetensors weights" >&2
  exit 2
fi
mkdir -p "$output_directory"
if find "$output_directory" -mindepth 1 -print -quit | grep -q .; then
  echo "output directory must be empty so cached predictions cannot skip model forwards" >&2
  exit 2
fi

model_directory=$(cd -- "$model_directory" && pwd)
output_directory=$(cd -- "$output_directory" && pwd)
runtime_result=$output_directory/ue5m3-olmes-runtime.json
olmes_import_root=$olmes_root
replay_work=
if [[ "$request_mode" == "frozen_request_archive" ]]; then
  request_manifest=${UE5M3_OLMES_REQUEST_MANIFEST:-}
  request_archive=${UE5M3_OLMES_REQUEST_ARCHIVE:-}
  if [[ ! -f "$request_manifest" || ! -f "$request_archive" ]]; then
    echo "frozen replay requires UE5M3_OLMES_REQUEST_MANIFEST and UE5M3_OLMES_REQUEST_ARCHIVE" >&2
    echo "expected manifest SHA-256: b7cd708300b7b63edd45e4d973de7195b2c98384f1a9b0773f49c5a8d0e47898" >&2
    echo "expected archive SHA-256: 0bf27af57eb1bb1b98872c4af12d419498652d935a6b745cc7ec4ecdb32d7483" >&2
    exit 2
  fi
  replay_work=$(mktemp -d)
  cleanup_replay_work() {
    if [[ -n "$replay_work" && -d "$replay_work" ]]; then
      rm -rf -- "$replay_work"
    fi
  }
  trap cleanup_replay_work EXIT
  python "$repository/reproduce/scripts/prepare_olmes_replay.py" prepare \
    --manifest "$request_manifest" \
    --archive "$request_archive" \
    --work-root "$replay_work" \
    --evaluation-dir "$output_directory"
  mkdir -p "$replay_work/shadow"
  cp -a "$olmes_root/oe_eval" "$replay_work/shadow/oe_eval"
  python "$repository/reproduce/scripts/prepare_olmes_replay.py" patch-olmes \
    --source "$replay_work/shadow/oe_eval/run_eval.py"
  olmes_import_root=$replay_work/shadow
  export OLMES_REUSE_RAW_REQUESTS=1
  export HF_HOME="$replay_work/extracted/payload/hf-cache"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export UE5M3_OLMES_FROZEN_BUNDLE_VERIFIED=1
fi
export UE5M3_PUBLIC_OLMES_RUNTIME=1
export UE5M3_PUBLIC_OLMES_MODEL_DIRECTORY="$model_directory"
export UE5M3_PUBLIC_OLMES_RUNTIME_RESULT="$runtime_result"
export UE5M3_OLMES_NUMERIC_PATH="$numeric_path"
export UE5M3_OLMES_REQUEST_MODE="$request_mode"
export PYTHONPATH="$olmes_import_root:$repository/reproduce/evaluation:$repository/src:$olmes_root${PYTHONPATH:+:$PYTHONPATH}"
command=(
  python -m oe_eval.launch
  --model "$model_directory"
  --model-type hf
  --model-args '{"dtype":"bfloat16","max_length":2048,"trust_remote_code":true}'
  --task core_9mcqa::olmes mmlu::olmes mmlu_pro:mc::none
  --random-subsample-seed 20260830
  --batch-size 8
  --gpus 1
  --num-workers 1
  --save-raw-requests true
  --output-dir "$output_directory"
)
if [[ "$dry_run" == "--dry-run" ]]; then
  command+=(--dry-run)
fi
"${command[@]}"
if [[ "$dry_run" != "--dry-run" && ! -s "$runtime_result" ]]; then
  echo "OLMES completed without the required runtime attestation: $runtime_result" >&2
  exit 1
fi
if [[ "$request_mode" == "frozen_request_archive" ]]; then
  python "$repository/reproduce/scripts/prepare_olmes_replay.py" verify \
    --manifest "$request_manifest" \
    --evaluation-dir "$output_directory"
fi
