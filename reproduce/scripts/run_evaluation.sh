#!/usr/bin/env bash
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage:" >&2
  echo "  $0 validation --checkpoint DIR --validation SHARDS... --numeric-path PATH --output FILE [options]" >&2
  echo "  $0 olmes COMPLETE_HF_MODEL_DIRECTORY OUTPUT_DIRECTORY [--dry-run]" >&2
  exit 2
fi

mode=$1
shift
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

case "$mode" in
  validation)
    # The Python CLI requires explicit checkpoint, data, numeric path, and
    # output arguments. It performs all model/eval/scaling lifecycle checks.
    export PYTHONPATH="$repository/src${PYTHONPATH:+:$PYTHONPATH}"
    exec python -m ue5m3_fp4.cli.evaluate "$@"
    ;;
  olmes)
    # Select one of the seven released paths with UE5M3_OLMES_NUMERIC_PATH.
    # Unsupported paths and unavailable frozen-request replay fail closed.
    exec "$repository/reproduce/evaluation/run_public_olmes.sh" "$@"
    ;;
  *)
    echo "unknown evaluation mode: $mode" >&2
    exit 2
    ;;
esac
